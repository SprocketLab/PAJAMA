########################
# Reward model training for Qwen2.5 (3B) with optional LoRA.
#
# Train split : train.jsonl  (= original train + val, fused by converting_data.py)
# Eval split  : test.jsonl   (GT-labeled, used for metrics during and after training)
# Metrics     : accuracy
# Reports     : W&B
########################
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config"})
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "HF model id, e.g. Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen2.5-7B-Instruct"})
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    train_set_path: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing train.jsonl and test.jsonl"})
    output_path: Optional[str] = field(
        default="./models/qwen_rm",
        metadata={"help": "Output directory for checkpoints and merged model"})
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="adamw_torch_fused")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(default=999999)
    eval_every_steps: Optional[int] = field(default=999999)
    eval_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "'epoch' or 'steps'"})
    train_subset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly sample this many examples from train set (default: use all)"})

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_project: Optional[str] = field(
        default="pajama-reward-model",
        metadata={"help": "W&B project name"})
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name (auto-generated if None)"})

    # ── LoRA ──────────────────────────────────────────────────────────────
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Use LoRA adapter"})
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.05)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.wandb_project:
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

# ── tokenizer ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Pad token: {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
print(f"Padding side: {tokenizer.padding_side}")

# ── dataset ───────────────────────────────────────────────────────────────────

train_path = script_args.train_set_path
output_name = script_args.output_path


def build_dataset(tokenizer, train_path):
    def tokenize(sample):
        bos = tokenizer.bos_token or ""
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False,
            add_generation_prompt=False).replace(bos, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False,
            add_generation_prompt=False).replace(bos, "")
        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    data_files = {
        "train": os.path.join(train_path, "train.jsonl"),
        "test":  os.path.join(train_path, "test.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"].shuffle(seed=42).map(tokenize, num_proc=8)
    eval_ds  = ds["test"].shuffle(seed=42).map(tokenize, num_proc=8)
    return train_ds, eval_ds


train_dataset, eval_dataset = build_dataset(tokenizer, train_path)

if script_args.train_subset_size is not None:
    full_size = len(train_dataset)
    n = min(script_args.train_subset_size, full_size)
    train_dataset = train_dataset.shuffle(seed=42).select(range(n))
    print(f"Train subset: {n} / {full_size} sampled")

print(f"Train: {len(train_dataset)}  |  Eval (test): {len(eval_dataset)}")

# ── model ─────────────────────────────────────────────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# ── training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    eval_strategy=script_args.eval_strategy,
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="wandb",
    run_name=script_args.wandb_run_name,
    eval_on_start=True,
)

# ── data collator ─────────────────────────────────────────────────────────────

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = []
        for f in features:
            merged.append({"input_ids": f["input_ids_j"],
                           "attention_mask": f["attention_mask_j"]})
            merged.append({"input_ids": f["input_ids_k"],
                           "attention_mask": f["attention_mask_k"]})
        batch = self.tokenizer.pad(
            merged, padding=self.padding, max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return {"input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "return_loss": True}


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    rewards_j = eval_pred.predictions[0].squeeze()  # (N,)
    rewards_k = eval_pred.predictions[1].squeeze()  # (N,)
    return {"accuracy": float(np.mean(rewards_j > rewards_k))}


# ── trainer ───────────────────────────────────────────────────────────────────

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)

trainer.train()

# ── save ──────────────────────────────────────────────────────────────────────

print("Saving last checkpoint ...")
save_path = os.path.join(output_name, "last_checkpoint")
if script_args.use_lora:
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Merging LoRA weights ...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(output_name, "merged")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")
else:
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
