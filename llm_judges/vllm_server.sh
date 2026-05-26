#!/bin/bash
# Default values based on your original command
CUDA_VISIBLE_DEVICES="0"
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
DTYPE="bfloat16"
CHUNKED_PREFILL="--enable-chunked-prefill"
SERVED_MODEL_NAME=""
MAX_MODEL_LEN="4096"

# Usage message
usage() {
    echo "Usage: $0 --model <model_name> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --model <model_name>              Model name or path"
    echo ""
    echo "Optional:"
    echo "  --served-model-name <name>        Name to use in API (default: same as model)"
    echo "  --host <host>                     Host address (default: 0.0.0.0)"
    echo "  --port <port>                     Port number (default: 8000)"
    echo "  --tensor-parallel-size <size>     Tensor parallel size (default: 1)"
    echo "  --data-parallel-size <size>       Data parallel size (default: 1)"
    echo "  --gpu-memory-utilization <value>  GPU memory utilization (default: 0.95)"
    echo "  --dtype <dtype>                   Data type (default: bfloat16)"
    echo "  --no-chunked-prefill              Disable chunked prefill"
    echo "  --cuda-devices <device_list>      CUDA devices (default: 0,1,2,3,4,5,6,7)"
    echo "  --max-model-len <size>            The maximum context length"
    echo ""
    echo "Example: $0 --model Qwen/Qwen2.5-7B-Instruct --served-model-name judge-model"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift ;;
        --data-parallel-size) DATA_PARALLEL_SIZE="$2"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        --dtype) DTYPE="$2"; shift ;;
        --no-chunked-prefill) CHUNKED_PREFILL=""; ;;
        --cuda-devices) CUDA_VISIBLE_DEVICES="$2"; shift ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check if model is provided
if [ -z "$MODEL" ]; then
    echo "Error: Model name is required."
    usage
fi

# Validate tensor-parallel-size against CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

if [ "$TENSOR_PARALLEL_SIZE" -gt "$NUM_GPUS" ]; then
    echo "Error: tensor-parallel-size ($TENSOR_PARALLEL_SIZE) cannot be greater than the number of GPUs ($NUM_GPUS)."
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# Build the served-model-name option
SERVED_MODEL_NAME_OPTION=""
if [ -n "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME_OPTION="--served-model-name $SERVED_MODEL_NAME"
fi

# Build the max-model-len option
MAX_MODEL_LEN_OPTION=""
if [ -n "$MAX_MODEL_LEN" ]; then
    MAX_MODEL_LEN_OPTION="--max-model-len $MAX_MODEL_LEN"
fi

# Build the vllm serve command
VLLM_COMMAND="vllm serve $MODEL \
  --host $HOST \
  --port $PORT \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --data-parallel-size $DATA_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $DTYPE \
  $MAX_MODEL_LEN_OPTION \
  $CHUNKED_PREFILL \
  $SERVED_MODEL_NAME_OPTION"

# Print the command for debugging
echo "Running command: $VLLM_COMMAND"

# Execute the command
eval $VLLM_COMMAND

# bash ./vllm_server.sh --model Qwen/Qwen2.5-1.5B-Instruct --served-model-name judge-model
# bash ./vllm_server.sh --model Qwen/Qwen2.5-3B-Instruct --served-model-name judge-model
# bash ./vllm_server.sh --model Qwen/Qwen2.5-7B-Instruct --served-model-name judge-model
# bash ./vllm_server.sh --model Qwen/Qwen2.5-14B-Instruct --served-model-name judge-model