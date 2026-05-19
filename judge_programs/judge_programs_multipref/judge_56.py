def judging_function(query, response):
    """
    Evaluates reasoning transparency by analyzing causal/logical connective chains,
    explicit reasoning markers, question-answer self-dialogue patterns, and
    the density of explanatory constructions throughout the response.
    
    This variant focuses on:
    1. Causal chain analysis (because/therefore/since/thus chains)
    2. Self-dialogue patterns (rhetorical questions followed by answers)
    3. Explanatory construction density ("this means", "in other words", "the reason is")
    4. Progressive deepening (do later parts build on earlier parts?)
    5. Conditional reasoning ("if...then", "given that", "assuming")
    6. Contrast/comparison reasoning ("however", "on the other hand", "whereas")
    7. Evidence/example integration ("for example", "for instance", "such as", "consider")
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        
        text = response.strip()
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count < 5:
            return 0.5
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Causal connective chain analysis ----
        # Look for sequences of causal reasoning words and measure their distribution
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bso that\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bthis leads to\b', r'\bwhich means\b', r'\bwhich causes\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\baccordingly\b', r'\bin turn\b',
            r'\bthat\'s why\b', r'\bthis is why\b', r'\bthe reason\b',
            r'\bcaused by\b', r'\bresulting in\b', r'\bleading to\b',
        ]
        
        causal_positions = []
        causal_count = 0
        for pattern in causal_markers:
            for m in re.finditer(pattern, text_lower):
                causal_count += 1
                causal_positions.append(m.start() / max(len(text_lower), 1))
        
        # Reward distributed causal reasoning (not all clustered in one spot)
        causal_spread = 0.0
        if len(causal_positions) >= 2:
            causal_positions.sort()
            gaps = [causal_positions[i+1] - causal_positions[i] for i in range(len(causal_positions)-1)]
            avg_gap = sum(gaps) / len(gaps)
            # Ideal: evenly spread. Penalize if all clustered.
            causal_spread = min(avg_gap * 3, 1.0)  # normalize
        elif len(causal_positions) == 1:
            causal_spread = 0.3
        
        causal_density = causal_count / max(num_sentences, 1)
        causal_score = min(causal_density * 4.0, 5.0) + causal_spread * 3.0  # max ~8
        
        # ---- 2. Explanatory constructions ----
        explanatory_patterns = [
            r'\bthis means\b', r'\bin other words\b', r'\bput (simply|differently|another way)\b',
            r'\bto clarify\b', r'\bto explain\b', r'\bwhat this means is\b',
            r'\bthe idea is\b', r'\bthe point is\b', r'\bessentially\b',
            r'\bin essence\b', r'\bsimply put\b', r'\bto put it\b',
            r'\bnamely\b', r'\bthat is to say\b', r'\bi\.e\.\b',
            r'\bspecifically\b', r'\bmore precisely\b', r'\bin particular\b',
            r'\bto be (more )?specific\b', r'\bwhat I mean\b',
            r'\blet me explain\b', r'\bhere\'s why\b', r'\bhere is why\b',
            r'\bthe key (point|idea|insight|takeaway)\b',
        ]
        
        explanatory_count = 0
        for pattern in explanatory_patterns:
            explanatory_count += len(re.findall(pattern, text_lower))
        
        explanatory_score = min(explanatory_count * 2.5, 8.0)
        
        # ---- 3. Conditional/hypothetical reasoning ----
        conditional_patterns = [
            r'\bif\b.{3,60}\bthen\b', r'\bgiven that\b', r'\bassuming\b',
            r'\bsuppose\b', r'\bin the case\b', r'\bwhen\b.{3,40}\bthen\b',
            r'\bprovided that\b', r'\bunless\b', r'\bwould\b.{3,30}\bif\b',
            r'\beven if\b', r'\bwhether or not\b',
        ]
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, text_lower))
        
        conditional_score = min(conditional_count * 2.0, 6.0)
        
        # ---- 4. Contrast/comparison reasoning ----
        contrast_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bwhereas\b',
            r'\bin contrast\b', r'\bconversely\b', r'\balternatively\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bdespite\b',
            r'\balthough\b', r'\bwhile\b.{5,40}\b(also|still|yet)\b',
            r'\bunlike\b', r'\bas opposed to\b', r'\brather than\b',
            r'\bon the contrary\b', r'\bbut\b',
        ]
        
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, text_lower))
        
        contrast_score = min(contrast_count * 1.5, 6.0)
        
        # ---- 5. Evidence/example integration ----
        evidence_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bconsider\b', r'\btake\b.{1,15}\bas an example\b',
            r'\bto illustrate\b', r'\billustrat', r'\bdemonstrat',
            r'\bevidence\b', r'\baccording to\b', r'\bresearch\b',
            r'\bdata\b', r'\bstud(y|ies)\b', r'\bstatistic',
            r'\ba good example\b', r'\bin practice\b',
            r'\blike\b.{1,5}\bwhen\b',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, text_lower))
        
        evidence_score = min(evidence_count * 1.8, 6.0)
        
        # ---- 6. Progressive structure detection ----
        # Check for sequential/progressive markers that indicate step-building
        progressive_patterns = [
            r'\bfirst(ly)?\b', r'\bsecond(ly)?\b', r'\bthird(ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blast(ly)?\b',
            r'\bto (start|begin)\b', r'\bmoving on\b', r'\bbuilding on\b',
            r'\bnow that we\b', r'\bwith that in mind\b', r'\bgiven this\b',
            r'\bfrom this\b', r'\bbased on (this|the above)\b',
            r'\bstep \d+\b', r'\b\d+\.\s', r'\b\d+\)\s',
        ]
        
        progressive_count = 0
        for pattern in progressive_patterns:
            progressive_count += len(re.findall(pattern, text_lower))
        
        progressive_score = min(progressive_count * 1.2, 7.0)
        
        # ---- 7. Self-dialogue / rhetorical question patterns ----
        # Detect "What does this mean?" or "Why?" followed by explanation
        rhetorical_q = re.findall(r'\b(what|why|how|when|where|which|who)\b[^.!?]{5,60}\?', text_lower)
        rhetorical_score = min(len(rhetorical_q) * 2.5, 6.0)
        
        # ---- 8. Intermediate conclusion markers ----
        intermediate_patterns = [
            r'\bso\b,?\s', r'\bwe can (see|conclude|determine|say)\b',
            r'\bthis (shows|tells|indicates|suggests|demonstrates|implies|confirms)\b',
            r'\bfrom (this|here)\b', r'\bat this point\b',
            r'\bso far\b', r'\bin summary\b', r'\bto summarize\b',
            r'\bthe takeaway\b', r'\bthe conclusion\b',
            r'\bwe (now |can )?(know|see|understand|have)\b',
            r'\bnote that\b', r'\bnotice that\b', r'\bobserve that\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bsignificantly\b',
            r'\bkeep in mind\b', r'\bremember\b',
        ]
        
        intermediate_count = 0
        for pattern in intermediate_patterns:
            intermediate_count += len(re.findall(pattern, text_lower))
        
        intermediate_score = min(intermediate_count * 1.5, 6.0)
        
        # ---- 9. Mathematical/logical notation (for technical queries) ----
        math_symbols = len(re.findall(r'[=+\-*/^]', text))
        formula_patterns = len(re.findall(r'\b\w+\s*=\s*[\d\w(]', text))
        latex_patterns = len(re.findall(r'\\[a-zA-Z]+', text))
        
        # Only count if query seems technical
        query_lower = query.lower()
        technical_query = any(w in query_lower for w in [
            'calculate', 'compute', 'find', 'solve', 'equation', 'formula',
            'speed', 'velocity', 'mass', 'energy', 'force', 'distance',
            'how many', 'how much', 'what is the', 'derive', 'prove',
            'kg', 'm/s', 'degrees', 'angle', 'coefficient'
        ])
        
        math_score = 0.0
        if technical_query:
            math_score = min((formula_patterns * 2.0 + latex_patterns * 0.5), 6.0)
        
        # ---- 10. Sentence-level logical flow ----
        # Check if sentences reference previous content (pronouns like "this", "these", "that result")
        reference_patterns = [
            r'^this\b', r'^these\b', r'^that\b', r'^those\b',
            r'^it\b', r'^such\b', r'^here\b',
        ]
        
        referencing_sentences = 0
        for s in sentences[1:]:  # skip first sentence
            s_lower = s.strip().lower()
            for pat in reference_patterns:
                if re.match(pat, s_lower):
                    referencing_sentences += 1
                    break
        
        flow_ratio = referencing_sentences / max(num_sentences - 1, 1)
        flow_score = flow_ratio * 5.0  # max 5
        
        # ---- 11. Engagement/framing signals ----
        framing_patterns = [
            r'\blet\'s\b', r'\blet us\b', r'\blet me\b',
            r'\bwe need to\b', r'\bwe can\b', r'\bwe should\b',
            r'\bthe (first|next|final) (step|thing|part)\b',
            r'\bto (understand|answer|solve|address) this\b',
            r'\bbreak (it|this) down\b', r'\bdive (in|into)\b',
            r'\bthink about\b', r'\bask ourselves\b',
        ]
        
        framing_count = 0
        for pattern in framing_patterns:
            framing_count += len(re.findall(pattern, text_lower))
        
        framing_score = min(framing_count * 2.0, 5.0)
        
        # ---- 12. Paragraph structure (multi-paragraph = more structured reasoning) ----
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        para_count = len(paragraphs)
        
        # Having 2-8 substantial paragraphs is good for reasoning
        if para_count >= 3:
            para_score = min(para_count * 0.8, 4.0)
        elif para_count == 2:
            para_score = 2.0
        else:
            para_score = 0.5
        
        # ---- 13. Opacity penalty ----
        # If response is long but has very few reasoning markers, penalize
        total_reasoning_markers = (causal_count + explanatory_count + conditional_count + 
                                   contrast_count + progressive_count + intermediate_count +
                                   len(rhetorical_q) + evidence_count + framing_count)
        
        reasoning_density = total_reasoning_markers / max(word_count / 100, 1)
        
        opacity_penalty = 0.0
        if word_count > 50 and reasoning_density < 1.0:
            opacity_penalty = (1.0 - reasoning_density) * 4.0
        
        # ---- 14. Bold/formatted key terms (shows structured presentation) ----
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', text))
        format_score = min(bold_count * 0.6, 4.0)
        
        # ---- Aggregate score ----
        raw_score = (
            causal_score * 1.3 +       # causal chains: important
            explanatory_score * 1.2 +   # explanations
            conditional_score * 1.0 +   # conditional reasoning
            contrast_score * 0.8 +      # contrasts
            evidence_score * 0.9 +      # evidence
            progressive_score * 1.1 +   # progressive structure
            rhetorical_score * 0.8 +    # self-dialogue
            intermediate_score * 1.1 +  # intermediate conclusions
            math_score * 1.0 +          # math/technical
            flow_score * 0.7 +          # sentence flow
            framing_score * 0.6 +       # framing
            para_score * 0.5 +          # paragraph structure
            format_score * 0.4 -        # formatting
            opacity_penalty * 1.0       # opacity penalty
        )
        
        # Length bonus: slightly reward longer responses (more room for reasoning)
        # but with diminishing returns
        length_bonus = math.log(max(word_count, 1) + 1) * 0.5
        raw_score += length_bonus
        
        # Normalize to 0-100 range
        # Typical raw scores range from about 0 to ~60 for very good responses
        normalized = max(0.0, min(100.0, raw_score * 1.5))
        
        return round(normalized, 2)
        
    except Exception:
        return 5.0