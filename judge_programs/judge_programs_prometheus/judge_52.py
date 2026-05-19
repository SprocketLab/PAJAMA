def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical connective chain analysis approach.
    
    This variant focuses on:
    1. Causal reasoning markers (because, since, therefore, thus, hence, etc.)
    2. Conditional reasoning patterns (if...then, when...would, etc.)
    3. Explicit acknowledgment/framing before answering
    4. Progressive elaboration depth (ideas building on previous ideas)
    5. Hedging and epistemic markers showing reasoning nuance
    6. Question-response alignment (addressing the query's implicit needs)
    """
    try:
        if not query or not response:
            return 0.0
        
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        import re
        from collections import Counter
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        if len(response_lower) < 10:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = response_lower.split()
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # ============================================================
        # 1. CAUSAL REASONING MARKERS (max 15 points)
        # Words/phrases that explain WHY something is the case
        # ============================================================
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bthis leads to\b', r'\bso that\b',
            r'\bfor this reason\b', r'\bthe reason\b', r'\bthat\'s why\b',
            r'\bthis is because\b', r'\bthis is why\b', r'\bit follows\b',
            r'\bin turn\b', r'\bas such\b', r'\bgiven that\b',
            r'\baccordingly\b', r'\bin order to\b',
        ]
        causal_count = 0
        for pattern in causal_markers:
            causal_count += len(re.findall(pattern, response_lower))
        
        # Normalize by response length, reward density
        causal_density = causal_count / num_sentences
        score += min(causal_density * 8, 15)
        
        # ============================================================
        # 2. CONDITIONAL / HYPOTHETICAL REASONING (max 10 points)
        # Shows the model is considering scenarios and implications
        # ============================================================
        conditional_patterns = [
            r'\bif\b.*\b(then|would|could|might|should|will|can)\b',
            r'\bwhen\b.*\b(would|could|might|should|will|can)\b',
            r'\bsuppose\b', r'\bimagine\b', r'\bconsider\b',
            r'\bin case\b', r'\bassuming\b', r'\bwhat if\b',
            r'\beven if\b', r'\bwhether\b',
        ]
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        conditional_density = conditional_count / num_sentences
        score += min(conditional_density * 6, 10)
        
        # ============================================================
        # 3. EPISTEMIC / HEDGING MARKERS (max 8 points)
        # Shows nuanced reasoning rather than blind assertions
        # ============================================================
        epistemic_markers = [
            r'\bit\'s important to\b', r'\bit\'s worth\b', r'\bkeep in mind\b',
            r'\bnote that\b', r'\bremember that\b', r'\bremember to\b',
            r'\bbe aware\b', r'\bunderstandably\b', r'\bunderstandable\b',
            r'\bnatural(ly)?\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\btend(s)? to\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\blikely\b', r'\bprobably\b',
        ]
        epistemic_count = 0
        for pattern in epistemic_markers:
            epistemic_count += len(re.findall(pattern, response_lower))
        
        epistemic_density = epistemic_count / num_sentences
        score += min(epistemic_density * 4, 8)
        
        # ============================================================
        # 4. PROGRESSIVE ELABORATION / BUILDING ON IDEAS (max 15 points)
        # Detect when sentences reference or build upon previous content
        # ============================================================
        referential_markers = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bsuch\b', r'\bthe above\b', r'\bas mentioned\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bbuilding on\b', r'\bexpanding on\b', r'\bwith that\b',
            r'\bwith this\b', r'\bbased on\b',
        ]
        
        # Check referential markers in non-first sentences (building on prior content)
        ref_count = 0
        for i, sent in enumerate(sentences):
            if i == 0:
                continue
            sent_lower = sent.lower()
            for pattern in referential_markers:
                if re.search(pattern, sent_lower):
                    ref_count += 1
                    break  # count once per sentence
        
        if num_sentences > 1:
            ref_ratio = ref_count / (num_sentences - 1)
        else:
            ref_ratio = 0
        score += ref_ratio * 15
        
        # ============================================================
        # 5. EXPLICIT ACKNOWLEDGMENT / FRAMING (max 10 points)
        # Does the response first acknowledge the situation before diving in?
        # ============================================================
        first_two_sentences = ' '.join(sentences[:2]).lower() if sentences else ''
        
        acknowledgment_patterns = [
            r'\bi (can )?(see|hear|understand|recognize|appreciate)\b',
            r'\bit\'s (completely |totally |perfectly )?(understandable|natural|okay|normal)\b',
            r'\bi\'m (genuinely |truly |really )?sorry\b',
            r'\bthat\'s (a |completely )?(valid|great|good|understandable)\b',
            r'\blet\'s\b', r'\bhere\'s\b', r'\bhere are\b',
            r'\bto (answer|address|help|assist|explain|clarify)\b',
            r'\bgreat question\b', r'\bgood question\b',
            r'\bimagine\b', r'\bthink of\b', r'\bconsider\b',
        ]
        
        ack_score = 0
        for pattern in acknowledgment_patterns:
            if re.search(pattern, first_two_sentences):
                ack_score += 3
        score += min(ack_score, 10)
        
        # ============================================================
        # 6. EXPLANATORY DEPTH via clause complexity (max 12 points)
        # Longer sentences with subordinate clauses indicate deeper reasoning
        # ============================================================
        subordinate_markers = [
            r'\bwhich\b', r'\bwhere\b', r'\bwhile\b', r'\balthough\b',
            r'\bwhereas\b', r'\bunless\b', r'\bprovided\b',
            r'\bso long as\b', r'\bas long as\b',
        ]
        
        complex_sentence_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            sent_words = sent_lower.split()
            # A complex sentence: has subordinate marker AND reasonable length
            if len(sent_words) >= 10:
                for pattern in subordinate_markers:
                    if re.search(pattern, sent_lower):
                        complex_sentence_count += 1
                        break
        
        complexity_ratio = complex_sentence_count / num_sentences
        score += complexity_ratio * 12
        
        # ============================================================
        # 7. STRUCTURED ENUMERATION (max 8 points)
        # Numbered steps or labeled sections show step-wise formulation
        # ============================================================
        numbered_pattern = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        labeled_pattern = re.findall(r'(?:^|\n)\s*(?:step|first|second|third|fourth|fifth|next|finally|lastly)\b', response_lower)
        
        enum_count = len(numbered_pattern) + len(labeled_pattern)
        if enum_count >= 4:
            score += 8
        elif enum_count >= 2:
            score += 5
        elif enum_count >= 1:
            score += 3
        
        # ============================================================
        # 8. ANALOGY / EXAMPLE USAGE (max 8 points)
        # Using analogies or examples to make reasoning visible
        # ============================================================
        analogy_patterns = [
            r'\blike\b.*\b(a|an|the)\b', r'\bjust like\b', r'\bsimilar to\b',
            r'\bthink of (it as|this as)\b', r'\bimagine\b',
            r'\bfor (example|instance)\b', r'\bsuch as\b',
            r'\banalog(y|ous)\b', r'\bin other words\b',
            r'\bto (put it|illustrate)\b', r'\bpicture\b',
        ]
        analogy_count = 0
        for pattern in analogy_patterns:
            analogy_count += len(re.findall(pattern, response_lower))
        
        score += min(analogy_count * 2.5, 8)
        
        # ============================================================
        # 9. NEGATIVE SIGNAL: Opaque/dismissive language (penalty up to -10)
        # Jumping to conclusions, dismissive phrases
        # ============================================================
        dismissive_patterns = [
            r'\bjust\b.*\bdo\b', r'\bjust\b.*\bget\b',
            r'\byou should be able to\b', r'\bsimply\b',
            r'\bjust remember\b', r'\bmaybe you\'re just\b',
            r'\bget over it\b', r'\bmove on\b(?!.*\bit\'s)',
            r'\bnothing wrong with\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        # Only penalize if there's a high density of dismissiveness
        if num_sentences > 0:
            dismissive_density = dismissive_count / num_sentences
            if dismissive_density > 0.3:
                score -= min(dismissive_density * 8, 10)
        
        # ============================================================
        # 10. NEGATIVE SIGNAL: Very short / shallow responses (penalty up to -8)
        # ============================================================
        if num_words < 30:
            score -= 5
        elif num_words < 50:
            score -= 2
        
        # Reward moderate-to-long responses that maintain quality
        if num_words > 80:
            score += 2
        if num_words > 120:
            score += 1
        
        # ============================================================
        # 11. EMPATHETIC REASONING MARKERS (max 7 points)
        # Shows the response is reasoning about the person's state
        # ============================================================
        empathy_reasoning = [
            r'\byou(\'re| are) feeling\b', r'\byour feelings\b',
            r'\bit\'s okay to\b', r'\bperfectly (fine|okay|normal|natural)\b',
            r'\bgive yourself\b', r'\ballow yourself\b',
            r'\byour experience\b', r'\byour situation\b',
            r'\bi can (see|hear|sense|understand)\b',
            r'\bwe (value|appreciate|understand)\b',
        ]
        
        empathy_count = 0
        for pattern in empathy_reasoning:
            empathy_count += len(re.findall(pattern, response_lower))
        
        score += min(empathy_count * 2.5, 7)
        
        # ============================================================
        # 12. COHERENCE FLOW: Sentence-to-sentence word overlap (max 7 points)
        # Measures if ideas flow logically from one sentence to the next
        # ============================================================
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'or', 'but', 'not', 'no', 'it', 'its', 'this', 'that',
                     'i', 'you', 'your', 'we', 'they', 'them', 'their', 'he', 'she',
                     'his', 'her', 'my', 'our', 'me', 'us', 'so', 'if', 'as'}
        
        if len(sentences) >= 2:
            overlap_scores = []
            for i in range(1, len(sentences)):
                prev_words = set(re.findall(r'\b\w+\b', sentences[i-1].lower())) - stopwords
                curr_words = set(re.findall(r'\b\w+\b', sentences[i].lower())) - stopwords
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words) / min(len(prev_words), len(curr_words))
                    overlap_scores.append(overlap)
            
            if overlap_scores:
                avg_overlap = sum(overlap_scores) / len(overlap_scores)
                # Sweet spot: some overlap (0.1-0.4) is good, too much is repetitive
                if 0.05 <= avg_overlap <= 0.5:
                    score += avg_overlap * 14  # max ~7 at 0.5
                elif avg_overlap > 0.5:
                    score += 4  # still some credit but diminishing
        
        # ============================================================
        # FINAL SCALING: Map to 1-5 range
        # ============================================================
        # Expected raw score range: roughly -5 to 80
        # Normalize to 1-5
        
        # Clamp raw score
        raw_score = max(score, 0)
        
        # Sigmoid-like mapping to 1-5
        import math
        # Map: 0 -> ~1, 15 -> ~2, 25 -> ~3, 35 -> ~4, 50+ -> ~5
        normalized = 1 + 4 * (1 - math.exp(-raw_score / 25))
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, round(normalized, 1)))
        
        return final_score
        
    except Exception:
        return 2.5