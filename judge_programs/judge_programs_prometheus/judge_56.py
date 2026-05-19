def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical chain analysis approach. This variant focuses on:
    1. Causal connective density (because, therefore, since, thus, hence, so that, etc.)
    2. Conditional reasoning markers (if...then, when...would, suppose, assume)
    3. Explicit reasoning verbs (consider, note, observe, recall, recognize, understand)
    4. Progressive elaboration patterns (first...then...finally, moreover, furthermore, additionally)
    5. Justification ratio (claims backed by reasons vs bare assertions)
    6. Question-response alignment via interrogative echo
    7. Metacognitive markers (let's think, let me explain, the reason is, this means)
    """
    try:
        if not query or not response:
            return 0.0
        
        if len(response.strip()) < 20:
            return 0.5
        
        resp_lower = response.lower()
        resp_len = len(response)
        
        import re
        
        # Split into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Causal Connective Density ===
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis leads to\b', r'\bwhich means\b', r'\bwhich causes\b',
            r'\bfor this reason\b', r'\bthat\'s why\b', r'\bthis is why\b',
            r'\baccordingly\b', r'\bit follows that\b', r'\bgiven that\b',
            r'\bowing to\b', r'\bthanks to\b', r'\bon account of\b',
            r'\bas such\b', r'\bso\b(?=,?\s)',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        # Normalize by sentence count
        causal_density = causal_count / num_sentences
        # Score: up to 15 points
        score += min(causal_density * 12, 15.0)
        
        # === 2. Conditional Reasoning Markers ===
        conditional_patterns = [
            r'\bif\b.*?\b(then|would|could|should|will|might)\b',
            r'\bwhen\b.*?\b(would|could|should|will)\b',
            r'\bsuppose\b', r'\bassume\b', r'\bimagine\b',
            r'\bin case\b', r'\bprovided that\b', r'\bunless\b',
            r'\bwhat if\b', r'\beven if\b',
        ]
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, resp_lower))
        
        cond_density = conditional_count / num_sentences
        score += min(cond_density * 8, 10.0)
        
        # === 3. Explicit Reasoning/Thinking Verbs ===
        reasoning_verbs = [
            r'\bconsider\b', r'\bnote that\b', r'\bobserve\b', r'\brecall\b',
            r'\brecognize\b', r'\brealize\b', r'\bunderstand\b',
            r'\banalyze\b', r'\bexamine\b', r'\bevaluate\b',
            r'\bthink about\b', r'\breflect\b', r'\breason\b',
            r'\bdetermine\b', r'\bidentify\b', r'\bdistinguish\b',
            r'\bcompare\b', r'\bcontrast\b', r'\bassess\b',
            r'\blet\'s\b', r'\blet me\b', r'\blet us\b',
        ]
        reasoning_verb_count = 0
        for pattern in reasoning_verbs:
            reasoning_verb_count += len(re.findall(pattern, resp_lower))
        
        rv_density = reasoning_verb_count / num_sentences
        score += min(rv_density * 10, 10.0)
        
        # === 4. Progressive Elaboration / Sequential Markers ===
        sequential_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfourth\b',
            r'\bnext\b', r'\bthen\b', r'\bafter that\b', r'\bfinally\b',
            r'\bto begin\b', r'\bto start\b', r'\bfollowing this\b',
            r'\bsubsequently\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin addition\b', r'\balso\b',
            r'\bon top of\b', r'\bbeyond that\b', r'\bwhat\'s more\b',
            r'\bstep\s*\d', r'\b\d+\.\s', r'\b\d+\)\s',
        ]
        seq_count = 0
        for pattern in sequential_markers:
            seq_count += len(re.findall(pattern, resp_lower))
        
        seq_density = seq_count / num_sentences
        score += min(seq_density * 8, 12.0)
        
        # === 5. Metacognitive / Explanatory Framing ===
        metacognitive = [
            r'\bthe reason\b', r'\bthis means\b', r'\bthis implies\b',
            r'\bin other words\b', r'\bput simply\b', r'\bto clarify\b',
            r'\bto explain\b', r'\bto illustrate\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bhere\'s\b',
            r'\bhere is\b', r'\bthe key\b', r'\bthe point\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bessentially\b',
            r'\bfundamentally\b', r'\bin essence\b',
            r'\bwhat this means\b', r'\bthe idea\b',
        ]
        meta_count = 0
        for pattern in metacognitive:
            meta_count += len(re.findall(pattern, resp_lower))
        
        meta_density = meta_count / num_sentences
        score += min(meta_density * 10, 12.0)
        
        # === 6. Justification Ratio ===
        # Sentences containing at least one causal/explanatory marker vs total
        justified_sentences = 0
        justification_patterns = causal_connectives + metacognitive + [
            r'\bthis is\b', r'\bit\'s\b.*\bbecause\b',
        ]
        for sent in sentences:
            sent_lower = sent.lower()
            for pattern in justification_patterns:
                if re.search(pattern, sent_lower):
                    justified_sentences += 1
                    break
        
        justification_ratio = justified_sentences / num_sentences
        score += justification_ratio * 10.0  # up to 10 points
        
        # === 7. Sentence Complexity and Clause Depth ===
        # Longer sentences with subordinate clauses indicate more reasoning
        avg_commas = sum(s.count(',') for s in sentences) / num_sentences
        avg_clauses_indicator = sum(
            len(re.findall(r'\b(which|that|who|whom|where|while|although|whereas|whether)\b', s.lower()))
            for s in sentences
        ) / num_sentences
        
        clause_score = min(avg_commas * 1.5 + avg_clauses_indicator * 3, 8.0)
        score += clause_score
        
        # === 8. Contrast and Nuance Markers ===
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\bnonetheless\b', r'\balthough\b', r'\bwhile\b',
            r'\bdespite\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bbut\b', r'\byet\b', r'\binstead\b', r'\brather\b',
            r'\bstill\b', r'\beven though\b', r'\bregardless\b',
        ]
        contrast_count = 0
        for pattern in contrast_markers:
            contrast_count += len(re.findall(pattern, resp_lower))
        
        contrast_density = contrast_count / num_sentences
        score += min(contrast_density * 6, 8.0)
        
        # === 9. Empathetic Acknowledgment before Reasoning ===
        # Check if response starts with acknowledgment (shows structured thinking)
        first_two_sentences = ' '.join(sentences[:2]).lower() if len(sentences) >= 2 else resp_lower[:200]
        acknowledgment_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bthat\'s\b.*\b(understandable|valid|natural|normal|okay|fine)\b',
            r'\bit\'s\b.*\b(understandable|valid|natural|normal|okay|fine)\b',
            r'\bsorry to hear\b', r'\bcompletely\b.*\b(understandable|natural)\b',
            r'\babsolutely\b.*\b(okay|fine|understandable|natural)\b',
            r'\bperfectly\b.*\b(okay|fine|understandable|natural)\b',
        ]
        ack_found = any(re.search(p, first_two_sentences) for p in acknowledgment_patterns)
        if ack_found:
            score += 3.0
        
        # === 10. Response Substance / Depth Penalty ===
        # Very short responses likely lack reasoning depth
        word_count = len(response.split())
        if word_count < 30:
            score *= 0.4
        elif word_count < 50:
            score *= 0.6
        elif word_count < 80:
            score *= 0.8
        
        # === 11. Opaqueness Penalty ===
        # If response has many imperative/directive sentences without explanation
        imperative_count = 0
        for sent in sentences:
            sent_stripped = sent.strip()
            # Starts with a verb (imperative)
            first_word = sent_stripped.split()[0].lower() if sent_stripped.split() else ''
            imperative_starters = [
                'get', 'buy', 'try', 'just', 'stop', 'go', 'do', 'make',
                'take', 'keep', 'call', 'read', 'find', 'start', 'use',
            ]
            if first_word in imperative_starters:
                # Check if this sentence also has justification
                has_justification = any(
                    re.search(p, sent.lower()) 
                    for p in causal_connectives + metacognitive
                )
                if not has_justification:
                    imperative_count += 1
        
        unjustified_imperative_ratio = imperative_count / num_sentences
        score -= unjustified_imperative_ratio * 5.0
        
        # === 12. Dismissiveness Penalty ===
        dismissive_patterns = [
            r'\bjust\b.*\bget over\b', r'\bjust\b.*\bmove on\b',
            r'\byou should be able\b', r'\bjust\b.*\bdo it\b',
            r'\bget yourself together\b', r'\bnothing wrong\b',
            r'\bmaybe you\'re just not\b', r'\byou\'re not using it correctly\b',
            r'\bjust keep\b.*\btrying\b',
        ]
        dismissive_count = sum(
            len(re.findall(p, resp_lower)) for p in dismissive_patterns
        )
        score -= dismissive_count * 3.0
        
        # === 13. Numbered/Structured Reasoning Bonus ===
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        if len(numbered_items) >= 2:
            score += min(len(numbered_items) * 1.5, 6.0)
        
        # === 14. Analogy / Example Usage ===
        analogy_patterns = [
            r'\blike\b.*\bswitch\b', r'\bimagine\b', r'\bthink of\b',
            r'\banalog\b', r'\bsimilar to\b', r'\bjust like\b',
            r'\bpicture\b', r'\bsuppose\b', r'\bpretend\b',
            r'\bfor example\b', r'\bfor instance\b',
        ]
        analogy_count = sum(
            len(re.findall(p, resp_lower)) for p in analogy_patterns
        )
        score += min(analogy_count * 2.0, 6.0)
        
        # Clamp to [0, 10] range
        score = max(0.0, min(score, 10.0))
        
        # Scale to 1-5 range to match the examples
        final_score = 1.0 + (score / 10.0) * 4.0
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5