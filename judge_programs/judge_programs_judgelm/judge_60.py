def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    sentence-level analysis of claim types, structural coherence patterns,
    and proportional uncertainty signaling.
    
    This variant focuses on:
    1. Sentence-level classification (factual assertion vs hedged vs speculative)
    2. Ratio of hedged sentences to total claim-bearing sentences
    3. Detection of epistemic verb patterns and modal constructions
    4. Penalizing absolutist language patterns
    5. Rewarding structured reasoning with appropriate qualifications
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 3:
            return 0.5
        
        # Split response into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) == 0:
            return 1.0
        
        # === Feature 1: Sentence-level epistemic classification ===
        # Epistemic modal verbs and constructions (not just single words)
        epistemic_patterns = [
            r'\b(?:might|may|could)\s+(?:be|have|suggest|indicate|mean)',
            r'\b(?:it\s+is\s+)?(?:possible|plausible|conceivable)\s+that\b',
            r'\b(?:appears?|seems?)\s+(?:to\s+(?:be|have|suggest))',
            r'\bto\s+(?:some|a\s+certain)\s+(?:extent|degree)\b',
            r'\b(?:generally|typically|usually|often|sometimes|occasionally)\b',
            r'\b(?:tends?\s+to|inclined?\s+to)\b',
            r'\b(?:in\s+(?:most|many|some)\s+cases)\b',
            r'\b(?:it\s+(?:depends|varies))\b',
            r'\b(?:not\s+(?:entirely|completely|necessarily|always))\b',
            r'\b(?:one\s+(?:could|might|may)\s+argue)\b',
            r'\b(?:there\s+(?:is|are)\s+(?:some|limited|growing)\s+evidence)\b',
            r'\b(?:research|studies|evidence)\s+(?:suggests?|indicates?|shows?)\b',
            r'\b(?:according\s+to)\b',
            r'\b(?:it\s+is\s+(?:widely|generally|commonly)\s+(?:believed|accepted|understood))\b',
            r'\b(?:as\s+far\s+as\s+(?:we|I)\s+know)\b',
            r'\b(?:to\s+(?:my|the\s+best\s+of\s+(?:my|our))\s+knowledge)\b',
        ]
        
        # Absolutist / overconfident patterns
        absolutist_patterns = [
            r'\b(?:always|never|certainly|definitely|absolutely|undoubtedly)\b',
            r'\b(?:there\s+is\s+no\s+(?:doubt|question))\b',
            r'\b(?:it\s+is\s+(?:clear|obvious|evident|certain)\s+that)\b',
            r'\b(?:without\s+(?:a\s+)?doubt)\b',
            r'\b(?:everyone\s+knows)\b',
            r'\b(?:the\s+(?:only|sole|single)\s+(?:reason|way|answer|explanation))\b',
            r'\b(?:guaranteed|proven\s+fact|indisputable)\b',
            r'\b(?:100\s*%|100\s*percent)\b',
            r'\b(?:impossib(?:le|ly))\b',
            r'\b(?:no\s+one\s+(?:can|could|would))\b',
        ]
        
        # Hedging / qualification words (lighter weight individual markers)
        hedging_words = [
            r'\blikely\b', r'\bunlikely\b', r'\bperhaps\b', r'\bprobably\b',
            r'\bpossibly\b', r'\barguably\b', r'\bapproximately\b', r'\broughly\b',
            r'\bestimated\b', r'\bsuggested\b', r'\bbelieved\b', r'\bthought\b',
            r'\bassumed\b', r'\bpresumably\b', r'\bseemingly\b', r'\bapparently\b',
            r'\bsupposedly\b', r'\ballegedly\b', r'\breportedly\b',
            r'\bmight\b', r'\bcould\b', r'\bmay\b',
        ]
        
        # Acknowledgment of complexity / nuance
        nuance_patterns = [
            r'\b(?:however|on\s+the\s+other\s+hand|that\s+said|nonetheless)\b',
            r'\b(?:it\s+is\s+(?:worth|important)\s+(?:noting|mentioning))\b',
            r'\b(?:there\s+(?:are|is)\s+(?:debate|disagreement|controversy))\b',
            r'\b(?:this\s+is\s+(?:a\s+)?(?:complex|nuanced|debat(?:able|ed)))\b',
            r'\b(?:opinions?\s+(?:vary|differ))\b',
            r'\b(?:it\s+is\s+difficult\s+to)\b',
            r'\b(?:not\s+(?:straightforward|simple|clear-cut))\b',
            r'\b(?:subjective)\b',
            r'\b(?:vary\s+depending)\b',
            r'\b(?:multiple\s+(?:factors|perspectives|viewpoints))\b',
            r'\b(?:on\s+(?:one|the\s+one)\s+hand)\b',
            r'\b(?:while\s+(?:some|others|many))\b',
        ]
        
        response_lower = response_stripped.lower()
        
        # Count pattern matches
        epistemic_count = 0
        for pat in epistemic_patterns:
            epistemic_count += len(re.findall(pat, response_lower))
        
        absolutist_count = 0
        for pat in absolutist_patterns:
            absolutist_count += len(re.findall(pat, response_lower))
        
        hedging_count = 0
        for pat in hedging_words:
            hedging_count += len(re.findall(pat, response_lower))
        
        nuance_count = 0
        for pat in nuance_patterns:
            nuance_count += len(re.findall(pat, response_lower))
        
        # === Feature 2: Sentence-level analysis ===
        claim_sentences = 0
        hedged_sentences = 0
        absolutist_sentences = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Is this a claim-bearing sentence? (contains a verb-like structure)
            is_claim = bool(re.search(r'\b(?:is|are|was|were|has|have|had|will|would|can|do|does|did|should|must)\b', sent_lower))
            if not is_claim:
                continue
            claim_sentences += 1
            
            # Check if hedged
            sent_hedged = False
            for pat in epistemic_patterns + hedging_words:
                if re.search(pat, sent_lower):
                    sent_hedged = True
                    break
            if sent_hedged:
                hedged_sentences += 1
            
            # Check if absolutist
            for pat in absolutist_patterns:
                if re.search(pat, sent_lower):
                    absolutist_sentences += 1
                    break
        
        # === Feature 3: Response completeness and structure ===
        word_count = len(response_stripped.split())
        
        # Very short responses get a base penalty (but not fatal)
        length_factor = min(1.0, word_count / 20.0)
        
        # Check for repetition (sign of low quality)
        words = response_lower.split()
        if len(words) > 10:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                max_bigram_freq = max(bigram_counts.values())
                repetition_ratio = max_bigram_freq / len(bigrams)
            else:
                repetition_ratio = 0
        else:
            repetition_ratio = 0
        
        # === Feature 4: Query-response relevance (basic) ===
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response_lower))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'been', 'this', 'that', 'with', 'they', 'from',
                      'what', 'which', 'when', 'where', 'how', 'who', 'will',
                      'would', 'there', 'their', 'about', 'into', 'more', 'some'}
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        # === Feature 5: Structural coherence indicators ===
        has_explanation = bool(re.search(r'\b(?:because|since|therefore|thus|hence|as\s+a\s+result|due\s+to|this\s+means)\b', response_lower))
        has_examples = bool(re.search(r'\b(?:for\s+example|for\s+instance|such\s+as|e\.g\.|including)\b', response_lower))
        has_conditional = bool(re.search(r'\b(?:if|unless|depending\s+on|in\s+(?:the\s+)?case)\b', response_lower))
        
        # === Feature 6: Check for garbage / code dump / HTML dump ===
        code_ratio = len(re.findall(r'[{}\[\]<>/\\=;#]', response_stripped)) / max(len(response_stripped), 1)
        is_code_heavy = code_ratio > 0.08
        
        # === Scoring ===
        score = 5.0  # Base score
        
        # Epistemic calibration bonus (main feature)
        num_sentences = max(len(sentences), 1)
        
        # Hedging density: proportion of epistemic/hedging markers per sentence
        hedging_density = (epistemic_count + hedging_count * 0.5) / num_sentences
        # Optimal hedging density: some but not excessive (0.2 to 1.5 per sentence)
        if hedging_density <= 1.5:
            hedging_score = hedging_density * 1.5  # Up to ~2.25 bonus
        else:
            hedging_score = 1.5 * 1.5 - (hedging_density - 1.5) * 0.3  # Slight penalty for over-hedging
        hedging_score = max(0, min(hedging_score, 2.5))
        score += hedging_score
        
        # Sentence-level hedging ratio
        if claim_sentences > 0:
            hedge_ratio = hedged_sentences / claim_sentences
            # Reward moderate hedging (not 0%, not 100%)
            if 0.1 <= hedge_ratio <= 0.7:
                score += 0.8
            elif hedge_ratio > 0:
                score += 0.3
        
        # Absolutist penalty
        absolutist_density = absolutist_count / num_sentences
        score -= min(absolutist_density * 1.2, 2.0)
        
        # Sentence-level absolutism penalty
        if claim_sentences > 0:
            abs_ratio = absolutist_sentences / claim_sentences
            score -= abs_ratio * 1.0
        
        # Nuance bonus
        nuance_bonus = min(nuance_count * 0.5, 1.5)
        score += nuance_bonus
        
        # Structural coherence bonuses
        if has_explanation:
            score += 0.3
        if has_examples:
            score += 0.2
        if has_conditional:
            score += 0.3
        
        # Length factor: short responses tend to be worse
        score *= (0.4 + 0.6 * length_factor)
        
        # Relevance factor
        score *= (0.5 + 0.5 * min(relevance, 1.0))
        
        # Repetition penalty
        if repetition_ratio > 0.15:
            score *= max(0.5, 1.0 - (repetition_ratio - 0.15) * 3)
        
        # Code dump penalty
        if is_code_heavy:
            score *= 0.6
        
        # Very short response penalty (< 10 words)
        if word_count < 10:
            score *= 0.5
        elif word_count < 5:
            score *= 0.25
        
        # Bonus for moderate-length, well-structured responses
        if 30 <= word_count <= 300 and len(sentences) >= 2:
            score += 0.5
        
        # Check for "difficulty" acknowledgment (strong epistemic signal)
        if re.search(r'\b(?:difficult|hard|challenging|tricky)\s+to\s+(?:say|determine|know|provide|give|answer)\b', response_lower):
            score += 0.8
        
        # Check for source attribution
        if re.search(r'\b(?:according\s+to|based\s+on|sources?\s+suggest|experts?\s+(?:say|believe|suggest))\b', response_lower):
            score += 0.5
        
        # Clamp to range
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 3.0