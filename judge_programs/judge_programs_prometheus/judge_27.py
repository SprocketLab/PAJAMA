def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Uses a substantially different approach from variants 1 and 2:
    - Focuses on discourse markers and epistemic language patterns
    - Analyzes hedging vs. absolutism ratio
    - Detects citation-like patterns and specific factual anchors
    - Penalizes hallucination red-flags and sensationalism
    - Evaluates empathy/acknowledgment patterns for subjective queries
    - Uses n-gram pattern matching for quality signals
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 1.0
        
        query = str(query)
        response = str(response)
        
        if len(response.strip()) < 20:
            return 1.0
        
        score = 5.0  # Start at midpoint
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = re.findall(r'\b[a-z]+\b', response_lower)
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        if num_words < 5:
            return 1.5
        
        # ===== 1. EPISTEMIC HEDGING ANALYSIS =====
        # Appropriate hedging signals intellectual honesty
        hedging_phrases = [
            r'\bperhaps\b', r'\bmaybe\b', r'\bmight\b', r'\bcould be\b',
            r'\bit\'?s possible\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\btend[s]? to\b',
            r'\bin many cases\b', r'\bsome\b', r'\blikely\b',
            r'\bprobably\b', r'\bappears?\b', r'\bseems?\b',
            r'\bsuggest[s]?\b', r'\bmay\b', r'\bcan\b',
            r'\bit depends\b', r'\bwhile\b', r'\balthough\b',
            r'\bhowever\b', r'\bon the other hand\b',
            r'\bnot necessarily\b', r'\bin general\b',
        ]
        hedge_count = 0
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_ratio = hedge_count / num_words
        # Moderate hedging is good (0.01-0.05 ratio)
        if 0.005 < hedge_ratio < 0.06:
            score += 0.6
        elif hedge_ratio >= 0.06:
            score += 0.2  # Too much hedging can seem uncertain
        
        # ===== 2. ABSOLUTISM / OVERCONFIDENCE DETECTION =====
        absolutist_phrases = [
            r'\balways\b', r'\bnever\b', r'\beveryone knows\b',
            r'\bobviously\b', r'\bclearly\b', r'\bundoubtedly\b',
            r'\bwithout a doubt\b', r'\babsolutely\b', r'\bdefinitely\b',
            r'\bno question\b', r'\bguaranteed\b', r'\bimpossible\b',
            r'\bthe only way\b', r'\bthe best\b', r'\bthe worst\b',
            r'\beveryone\b', r'\bnobody\b', r'\bnothing\b',
            r'\beverything\b', r'\bcompletely\b', r'\btotally\b',
            r'\bjust\b.*\bjust\b',  # repeated "just" is dismissive
        ]
        absolutist_count = 0
        for pattern in absolutist_phrases:
            absolutist_count += len(re.findall(pattern, response_lower))
        
        absolutist_ratio = absolutist_count / num_words
        if absolutist_ratio > 0.03:
            score -= 1.0
        elif absolutist_ratio > 0.015:
            score -= 0.5
        
        # ===== 3. DISMISSIVE LANGUAGE DETECTION =====
        dismissive_patterns = [
            r'\bjust get over\b', r'\bjust move on\b', r'\bjust do\b',
            r'\bstop being\b', r'\byou should be\b', r'\bthat\'?s life\b',
            r'\bdeal with it\b', r'\bget over it\b', r'\bman up\b',
            r'\bit\'?s not that\b.*\bbad\b', r'\byou\'?re overreacting\b',
            r'\bjust\s+\w+\s+it\b',  # "just handle it" patterns
            r'\byou need to get\b.*\btogether\b',
            r'\bnothing wrong with\b',
        ]
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        if dismissive_count > 0:
            score -= dismissive_count * 0.7
        
        # ===== 4. EMPATHY AND ACKNOWLEDGMENT PATTERNS =====
        empathy_markers = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'?s understandable\b',
            r'\bcompletely understandable\b', r'\bit\'?s okay\b',
            r'\bit\'?s natural\b', r'\bit\'?s perfectly\b',
            r'\bi\'?m sorry\b', r'\bi hear\b', r'\bi can hear\b',
            r'\byour feelings\b', r'\byour experience\b',
            r'\bwe value\b', r'\bwe appreciate\b',
            r'\bsincerely\b', r'\bgenuinely\b',
            r'\blet\'?s\b',  # collaborative language
            r'\btogether\b', r'\bwe can\b',
            r'\bfeel free\b', r'\bdon\'?t hesitate\b',
        ]
        empathy_count = 0
        for pattern in empathy_markers:
            empathy_count += len(re.findall(pattern, response_lower))
        
        # Check if query seems emotional/personal
        emotional_query_signals = [
            r'\bfeel\b', r'\bfrustrat\b', r'\bsad\b', r'\bstress\b',
            r'\bworr\b', r'\banxi\b', r'\bhelp\b', r'\bcomfort\b',
            r'\bdevast\b', r'\blonely\b', r'\bdespair\b', r'\bpain\b',
            r'\bheartbr\b', r'\bexhaust\b', r'\bstruggl\b',
            r'\bseek\b', r'\bsupport\b', r'\badvice\b',
        ]
        is_emotional_query = sum(1 for p in emotional_query_signals 
                                  if re.search(p, query_lower)) >= 2
        
        if is_emotional_query:
            if empathy_count >= 3:
                score += 1.2
            elif empathy_count >= 1:
                score += 0.5
            else:
                score -= 0.8
        else:
            if empathy_count >= 1:
                score += 0.3
        
        # ===== 5. STRUCTURAL QUALITY INDICATORS =====
        # Numbered lists / structured advice (different from simple bullet detection)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+\w', response)
        has_structure = len(numbered_items) >= 2
        
        # Colon-based explanations (e.g., "Concept: explanation")
        colon_explanations = re.findall(r'\b\w+\s*:\s+[A-Z]', response)
        
        if has_structure:
            score += 0.5
        if len(colon_explanations) >= 2:
            score += 0.3
        
        # ===== 6. SPECIFICITY AND CONCRETENESS =====
        # Numbers and specific quantities (factual anchors)
        specific_numbers = re.findall(r'\b\d+\.?\d*\b', response)
        # Named entities (capitalized multi-word phrases)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response)
        
        specificity_score = min(len(specific_numbers) * 0.1 + len(named_entities) * 0.15, 0.8)
        score += specificity_score
        
        # ===== 7. DISCOURSE COHERENCE via TRANSITION WORDS =====
        transitions = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bin addition\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bin other words\b',
            r'\btherefore\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bin conclusion\b', r'\bto summarize\b',
            r'\bon one hand\b', r'\bin contrast\b',
            r'\bspecifically\b', r'\bnamely\b',
            r'\bhere\b', r'\bnow\b.*\blet\b',
            r'\bremember\b', r'\bkeep in mind\b',
            r'\bthis means\b', r'\bthis is\b',
        ]
        transition_count = 0
        for pattern in transitions:
            transition_count += len(re.findall(pattern, response_lower))
        
        transition_ratio = transition_count / num_sentences
        if transition_ratio > 0.3:
            score += 0.6
        elif transition_ratio > 0.15:
            score += 0.3
        
        # ===== 8. RESPONSE COMPLETENESS =====
        # Check if response seems cut off (ends mid-sentence without punctuation)
        stripped_response = response.rstrip()
        if stripped_response and stripped_response[-1] not in '.!?)"\'':
            # Likely truncated - mild penalty as both responses may be truncated
            pass  # Don't penalize since examples show truncation is common
        
        # Response length adequacy relative to query complexity
        query_words = len(re.findall(r'\b\w+\b', query_lower))
        response_adequacy = num_words / max(query_words, 1)
        if response_adequacy > 1.5:
            score += 0.4
        elif response_adequacy < 0.5:
            score -= 0.5
        
        # ===== 9. SENSATIONALISM AND CONSPIRACY DETECTION =====
        sensational_patterns = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind-?blowing\b',
            r'\bthey don\'?t want you to know\b', r'\bsecret\b.*\bthey\b',
            r'\bwake up\b', r'\bsheeple\b', r'\bconspiracy\b',
            r'\bcover[- ]?up\b', r'\bhidden truth\b', r'\bexposed\b',
            r'\b(?:big )?pharma\b', r'\bmainstream media\b',
            r'\bmiracl\b', r'\bbreakthrough\b.*\bthey\b',
        ]
        sensational_count = 0
        for pattern in sensational_patterns:
            sensational_count += len(re.findall(pattern, response_lower))
        
        if sensational_count > 0:
            score -= sensational_count * 1.0
        
        # ===== 10. HALLUCINATION RED FLAGS =====
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+\s*%', response)
        unsourced_claims = re.findall(r'\bstudies show\b|\bresearch proves\b|\bscientists say\b', response_lower)
        
        if len(precise_stats) > 2 and len(unsourced_claims) > 0:
            score -= 0.8
        
        # Fabricated-sounding specific directions/instructions without context
        if re.search(r'\btake a (?:left|right)\b.*\bturn\b', response_lower):
            # Could be hallucinated directions
            if 'direction' not in query_lower and 'navigate' not in query_lower:
                score -= 1.5
        
        # ===== 11. QUERY-RESPONSE ALIGNMENT =====
        # Check if response addresses the query's core concern
        query_key_terms = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        response_terms = set(words)
        
        if len(query_key_terms) > 0:
            overlap = len(query_key_terms & response_terms) / len(query_key_terms)
            score += overlap * 0.8
        
        # ===== 12. TONE APPROPRIATENESS =====
        # Casual/dismissive tone detection via short choppy sentences
        avg_sentence_len = num_words / num_sentences
        
        very_short_sentences = sum(1 for s in sentences if len(s.split()) <= 4)
        short_ratio = very_short_sentences / num_sentences
        
        if short_ratio > 0.5 and is_emotional_query:
            score -= 0.6  # Too terse for emotional context
        
        # ===== 13. EXPLANATION DEPTH =====
        # Causal/explanatory language
        explanatory_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\bdue to\b',
            r'\bthis (?:is|means|helps|allows|ensures)\b',
            r'\bso that\b', r'\bin order to\b', r'\bwhich (?:means|helps|allows)\b',
            r'\bthe reason\b', r'\bas a result\b',
            r'\bimagine\b', r'\bthink of\b', r'\bconsider\b',
        ]
        explanatory_count = 0
        for pattern in explanatory_patterns:
            explanatory_count += len(re.findall(pattern, response_lower))
        
        explanation_ratio = explanatory_count / num_sentences
        if explanation_ratio > 0.3:
            score += 0.7
        elif explanation_ratio > 0.15:
            score += 0.3
        
        # ===== 14. ACTIONABLE ADVICE DETECTION =====
        actionable_patterns = [
            r'\btry\b', r'\bconsider\b', r'\byou (?:can|could|might)\b',
            r'\bstart (?:by|with)\b', r'\bmake sure\b',
            r'\bone (?:way|approach|method)\b', r'\bhere(?:\'s| are)\b',
            r'\bstep\b', r'\btip\b', r'\bstrateg\b',
            r'\bbreak\b.*\bdown\b', r'\bfocus on\b',
        ]
        actionable_count = 0
        for pattern in actionable_patterns:
            actionable_count += len(re.findall(pattern, response_lower))
        
        if actionable_count >= 3:
            score += 0.5
        elif actionable_count >= 1:
            score += 0.2
        
        # ===== 15. SECOND-PERSON ENGAGEMENT =====
        # Responses that directly address the user tend to be more helpful
        you_count = len(re.findall(r'\byou(?:r|\'re|\'ve|\'ll)?\b', response_lower))
        you_ratio = you_count / num_words
        
        if 0.02 < you_ratio < 0.08:
            score += 0.4
        elif you_ratio >= 0.08:
            score += 0.1  # Slightly too much
        
        # ===== 16. NEGATIVE COMMAND DETECTION =====
        # "You need to..." "You should..." in commanding tone
        commanding = re.findall(r'\byou (?:need|should|must|have) to\b', response_lower)
        if len(commanding) > 2:
            score -= 0.4
        
        # ===== FINAL NORMALIZATION =====
        # Clamp to 1-10 range and round
        score = max(1.0, min(10.0, score))
        
        # Map to 1-5 scale to match training data
        # Linear mapping from [1,10] to [1,5]
        final_score = 1.0 + (score - 1.0) * (4.0 / 9.0)
        final_score = max(1.0, min(5.0, round(final_score, 2)))
        
        return final_score
        
    except Exception as e:
        return 3.0  # Safe midpoint on error