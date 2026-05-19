def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Claim density analysis - ratio of assertive claims to total sentences
    2. Evidence attribution patterns (citations, references to sources)
    3. Conditional/nuanced reasoning detection (if-then, depends on, etc.)
    4. Spectrum of certainty language (not just binary hedge/confident)
    5. Question complexity vs response certainty mismatch detection
    6. Structural sophistication (organized thinking suggests calibrated reasoning)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        resp_len = len(response)
        
        # Split into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Claim density & assertive statement analysis
        # Count absolute/universal claims vs qualified ones
        # ============================================================
        
        absolute_markers = [
            r'\bis\b(?!\s+(?:likely|possibly|probably|perhaps|often|sometimes|generally))',
            r'\balways\b', r'\bnever\b', r'\bevery\b', r'\bno one\b',
            r'\bwithout a doubt\b', r'\bundeniably\b', r'\bunquestionably\b',
            r'\bobviously\b', r'\bclearly\b', r'\bdefinitely\b',
            r'\babsolutely\b', r'\bcertainly\b', r'\bwithout question\b',
            r'\bthe fact is\b', r'\bthe truth is\b', r'\bit is clear\b',
            r'\bno doubt\b', r'\bplain and simple\b', r'\bperiod\b',
            r'\bguaranteed\b', r'\bproven\b', r'\bimpossible\b',
            r'\bmust be\b', r'\bcannot be\b',
        ]
        
        absolute_count = 0
        for pattern in absolute_markers:
            absolute_count += len(re.findall(pattern, response_lower))
        
        # Normalize by sentence count
        absolute_density = absolute_count / num_sentences
        # Penalize high absolute density (score 0-1, lower is better for calibration)
        absolute_penalty = min(absolute_density * 0.8, 2.0)
        
        # ============================================================
        # FEATURE 2: Evidence attribution & source referencing
        # ============================================================
        
        evidence_patterns = [
            r'\baccording to\b', r'\bresearch\s+(suggests?|shows?|indicates?|finds?)\b',
            r'\bstudies?\s+(suggest|show|indicate|find|have shown)\b',
            r'\bexperts?\s+(suggest|say|believe|recommend|note)\b',
            r'\bevidence\s+(suggests?|shows?|indicates?)\b',
            r'\bdata\s+(suggests?|shows?|indicates?)\b',
            r'\bgenerally\s+(?:speaking|accepted|considered|regarded)\b',
            r'\bin\s+(?:many|most|some)\s+cases\b',
            r'\btypically\b', r'\bcommonly\b', r'\btraditionally\b',
            r'\bhistorically\b', r'\bgenerally\b',
            r'\bwidely\s+(?:considered|regarded|accepted|believed)\b',
            r'\bsources?\s+(?:suggest|indicate|say)\b',
            r'\breported\b', r'\bdocumented\b',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, response_lower))
        
        evidence_score = min(evidence_count * 0.4, 3.0)
        
        # ============================================================
        # FEATURE 3: Conditional & nuanced reasoning
        # ============================================================
        
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bdepends?\s+on\b', r'\bdepending\s+on\b',
            r'\bin\s+some\s+cases\b', r'\bin\s+certain\s+(?:cases|situations|contexts)\b',
            r'\bit\s+depends\b', r'\bvaries?\b', r'\bcontext\b',
            r'\bon\s+the\s+other\s+hand\b', r'\bhowever\b', r'\balthough\b',
            r'\bwhile\b.*\b(?:also|but|however)\b', r'\bthat\s+said\b',
            r'\bnevertheless\b', r'\bnuance\b', r'\bsubtlet(?:y|ies)\b',
            r'\btrade-?off\b', r'\bbalance\b.*\bbetween\b',
            r'\bpros?\s+and\s+cons?\b', r'\badvantages?\s+and\s+disadvantages?\b',
            r'\bon\s+one\s+hand\b', r'\bconversely\b',
            r'\bwhereas\b', r'\balternatively\b',
            r'\bnot\s+necessarily\b', r'\bnot\s+always\b',
        ]
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        conditional_score = min(conditional_count * 0.35, 3.0)
        
        # ============================================================
        # FEATURE 4: Graduated certainty spectrum
        # Instead of binary hedge/confident, map to a spectrum
        # ============================================================
        
        high_certainty = [
            r'\bwill\b', r'\bmust\b', r'\bis\s+(?:the|a)\s+fact\b',
            r'\bundoubtedly\b', r'\bwithout\s+(?:a\s+)?doubt\b',
        ]
        
        medium_certainty = [
            r'\blikely\b', r'\bprobably\b', r'\bgenerally\b',
            r'\busually\b', r'\btends?\s+to\b', r'\boften\b',
            r'\bin\s+most\s+cases\b', r'\btypically\b',
        ]
        
        low_certainty = [
            r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bmay\b', r'\bsometimes\b', r'\boccasionally\b',
            r'\bappears?\s+to\b', r'\bseems?\s+to\b',
            r'\bsuggest\b', r'\bpotentially\b',
            r'\bi\s+think\b', r'\bi\s+believe\b',
            r'\bit\'s\s+(?:possible|plausible)\b',
        ]
        
        high_cert_count = sum(len(re.findall(p, response_lower)) for p in high_certainty)
        med_cert_count = sum(len(re.findall(p, response_lower)) for p in medium_certainty)
        low_cert_count = sum(len(re.findall(p, response_lower)) for p in low_certainty)
        
        total_cert_markers = high_cert_count + med_cert_count + low_cert_count + 0.001
        
        # A well-calibrated response uses a mix, especially medium and low certainty
        certainty_diversity = 0.0
        for count in [high_cert_count, med_cert_count, low_cert_count]:
            if count > 0:
                p = count / total_cert_markers
                certainty_diversity -= p * math.log(p + 1e-10)
        
        # Normalize entropy (max entropy for 3 categories is ln(3) ≈ 1.099)
        max_entropy = math.log(3)
        certainty_diversity_norm = certainty_diversity / max_entropy if max_entropy > 0 else 0
        
        certainty_spectrum_score = certainty_diversity_norm * 2.0
        
        # Bonus for having medium/low certainty markers present
        has_graduated_language = min((med_cert_count + low_cert_count) * 0.25, 2.0)
        
        # ============================================================
        # FEATURE 5: Query complexity vs response certainty mismatch
        # ============================================================
        
        # Detect if query is about subjective/ambiguous/speculative topics
        ambiguity_indicators = [
            r'\bdo you think\b', r'\bwhat do you\b', r'\bshould\b',
            r'\bopinion\b', r'\bbelieve\b', r'\bfeel\b',
            r'\bwhat\s+(?:is|are)\s+the\s+best\b', r'\bhow\s+(?:can|should|would)\b',
            r'\bwhy\s+(?:is|are|do|does|did)\b',
            r'\bwhat\s+(?:would|could|might)\b',
            r'\bpossible\b', r'\bfuture\b', r'\bpredict\b',
            r'\bcontrovers\b', r'\bdebat\b',
            r'\bwhat\'s\s+happening\b', r'\bwhat\s+happened\b',
        ]
        
        query_ambiguity = sum(1 for p in ambiguity_indicators if re.search(p, query_lower))
        is_ambiguous_query = query_ambiguity >= 1
        
        # If query is ambiguous but response is highly certain, penalize
        mismatch_penalty = 0.0
        if is_ambiguous_query:
            if high_cert_count > med_cert_count + low_cert_count and absolute_count > 2:
                mismatch_penalty = 1.5
            # Reward acknowledging ambiguity
            if low_cert_count > 0 or med_cert_count > 0:
                mismatch_penalty -= 0.5
        
        # ============================================================
        # FEATURE 6: Structural sophistication
        # Well-organized responses with steps, sections, multiple perspectives
        # ============================================================
        
        # Count organizational markers
        has_numbered_list = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)) > 0
        has_headers = len(re.findall(r'(?:^|\n)\s*#{1,4}\s', response)) > 0
        has_bold = len(re.findall(r'\*\*[^*]+\*\*', response)) > 0
        has_bullet_list = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response)) > 0
        
        structure_score = 0.0
        if has_numbered_list:
            structure_score += 0.5
        if has_headers:
            structure_score += 0.5
        if has_bold:
            structure_score += 0.3
        if has_bullet_list:
            structure_score += 0.3
        
        # ============================================================
        # FEATURE 7: Response length and information density
        # ============================================================
        
        # Moderate length is generally better
        words = response.split()
        word_count = len(words)
        
        # Optimal range: 80-300 words
        if word_count < 20:
            length_score = 0.0
        elif word_count < 50:
            length_score = 0.5
        elif word_count < 80:
            length_score = 1.0
        elif word_count <= 300:
            length_score = 1.5
        elif word_count <= 500:
            length_score = 1.2
        else:
            length_score = 1.0
        
        # ============================================================
        # FEATURE 8: Explanation depth - causal reasoning words
        # ============================================================
        
        causal_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas\s+a\s+result\b', r'\bconsequently\b', r'\bdue\s+to\b',
            r'\bthis\s+(?:means|implies|suggests|indicates)\b',
            r'\bfor\s+(?:this|that)\s+reason\b', r'\bso\s+that\b',
            r'\bin\s+order\s+to\b', r'\bleading\s+to\b',
            r'\bwhich\s+(?:means|causes|leads|results)\b',
        ]
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_patterns)
        causal_score = min(causal_count * 0.3, 2.0)
        
        # ============================================================
        # FEATURE 9: Engagement & reader-awareness
        # ============================================================
        
        engagement_patterns = [
            r'\bhere\s+(?:are|is)\b', r'\blet\'s\b', r'\byou\s+(?:can|might|could|may)\b',
            r'\byou\'ll\b', r'\byour\b', r'\bconsider\b',
            r'\bkeep\s+in\s+mind\b', r'\bnote\s+that\b', r'\bimportant(?:ly)?\b',
            r'\bremember\b', r'\bfor\s+(?:example|instance)\b',
            r'\bsuch\s+as\b', r'\be\.g\.\b', r'\bi\.e\.\b',
        ]
        
        engagement_count = sum(len(re.findall(p, response_lower)) for p in engagement_patterns)
        engagement_score = min(engagement_count * 0.2, 1.5)
        
        # ============================================================
        # FEATURE 10: Lexical sophistication (unique word ratio)
        # ============================================================
        
        words_lower = [w.lower().strip('.,!?;:()[]{}"\'-') for w in words]
        words_lower = [w for w in words_lower if len(w) > 0]
        if len(words_lower) > 0:
            unique_ratio = len(set(words_lower)) / len(words_lower)
        else:
            unique_ratio = 0
        
        # Higher unique ratio suggests more diverse vocabulary
        vocab_score = unique_ratio * 1.5
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        total_score = (
            evidence_score           # 0-3: evidence attribution
            + conditional_score      # 0-3: conditional/nuanced reasoning
            + certainty_spectrum_score  # 0-2: diversity of certainty levels
            + has_graduated_language  # 0-2: medium/low certainty markers
            + structure_score        # 0-1.6: structural organization
            + length_score           # 0-1.5: appropriate length
            + causal_score           # 0-2: causal reasoning
            + engagement_score       # 0-1.5: reader engagement
            + vocab_score            # 0-1.5: vocabulary diversity
            - absolute_penalty       # 0-2: penalty for absolutism
            - mismatch_penalty       # 0-1.5: penalty for certainty mismatch
        )
        
        # Clamp to 0-10 range
        total_score = max(0.0, min(10.0, total_score))
        
        return round(total_score, 3)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            return max(0.0, min(10.0, len(response) / 200.0))
        except:
            return 2.0