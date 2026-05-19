def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Claim density analysis - ratio of assertive claims to total sentences
    2. Source attribution and evidence referencing patterns
    3. Conditional/nuanced reasoning detection (if-then, depends on, etc.)
    4. Epistemic verb analysis (know, believe, think, suggest, indicate)
    5. Gradation of certainty levels across the response
    6. Question complexity vs response certainty mismatch detection
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
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Epistemic verb spectrum analysis
        # Categorize verbs by their epistemic strength
        # ============================================================
        
        # Strong epistemic verbs (appropriate uncertainty)
        soft_epistemic = [
            r'\bmight\b', r'\bcould\b', r'\bmay\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\btends to\b', r'\bappears to\b',
            r'\bseems to\b', r'\bseems like\b', r'\bit\'s possible\b',
            r'\bone possibility\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\bsometimes\b',
            r'\bin some cases\b', r'\bin many cases\b',
            r'\bcan vary\b', r'\bvaries\b', r'\bdepending on\b',
            r'\bto some extent\b', r'\bto a degree\b',
        ]
        
        # Evidence/source referencing patterns
        evidence_patterns = [
            r'\bresearch\s+(suggests?|shows?|indicates?|has\s+shown)\b',
            r'\bstudies\s+(suggest|show|indicate|have\s+shown)\b',
            r'\baccording to\b', r'\bevidence\s+(suggests?|indicates?)\b',
            r'\bexperts\s+(say|suggest|believe|recommend)\b',
            r'\bdata\s+(suggests?|shows?|indicates?)\b',
            r'\bliterature\b', r'\bfindings\b',
            r'\bsource\b', r'\breported\b', r'\bdocumented\b',
        ]
        
        # Conditional/nuanced reasoning
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bit depends\b', r'\bdepends on\b',
            r'\bin this case\b', r'\bin that case\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\balthough\b',
            r'\bwhile\b.*\b(also|but|however)\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bthere are (several|multiple|different|various)\b',
            r'\bconversely\b', r'\balternatively\b',
            r'\bwhereas\b', r'\bnonetheless\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bthat being said\b',
            r'\bon one hand\b', r'\bbalance\b',
        ]
        
        # Overconfident/absolutist patterns
        absolutist_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b',
            r'\babsolutely\b', r'\bwithout (a )?doubt\b',
            r'\bcertainly\b', r'\bundoubtedly\b', r'\bunquestionably\b',
            r'\bobviously\b', r'\bclearly\b', r'\bof course\b',
            r'\beveryone knows\b', r'\bit is (a )?fact\b',
            r'\bthe truth is\b', r'\bno question\b',
            r'\bguaranteed\b', r'\bwithout exception\b',
            r'\bin every case\b', r'\bno doubt\b',
        ]
        
        # Count matches for each category
        soft_count = sum(len(re.findall(p, response_lower)) for p in soft_epistemic)
        evidence_count = sum(len(re.findall(p, response_lower)) for p in evidence_patterns)
        conditional_count = sum(len(re.findall(p, response_lower)) for p in conditional_patterns)
        absolutist_count = sum(len(re.findall(p, response_lower)) for p in absolutist_patterns)
        
        # ============================================================
        # FEATURE 2: Claim density - declarative assertions per sentence
        # ============================================================
        
        # Patterns that indicate strong declarative claims
        declarative_markers = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bwill\b', r'\bmust\b', r'\bshould\b', r'\bneed to\b',
        ]
        
        declarative_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_declarative = any(re.search(p, sent_lower) for p in declarative_markers)
            # Check if sentence also has softening
            has_softening = any(re.search(p, sent_lower) for p in soft_epistemic)
            if has_declarative and not has_softening:
                declarative_count += 1
        
        claim_density = declarative_count / num_sentences
        
        # ============================================================
        # FEATURE 3: Perspective acknowledgment
        # ============================================================
        
        perspective_patterns = [
            r'\bfrom (one|a|another|this|that) perspective\b',
            r'\bsome (people|experts|researchers|argue|believe|think|say)\b',
            r'\bothers (argue|believe|think|say|suggest|contend)\b',
            r'\bthere (is|are) (debate|disagreement|discussion)\b',
            r'\bopinions (vary|differ)\b', r'\bviews (vary|differ)\b',
            r'\bone (view|school of thought|argument)\b',
            r'\banother (view|school of thought|argument)\b',
            r'\bproponents\b', r'\bcritics\b', r'\bskeptics\b',
            r'\badvocates\b', r'\bsupporters\b', r'\bopponents\b',
        ]
        
        perspective_count = sum(len(re.findall(p, response_lower)) for p in perspective_patterns)
        
        # ============================================================
        # FEATURE 4: Query ambiguity/complexity assessment
        # ============================================================
        
        # Detect if query is about subjective/ambiguous/speculative topics
        ambiguity_indicators = [
            r'\bwhat do you think\b', r'\bdo you believe\b',
            r'\bshould\b', r'\bis it (true|possible|likely)\b',
            r'\bwhy (do|does|is|are)\b', r'\bhow (do|does|can|should)\b',
            r'\bwhat (causes?|leads?)\b', r'\bopinion\b',
            r'\bfuture\b', r'\bpredict\b', r'\bwill\b',
            r'\bbest\b', r'\bworst\b', r'\bmost\b',
        ]
        
        query_ambiguity = sum(1 for p in ambiguity_indicators if re.search(p, query_lower))
        is_ambiguous_query = query_ambiguity >= 1
        
        # Factual/procedural query indicators
        factual_indicators = [
            r'\bhow (to|do I|can I)\b', r'\bwhat (is|are) the steps\b',
            r'\brecipe\b', r'\bdirections\b', r'\binstructions\b',
            r'\bcalculate\b', r'\bsolve\b', r'\bfind\b',
        ]
        is_factual_query = sum(1 for p in factual_indicators if re.search(p, query_lower)) >= 1
        
        # ============================================================
        # FEATURE 5: Structural sophistication (different from other variants)
        # Measure: paragraph transitions, logical connectors, multi-part reasoning
        # ============================================================
        
        logical_connectors = [
            r'\bfirst(ly)?\b', r'\bsecond(ly)?\b', r'\bthird(ly)?\b',
            r'\badditionally\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bin addition\b', r'\bfor (example|instance)\b',
            r'\bspecifically\b', r'\bin particular\b',
            r'\bnamely\b', r'\bsuch as\b', r'\bincluding\b',
            r'\bin summary\b', r'\boverall\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\btherefore\b', r'\bthus\b',
            r'\bas a result\b', r'\bconsequently\b',
        ]
        
        connector_count = sum(len(re.findall(p, response_lower)) for p in logical_connectors)
        
        # ============================================================
        # FEATURE 6: Self-awareness / limitation acknowledgment
        # ============================================================
        
        limitation_patterns = [
            r'\bi\'m not (sure|certain|aware)\b',
            r'\bi don\'t (know|have)\b',
            r'\bthis (may|might) not be\b',
            r'\bkeep in mind\b', r'\bnote that\b',
            r'\bit\'s (worth|important to) not(e|ing)\b',
            r'\blimitation\b', r'\bcaveat\b', r'\bdisclaimer\b',
            r'\bplease (note|consult|check|verify)\b',
            r'\bconsult (a|your|an)\b', r'\bverify\b',
            r'\bthis is not (medical|legal|financial) advice\b',
        ]
        
        limitation_count = sum(len(re.findall(p, response_lower)) for p in limitation_patterns)
        
        # ============================================================
        # FEATURE 7: Proportional word diversity (lexical richness)
        # More diverse vocabulary often correlates with more nuanced responses
        # ============================================================
        
        words = re.findall(r'\b[a-z]+\b', response_lower)
        num_words = max(len(words), 1)
        unique_words = len(set(words))
        
        # Type-token ratio adjusted for length
        if num_words > 0:
            ttr = unique_words / math.sqrt(num_words)  # Guiraud's index
        else:
            ttr = 0
        
        # ============================================================
        # FEATURE 8: Response completeness and engagement
        # ============================================================
        
        # Check for truncation
        is_truncated = response.rstrip()[-1] not in '.!?"\')' if response.rstrip() else True
        
        # Engagement markers
        engagement_patterns = [
            r'\blet\'s\b', r'\byou (can|might|could|may)\b',
            r'\bhere (are|is)\b', r'\bconsider\b',
            r'\bthink about\b', r'\bimagine\b',
            r'\bfor (your|reference)\b',
        ]
        engagement_count = sum(len(re.findall(p, response_lower)) for p in engagement_patterns)
        
        # ============================================================
        # SCORING
        # ============================================================
        
        score = 50.0  # Base score
        
        # Soft epistemic language (good for uncertainty communication)
        soft_per_sent = soft_count / num_sentences
        score += min(soft_per_sent * 8, 12)  # Up to +12
        
        # Evidence referencing (good)
        score += min(evidence_count * 3, 9)  # Up to +9
        
        # Conditional/nuanced reasoning (good)
        cond_per_sent = conditional_count / num_sentences
        score += min(cond_per_sent * 10, 10)  # Up to +10
        
        # Perspective acknowledgment (good for ambiguous queries)
        if is_ambiguous_query:
            score += min(perspective_count * 3, 6)
        else:
            score += min(perspective_count * 1.5, 3)
        
        # Absolutist language penalty
        abs_per_sent = absolutist_count / num_sentences
        if is_ambiguous_query:
            score -= min(abs_per_sent * 12, 15)  # Heavier penalty for ambiguous topics
        else:
            score -= min(abs_per_sent * 5, 8)
        
        # Claim density penalty (high unhedged claim density is bad)
        if claim_density > 0.8:
            score -= (claim_density - 0.8) * 10
        
        # Logical connectors (structural quality)
        conn_per_sent = connector_count / num_sentences
        score += min(conn_per_sent * 4, 6)  # Up to +6
        
        # Limitation acknowledgment (good)
        score += min(limitation_count * 2, 4)  # Up to +4
        
        # Lexical richness bonus
        if ttr > 6:
            score += min((ttr - 6) * 2, 5)  # Up to +5
        
        # Engagement bonus
        score += min(engagement_count * 1.0, 3)  # Up to +3
        
        # Truncation penalty
        if is_truncated:
            score -= 2
        
        # Length considerations - very short responses may lack nuance
        if num_words < 20:
            score -= 5
        elif num_words < 50:
            score -= 2
        
        # Moderate length bonus (enough room for nuance)
        if 80 < num_words < 500:
            score += 3
        elif num_words >= 500:
            score += 2
        
        # For factual/procedural queries, reward clear structure more
        if is_factual_query:
            # Check for numbered steps or structured format
            has_steps = bool(re.search(r'(\d+[\.\)]\s|\bstep\s+\d)', response_lower))
            if has_steps:
                score += 3
        
        # Bonus for balanced response (has both confident and hedged language)
        if soft_count > 0 and declarative_count > 0:
            balance_ratio = min(soft_count, declarative_count) / max(soft_count, declarative_count)
            score += balance_ratio * 4  # Up to +4 for good balance
        
        # Clamp score
        score = max(0, min(100, score))
        
        return round(score, 2)
        
    except Exception:
        return 50.0