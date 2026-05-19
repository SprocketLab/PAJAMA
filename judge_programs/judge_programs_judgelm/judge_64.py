def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using a
    novel approach based on:
    1. Claim density analysis (ratio of assertive statements to total content)
    2. Epistemic stance markers categorized by strength
    3. Source attribution patterns
    4. Conditional/qualifying clause detection
    5. Response coherence and completeness signals
    
    This variant focuses on STRUCTURAL patterns of epistemic reasoning
    rather than simple keyword counting.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        resp_stripped = response.strip()
        if len(resp_stripped) < 2:
            return 0.5
        
        resp_lower = resp_stripped.lower()
        query_lower = query.lower().strip()
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', resp_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', resp_lower)
        num_words = max(len(words), 1)
        
        # ============================================================
        # FEATURE 1: Epistemic Stance Spectrum Analysis
        # Categorize epistemic markers by their commitment level
        # ============================================================
        
        # Strong commitment (overconfident) markers
        strong_commit = [
            r'\bdefinitely\b', r'\bcertainly\b', r'\babsolutely\b',
            r'\bundoubtedly\b', r'\bwithout\s+(?:a\s+)?doubt\b',
            r'\bobviously\b', r'\bclearly\b', r'\bof\s+course\b',
            r'\balways\b', r'\bnever\b', r'\beveryone\s+knows\b',
            r'\bit\s+is\s+(?:a\s+)?fact\b', r'\bundeniably\b',
            r'\bincontrovertibly\b', r'\bguaranteed\b',
            r'\bno\s+question\b', r'\bwithout\s+exception\b',
            r'\bproven\b'
        ]
        
        # Moderate commitment markers (appropriate confidence)
        moderate_commit = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bcommonly\b', r'\bwidely\b', r'\boften\b',
            r'\bin\s+most\s+cases\b', r'\btends?\s+to\b',
            r'\bfor\s+the\s+most\s+part\b', r'\blargely\b',
            r'\bmainly\b', r'\bprimarily\b', r'\bnormally\b'
        ]
        
        # Epistemic hedges (appropriate uncertainty)
        hedges = [
            r'\bperhaps\b', r'\bpossibly\b', r'\bprobably\b',
            r'\blikely\b', r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bappears?\s+to\b', r'\bseems?\s+to\b',
            r'\bit\s+is\s+(?:also\s+)?possible\b',
            r'\bto\s+some\s+(?:extent|degree)\b',
            r'\bapproximately\b', r'\broughly\b', r'\baround\b',
            r'\bestimated\b', r'\bin\s+some\s+cases\b'
        ]
        
        # Source/evidence attribution markers
        source_markers = [
            r'\baccording\s+to\b', r'\bresearch\s+(?:suggests?|shows?|indicates?)\b',
            r'\bstudies?\s+(?:suggest|show|indicate|find|found)\b',
            r'\bexperts?\s+(?:say|believe|suggest|argue|note)\b',
            r'\bevidence\s+(?:suggests?|shows?|indicates?)\b',
            r'\bhistorically\b', r'\btraditionally\b',
            r'\bit\s+(?:is|has\s+been)\s+(?:widely\s+)?(?:reported|documented|noted|observed)\b',
            r'\bsources?\s+(?:say|suggest|indicate|report)\b',
            r'\bliterature\s+(?:suggests?|shows?)\b',
            r'\bdata\s+(?:suggests?|shows?|indicates?)\b'
        ]
        
        # Qualifying/conditional clauses
        qualifiers = [
            r'\bhowever\b', r'\balthough\b', r'\bthough\b',
            r'\bnevertheless\b', r'\bon\s+the\s+other\s+hand\b',
            r'\bthat\s+said\b', r'\bwhile\b.*\b(?:also|still|yet)\b',
            r'\bdepending\s+on\b', r'\bit\s+depends\b',
            r'\bin\s+(?:some|certain)\s+(?:cases|situations|contexts)\b',
            r'\bnot\s+(?:always|necessarily|entirely)\b',
            r'\bto\s+(?:a|some)\s+(?:degree|extent)\b',
            r'\bwith\s+(?:some|certain)\s+(?:exceptions?|caveats?|limitations?)\b',
            r'\bsubjective\b', r'\bcontroversial\b', r'\bdebatable\b',
            r'\bvarying?\b.*\bopinions?\b'
        ]
        
        def count_patterns(patterns, text):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text, re.IGNORECASE))
            return count
        
        strong_count = count_patterns(strong_commit, resp_lower)
        moderate_count = count_patterns(moderate_commit, resp_lower)
        hedge_count = count_patterns(hedges, resp_lower)
        source_count = count_patterns(source_markers, resp_lower)
        qualifier_count = count_patterns(qualifiers, resp_lower)
        
        # ============================================================
        # FEATURE 2: Claim Density via Declarative Statement Detection
        # ============================================================
        
        # Count declarative "X is Y" patterns (assertive claims)
        declarative_patterns = [
            r'\b(?:is|are|was|were)\s+(?:a|an|the)\b',
            r'\b(?:is|are|was|were)\s+(?:not\s+)?(?:considered|regarded|known)\b',
            r'\bthis\s+(?:is|means|shows|proves)\b',
            r'\bthe\s+(?:answer|reason|cause|result|fact)\s+is\b',
        ]
        declarative_count = count_patterns(declarative_patterns, resp_lower)
        
        # Claim density: ratio of assertive statements per sentence
        claim_density = declarative_count / num_sentences
        
        # ============================================================
        # FEATURE 3: Structural Coherence Signals
        # ============================================================
        
        # Check for repetitive content (sign of low quality)
        if num_sentences >= 3:
            sent_set = set()
            duplicates = 0
            for s in sentences:
                s_norm = re.sub(r'\s+', ' ', s.lower().strip())
                if s_norm in sent_set:
                    duplicates += 1
                sent_set.add(s_norm)
            repetition_ratio = duplicates / num_sentences
        else:
            repetition_ratio = 0.0
        
        # Check for truncation (incomplete response)
        truncated = 0
        if resp_stripped and resp_stripped[-1] not in '.!?"\')]}':
            # Might be truncated
            last_chars = resp_stripped[-20:]
            if re.search(r'\b\w+$', last_chars) and len(resp_stripped) > 100:
                truncated = 0.3  # mild penalty for possible truncation
        
        # Check for garbage/code/HTML content
        html_tags = len(re.findall(r'<[^>]+>', response))
        code_indicators = len(re.findall(r'(?:def |import |class |function |var |const |let )', resp_lower))
        garbage_ratio = (html_tags + code_indicators) / max(num_words, 1)
        
        # ============================================================
        # FEATURE 4: Response Substance and Relevance
        # ============================================================
        
        # Query word overlap (basic relevance)
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'been', 'from', 'this', 'that', 'with', 'they',
                      'what', 'how', 'who', 'where', 'when', 'why', 'which',
                      'will', 'would', 'could', 'should', 'about', 'into',
                      'more', 'some', 'than', 'them', 'then', 'these', 'also',
                      'make', 'just', 'any', 'each', 'does', 'did'}
        query_content_words = query_words - stop_words
        resp_words_set = set(words)
        
        if query_content_words:
            relevance = len(query_content_words & resp_words_set) / len(query_content_words)
        else:
            relevance = 0.5
        
        # Response length adequacy
        if num_words < 5:
            length_score = 0.1
        elif num_words < 15:
            length_score = 0.3
        elif num_words < 30:
            length_score = 0.5
        elif num_words < 80:
            length_score = 0.75
        elif num_words < 200:
            length_score = 1.0
        elif num_words < 400:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ============================================================
        # FEATURE 5: Proportional Epistemic Balance Score
        # ============================================================
        
        total_epistemic_markers = strong_count + moderate_count + hedge_count + source_count + qualifier_count
        
        if total_epistemic_markers > 0:
            # Ideal: low strong commitment, balanced moderate + hedges + qualifiers
            strong_ratio = strong_count / total_epistemic_markers
            good_markers = (moderate_count + hedge_count + source_count + qualifier_count)
            good_ratio = good_markers / total_epistemic_markers
            
            # Penalize high proportion of overconfident markers
            epistemic_balance = good_ratio - 0.5 * strong_ratio
        else:
            # No epistemic markers at all - neutral (short factual answers may not need them)
            if num_words < 20:
                epistemic_balance = 0.5  # short answers get a pass
            else:
                epistemic_balance = 0.3  # longer answers should have some markers
        
        # Bonus for source attribution
        source_bonus = min(source_count * 0.15, 0.5)
        
        # Bonus for qualifiers (showing nuanced thinking)
        qualifier_bonus = min(qualifier_count * 0.1, 0.4)
        
        # ============================================================
        # FEATURE 6: Topic Sensitivity Detection
        # ============================================================
        
        # Some topics inherently require more hedging
        sensitive_topics = [
            r'\b(?:believe|opinion|think|feel)\b',
            r'\b(?:best|worst|greatest|most\s+important)\b',
            r'\b(?:should|ought|must)\b',
            r'\b(?:controversial|debate|disagree)\b',
            r'\b(?:predict|future|forecast)\b',
            r'\b(?:cause|reason|why)\b',
            r'\b(?:how\s+many|how\s+much|exact|exactly)\b',
        ]
        
        topic_sensitivity = 0
        for p in sensitive_topics:
            if re.search(p, query_lower):
                topic_sensitivity += 1
        
        # For sensitive topics, reward hedging more
        sensitivity_multiplier = 1.0 + 0.1 * min(topic_sensitivity, 3)
        
        # ============================================================
        # FEATURE 7: Information Density (unique content words / total words)
        # ============================================================
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        if content_words:
            unique_content = len(set(content_words))
            info_density = unique_content / len(content_words)
        else:
            info_density = 0.0
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        # Base score from response substance
        base_score = (
            2.5 * length_score +          # 0-2.5: adequate length
            1.5 * relevance +              # 0-1.5: relevance to query
            1.0 * info_density             # 0-1.0: information richness
        )  # max ~5.0
        
        # Epistemic calibration score
        epistemic_score = (
            1.5 * max(epistemic_balance, 0) * sensitivity_multiplier +  # 0-~2.0
            source_bonus +                                                # 0-0.5
            qualifier_bonus                                               # 0-0.4
        )  # max ~2.9
        
        # Penalties
        overconfidence_penalty = min(strong_count * 0.15, 1.0) if num_words > 20 else 0
        repetition_penalty = repetition_ratio * 2.0
        garbage_penalty = min(garbage_ratio * 3.0, 2.0)
        
        # Very short non-substantive responses
        if num_words < 5 and len(resp_stripped) < 20:
            base_score *= 0.2
        
        # Compute final score
        raw_score = base_score + epistemic_score - overconfidence_penalty - repetition_penalty - garbage_penalty - truncated
        
        # Normalize to 0-10 range
        # Expected raw range: roughly -2 to 8
        score = max(0.0, min(10.0, raw_score + 1.0))
        
        # Additional adjustments for extreme cases
        # Single word / very terse responses that don't address the query
        if num_words <= 3 and relevance < 0.3:
            score = min(score, 1.5)
        
        # Responses that are mostly HTML/code when not asked for it
        query_asks_code = bool(re.search(r'\b(?:code|html|program|script|function|tag)\b', query_lower))
        if garbage_ratio > 0.1 and not query_asks_code:
            score *= 0.5
        
        return round(score, 2)
        
    except Exception:
        return 3.0