def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    sentence-level analysis of claim types and appropriate hedging patterns.
    
    This variant focuses on:
    1. Sentence-level classification (factual vs speculative vs hedged)
    2. Structural coherence and completeness signals
    3. Ratio-based calibration scoring
    4. Penalizing absolute/dogmatic phrasing on ambiguous topics
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        resp_stripped = response.strip()
        if len(resp_stripped) < 2:
            return 0.5
        
        resp_lower = response.lower()
        query_lower = query.lower()
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', resp_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # ---- Feature 1: Appropriate uncertainty markers (per-sentence density) ----
        # Epistemic hedging phrases (nuanced, multi-word)
        epistemic_phrases = [
            r'\bit is (difficult|hard|challenging) to\b',
            r'\b(may|might|could) (be|have|indicate|suggest|vary)\b',
            r'\b(research|studies|evidence) suggest[s]?\b',
            r'\bgenerally (speaking|considered|regarded)\b',
            r'\bin (most|many|some) cases\b',
            r'\btend[s]? to\b',
            r'\b(likely|unlikely|probable|possibly|perhaps)\b',
            r'\bto (some|a certain) (extent|degree)\b',
            r'\bdepending on\b',
            r'\bnot (necessarily|always|entirely)\b',
            r'\b(can|could) vary\b',
            r'\bone (possible|potential) \w+\b',
            r'\b(often|sometimes|occasionally|frequently|typically|usually)\b',
            r'\bit (seems|appears) (that|to)\b',
            r'\b(arguably|approximately|roughly|around)\b',
            r'\bcan be (subjective|debatable|controversial)\b',
            r'\bwithout (controversy|debate)\b',
            r'\bhas been (criticized|debated|questioned)\b',
            r'\baccording to\b',
            r'\b(some|many|most|certain) (people|experts|scholars|researchers)\b',
            r'\bin general\b',
            r'\bas far as\b',
            r'\bwhile (it|this|there)\b',
        ]
        
        hedging_count = 0
        for pattern in epistemic_phrases:
            hedging_count += len(re.findall(pattern, resp_lower))
        
        hedging_density = hedging_count / num_sentences
        
        # ---- Feature 2: Overconfidence / absolutist markers ----
        absolutist_patterns = [
            r'\b(always|never|definitely|certainly|absolutely|undoubtedly)\b',
            r'\b(everyone knows|obviously|clearly|of course|without (a )?doubt)\b',
            r'\b(the fact is|the truth is|it is (a )?fact)\b',
            r'\b(there is no (question|doubt))\b',
            r'\b(100%|guaranteed|proven beyond)\b',
            r'\bno one (can|could|would|should)\b',
            r'\bevery (single|one)\b',
        ]
        
        absolutist_count = 0
        for pattern in absolutist_patterns:
            absolutist_count += len(re.findall(pattern, resp_lower))
        
        absolutist_density = absolutist_count / num_sentences
        
        # ---- Feature 3: Query ambiguity detection ----
        ambiguity_signals = [
            r'\b(how many|what is the|is it|can you|should|would)\b',
            r'\b(opinion|think|believe|feel|best|worst)\b',
            r'\b(controversial|debate|argument)\b',
            r'\b(history|meaning|significance)\b',
            r'\b(more about|information|explain)\b',
        ]
        
        query_ambiguity = 0
        for pattern in ambiguity_signals:
            if re.search(pattern, query_lower):
                query_ambiguity += 1
        query_ambiguity_score = min(query_ambiguity / 3.0, 1.0)
        
        # ---- Feature 4: Structural quality / completeness ----
        words = resp_lower.split()
        num_words = len(words)
        
        # Penalize very short responses
        if num_words < 5:
            length_score = 0.15
        elif num_words < 15:
            length_score = 0.4
        elif num_words < 30:
            length_score = 0.65
        elif num_words < 200:
            length_score = 1.0
        else:
            length_score = 0.85  # slightly penalize very long
        
        # Check if response seems truncated
        truncation_penalty = 0.0
        if resp_stripped[-1] not in '.!?")\']':
            # Might be truncated
            truncation_penalty = 0.05
        
        # ---- Feature 5: Content relevance (does it address the query?) ----
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        resp_words_set = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
        
        # Remove very common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'this', 'that', 'with', 'they', 'from',
                     'what', 'how', 'who', 'where', 'when', 'which', 'will',
                     'about', 'there', 'their', 'would', 'make', 'like', 'also',
                     'into', 'than', 'then', 'some', 'could', 'them', 'other'}
        
        query_content = query_words - stopwords
        resp_content = resp_words_set - stopwords
        
        if query_content:
            relevance = len(query_content & resp_content) / len(query_content)
        else:
            relevance = 0.5
        
        # ---- Feature 6: Sentence variety and information density ----
        # Check for repetition (low quality signal)
        if num_sentences > 2:
            unique_sentence_starts = set()
            for s in sentences:
                words_in_s = s.split()
                if len(words_in_s) >= 3:
                    unique_sentence_starts.add(' '.join(words_in_s[:3]).lower())
            repetition_ratio = len(unique_sentence_starts) / num_sentences
        else:
            repetition_ratio = 1.0
        
        # ---- Feature 7: Garbage / off-topic detection ----
        garbage_signals = 0
        
        # HTML/code in non-code responses
        if re.search(r'<(h[1-6]|div|span|blockquote|p)>', resp_lower):
            if 'html' not in query_lower and 'tag' not in query_lower:
                garbage_signals += 2
        
        # Random code patterns when not asked for code
        if re.search(r'(import |def |class |function\()', resp_lower):
            if 'code' not in query_lower and 'program' not in query_lower and 'function' not in query_lower:
                garbage_signals += 2
        
        # Excessive "Output:" or "Input:" patterns
        output_count = len(re.findall(r'\b(output|input)\s*:', resp_lower))
        if output_count > 2:
            garbage_signals += 1
        
        # Repeated question-answer format (echo/loop)
        qa_pattern_count = len(re.findall(r'\b(question|answer)\s*:', resp_lower))
        if qa_pattern_count > 2:
            garbage_signals += 1
        
        garbage_penalty = min(garbage_signals * 0.15, 0.6)
        
        # ---- Feature 8: Discourse connectors indicating structured reasoning ----
        discourse_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bon the other hand\b', r'\bin addition\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bconversely\b', r'\balternatively\b',
            r'\bfor (example|instance)\b', r'\bsuch as\b',
            r'\bin contrast\b', r'\bsimilarly\b',
            r'\bspecifically\b', r'\bnotably\b',
        ]
        
        discourse_count = 0
        for pattern in discourse_markers:
            discourse_count += len(re.findall(pattern, resp_lower))
        
        discourse_score = min(discourse_count * 0.12, 0.5)
        
        # ---- Feature 9: Acknowledgment of limitations ----
        limitation_patterns = [
            r'\b(exact|precise) (count|number|figure|answer) .{0,30}(difficult|hard|vary|subjective)\b',
            r'\bnot without (its )?(limitations|flaws|criticism)\b',
            r'\b(it is|it\'s) (important|worth) (to note|noting)\b',
            r'\b(keep in mind|bear in mind)\b',
            r'\b(this|it) (depends|varies)\b',
            r'\b(complex|nuanced|multifaceted)\b',
        ]
        
        limitation_count = 0
        for pattern in limitation_patterns:
            if re.search(pattern, resp_lower):
                limitation_count += 1
        
        limitation_score = min(limitation_count * 0.15, 0.4)
        
        # ====== COMPOSITE SCORING ======
        
        # Base score from length/structure
        base_score = length_score * 3.5  # 0 to 3.5
        
        # Relevance contribution
        relevance_score = relevance * 2.0  # 0 to 2.0
        
        # Hedging reward (more valuable when query is ambiguous)
        hedging_reward = min(hedging_density * 0.8, 1.2)
        if query_ambiguity_score > 0.3:
            hedging_reward *= (1.0 + query_ambiguity_score * 0.5)
        hedging_reward = min(hedging_reward, 1.5)
        
        # Overconfidence penalty (worse when query is ambiguous)
        overconfidence_penalty = absolutist_density * 0.4
        if query_ambiguity_score > 0.3:
            overconfidence_penalty *= (1.0 + query_ambiguity_score * 0.8)
        overconfidence_penalty = min(overconfidence_penalty, 1.5)
        
        # Repetition penalty
        repetition_penalty = max(0, (1.0 - repetition_ratio) * 0.8)
        
        # Combine
        score = (
            base_score
            + relevance_score
            + hedging_reward
            + discourse_score
            + limitation_score
            - overconfidence_penalty
            - repetition_penalty
            - garbage_penalty
            - truncation_penalty
        )
        
        # Normalize to 0-10 range
        # Theoretical max: ~3.5 + 2.0 + 1.5 + 0.5 + 0.4 = 7.9
        # Theoretical min: 0 - penalties
        score = max(0.0, min(10.0, score * 1.25 + 0.5))
        
        # Final rounding
        return round(score, 2)
        
    except Exception:
        return 3.0