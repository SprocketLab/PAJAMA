def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a STRUCTURAL ANALYSIS approach:
    - Sentence-level epistemic classification (each sentence tagged as fact/hedged/speculative/overconfident)
    - Query ambiguity detection to determine expected calibration level
    - Discourse coherence of epistemic stance across the response
    - Proportional balance scoring based on topic type
    - Information density and completeness metrics
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        if len(response_clean) < 3:
            return 0.5
        
        # ============================================================
        # 1. QUERY AMBIGUITY / TOPIC TYPE CLASSIFICATION
        # Determines what level of epistemic calibration we expect
        # ============================================================
        
        # Factual queries - expect confident, well-established claims
        factual_signals = [
            r'\bwhat is\b', r'\bdefine\b', r'\bdescribe\b', r'\bexplain\b',
            r'\bhow does\b', r'\bhow do\b', r'\bwhat are\b', r'\blist\b',
            r'\bname\b', r'\bprovide\b', r'\bgive\b', r'\brewrite\b',
            r'\bconvert\b', r'\btranslate\b', r'\bcreate\b', r'\bwrite\b',
            r'\bgenerate\b', r'\bcome up with\b', r'\bmake\b', r'\bcrop\b',
        ]
        
        # Speculative/opinion queries - expect hedging and nuance
        speculative_signals = [
            r'\bwhy might\b', r'\bdo you think\b', r'\bwhat if\b',
            r'\bcould\b.*\b(happen|be|cause)\b', r'\bpredict\b',
            r'\bfuture\b', r'\bopinion\b', r'\bbelieve\b',
            r'\bhypothetical\b', r'\bspeculate\b', r'\bpossible\b',
            r'\blikely\b', r'\bshould\b.*\b(we|society|government)\b',
            r'\bdebate\b', r'\bcontroversial\b', r'\bargue\b',
        ]
        
        # Comparative queries - expect balanced treatment
        comparative_signals = [
            r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b',
            r'\bsimilar\b', r'\bversus\b', r'\bvs\b', r'\bpros and cons\b',
            r'\badvantages\b.*\bdisadvantages\b',
        ]
        
        ql = query_clean.lower()
        
        factual_score = sum(1 for p in factual_signals if re.search(p, ql))
        speculative_score = sum(1 for p in speculative_signals if re.search(p, ql))
        comparative_score = sum(1 for p in comparative_signals if re.search(p, ql))
        
        # Classify query type
        if speculative_score > factual_score and speculative_score > comparative_score:
            query_type = 'speculative'
        elif comparative_score > 0:
            query_type = 'comparative'
        else:
            query_type = 'factual'
        
        # ============================================================
        # 2. SENTENCE-LEVEL EPISTEMIC CLASSIFICATION
        # Tag each sentence with its epistemic stance
        # ============================================================
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        if not sentences:
            return 1.0
        
        # Epistemic markers for classification
        hedging_markers = [
            'likely', 'unlikely', 'perhaps', 'possibly', 'probably',
            'may', 'might', 'could be', 'appears to', 'seems to',
            'suggests', 'indicates', 'tends to', 'generally',
            'in some cases', 'often', 'sometimes', 'typically',
            'research suggests', 'studies suggest', 'evidence suggests',
            'it is believed', 'it is thought', 'arguably',
            'to some extent', 'in many ways', 'can be seen as',
            'one interpretation', 'some argue', 'some believe',
        ]
        
        overconfidence_markers = [
            'always', 'never', 'certainly', 'definitely', 'undoubtedly',
            'without question', 'without doubt', 'absolutely',
            'unquestionably', 'indisputably', 'obviously',
            'clearly the best', 'clearly the worst', 'the only way',
            'everyone knows', 'it is a fact that', 'proven that',
            'no one can deny', 'there is no doubt',
        ]
        
        source_attribution = [
            'according to', 'research shows', 'studies show',
            'data suggests', 'experts', 'scientists',
            'researchers', 'evidence', 'findings',
        ]
        
        conditional_markers = [
            'if', 'depending on', 'in the case of', 'when',
            'under certain', 'in certain', 'provided that',
            'assuming', 'given that',
        ]
        
        # Classify each sentence
        sentence_tags = []
        for sent in sentences:
            sl = sent.lower()
            tag = {
                'hedged': 0,
                'overconfident': 0,
                'sourced': 0,
                'conditional': 0,
                'assertive': 0,
                'length': len(sent.split()),
            }
            
            for marker in hedging_markers:
                if marker in sl:
                    tag['hedged'] += 1
            
            for marker in overconfidence_markers:
                if marker in sl:
                    tag['overconfident'] += 1
            
            for marker in source_attribution:
                if marker in sl:
                    tag['sourced'] += 1
            
            for marker in conditional_markers:
                if re.search(r'\b' + re.escape(marker) + r'\b', sl):
                    tag['conditional'] += 1
            
            # Check for bare assertions (no epistemic marking at all)
            if tag['hedged'] == 0 and tag['overconfident'] == 0 and tag['sourced'] == 0 and tag['conditional'] == 0:
                tag['assertive'] = 1
            
            sentence_tags.append(tag)
        
        n_sentences = len(sentence_tags)
        
        # Aggregate sentence-level stats
        n_hedged = sum(1 for t in sentence_tags if t['hedged'] > 0)
        n_overconfident = sum(1 for t in sentence_tags if t['overconfident'] > 0)
        n_sourced = sum(1 for t in sentence_tags if t['sourced'] > 0)
        n_conditional = sum(1 for t in sentence_tags if t['conditional'] > 0)
        n_assertive = sum(1 for t in sentence_tags if t['assertive'] > 0)
        
        hedged_ratio = n_hedged / n_sentences
        overconfident_ratio = n_overconfident / n_sentences
        sourced_ratio = n_sourced / n_sentences
        conditional_ratio = n_conditional / n_sentences
        assertive_ratio = n_assertive / n_sentences
        
        # ============================================================
        # 3. EPISTEMIC CALIBRATION SCORE
        # Score depends on query type
        # ============================================================
        
        calibration_score = 50.0  # base
        
        if query_type == 'speculative':
            # For speculative queries, we want hedging and nuance
            calibration_score += hedged_ratio * 15
            calibration_score += conditional_ratio * 10
            calibration_score += sourced_ratio * 10
            calibration_score -= overconfident_ratio * 20
            # Pure assertions on speculative topics are slightly bad
            calibration_score -= assertive_ratio * 5
            
        elif query_type == 'comparative':
            # For comparative queries, we want balance and some hedging
            calibration_score += hedged_ratio * 10
            calibration_score += conditional_ratio * 8
            calibration_score -= overconfident_ratio * 15
            # Some assertions are fine for stating differences
            
        else:  # factual
            # For factual queries, confident assertions are fine
            # But overconfidence markers are still slightly negative
            calibration_score -= overconfident_ratio * 10
            # Hedging on clearly factual topics is slightly negative
            # but not as bad as overconfidence on speculative topics
            calibration_score += hedged_ratio * 2
            calibration_score += sourced_ratio * 5
        
        # ============================================================
        # 4. STRUCTURAL QUALITY & INFORMATION DENSITY
        # ============================================================
        
        words = response_clean.split()
        n_words = len(words)
        
        # Unique word ratio (vocabulary richness)
        unique_words = set(w.lower().strip('.,!?;:') for w in words)
        vocab_richness = len(unique_words) / max(n_words, 1)
        
        # Repetition penalty - detect repeated phrases
        bigrams = []
        lower_words = [w.lower().strip('.,!?;:') for w in words]
        for i in range(len(lower_words) - 1):
            bigrams.append(lower_words[i] + ' ' + lower_words[i+1])
        
        if bigrams:
            bigram_counts = Counter(bigrams)
            max_bigram_repeat = max(bigram_counts.values())
            repetition_penalty = max(0, (max_bigram_repeat - 3) * 3)
        else:
            repetition_penalty = 0
        
        # Trigram repetition (catches more subtle repetition)
        trigrams = []
        for i in range(len(lower_words) - 2):
            trigrams.append(' '.join(lower_words[i:i+3]))
        
        if trigrams:
            trigram_counts = Counter(trigrams)
            max_trigram_repeat = max(trigram_counts.values())
            repetition_penalty += max(0, (max_trigram_repeat - 2) * 4)
        
        # ============================================================
        # 5. RESPONSE COMPLETENESS & SUBSTANCE
        # ============================================================
        
        # Length scoring - prefer substantive responses but not excessively long
        if n_words < 5:
            length_score = 2
        elif n_words < 15:
            length_score = 8
        elif n_words < 30:
            length_score = 14
        elif n_words < 80:
            length_score = 18
        elif n_words < 150:
            length_score = 16
        else:
            length_score = 13
        
        # Number of distinct points/ideas (approximated by sentence count with substance)
        substantive_sentences = sum(1 for t in sentence_tags if t['length'] >= 5)
        idea_density = min(substantive_sentences / max(n_sentences, 1), 1.0)
        
        # ============================================================
        # 6. DISCOURSE COHERENCE OF EPISTEMIC STANCE
        # Check if the response maintains a consistent epistemic stance
        # or appropriately varies it
        # ============================================================
        
        # Detect epistemic shifts (e.g., going from hedged to overconfident)
        stance_sequence = []
        for t in sentence_tags:
            if t['overconfident'] > 0:
                stance_sequence.append('O')
            elif t['hedged'] > 0:
                stance_sequence.append('H')
            elif t['sourced'] > 0:
                stance_sequence.append('S')
            elif t['conditional'] > 0:
                stance_sequence.append('C')
            else:
                stance_sequence.append('A')
        
        # Count jarring transitions (hedged -> overconfident or vice versa)
        jarring_transitions = 0
        for i in range(len(stance_sequence) - 1):
            if (stance_sequence[i] == 'H' and stance_sequence[i+1] == 'O') or \
               (stance_sequence[i] == 'O' and stance_sequence[i+1] == 'H'):
                jarring_transitions += 1
        
        coherence_penalty = jarring_transitions * 3
        
        # ============================================================
        # 7. SPECIFICITY AND DETAIL ANALYSIS
        # ============================================================
        
        # Check for specific details (numbers, examples, proper nouns)
        has_numbers = len(re.findall(r'\b\d+\b', response_clean))
        has_examples = len(re.findall(r'\b(for example|such as|e\.g\.|for instance|like)\b', response_clean.lower()))
        has_contrast = len(re.findall(r'\b(however|but|although|while|whereas|on the other hand|in contrast|conversely)\b', response_clean.lower()))
        has_cause = len(re.findall(r'\b(because|therefore|thus|hence|as a result|consequently|due to|since)\b', response_clean.lower()))
        
        specificity_score = min(has_numbers * 1.5 + has_examples * 2 + has_contrast * 2 + has_cause * 1.5, 12)
        
        # ============================================================
        # 8. QUERY-RESPONSE RELEVANCE (topic coverage)
        # ============================================================
        
        # Extract content words from query
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not',
            'so', 'if', 'that', 'this', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
            'about', 'up', 'also',
        }
        
        query_words = set(
            w.lower().strip('.,!?;:()[]"\'')
            for w in query_clean.split()
            if len(w.strip('.,!?;:()[]"\'')) > 2
        ) - stop_words
        
        response_words_set = set(
            w.lower().strip('.,!?;:()[]"\'')
            for w in words
            if len(w.strip('.,!?;:()[]"\'')) > 2
        ) - stop_words
        
        if query_words:
            topic_coverage = len(query_words & response_words_set) / len(query_words)
        else:
            topic_coverage = 0.5
        
        relevance_score = topic_coverage * 12
        
        # ============================================================
        # 9. FINAL COMPOSITE SCORE
        # ============================================================
        
        # Weighted combination
        final_score = (
            calibration_score * 0.25 +       # Epistemic calibration (0-~70 range)
            length_score * 0.15 +             # Length/substance (0-18)
            vocab_richness * 15 * 0.10 +      # Vocabulary richness (0-15)
            specificity_score * 0.12 +        # Specificity (0-12)
            relevance_score * 0.15 +          # Relevance (0-12)
            idea_density * 10 * 0.10 +        # Idea density (0-10)
            - repetition_penalty * 0.08 +     # Repetition penalty
            - coherence_penalty * 0.05        # Coherence penalty
        )
        
        # Bonus for having multiple sentences with good structure
        if n_sentences >= 3 and substantive_sentences >= 2:
            final_score += 2.0
        
        # Penalty for extremely short or empty-ish responses
        if n_words < 5:
            final_score *= 0.3
        elif n_words < 10:
            final_score *= 0.6
        
        # Penalty for garbled/broken text
        if response_clean.endswith(('...', '…')) or (len(response_clean) > 50 and not response_clean[-1] in '.!?"\''):
            # Truncated response
            final_score *= 0.85
        
        # Severe repetition detection (same word 5+ times)
        word_counts = Counter(lower_words)
        most_common_word = word_counts.most_common(1)
        if most_common_word:
            mc_word, mc_count = most_common_word[0]
            if mc_word not in stop_words and mc_count > 4:
                final_score *= 0.7
        
        # Normalize to 0-100 range
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            return min(max(len(str(response).split()) * 0.5, 1.0), 50.0)
        except Exception:
            return 5.0