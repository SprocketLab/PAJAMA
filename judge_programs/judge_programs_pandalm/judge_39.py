def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker analysis (causal, contrastive, additive, temporal)
    - Sentence-level progression analysis (information flow)
    - Contradiction detection via antonym/negation patterns
    - Structural depth (subordinate clauses, conditional reasoning)
    - Repetition penalty (detecting degenerate/circular text)
    - Propositional density (unique claims per sentence)
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0

        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0

        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space or end
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sents if s.strip()]

        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())

        sentences = split_sentences(response_stripped)
        all_words = tokenize(response_stripped)
        query_words = tokenize(query)

        if len(all_words) == 0:
            return 0.5

        num_sentences = len(sentences)

        # ============================================================
        # 1. DISCOURSE MARKER RICHNESS (different categories scored separately)
        # ============================================================
        causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'owing to',
            'for this reason', 'accordingly', 'thereby', 'leads to',
            'causes', 'results in', 'in order to'
        ]
        contrastive_markers = [
            'however', 'but', 'although', 'though', 'whereas',
            'on the other hand', 'in contrast', 'nevertheless',
            'nonetheless', 'yet', 'despite', 'while', 'conversely',
            'on the contrary', 'rather than', 'instead', 'unlike'
        ]
        additive_markers = [
            'furthermore', 'moreover', 'additionally', 'in addition',
            'also', 'besides', 'as well as', 'not only', 'along with',
            'coupled with', 'equally important'
        ]
        temporal_markers = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'subsequently', 'previously', 'before', 'after', 'meanwhile',
            'initially', 'eventually', 'lastly', 'in the end', 'at first'
        ]
        elaboration_markers = [
            'for example', 'for instance', 'such as', 'specifically',
            'in particular', 'namely', 'to illustrate', 'that is',
            'in other words', 'more specifically', 'this means'
        ]
        conclusion_markers = [
            'in conclusion', 'to summarize', 'overall', 'in summary',
            'to conclude', 'all in all', 'ultimately', 'in short'
        ]

        response_lower = response_stripped.lower()

        def count_markers(markers):
            count = 0
            for m in markers:
                count += len(re.findall(r'\b' + re.escape(m) + r'\b', response_lower))
            return count

        causal_count = count_markers(causal_markers)
        contrastive_count = count_markers(contrastive_markers)
        additive_count = count_markers(additive_markers)
        temporal_count = count_markers(temporal_markers)
        elaboration_count = count_markers(elaboration_markers)
        conclusion_count = count_markers(conclusion_markers)

        # Count distinct categories used (not just total markers)
        categories_used = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                           temporal_count, elaboration_count, conclusion_count] if c > 0)

        total_markers = (causal_count + contrastive_count + additive_count +
                         temporal_count + elaboration_count + conclusion_count)

        # Normalize by sentence count
        marker_density = total_markers / max(num_sentences, 1)
        # Score: reward diversity of marker types and density, with diminishing returns
        discourse_score = (min(marker_density, 1.5) / 1.5) * 5 + (categories_used / 6) * 5
        # Max ~10

        # ============================================================
        # 2. SUBORDINATE CLAUSE DEPTH (complex reasoning indicator)
        # ============================================================
        subordinators = [
            'which', 'that', 'who', 'whom', 'whose', 'where', 'when',
            'if', 'unless', 'although', 'because', 'since', 'while',
            'whereas', 'whether', 'even though', 'so that', 'in order that',
            'provided that', 'as long as', 'given that'
        ]
        subordinate_count = 0
        for sub in subordinators:
            subordinate_count += len(re.findall(r'\b' + re.escape(sub) + r'\b', response_lower))

        # Also count commas as proxy for clause complexity
        comma_count = response_stripped.count(',')
        clause_complexity = subordinate_count / max(num_sentences, 1)
        comma_density = comma_count / max(num_sentences, 1)

        # Score structural depth
        depth_score = min(clause_complexity * 2.5, 7) + min(comma_density * 0.8, 3)
        # Max ~10

        # ============================================================
        # 3. INFORMATION PROGRESSION (new content per sentence)
        # ============================================================
        if num_sentences >= 2:
            sentence_word_sets = []
            for s in sentences:
                words = set(tokenize(s))
                # Remove very common stop words
                stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'it', 'its', 'this', 'that', 'and', 'or', 'not', 'as'}
                words = words - stops
                sentence_word_sets.append(words)

            # Measure new information introduced in each sentence
            cumulative = set()
            new_info_ratios = []
            for i, ws in enumerate(sentence_word_sets):
                if len(ws) == 0:
                    new_info_ratios.append(0)
                else:
                    new_words = ws - cumulative
                    new_info_ratios.append(len(new_words) / len(ws))
                cumulative |= ws

            # Average new info ratio (excluding first sentence which is always 1.0)
            if len(new_info_ratios) > 1:
                avg_new_info = sum(new_info_ratios[1:]) / len(new_info_ratios[1:])
            else:
                avg_new_info = new_info_ratios[0] if new_info_ratios else 0

            progression_score = avg_new_info * 10
        else:
            # Single sentence - moderate score
            progression_score = 4.0

        # ============================================================
        # 4. REPETITION PENALTY (detect degenerate/circular text)
        # ============================================================
        # Check for repeated phrases (3-grams)
        if len(all_words) >= 3:
            trigrams = [tuple(all_words[i:i+3]) for i in range(len(all_words) - 2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            unique_trigrams = len(trigram_counts)

            if total_trigrams > 0:
                trigram_uniqueness = unique_trigrams / total_trigrams
            else:
                trigram_uniqueness = 1.0

            # Check for repeated sentences
            sentence_texts = [s.lower().strip() for s in sentences]
            unique_sentences = len(set(sentence_texts))
            sentence_uniqueness = unique_sentences / max(num_sentences, 1)

            # Check for repeated words
            word_counts = Counter(all_words)
            # Remove stop words for this analysis
            stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'its', 'this', 'that', 'and', 'or', 'not', 'as'}
            content_words = {w: c for w, c in word_counts.items() if w not in stops}
            if content_words:
                max_repeat = max(content_words.values())
                total_content = sum(content_words.values())
                max_repeat_ratio = max_repeat / total_content
            else:
                max_repeat_ratio = 0

            repetition_penalty = 0
            # Penalize low trigram uniqueness heavily
            if trigram_uniqueness < 0.5:
                repetition_penalty += (0.5 - trigram_uniqueness) * 30
            # Penalize repeated sentences
            if sentence_uniqueness < 0.8:
                repetition_penalty += (0.8 - sentence_uniqueness) * 15
            # Penalize dominant single word
            if max_repeat_ratio > 0.3:
                repetition_penalty += (max_repeat_ratio - 0.3) * 10
        else:
            repetition_penalty = 0

        # ============================================================
        # 5. QUERY RELEVANCE via argument alignment
        # ============================================================
        query_content = set(query_words) - {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                              'to', 'of', 'in', 'for', 'on', 'with', 'at',
                                              'by', 'from', 'it', 'this', 'that', 'and', 'or',
                                              'what', 'how', 'why', 'when', 'where', 'who',
                                              'which', 'do', 'does', 'did', 'can', 'could',
                                              'would', 'should', 'will', 'shall', 'may', 'might'}
        response_content = set(all_words) - {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                               'to', 'of', 'in', 'for', 'on', 'with', 'at',
                                               'by', 'from', 'it', 'this', 'that', 'and', 'or'}

        if query_content:
            query_coverage = len(query_content & response_content) / len(query_content)
        else:
            query_coverage = 0.5

        relevance_score = query_coverage * 8  # Max 8

        # ============================================================
        # 6. RESPONSE COMPLETENESS (length adequacy)
        # ============================================================
        word_count = len(all_words)
        # Reward adequate length with diminishing returns
        if word_count < 5:
            length_score = word_count * 0.4  # Very short = low
        elif word_count < 20:
            length_score = 2 + (word_count - 5) * 0.2
        elif word_count < 80:
            length_score = 5 + (word_count - 20) * 0.05
        else:
            length_score = 8 + min((word_count - 80) * 0.01, 2)
        length_score = min(length_score, 10)

        # ============================================================
        # 7. LOGICAL CONNECTIVE PATTERNS (premise-conclusion structure)
        # ============================================================
        # Detect if-then patterns, premise-conclusion patterns
        conditional_patterns = [
            r'\bif\b.*\bthen\b',
            r'\bwhen\b.*\b(?:will|would|can|could|should)\b',
            r'\bgiven\b.*\b(?:then|therefore|thus)\b',
            r'\b(?:this|these|that|those)\s+(?:means?|implies?|suggests?|indicates?)\b',
            r'\b(?:the reason|one reason|another reason)\b',
            r'\b(?:on one hand|on the other)\b',
            r'\b(?:not only|but also)\b',
        ]
        logical_pattern_count = 0
        for pat in conditional_patterns:
            if re.search(pat, response_lower):
                logical_pattern_count += 1

        logical_pattern_score = min(logical_pattern_count * 2.5, 8)

        # ============================================================
        # 8. SENTENCE LENGTH VARIANCE (indicates structural variety)
        # ============================================================
        if num_sentences >= 2:
            sent_lengths = [len(tokenize(s)) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                cv = math.sqrt(variance) / mean_len  # coefficient of variation
                # Some variance is good (0.2-0.6), too much or too little is bad
                if cv < 0.1:
                    variety_score = 2
                elif cv < 0.5:
                    variety_score = 2 + (cv - 0.1) * 15  # up to 8
                elif cv < 0.8:
                    variety_score = 8
                else:
                    variety_score = max(8 - (cv - 0.8) * 5, 2)
            else:
                variety_score = 1
        else:
            variety_score = 3

        # ============================================================
        # COMBINE SCORES
        # ============================================================
        # Weights chosen to emphasize logical structure
        raw_score = (
            discourse_score * 0.18 +        # discourse markers
            depth_score * 0.15 +             # clause complexity
            progression_score * 0.15 +       # information flow
            relevance_score * 0.12 +         # query relevance
            length_score * 0.12 +            # completeness
            logical_pattern_score * 0.13 +   # logical patterns
            variety_score * 0.08 +           # sentence variety
            0                                # padding
        )

        # Apply repetition penalty
        final_score = max(raw_score - repetition_penalty, 0)

        # Scale to 0-100
        final_score = min(final_score * 10, 100)

        # Ensure minimum discrimination
        if word_count <= 2:
            final_score = min(final_score, 5)
        elif word_count <= 5:
            final_score = min(final_score, 15)

        return round(final_score, 2)

    except Exception:
        return 0.0