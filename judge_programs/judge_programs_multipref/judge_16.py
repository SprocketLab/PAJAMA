def judging_function(query, response):
    """
    Evaluates language quality and readability using a unique approach focused on:
    - Punctuation correctness and variety
    - Sentence structure diversity (length variance, starter variety)
    - Cohesion markers and discourse connectives
    - Character-level n-gram entropy (proxy for natural language flow)
    - Comma density and clause complexity
    - Question-response alignment
    """
    import re
    import math
    import collections
    import string

    try:
        if not response or not isinstance(response, str):
            return 0.0

        response = response.strip()
        if len(response) < 10:
            return 0.5

        # --- Tokenization ---
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", response)
        if len(words) < 3:
            return 1.0

        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if len(sentences) == 0:
            sentences = [response]

        total_words = len(words)
        lower_words = [w.lower() for w in words]

        # ============================================================
        # FEATURE 1: Punctuation variety and density
        # ============================================================
        punct_chars = [c for c in response if c in '.,;:!?—–-()[]"\'']
        punct_density = len(punct_chars) / max(len(response), 1) * 100
        # Ideal punctuation density is roughly 4-8%
        punct_density_score = max(0, 10 - abs(punct_density - 6) * 1.5)

        # Variety of punctuation used
        unique_puncts = set(punct_chars)
        punct_variety_score = min(len(unique_puncts) / 6.0, 1.0) * 10

        # ============================================================
        # FEATURE 2: Sentence starter diversity
        # ============================================================
        starters = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                starters.append(s_words[0].lower())

        if len(starters) > 1:
            unique_starters = len(set(starters))
            starter_diversity = unique_starters / len(starters)
        else:
            starter_diversity = 0.5
        starter_score = starter_diversity * 10

        # ============================================================
        # FEATURE 3: Sentence length variance (structural variety)
        # ============================================================
        sent_lengths = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            sent_lengths.append(len(s_words))

        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Some variance is good (5-15 std dev), too little or too much is bad
            if mean_len > 0:
                coeff_var = std_dev / mean_len
            else:
                coeff_var = 0
            # Ideal CV around 0.3-0.6
            length_var_score = max(0, 10 - abs(coeff_var - 0.45) * 15)
        else:
            length_var_score = 3.0

        # Average sentence length score (ideal: 12-22 words)
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        if 12 <= avg_sent_len <= 22:
            avg_len_score = 10.0
        elif avg_sent_len < 12:
            avg_len_score = max(0, 10 - (12 - avg_sent_len) * 1.0)
        else:
            avg_len_score = max(0, 10 - (avg_sent_len - 22) * 0.5)

        # ============================================================
        # FEATURE 4: Cohesion / discourse connectives
        # ============================================================
        connectives = [
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'similarly', 'likewise',
            'conversely', 'instead', 'otherwise', 'thus', 'hence', 'accordingly',
            'specifically', 'notably', 'importantly', 'essentially',
            'although', 'whereas', 'while', 'since', 'because',
            'also', 'then', 'next', 'finally', 'first', 'second', 'third',
            'in addition', 'for example', 'for instance', 'in contrast',
            'on the other hand', 'as a result', 'in fact', 'of course',
            'in other words', 'that said', 'to summarize', 'in particular'
        ]
        response_lower = response.lower()
        connective_count = 0
        unique_connectives = set()
        for conn in connectives:
            occurrences = response_lower.count(conn)
            if occurrences > 0:
                connective_count += occurrences
                unique_connectives.add(conn)

        # Connective density per 100 words
        conn_density = connective_count / max(total_words, 1) * 100
        # Ideal: 2-6 per 100 words
        conn_density_score = max(0, 10 - abs(conn_density - 4) * 2)
        conn_variety_score = min(len(unique_connectives) / 5.0, 1.0) * 10

        cohesion_score = (conn_density_score + conn_variety_score) / 2

        # ============================================================
        # FEATURE 5: Character trigram entropy (language naturalness)
        # ============================================================
        clean_text = re.sub(r'[^a-z ]', '', response_lower)
        if len(clean_text) > 10:
            trigrams = [clean_text[i:i+3] for i in range(len(clean_text) - 2)]
            trigram_counts = collections.Counter(trigrams)
            total_trigrams = len(trigrams)
            entropy = 0.0
            for count in trigram_counts.values():
                p = count / total_trigrams
                if p > 0:
                    entropy -= p * math.log2(p)
            # Natural English text typically has trigram entropy of 7-10
            max_possible = math.log2(min(total_trigrams, 17576))  # 26^3
            if max_possible > 0:
                normalized_entropy = entropy / max_possible
            else:
                normalized_entropy = 0
            # Higher normalized entropy (0.6-0.85) suggests richer text
            entropy_score = min(normalized_entropy / 0.8, 1.0) * 10
        else:
            entropy_score = 5.0

        # ============================================================
        # FEATURE 6: Comma-based clause complexity
        # ============================================================
        commas_per_sentence = []
        for s in sentences:
            comma_count = s.count(',')
            commas_per_sentence.append(comma_count)

        avg_commas = sum(commas_per_sentence) / max(len(commas_per_sentence), 1)
        # Ideal: 1-3 commas per sentence indicates good clause structure
        if 1.0 <= avg_commas <= 3.0:
            clause_score = 10.0
        elif avg_commas < 1.0:
            clause_score = max(0, avg_commas * 10)
        else:
            clause_score = max(0, 10 - (avg_commas - 3.0) * 2)

        # ============================================================
        # FEATURE 7: Capitalization correctness
        # ============================================================
        cap_errors = 0
        for s in sentences:
            s_stripped = s.lstrip(' \t*#->0123456789.')
            if s_stripped and s_stripped[0].isalpha() and s_stripped[0].islower():
                cap_errors += 1

        # Check for random mid-sentence capitalizations (excluding after periods, known abbreviations)
        # Simple heuristic: ratio of capitalized words that aren't sentence starters or "I"
        non_starter_caps = 0
        for i, w in enumerate(words):
            if i == 0:
                continue
            if w[0].isupper() and w.lower() not in {'i'}:
                # Check if previous character was sentence-ending
                prev_text = response[:response.find(w, max(0, response.find(words[i-1])))]
                # Just count — some are legitimate (proper nouns), but excessive is bad
                non_starter_caps += 1

        cap_ratio = non_starter_caps / max(total_words, 1)
        # Some capitalization is normal (5-15% for proper nouns, etc.)
        if cap_ratio <= 0.20:
            cap_score = 10.0 - cap_errors * 2
        else:
            cap_score = max(0, 10 - (cap_ratio - 0.20) * 30 - cap_errors * 2)
        cap_score = max(0, min(10, cap_score))

        # ============================================================
        # FEATURE 8: Engagement / conversational tone
        # ============================================================
        # Presence of direct address, rhetorical questions, exclamations
        engagement_markers = 0
        if re.search(r'\byou\b', response_lower):
            engagement_markers += 1
        if re.search(r'\byour\b', response_lower):
            engagement_markers += 1
        if '!' in response:
            engagement_markers += 1
        if re.search(r"\blet's\b", response_lower):
            engagement_markers += 1
        if re.search(r'\bhere\b.*\b(are|is)\b', response_lower):
            engagement_markers += 1
        if re.search(r'\bcertainly\b|\babsolutely\b|\bgreat\b|\bawesome\b', response_lower):
            engagement_markers += 1

        engagement_score = min(engagement_markers / 3.0, 1.0) * 10

        # ============================================================
        # FEATURE 9: Structural organization signals
        # ============================================================
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_headers = bool(re.search(r'^#{1,4}\s', response, re.MULTILINE))
        has_paragraphs = response.count('\n\n') >= 1

        org_signals = sum([has_numbered_list, has_bold, has_headers, has_paragraphs])
        # Reward some organization but not excessively
        org_score = min(org_signals / 2.5, 1.0) * 10

        # ============================================================
        # FEATURE 10: Lexical sophistication (beyond simple type-token)
        # Using hapax legomena ratio and average word length distribution
        # ============================================================
        word_freq = collections.Counter(lower_words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(len(word_freq), 1)

        # Word length distribution - standard deviation
        word_lengths = [len(w) for w in words]
        avg_wl = sum(word_lengths) / max(len(word_lengths), 1)
        if len(word_lengths) > 1:
            wl_var = sum((x - avg_wl) ** 2 for x in word_lengths) / len(word_lengths)
            wl_std = math.sqrt(wl_var)
        else:
            wl_std = 0

        # Hapax ratio of 0.5-0.7 is typical of good writing
        hapax_score = max(0, 10 - abs(hapax_ratio - 0.6) * 20)

        # Word length std of 2-3 indicates good vocabulary mix
        wl_variety_score = max(0, 10 - abs(wl_std - 2.5) * 4)

        lexical_score = (hapax_score + wl_variety_score) / 2

        # ============================================================
        # FEATURE 11: Repetition penalty
        # ============================================================
        # Check for repeated phrases (3-grams)
        if total_words >= 6:
            trigram_words = [' '.join(lower_words[i:i+3]) for i in range(total_words - 2)]
            trigram_word_counts = collections.Counter(trigram_words)
            repeated_trigrams = sum(1 for c in trigram_word_counts.values() if c > 2)
            repetition_penalty = min(repeated_trigrams * 1.5, 8)
        else:
            repetition_penalty = 0

        # ============================================================
        # FEATURE 12: Response length adequacy relative to query
        # ============================================================
        query_words = len(re.findall(r"[a-zA-Z']+", query)) if query else 5
        # Longer queries might need longer responses
        length_ratio = total_words / max(query_words, 1)
        # Ideal ratio: 5-20x query length
        if 5 <= length_ratio <= 20:
            length_adequacy = 10.0
        elif length_ratio < 5:
            length_adequacy = max(0, length_ratio * 2)
        else:
            length_adequacy = max(0, 10 - (length_ratio - 20) * 0.3)

        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        weights = {
            'punct_density': 0.06,
            'punct_variety': 0.04,
            'starter_diversity': 0.08,
            'length_variance': 0.06,
            'avg_sent_len': 0.08,
            'cohesion': 0.10,
            'entropy': 0.08,
            'clause_complexity': 0.06,
            'capitalization': 0.05,
            'engagement': 0.10,
            'organization': 0.08,
            'lexical': 0.08,
            'length_adequacy': 0.06,
        }

        raw_score = (
            weights['punct_density'] * punct_density_score +
            weights['punct_variety'] * punct_variety_score +
            weights['starter_diversity'] * starter_score +
            weights['length_variance'] * length_var_score +
            weights['avg_sent_len'] * avg_len_score +
            weights['cohesion'] * cohesion_score +
            weights['entropy'] * entropy_score +
            weights['clause_complexity'] * clause_score +
            weights['capitalization'] * cap_score +
            weights['engagement'] * engagement_score +
            weights['organization'] * org_score +
            weights['lexical'] * lexical_score +
            weights['length_adequacy'] * length_adequacy
        )

        # Total weight sums to ~0.93, normalize
        total_weight = sum(weights.values())
        raw_score = raw_score / total_weight

        # Apply repetition penalty
        raw_score = max(0, raw_score - repetition_penalty)

        # Scale to 0-100
        final_score = max(0.0, min(100.0, raw_score * 10))

        return round(final_score, 2)

    except Exception:
        return 25.0