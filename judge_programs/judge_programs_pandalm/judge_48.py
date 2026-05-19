def judging_function(query, response):
    """
    Evaluates clarity and conciseness using a unique approach based on:
    - Compression ratio (repetition detection via LZ-like analysis)
    - Syntactic complexity via clause/conjunction density
    - Information density (unique content words per total words)
    - Response completeness relative to query
    - Penalizing degenerate/empty responses
    - Sentence-level coherence via progressive information gain
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not response.strip():
            return 0.0

        response_text = response.strip()
        query_text = query.strip() if query else ""

        # Tokenize
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+', text.lower())

        response_words = tokenize(response_text)
        query_words = tokenize(query_text)

        if len(response_words) == 0:
            return 0.5

        # ---- Feature 1: LZ-complexity based repetition detection ----
        # Approximate Lempel-Ziv complexity on the word sequence
        # Higher complexity = less repetitive = better
        def lz_complexity(sequence):
            if not sequence:
                return 0
            n = len(sequence)
            i = 0
            complexity = 0
            while i < n:
                # Find longest match starting at i in sequence[:i]
                best_len = 0
                prefix = sequence[:i] if i > 0 else []
                for start in range(len(prefix)):
                    match_len = 0
                    while (i + match_len < n and
                           start + match_len < len(prefix) and
                           sequence[start + match_len] == sequence[i + match_len]):
                        match_len += 1
                    best_len = max(best_len, match_len)
                i += max(best_len, 1)
                complexity += 1
            return complexity

        # Limit to first 200 words for performance
        lz_c = lz_complexity(response_words[:200])
        max_lz = len(response_words[:200])
        if max_lz > 0:
            lz_ratio = lz_c / max_lz  # 1.0 = all unique, lower = more repetitive
        else:
            lz_ratio = 0.5

        # ---- Feature 2: Conjunction/subordination density ----
        # Too many conjunctions per sentence = convoluted
        conjunctions = {'and', 'but', 'or', 'nor', 'yet', 'so', 'because',
                        'although', 'though', 'while', 'whereas', 'since',
                        'unless', 'if', 'when', 'whenever', 'wherever',
                        'however', 'moreover', 'furthermore', 'nevertheless',
                        'nonetheless', 'therefore', 'consequently', 'thus'}
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        conj_count = sum(1 for w in response_words if w in conjunctions)
        conj_density = conj_count / num_sentences  # conjunctions per sentence
        # Ideal: 1-2 per sentence. Penalize if too high
        conj_penalty = max(0, (conj_density - 2.0) * 0.1)

        # ---- Feature 3: Unique content word ratio (information density) ----
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'it',
                     'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                     'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                     'my', 'your', 'his', 'our', 'their', 'not', 'no', 'and',
                     'but', 'or', 'if', 'then', 'than', 'so', 'also', 'more',
                     'most', 'very', 'just', 'about', 'up', 'out', 'which',
                     'who', 'what', 'where', 'when', 'how', 'all', 'each',
                     'both', 'few', 'some', 'such', 'other', 'only'}

        content_words = [w for w in response_words if w not in stopwords]
        unique_content = set(content_words)
        
        if len(content_words) > 0:
            content_uniqueness = len(unique_content) / len(content_words)
        else:
            content_uniqueness = 0.0

        # ---- Feature 4: Progressive information gain across sentences ----
        # Each sentence should add new information
        if len(sentences) > 1:
            seen_content = set()
            gains = []
            for sent in sentences:
                sent_words = set(tokenize(sent)) - stopwords
                if len(sent_words) > 0:
                    new_words = sent_words - seen_content
                    gain = len(new_words) / len(sent_words)
                    gains.append(gain)
                    seen_content.update(sent_words)
            if gains:
                avg_gain = sum(gains) / len(gains)
            else:
                avg_gain = 0.5
        else:
            avg_gain = 0.7  # Single sentence: neutral

        # ---- Feature 5: Exact phrase repetition detection ----
        # Check for repeated 3-grams and 4-grams
        def count_repeated_ngrams(words, n):
            if len(words) < n:
                return 0
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            return repeated

        repeated_3 = count_repeated_ngrams(response_words, 3)
        repeated_4 = count_repeated_ngrams(response_words, 4)
        total_possible = max(len(response_words) - 2, 1)
        repetition_ratio = (repeated_3 * 0.5 + repeated_4 * 1.0) / total_possible

        # ---- Feature 6: Response adequacy (not too short, not too long) ----
        # Based on query complexity
        query_content_words = [w for w in query_words if w not in stopwords]
        query_complexity = len(query_content_words)
        
        # Expected response length range
        min_expected = max(5, query_complexity * 2)
        max_expected = max(80, query_complexity * 25)
        
        word_count = len(response_words)
        if word_count < 3:
            length_score = 0.1
        elif word_count < min_expected:
            length_score = 0.3 + 0.4 * (word_count / min_expected)
        elif word_count <= max_expected:
            length_score = 1.0
        else:
            # Gradually penalize excessive length
            excess = (word_count - max_expected) / max_expected
            length_score = max(0.3, 1.0 - excess * 0.3)

        # ---- Feature 7: Query relevance via keyword coverage ----
        if query_content_words:
            query_content_set = set(query_content_words)
            covered = sum(1 for w in query_content_set if w in set(response_words))
            relevance = covered / len(query_content_set)
        else:
            relevance = 0.5

        # ---- Feature 8: Average words per sentence (clarity proxy) ----
        words_per_sentence = word_count / num_sentences
        # Ideal: 10-25 words per sentence
        if words_per_sentence < 5:
            wps_score = 0.5
        elif words_per_sentence <= 25:
            wps_score = 1.0
        elif words_per_sentence <= 40:
            wps_score = 1.0 - (words_per_sentence - 25) / 30
        else:
            wps_score = 0.4

        # ---- Feature 9: Degenerate content detection ----
        # Check if response is mostly the same word repeated
        if response_words:
            most_common_word, most_common_count = Counter(response_words).most_common(1)[0]
            dominance = most_common_count / len(response_words)
            if most_common_word not in stopwords and dominance > 0.3:
                degenerate_penalty = (dominance - 0.3) * 2
            else:
                degenerate_penalty = 0.0
        else:
            degenerate_penalty = 0.0

        # ---- Feature 10: Sentence variety (std dev of sentence lengths) ----
        if len(sentences) > 1:
            sent_lengths = [len(tokenize(s)) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((sl - mean_sl) ** 2 for sl in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Some variety is good, but extreme variance is bad
            if mean_sl > 0:
                cv = std_sl / mean_sl  # coefficient of variation
                if cv < 0.5:
                    variety_score = 0.8 + cv * 0.4  # mild bonus for variety
                else:
                    variety_score = max(0.5, 1.0 - (cv - 0.5) * 0.3)
            else:
                variety_score = 0.5
        else:
            variety_score = 0.7

        # ---- Combine all features ----
        # Weights chosen to emphasize the most discriminative features
        score = (
            lz_ratio * 2.5 +              # [0, 2.5] - repetition detection
            content_uniqueness * 1.5 +      # [0, 1.5] - information density
            avg_gain * 1.5 +                # [0, 1.5] - progressive info
            length_score * 1.5 +            # [0, 1.5] - appropriate length
            relevance * 1.0 +               # [0, 1.0] - query relevance
            wps_score * 0.8 +               # [0, 0.8] - sentence clarity
            variety_score * 0.5 +           # [0, 0.5] - sentence variety
            - repetition_ratio * 3.0 +      # penalty for phrase repetition
            - conj_penalty * 1.0 +          # penalty for convoluted structure
            - degenerate_penalty * 3.0      # penalty for degenerate content
        )
        # Theoretical max ~9.8, typical range 3-9

        # Normalize to 0-10 scale
        score = max(0.0, min(10.0, score))

        return round(score, 3)

    except Exception:
        return 3.0