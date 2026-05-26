def judging_function(query, response):
    """
    Evaluate response relevance to query using word overlap, topic alignment,
    and content quality signals. Returns a score where higher = better.
    
    Strategy: Multi-signal approach combining:
    1. TF-based keyword overlap between query and response
    2. Query intent coverage analysis
    3. Response substance and coherence metrics
    4. Penalty signals for off-topic or degenerate content
    """
    try:
        import re
        import math
        from collections import Counter
        
        # Handle edge cases
        if not query or not response:
            return 0.0
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        query = query.strip()
        response = response.strip()
        
        if len(response) == 0:
            return 0.0
        if len(response) <= 2:
            return 0.5
        
        # --- Tokenization ---
        def tokenize(text):
            """Lowercase, remove punctuation, split into words."""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            tokens = text.split()
            return tokens
        
        # Common English stopwords
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'am', 'an', 'any', 'anything',
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if len(query_tokens) == 0:
            return 5.0  # Can't evaluate relevance without query
        if len(response_tokens) == 0:
            return 0.5
        
        # Content words (non-stopwords)
        query_content = [t for t in query_tokens if t not in STOPWORDS and len(t) > 1]
        response_content = [t for t in response_tokens if t not in STOPWORDS and len(t) > 1]
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        # --- Signal 1: Keyword Overlap (Jaccard-like + recall) ---
        if len(query_content_set) > 0:
            # How many query content words appear in the response
            query_recall = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            query_recall = 0.5  # neutral if no content words in query
        
        # Weighted overlap: count frequency-weighted matches
        response_content_counter = Counter(response_content)
        query_content_counter = Counter(query_content)
        
        weighted_match = 0.0
        total_query_weight = 0.0
        for word, count in query_content_counter.items():
            total_query_weight += count
            if word in response_content_counter:
                weighted_match += min(count, response_content_counter[word])
        
        weighted_recall = weighted_match / max(total_query_weight, 1)
        
        # --- Signal 2: N-gram overlap (bigrams) ---
        def get_bigrams(tokens):
            return [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]
        
        query_bigrams = set(get_bigrams(query_tokens))
        response_bigrams = set(get_bigrams(response_tokens))
        
        if len(query_bigrams) > 0:
            bigram_recall = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_recall = 0.0
        
        # --- Signal 3: Topic word presence ---
        # Extract likely topic words from query (longer, less common words)
        topic_words = [w for w in query_content if len(w) >= 4]
        if len(topic_words) > 0:
            topic_coverage = sum(1 for w in set(topic_words) if w in response_content_set) / len(set(topic_words))
        else:
            topic_coverage = query_recall  # fallback
        
        # --- Signal 4: Response length quality ---
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Ideal response is typically longer than the query but not excessively so
        # Very short responses are usually bad
        if resp_len <= 3:
            length_score = 0.15
        elif resp_len <= 8:
            length_score = 0.35
        elif resp_len <= 15:
            length_score = 0.55
        elif resp_len <= 50:
            length_score = 0.8
        elif resp_len <= 150:
            length_score = 1.0
        elif resp_len <= 300:
            length_score = 0.9
        else:
            length_score = 0.75
        
        # --- Signal 5: Repetition penalty ---
        # Check for excessive repetition in response
        if len(response_tokens) > 5:
            unique_ratio = len(set(response_tokens)) / len(response_tokens)
        else:
            unique_ratio = 1.0
        
        # Severe repetition penalty
        repetition_penalty = 0.0
        if unique_ratio < 0.3:
            repetition_penalty = 0.4
        elif unique_ratio < 0.5:
            repetition_penalty = 0.2
        elif unique_ratio < 0.6:
            repetition_penalty = 0.1
        
        # Check for repeated phrases/sentences
        sentences = re.split(r'[.!?\n]+', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        if len(sentences) > 1:
            unique_sentences = set(sentences)
            sentence_repetition = 1.0 - (len(unique_sentences) / len(sentences))
            if sentence_repetition > 0.5:
                repetition_penalty += 0.2
        
        # --- Signal 6: Garbage/noise detection ---
        noise_penalty = 0.0
        
        # Check for HTML/code when not asked for
        query_lower = query.lower()
        response_lower = response.lower()
        
        asks_for_code = any(kw in query_lower for kw in ['code', 'html', 'python', 'program', 'script', 'tag', 'function', 'write a'])
        has_code = bool(re.search(r'(import |def |class |<[a-z]+>|function\s*\(|var\s+|#include)', response_lower))
        
        if has_code and not asks_for_code:
            noise_penalty += 0.15
        
        # Check for "Input:/Output:" repetition patterns (common in bad outputs)
        io_pattern_count = len(re.findall(r'(input:|output:|question:|answer:)', response_lower))
        if io_pattern_count > 3:
            noise_penalty += 0.15
        
        # --- Signal 7: Semantic coherence via shared vocabulary domains ---
        # Build simple topic vectors using character trigrams
        def char_trigrams(text, max_chars=500):
            text = text.lower()[:max_chars]
            trigrams = Counter()
            for i in range(len(text) - 2):
                trigrams[text[i:i+3]] += 1
            return trigrams
        
        q_trigrams = char_trigrams(query)
        r_trigrams = char_trigrams(response)
        
        # Cosine similarity of character trigram vectors
        common_keys = set(q_trigrams.keys()) & set(r_trigrams.keys())
        dot_product = sum(q_trigrams[k] * r_trigrams[k] for k in common_keys)
        mag_q = math.sqrt(sum(v**2 for v in q_trigrams.values()))
        mag_r = math.sqrt(sum(v**2 for v in r_trigrams.values()))
        
        if mag_q > 0 and mag_r > 0:
            trigram_similarity = dot_product / (mag_q * mag_r)
        else:
            trigram_similarity = 0.0
        
        # --- Signal 8: Direct address of query intent ---
        # Check if response contains question words that echo the query
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does'}
        query_has_question = any(w in question_words for w in query_tokens[:5])
        
        # If query is a question, the response should NOT just be a single word
        single_word_answer_penalty = 0.0
        if query_has_question and len(response_tokens) <= 2:
            single_word_answer_penalty = 0.25
        
        # --- Signal 9: Check for "meta" non-answers ---
        meta_phrases = ['you can tell us', 'let me know', 'i don\'t know', 'no answer', 'n/a']
        meta_penalty = 0.0
        for phrase in meta_phrases:
            if phrase in response_lower:
                meta_penalty += 0.1
        
        # --- Signal 10: Substantive content ratio ---
        # Ratio of content words to total words
        if len(response_tokens) > 0:
            content_ratio = len(response_content) / len(response_tokens)
        else:
            content_ratio = 0.0
        
        # --- Combine signals into final score ---
        # Relevance component (0-1 scale)
        relevance = (
            0.25 * query_recall +
            0.15 * weighted_recall +
            0.15 * bigram_recall +
            0.20 * topic_coverage +
            0.25 * trigram_similarity
        )
        
        # Quality component (0-1 scale)
        quality = (
            0.40 * length_score +
            0.30 * min(content_ratio * 1.5, 1.0) +
            0.30 * unique_ratio
        )
        
        # Combined score before penalties
        raw_score = 0.65 * relevance + 0.35 * quality
        
        # Apply penalties
        total_penalty = repetition_penalty + noise_penalty + single_word_answer_penalty + meta_penalty
        total_penalty = min(total_penalty, 0.7)  # Cap penalty
        
        penalized_score = raw_score * (1.0 - total_penalty)
        
        # Scale to 0-10 range with some non-linearity to spread scores
        # Use a sigmoid-like mapping to make discrimination clearer
        scaled = penalized_score * 10.0
        
        # Apply a slight power curve to spread the middle range
        if scaled > 0:
            final_score = 10.0 * (scaled / 10.0) ** 0.8
        else:
            final_score = 0.0
        
        # Clamp to range
        final_score = max(0.0, min(10.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 2)
        
    except Exception:
        # Never crash - return a neutral score on error
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except Exception:
            return 3.0