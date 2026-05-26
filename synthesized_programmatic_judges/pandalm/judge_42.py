def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, redundancy detection,
    and structural quality metrics. Uses compression-ratio inspired approach and
    unique information content measurement.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        response = response.strip()
        query = query.strip() if query else ""
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        if not words:
            return 0.5
        
        total_words = len(words)
        unique_words = set(words)
        num_unique = len(unique_words)
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Information Entropy (higher = more diverse vocabulary = clearer communication) ----
        word_counts = Counter(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize entropy by max possible entropy
        max_entropy = math.log2(total_words) if total_words > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # ---- 2. Repetition penalty using n-gram redundancy ----
        def ngram_redundancy(tokens, n):
            if len(tokens) < n:
                return 0.0
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            if not ngrams:
                return 0.0
            unique_ngrams = set(ngrams)
            # Ratio of repeated ngrams
            redundancy = 1.0 - (len(unique_ngrams) / len(ngrams))
            return redundancy
        
        bigram_redundancy = ngram_redundancy(words, 2)
        trigram_redundancy = ngram_redundancy(words, 3)
        
        # Weighted redundancy score (0 = no redundancy, 1 = fully redundant)
        redundancy_score = 0.4 * bigram_redundancy + 0.6 * trigram_redundancy
        
        # ---- 3. Phrase-level repetition detection ----
        # Look for repeated phrases (4+ word sequences)
        phrase_rep_penalty = 0.0
        if total_words >= 8:
            four_grams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
            four_gram_counts = Counter(four_grams)
            repeated_4grams = sum(c - 1 for c in four_gram_counts.values() if c > 1)
            phrase_rep_penalty = min(repeated_4grams / max(len(four_grams), 1) * 3, 1.0)
        
        # ---- 4. Sentence-level clarity metrics ----
        avg_sentence_length = total_words / num_sentences
        
        # Ideal sentence length: 10-20 words. Penalize very long or very short.
        if avg_sentence_length < 3:
            sentence_length_score = 0.3
        elif avg_sentence_length <= 8:
            sentence_length_score = 0.6 + 0.4 * ((avg_sentence_length - 3) / 5)
        elif avg_sentence_length <= 22:
            sentence_length_score = 1.0
        elif avg_sentence_length <= 40:
            sentence_length_score = 1.0 - 0.5 * ((avg_sentence_length - 22) / 18)
        else:
            sentence_length_score = 0.3
        
        # ---- 5. Lexical density (content words vs function words) ----
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'they', 'them', 'their', 'while', 'also', 'then'
        }
        content_words = [w for w in words if w not in function_words]
        lexical_density = len(content_words) / total_words if total_words > 0 else 0
        # Ideal lexical density around 0.4-0.6
        if 0.35 <= lexical_density <= 0.65:
            density_score = 1.0
        elif lexical_density < 0.35:
            density_score = 0.5 + 0.5 * (lexical_density / 0.35)
        else:
            density_score = max(0.5, 1.0 - (lexical_density - 0.65) * 2)
        
        # ---- 6. Response substantiveness (not too short, not too long) ----
        # Penalize extremely short responses that may lack substance
        if total_words < 3:
            substance_score = 0.2
        elif total_words < 8:
            substance_score = 0.4 + 0.1 * (total_words - 3)
        elif total_words <= 150:
            substance_score = 1.0
        elif total_words <= 300:
            substance_score = 1.0 - 0.3 * ((total_words - 150) / 150)
        else:
            substance_score = 0.5
        
        # ---- 7. Unique information ratio (type-token ratio variant) ----
        # Use root TTR for better normalization across lengths
        root_ttr = num_unique / math.sqrt(total_words) if total_words > 0 else 0
        # Typical root TTR ranges from ~3 to ~10
        ttr_score = min(root_ttr / 8.0, 1.0)
        
        # ---- 8. Filler/hedge word penalty ----
        filler_words = [
            'basically', 'actually', 'really', 'very', 'quite', 'rather',
            'somewhat', 'perhaps', 'maybe', 'possibly', 'probably',
            'essentially', 'literally', 'honestly', 'frankly', 'simply',
            'obviously', 'clearly', 'naturally', 'certainly', 'definitely',
            'absolutely', 'totally', 'completely', 'extremely', 'incredibly'
        ]
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / total_words if total_words > 0 else 0
        filler_penalty = min(filler_ratio * 10, 1.0)  # 0 to 1 penalty
        
        # ---- 9. Sentence start diversity ----
        if num_sentences >= 2:
            sentence_starts = []
            for s in sentences:
                s_words = re.findall(r'[a-zA-Z]+', s.lower())
                if s_words:
                    # Use first two words as sentence start signature
                    start = tuple(s_words[:min(2, len(s_words))])
                    sentence_starts.append(start)
            if sentence_starts:
                unique_starts = len(set(sentence_starts))
                start_diversity = unique_starts / len(sentence_starts)
            else:
                start_diversity = 0.5
        else:
            start_diversity = 0.7  # Single sentence, neutral
        
        # ---- 10. Query relevance via keyword overlap ----
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - function_words
        response_content = set(content_words)
        if query_words:
            relevance = len(query_words & response_content) / len(query_words)
        else:
            relevance = 0.5
        
        # ---- Combine scores ----
        # Weights chosen to emphasize clarity dimensions
        score = (
            normalized_entropy * 15 +          # Vocabulary diversity (0-15)
            (1 - redundancy_score) * 20 +      # Low redundancy (0-20)
            (1 - phrase_rep_penalty) * 15 +     # No phrase repetition (0-15)
            sentence_length_score * 10 +        # Good sentence length (0-10)
            density_score * 8 +                 # Lexical density (0-8)
            substance_score * 12 +              # Substantive content (0-12)
            ttr_score * 8 +                     # Vocabulary richness (0-8)
            (1 - filler_penalty) * 5 +          # No filler words (0-5)
            start_diversity * 5 +               # Sentence variety (0-5)
            relevance * 2                       # Query relevance (0-2)
        )
        # Max theoretical: 100
        
        # Clamp
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: return a middling score
        try:
            if response and response.strip():
                return 30.0
            return 0.0
        except Exception:
            return 0.0