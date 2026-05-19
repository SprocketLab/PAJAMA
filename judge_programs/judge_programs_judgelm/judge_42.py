def judging_function(query, response):
    """
    Evaluate clarity and conciseness using sentence-level analysis,
    information density metrics, and structural quality signals.
    
    This variant focuses on:
    - Sentence-level clarity (length distribution, complexity)
    - Information density (unique content words ratio)
    - Structural coherence signals
    - Noise detection (HTML, code, garbage patterns)
    - Response completeness relative to query
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # === 1. BASIC LENGTH ADEQUACY ===
        resp_len = len(response_stripped)
        word_tokens = response_stripped.split()
        num_words = len(word_tokens)
        
        if num_words == 0:
            return 0.0
        
        # Very short responses (1-2 words) are usually low quality
        if num_words <= 2:
            # But check if query expects a short answer (identification tasks)
            query_lower = query.lower()
            expects_short = any(kw in query_lower for kw in [
                'identify', 'which', 'name the', 'biggest', 'largest', 
                'smallest', 'what is the name'
            ])
            if expects_short and num_words >= 1:
                length_score = 4.0
            else:
                length_score = 1.0
        elif num_words <= 5:
            length_score = 3.5
        elif num_words <= 200:
            length_score = 7.0
        elif num_words <= 400:
            length_score = 5.5
        else:
            length_score = 4.0  # Too long often means bloated
        
        # === 2. SENTENCE QUALITY ANALYSIS ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length (words per sentence) - moderate is best
        words_per_sentence = num_words / num_sentences
        if 8 <= words_per_sentence <= 25:
            sentence_len_score = 8.0
        elif 5 <= words_per_sentence < 8 or 25 < words_per_sentence <= 35:
            sentence_len_score = 6.0
        elif 3 <= words_per_sentence < 5 or 35 < words_per_sentence <= 50:
            sentence_len_score = 4.0
        else:
            sentence_len_score = 2.0
        
        # Sentence length variance - consistent is better
        if num_sentences > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((sl - mean_sl) ** 2 for sl in sent_lengths) / len(sent_lengths)
            cv = math.sqrt(variance) / max(mean_sl, 1)
            if cv < 0.5:
                uniformity_score = 8.0
            elif cv < 1.0:
                uniformity_score = 6.0
            else:
                uniformity_score = 3.0
        else:
            uniformity_score = 5.0
        
        # === 3. INFORMATION DENSITY ===
        # Ratio of unique content words to total words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'while',
            'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'these', 'those', 'am', 'about', 'up',
        }
        
        lower_tokens = [w.lower().strip(string.punctuation) for w in word_tokens]
        lower_tokens = [w for w in lower_tokens if w]
        content_words = [w for w in lower_tokens if w not in stop_words and len(w) > 1]
        
        if len(lower_tokens) > 0:
            content_ratio = len(content_words) / len(lower_tokens)
        else:
            content_ratio = 0
        
        # Unique content words ratio (penalizes repetition)
        if len(content_words) > 0:
            unique_content = len(set(content_words))
            uniqueness_ratio = unique_content / len(content_words)
        else:
            uniqueness_ratio = 0
        
        # Information density score
        density_score = (content_ratio * 5 + uniqueness_ratio * 5)
        density_score = min(10.0, max(0.0, density_score))
        
        # === 4. REPETITION DETECTION (n-gram based) ===
        # Check for repeated bigrams and trigrams
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        repetition_penalty = 0.0
        
        if len(lower_tokens) >= 4:
            bigrams = get_ngrams(lower_tokens, 2)
            if bigrams:
                bigram_counts = Counter(bigrams)
                repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
                bigram_rep_ratio = repeated_bigrams / max(len(set(bigrams)), 1)
                repetition_penalty += bigram_rep_ratio * 3
        
        if len(lower_tokens) >= 6:
            trigrams = get_ngrams(lower_tokens, 3)
            if trigrams:
                trigram_counts = Counter(trigrams)
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
                trigram_rep_ratio = repeated_trigrams / max(len(set(trigrams)), 1)
                repetition_penalty += trigram_rep_ratio * 4
        
        # Check for repeated sentences
        if num_sentences > 1:
            sent_normalized = [re.sub(r'\s+', ' ', s.lower().strip()) for s in sentences]
            unique_sents = len(set(sent_normalized))
            sent_rep_ratio = 1 - (unique_sents / num_sentences)
            repetition_penalty += sent_rep_ratio * 5
        
        repetition_penalty = min(repetition_penalty, 6.0)
        
        # === 5. NOISE AND GARBAGE DETECTION ===
        noise_penalty = 0.0
        
        # HTML tags
        html_tags = re.findall(r'<[^>]+>', response_stripped)
        if len(html_tags) > 2:
            noise_penalty += min(len(html_tags) * 0.5, 3.0)
        
        # Code-like patterns (imports, function defs, etc.) when not asked for
        query_asks_code = any(kw in query.lower() for kw in ['code', 'python', 'function', 'program', 'script', 'html'])
        if not query_asks_code:
            code_patterns = re.findall(r'\b(import |def |class |return |print\(|if __name__|\.open\()', response_stripped)
            if len(code_patterns) > 1:
                noise_penalty += min(len(code_patterns) * 1.0, 4.0)
        
        # Excessive special characters
        special_chars = sum(1 for c in response_stripped if c in '{}[]#@$%^&*~`|\\')
        special_ratio = special_chars / max(resp_len, 1)
        if special_ratio > 0.05:
            noise_penalty += min(special_ratio * 20, 3.0)
        
        # "Input:" / "Output:" spam patterns
        io_pattern_count = len(re.findall(r'(Input:|Output:|Question:|Answer:)', response_stripped))
        if io_pattern_count > 3:
            noise_penalty += min((io_pattern_count - 3) * 0.8, 3.0)
        
        # Random incomplete text (ending mid-word or with fragments)
        # Not penalizing truncation too heavily since it might be a display limit
        
        noise_penalty = min(noise_penalty, 6.0)
        
        # === 6. RELEVANCE SIGNAL ===
        # Check if response shares key terms with query
        query_tokens = [w.lower().strip(string.punctuation) for w in query.split()]
        query_content = set(w for w in query_tokens if w not in stop_words and len(w) > 2)
        resp_content_set = set(content_words)
        
        if query_content:
            overlap = len(query_content & resp_content_set)
            relevance_ratio = overlap / len(query_content)
            relevance_score = min(relevance_ratio * 8, 8.0)
        else:
            relevance_score = 5.0
        
        # === 7. CLARITY INDICATORS ===
        clarity_score = 7.0  # Start neutral
        
        # Penalize responses that start with lowercase (often fragments)
        if response_stripped[0].islower() and response_stripped[0].isalpha():
            clarity_score -= 1.0
        
        # Reward responses that have proper sentence structure
        # (starts with capital, has periods)
        if response_stripped[0].isupper():
            clarity_score += 0.5
        
        # Check for complete sentences (ending with punctuation)
        if response_stripped[-1] in '.!?':
            clarity_score += 0.5
        elif response_stripped[-1].isalpha():
            clarity_score -= 0.5  # Truncated
        
        # Penalize single character or meaningless responses
        if response_stripped in {'.', ',', '-', '—', '...', 'no', 'yes'} and num_words <= 1:
            clarity_score = 1.0
        
        # Average word length (very short avg = vague, very long avg = jargon-heavy)
        if lower_tokens:
            avg_word_len = sum(len(w) for w in lower_tokens) / len(lower_tokens)
            if 3.5 <= avg_word_len <= 7.0:
                clarity_score += 1.0
            elif avg_word_len < 2.5 or avg_word_len > 9.0:
                clarity_score -= 1.0
        
        clarity_score = max(0.0, min(10.0, clarity_score))
        
        # === 8. CONCISENESS BONUS ===
        # If response answers well in fewer words, that's good
        # Estimate: responses between 20-150 words for typical queries are concise
        query_complexity = len(query.split())
        
        if query_complexity <= 10:
            ideal_range = (10, 120)
        else:
            ideal_range = (20, 200)
        
        if ideal_range[0] <= num_words <= ideal_range[1]:
            conciseness_bonus = 1.5
        elif num_words < ideal_range[0]:
            conciseness_bonus = -0.5
        elif num_words > ideal_range[1] * 2:
            conciseness_bonus = -2.0
        else:
            conciseness_bonus = 0.0
        
        # === FINAL SCORE COMPOSITION ===
        # Weights chosen to emphasize clarity and conciseness
        final_score = (
            length_score * 0.15 +
            sentence_len_score * 0.10 +
            uniformity_score * 0.05 +
            density_score * 0.15 +
            clarity_score * 0.20 +
            relevance_score * 0.15 +
            (10.0 - repetition_penalty * 1.67) * 0.10 +  # Convert penalty to score
            (10.0 - noise_penalty * 1.67) * 0.10 +       # Convert penalty to score
            conciseness_bonus
        )
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        # Apply floor: truly garbage responses should score very low
        if num_words <= 1 and relevance_score < 2:
            final_score = min(final_score, 1.5)
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0