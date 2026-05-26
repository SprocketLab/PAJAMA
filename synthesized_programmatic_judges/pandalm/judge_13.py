def judging_function(query, response):
    """
    Evaluate language quality and readability using:
    - Coleman-Liau Index (character-based readability, different from Flesch)
    - Gunning Fog Index approximation
    - Sentence structure variety (std dev of sentence lengths)
    - Lexical sophistication (longer unique words ratio)
    - Repetition penalty (duplicate n-grams)
    - Punctuation variety and correctness
    - Conjunction/connective diversity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str) or response.strip() == "":
            return 0.0
        
        text = response.strip()
        
        # Basic tokenization
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        if len(words) == 0:
            return 0.5
        
        # Sentence splitting - more sophisticated
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(re.findall(r'[a-zA-Z]+', s)) > 0]
        if len(sentences) == 0:
            sentences = [text]
        
        num_words = len(words)
        num_sentences = len(sentences)
        num_chars = sum(len(w) for w in words)  # letters only
        
        # ---- 1. Coleman-Liau Index ----
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg letters per 100 words, S = avg sentences per 100 words
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        # Ideal range: 8-14 for good readability
        # Score: penalize if too low or too high
        cli_score = max(0, 10 - abs(coleman_liau - 11) * 0.8)
        
        # ---- 2. Gunning Fog approximation ----
        # Complex words: words with 3+ syllables (approximate by character count >= 7)
        complex_words = [w for w in words if len(w) >= 7]
        complex_ratio = len(complex_words) / num_words if num_words > 0 else 0
        avg_sent_len = num_words / num_sentences if num_sentences > 0 else num_words
        fog_index = 0.4 * (avg_sent_len + 100 * complex_ratio)
        # Ideal: 8-14
        fog_score = max(0, 10 - abs(fog_index - 11) * 0.6)
        
        # ---- 3. Sentence length variety ----
        sent_lengths = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s)
            sent_lengths.append(len(s_words))
        
        if len(sent_lengths) >= 2:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Some variety is good, but not extreme
            # Coefficient of variation
            cv = std_sl / mean_sl if mean_sl > 0 else 0
            variety_score = min(10, cv * 20)  # cv of 0.5 -> 10
        else:
            variety_score = 2.0  # single sentence gets low variety score
        
        # ---- 4. Lexical sophistication ----
        # Ratio of words with length >= 6 that are unique
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        
        # Type-token ratio adjusted: use root TTR (Guiraud's index)
        guiraud = len(unique_words) / math.sqrt(num_words) if num_words > 0 else 0
        # Typical range: 4-8 for good text
        lexical_score = min(10, guiraud * 1.5)
        
        # ---- 5. Repetition penalty ----
        # Bigram and trigram repetition
        if len(lower_words) >= 2:
            bigrams = [f"{lower_words[i]} {lower_words[i+1]}" for i in range(len(lower_words)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
            bigram_rep_ratio = repeated_bigrams / total_bigrams if total_bigrams > 0 else 0
        else:
            bigram_rep_ratio = 0
        
        if len(lower_words) >= 3:
            trigrams = [f"{lower_words[i]} {lower_words[i+1]} {lower_words[i+2]}" for i in range(len(lower_words)-2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0
        else:
            trigram_rep_ratio = 0
        
        # Heavy penalty for repetition
        repetition_penalty = (bigram_rep_ratio * 5 + trigram_rep_ratio * 15)
        repetition_score = max(0, 10 - repetition_penalty)
        
        # ---- 6. Punctuation variety and usage ----
        punct_types_used = set()
        for ch in text:
            if ch in '.,;:!?—-()""\'':
                punct_types_used.add(ch)
        
        # Count commas, semicolons, colons per sentence
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        
        punct_variety = len(punct_types_used)
        # More variety of punctuation = better (up to a point)
        punct_score = min(10, punct_variety * 1.8)
        
        # Bonus for appropriate comma usage (roughly 1 comma per 10-15 words is natural)
        expected_commas = num_words / 12
        comma_ratio = comma_count / expected_commas if expected_commas > 0 else 0
        comma_bonus = max(0, 2 - abs(comma_ratio - 1) * 2)
        punct_score = min(10, punct_score + comma_bonus)
        
        # ---- 7. Connective/discourse marker diversity ----
        connectives = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'nevertheless', 'consequently', 'meanwhile', 'alternatively',
            'specifically', 'particularly', 'notably', 'importantly',
            'although', 'whereas', 'while', 'since', 'because',
            'thus', 'hence', 'accordingly', 'indeed', 'certainly',
            'overall', 'finally', 'initially', 'subsequently',
            'for example', 'in addition', 'on the other hand',
            'as a result', 'in contrast', 'in fact', 'such as',
            'also', 'but', 'yet', 'still', 'then', 'next'
        }
        
        lower_text = text.lower()
        connectives_found = set()
        for c in connectives:
            if c in lower_text:
                connectives_found.add(c)
        
        connective_score = min(10, len(connectives_found) * 2.0)
        
        # ---- 8. Completeness check ----
        # Check if response ends mid-sentence (truncation)
        last_char = text.rstrip()[-1] if text.rstrip() else ''
        completeness_score = 10.0
        if last_char not in '.!?"\')':
            completeness_score = 4.0  # likely truncated
        
        # Check for word-level repetition (same word repeated many times)
        word_counts = Counter(lower_words)
        # Exclude common function words
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'and', 'or', 'but', 'not', 'that', 'this', 'it', 'its',
                         'as', 'if', 'their', 'they', 'them', 'he', 'she', 'his',
                         'her', 'we', 'our', 'you', 'your', 'i', 'my', 'me'}
        
        content_words = {w: c for w, c in word_counts.items() if w not in function_words}
        if content_words:
            max_content_freq = max(content_words.values())
            if max_content_freq > 3 and num_words > 10:
                dominance = max_content_freq / num_words
                if dominance > 0.1:
                    repetition_score = max(0, repetition_score - dominance * 30)
        
        # ---- 9. Response length adequacy ----
        # Very short responses are usually worse
        length_score = min(10, math.log(num_words + 1) * 2.5)
        # But penalize extremely long responses slightly
        if num_words > 300:
            length_score = max(5, length_score - (num_words - 300) * 0.005)
        
        # ---- 10. Average word length (sophistication proxy) ----
        avg_word_len = num_chars / num_words if num_words > 0 else 0
        # Ideal: 4.5-5.5
        awl_score = max(0, 10 - abs(avg_word_len - 5.0) * 2.5)
        
        # ---- Combine scores with weights ----
        weights = {
            'cli': 0.10,
            'fog': 0.08,
            'variety': 0.10,
            'lexical': 0.15,
            'repetition': 0.18,
            'punct': 0.07,
            'connective': 0.07,
            'completeness': 0.10,
            'length': 0.10,
            'awl': 0.05,
        }
        
        scores = {
            'cli': cli_score,
            'fog': fog_score,
            'variety': variety_score,
            'lexical': lexical_score,
            'repetition': repetition_score,
            'punct': punct_score,
            'connective': connective_score,
            'completeness': completeness_score,
            'length': length_score,
            'awl': awl_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Scale to 0-100
        final_score = final_score * 10
        
        # Clamp
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 25.0
        except Exception:
            pass
        return 0.0