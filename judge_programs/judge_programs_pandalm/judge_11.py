def judging_function(query, response):
    """
    Evaluates language quality and readability using:
    - Sentence structure variety (length variance)
    - Punctuation diversity and correctness
    - Type-token ratio with hapax legomena ratio
    - Average word length distribution (not just mean)
    - Capitalization correctness
    - Repetition penalty (n-gram repetition)
    - Coleman-Liau readability index (different from Flesch)
    - Paragraph/structural coherence signals
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        # Basic tokenization
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        if len(words) == 0:
            return 0.5
        
        num_words = len(words)
        words_lower = [w.lower() for w in words]
        
        # --- 1. Sentence structure variety (std dev of sentence lengths) ---
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        sent_lengths = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s)
            sent_lengths.append(len(sw))
        
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        
        # Sentence length variety score
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Some variety is good, too much is chaotic
            variety_score = min(std_sl / 5.0, 1.0) * 10  # 0-10
        else:
            variety_score = 2.0  # Single sentence gets low variety score
        
        # Penalize very short or very long average sentence length
        if avg_sent_len < 3:
            sent_len_penalty = -5
        elif avg_sent_len > 40:
            sent_len_penalty = -3
        elif 8 <= avg_sent_len <= 25:
            sent_len_penalty = 3
        else:
            sent_len_penalty = 0
        
        # --- 2. Coleman-Liau Index ---
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg number of letters per 100 words
        # S = avg number of sentences per 100 words
        total_letters = sum(len(w) for w in words)
        L = (total_letters / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        
        # Ideal CLI is around 8-14 for general audience
        if 6 <= cli <= 16:
            cli_score = 10
        elif 4 <= cli <= 18:
            cli_score = 7
        elif 2 <= cli <= 22:
            cli_score = 4
        else:
            cli_score = 1
        
        # --- 3. Vocabulary richness: Type-Token Ratio + Hapax ratio ---
        word_freq = Counter(words_lower)
        unique_words = len(word_freq)
        
        # Type-token ratio (adjusted for text length using root TTR / Guiraud's index)
        guiraud = unique_words / math.sqrt(num_words) if num_words > 0 else 0
        # Typical range: 3-10 for good text
        ttr_score = min(guiraud / 7.0, 1.0) * 10  # 0-10
        
        # Hapax legomena ratio (words appearing exactly once)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(unique_words, 1)
        # Higher hapax ratio = richer vocabulary
        hapax_score = hapax_ratio * 8  # 0-8
        
        # --- 4. N-gram repetition penalty ---
        # Check bigram and trigram repetition
        def ngram_repetition_score(tokens, n):
            if len(tokens) < n:
                return 0.0
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            ngram_freq = Counter(ngrams)
            total_ngrams = len(ngrams)
            if total_ngrams == 0:
                return 0.0
            repeated = sum(c - 1 for c in ngram_freq.values() if c > 1)
            repetition_ratio = repeated / total_ngrams
            return repetition_ratio
        
        bigram_rep = ngram_repetition_score(words_lower, 2)
        trigram_rep = ngram_repetition_score(words_lower, 3)
        fourgram_rep = ngram_repetition_score(words_lower, 4)
        
        # Heavy penalty for high repetition
        repetition_penalty = -(bigram_rep * 8 + trigram_rep * 15 + fourgram_rep * 25)
        
        # --- 5. Punctuation diversity and correctness ---
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-—()""\'':
                punct_types.add(ch)
        
        punct_diversity_score = min(len(punct_types) / 4.0, 1.0) * 5  # 0-5
        
        # Check comma usage relative to sentence length
        comma_count = text.count(',')
        comma_ratio = comma_count / max(num_sentences, 1)
        if 0.5 <= comma_ratio <= 3.0:
            comma_score = 3
        elif comma_ratio < 0.5 and num_words > 20:
            comma_score = 1
        else:
            comma_score = 1.5
        
        # --- 6. Capitalization correctness ---
        cap_errors = 0
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isalpha() and not s_stripped[0].isupper():
                cap_errors += 1
        
        cap_score = max(0, 5 - cap_errors * 2)  # 0-5
        
        # --- 7. Word length distribution (entropy-based) ---
        word_lengths = [len(w) for w in words]
        wl_freq = Counter(word_lengths)
        total_wl = sum(wl_freq.values())
        wl_entropy = 0
        for count in wl_freq.values():
            p = count / total_wl
            if p > 0:
                wl_entropy -= p * math.log2(p)
        # Higher entropy = more diverse word lengths
        wl_score = min(wl_entropy / 3.0, 1.0) * 8  # 0-8
        
        # --- 8. Content length adequacy ---
        # Longer, more substantive responses tend to be better (up to a point)
        if num_words < 5:
            length_score = 1
        elif num_words < 15:
            length_score = 4
        elif num_words < 30:
            length_score = 7
        elif num_words < 100:
            length_score = 10
        elif num_words < 200:
            length_score = 9
        else:
            length_score = 7
        
        # --- 9. Discourse markers / connectives ---
        connectives = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'although', 'because',
            'since', 'while', 'whereas', 'thus', 'hence', 'also', 'but',
            'yet', 'still', 'instead', 'otherwise', 'similarly', 'likewise',
            'specifically', 'particularly', 'notably', 'indeed', 'certainly',
            'for example', 'in addition', 'on the other hand', 'as a result',
            'in contrast', 'such as', 'including', 'both', 'either', 'neither'
        }
        connective_count = sum(1 for w in words_lower if w in connectives)
        connective_ratio = connective_count / max(num_words, 1)
        # Some connectives are good, not too many
        if 0.02 <= connective_ratio <= 0.08:
            connective_score = 6
        elif connective_count > 0:
            connective_score = 3
        else:
            connective_score = 1
        
        # --- 10. Detect degenerate/broken text ---
        # Check for excessive repetition of same word
        if word_freq and num_words > 3:
            most_common_word, most_common_count = word_freq.most_common(1)[0]
            # Exclude common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 
                         'and', 'in', 'for', 'on', 'with', 'that', 'it', 'be', 'as',
                         'at', 'by', 'from', 'or', 'not', 'this', 'their', 'its'}
            non_stop_freq = {w: c for w, c in word_freq.items() if w not in stop_words}
            if non_stop_freq:
                top_ns_word, top_ns_count = max(non_stop_freq.items(), key=lambda x: x[1])
                dominance = top_ns_count / num_words
                if dominance > 0.3:
                    degenerate_penalty = -15
                elif dominance > 0.2:
                    degenerate_penalty = -8
                else:
                    degenerate_penalty = 0
            else:
                degenerate_penalty = 0
        else:
            degenerate_penalty = 0
        
        # Check for text that appears truncated (ends mid-word or mid-sentence without punctuation)
        truncation_penalty = 0
        if text and text[-1] not in '.!?)"\'':
            # Might be truncated
            if len(text) > 50:
                truncation_penalty = -3
        
        # --- Combine all scores ---
        total = (
            variety_score +        # 0-10
            sent_len_penalty +     # -5 to 3
            cli_score +            # 1-10
            ttr_score +            # 0-10
            hapax_score +          # 0-8
            repetition_penalty +   # -48 to 0
            punct_diversity_score + # 0-5
            comma_score +          # 1-3
            cap_score +            # 0-5
            wl_score +             # 0-8
            length_score +         # 1-10
            connective_score +     # 1-6
            degenerate_penalty +   # -15 to 0
            truncation_penalty     # -3 to 0
        )
        
        # Normalize to roughly 0-100 range
        # Theoretical max: ~10+3+10+10+8+0+5+3+5+8+10+6+0+0 = 78
        # Theoretical min: ~0-5+1+0+0-48+0+1+0+0+1+1-15-3 = -67
        # Shift and scale
        normalized = (total + 70) / 148 * 100
        
        # Clamp to 0-100
        return max(0.0, min(100.0, round(normalized, 2)))
        
    except Exception:
        return 25.0