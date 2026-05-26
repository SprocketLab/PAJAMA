def judging_function(query, response):
    """
    Evaluate language quality and readability using a substantially different approach:
    - Coleman-Liau Index (instead of Flesch)
    - Automated Readability Index (ARI)
    - Punctuation diversity and correctness
    - Sentence structure variance (std dev of sentence lengths)
    - Lexical sophistication (longer unique words ratio)
    - Repetition penalty (consecutive repeated words/phrases)
    - Coherence via sentence-to-sentence word overlap
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        # === Basic tokenization ===
        # Split into sentences using multiple delimiters
        raw_sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            return 1.0
        
        # Tokenize words (alphanumeric only)
        all_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
        
        if not all_words:
            return 1.0
        
        num_words = len(all_words)
        num_sentences = max(len(sentences), 1)
        num_chars = sum(len(w) for w in all_words)
        
        # === 1. Coleman-Liau Index ===
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg number of letters per 100 words
        # S = avg number of sentences per 100 words
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        # Ideal CLI for general audience: 7-12
        # Score: penalize if too high (>16) or too low (<3)
        if cli < 3:
            cli_score = max(0, cli / 3) * 5
        elif 3 <= cli <= 14:
            cli_score = 5 + 5 * (1 - abs(cli - 8.5) / 5.5)
        else:
            cli_score = max(0, 10 - (cli - 14) * 0.5)
        cli_score = max(0, min(10, cli_score))
        
        # === 2. Automated Readability Index ===
        ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        # Ideal ARI: 6-12
        if ari < 2:
            ari_score = max(0, ari) * 2.5
        elif 2 <= ari <= 14:
            ari_score = 5 + 5 * (1 - abs(ari - 8) / 6)
        else:
            ari_score = max(0, 10 - (ari - 14) * 0.5)
        ari_score = max(0, min(10, ari_score))
        
        # === 3. Sentence structure variance ===
        sent_word_counts = []
        for s in sentences:
            words_in_sent = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s)
            sent_word_counts.append(len(words_in_sent))
        
        avg_sent_len = sum(sent_word_counts) / len(sent_word_counts) if sent_word_counts else 0
        
        if len(sent_word_counts) > 1:
            variance = sum((x - avg_sent_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            # Coefficient of variation - normalized variance
            cv = std_dev / avg_sent_len if avg_sent_len > 0 else 0
            # Good variety: cv between 0.2 and 0.6
            if cv < 0.05:
                variety_score = 2.0  # Very monotonous
            elif cv < 0.2:
                variety_score = 2 + (cv - 0.05) / 0.15 * 4
            elif cv <= 0.6:
                variety_score = 6 + (cv - 0.2) / 0.4 * 4
            else:
                variety_score = max(4, 10 - (cv - 0.6) * 5)
        else:
            variety_score = 3.0  # Single sentence, can't assess variety
        variety_score = max(0, min(10, variety_score))
        
        # === 4. Punctuation diversity and usage ===
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-()"\'"':
                punct_types.add(ch)
        
        # Count specific punctuation
        comma_count = text.count(',')
        period_count = text.count('.')
        
        # Punctuation per sentence
        total_punct = sum(1 for ch in text if ch in '.,;:!?')
        punct_per_sent = total_punct / num_sentences
        
        # Diversity of punctuation (more types = better, up to a point)
        punct_diversity = len(punct_types)
        
        # Score: good punctuation usage
        punct_score = min(10, punct_diversity * 1.2 + min(3, punct_per_sent * 0.8))
        
        # Penalize no commas in longer text
        if num_words > 30 and comma_count == 0:
            punct_score *= 0.7
        
        punct_score = max(0, min(10, punct_score))
        
        # === 5. Lexical sophistication ===
        # Proportion of words with 7+ characters (sophisticated words)
        sophisticated = [w for w in all_words if len(w) >= 7]
        sophistication_ratio = len(sophisticated) / num_words
        
        # Unique sophisticated words
        unique_sophisticated = len(set(sophisticated))
        
        # Score: ideal ratio around 0.15-0.35
        if sophistication_ratio < 0.05:
            soph_score = sophistication_ratio / 0.05 * 4
        elif sophistication_ratio <= 0.4:
            soph_score = 4 + (sophistication_ratio - 0.05) / 0.35 * 6
        else:
            soph_score = max(5, 10 - (sophistication_ratio - 0.4) * 10)
        
        # Bonus for unique sophisticated words
        if unique_sophisticated >= 3:
            soph_score = min(10, soph_score + 1)
        
        soph_score = max(0, min(10, soph_score))
        
        # === 6. Repetition penalty ===
        # Check for repeated consecutive words
        consecutive_repeats = 0
        for i in range(1, len(all_words)):
            if all_words[i] == all_words[i-1] and all_words[i] not in {'the', 'a', 'an', 'to', 'and', 'or', 'is', 'was', 'that'}:
                consecutive_repeats += 1
        
        # Check for repeated bigrams
        bigrams = [all_words[i] + ' ' + all_words[i+1] for i in range(len(all_words)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
        
        # Check for repeated trigrams (strong repetition signal)
        trigrams = [' '.join(all_words[i:i+3]) for i in range(len(all_words)-2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
        
        # Repetition ratio
        word_counts = Counter(all_words)
        # Exclude common stop words for repetition check
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 
                       'that', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'this', 'from',
                       'or', 'be', 'not', 'but', 'have', 'has', 'had', 'do', 'does', 'did',
                       'will', 'would', 'could', 'should', 'may', 'might', 'can', 'their',
                       'they', 'them', 'its', 'his', 'her', 'he', 'she', 'we', 'you', 'i',
                       'my', 'your', 'our', 'more', 'than', 'also', 'such'}
        content_words = [w for w in all_words if w not in stop_words and len(w) > 2]
        content_counts = Counter(content_words)
        
        if content_words:
            max_content_freq = max(content_counts.values()) if content_counts else 0
            max_repeat_ratio = max_content_freq / len(content_words)
        else:
            max_repeat_ratio = 0
        
        rep_penalty = 0
        rep_penalty += min(5, consecutive_repeats * 1.5)
        rep_penalty += min(3, repeated_bigrams * 1.0)
        rep_penalty += min(4, repeated_trigrams * 0.8)
        rep_penalty += min(3, max(0, max_repeat_ratio - 0.15) * 20)
        
        repetition_score = max(0, 10 - rep_penalty)
        
        # === 7. Coherence: sentence-to-sentence word overlap ===
        if len(sentences) > 1:
            overlaps = []
            for i in range(len(sentences) - 1):
                words_a = set(re.findall(r"[a-zA-Z]+", sentences[i].lower())) - stop_words
                words_b = set(re.findall(r"[a-zA-Z]+", sentences[i+1].lower())) - stop_words
                if words_a and words_b:
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    overlaps.append(overlap)
            
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                # Ideal overlap: 0.1-0.4 (some connection but not too repetitive)
                if avg_overlap < 0.05:
                    coherence_score = 3.0  # Disconnected
                elif avg_overlap <= 0.4:
                    coherence_score = 3 + (avg_overlap - 0.05) / 0.35 * 7
                elif avg_overlap <= 0.7:
                    coherence_score = 10 - (avg_overlap - 0.4) / 0.3 * 3
                else:
                    coherence_score = max(3, 7 - (avg_overlap - 0.7) * 10)
            else:
                coherence_score = 5.0
        else:
            coherence_score = 5.0
        coherence_score = max(0, min(10, coherence_score))
        
        # === 8. Capitalization correctness ===
        cap_errors = 0
        # Check sentence starts
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isalpha() and not s_stripped[0].isupper():
                cap_errors += 1
        
        # Check for ALL CAPS words (more than 2 chars, not acronyms)
        all_caps_words = [w for w in re.findall(r'\b[A-Z]{3,}\b', text) if len(w) > 3]
        cap_errors += len(all_caps_words)
        
        cap_score = max(0, 10 - cap_errors * 2)
        
        # === 9. Response completeness ===
        # Check if response seems truncated
        last_char = text[-1] if text else ''
        completeness_score = 10.0
        if last_char not in '.!?")\']':
            completeness_score -= 3.0
        # Check for very short responses relative to query
        query_words = len(re.findall(r'[a-zA-Z]+', query)) if query else 1
        response_query_ratio = num_words / max(query_words, 1)
        if response_query_ratio < 0.5:
            completeness_score -= 2.0
        if num_words < 5:
            completeness_score -= 3.0
        completeness_score = max(0, min(10, completeness_score))
        
        # === 10. Average word length distribution (Zipf-like naturalness) ===
        word_lengths = [len(w) for w in all_words]
        length_dist = Counter(word_lengths)
        # Natural English has most words 2-5 chars
        short_ratio = sum(1 for l in word_lengths if 2 <= l <= 5) / num_words
        # Ideal: ~0.55-0.70
        if 0.45 <= short_ratio <= 0.75:
            naturalness_score = 8 + (1 - abs(short_ratio - 0.6) / 0.15) * 2
        else:
            naturalness_score = max(3, 8 - abs(short_ratio - 0.6) * 15)
        naturalness_score = max(0, min(10, naturalness_score))
        
        # === Combine scores with weights ===
        weights = {
            'cli': 0.08,
            'ari': 0.08,
            'variety': 0.12,
            'punct': 0.08,
            'sophistication': 0.10,
            'repetition': 0.20,
            'coherence': 0.10,
            'capitalization': 0.06,
            'completeness': 0.10,
            'naturalness': 0.08,
        }
        
        scores = {
            'cli': cli_score,
            'ari': ari_score,
            'variety': variety_score,
            'punct': punct_score,
            'sophistication': soph_score,
            'repetition': repetition_score,
            'coherence': coherence_score,
            'capitalization': cap_score,
            'completeness': completeness_score,
            'naturalness': naturalness_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Scale to 0-100
        final_score = final_score * 10
        
        # Clamp
        final_score = max(0, min(100, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Minimal fallback: score based on basic text properties
            if not response or not response.strip():
                return 0.0
            words = response.split()
            return min(50, len(words) * 0.5 + 10)
        except Exception:
            return 25.0