def judging_function(query, response):
    """
    Evaluates language quality and readability using a substantially different approach:
    - Punctuation correctness and density
    - Capitalization patterns (proper sentence starts)
    - Repetition detection (repeated phrases/lines)
    - Character-level entropy (information density)
    - Conjunction/connective variety
    - Sentence structure variance (coefficient of variation of sentence lengths)
    - Proportion of "content" vs "function" words
    - Penalize HTML/code artifacts, raw markup
    - Reward well-formed paragraph structure
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        # Very short responses get low scores
        if len(text) < 5:
            return 0.5
        
        score = 0.0
        
        # ===== 1. Character-level entropy (information density) =====
        char_counts = Counter(text.lower())
        total_chars = len(text)
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Normalize: good English text typically has entropy ~4.0-5.0
        # Very repetitive text has low entropy
        entropy_score = min(char_entropy / 4.5, 1.0) * 10
        score += entropy_score * 0.8
        
        # ===== 2. Capitalization quality =====
        # Check that sentences start with capital letters
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 0:
            cap_correct = 0
            for s in sentences:
                if s and s[0].isupper():
                    cap_correct += 1
            cap_ratio = cap_correct / len(sentences)
            score += cap_ratio * 8 * 0.5
        
        # ===== 3. Punctuation density and variety =====
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return 0.5
        
        punct_chars = set('.,;:!?-—()""\'')
        punct_found = set()
        punct_count = 0
        for ch in text:
            if ch in punct_chars:
                punct_count += 1
                punct_found.add(ch)
        
        # Punctuation density: roughly 1 punct per 8-15 words is normal
        punct_density = punct_count / max(num_words, 1)
        # Ideal range: 0.05 to 0.2
        if 0.04 <= punct_density <= 0.25:
            punct_density_score = 1.0
        elif punct_density < 0.04:
            punct_density_score = punct_density / 0.04
        else:
            punct_density_score = max(0, 1.0 - (punct_density - 0.25) / 0.5)
        
        # Variety of punctuation used
        punct_variety_score = min(len(punct_found) / 4.0, 1.0)
        
        score += (punct_density_score * 0.5 + punct_variety_score * 0.5) * 6 * 0.5
        
        # ===== 4. Repetition detection =====
        # Check for repeated lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counter = Counter(lines)
            most_common_count = line_counter.most_common(1)[0][1]
            repetition_ratio = most_common_count / len(lines)
            if repetition_ratio > 0.5:
                score -= 3.0  # Heavy penalty for repeated lines
        
        # Check for repeated trigrams (word-level)
        if num_words >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counter = Counter(trigrams)
            if trigrams:
                total_trigrams = len(trigrams)
                unique_trigrams = len(trigram_counter)
                trigram_uniqueness = unique_trigrams / total_trigrams
                # If very repetitive (low uniqueness), penalize
                if trigram_uniqueness < 0.5:
                    score -= (0.5 - trigram_uniqueness) * 6
        
        # ===== 5. Sentence structure variance =====
        sentence_lengths = []
        raw_sentences = re.split(r'[.!?]+', text)
        for s in raw_sentences:
            sw = s.strip().split()
            if len(sw) > 0:
                sentence_lengths.append(len(sw))
        
        if len(sentence_lengths) >= 2:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len  # coefficient of variation
                # Some variance is good (0.2-0.8), too uniform or too wild is bad
                if 0.15 <= cv <= 1.0:
                    variety_score = 1.0
                elif cv < 0.15:
                    variety_score = cv / 0.15
                else:
                    variety_score = max(0, 1.0 - (cv - 1.0) / 2.0)
                score += variety_score * 6 * 0.5
            
            # Reward having multiple sentences (shows developed response)
            num_sents = len(sentence_lengths)
            if num_sents >= 3:
                score += 1.5
            elif num_sents >= 2:
                score += 0.8
        elif len(sentence_lengths) == 1:
            # Single sentence - check if it's reasonable length
            if 5 <= sentence_lengths[0] <= 30:
                score += 1.0
        
        # ===== 6. Content word ratio =====
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'
        }
        
        lower_words = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words]
        lower_words = [w for w in lower_words if w]
        
        if lower_words:
            content_words = [w for w in lower_words if w not in function_words and len(w) > 1]
            content_ratio = len(content_words) / len(lower_words)
            # Good ratio is around 0.4-0.7
            if 0.35 <= content_ratio <= 0.75:
                score += 2.0
            elif content_ratio > 0.75:
                score += 1.0  # Might be keyword-heavy
            else:
                score += content_ratio * 3.0
        
        # ===== 7. Penalize code/HTML artifacts =====
        html_pattern = re.findall(r'<[^>]+>', text)
        code_indicators = ['import ', 'def ', 'class ', 'return ', 'print(', '#!/', 'void ', 'int ']
        
        html_ratio = len(''.join(html_pattern)) / max(len(text), 1)
        if html_ratio > 0.15:
            score -= 3.0
        elif html_ratio > 0.05:
            score -= 1.5
        
        code_count = sum(1 for ind in code_indicators if ind in text)
        if code_count >= 3:
            score -= 2.5
        elif code_count >= 1:
            score -= 0.5
        
        # ===== 8. Word length distribution (proxy for vocabulary sophistication) =====
        if lower_words:
            word_lengths = [len(w) for w in lower_words if w.isalpha()]
            if word_lengths:
                avg_word_len = sum(word_lengths) / len(word_lengths)
                # Good English prose: avg word length ~4.5-6.0
                if 4.0 <= avg_word_len <= 6.5:
                    score += 2.0
                elif 3.0 <= avg_word_len < 4.0:
                    score += 1.0
                elif avg_word_len > 6.5:
                    score += 1.5
                
                # Also check for presence of longer words (vocabulary richness)
                long_words = [w for w in word_lengths if w >= 8]
                long_word_ratio = len(long_words) / len(word_lengths)
                score += min(long_word_ratio * 10, 2.0)
        
        # ===== 9. Response length adequacy =====
        # Very short responses tend to be lower quality
        if num_words < 3:
            score -= 2.0
        elif num_words < 10:
            # Short but might be ok for simple queries
            score += 0.5
        elif 10 <= num_words <= 300:
            score += 2.0
        elif num_words > 300:
            score += 1.5  # Slightly less for very long (might be rambling)
        
        # ===== 10. Detect "Output:" prefix repetition pattern =====
        output_prefix_count = text.count('Output:')
        if output_prefix_count > 2:
            score -= 1.5
        
        # Detect "Question:" / "Answer:" repetitive patterns
        qa_pattern_count = len(re.findall(r'(?:Question|Answer|Input|Output):', text))
        if qa_pattern_count > 4:
            score -= 2.0
        
        # ===== 11. Proportion of alphabetic characters =====
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        alpha_ratio = alpha_chars / max(len(text), 1)
        if alpha_ratio < 0.5:
            score -= 2.0  # Too many non-alpha characters (code, symbols)
        elif alpha_ratio >= 0.7:
            score += 1.0
        
        # ===== 12. Unique word ratio (type-token but weighted differently) =====
        if lower_words and len(lower_words) >= 5:
            # Use a windowed approach: check uniqueness in sliding windows
            window_size = min(50, len(lower_words))
            windows = []
            for i in range(0, len(lower_words) - window_size + 1, max(window_size // 2, 1)):
                window = lower_words[i:i + window_size]
                unique_in_window = len(set(window)) / len(window)
                windows.append(unique_in_window)
            
            if windows:
                avg_window_uniqueness = sum(windows) / len(windows)
                # Good prose: ~0.5-0.8 unique ratio in windows
                score += min(avg_window_uniqueness, 0.85) * 3.0
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Map from [0,10] through a curve that emphasizes differences
        normalized = score / 10.0
        # Slight S-curve
        if normalized <= 0.5:
            adjusted = 2 * normalized * normalized
        else:
            adjusted = 1 - 2 * (1 - normalized) * (1 - normalized)
        
        final_score = adjusted * 10.0
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0