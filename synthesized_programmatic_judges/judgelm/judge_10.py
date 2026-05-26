def judging_function(query, response):
    """
    Evaluate language quality and readability using a combination of:
    - Gunning Fog Index (different from Flesch used in Variant 1)
    - Punctuation correctness heuristics
    - Sentence structure variety (std dev of sentence lengths)
    - Hapax legomena ratio (words appearing exactly once)
    - Character-level entropy (information density)
    - Repetition penalty (detecting repeated phrases/sentences)
    - Capitalization correctness
    - Response completeness signals
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        text = response.strip()
        
        # ---- Basic tokenization ----
        # Split into sentences using multiple delimiters
        sentence_endings = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            return 0.5
        
        # Words
        word_pattern = re.compile(r"[a-zA-Z']+")
        all_words = word_pattern.findall(text)
        words_lower = [w.lower() for w in all_words]
        
        if len(all_words) < 1:
            return 0.5
        
        num_words = len(all_words)
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Gunning Fog Index ----
        def count_syllables_word(word):
            word = word.lower()
            if len(word) <= 2:
                return 1
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            # Silent e
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)
        
        complex_words = sum(1 for w in all_words if count_syllables_word(w) >= 3)
        avg_sentence_len = num_words / num_sentences
        complex_word_pct = (complex_words / num_words) * 100 if num_words > 0 else 0
        
        fog_index = 0.4 * (avg_sentence_len + complex_word_pct)
        
        # Ideal fog: 8-14 (readable but not too simple)
        if fog_index < 4:
            fog_score = fog_index / 4 * 4  # too simple
        elif fog_index <= 14:
            fog_score = 7 + 3 * (1 - abs(fog_index - 11) / 7)  # peak around 11
        elif fog_index <= 20:
            fog_score = 7 - (fog_index - 14) * 0.5
        else:
            fog_score = max(0, 4 - (fog_index - 20) * 0.2)
        
        fog_score = max(0, min(10, fog_score))
        
        # ---- 2. Character-level entropy (information density) ----
        char_counts = Counter(text.lower())
        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Good text typically has entropy 4-5 bits
        # Very low entropy = repetitive, very high = noise
        if entropy < 2:
            entropy_score = entropy * 2
        elif entropy <= 5.5:
            entropy_score = 4 + (entropy - 2) * 1.7
        else:
            entropy_score = max(3, 10 - (entropy - 5.5) * 2)
        entropy_score = max(0, min(10, entropy_score))
        
        # ---- 3. Sentence length variety (std dev) ----
        sent_lengths = []
        for s in sentences:
            ws = word_pattern.findall(s)
            sent_lengths.append(len(ws))
        
        if len(sent_lengths) >= 2:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Some variety is good, but not too much
            # Coefficient of variation
            cv = std_sl / mean_sl if mean_sl > 0 else 0
            if cv < 0.1:
                variety_score = 4  # too uniform
            elif cv < 0.5:
                variety_score = 4 + (cv - 0.1) * 15  # up to 10
            elif cv < 1.0:
                variety_score = 10 - (cv - 0.5) * 6
            else:
                variety_score = max(2, 7 - cv * 2)
        else:
            variety_score = 4  # single sentence, neutral
        variety_score = max(0, min(10, variety_score))
        
        # ---- 4. Hapax legomena ratio (vocabulary richness, different from TTR) ----
        word_freq = Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / len(word_freq) if word_freq else 0
        
        # High hapax ratio = rich vocabulary
        hapax_score = hapax_ratio * 10
        hapax_score = max(0, min(10, hapax_score))
        
        # ---- 5. Repetition penalty ----
        # Check for repeated sentences or large repeated phrases
        repetition_penalty = 0
        
        # Sentence-level repetition
        sent_texts = [s.lower().strip() for s in sentences]
        sent_counter = Counter(sent_texts)
        repeated_sents = sum(c - 1 for c in sent_counter.values() if c > 1)
        repetition_penalty += repeated_sents * 1.5
        
        # N-gram repetition (trigrams)
        if len(words_lower) >= 3:
            trigrams = [tuple(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            tri_counter = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in tri_counter.values() if c > 2)
            repetition_penalty += (repeated_trigrams / max(total_trigrams, 1)) * 10
        
        # Check for exact substring repetition (copy-paste detection)
        if len(text) > 50:
            half = len(text) // 2
            first_half = text[:half]
            second_half = text[half:]
            # Simple overlap check
            overlap_len = min(len(first_half), len(second_half), 40)
            if overlap_len > 20 and first_half[:overlap_len] == second_half[:overlap_len]:
                repetition_penalty += 3
        
        repetition_score = max(0, 10 - repetition_penalty)
        
        # ---- 6. Capitalization and punctuation correctness ----
        cap_punct_score = 10.0
        
        # Check sentence capitalization
        cap_errors = 0
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isalpha() and not s_stripped[0].isupper():
                cap_errors += 1
        if num_sentences > 0:
            cap_error_rate = cap_errors / num_sentences
            cap_punct_score -= cap_error_rate * 3
        
        # Check for proper ending punctuation in the overall text
        last_char = text.rstrip()[-1] if text.rstrip() else ''
        if last_char not in '.!?"\')':
            cap_punct_score -= 1.0
        
        # Penalize excessive special characters / code-like content
        special_chars = sum(1 for c in text if c in '{}[]<>|\\@#$%^&*~`')
        special_ratio = special_chars / total_chars if total_chars > 0 else 0
        if special_ratio > 0.05:
            cap_punct_score -= min(4, special_ratio * 40)
        
        # Penalize no punctuation at all
        punct_count = sum(1 for c in text if c in '.,;:!?')
        if punct_count == 0 and num_words > 5:
            cap_punct_score -= 2
        
        cap_punct_score = max(0, min(10, cap_punct_score))
        
        # ---- 7. Response length adequacy ----
        # Very short responses tend to be low quality
        if num_words < 3:
            length_score = 1.0
        elif num_words < 8:
            length_score = 3.0
        elif num_words < 15:
            length_score = 5.0
        elif num_words < 30:
            length_score = 7.0
        elif num_words <= 200:
            length_score = 9.0
        elif num_words <= 400:
            length_score = 8.0
        else:
            length_score = 7.0
        
        # But if query asks for short response, don't penalize brevity as much
        query_lower = query.lower() if query else ""
        if any(kw in query_lower for kw in ['short', 'brief', 'concise', 'one word', 'identify', 'name']):
            if num_words >= 3:
                length_score = max(length_score, 6.0)
        
        # ---- 8. Coherence signal: ratio of common English words ----
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
            'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
            'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
            'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
            'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'has',
            'had', 'did', 'does', 'may', 'might', 'should', 'must', 'shall',
            'more', 'very', 'much', 'many', 'such', 'each', 'every', 'both',
            'few', 'own', 'same', 'still', 'found', 'where', 'here', 'while'
        }
        if words_lower:
            common_ratio = sum(1 for w in words_lower if w in common_words) / len(words_lower)
        else:
            common_ratio = 0
        
        # Natural English text typically has 40-60% common words
        if common_ratio < 0.15:
            coherence_score = 3
        elif common_ratio < 0.3:
            coherence_score = 5
        elif common_ratio <= 0.65:
            coherence_score = 8 + (common_ratio - 0.3) * 5.7
        else:
            coherence_score = max(5, 10 - (common_ratio - 0.65) * 10)
        coherence_score = max(0, min(10, coherence_score))
        
        # ---- Combine scores with weights ----
        final_score = (
            fog_score * 0.12 +
            entropy_score * 0.12 +
            variety_score * 0.10 +
            hapax_score * 0.08 +
            repetition_score * 0.18 +
            cap_punct_score * 0.15 +
            length_score * 0.15 +
            coherence_score * 0.10
        )
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a neutral score based on response length
        try:
            words = response.split() if response else []
            if len(words) < 3:
                return 1.0
            elif len(words) < 10:
                return 3.0
            else:
                return 5.0
        except Exception:
            return 2.0