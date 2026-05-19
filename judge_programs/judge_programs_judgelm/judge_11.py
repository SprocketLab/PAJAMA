def judging_function(query, response):
    """
    Evaluate language quality and readability using a combination of:
    - Gunning Fog Index (different from Flesch used in variants 1&2)
    - Coleman-Liau Index
    - Type-token ratio with hapax legomena ratio
    - Punctuation diversity and correctness
    - Sentence structure variety (std dev of sentence lengths)
    - Repetition penalty (n-gram repetition detection)
    - Content density (ratio of content words to total words)
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        # Very short responses get penalized
        if len(text) < 3:
            return 0.5
        
        # === Tokenization ===
        # Split into sentences
        sentence_endings = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentence_endings if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Split into words
        words = re.findall(r"[a-zA-Z']+", text.lower())
        num_words = len(words)
        
        if num_words == 0:
            return 0.5
        
        # === Syllable counting function ===
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing e
            if word.endswith('e') and len(word) > 2:
                word = word[:-1]
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(count, 1)
        
        syllable_counts = [count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        
        # === Complex words (3+ syllables) for Gunning Fog ===
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        complex_word_pct = complex_words / num_words if num_words > 0 else 0
        
        avg_sentence_length = num_words / num_sentences
        
        # Gunning Fog Index: 0.4 * (ASL + 100 * complex_word_pct)
        gunning_fog = 0.4 * (avg_sentence_length + 100 * complex_word_pct)
        
        # === Coleman-Liau Index ===
        # Uses characters per word and sentences per word
        total_chars = sum(len(w) for w in words)
        avg_chars_per_100words = (total_chars / num_words) * 100 if num_words > 0 else 0
        avg_sents_per_100words = (num_sentences / num_words) * 100 if num_words > 0 else 0
        coleman_liau = 0.0588 * avg_chars_per_100words - 0.296 * avg_sents_per_100words - 15.8
        
        # === Readability score (combine Gunning Fog and Coleman-Liau) ===
        # Ideal readability: Gunning Fog between 8-14, Coleman-Liau between 8-14
        # Penalize extremes
        def readability_score_from_index(idx, ideal_low=7, ideal_high=14):
            if ideal_low <= idx <= ideal_high:
                return 1.0
            elif idx < ideal_low:
                return max(0.2, 1.0 - (ideal_low - idx) * 0.1)
            else:
                return max(0.2, 1.0 - (idx - ideal_high) * 0.05)
        
        fog_score = readability_score_from_index(gunning_fog, 7, 15)
        cl_score = readability_score_from_index(coleman_liau, 6, 14)
        readability_combined = 0.5 * fog_score + 0.5 * cl_score
        
        # === Vocabulary richness: Hapax legomena ratio ===
        word_freq = collections.Counter(words)
        unique_words = len(word_freq)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        
        # Type-token ratio
        ttr = unique_words / num_words if num_words > 0 else 0
        # Hapax ratio (proportion of words appearing only once)
        hapax_ratio = hapax / num_words if num_words > 0 else 0
        
        # For very short texts, TTR is naturally high, so adjust
        if num_words < 10:
            vocab_score = ttr * 0.5  # Penalize very short
        else:
            # Combine TTR and hapax ratio
            vocab_score = 0.6 * min(ttr / 0.7, 1.0) + 0.4 * min(hapax_ratio / 0.6, 1.0)
        
        # === Sentence variety (std dev of sentence lengths) ===
        if num_sentences > 1:
            sent_word_counts = []
            for s in sentences:
                s_words = re.findall(r"[a-zA-Z']+", s)
                sent_word_counts.append(len(s_words))
            
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance)
            
            # Some variety is good, but not too much (which indicates broken text)
            # Coefficient of variation
            cv = std_sl / mean_sl if mean_sl > 0 else 0
            if cv < 0.1:
                variety_score = 0.4  # Too uniform
            elif cv < 0.5:
                variety_score = 0.7 + 0.6 * (cv - 0.1) / 0.4  # Sweet spot
            elif cv < 1.0:
                variety_score = 1.0 - 0.3 * (cv - 0.5) / 0.5
            else:
                variety_score = 0.4  # Too chaotic
        else:
            variety_score = 0.3  # Single sentence gets low variety
        
        # === Punctuation diversity ===
        punct_chars = set()
        punct_count = 0
        for ch in text:
            if ch in '.,;:!?-–—()[]"\'':
                punct_chars.add(ch)
                punct_count += 1
        
        punct_diversity = min(len(punct_chars) / 5.0, 1.0)  # Normalize to ~5 different punct types
        punct_density = punct_count / num_words if num_words > 0 else 0
        
        # Good punct density is roughly 0.05-0.25
        if punct_density < 0.02:
            punct_density_score = 0.3
        elif punct_density < 0.05:
            punct_density_score = 0.6
        elif punct_density <= 0.3:
            punct_density_score = 1.0
        else:
            punct_density_score = max(0.3, 1.0 - (punct_density - 0.3) * 2)
        
        punctuation_score = 0.5 * punct_diversity + 0.5 * punct_density_score
        
        # === Repetition penalty (bigram and trigram repetition) ===
        if num_words >= 4:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
            
            bigram_freq = collections.Counter(bigrams)
            trigram_freq = collections.Counter(trigrams)
            
            # Fraction of repeated bigrams
            repeated_bigrams = sum(c - 1 for c in bigram_freq.values() if c > 1)
            bigram_rep_ratio = repeated_bigrams / len(bigrams) if bigrams else 0
            
            repeated_trigrams = sum(c - 1 for c in trigram_freq.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / len(trigrams) if trigrams else 0
            
            # High repetition = bad
            repetition_penalty = min(1.0, bigram_rep_ratio * 1.5 + trigram_rep_ratio * 3.0)
        else:
            repetition_penalty = 0.0
        
        repetition_score = 1.0 - repetition_penalty
        
        # === Content density: ratio of non-stopwords ===
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'about', 'up', 'out', 'then', 'there', 'here', 'also'
        }
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        content_ratio = len(content_words) / num_words if num_words > 0 else 0
        content_density_score = min(content_ratio / 0.5, 1.0)  # Normalize around 50%
        
        # === Detect garbage/code/HTML ===
        # Check for excessive non-alphabetic characters
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        total_chars_all = len(text)
        alpha_ratio = alpha_chars / total_chars_all if total_chars_all > 0 else 0
        
        # Check for HTML tags
        html_tags = len(re.findall(r'<[^>]+>', text))
        html_penalty = min(html_tags * 0.1, 0.5)
        
        # Check for code-like patterns
        code_patterns = len(re.findall(r'(def |import |class |return |if |for |while |print\(|\.py|#include)', text))
        code_penalty = min(code_patterns * 0.08, 0.5)
        
        # Format quality: starts with capital or reasonable character
        starts_well = 1.0 if text[0].isupper() or text[0] == '"' or text[0] == "'" else 0.5
        
        # Check for proper ending
        ends_well = 1.0 if text[-1] in '.!?")\'' else 0.6
        
        formatting_score = (starts_well * 0.4 + ends_well * 0.4 + (1.0 - html_penalty) * 0.1 + (1.0 - code_penalty) * 0.1)
        
        # === Length appropriateness ===
        # Very short responses are usually low quality; moderate length is ideal
        if num_words < 3:
            length_score = 0.15
        elif num_words < 8:
            length_score = 0.35
        elif num_words < 15:
            length_score = 0.55
        elif num_words < 30:
            length_score = 0.75
        elif num_words <= 200:
            length_score = 1.0
        elif num_words <= 400:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # === Capitalization quality ===
        # Check that sentences start with capitals
        cap_correct = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isupper():
                cap_correct += 1
        cap_score = cap_correct / num_sentences if num_sentences > 0 else 0.5
        
        # === Final weighted combination ===
        final_score = (
            readability_combined * 1.5 +      # Readability indices
            vocab_score * 1.5 +                # Vocabulary richness
            variety_score * 1.0 +              # Sentence variety
            punctuation_score * 0.8 +          # Punctuation quality
            repetition_score * 2.0 +           # Repetition penalty (important)
            content_density_score * 0.8 +      # Content density
            formatting_score * 1.0 +           # Formatting quality
            length_score * 1.2 +               # Length appropriateness
            cap_score * 0.7                    # Capitalization
        )
        
        # Normalize: max possible ≈ 1.5+1.5+1.0+0.8+2.0+0.8+1.0+1.2+0.7 = 10.5
        max_possible = 10.5
        normalized = (final_score / max_possible) * 10.0
        
        # Apply alpha ratio penalty for garbage text
        if alpha_ratio < 0.5:
            normalized *= alpha_ratio * 1.5
        
        # Clamp to [0, 10]
        normalized = max(0.0, min(10.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        return 2.0