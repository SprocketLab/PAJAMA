def judging_function(query, response):
    """
    Evaluates language quality and readability using a substantially different approach:
    - Gunning Fog Index (instead of Flesch)
    - Coleman-Liau Index
    - Sentence structure variety (std dev of sentence lengths)
    - Punctuation sophistication (use of semicolons, colons, dashes, parentheses)
    - Repetition penalty (duplicate n-grams and repeated words)
    - Coherence via sentence-start diversity
    - Word sophistication via average word length distribution entropy
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        # Basic tokenization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        words = re.findall(r"[a-zA-Z']+", text)
        if len(words) < 3:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # --- Helper: count syllables ---
        def count_syllables(word):
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
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)
        
        syllable_counts = [count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        
        # --- 1. Gunning Fog Index ---
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        avg_sentence_len = num_words / num_sentences
        fog_index = 0.4 * (avg_sentence_len + 100 * (complex_words / max(num_words, 1)))
        
        # Ideal fog: 8-14 (clear but not too simple)
        if fog_index < 4:
            fog_score = fog_index / 4.0 * 5  # too simple
        elif fog_index <= 14:
            fog_score = 10.0
        elif fog_index <= 20:
            fog_score = 10.0 - (fog_index - 14) / 6.0 * 5
        else:
            fog_score = max(0, 5.0 - (fog_index - 20) / 10.0 * 5)
        
        # --- 2. Coleman-Liau Index ---
        num_chars = sum(len(w) for w in words)
        L = (num_chars / num_words) * 100  # avg letters per 100 words
        S = (num_sentences / num_words) * 100  # avg sentences per 100 words
        cli = 0.0588 * L - 0.296 * S - 15.8
        
        # Ideal CLI: 7-12
        if cli < 3:
            cli_score = max(0, cli / 3.0 * 4)
        elif cli <= 12:
            cli_score = 10.0
        elif cli <= 18:
            cli_score = 10.0 - (cli - 12) / 6.0 * 5
        else:
            cli_score = max(0, 5.0 - (cli - 18) / 10.0 * 5)
        
        # --- 3. Sentence length variety (std dev) ---
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 2:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance)
            # Some variety is good; too much is chaotic
            # Ideal std_dev: 3-10
            if std_sl < 1:
                variety_score = 3.0  # monotonous
            elif std_sl <= 10:
                variety_score = 3.0 + (std_sl - 1) / 9.0 * 7.0
            elif std_sl <= 20:
                variety_score = 10.0 - (std_sl - 10) / 10.0 * 5
            else:
                variety_score = 5.0
        else:
            variety_score = 4.0  # single sentence, neutral
        
        # --- 4. Punctuation sophistication ---
        # Count advanced punctuation marks
        semicolons = text.count(';')
        colons = text.count(':')
        dashes = text.count('—') + text.count('–') + text.count(' - ')
        parens = text.count('(') + text.count(')')
        commas = text.count(',')
        
        advanced_punct = semicolons + colons + dashes + parens // 2
        # Commas per sentence (good writing uses commas well)
        commas_per_sent = commas / num_sentences
        
        punct_score = 4.0  # baseline
        punct_score += min(3.0, advanced_punct * 1.0)  # bonus for sophisticated punctuation
        if 0.5 <= commas_per_sent <= 3.0:
            punct_score += 2.0
        elif commas_per_sent > 0:
            punct_score += 1.0
        punct_score = min(10.0, punct_score)
        
        # --- 5. Repetition penalty ---
        # Word-level repetition
        word_freq = Counter(words_lower)
        # Filter out common stop words for repetition check
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                      'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
                      'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
                      'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                      'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'he',
                      'him', 'his', 'she', 'her', 'i', 'me', 'my', 'you', 'your'}
        
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 2]
        content_freq = Counter(content_words)
        
        if content_words:
            max_content_freq = max(content_freq.values())
            # Ratio of most repeated content word to total content words
            repetition_ratio = max_content_freq / len(content_words)
        else:
            repetition_ratio = 0
        
        # Trigram repetition
        trigrams = [tuple(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
        trigram_freq = Counter(trigrams)
        if trigrams:
            repeated_trigrams = sum(1 for v in trigram_freq.values() if v > 1)
            trigram_rep_ratio = repeated_trigrams / len(trigram_freq) if trigram_freq else 0
        else:
            trigram_rep_ratio = 0
        
        # Score: less repetition = better
        rep_penalty = repetition_ratio * 15 + trigram_rep_ratio * 10
        repetition_score = max(0, 10.0 - rep_penalty)
        
        # --- 6. Sentence-start diversity ---
        sentence_starts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sentence_starts.append(sw[0].lower())
        
        if len(sentence_starts) >= 2:
            unique_starts = len(set(sentence_starts))
            start_diversity = unique_starts / len(sentence_starts)
            start_score = start_diversity * 10.0
        else:
            start_score = 5.0
        
        # --- 7. Word length distribution entropy ---
        # Measures vocabulary sophistication through distribution of word lengths
        word_lengths = [len(w) for w in words]
        length_freq = Counter(word_lengths)
        total = sum(length_freq.values())
        entropy = 0
        for count in length_freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Ideal entropy: 2.5-3.5 (good mix of short and long words)
        if entropy < 1.0:
            entropy_score = entropy / 1.0 * 3
        elif entropy <= 3.5:
            entropy_score = 3.0 + (entropy - 1.0) / 2.5 * 7.0
        else:
            entropy_score = 10.0
        entropy_score = min(10.0, entropy_score)
        
        # --- 8. Response length adequacy ---
        # Very short responses tend to be lower quality
        if num_words < 5:
            length_score = 2.0
        elif num_words < 10:
            length_score = 5.0
        elif num_words < 20:
            length_score = 7.0
        elif num_words <= 200:
            length_score = 10.0
        elif num_words <= 500:
            length_score = 9.0
        else:
            length_score = 8.0
        
        # --- 9. Grammar heuristics ---
        grammar_score = 10.0
        
        # Check for sentence capitalization
        cap_violations = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isalpha() and not s[0].isupper():
                cap_violations += 1
        if num_sentences > 0:
            grammar_score -= (cap_violations / num_sentences) * 4
        
        # Check for double spaces
        double_spaces = len(re.findall(r'  +', text))
        grammar_score -= min(2.0, double_spaces * 0.5)
        
        # Check text ends with proper punctuation
        if text and text[-1] not in '.!?:"\')':
            grammar_score -= 1.0
        
        grammar_score = max(0, grammar_score)
        
        # --- Combine scores with weights ---
        weights = {
            'fog': 0.10,
            'cli': 0.10,
            'variety': 0.10,
            'punct': 0.08,
            'repetition': 0.18,
            'start_div': 0.10,
            'entropy': 0.10,
            'length': 0.12,
            'grammar': 0.12,
        }
        
        scores = {
            'fog': fog_score,
            'cli': cli_score,
            'variety': variety_score,
            'punct': punct_score,
            'repetition': repetition_score,
            'start_div': start_score,
            'entropy': entropy_score,
            'length': length_score,
            'grammar': grammar_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-100
        final_score = final_score * 10.0
        
        return round(max(0.0, min(100.0, final_score)), 2)
    
    except Exception:
        return 25.0