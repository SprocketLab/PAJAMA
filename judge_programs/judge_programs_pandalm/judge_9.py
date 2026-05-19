def judging_function(query, response):
    """
    Evaluate language quality and readability of an LLM response.
    Uses a combination of:
    - Flesch-like readability scoring
    - Type-token ratio (vocabulary richness)
    - Sentence variety (length variance)
    - Grammar/punctuation heuristics
    - Repetition penalty
    - Appropriate length assessment
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        import re
        import math
        import collections
        import string
        
        # ============================================================
        # 1. Basic tokenization
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r"[a-zA-Z'-]+", response)
        num_words = len(words)
        
        if num_words == 0:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        # ============================================================
        # 2. Syllable counting for Flesch-like measure
        # ============================================================
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing 'e'
            if word.endswith('e') and not word.endswith('le'):
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
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / num_words
        avg_words_per_sentence = num_words / num_sentences
        
        # Flesch Reading Ease (modified to 0-100 scale, higher = more readable)
        flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        flesch_score = max(0, min(100, flesch))
        # Normalize to 0-10; sweet spot around 50-70 (not too simple, not too complex)
        # Penalize extremes
        if flesch_score > 80:
            flesch_component = 6.0 + (flesch_score - 80) * 0.02  # slightly less for very simple
        elif flesch_score >= 30:
            flesch_component = 4.0 + (flesch_score - 30) * 0.08  # 4-8 range for good readability
        else:
            flesch_component = max(1.0, flesch_score * 0.133)  # penalize very hard text
        flesch_component = min(flesch_component, 8.0)
        
        # ============================================================
        # 3. Type-Token Ratio (vocabulary richness)
        # ============================================================
        unique_words = set(words_lower)
        num_unique = len(unique_words)
        
        # Use root TTR to handle length variation
        if num_words > 0:
            root_ttr = num_unique / math.sqrt(num_words)
        else:
            root_ttr = 0
        
        # Normalize: typical root_ttr ranges from ~2 (repetitive) to ~8+ (rich)
        ttr_score = min(10.0, root_ttr * 1.2)
        
        # ============================================================
        # 4. Sentence length variety
        # ============================================================
        sent_lengths = [len(re.findall(r"[a-zA-Z'-]+", s)) for s in sentences]
        sent_lengths = [l for l in sent_lengths if l > 0]
        
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Some variety is good (cv 0.2-0.6), too much or too little is bad
            if cv < 0.05:
                variety_score = 3.0
            elif cv < 0.15:
                variety_score = 5.0
            elif cv < 0.5:
                variety_score = 7.0 + (cv - 0.15) * 5.0  # up to ~8.75
            elif cv < 0.8:
                variety_score = 7.0
            else:
                variety_score = 5.0  # too erratic
            variety_score = min(variety_score, 9.0)
        else:
            # Single sentence - neutral
            variety_score = 5.0
        
        # ============================================================
        # 5. Repetition penalty
        # ============================================================
        # Check for repeated words
        word_counts = collections.Counter(words_lower)
        # Exclude common stop words from repetition check
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                      'than', 'too', 'very', 'just', 'that', 'this', 'these', 'those',
                      'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your',
                      'he', 'she', 'his', 'her', 'i', 'me', 'my', 'also', 'which',
                      'who', 'whom', 'what', 'when', 'where', 'how', 'if', 'then',
                      'about', 'up', 'out', 'down', 'over', 'under'}
        
        content_words = {w: c for w, c in word_counts.items() if w not in stop_words and len(w) > 2}
        
        # Detect excessive repetition
        rep_penalty = 0.0
        if content_words:
            max_repeat = max(content_words.values())
            total_content = sum(content_words.values())
            if total_content > 0:
                # Ratio of most repeated content word to total content words
                max_ratio = max_repeat / total_content
                if max_ratio > 0.5 and max_repeat > 3:
                    rep_penalty += 3.0
                elif max_ratio > 0.3 and max_repeat > 3:
                    rep_penalty += 1.5
                elif max_repeat > 5:
                    rep_penalty += 1.0
        
        # Check for repeated phrases (bigrams and trigrams)
        if num_words >= 4:
            bigrams = [tuple(words_lower[i:i+2]) for i in range(len(words_lower)-1)]
            bigram_counts = collections.Counter(bigrams)
            max_bigram_rep = max(bigram_counts.values()) if bigram_counts else 1
            if max_bigram_rep > 3:
                rep_penalty += min(3.0, (max_bigram_rep - 3) * 0.5)
        
        if num_words >= 6:
            trigrams = [tuple(words_lower[i:i+3]) for i in range(len(words_lower)-2)]
            trigram_counts = collections.Counter(trigrams)
            max_trigram_rep = max(trigram_counts.values()) if trigram_counts else 1
            if max_trigram_rep > 2:
                rep_penalty += min(4.0, (max_trigram_rep - 2) * 1.0)
        
        rep_penalty = min(rep_penalty, 8.0)
        
        # ============================================================
        # 6. Punctuation and grammar heuristics
        # ============================================================
        punct_score = 5.0
        
        # Check for proper capitalization at sentence start
        raw_sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        cap_count = sum(1 for s in raw_sentences if s and s[0].isupper())
        if raw_sentences:
            cap_ratio = cap_count / len(raw_sentences)
            punct_score += cap_ratio * 2.0  # up to +2
        
        # Check for ending punctuation
        if response.rstrip()[-1] in '.!?':
            punct_score += 1.0
        
        # Check for comma usage (indicates complex sentences)
        comma_count = response.count(',')
        if num_sentences > 0:
            commas_per_sent = comma_count / num_sentences
            if 0.5 <= commas_per_sent <= 3.0:
                punct_score += 1.0
        
        punct_score = min(punct_score, 9.0)
        
        # ============================================================
        # 7. Length appropriateness
        # ============================================================
        # Responses that are too short or absurdly long get penalized
        length_score = 5.0
        
        if num_words < 5:
            length_score = 2.0
        elif num_words < 10:
            length_score = 4.0
        elif num_words < 20:
            length_score = 6.0
        elif num_words <= 100:
            length_score = 8.0
        elif num_words <= 200:
            length_score = 7.0
        elif num_words <= 500:
            length_score = 6.0
        else:
            length_score = 5.0
        
        # Bonus for multi-sentence responses (shows structure)
        if num_sentences >= 2:
            length_score += 1.0
        if num_sentences >= 3:
            length_score += 0.5
        
        length_score = min(length_score, 10.0)
        
        # ============================================================
        # 8. Structural markers (lists, transitions, etc.)
        # ============================================================
        structure_bonus = 0.0
        
        # Transition words
        transitions = ['however', 'moreover', 'furthermore', 'additionally', 'therefore',
                       'consequently', 'nevertheless', 'meanwhile', 'alternatively',
                       'specifically', 'for example', 'for instance', 'in addition',
                       'on the other hand', 'in contrast', 'as a result', 'in conclusion',
                       'first', 'second', 'third', 'finally', 'also', 'while', 'although',
                       'whereas', 'despite', 'instead', 'thus', 'hence', 'accordingly']
        
        response_lower = response.lower()
        transition_count = sum(1 for t in transitions if t in response_lower)
        structure_bonus += min(2.0, transition_count * 0.5)
        
        # Check for list-like structure
        if re.search(r'\d+[.)]\s', response):
            structure_bonus += 0.5
        if re.search(r'[-•]\s', response):
            structure_bonus += 0.5
        
        structure_bonus = min(structure_bonus, 2.5)
        
        # ============================================================
        # 9. Detect garbage/nonsense
        # ============================================================
        garbage_penalty = 0.0
        
        # Very long words that are likely gibberish
        long_word_count = sum(1 for w in words if len(w) > 20)
        if long_word_count > 0:
            garbage_penalty += long_word_count * 0.5
        
        # Check if response is mostly non-alphabetic
        alpha_chars = sum(1 for c in response if c.isalpha())
        total_chars = len(response)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.4:
                garbage_penalty += 3.0
        
        # Check for excessive special characters or markup
        special_pattern_count = len(re.findall(r'[!]{3,}|[.]{4,}|[#]{3,}', response))
        garbage_penalty += special_pattern_count * 0.5
        
        # Truncation detection (ends mid-word or mid-sentence without punctuation)
        truncation_penalty = 0.0
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?"\')':
            # Might be truncated
            last_word = words[-1] if words else ""
            if len(last_word) < 3 and stripped[-1].isalpha():
                truncation_penalty = 1.5
            elif stripped[-1].isalpha():
                truncation_penalty = 0.5
        
        garbage_penalty = min(garbage_penalty, 6.0)
        
        # ============================================================
        # 10. Combine all components
        # ============================================================
        # Weights emphasizing readability and language quality
        score = (
            flesch_component * 0.20 +      # Readability
            ttr_score * 0.20 +              # Vocabulary richness
            variety_score * 0.10 +          # Sentence variety
            punct_score * 0.15 +            # Grammar/punctuation
            length_score * 0.20 +           # Appropriate length
            structure_bonus * 0.15          # Structural quality (max ~2.5, weighted)
        )
        
        # Apply penalties
        score -= rep_penalty * 0.8
        score -= garbage_penalty * 0.7
        score -= truncation_penalty
        
        # Clamp to [0, 10]
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
    
    except Exception:
        return 2.0