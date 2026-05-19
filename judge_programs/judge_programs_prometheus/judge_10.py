def judging_function(query, response):
    """
    Evaluates language quality and readability using a different approach than Flesch/syllable-based methods.
    
    This variant focuses on:
    - Gunning Fog-inspired complexity analysis
    - Sentence structure variety (length variance)
    - Punctuation sophistication
    - Transition/cohesion word usage
    - Type-token ratio with hapax legomena ratio
    - Grammar heuristics (repeated words, capitalization errors)
    - Paragraph structure
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # === Tokenization ===
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        words = re.findall(r"[a-zA-Z']+", response)
        words_lower = [w.lower() for w in words]
        
        if len(words) < 3:
            return 0.5
        
        num_sentences = max(len(sentences), 1)
        num_words = len(words)
        
        # === 1. Sentence Length Variety (std dev of sentence lengths) ===
        sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
        sent_lengths = [sl for sl in sent_lengths if sl > 0]
        
        if len(sent_lengths) >= 2:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance_sl)
            # Good variety: std between 3-12 words
            variety_score = min(std_sl / 8.0, 1.0) * 10
            # Penalize if all sentences are same length
            if std_sl < 1.0:
                variety_score *= 0.3
        else:
            variety_score = 3.0  # Only one sentence, moderate penalty
        
        # === 2. Average sentence length score (ideal: 12-22 words) ===
        avg_sent_len = num_words / num_sentences
        if 12 <= avg_sent_len <= 22:
            avg_len_score = 10.0
        elif 8 <= avg_sent_len < 12 or 22 < avg_sent_len <= 30:
            avg_len_score = 7.0
        elif 5 <= avg_sent_len < 8 or 30 < avg_sent_len <= 40:
            avg_len_score = 4.0
        else:
            avg_len_score = 2.0
        
        # === 3. Complex word ratio (Gunning Fog inspired - words with 3+ syllables) ===
        def count_syllables_approx(word):
            word = word.lower()
            if len(word) <= 2:
                return 1
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e') and count > 1:
                count -= 1
            if word.endswith('le') and len(word) > 3 and word[-3] not in vowels:
                count += 1
            return max(count, 1)
        
        complex_words = sum(1 for w in words_lower if count_syllables_approx(w) >= 3)
        complex_ratio = complex_words / num_words
        # Ideal: 10-25% complex words
        if 0.08 <= complex_ratio <= 0.25:
            complexity_score = 10.0
        elif 0.04 <= complex_ratio < 0.08 or 0.25 < complex_ratio <= 0.35:
            complexity_score = 7.0
        elif complex_ratio < 0.04:
            complexity_score = 4.0
        else:
            complexity_score = 3.0
        
        # === 4. Vocabulary richness: Hapax Legomena Ratio ===
        freq = collections.Counter(words_lower)
        hapax = sum(1 for w, c in freq.items() if c == 1)
        hapax_ratio = hapax / len(freq) if freq else 0
        # Higher hapax ratio = richer vocabulary
        vocab_richness_score = min(hapax_ratio * 12, 10.0)
        
        # Also: Yule's K measure approximation (lower = richer)
        N = num_words
        freq_spectrum = collections.Counter(freq.values())
        M2 = sum(i * i * freq_spectrum[i] for i in freq_spectrum)
        if N > 0:
            yules_k = 10000 * (M2 - N) / (N * N) if N > 1 else 100
        else:
            yules_k = 100
        # Lower Yule's K is better; typical range 50-200
        if yules_k < 80:
            yule_score = 10.0
        elif yules_k < 120:
            yule_score = 8.0
        elif yules_k < 180:
            yule_score = 6.0
        else:
            yule_score = 3.0
        
        # === 5. Transition/Cohesion words ===
        transition_words = {
            'however', 'moreover', 'furthermore', 'additionally', 'therefore',
            'consequently', 'nevertheless', 'meanwhile', 'alternatively', 'specifically',
            'similarly', 'likewise', 'conversely', 'instead', 'accordingly',
            'thus', 'hence', 'besides', 'indeed', 'certainly',
            'notably', 'importantly', 'significantly', 'ultimately', 'essentially',
            'firstly', 'secondly', 'thirdly', 'finally', 'overall',
            'although', 'despite', 'whereas', 'while', 'since',
            'because', 'unless', 'provided', 'assuming', 'considering',
            'remember', 'imagine', 'consider', 'notice', 'importantly'
        }
        transition_count = sum(1 for w in words_lower if w in transition_words)
        transition_density = transition_count / num_sentences
        # Ideal: 0.3-1.5 transitions per sentence
        if 0.2 <= transition_density <= 1.5:
            transition_score = min(transition_density * 7, 10.0)
        elif transition_density > 1.5:
            transition_score = 6.0
        else:
            transition_score = 2.0
        
        # === 6. Punctuation sophistication ===
        commas = response.count(',')
        semicolons = response.count(';')
        colons = response.count(':')
        dashes = response.count('—') + response.count('–') + response.count(' - ')
        parens = response.count('(')
        
        punct_variety = sum(1 for p in [commas, semicolons, colons, dashes, parens] if p > 0)
        comma_density = commas / num_sentences
        
        punct_score = min(punct_variety * 2.0, 6.0)
        if 0.5 <= comma_density <= 3.0:
            punct_score += 4.0
        elif comma_density > 3.0:
            punct_score += 2.0
        else:
            punct_score += 1.0
        punct_score = min(punct_score, 10.0)
        
        # === 7. Grammar heuristics ===
        grammar_score = 10.0
        
        # Check for repeated adjacent words
        for i in range(len(words_lower) - 1):
            if words_lower[i] == words_lower[i + 1] and words_lower[i] not in {'very', 'really', 'had', 'that'}:
                grammar_score -= 0.5
        
        # Check sentences start with capital
        cap_starts = 0
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isupper():
                cap_starts += 1
        cap_ratio = cap_starts / num_sentences if num_sentences > 0 else 0
        grammar_score *= max(cap_ratio, 0.5)
        
        # Penalize very short response
        if num_words < 20:
            grammar_score *= 0.7
        
        grammar_score = max(grammar_score, 0.0)
        grammar_score = min(grammar_score, 10.0)
        
        # === 8. Paragraph and structure awareness ===
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Check for numbered lists or bullet points
        has_list = bool(re.search(r'^\s*(\d+[.):]|\-|\*|•)', response, re.MULTILINE))
        
        structure_score = 5.0
        if num_paragraphs >= 2:
            structure_score += 2.0
        if num_paragraphs >= 3:
            structure_score += 1.5
        if has_list:
            structure_score += 1.5
        structure_score = min(structure_score, 10.0)
        
        # === 9. Word length distribution (character-level diversity) ===
        word_lengths = [len(w) for w in words]
        avg_word_len = sum(word_lengths) / len(word_lengths)
        # Ideal average word length: 4.5-6.5 characters
        if 4.5 <= avg_word_len <= 6.5:
            word_len_score = 10.0
        elif 3.5 <= avg_word_len < 4.5 or 6.5 < avg_word_len <= 7.5:
            word_len_score = 7.0
        else:
            word_len_score = 4.0
        
        # Word length variance
        if len(word_lengths) >= 2:
            wl_mean = avg_word_len
            wl_var = sum((x - wl_mean) ** 2 for x in word_lengths) / len(word_lengths)
            wl_std = math.sqrt(wl_var)
            # Good variety: std around 2-4
            if 2.0 <= wl_std <= 4.0:
                word_len_score = min(word_len_score + 2, 10.0)
        
        # === 10. Empathy/engagement markers (for conversational quality) ===
        empathy_words = {
            'understand', 'sorry', 'hear', 'feel', 'appreciate', 'acknowledge',
            'completely', 'absolutely', 'genuinely', 'truly', 'certainly',
            'please', 'thank', 'welcome', 'glad', 'happy',
            'understandable', 'natural', 'valid', 'okay', 'fine'
        }
        empathy_count = sum(1 for w in words_lower if w in empathy_words)
        empathy_density = empathy_count / num_words
        # Bonus for engagement, but don't over-reward
        engagement_bonus = min(empathy_density * 50, 3.0)
        
        # === 11. Response length adequacy ===
        length_score = 5.0
        if 50 <= num_words <= 300:
            length_score = 10.0
        elif 30 <= num_words < 50 or 300 < num_words <= 500:
            length_score = 7.0
        elif 15 <= num_words < 30:
            length_score = 5.0
        else:
            length_score = 3.0
        
        # === Combine all scores with weights ===
        weights = {
            'variety': 0.10,
            'avg_len': 0.08,
            'complexity': 0.10,
            'vocab_richness': 0.08,
            'yule': 0.07,
            'transition': 0.12,
            'punctuation': 0.08,
            'grammar': 0.10,
            'structure': 0.10,
            'word_len': 0.05,
            'length': 0.07,
            'engagement': 0.05,
        }
        
        raw_score = (
            weights['variety'] * variety_score +
            weights['avg_len'] * avg_len_score +
            weights['complexity'] * complexity_score +
            weights['vocab_richness'] * vocab_richness_score +
            weights['yule'] * yule_score +
            weights['transition'] * transition_score +
            weights['punctuation'] * punct_score +
            weights['grammar'] * grammar_score +
            weights['structure'] * structure_score +
            weights['word_len'] * word_len_score +
            weights['length'] * length_score +
            engagement_bonus
        )
        
        # Normalize to 1-5 scale
        # raw_score theoretical range: ~0 to ~13
        final_score = max(1.0, min(5.0, raw_score * 0.45 + 0.5))
        
        return round(final_score, 2)
    
    except Exception:
        return 3.0