def judging_function(query, response):
    """
    Evaluate language quality and readability of an LLM response.
    
    Uses a combination of:
    - Flesch-like readability scoring
    - Vocabulary richness (type-token ratio)
    - Sentence variety (length variance)
    - Grammar/punctuation heuristics
    - Structural quality indicators
    
    Returns a score where HIGHER = BETTER quality.
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # ---- Helper functions ----
        
        def count_syllables(word):
            """Estimate syllable count for a word."""
            word = word.lower().strip()
            if not word:
                return 1
            # Remove trailing 'e' (silent e)
            if len(word) > 2 and word.endswith('e'):
                word = word[:-1]
            vowels = 'aeiouoy'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(count, 1)
        
        def get_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'[.!?]+', text)
            sents = [s.strip() for s in sents if s.strip()]
            return sents
        
        def get_words(text):
            """Extract words from text."""
            words = re.findall(r"[a-zA-Z']+", text)
            return [w for w in words if len(w) > 0]
        
        # ---- Extract basic components ----
        
        sentences = get_sentences(response)
        words = get_words(response)
        
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        num_chars = len(response)
        
        # ---- 1. Flesch-like Readability (adapted) ----
        # Target: moderately readable (not too simple, not too complex)
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / num_words
        avg_words_per_sentence = num_words / num_sentences
        
        # Flesch Reading Ease (original scale ~0-100, higher = easier)
        flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        # Clamp to 0-100
        flesch = max(0, min(100, flesch))
        
        # We want moderate readability (50-70 is ideal for informative text)
        # Score peaks around 55-65
        if flesch < 20:
            readability_score = 2.0
        elif flesch < 35:
            readability_score = 4.0
        elif flesch < 50:
            readability_score = 6.5
        elif flesch < 70:
            readability_score = 8.5
        elif flesch < 85:
            readability_score = 7.0
        else:
            readability_score = 5.0  # Too simple
        
        # ---- 2. Vocabulary Richness (Type-Token Ratio) ----
        
        words_lower = [w.lower() for w in words]
        unique_words = set(words_lower)
        
        # Standard TTR
        ttr = len(unique_words) / num_words if num_words > 0 else 0
        
        # For longer texts, TTR naturally decreases, so use root TTR (Guiraud's index)
        guiraud = len(unique_words) / math.sqrt(num_words) if num_words > 0 else 0
        
        # Score TTR component (higher diversity = better, up to a point)
        # Guiraud index typically 4-12 for good text
        if guiraud < 3:
            vocab_score = 3.0
        elif guiraud < 5:
            vocab_score = 5.0
        elif guiraud < 8:
            vocab_score = 8.0
        elif guiraud < 12:
            vocab_score = 9.0
        else:
            vocab_score = 7.5  # Might be too scattered
        
        # ---- 3. Sentence Variety ----
        
        sent_lengths = [len(get_words(s)) for s in sentences]
        
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / mean_len if mean_len > 0 else 0
            
            # Good variety: CV between 0.3 and 0.7
            if cv < 0.1:
                variety_score = 4.0  # Too monotonous
            elif cv < 0.3:
                variety_score = 6.5
            elif cv < 0.6:
                variety_score = 8.5
            elif cv < 0.9:
                variety_score = 7.0
            else:
                variety_score = 5.0  # Too erratic
        else:
            variety_score = 3.0  # Only one sentence
        
        # ---- 4. Grammar & Punctuation Heuristics ----
        
        grammar_score = 7.0  # Start with decent baseline
        
        # Check for proper capitalization at sentence starts
        cap_count = 0
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isupper():
                cap_count += 1
        cap_ratio = cap_count / num_sentences
        grammar_score += (cap_ratio - 0.5) * 2  # Bonus for good capitalization
        
        # Check punctuation density (should have reasonable punctuation)
        punct_chars = sum(1 for c in response if c in '.,;:!?')
        punct_ratio = punct_chars / num_chars if num_chars > 0 else 0
        # Good range: 0.02 - 0.06
        if 0.02 <= punct_ratio <= 0.06:
            grammar_score += 1.0
        elif punct_ratio < 0.01:
            grammar_score -= 1.5  # Almost no punctuation
        
        # Check for common issues
        # Double spaces
        double_spaces = len(re.findall(r'  +', response))
        if double_spaces > 2:
            grammar_score -= 0.5
        
        # Missing space after punctuation
        missing_space = len(re.findall(r'[.!?,;:][A-Za-z]', response))
        if missing_space > 1:
            grammar_score -= 0.5 * min(missing_space, 3)
        
        # Repeated words (stuttering)
        repeated = len(re.findall(r'\b(\w+)\s+\1\b', response, re.IGNORECASE))
        if repeated > 0:
            grammar_score -= 0.5 * min(repeated, 3)
        
        grammar_score = max(2.0, min(10.0, grammar_score))
        
        # ---- 5. Structural Quality ----
        
        structure_score = 5.0
        
        # Paragraph breaks indicate good structure
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if num_paragraphs >= 2:
            structure_score += 1.5
        if num_paragraphs >= 3:
            structure_score += 0.5
        
        # Numbered or bulleted lists (structured advice)
        has_list = bool(re.search(r'^\s*[\d]+[.)]\s', response, re.MULTILINE))
        has_bullets = bool(re.search(r'^\s*[-•*]\s', response, re.MULTILINE))
        if has_list or has_bullets:
            structure_score += 1.0
        
        # Adequate length (not too short, not rambling)
        if num_words < 20:
            structure_score -= 2.0
        elif num_words < 40:
            structure_score -= 0.5
        elif num_words > 300:
            structure_score -= 0.5  # Might be too verbose
        
        # Appropriate sentence count
        if num_sentences >= 3:
            structure_score += 0.5
        if num_sentences >= 5:
            structure_score += 0.5
        
        structure_score = max(2.0, min(10.0, structure_score))
        
        # ---- 6. Tone & Engagement Indicators ----
        
        engagement_score = 5.0
        
        # Empathetic/professional language markers
        empathy_markers = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'s\b',
            r'\bcompletely\b', r'\babsolutely\b', r'\bgenuinely\b',
            r'\bunderstandable\b', r'\bperfectly\b', r'\bnatural\b',
            r'\bimportant\b', r'\bremember\b', r'\bconsider\b',
        ]
        empathy_count = sum(1 for pat in empathy_markers if re.search(pat, response, re.IGNORECASE))
        engagement_score += min(empathy_count * 0.4, 2.0)
        
        # Transition words (coherence)
        transitions = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin addition\b', r'\bfor instance\b',
            r'\bfor example\b', r'\btherefore\b', r'\bconsequently\b',
            r'\bnevertheless\b', r'\bmeanwhile\b', r'\binstead\b',
            r'\balso\b', r'\bfirst\b', r'\bnext\b', r'\bfinally\b',
            r'\bhere\b', r'\bnow\b', r'\bthen\b',
        ]
        trans_count = sum(1 for pat in transitions if re.search(pat, response, re.IGNORECASE))
        engagement_score += min(trans_count * 0.3, 2.0)
        
        # Avoid overly dismissive language
        dismissive = [
            r'\bjust\s+get\s+over\b', r'\bit\'s\s+not\s+a\s+big\s+deal\b',
            r'\bstop\s+complaining\b', r'\bget\s+yourself\s+together\b',
            r'\bmaybe\s+you\'re\s+just\b',
        ]
        dismiss_count = sum(1 for pat in dismissive if re.search(pat, response, re.IGNORECASE))
        engagement_score -= dismiss_count * 1.5
        
        engagement_score = max(2.0, min(10.0, engagement_score))
        
        # ---- 7. Word length distribution (sophistication) ----
        
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / num_words if num_words > 0 else 0
        
        # Good prose: avg word length 4-6
        if avg_word_length < 3.5:
            word_len_score = 4.0
        elif avg_word_length < 4.5:
            word_len_score = 6.5
        elif avg_word_length < 5.5:
            word_len_score = 8.0
        elif avg_word_length < 6.5:
            word_len_score = 7.5
        else:
            word_len_score = 5.0  # Too many long words
        
        # Proportion of "long" words (>6 chars) — sophistication indicator
        long_words = sum(1 for l in word_lengths if l > 6)
        long_ratio = long_words / num_words if num_words > 0 else 0
        if 0.15 <= long_ratio <= 0.35:
            word_len_score += 1.0
        elif long_ratio > 0.45:
            word_len_score -= 0.5
        
        word_len_score = max(2.0, min(10.0, word_len_score))
        
        # ---- Combine all scores with weights ----
        
        weights = {
            'readability': 0.18,
            'vocabulary': 0.15,
            'variety': 0.12,
            'grammar': 0.18,
            'structure': 0.15,
            'engagement': 0.12,
            'word_length': 0.10,
        }
        
        scores = {
            'readability': readability_score,
            'vocabulary': vocab_score,
            'variety': variety_score,
            'grammar': grammar_score,
            'structure': structure_score,
            'engagement': engagement_score,
            'word_length': word_len_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Scale to 0-10 range (scores are already roughly 2-10)
        # Normalize: map [3, 9.5] -> [1, 5] for the expected output range
        final_mapped = 1.0 + (final_score - 3.0) * (4.0 / 6.5)
        final_mapped = max(1.0, min(5.0, final_mapped))
        
        return round(final_mapped, 2)
    
    except Exception:
        # Never crash — return a neutral score
        return 3.0