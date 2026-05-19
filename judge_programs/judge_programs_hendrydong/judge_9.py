def judging_function(query, response):
    """
    Evaluate language quality and readability of an LLM response.
    
    This variant focuses on:
    - Flesch-Kincaid readability metrics
    - Type-token ratio (vocabulary richness)
    - Grammar/spelling heuristics
    - Sentence structure variety
    - Punctuation quality
    - Overall text sophistication
    
    Returns a score where HIGHER = BETTER quality.
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        # ============================================================
        # 1. SYLLABLE COUNTING
        # ============================================================
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing 'e' (silent e)
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
            return max(1, count)
        
        # ============================================================
        # 2. BASIC TEXT STATISTICS
        # ============================================================
        # Sentence splitting
        sentence_endings = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 2]
        num_sentences = max(1, len(sentences))
        
        # Word extraction
        words = re.findall(r"[a-zA-Z']+", response)
        num_words = len(words)
        if num_words < 3:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        # Character count (letters only)
        num_chars = sum(len(w) for w in words)
        
        # Syllable count
        total_syllables = sum(count_syllables(w) for w in words)
        
        # ============================================================
        # 3. FLESCH READING EASE (modified)
        # ============================================================
        avg_sentence_length = num_words / num_sentences
        avg_syllables_per_word = total_syllables / num_words
        
        # Standard Flesch Reading Ease
        flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        # Normalize to 0-10 range; ideal range is roughly 30-70 for informative text
        # Too high (>80) = too simple, too low (<20) = too complex
        # Best score around 40-65
        if flesch < 0:
            flesch_score = 2.0
        elif flesch > 100:
            flesch_score = 3.0
        elif 30 <= flesch <= 70:
            flesch_score = 10.0
        elif 20 <= flesch < 30 or 70 < flesch <= 80:
            flesch_score = 7.0
        elif 10 <= flesch < 20 or 80 < flesch <= 90:
            flesch_score = 5.0
        else:
            flesch_score = 3.0
        
        # ============================================================
        # 4. TYPE-TOKEN RATIO (vocabulary richness)
        # ============================================================
        unique_words = set(words_lower)
        num_unique = len(unique_words)
        
        # Use root TTR to handle length variation: TTR = unique / sqrt(2 * total)
        if num_words > 0:
            root_ttr = num_unique / math.sqrt(2 * num_words)
        else:
            root_ttr = 0
        
        # Root TTR typically ranges from ~3 to ~10 for good text
        ttr_score = min(10.0, max(0.0, root_ttr * 1.5))
        
        # ============================================================
        # 5. SENTENCE LENGTH VARIETY
        # ============================================================
        sentence_lengths = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            sentence_lengths.append(len(s_words))
        
        if len(sentence_lengths) > 1:
            mean_sl = sum(sentence_lengths) / len(sentence_lengths)
            variance_sl = sum((x - mean_sl) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_sl = math.sqrt(variance_sl)
            # Coefficient of variation
            cv = std_sl / max(1, mean_sl)
            # Good variety: cv around 0.3-0.7
            if 0.2 <= cv <= 0.8:
                variety_score = 10.0
            elif 0.1 <= cv < 0.2 or 0.8 < cv <= 1.0:
                variety_score = 7.0
            elif cv < 0.1:
                variety_score = 4.0  # All sentences same length
            else:
                variety_score = 5.0
        else:
            # Only one sentence - penalize slightly but don't crush
            variety_score = 4.0
        
        # ============================================================
        # 6. AVERAGE WORD LENGTH (sophistication proxy)
        # ============================================================
        avg_word_length = num_chars / num_words
        # Good prose typically 4.5-6.0 average word length
        if 4.5 <= avg_word_length <= 6.0:
            wordlen_score = 10.0
        elif 4.0 <= avg_word_length < 4.5 or 6.0 < avg_word_length <= 6.5:
            wordlen_score = 8.0
        elif 3.5 <= avg_word_length < 4.0 or 6.5 < avg_word_length <= 7.0:
            wordlen_score = 6.0
        else:
            wordlen_score = 4.0
        
        # ============================================================
        # 7. PUNCTUATION QUALITY
        # ============================================================
        # Check for proper use of commas, semicolons, colons, etc.
        comma_count = response.count(',')
        semicolon_count = response.count(';')
        colon_count = response.count(':')
        dash_count = response.count('—') + response.count('--') + response.count(' - ')
        paren_count = response.count('(')
        
        # Punctuation diversity
        punct_types_used = sum(1 for c in [comma_count, semicolon_count, colon_count, dash_count, paren_count] if c > 0)
        
        # Comma rate per sentence
        comma_rate = comma_count / num_sentences
        
        punct_score = 4.0
        if punct_types_used >= 3:
            punct_score = 10.0
        elif punct_types_used == 2:
            punct_score = 7.5
        elif punct_types_used == 1:
            punct_score = 5.5
        
        # Bonus for reasonable comma usage
        if 0.5 <= comma_rate <= 3.0:
            punct_score = min(10.0, punct_score + 1.0)
        
        # ============================================================
        # 8. SPELLING / TYPO HEURISTICS
        # ============================================================
        # Check for common error patterns
        error_count = 0
        
        # Double spaces
        error_count += len(re.findall(r'  +', response))
        
        # Missing space after punctuation
        error_count += len(re.findall(r'[.!?,;:][a-zA-Z]', response))
        
        # Repeated words (e.g., "the the")
        for i in range(len(words_lower) - 1):
            if words_lower[i] == words_lower[i + 1] and words_lower[i] not in {'had', 'that', 'very', 'really', 'so'}:
                error_count += 1
        
        # Unclosed parentheses/brackets
        open_parens = response.count('(') - response.count(')')
        open_brackets = response.count('[') - response.count(']')
        error_count += abs(open_parens) + abs(open_brackets)
        
        # Common misspelling patterns (double letters where shouldn't be, etc.)
        common_typos = [r'\bteh\b', r'\brecieve\b', r'\boccured\b', r'\bseperately\b',
                        r'\bdefin[ai]tly\b', r'\baccommodate\b', r'\boccassion\b',
                        r'\bcrossbow\b']  # not a typo but check misspellings
        misspell_patterns = [r'\bteh\b', r'\brecieve\b', r'\boccured\b', r'\bseperatel\b',
                            r'\bdefinately\b', r'\bwieled\b', r'\bwielded\b']
        
        for pattern in [r'\bteh\b', r'\brecieve\b', r'\boccured\b', r'\bdefinately\b',
                        r'\bseperately\b', r'\bcrosssbow\b', r'\bcrowssbow\b']:
            if re.search(pattern, response, re.IGNORECASE):
                error_count += 2
        
        # Error rate
        error_rate = error_count / max(1, num_words) * 100
        spelling_score = max(0, 10.0 - error_rate * 5)
        
        # ============================================================
        # 9. STRUCTURAL QUALITY
        # ============================================================
        structural_score = 5.0
        
        # Proper capitalization at sentence starts
        cap_count = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isupper():
                cap_count += 1
        cap_ratio = cap_count / max(1, num_sentences)
        structural_score += cap_ratio * 2.0
        
        # Has proper ending punctuation
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        if last_char in '.!?"\'':
            structural_score += 1.0
        
        # Use of markdown/formatting (lists, bold, italics)
        if re.search(r'\*\*.*?\*\*', response) or re.search(r'\*[^*]+\*', response):
            structural_score += 0.5
        if re.search(r'^\s*[-*]\s', response, re.MULTILINE):
            structural_score += 0.5
        
        structural_score = min(10.0, structural_score)
        
        # ============================================================
        # 10. RESPONSE LENGTH ADEQUACY
        # ============================================================
        # Longer, more substantive responses tend to be better
        # But not excessively so
        if num_words < 10:
            length_score = 2.0
        elif num_words < 20:
            length_score = 4.0
        elif num_words < 40:
            length_score = 6.0
        elif num_words < 80:
            length_score = 8.0
        elif num_words < 200:
            length_score = 10.0
        elif num_words < 500:
            length_score = 9.0
        else:
            length_score = 8.0
        
        # ============================================================
        # 11. DISCOURSE MARKERS / CONNECTIVES
        # ============================================================
        connectives = ['however', 'therefore', 'moreover', 'furthermore', 'additionally',
                       'nevertheless', 'consequently', 'meanwhile', 'similarly', 'likewise',
                       'in contrast', 'on the other hand', 'for example', 'for instance',
                       'in other words', 'that is', 'specifically', 'in particular',
                       'as a result', 'in addition', 'essentially', 'typically',
                       'generally', 'particularly', 'importantly', 'significantly',
                       'interestingly', 'notably', 'ultimately', 'basically']
        
        response_lower = response.lower()
        connective_count = sum(1 for c in connectives if c in response_lower)
        connective_score = min(10.0, 3.0 + connective_count * 1.5)
        
        # ============================================================
        # 12. SOPHISTICATION: proportion of "complex" words (3+ syllables)
        # ============================================================
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        complex_ratio = complex_words / num_words
        # Ideal: 15-30% complex words
        if 0.10 <= complex_ratio <= 0.35:
            complexity_score = 10.0
        elif 0.05 <= complex_ratio < 0.10 or 0.35 < complex_ratio <= 0.45:
            complexity_score = 7.0
        elif complex_ratio < 0.05:
            complexity_score = 4.0
        else:
            complexity_score = 5.0
        
        # ============================================================
        # FINAL WEIGHTED SCORE
        # ============================================================
        weights = {
            'flesch': 0.10,
            'ttr': 0.12,
            'variety': 0.08,
            'wordlen': 0.08,
            'punct': 0.08,
            'spelling': 0.12,
            'structural': 0.10,
            'length': 0.12,
            'connective': 0.08,
            'complexity': 0.08,
        }
        # Verify weights sum to ~1.0 (leaving room for bonus)
        # 0.10+0.12+0.08+0.08+0.08+0.12+0.10+0.12+0.08+0.08 = 0.96
        
        scores = {
            'flesch': flesch_score,
            'ttr': ttr_score,
            'variety': variety_score,
            'wordlen': wordlen_score,
            'punct': punct_score,
            'spelling': spelling_score,
            'structural': structural_score,
            'length': length_score,
            'connective': connective_score,
            'complexity': complexity_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        # Small bonus for having multiple sentences (sign of developed thought)
        if num_sentences >= 3:
            final_score = min(10.0, final_score + 0.3)
        if num_sentences >= 5:
            final_score = min(10.0, final_score + 0.2)
        
        return round(final_score, 3)
    
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 20:
                return 3.0
        except Exception:
            pass
        return 1.0