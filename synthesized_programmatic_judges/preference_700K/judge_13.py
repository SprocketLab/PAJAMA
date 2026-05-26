def judging_function(query, response):
    """
    Evaluate language quality and readability using:
    - Gunning Fog Index (different from Flesch used in other variants)
    - Punctuation variety and correctness
    - Sentence structure variety (std dev of sentence lengths)
    - Cohesion markers (pronouns, conjunctions, discourse markers)
    - Lexical sophistication (longer words ratio, uncommon letter patterns)
    - Completeness heuristics
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
        
        # === Tokenization ===
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        if not sentences:
            sentences = [text]
        
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        if not words:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        num_words = len(words_lower)
        num_sentences = max(len(sentences), 1)
        
        # === 1. Gunning Fog Index (different from Flesch) ===
        # Complex words = words with 3+ syllables (excluding common suffixes)
        def count_syllables(word):
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
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            return max(count, 1)
        
        complex_words = 0
        for w in words_lower:
            syl = count_syllables(w)
            if syl >= 3:
                # Exclude common suffixes that inflate syllable count
                if not (w.endswith('ing') or w.endswith('ed') or w.endswith('es') or w.endswith('ly')):
                    complex_words += 1
                elif syl >= 4:
                    complex_words += 1
        
        avg_sentence_len = num_words / num_sentences
        complex_word_pct = (complex_words / max(num_words, 1)) * 100
        fog_index = 0.4 * (avg_sentence_len + complex_word_pct)
        
        # Ideal fog: 8-14 (readable but not simplistic)
        if fog_index < 4:
            fog_score = 3.0
        elif fog_index < 8:
            fog_score = 5.0 + (fog_index - 4) * 0.625  # 5 to 7.5
        elif fog_index <= 14:
            fog_score = 7.5 + (fog_index - 8) * (2.5 / 6)  # 7.5 to 10
        elif fog_index <= 20:
            fog_score = 10.0 - (fog_index - 14) * 0.5  # 10 to 7
        else:
            fog_score = max(2.0, 7.0 - (fog_index - 20) * 0.3)
        
        # === 2. Punctuation variety and density ===
        punct_marks = re.findall(r'[,;:\-\—\(\)\[\]\"\'`]', text)
        punct_density = len(punct_marks) / max(num_words, 1)
        
        # Count distinct punctuation types used
        punct_types = set(re.findall(r'[,;:\-\—\(\)\[\]\"\'…–]', text))
        punct_variety = len(punct_types)
        
        # Ideal density: 0.05 - 0.20
        if punct_density < 0.02:
            punct_score = 3.0
        elif punct_density < 0.05:
            punct_score = 5.0
        elif punct_density <= 0.20:
            punct_score = 7.0 + min(punct_variety, 5) * 0.4
        elif punct_density <= 0.35:
            punct_score = 7.0
        else:
            punct_score = 4.0
        
        punct_score = min(punct_score, 10.0)
        
        # === 3. Sentence length variety (std dev) ===
        sent_word_counts = []
        for s in sentences:
            sw = len(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s))
            if sw > 0:
                sent_word_counts.append(sw)
        
        if len(sent_word_counts) >= 2:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance)
            # Coefficient of variation
            cv = std_sl / max(mean_sl, 1)
            # Ideal CV: 0.3-0.7 (good variety but not chaotic)
            if cv < 0.1:
                variety_score = 4.0
            elif cv < 0.3:
                variety_score = 6.0 + (cv - 0.1) * 10  # 6 to 8
            elif cv <= 0.7:
                variety_score = 8.0 + (cv - 0.3) * 5  # 8 to 10
            elif cv <= 1.0:
                variety_score = 10.0 - (cv - 0.7) * 6.67  # 10 to 8
            else:
                variety_score = max(3.0, 8.0 - (cv - 1.0) * 5)
        elif len(sent_word_counts) == 1:
            variety_score = 5.0  # Single sentence - neutral
        else:
            variety_score = 3.0
        
        # === 4. Cohesion and discourse markers ===
        discourse_markers = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'nevertheless', 'consequently', 'meanwhile', 'specifically',
            'essentially', 'particularly', 'alternatively', 'similarly',
            'conversely', 'notably', 'importantly', 'indeed', 'thus',
            'hence', 'accordingly', 'likewise', 'nonetheless', 'regardless',
            'overall', 'ultimately', 'typically', 'generally', 'basically',
            'primarily', 'certainly', 'clearly', 'obviously', 'apparently',
            'fortunately', 'unfortunately', 'interestingly', 'surprisingly',
            'significantly', 'arguably'
        }
        
        connectives = {
            'because', 'since', 'although', 'though', 'while', 'whereas',
            'unless', 'whether', 'besides', 'despite', 'except',
            'rather', 'instead', 'otherwise', 'perhaps', 'maybe'
        }
        
        # Transition phrases (check in text)
        transition_phrases = [
            'for example', 'for instance', 'in addition', 'on the other hand',
            'in other words', 'as a result', 'in fact', 'of course',
            'at the same time', 'in particular', 'to be fair', 'that said',
            'having said that', 'the trade-off', 'in contrast', 'as such',
            'that being said', 'to put it', 'in terms of', 'with that'
        ]
        
        text_lower = text.lower()
        
        dm_count = sum(1 for w in words_lower if w in discourse_markers)
        conn_count = sum(1 for w in words_lower if w in connectives)
        phrase_count = sum(1 for p in transition_phrases if p in text_lower)
        
        cohesion_total = dm_count + conn_count + phrase_count * 2
        cohesion_density = cohesion_total / max(num_sentences, 1)
        
        # Ideal: 0.3 - 1.5 per sentence
        if cohesion_density < 0.1:
            cohesion_score = 3.0
        elif cohesion_density < 0.3:
            cohesion_score = 5.0
        elif cohesion_density <= 1.5:
            cohesion_score = 7.0 + (cohesion_density - 0.3) * 2.5
        elif cohesion_density <= 3.0:
            cohesion_score = 9.0
        else:
            cohesion_score = 7.0
        cohesion_score = min(cohesion_score, 10.0)
        
        # === 5. Lexical sophistication ===
        # Ratio of words with 7+ characters (sophisticated vocabulary)
        long_words = sum(1 for w in words_lower if len(w) >= 7)
        long_word_ratio = long_words / max(num_words, 1)
        
        # Hapax legomena ratio (words appearing only once) - indicates rich vocabulary
        word_freq = Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(len(word_freq), 1)
        
        # Type-token ratio with correction for text length (Guiraud's index)
        unique_words = len(word_freq)
        guiraud = unique_words / max(math.sqrt(num_words), 1)
        
        # Combine sophistication metrics
        # long_word_ratio ideal: 0.15-0.35
        if long_word_ratio < 0.05:
            lw_score = 3.0
        elif long_word_ratio < 0.15:
            lw_score = 5.0 + (long_word_ratio - 0.05) * 30
        elif long_word_ratio <= 0.35:
            lw_score = 8.0 + (long_word_ratio - 0.15) * 10
        else:
            lw_score = max(5.0, 10.0 - (long_word_ratio - 0.35) * 10)
        
        # Guiraud ideal: 5-10
        if guiraud < 3:
            guiraud_score = 4.0
        elif guiraud < 5:
            guiraud_score = 6.0
        elif guiraud <= 10:
            guiraud_score = 8.0 + (guiraud - 5) * 0.4
        else:
            guiraud_score = 9.0
        guiraud_score = min(guiraud_score, 10.0)
        
        sophistication_score = lw_score * 0.5 + guiraud_score * 0.5
        
        # === 6. Spelling / error heuristics ===
        # Check for common error patterns
        error_count = 0
        # Double spaces
        error_count += len(re.findall(r'  +', text)) * 0.5
        # Missing space after punctuation
        error_count += len(re.findall(r'[.!?,;:][a-zA-Z]', text))
        # Repeated words
        for i in range(1, len(words_lower)):
            if words_lower[i] == words_lower[i-1] and words_lower[i] not in {'the', 'that', 'had', 'is', 'very'}:
                error_count += 1
        # Unclosed parentheses/brackets
        if text.count('(') != text.count(')'):
            error_count += 1
        if text.count('[') != text.count(']'):
            error_count += 1
        
        error_rate = error_count / max(num_sentences, 1)
        error_score = max(2.0, 10.0 - error_rate * 3)
        
        # === 7. Response length and completeness ===
        # Longer, more complete responses tend to be better
        # But check if response seems truncated
        is_truncated = text.endswith(('...', '..', ' the', ' a', ' an', ' to', ' of', ' in', ' and', ' or', ' but', ' with', ' for', ' is', ' was', ' that', ' this'))
        
        if num_words < 10:
            length_score = 3.0
        elif num_words < 25:
            length_score = 5.0
        elif num_words < 50:
            length_score = 6.5
        elif num_words < 100:
            length_score = 7.5
        elif num_words < 200:
            length_score = 8.5
        else:
            length_score = 9.0
        
        if is_truncated:
            length_score -= 0.5  # Slight penalty, many good responses get truncated
        
        # === 8. Sentence opening variety ===
        if len(sentences) >= 3:
            first_words = []
            for s in sentences:
                fw = re.findall(r'[a-zA-Z]+', s)
                if fw:
                    first_words.append(fw[0].lower())
            if first_words:
                unique_openers = len(set(first_words))
                opener_ratio = unique_openers / len(first_words)
                opener_score = 4.0 + opener_ratio * 6.0  # 4-10
            else:
                opener_score = 5.0
        else:
            opener_score = 6.0  # Neutral for short responses
        
        # === 9. Use of specific/concrete language ===
        # Check for numbers, proper nouns, specific references
        has_numbers = len(re.findall(r'\d+', text))
        has_proper_nouns = len(re.findall(r'(?<!\. )[A-Z][a-z]{2,}', text[1:]))  # Skip first char
        has_quotes = text.count('"') + text.count("'") + text.count('*')
        
        specificity = min(has_numbers * 0.5 + has_proper_nouns * 0.3 + has_quotes * 0.2, 5)
        specificity_score = 5.0 + specificity
        
        # === Combine all scores with weights ===
        final_score = (
            fog_score * 0.12 +           # Readability level
            punct_score * 0.08 +          # Punctuation usage
            variety_score * 0.12 +        # Sentence variety
            cohesion_score * 0.15 +       # Discourse cohesion
            sophistication_score * 0.15 + # Vocabulary sophistication
            error_score * 0.08 +          # Error-free writing
            length_score * 0.15 +         # Completeness
            opener_score * 0.07 +         # Sentence opening variety
            specificity_score * 0.08      # Concrete language
        )
        
        # Scale to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        # Fallback: return a middling score based on length
        try:
            return min(5.0, len(str(response)) / 100.0)
        except Exception:
            return 2.0