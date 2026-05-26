def judging_function(query, response):
    """
    Evaluates language quality and readability using a combination of:
    - Punctuation variety and correctness
    - Sentence structure diversity (length variance)
    - Transition word usage
    - Paragraph structure
    - Type-token ratio with hapax legomena ratio
    - Gunning Fog Index (different from Flesch)
    - Discourse markers and connective density
    - Capitalization correctness
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.5
        
        # ---- Helper: count syllables (simple heuristic) ----
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing silent e
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
            return max(1, count)
        
        # ---- Tokenization ----
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r"[a-zA-Z']+", text)
        num_words = max(len(words), 1)
        
        words_lower = [w.lower() for w in words]
        
        # ---- 1. Gunning Fog Index (different from Flesch) ----
        # Fog = 0.4 * (ASL + PHW)
        # ASL = average sentence length, PHW = % hard words (3+ syllables)
        asl = num_words / num_sentences
        hard_words = sum(1 for w in words_lower if count_syllables(w) >= 3)
        phw = (hard_words / num_words) * 100
        fog_index = 0.4 * (asl + phw)
        
        # Ideal fog: 8-14 (readable but not too simple)
        if fog_index < 6:
            fog_score = 3.0
        elif fog_index <= 8:
            fog_score = 5.0 + (fog_index - 6) * 1.25
        elif fog_index <= 14:
            fog_score = 7.5 + (1 - abs(fog_index - 11) / 3) * 2.5
        elif fog_index <= 18:
            fog_score = 7.5 - (fog_index - 14) * 0.5
        else:
            fog_score = max(2.0, 5.5 - (fog_index - 18) * 0.3)
        
        # ---- 2. Sentence length variance (structural diversity) ----
        sent_lengths = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            sent_lengths.append(len(sw))
        
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Coefficient of variation
            cv = std_sl / max(mean_sl, 1)
            # Good variety: cv around 0.3-0.6
            if cv < 0.1:
                variety_score = 3.0
            elif cv < 0.3:
                variety_score = 5.0 + (cv - 0.1) * 15
            elif cv <= 0.6:
                variety_score = 8.0 + (cv - 0.3) * 3.33
            elif cv <= 1.0:
                variety_score = 9.0 - (cv - 0.6) * 5
            else:
                variety_score = max(2.0, 7.0 - cv * 2)
        else:
            variety_score = 3.0
        
        # ---- 3. Transition / discourse markers ----
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'specifically',
            'alternatively', 'similarly', 'conversely', 'notably', 'importantly',
            'essentially', 'ultimately', 'overall', 'indeed', 'certainly',
            'although', 'whereas', 'while', 'since', 'because', 'thus',
            'hence', 'accordingly', 'besides', 'nonetheless', 'regardless',
            'first', 'second', 'third', 'finally', 'next', 'then',
            'in addition', 'for example', 'for instance', 'in contrast',
            'on the other hand', 'as a result', 'in fact', 'of course',
            'in particular', 'in other words', 'that said', 'to summarize'
        }
        
        text_lower = text.lower()
        transition_count = 0
        for tw in transition_words:
            if ' ' in tw:
                transition_count += text_lower.count(tw)
            else:
                transition_count += words_lower.count(tw)
        
        transition_density = transition_count / num_sentences
        # Ideal: 0.3-0.8 per sentence
        if transition_density < 0.1:
            transition_score = 3.0
        elif transition_density < 0.3:
            transition_score = 5.0 + (transition_density - 0.1) * 15
        elif transition_density <= 0.8:
            transition_score = 8.0 + (transition_density - 0.3) * 4
        elif transition_density <= 1.5:
            transition_score = 10.0 - (transition_density - 0.8) * 2
        else:
            transition_score = max(3.0, 8.6 - transition_density * 2)
        
        # ---- 4. Punctuation variety ----
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-()[]{}"\'/':
                punct_types.add(ch)
        
        # Count specific punctuation uses
        colon_count = text.count(':')
        semicolon_count = text.count(';')
        dash_count = text.count('-') + text.count('—') + text.count('–')
        paren_count = text.count('(') + text.count(')')
        
        punct_variety = len(punct_types)
        # Also check for markdown formatting (bold, headers, lists)
        has_bold = '**' in text or '__' in text
        has_headers = bool(re.search(r'^#{1,4}\s', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE)) or bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
        
        formatting_bonus = (1.0 if has_bold else 0) + (1.0 if has_headers else 0) + (0.8 if has_lists else 0)
        
        punct_score = min(10.0, 3.0 + punct_variety * 0.5 + formatting_bonus + 
                         min(colon_count, 3) * 0.2 + min(semicolon_count, 2) * 0.3)
        
        # ---- 5. Hapax legomena ratio & vocabulary richness ----
        word_freq = Counter(words_lower)
        unique_words = len(word_freq)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        
        # Type-token ratio (with correction for text length using Guiraud's index)
        guiraud = unique_words / math.sqrt(num_words) if num_words > 0 else 0
        hapax_ratio = hapax / unique_words if unique_words > 0 else 0
        
        # Guiraud ideal: 4-8 for good text
        if guiraud < 2:
            vocab_score = 3.0
        elif guiraud < 4:
            vocab_score = 4.0 + (guiraud - 2) * 1.5
        elif guiraud <= 8:
            vocab_score = 7.0 + (guiraud - 4) * 0.5
        else:
            vocab_score = min(10.0, 9.0 + (guiraud - 8) * 0.1)
        
        # Bonus for good hapax ratio (indicates rich vocabulary)
        if hapax_ratio > 0.5:
            vocab_score = min(10.0, vocab_score + 0.5)
        
        # ---- 6. Capitalization correctness ----
        cap_errors = 0
        # Check sentences start with capital
        for s in sentences:
            s = s.strip()
            if s and s[0].isalpha() and s[0].islower():
                cap_errors += 1
        
        cap_error_rate = cap_errors / num_sentences
        cap_score = max(3.0, 10.0 - cap_error_rate * 10)
        
        # ---- 7. Opening engagement ----
        # Does the response start with an engaging opener?
        first_50 = text[:min(150, len(text))].lower()
        engagement_patterns = [
            r'\bcertainly\b', r'\bgreat\b', r'\bexcellent\b', r'\babsolutely\b',
            r'\bhere\s+are\b', r'\blet\'?s\b', r'\bthe\s+art\b', r'\bawesome\b',
            r'\bthat\'?s\s+a\b', r'\bwhat\s+a\b', r'\binteresting\b',
            r'\bgood\s+question\b', r'\bof\s+course\b'
        ]
        engagement_score = 5.0
        for pat in engagement_patterns:
            if re.search(pat, first_50):
                engagement_score = 8.0
                break
        
        # Check if response directly addresses the query
        query_words = set(re.findall(r'[a-z]+', query.lower())) if query else set()
        response_start_words = set(re.findall(r'[a-z]+', first_50))
        overlap = len(query_words & response_start_words)
        if overlap >= 2:
            engagement_score = min(10.0, engagement_score + 1.0)
        
        # ---- 8. Paragraph structure ----
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if num_paragraphs >= 3:
            para_score = 8.0 + min(2.0, (num_paragraphs - 3) * 0.3)
        elif num_paragraphs == 2:
            para_score = 6.5
        else:
            # Single block - check length
            if num_words > 100:
                para_score = 4.0  # Long single block is bad
            else:
                para_score = 6.0
        
        # ---- 9. Spelling heuristic (common error patterns) ----
        # Check for repeated characters (e.g., "reallly", "goood")
        repeated_char_pattern = re.findall(r'([a-zA-Z])\1{2,}', text)
        # Check for common misspelling patterns
        spelling_penalty = len(repeated_char_pattern) * 0.5
        spelling_score = max(3.0, 10.0 - spelling_penalty)
        
        # ---- 10. Response completeness ----
        # Does the response end mid-sentence? (truncation check)
        last_chars = text[-5:].strip() if len(text) >= 5 else text
        ends_properly = last_chars[-1] in '.!?:"\')' if last_chars else False
        completeness_score = 8.0 if ends_properly else 4.0
        
        # ---- Weighted combination ----
        weights = {
            'fog': 0.12,
            'variety': 0.13,
            'transition': 0.13,
            'punct': 0.10,
            'vocab': 0.12,
            'cap': 0.05,
            'engagement': 0.10,
            'para': 0.10,
            'spelling': 0.05,
            'completeness': 0.10,
        }
        
        scores = {
            'fog': fog_score,
            'variety': variety_score,
            'transition': transition_score,
            'punct': punct_score,
            'vocab': vocab_score,
            'cap': cap_score,
            'engagement': engagement_score,
            'para': para_score,
            'spelling': spelling_score,
            'completeness': completeness_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 4)
    
    except Exception:
        return 5.0