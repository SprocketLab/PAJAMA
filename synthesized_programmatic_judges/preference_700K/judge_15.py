def judging_function(query, response):
    """
    Evaluate language quality and readability using a substantially different approach:
    - Coleman-Liau Index (character-based readability, not syllable-based like Flesch)
    - Automated Readability Index (ARI)
    - Punctuation sophistication (use of semicolons, colons, dashes, parentheses)
    - Sentence structure variation (coefficient of variation of sentence lengths)
    - Hapax legomena ratio (words used exactly once / total words) as vocabulary richness
    - Connective/cohesion word density
    - Spelling heuristics based on unusual character patterns
    - Response completeness relative to query complexity
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        # ---- Basic tokenization ----
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        if not sentences:
            sentences = [response]
        
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", response)
        if not words:
            return 1.0
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        num_chars_in_words = sum(len(w) for w in words)
        
        words_lower = [w.lower() for w in words]
        word_freq = Counter(words_lower)
        unique_words = len(word_freq)
        
        # ---- 1. Coleman-Liau Index ----
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg letters per 100 words, S = avg sentences per 100 words
        L = (num_chars_in_words / num_words) * 100
        S = (num_sentences / num_words) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        # Ideal range for general audience: 8-14. Penalize extremes.
        cli_score = max(0, 10 - abs(coleman_liau - 11) * 0.8)
        
        # ---- 2. Automated Readability Index ----
        # ARI = 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
        ari = 4.71 * (num_chars_in_words / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        ari_score = max(0, 10 - abs(ari - 10) * 0.7)
        
        # ---- 3. Punctuation sophistication ----
        # Count sophisticated punctuation marks (not just periods/commas)
        semicolons = response.count(';')
        colons = response.count(':')
        dashes = response.count('—') + response.count('–') + len(re.findall(r' - ', response))
        parentheses = response.count('(')
        quotes = response.count('"') + response.count("'") + response.count('"') + response.count('"')
        commas = response.count(',')
        
        # Normalize by word count
        sophisticated_punct = semicolons + colons + dashes + parentheses
        punct_density = sophisticated_punct / max(num_words, 1) * 100
        comma_density = commas / max(num_words, 1) * 100
        
        # Good writing uses some sophisticated punctuation but not excessively
        punct_score = min(punct_density * 2.5, 6.0) + min(comma_density * 0.8, 4.0)
        punct_score = min(punct_score, 10.0)
        
        # ---- 4. Sentence length variation (coefficient of variation) ----
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 2:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance_sl)
            cv_sl = std_sl / max(mean_sl, 1)
            # Good writing has moderate variation (CV around 0.3-0.7)
            if cv_sl < 0.1:
                variation_score = 3.0  # Too uniform
            elif cv_sl < 0.3:
                variation_score = 5.0 + (cv_sl - 0.1) * 15
            elif cv_sl <= 0.7:
                variation_score = 8.0 + min((cv_sl - 0.3) * 5, 2.0)
            else:
                variation_score = max(10.0 - (cv_sl - 0.7) * 5, 4.0)
        elif len(sent_word_counts) == 1:
            variation_score = 5.0  # Can't measure variation with one sentence
        else:
            variation_score = 3.0
        
        # ---- 5. Hapax legomena ratio ----
        # Words used exactly once — indicates vocabulary richness
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(unique_words, 1)
        # In good writing, hapax ratio is typically 0.4-0.7
        if hapax_ratio < 0.3:
            hapax_score = 4.0 + hapax_ratio * 10
        elif hapax_ratio <= 0.75:
            hapax_score = 7.0 + (hapax_ratio - 0.3) * 6.67
        else:
            hapax_score = max(10.0 - (hapax_ratio - 0.75) * 8, 5.0)
        
        # ---- 6. Cohesion / connective word density ----
        cohesion_words = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'nonetheless', 'meanwhile', 'similarly',
            'conversely', 'specifically', 'particularly', 'essentially', 'ultimately',
            'accordingly', 'alternatively', 'subsequently', 'notably', 'importantly',
            'indeed', 'thus', 'hence', 'although', 'whereas', 'while',
            'because', 'since', 'despite', 'rather', 'instead',
            'for example', 'for instance', 'in addition', 'on the other hand',
            'in contrast', 'as a result', 'in fact', 'of course',
            'that said', 'in other words', 'to be fair'
        }
        
        response_lower = response.lower()
        cohesion_count = 0
        for cw in cohesion_words:
            if ' ' in cw:
                cohesion_count += response_lower.count(cw)
            else:
                cohesion_count += words_lower.count(cw)
        
        cohesion_density = cohesion_count / max(num_sentences, 1)
        cohesion_score = min(cohesion_density * 5.0, 10.0)
        # Bonus for having any cohesion at all
        if cohesion_count > 0:
            cohesion_score = max(cohesion_score, 3.0)
        
        # ---- 7. Spelling / typo heuristic ----
        # Detect likely misspellings via unusual patterns
        suspicious_patterns = [
            r'[a-z]{4,}[A-Z]',  # random capitals mid-word
            r'([a-zA-Z])\1{3,}',  # 4+ repeated letters
            r'[bcdfghjklmnpqrstvwxyz]{5,}',  # 5+ consonants in a row (unusual)
            r'[aeiou]{4,}',  # 4+ vowels in a row (unusual)
        ]
        
        typo_count = 0
        for w in words:
            if len(w) < 2:
                continue
            for pat in suspicious_patterns:
                if re.search(pat, w):
                    typo_count += 1
                    break
        
        # Also check for double spaces, missing space after punctuation
        double_spaces = len(re.findall(r'  +', response))
        missing_space_after_punct = len(re.findall(r'[.!?,;:][a-zA-Z]', response))
        
        error_rate = (typo_count + double_spaces * 0.5 + missing_space_after_punct * 0.5) / max(num_words, 1) * 100
        spelling_score = max(10.0 - error_rate * 3.0, 0.0)
        
        # ---- 8. Response length / completeness ----
        # Longer, more substantive responses tend to be better (up to a point)
        # Use log scale to avoid over-rewarding extremely long responses
        length_factor = math.log(max(num_words, 1) + 1, 2)
        # Sweet spot: 50-300 words
        if num_words < 10:
            length_score = 2.0
        elif num_words < 30:
            length_score = 4.0 + (num_words - 10) * 0.15
        elif num_words < 50:
            length_score = 7.0 + (num_words - 30) * 0.05
        elif num_words <= 300:
            length_score = 8.0 + min((num_words - 50) * 0.008, 2.0)
        else:
            length_score = 9.5 + min((num_words - 300) * 0.001, 0.5)
        length_score = min(length_score, 10.0)
        
        # ---- 9. Average word length (proxy for vocabulary sophistication) ----
        avg_word_len = num_chars_in_words / max(num_words, 1)
        # Good range: 4.5 - 6.0
        if avg_word_len < 3.5:
            wordlen_score = 4.0
        elif avg_word_len < 4.5:
            wordlen_score = 4.0 + (avg_word_len - 3.5) * 4.0
        elif avg_word_len <= 6.0:
            wordlen_score = 8.0 + (avg_word_len - 4.5) * 1.33
        else:
            wordlen_score = max(10.0 - (avg_word_len - 6.0) * 2.0, 5.0)
        
        # ---- 10. Structural formatting bonus ----
        # Check for markdown, lists, code blocks, etc.
        has_markdown_bold = '**' in response or '__' in response
        has_markdown_italic = bool(re.search(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        has_code_block = '```' in response
        has_numbered_list = bool(re.search(r'\n\s*\d+[.)]\s', response))
        has_bullet_list = bool(re.search(r'\n\s*[-*•]\s', response))
        
        formatting_score = 0.0
        if has_markdown_bold:
            formatting_score += 2.0
        if has_markdown_italic:
            formatting_score += 1.5
        if has_code_block:
            formatting_score += 2.0
        if has_numbered_list:
            formatting_score += 2.0
        if has_bullet_list:
            formatting_score += 2.0
        formatting_score = min(formatting_score, 6.0)
        
        # ---- Combine scores with weights ----
        weights = {
            'cli': 0.08,
            'ari': 0.08,
            'punct': 0.10,
            'variation': 0.12,
            'hapax': 0.10,
            'cohesion': 0.14,
            'spelling': 0.08,
            'length': 0.15,
            'wordlen': 0.08,
            'formatting': 0.07,
        }
        
        scores = {
            'cli': cli_score,
            'ari': ari_score,
            'punct': punct_score,
            'variation': variation_score,
            'hapax': hapax_score,
            'cohesion': cohesion_score,
            'spelling': spelling_score,
            'length': length_score,
            'wordlen': wordlen_score,
            'formatting': formatting_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
    
    except Exception:
        try:
            # Fallback: simple length-based score
            return min(len(response.split()) / 30.0, 5.0) if response else 0.0
        except Exception:
            return 0.0