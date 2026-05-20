def judging_function(query, response):
    """
    Evaluate language quality and readability using:
    - Punctuation correctness and variety
    - Spelling error heuristics (unusual character patterns)
    - Sentence structure variety (length variance)
    - Cohesion markers and discourse connectives
    - Type-token ratio with hapax legomena ratio
    - Gunning Fog-like complexity measure
    - Character-level n-gram entropy for naturalness
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        # ---- Tokenization ----
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", text)
        if len(words) < 2:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        num_words = len(words_lower)
        
        # ---- Sentence splitting (more robust) ----
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=\n)\s*(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if len(sentences) == 0:
            sentences = [text]
        num_sentences = len(sentences)
        
        # ---- 1. Sentence length variety (std dev of sentence word counts) ----
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            sent_word_counts.append(len(sw))
        
        avg_sent_len = sum(sent_word_counts) / max(len(sent_word_counts), 1)
        
        if len(sent_word_counts) > 1:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sent_len = math.sqrt(variance_sl)
        else:
            std_sent_len = 0.0
        
        # Reward moderate sentence length variety (not too uniform, not too wild)
        # Optimal std around 5-15
        variety_score = min(std_sent_len / 10.0, 1.5) if std_sent_len < 30 else max(0, 1.5 - (std_sent_len - 30) / 30)
        variety_score = max(0, variety_score)
        
        # Penalize very short average sentence length or very long
        avg_len_score = 0
        if 8 <= avg_sent_len <= 25:
            avg_len_score = 1.0
        elif avg_sent_len < 8:
            avg_len_score = avg_sent_len / 8.0
        else:
            avg_len_score = max(0, 1.0 - (avg_sent_len - 25) / 30)
        
        # ---- 2. Punctuation quality ----
        # Check for proper punctuation usage
        punct_types_used = set()
        for ch in text:
            if ch in '.,;:!?-()"\'"':
                punct_types_used.add(ch)
        
        punct_variety = min(len(punct_types_used) / 5.0, 1.0)
        
        # Ratio of sentences ending with proper punctuation
        proper_endings = sum(1 for s in sentences if s.strip() and s.strip()[-1] in '.!?')
        ending_ratio = proper_endings / max(num_sentences, 1)
        
        # Check for comma usage (indicates clause structure)
        comma_count = text.count(',')
        comma_ratio = comma_count / max(num_sentences, 1)
        comma_score = min(comma_ratio / 2.0, 1.0)  # ~2 commas per sentence is good
        
        punctuation_score = (punct_variety * 0.3 + ending_ratio * 0.4 + comma_score * 0.3)
        
        # ---- 3. Spelling heuristic: detect unusual letter patterns ----
        # Common misspelling patterns: double letters that shouldn't be, unusual trigrams
        misspelling_indicators = 0
        for w in words_lower:
            if len(w) < 2:
                continue
            # Triple letters
            if re.search(r'(.)\1\1', w):
                misspelling_indicators += 1
            # Unusual bigrams that rarely appear in English
            unusual_bigrams = ['xz', 'zx', 'qk', 'jx', 'vq', 'wz', 'zw', 'xj']
            for bg in unusual_bigrams:
                if bg in w:
                    misspelling_indicators += 1
            # Words with no vowels (except common ones)
            if len(w) > 2 and not re.search(r'[aeiouy]', w):
                misspelling_indicators += 1
        
        spelling_score = max(0, 1.0 - misspelling_indicators / max(num_words * 0.05, 1))
        
        # ---- 4. Vocabulary richness: Hapax legomena ratio + modified TTR ----
        word_freq = Counter(words_lower)
        unique_words = len(word_freq)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        
        # Yule's K measure (vocabulary richness)
        if num_words > 0 and unique_words > 0:
            # Root TTR (Guiraud's index)
            guiraud = unique_words / math.sqrt(num_words)
            guiraud_score = min(guiraud / 8.0, 1.0)
            
            # Hapax ratio
            hapax_ratio = hapax / max(unique_words, 1)
            hapax_score = min(hapax_ratio / 0.6, 1.0)  # 60%+ hapax is rich
        else:
            guiraud_score = 0
            hapax_score = 0
        
        vocab_score = guiraud_score * 0.6 + hapax_score * 0.4
        
        # ---- 5. Discourse connectives and cohesion markers ----
        connectives = [
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'additionally', 'meanwhile', 'although', 'whereas',
            'despite', 'instead', 'otherwise', 'similarly', 'likewise',
            'specifically', 'particularly', 'essentially', 'basically',
            'for example', 'for instance', 'in addition', 'on the other hand',
            'in contrast', 'as a result', 'in fact', 'of course',
            'that said', 'in other words', 'to be fair', 'more importantly',
            'first', 'second', 'third', 'finally', 'also', 'thus',
            'hence', 'yet', 'still', 'indeed', 'certainly', 'clearly',
            'apparently', 'typically', 'generally', 'usually', 'often',
            'while', 'since', 'because', 'though', 'unless', 'whether',
            'rather', 'perhaps', 'arguably'
        ]
        
        text_lower = text.lower()
        connective_count = 0
        for conn in connectives:
            # Count as whole word/phrase
            connective_count += len(re.findall(r'\b' + re.escape(conn) + r'\b', text_lower))
        
        connective_density = connective_count / max(num_sentences, 1)
        cohesion_score = min(connective_density / 1.5, 1.0)
        
        # ---- 6. Gunning Fog-like complexity ----
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
            # Adjust for silent e
            if word.endswith('e') and count > 1:
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            return max(count, 1)
        
        complex_words = sum(1 for w in words_lower if count_syllables_approx(w) >= 3)
        complex_ratio = complex_words / max(num_words, 1)
        
        # Gunning Fog approximation
        fog_index = 0.4 * (avg_sent_len + 100 * complex_ratio)
        
        # Optimal fog: 8-14 (readable but not simplistic)
        if 8 <= fog_index <= 14:
            fog_score = 1.0
        elif fog_index < 8:
            fog_score = fog_index / 8.0
        else:
            fog_score = max(0, 1.0 - (fog_index - 14) / 20)
        
        # ---- 7. Character bigram entropy (naturalness of text) ----
        clean_text = re.sub(r'[^a-z ]', '', text_lower)
        if len(clean_text) > 2:
            bigrams = [clean_text[i:i+2] for i in range(len(clean_text) - 1)]
            bg_freq = Counter(bigrams)
            total_bg = len(bigrams)
            entropy = 0
            for bg, cnt in bg_freq.items():
                p = cnt / total_bg
                if p > 0:
                    entropy -= p * math.log2(p)
            # English text typically has bigram entropy around 4-6
            if 3.5 <= entropy <= 7.0:
                entropy_score = 1.0
            elif entropy < 3.5:
                entropy_score = entropy / 3.5
            else:
                entropy_score = max(0, 1.0 - (entropy - 7.0) / 3.0)
        else:
            entropy_score = 0.3
        
        # ---- 8. Sentence starter variety ----
        if len(sentences) >= 3:
            starters = []
            for s in sentences:
                first_word_match = re.match(r'[^a-zA-Z]*([a-zA-Z]+)', s)
                if first_word_match:
                    starters.append(first_word_match.group(1).lower())
            if starters:
                unique_starters = len(set(starters))
                starter_variety = unique_starters / len(starters)
            else:
                starter_variety = 0.5
        else:
            starter_variety = 0.7  # neutral for short texts
        
        # ---- 9. Capitalization correctness ----
        # Check if sentences start with capital letters
        cap_correct = 0
        for s in sentences:
            stripped = s.lstrip()
            if stripped and stripped[0].isupper():
                cap_correct += 1
        cap_score = cap_correct / max(num_sentences, 1)
        
        # ---- 10. Response length bonus (longer, substantive responses tend to be better) ----
        length_bonus = min(num_words / 80.0, 1.0)  # up to 80 words for full bonus
        
        # ---- 11. Parenthetical/qualifying expressions (sign of nuanced writing) ----
        parenthetical_patterns = [
            r'\(.*?\)', r'--.*?--', r'—.*?—',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\betc\.\b'
        ]
        paren_count = 0
        for pat in parenthetical_patterns:
            paren_count += len(re.findall(pat, text))
        paren_score = min(paren_count / 3.0, 1.0)
        
        # ---- 12. Formatting signals (lists, bold, italics, code blocks) ----
        formatting_signals = 0
        if re.search(r'\*\*.*?\*\*', text) or re.search(r'\*[^*]+\*', text):
            formatting_signals += 1
        if re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
            formatting_signals += 1
        if re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE):
            formatting_signals += 1
        if '```' in text:
            formatting_signals += 1
        format_score = min(formatting_signals / 2.0, 1.0)
        
        # ---- Aggregate Score ----
        # Weight the components
        score = (
            avg_len_score * 1.2 +       # Sentence length appropriateness
            variety_score * 1.0 +         # Sentence variety
            punctuation_score * 1.3 +     # Punctuation quality
            spelling_score * 0.8 +        # Spelling correctness
            vocab_score * 1.2 +           # Vocabulary richness
            cohesion_score * 1.5 +        # Discourse cohesion
            fog_score * 0.8 +             # Readability complexity
            entropy_score * 0.5 +         # Text naturalness
            starter_variety * 0.6 +       # Sentence starter variety
            cap_score * 0.5 +             # Capitalization
            length_bonus * 1.2 +          # Response substance
            paren_score * 0.4 +           # Nuance markers
            format_score * 0.3            # Formatting
        )
        
        max_possible = (1.2 + 1.5 + 1.3 + 0.8 + 1.2 + 1.5 + 0.8 + 0.5 + 0.6 + 0.5 + 1.2 + 0.4 + 0.3)
        
        # Normalize to 0-10
        final_score = (score / max_possible) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
    
    except Exception:
        return 3.0