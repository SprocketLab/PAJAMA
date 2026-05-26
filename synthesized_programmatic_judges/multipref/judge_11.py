def judging_function(query, response):
    """
    Evaluates language quality and readability using a combination of:
    - Gunning Fog Index (different from Flesch used in other variants)
    - Punctuation variety and correctness
    - Sentence structure diversity (std dev of sentence lengths)
    - Transition/cohesion word usage
    - Type-token ratio with hapax legomena ratio
    - Character-level entropy as a proxy for vocabulary richness
    - Paragraph structure analysis
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.0
        
        # --- Tokenization ---
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        words_lower = [w.lower() for w in words]
        
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        
        # --- Syllable counter (different heuristic: vowel group based with more exceptions) ---
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing silent e
            if word.endswith('e') and not word.endswith('le') and len(word) > 3:
                word = word[:-1]
            # Count vowel groups
            vowel_groups = re.findall(r'[aeiouy]+', word)
            count = len(vowel_groups)
            # Adjust for common patterns
            if word.endswith('ed') and len(word) > 3 and word[-3] not in 'dt':
                count = max(count - 1, 1)
            if word.endswith('ion'):
                count += 1
            if word.endswith('ious') or word.endswith('eous'):
                count += 1
            return max(count, 1)
        
        syllable_counts = [count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        
        # --- 1. Gunning Fog Index ---
        # Fog = 0.4 * (avg_sentence_length + percentage_complex_words)
        # Complex words = 3+ syllables
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        pct_complex = (complex_words / num_words) * 100
        avg_sentence_length = num_words / num_sentences
        
        fog_index = 0.4 * (avg_sentence_length + pct_complex)
        
        # Ideal fog: 8-14 (accessible but not too simple)
        if fog_index < 6:
            fog_score = fog_index / 6.0 * 5  # too simple
        elif fog_index <= 14:
            fog_score = 10 - abs(fog_index - 11) * (5 / 6)  # sweet spot around 11
        else:
            fog_score = max(0, 10 - (fog_index - 14) * 0.5)  # too complex
        
        # --- 2. Sentence length diversity (std dev) ---
        sent_word_counts = []
        for sent in sentences:
            sw = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", sent)
            sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) > 1:
            mean_swc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_swc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sent_len = math.sqrt(variance)
            # Some variety is good, but not too wild
            # Ideal std dev: 3-10
            if std_sent_len < 1:
                diversity_score = 2.0
            elif std_sent_len <= 10:
                diversity_score = min(10, 2 + std_sent_len * 0.8)
            else:
                diversity_score = max(3, 10 - (std_sent_len - 10) * 0.3)
        else:
            diversity_score = 2.0
        
        # --- 3. Punctuation variety and usage ---
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-–—()[]{}"\'/':
                punct_types.add(ch)
        
        # Count specific punctuation
        commas = text.count(',')
        colons = text.count(':')
        semicolons = text.count(';')
        dashes = text.count('-') + text.count('–') + text.count('—')
        parens = text.count('(') + text.count(')')
        
        punct_variety = len(punct_types)
        # Normalize by text length
        punct_density = sum(1 for ch in text if ch in string.punctuation) / max(len(text), 1)
        
        # Good punct density: 0.03-0.10
        if punct_density < 0.02:
            punct_density_score = 3.0
        elif punct_density <= 0.10:
            punct_density_score = 7.0 + (punct_density - 0.02) * 30
        else:
            punct_density_score = max(4, 10 - (punct_density - 0.10) * 30)
        
        punct_variety_score = min(10, punct_variety * 1.2)
        punct_score = 0.5 * punct_density_score + 0.5 * punct_variety_score
        
        # --- 4. Transition/cohesion words ---
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'although', 'whereas',
            'similarly', 'likewise', 'conversely', 'instead', 'otherwise',
            'specifically', 'particularly', 'notably', 'importantly', 'significantly',
            'first', 'second', 'third', 'finally', 'next', 'then', 'also',
            'besides', 'indeed', 'certainly', 'clearly', 'obviously',
            'essentially', 'basically', 'overall', 'thus', 'hence',
            'accordingly', 'subsequently', 'previously', 'initially',
            'ultimately', 'primarily', 'generally', 'typically',
            'for example', 'for instance', 'in addition', 'in contrast',
            'on the other hand', 'as a result', 'in fact', 'of course',
            'in particular', 'in other words'
        }
        
        text_lower = text.lower()
        transition_count = 0
        for tw in transition_words:
            transition_count += len(re.findall(r'\b' + re.escape(tw) + r'\b', text_lower))
        
        transition_density = transition_count / num_sentences
        # Ideal: 0.3-1.5 transitions per sentence
        if transition_density < 0.1:
            transition_score = 3.0
        elif transition_density <= 1.5:
            transition_score = 3 + transition_density * 4.67
        else:
            transition_score = max(5, 10 - (transition_density - 1.5) * 2)
        
        # --- 5. Hapax legomena ratio (words appearing exactly once) ---
        word_freq = collections.Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        type_count = len(word_freq)
        
        hapax_ratio = hapax / max(type_count, 1)
        # Higher hapax ratio = richer vocabulary (but diminishing returns)
        # Typical good range: 0.4-0.8
        hapax_score = min(10, hapax_ratio * 12)
        
        # Type-token ratio
        ttr = type_count / max(num_words, 1)
        # Ideal TTR depends on length; for moderate texts, 0.4-0.7 is good
        ttr_score = min(10, ttr * 14)
        
        vocab_score = 0.5 * hapax_score + 0.5 * ttr_score
        
        # --- 6. Character-level entropy ---
        char_freq = collections.Counter(text.lower())
        total_chars = sum(char_freq.values())
        entropy = 0.0
        for count in char_freq.values():
            if count > 0:
                p = count / total_chars
                entropy -= p * math.log2(p)
        
        # Good English text typically has entropy around 4.0-4.5
        # Normalize to 0-10
        entropy_score = min(10, max(0, (entropy - 2.5) * 4))
        
        # --- 7. Formatting and structure ---
        # Check for markdown headers, bullet points, numbered lists
        has_headers = bool(re.search(r'#{1,4}\s', text))
        has_bullets = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
        has_numbered = bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', text))
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        structure_score = 3.0  # base
        if has_headers:
            structure_score += 2.0
        if has_bullets or has_numbered:
            structure_score += 2.0
        if has_bold:
            structure_score += 1.5
        if num_paragraphs >= 2:
            structure_score += 1.5
        structure_score = min(10, structure_score)
        
        # --- 8. Sentence opener variety ---
        openers = []
        for sent in sentences:
            first_word = re.match(r'[a-zA-Z]+', sent.strip())
            if first_word:
                openers.append(first_word.group().lower())
        
        if openers:
            unique_openers = len(set(openers))
            opener_ratio = unique_openers / len(openers)
            opener_score = min(10, opener_ratio * 12)
        else:
            opener_score = 5.0
        
        # --- 9. Average word length (proxy for sophistication) ---
        avg_word_len = sum(len(w) for w in words) / num_words
        # Ideal: 4.5-6.0
        if avg_word_len < 3.5:
            word_len_score = 3.0
        elif avg_word_len <= 6.0:
            word_len_score = 3 + (avg_word_len - 3.5) * 2.8
        else:
            word_len_score = max(5, 10 - (avg_word_len - 6.0) * 1.5)
        
        # --- 10. Spelling heuristic: unusual character patterns ---
        # Count words with unlikely letter triples
        unusual_patterns = 0
        for w in words_lower:
            if len(w) >= 3:
                # Triple same letter
                if re.search(r'(.)\1\1', w):
                    unusual_patterns += 1
                # No vowels in word of 4+ chars
                if len(w) >= 4 and not re.search(r'[aeiouy]', w):
                    unusual_patterns += 1
        
        spelling_penalty = min(5, unusual_patterns * 0.5)
        
        # --- Combine scores with weights ---
        # Weights emphasize different aspects than other variants
        final_score = (
            fog_score * 0.12 +           # readability level
            diversity_score * 0.12 +      # sentence variety
            punct_score * 0.08 +          # punctuation quality
            transition_score * 0.14 +     # cohesion
            vocab_score * 0.12 +          # vocabulary richness
            entropy_score * 0.08 +        # character diversity
            structure_score * 0.14 +      # formatting
            opener_score * 0.08 +         # sentence opener variety
            word_len_score * 0.07 +       # word sophistication
            5.0 * 0.05                    # base component
        ) - spelling_penalty * 0.1
        
        # Length bonus: slightly reward completeness (but not too much)
        length_factor = min(1.0, num_words / 50)  # ramp up to 50 words
        if num_words < 20:
            length_factor = num_words / 30.0
        
        final_score = final_score * (0.7 + 0.3 * length_factor)
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 4)
        
    except Exception:
        return 5.0