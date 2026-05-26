def judging_function(query, response):
    """
    Evaluate language quality and readability using a different approach:
    - Coleman-Liau Index (instead of Flesch)
    - Character-level analysis (letter frequency distribution entropy)
    - Punctuation variety and density
    - Sentence structure variety (std dev of sentence lengths)
    - Hapax legomena ratio (words appearing exactly once)
    - Repetition penalty (detecting repeated phrases/sentences)
    - Coherence via transition words
    - Penalty for HTML/code artifacts
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        # Very short responses get low scores
        if len(text) < 3:
            return 0.5
        
        # === Basic tokenization ===
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        words = re.findall(r"[a-zA-Z']+", text)
        words_lower = [w.lower() for w in words]
        
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        num_chars_in_words = sum(len(w) for w in words)
        
        # === 1. Coleman-Liau Index ===
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg number of letters per 100 words
        # S = avg number of sentences per 100 words
        L = (num_chars_in_words / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        # Ideal CLI around 8-12 for general audience
        cli_clamped = max(0, min(cli, 20))
        # Score: best around 7-12
        if cli_clamped < 2:
            cli_score = cli_clamped * 2  # 0-4
        elif cli_clamped <= 12:
            cli_score = 6 + (cli_clamped - 2) * 0.4  # 6-10
        else:
            cli_score = max(4, 10 - (cli_clamped - 12) * 0.5)
        
        # === 2. Character-level entropy (letter frequency distribution) ===
        letter_counts = collections.Counter(c.lower() for c in text if c.isalpha())
        total_letters = sum(letter_counts.values())
        if total_letters > 0:
            char_entropy = 0
            for count in letter_counts.values():
                p = count / total_letters
                if p > 0:
                    char_entropy -= p * math.log2(p)
            # English text typically has entropy around 4.0-4.5
            # Max possible ~4.7 for 26 letters
            char_entropy_score = min(10, (char_entropy / 4.2) * 10)
        else:
            char_entropy_score = 0
        
        # === 3. Punctuation variety and density ===
        punct_chars = set()
        punct_count = 0
        for c in text:
            if c in '.,;:!?-()"\'"—–…':
                punct_chars.add(c)
                punct_count += 1
        
        punct_variety = len(punct_chars)
        punct_density = punct_count / max(num_words, 1)
        
        # Good punctuation density: 0.05 - 0.25
        if punct_density < 0.02:
            density_score = 2
        elif punct_density <= 0.3:
            density_score = 7 + min(3, punct_variety * 0.5)
        else:
            density_score = max(3, 8 - (punct_density - 0.3) * 10)
        
        punct_score = min(10, density_score)
        
        # === 4. Sentence length variety (standard deviation) ===
        if num_sentences >= 2:
            sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
            sent_lengths = [sl for sl in sent_lengths if sl > 0]
            if len(sent_lengths) >= 2:
                mean_sl = sum(sent_lengths) / len(sent_lengths)
                variance = sum((sl - mean_sl) ** 2 for sl in sent_lengths) / len(sent_lengths)
                std_sl = math.sqrt(variance)
                # Some variety is good (std 3-10), too much or too little is bad
                if std_sl < 1:
                    variety_score = 4
                elif std_sl <= 10:
                    variety_score = 6 + (std_sl - 1) * 0.44
                else:
                    variety_score = max(4, 10 - (std_sl - 10) * 0.3)
            else:
                variety_score = 3
        else:
            # Single sentence - can still be fine for short responses
            variety_score = 4
        
        # === 5. Hapax legomena ratio ===
        word_freq = collections.Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(len(word_freq), 1)
        # Higher hapax ratio = richer vocabulary (for reasonable text lengths)
        hapax_score = min(10, hapax_ratio * 12)
        
        # === 6. Repetition penalty ===
        # Detect repeated sentences or phrases
        repetition_penalty = 0
        
        # Check for repeated sentences
        sent_counter = collections.Counter(s.strip().lower() for s in sentences if len(s.strip()) > 10)
        for s, count in sent_counter.items():
            if count > 1:
                repetition_penalty += (count - 1) * 2
        
        # Check for repeated n-grams (trigrams)
        if len(words_lower) >= 3:
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            tri_counter = collections.Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in tri_counter.values() if c > 2)
            repetition_ratio = repeated_trigrams / max(total_trigrams, 1)
            repetition_penalty += repetition_ratio * 15
        
        repetition_penalty = min(repetition_penalty, 8)
        
        # === 7. Transition/cohesion words ===
        transition_words = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'nevertheless', 'consequently', 'meanwhile', 'although', 'because',
            'since', 'while', 'thus', 'hence', 'indeed', 'similarly',
            'specifically', 'particularly', 'notably', 'importantly',
            'first', 'second', 'third', 'finally', 'also', 'then',
            'next', 'overall', 'in', 'for', 'yet', 'but', 'so',
            'still', 'instead', 'rather', 'otherwise'
        }
        transition_count = sum(1 for w in words_lower if w in transition_words)
        transition_ratio = transition_count / max(num_words, 1)
        # Good range: 0.02 - 0.10
        if transition_ratio < 0.01:
            transition_score = 3
        elif transition_ratio <= 0.12:
            transition_score = 7 + transition_ratio * 25
        else:
            transition_score = max(5, 10 - (transition_ratio - 0.12) * 20)
        transition_score = min(10, transition_score)
        
        # === 8. Code/HTML artifact penalty ===
        artifact_penalty = 0
        # HTML tags
        html_tags = re.findall(r'<[^>]+>', text)
        if html_tags:
            artifact_penalty += min(3, len(html_tags) * 0.3)
        
        # Code patterns
        code_patterns = re.findall(r'(import |def |class |print\(|\.open\(|\.close\(|\{.*\}|```)', text)
        if code_patterns:
            # Only penalize if the query doesn't ask for code
            query_lower = query.lower() if query else ""
            if not any(kw in query_lower for kw in ['code', 'program', 'function', 'html', 'script', 'python']):
                artifact_penalty += min(3, len(code_patterns) * 0.5)
        
        # === 9. Length adequacy ===
        # Reasonable response length relative to query
        if num_words < 3:
            length_score = 2
        elif num_words < 10:
            length_score = 4
        elif num_words < 20:
            length_score = 6
        elif num_words <= 200:
            length_score = 8
        else:
            length_score = 7  # Very long might be okay but slightly penalize
        
        # === 10. Capitalization quality ===
        cap_score = 7  # default
        if sentences:
            properly_capitalized = 0
            for s in sentences:
                stripped = s.strip()
                if stripped and stripped[0].isupper():
                    properly_capitalized += 1
            cap_ratio = properly_capitalized / len(sentences)
            cap_score = 4 + cap_ratio * 6
        
        # === Combine scores ===
        # Weighted combination
        raw_score = (
            cli_score * 0.10 +
            char_entropy_score * 0.08 +
            punct_score * 0.10 +
            variety_score * 0.12 +
            hapax_score * 0.10 +
            transition_score * 0.10 +
            length_score * 0.20 +
            cap_score * 0.10 +
            # Small bonus for having multiple sentences
            min(2, (num_sentences - 1) * 0.5) * 0.10
        )
        
        # Apply penalties
        final_score = raw_score - repetition_penalty - artifact_penalty
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        try:
            # Fallback: simple length-based score
            if response and len(response.strip()) > 5:
                return 3.0
            return 1.0
        except Exception:
            return 1.0