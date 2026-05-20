def judging_function(query, response):
    """
    Evaluate language quality and readability using:
    - Coleman-Liau Index (character-based readability, different from Flesch)
    - Gunning Fog Index approximation
    - Type-token ratio with hapax legomena ratio
    - Punctuation variety and correctness
    - Sentence structure diversity (std dev of sentence lengths)
    - Discourse markers and cohesion signals
    - Error detection (repeated words, capitalization issues)
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        # ---- Basic tokenization ----
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        num_words = max(len(words), 1)
        
        # Character count (letters and digits only)
        num_chars = sum(1 for c in text if c.isalpha())
        
        # ---- 1. Coleman-Liau Index ----
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg letters per 100 words, S = avg sentences per 100 words
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        # Ideal range ~8-14 for general audience; penalize extremes
        cli_score = max(0, 10 - abs(coleman_liau - 11) * 0.8)
        
        # ---- 2. Gunning Fog approximation ----
        # Complex words: words with 3+ syllables (approximated by length >= 7)
        complex_words = sum(1 for w in words if len(w) >= 7)
        avg_sentence_len = num_words / num_sentences
        fog_index = 0.4 * (avg_sentence_len + 100 * (complex_words / num_words))
        # Ideal ~10-15
        fog_score = max(0, 10 - abs(fog_index - 12) * 0.5)
        
        # ---- 3. Lexical richness: hapax legomena ratio ----
        words_lower = [w.lower() for w in words]
        freq = collections.Counter(words_lower)
        hapax = sum(1 for w, c in freq.items() if c == 1)
        hapax_ratio = hapax / num_words if num_words > 0 else 0
        # Also type-token ratio but using root TTR (Guiraud's index)
        types = len(freq)
        guiraud = types / math.sqrt(num_words) if num_words > 0 else 0
        # Guiraud typically 4-10 for good text
        guiraud_score = min(10, guiraud * 1.2)
        hapax_score = min(10, hapax_ratio * 15)  # higher hapax = richer vocab
        lexical_score = (guiraud_score * 0.6 + hapax_score * 0.4)
        
        # ---- 4. Sentence length diversity (std dev) ----
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
            cv = std_sl / mean_sl if mean_sl > 0 else 0
            # Some variation is good (cv ~0.3-0.7), too little or too much is bad
            diversity_score = max(0, 10 - abs(cv - 0.5) * 12)
        else:
            diversity_score = 3.0  # single sentence gets moderate score
        
        # ---- 5. Punctuation variety and density ----
        punct_chars = set()
        punct_count = 0
        for c in text:
            if c in '.,;:!?—–-()[]{}"\'/':
                punct_chars.add(c)
                punct_count += 1
        
        punct_variety = len(punct_chars)
        punct_density = punct_count / num_words if num_words > 0 else 0
        # Good punct density ~0.1-0.3, variety 3-8
        punct_variety_score = min(10, punct_variety * 1.5)
        punct_density_score = max(0, 10 - abs(punct_density - 0.2) * 25)
        punctuation_score = punct_variety_score * 0.5 + punct_density_score * 0.5
        
        # ---- 6. Discourse markers and cohesion ----
        cohesion_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\btherefore\b', r'\bthus\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bnevertheless\b',
            r'\bmeanwhile\b', r'\bsimilarly\b', r'\blikewise\b',
            r'\bin particular\b', r'\bnotably\b', r'\bindeed\b',
            r'\bwhile\b', r'\balthough\b', r'\bwhereas\b',
            r'\bfirst\b', r'\bsecond\b', r'\bfinally\b', r'\blastly\b',
            r'\bessentially\b', r'\bultimately\b', r'\boverall\b',
            r'\bin other words\b', r'\bthat is\b', r'\bnamely\b',
            r'\btypically\b', r'\bgenerally\b', r'\busually\b',
        ]
        
        text_lower = text.lower()
        marker_count = 0
        unique_markers = 0
        for pattern in cohesion_markers:
            matches = re.findall(pattern, text_lower)
            if matches:
                unique_markers += 1
                marker_count += len(matches)
        
        # Normalize by number of sentences
        marker_density = marker_count / num_sentences if num_sentences > 0 else 0
        cohesion_score = min(10, unique_markers * 1.5 + marker_density * 2)
        
        # ---- 7. Error detection ----
        error_penalty = 0
        
        # Repeated adjacent words (e.g., "the the")
        repeated = re.findall(r'\b(\w+)\s+\1\b', text_lower)
        error_penalty += len(repeated) * 1.5
        
        # Sentences not starting with uppercase (after first)
        for s in sentences[1:]:
            if s and s[0].islower():
                error_penalty += 0.5
        
        # Double spaces
        double_spaces = len(re.findall(r'  +', text))
        error_penalty += double_spaces * 0.3
        
        # Missing space after punctuation
        missing_space = len(re.findall(r'[.!?,;:][a-zA-Z]', text))
        error_penalty += missing_space * 0.5
        
        # Common misspelling patterns (double letters that look wrong)
        # Simple: check for unusual character trigrams
        # Skip heavy spell-check; just penalize obvious patterns
        
        error_score = max(0, 10 - error_penalty)
        
        # ---- 8. Response length bonus (longer, substantive responses tend to be better) ----
        # Logarithmic scaling to avoid over-rewarding very long responses
        length_factor = min(10, math.log(num_words + 1, 2) * 1.0)
        # Penalize very short responses
        if num_words < 15:
            length_factor *= 0.5
        elif num_words < 30:
            length_factor *= 0.75
        
        # ---- 9. Word length distribution (entropy-based) ----
        word_lengths = [len(w) for w in words]
        wl_freq = collections.Counter(word_lengths)
        total_wl = sum(wl_freq.values())
        wl_entropy = 0
        for count in wl_freq.values():
            p = count / total_wl
            if p > 0:
                wl_entropy -= p * math.log2(p)
        # Higher entropy = more diverse word lengths, typically 2.5-3.5 for good text
        wl_entropy_score = min(10, wl_entropy * 3.0)
        
        # ---- 10. Sentence opening diversity ----
        if len(sentences) >= 3:
            openers = []
            for s in sentences:
                first_word = re.match(r'[a-zA-Z]+', s)
                if first_word:
                    openers.append(first_word.group().lower())
            if openers:
                opener_types = len(set(openers))
                opener_diversity = opener_types / len(openers)
                opener_score = opener_diversity * 10
            else:
                opener_score = 5
        else:
            opener_score = 5
        
        # ---- Weighted combination ----
        weights = {
            'cli': 0.10,
            'fog': 0.08,
            'lexical': 0.14,
            'diversity': 0.10,
            'punctuation': 0.08,
            'cohesion': 0.12,
            'errors': 0.10,
            'length': 0.12,
            'wl_entropy': 0.08,
            'opener': 0.08,
        }
        
        scores = {
            'cli': cli_score,
            'fog': fog_score,
            'lexical': lexical_score,
            'diversity': diversity_score,
            'punctuation': punctuation_score,
            'cohesion': cohesion_score,
            'errors': error_score,
            'length': length_factor,
            'wl_entropy': wl_entropy_score,
            'opener': opener_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = max(0, min(10, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            return min(10, len(response.split()) / 15.0)
        except Exception:
            return 1.0