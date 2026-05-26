def judging_function(query, response):
    """
    Evaluate language quality and readability using a substantially different approach:
    - Coleman-Liau Index (instead of Flesch)
    - Automated Readability Index (ARI)
    - Punctuation diversity and correctness
    - Sentence structure variation (std dev of sentence lengths)
    - Lexical sophistication (longer unique words ratio)
    - Repetition penalty (consecutive repeated words/phrases)
    - Discourse markers and cohesion signals
    - Character-level entropy for text richness
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) == 0:
            return 0.0
        
        # Basic tokenization
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        num_words = len(words)
        
        if num_words == 0:
            return 0.5
        
        # Very short responses get penalized
        if num_words < 3:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        # --- 1. Coleman-Liau Index ---
        # CLI = 0.0588*L - 0.296*S - 15.8
        # L = avg number of letters per 100 words
        # S = avg number of sentences per 100 words
        num_letters = sum(len(w) for w in words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        L = (num_letters / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        # Ideal CLI for general audience: 8-12
        cli_score = max(0, 10 - abs(cli - 10) * 0.8)
        cli_score = min(cli_score, 10)
        
        # --- 2. Automated Readability Index ---
        num_chars = sum(len(w) for w in words)
        ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        # Ideal ARI: 7-12
        ari_score = max(0, 10 - abs(ari - 9.5) * 0.7)
        ari_score = min(ari_score, 10)
        
        # --- 3. Sentence length variation (coefficient of variation) ---
        sent_words = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s)
            if sw:
                sent_words.append(len(sw))
        
        if len(sent_words) >= 2:
            mean_sl = sum(sent_words) / len(sent_words)
            variance = sum((x - mean_sl) ** 2 for x in sent_words) / len(sent_words)
            std_sl = math.sqrt(variance)
            cv = std_sl / max(mean_sl, 1)
            # Some variation is good (cv 0.2-0.6), too much or too little is bad
            if cv < 0.05:
                variation_score = 3.0
            elif cv < 0.15:
                variation_score = 5.0
            elif cv < 0.5:
                variation_score = 8.0 + 2.0 * min((cv - 0.15) / 0.35, 1.0)
            elif cv < 0.8:
                variation_score = 8.0
            else:
                variation_score = max(4.0, 8.0 - (cv - 0.8) * 5)
        else:
            variation_score = 4.0  # Single sentence, limited variation
        
        # --- 4. Punctuation diversity ---
        punct_chars = re.findall(r'[,;:\-\(\)\"\'!?\.]', text)
        punct_types = set(punct_chars)
        punct_diversity = min(len(punct_types) / 5.0, 1.0) * 10
        
        # Comma usage rate (good writing uses commas appropriately)
        comma_count = text.count(',')
        comma_rate = comma_count / max(num_sentences, 1)
        # Ideal: 0.5-2 commas per sentence
        if 0.3 <= comma_rate <= 2.5:
            comma_score = 8.0
        elif comma_rate < 0.3:
            comma_score = 4.0 + comma_rate * 10
        else:
            comma_score = max(3.0, 8.0 - (comma_rate - 2.5) * 2)
        
        punct_score = (punct_diversity * 0.6 + comma_score * 0.4)
        
        # --- 5. Repetition penalty (bigram and trigram repetition) ---
        # Count repeated consecutive bigrams
        bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
        trigrams = [(words_lower[i], words_lower[i+1], words_lower[i+2]) for i in range(len(words_lower)-2)]
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Fraction of bigrams that appear more than twice
        if bigrams:
            repeated_bigrams = sum(1 for bg, c in bigram_counts.items() if c > 2)
            bigram_rep_ratio = repeated_bigrams / len(bigram_counts) if bigram_counts else 0
        else:
            bigram_rep_ratio = 0
        
        if trigrams:
            repeated_trigrams = sum(1 for tg, c in trigram_counts.items() if c > 1)
            trigram_rep_ratio = repeated_trigrams / len(trigram_counts) if trigram_counts else 0
        else:
            trigram_rep_ratio = 0
        
        # Also check for word-level repetition (same word repeated consecutively)
        consecutive_repeats = 0
        for i in range(1, len(words_lower)):
            if words_lower[i] == words_lower[i-1]:
                consecutive_repeats += 1
        consec_ratio = consecutive_repeats / max(num_words, 1)
        
        repetition_penalty = (bigram_rep_ratio * 3 + trigram_rep_ratio * 4 + consec_ratio * 3) * 10
        repetition_score = max(0, 10 - repetition_penalty)
        
        # --- 6. Lexical sophistication ---
        # Ratio of words with 7+ characters (more sophisticated vocabulary)
        sophisticated_words = [w for w in words_lower if len(w) >= 7]
        sophistication_ratio = len(sophisticated_words) / max(num_words, 1)
        # Ideal: 15-35%
        if sophistication_ratio < 0.05:
            sophistication_score = 3.0
        elif sophistication_ratio < 0.15:
            sophistication_score = 5.0 + (sophistication_ratio - 0.05) * 50
        elif sophistication_ratio <= 0.35:
            sophistication_score = 10.0
        elif sophistication_ratio <= 0.5:
            sophistication_score = 10.0 - (sophistication_ratio - 0.35) * 20
        else:
            sophistication_score = max(3.0, 7.0 - (sophistication_ratio - 0.5) * 10)
        
        # --- 7. Character-level entropy ---
        char_counts = Counter(text.lower())
        total_chars = len(text)
        char_entropy = 0
        for ch, count in char_counts.items():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Good text typically has entropy 4.0-5.0
        if char_entropy < 2.0:
            entropy_score = 2.0
        elif char_entropy < 3.5:
            entropy_score = 4.0 + (char_entropy - 2.0) * 3
        elif char_entropy <= 5.0:
            entropy_score = 9.0 + min((char_entropy - 3.5) / 1.5, 1.0)
        else:
            entropy_score = max(6.0, 10.0 - (char_entropy - 5.0) * 2)
        
        # --- 8. Discourse and cohesion markers ---
        cohesion_markers = [
            r'\bmoreover\b', r'\bfurthermore\b', r'\bhowever\b', r'\btherefore\b',
            r'\bin addition\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bas a result\b',
            r'\bconsequently\b', r'\bnevertheless\b', r'\bmeanwhile\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bthat is\b',
            r'\balso\b', r'\bwhile\b', r'\balthough\b', r'\bsince\b',
            r'\bbecause\b', r'\bthus\b', r'\bhence\b', r'\byet\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bwhich\b',
            r'\bin other words\b', r'\boverall\b', r'\bfinally\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
        ]
        
        text_lower = text.lower()
        marker_count = 0
        unique_markers = 0
        for pattern in cohesion_markers:
            matches = re.findall(pattern, text_lower)
            if matches:
                unique_markers += 1
                marker_count += len(matches)
        
        marker_density = marker_count / max(num_sentences, 1)
        marker_variety = unique_markers
        
        # Score: having some cohesion markers is good
        if marker_count == 0:
            cohesion_score = 3.0
        elif marker_density <= 0.5:
            cohesion_score = 5.0 + marker_variety * 0.5
        elif marker_density <= 2.0:
            cohesion_score = 7.0 + min(marker_variety * 0.3, 3.0)
        else:
            cohesion_score = max(5.0, 8.0 - (marker_density - 2.0) * 1.5)
        cohesion_score = min(cohesion_score, 10.0)
        
        # --- 9. Sentence beginning diversity ---
        if len(sentences) >= 2:
            first_words = []
            for s in sentences:
                sw = re.findall(r"[a-zA-Z]+", s)
                if sw:
                    first_words.append(sw[0].lower())
            if first_words:
                unique_starts = len(set(first_words))
                start_diversity = unique_starts / len(first_words)
                start_score = start_diversity * 10
            else:
                start_score = 5.0
        else:
            start_score = 5.0
        
        # --- 10. Response completeness (ends properly) ---
        completeness_score = 5.0
        stripped = text.rstrip()
        if stripped and stripped[-1] in '.!?"\'':
            completeness_score = 10.0
        elif stripped and stripped[-1] in ',;:':
            completeness_score = 3.0  # Truncated
        else:
            completeness_score = 4.0
        
        # --- 11. Length appropriateness ---
        # Reward moderate length, penalize very short or excessively long
        if num_words < 5:
            length_score = 2.0
        elif num_words < 10:
            length_score = 4.0
        elif num_words < 20:
            length_score = 6.0
        elif num_words <= 100:
            length_score = 8.0 + min((num_words - 20) / 80, 1.0) * 2
        elif num_words <= 300:
            length_score = 9.0
        else:
            length_score = max(6.0, 9.0 - (num_words - 300) / 200)
        
        # --- 12. Type-Token Ratio (unique approach: use hapax legomena ratio) ---
        word_freq = Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(num_words, 1)
        # Good writing: 40-70% hapax
        if hapax_ratio < 0.2:
            hapax_score = 3.0
        elif hapax_ratio < 0.4:
            hapax_score = 5.0 + (hapax_ratio - 0.2) * 25
        elif hapax_ratio <= 0.75:
            hapax_score = 10.0
        else:
            hapax_score = max(6.0, 10.0 - (hapax_ratio - 0.75) * 16)
        
        # --- Combine all scores with weights ---
        weights = {
            'cli': 0.08,
            'ari': 0.07,
            'variation': 0.10,
            'punct': 0.08,
            'repetition': 0.14,
            'sophistication': 0.08,
            'entropy': 0.07,
            'cohesion': 0.10,
            'start_diversity': 0.06,
            'completeness': 0.08,
            'length': 0.08,
            'hapax': 0.06,
        }
        
        scores = {
            'cli': cli_score,
            'ari': ari_score,
            'variation': variation_score,
            'punct': punct_score,
            'repetition': repetition_score,
            'sophistication': sophistication_score,
            'entropy': entropy_score,
            'cohesion': cohesion_score,
            'start_diversity': start_score,
            'completeness': completeness_score,
            'length': length_score,
            'hapax': hapax_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-100 for better discrimination
        final_score = final_score * 10
        
        return round(max(0.0, min(100.0, final_score)), 2)
    
    except Exception:
        return 5.0