def judging_function(query, response):
    """
    Evaluates language quality and readability using:
    - Coleman-Liau Index (character-based readability, different from Flesch which is syllable-based)
    - Punctuation correctness and variety
    - Sentence structure analysis (length variance, not just average)
    - Lexical sophistication (word length distribution, not just type-token ratio)
    - Repetition penalty (n-gram repetition detection)
    - Capitalization correctness
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        # ---- Basic tokenization ----
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", response)
        num_words = len(words)
        
        if num_words == 0:
            return 0.5
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Coleman-Liau Index (character-based, not syllable-based) ----
        num_chars = sum(len(w) for w in words)
        
        if num_words >= 3:
            L = (num_chars / num_words) * 100  # avg letters per 100 words
            S = (num_sentences / num_words) * 100  # avg sentences per 100 words
            cli = 0.0588 * L - 0.296 * S - 15.8
            # Ideal CLI for general audience: 8-12
            cli_score = max(0, 10 - abs(cli - 10) * 0.8)
            cli_score = min(cli_score, 10)
        else:
            cli_score = 2.0
        
        # ---- 2. Sentence length variance (good writing has variety) ----
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 2:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            # Coefficient of variation - normalized variance
            cv = std_dev / max(mean_len, 1)
            # Sweet spot: CV between 0.2 and 0.6 indicates good variety
            if cv < 0.05:
                variance_score = 3.0  # too uniform
            elif cv < 0.2:
                variance_score = 5.0 + (cv - 0.05) * 33.3
            elif cv <= 0.6:
                variance_score = 10.0
            else:
                variance_score = max(3.0, 10.0 - (cv - 0.6) * 10)
        elif len(sent_word_counts) == 1:
            variance_score = 5.0
        else:
            variance_score = 3.0
        
        # ---- 3. Lexical sophistication: word length distribution ----
        word_lengths = [len(w) for w in words]
        avg_word_len = sum(word_lengths) / num_words
        
        # Count words by length buckets
        short_words = sum(1 for l in word_lengths if l <= 3)  # a, the, is
        medium_words = sum(1 for l in word_lengths if 4 <= l <= 7)
        long_words = sum(1 for l in word_lengths if l >= 8)
        
        short_ratio = short_words / num_words
        medium_ratio = medium_words / num_words
        long_ratio = long_words / num_words
        
        # Good writing has a mix; too many short = simplistic, too many long = pretentious
        # Ideal: ~40% short, ~40% medium, ~20% long
        mix_score = 10.0
        if short_ratio > 0.7:
            mix_score -= (short_ratio - 0.7) * 15
        if long_ratio > 0.5:
            mix_score -= (long_ratio - 0.5) * 10
        if medium_ratio < 0.15 and num_words > 5:
            mix_score -= 2.0
        
        # Bonus for good average word length (4.5-6.0 is sophisticated but readable)
        if 4.5 <= avg_word_len <= 6.0:
            mix_score += 1.0
        
        mix_score = max(0, min(10, mix_score))
        
        # ---- 4. N-gram repetition penalty ----
        lower_words = [w.lower() for w in words]
        
        # Bigram repetition
        bigrams = []
        for i in range(len(lower_words) - 1):
            bigrams.append((lower_words[i], lower_words[i + 1]))
        
        trigrams = []
        for i in range(len(lower_words) - 2):
            trigrams.append((lower_words[i], lower_words[i + 1], lower_words[i + 2]))
        
        # Calculate repetition ratios
        if len(bigrams) > 1:
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
            bigram_rep_ratio = repeated_bigrams / len(bigrams)
        else:
            bigram_rep_ratio = 0
        
        if len(trigrams) > 1:
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / len(trigrams)
        else:
            trigram_rep_ratio = 0
        
        # Word-level repetition (beyond common words)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                       'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'and',
                       'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither',
                       'it', 'its', 'this', 'that', 'these', 'those', 'their', 'they', 'them',
                       'he', 'she', 'his', 'her', 'we', 'our', 'you', 'your', 'i', 'my', 'me'}
        
        content_words = [w for w in lower_words if w not in common_words and len(w) > 2]
        if len(content_words) > 2:
            content_unique = len(set(content_words))
            content_ttr = content_unique / len(content_words)
        else:
            content_ttr = 1.0
        
        repetition_score = 10.0
        repetition_score -= bigram_rep_ratio * 15
        repetition_score -= trigram_rep_ratio * 25
        repetition_score -= max(0, (1 - content_ttr) - 0.3) * 15  # penalize low content TTR
        repetition_score = max(0, min(10, repetition_score))
        
        # ---- 5. Punctuation and capitalization correctness ----
        punct_score = 10.0
        
        # Check sentence-initial capitalization
        if sentences:
            cap_correct = 0
            for s in sentences:
                s_stripped = s.lstrip()
                if s_stripped and s_stripped[0].isupper():
                    cap_correct += 1
            cap_ratio = cap_correct / len(sentences)
            punct_score *= (0.5 + 0.5 * cap_ratio)
        
        # Check if response ends with proper punctuation
        if response[-1] in '.!?':
            punct_score += 0  # no penalty
        else:
            punct_score -= 2.0
        
        # Punctuation variety (commas, semicolons, colons, dashes)
        has_comma = ',' in response
        has_semicolon = ';' in response
        has_colon = ':' in response
        has_dash = '—' in response or ' - ' in response or '–' in response
        has_parens = '(' in response and ')' in response
        
        punct_variety = sum([has_comma, has_semicolon, has_colon, has_dash, has_parens])
        if num_words > 15:
            punct_score += min(2.0, punct_variety * 0.5)
        
        punct_score = max(0, min(10, punct_score))
        
        # ---- 6. Response completeness (not cut off) ----
        completeness_score = 10.0
        
        # Check if response appears truncated
        if response[-1] not in '.!?")\']':
            # Might be truncated
            last_word = words[-1] if words else ""
            if len(last_word) > 0 and response[-1].isalpha():
                completeness_score -= 3.0
        
        # Check for very abrupt ending
        if num_words < 5:
            completeness_score -= 2.0
        
        # Check for excessive length without substance (padding)
        if num_words > 10:
            unique_ratio = len(set(lower_words)) / num_words
            if unique_ratio < 0.3:
                completeness_score -= 4.0
        
        completeness_score = max(0, min(10, completeness_score))
        
        # ---- 7. Content density: ratio of content words to total ----
        if num_words > 0:
            content_density = len(content_words) / num_words
            # Good range: 0.35-0.55
            if 0.30 <= content_density <= 0.60:
                density_score = 10.0
            elif content_density < 0.30:
                density_score = 5.0 + content_density * 16.7
            else:
                density_score = max(5.0, 10.0 - (content_density - 0.60) * 15)
        else:
            density_score = 3.0
        
        # ---- 8. Appropriate length bonus ----
        # Longer, well-formed responses tend to be better (from examples)
        length_bonus = 0
        if num_words >= 10:
            length_bonus = min(5.0, math.log(num_words / 10 + 1) * 3)
        
        # ---- Combine scores with weights ----
        # Different weighting than Flesch-based approach
        total = (
            cli_score * 0.12 +          # Coleman-Liau readability
            variance_score * 0.12 +      # Sentence variety
            mix_score * 0.12 +           # Word length sophistication
            repetition_score * 0.20 +    # Repetition penalty (important)
            punct_score * 0.12 +         # Punctuation/capitalization
            completeness_score * 0.15 +  # Not truncated
            density_score * 0.07 +       # Content density
            length_bonus * 0.10          # Length bonus
        )
        
        # Scale to 0-100
        final_score = total * 10
        
        # Floor for non-empty responses
        if num_words >= 3:
            final_score = max(final_score, 5.0)
        
        return round(final_score, 2)
        
    except Exception:
        try:
            if response and len(response.strip()) > 0:
                return 25.0
            return 0.0
        except Exception:
            return 0.0