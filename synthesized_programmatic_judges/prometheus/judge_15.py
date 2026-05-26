def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique combination of:
    - Punctuation diversity and correctness
    - Sentence structure variation (std dev of sentence lengths)
    - Lexical sophistication (longer unique words ratio)
    - Discourse coherence (connective/cohesion markers)
    - Repetition penalty (bigram/trigram repetition)
    - Formality and politeness markers
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+", response)
        if len(words) < 3:
            return 0.5
        
        words_lower = [w.lower() for w in words]
        num_words = len(words_lower)
        
        # 1. Punctuation diversity and density score (0-10)
        # Measures how well the response uses varied punctuation
        punct_chars = re.findall(r'[.,;:!?\-\(\)\"\'…—]', response)
        punct_types = set(punct_chars)
        punct_diversity = min(len(punct_types) / 5.0, 1.0)  # normalize by 5 types
        punct_density = len(punct_chars) / max(num_words, 1)
        # Good punct density is roughly 0.1-0.3
        punct_density_score = 1.0 - min(abs(punct_density - 0.18) / 0.18, 1.0)
        punctuation_score = (punct_diversity * 0.6 + punct_density_score * 0.4) * 10
        
        # 2. Sentence structure variation (0-10)
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        if len(sentences) >= 2:
            sent_lengths = [len(re.findall(r'\S+', s)) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Good variation: std dev around 4-8 words
            variation_score = min(std_sl / 6.0, 1.0) * 10
            # Penalize very short or very long average sentences
            avg_len_score = 1.0 - min(abs(mean_sl - 16) / 16, 1.0)
            sentence_score = variation_score * 0.5 + avg_len_score * 10 * 0.5
        else:
            sentence_score = 2.0  # single sentence gets low score
        
        # 3. Lexical sophistication (0-10)
        # Ratio of words with 3+ syllables (approximated by length >= 7)
        sophisticated_words = [w for w in words_lower if len(w) >= 7]
        sophistication_ratio = len(sophisticated_words) / max(num_words, 1)
        # Good ratio: 0.1-0.25
        if sophistication_ratio < 0.05:
            lex_score = sophistication_ratio / 0.05 * 4
        elif sophistication_ratio <= 0.30:
            lex_score = 4 + (sophistication_ratio - 0.05) / 0.25 * 6
        else:
            lex_score = max(10 - (sophistication_ratio - 0.30) * 20, 4)
        
        # Type-token ratio on unique sophisticated words
        unique_soph = set(sophisticated_words)
        if len(sophisticated_words) > 0:
            soph_ttr = len(unique_soph) / len(sophisticated_words)
        else:
            soph_ttr = 0
        lex_score = lex_score * 0.7 + soph_ttr * 10 * 0.3
        
        # 4. Discourse coherence markers (0-10)
        # Check for cohesive devices: conjunctions, discourse markers, anaphora
        cohesion_markers = [
            r'\bhowever\b', r'\btherefore\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\bfor instance\b', r'\bfor example\b',
            r'\bon the other hand\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bnevertheless\b', r'\bmeanwhile\b', r'\bspecifically\b',
            r'\bin fact\b', r'\bindeed\b', r'\bthus\b', r'\bhence\b',
            r'\balthough\b', r'\bwhile\b', r'\bsince\b', r'\bbecause\b',
            r'\bthat said\b', r'\bin other words\b', r'\bultimately\b',
            r'\bfirst\b', r'\bsecond\b', r'\bnext\b', r'\bfinally\b',
            r'\badditionally\b', r'\bsimilarly\b', r'\blikewise\b',
            r'\bthis\b', r'\bthese\b', r'\bthat\b', r'\bsuch\b',
        ]
        
        response_lower = response.lower()
        marker_count = 0
        unique_markers = 0
        for pattern in cohesion_markers:
            matches = re.findall(pattern, response_lower)
            if matches:
                unique_markers += 1
                marker_count += len(matches)
        
        marker_density = marker_count / max(num_words, 1)
        marker_variety = unique_markers / max(len(cohesion_markers) * 0.15, 1)
        coherence_score = min((marker_density * 30 + min(marker_variety, 1.0) * 5), 10)
        
        # 5. Repetition penalty (0-10, higher = less repetition = better)
        # Bigram repetition
        bigrams = [tuple(words_lower[i:i+2]) for i in range(len(words_lower)-1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
            bigram_rep_ratio = repeated_bigrams / max(len(set(bigrams)), 1)
        else:
            bigram_rep_ratio = 0
        
        # Trigram repetition
        trigrams = [tuple(words_lower[i:i+3]) for i in range(len(words_lower)-2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / max(len(set(trigrams)), 1)
        else:
            trigram_rep_ratio = 0
        
        # Word-level repetition (excluding common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'no', 'if', 'it', 'its', 'this',
                      'that', 'these', 'those', 'i', 'you', 'your', 'we', 'they',
                      'he', 'she', 'my', 'me', 'our', 'their', 'his', 'her', 'them',
                      'as', 'so', 'up', 'out', 'about', 'just', 'than', 'then',
                      'also', 'very', 'more', 'some', 'any', 'all', 'each', 'every'}
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 2]
        if content_words:
            content_ttr = len(set(content_words)) / len(content_words)
        else:
            content_ttr = 1.0
        
        repetition_score = (1.0 - bigram_rep_ratio * 0.4 - trigram_rep_ratio * 0.6) * 5 + content_ttr * 5
        repetition_score = max(min(repetition_score, 10), 0)
        
        # 6. Empathy / Engagement / Politeness markers (0-10)
        # Contextual quality: does the response engage with the query appropriately?
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r"\bi'm sorry\b",
            r'\bunderstandable\b', r'\bcompletely\b', r'\babsolutely\b',
            r'\bperfectly\b', r'\bof course\b', r'\blet\'s\b', r'\blet us\b',
            r'\bplease\b', r'\bthank\b', r'\bappreciate\b', r'\bhappy to\b',
            r'\bglad\b', r'\bwelcome\b', r'\bfeel free\b', r'\bdon\'t hesitate\b',
            r'\bhere to help\b', r'\bremember\b', r'\bimportant\b',
            r'\bit\'s okay\b', r"\bit's natural\b", r"\bit's completely\b",
        ]
        
        empathy_count = 0
        for pattern in empathy_patterns:
            if re.search(pattern, response_lower):
                empathy_count += 1
        
        empathy_score = min(empathy_count / 3.0, 1.0) * 10
        
        # 7. Structural organization (0-10)
        # Check for numbered lists, colons, paragraph breaks
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[.)]\s', response))
        has_paragraphs = response.count('\n\n') >= 1
        has_colons = ':' in response
        
        # Opening and closing quality
        first_sentence = sentences[0] if sentences else ""
        first_words = re.findall(r'\S+', first_sentence)
        has_good_opening = len(first_words) >= 4
        
        structure_score = 0
        if has_numbering:
            structure_score += 3
        if has_paragraphs:
            structure_score += 3
        if has_colons:
            structure_score += 1.5
        if has_good_opening:
            structure_score += 2.5
        structure_score = min(structure_score, 10)
        
        # 8. Capitalization and basic grammar signals (0-10)
        # Check sentence-initial capitalization
        cap_correct = 0
        cap_total = 0
        for s in sentences:
            s = s.strip()
            if s:
                cap_total += 1
                if s[0].isupper():
                    cap_correct += 1
        
        cap_ratio = cap_correct / max(cap_total, 1)
        
        # Check for common grammar issues
        grammar_issues = 0
        # Double spaces
        grammar_issues += len(re.findall(r'  +', response)) * 0.5
        # Missing space after punctuation
        grammar_issues += len(re.findall(r'[.!?,;][A-Za-z]', response))
        # Repeated words
        grammar_issues += len(re.findall(r'\b(\w+)\s+\1\b', response_lower))
        
        grammar_penalty = min(grammar_issues / 5.0, 1.0)
        grammar_score = (cap_ratio * 0.5 + (1.0 - grammar_penalty) * 0.5) * 10
        
        # 9. Response length appropriateness (0-10)
        # Very short or extremely long responses are penalized
        if num_words < 20:
            length_score = num_words / 20 * 4
        elif num_words < 40:
            length_score = 4 + (num_words - 20) / 20 * 3
        elif num_words <= 200:
            length_score = 7 + min((num_words - 40) / 160 * 3, 3)
        else:
            length_score = max(10 - (num_words - 200) / 200 * 3, 5)
        
        # Weighted combination
        final_score = (
            punctuation_score * 0.08 +
            sentence_score * 0.15 +
            lex_score * 0.12 +
            coherence_score * 0.15 +
            repetition_score * 0.12 +
            empathy_score * 0.10 +
            structure_score * 0.10 +
            grammar_score * 0.10 +
            length_score * 0.08
        )
        
        # Scale to 1-5 range to match the examples
        # Current range is roughly 0-10, map to 1-5
        scaled_score = 1.0 + (final_score / 10.0) * 4.0
        scaled_score = max(1.0, min(5.0, scaled_score))
        
        return round(scaled_score, 2)
        
    except Exception:
        return 2.5