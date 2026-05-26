def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, structural quality,
    and communication efficiency metrics.
    
    This variant focuses on:
    - Information density (unique content words per total words ratio)
    - Repetition penalty using sliding window duplicate detection
    - Sentence structure quality (clause complexity analysis)
    - Response adequacy relative to query
    - Filler phrase detection
    - Entropy-based content richness
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            query = ""
        
        response_text = response.strip()
        query_text = query.strip()
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response_text.lower())
        query_words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', query_text.lower())
        
        if len(words) == 0:
            return 1.0
        
        # --- Feature 1: Character-level entropy (measures information richness) ---
        char_counts = Counter(response_text.lower())
        total_chars = len(response_text.lower())
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Normalize: English text typically has entropy 3.5-4.5
        entropy_score = min(char_entropy / 4.5, 1.0)
        
        # --- Feature 2: Sliding window repetition detection ---
        # Look for repeated sequences of 3+ words
        repetition_penalty = 0.0
        if len(words) >= 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            repetition_penalty = min(repetition_ratio * 3.0, 1.0)
        
        # Also check for repeated longer phrases (5-grams)
        if len(words) >= 10:
            fivegrams = [tuple(words[i:i+5]) for i in range(len(words) - 4)]
            fivegram_counts = Counter(fivegrams)
            repeated_5g = sum(c - 1 for c in fivegram_counts.values() if c > 1)
            rep5_ratio = repeated_5g / max(len(fivegrams), 1)
            repetition_penalty = min(repetition_penalty + rep5_ratio * 5.0, 1.0)
        
        # --- Feature 3: Filler and hedge phrase detection ---
        filler_patterns = [
            r'\bin many ways\b', r'\bit is important to note\b',
            r'\bit should be noted\b', r'\bas a matter of fact\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bat the end of the day\b', r'\bfor the purpose of\b',
            r'\bin terms of\b', r'\bwith regard to\b',
            r'\bit goes without saying\b', r'\bneedless to say\b',
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b',
            r'\breally\b', r'\bjust\b', r'\bvery\b',
            r'\bsort of\b', r'\bkind of\b',
        ]
        filler_count = 0
        response_lower = response_text.lower()
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, response_lower))
        filler_ratio = filler_count / max(len(words), 1)
        filler_penalty = min(filler_ratio * 8.0, 0.5)
        
        # --- Feature 4: Sentence quality analysis ---
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Average words per sentence - ideal range 10-25
        avg_words_per_sent = len(words) / num_sentences
        if avg_words_per_sent < 3:
            sentence_length_score = 0.3
        elif avg_words_per_sent < 8:
            sentence_length_score = 0.6
        elif avg_words_per_sent <= 25:
            sentence_length_score = 1.0
        elif avg_words_per_sent <= 40:
            sentence_length_score = 0.7
        else:
            sentence_length_score = 0.4
        
        # Sentence length variance - some variety is good but not too much
        if num_sentences > 1:
            sent_lengths = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
            cv = math.sqrt(variance) / max(mean_sl, 1)  # coefficient of variation
            # Moderate variation (0.2-0.6) is ideal
            if 0.15 <= cv <= 0.7:
                variety_score = 1.0
            elif cv < 0.15:
                variety_score = 0.7  # too uniform
            else:
                variety_score = max(0.4, 1.0 - (cv - 0.7) * 0.5)
        else:
            variety_score = 0.7
        
        # --- Feature 5: Content word density ---
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
            'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
            'them', 'their', 'theirs', 'themselves', 'also', 'while'
        }
        content_words = [w for w in words if w not in stop_words]
        unique_content_words = set(content_words)
        
        content_density = len(content_words) / max(len(words), 1)
        # Ideal density around 0.4-0.6
        if 0.35 <= content_density <= 0.65:
            density_score = 1.0
        elif content_density < 0.35:
            density_score = 0.5 + content_density
        else:
            density_score = max(0.6, 1.0 - (content_density - 0.65) * 2)
        
        # --- Feature 6: Unique content ratio (penalize word-level repetition) ---
        if len(content_words) > 0:
            unique_content_ratio = len(unique_content_words) / len(content_words)
        else:
            unique_content_ratio = 0.5
        # Scale: very low ratio = heavy repetition
        content_uniqueness_score = min(unique_content_ratio * 1.2, 1.0)
        
        # --- Feature 7: Response adequacy / relevance to query ---
        query_content = set(w for w in query_words if w not in stop_words and len(w) > 2)
        response_content = set(w for w in words if w not in stop_words and len(w) > 2)
        
        if query_content:
            # How many query content words appear in response
            coverage = len(query_content & response_content) / len(query_content)
            # New information beyond query
            if response_content:
                novelty = len(response_content - query_content) / len(response_content)
            else:
                novelty = 0.0
            relevance_score = 0.5 * min(coverage * 1.5, 1.0) + 0.5 * min(novelty * 1.3, 1.0)
        else:
            relevance_score = 0.7  # neutral if no query content
        
        # --- Feature 8: Response length adequacy ---
        # Not too short (empty/trivial), not excessively long
        word_count = len(words)
        if word_count < 3:
            length_score = 0.15
        elif word_count < 8:
            length_score = 0.4
        elif word_count < 15:
            length_score = 0.7
        elif word_count <= 150:
            length_score = 1.0
        elif word_count <= 300:
            length_score = 0.85
        else:
            length_score = max(0.5, 0.85 - (word_count - 300) / 1000)
        
        # --- Feature 9: Consecutive duplicate word detection ---
        consec_dup_count = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1] and words[i] not in stop_words:
                consec_dup_count += 1
        consec_penalty = min(consec_dup_count / max(len(words), 1) * 10, 0.5)
        
        # --- Feature 10: Sentence-level semantic repetition ---
        # Compare sentences pairwise using content word overlap
        sent_repetition_penalty = 0.0
        if num_sentences > 1:
            sent_content_sets = []
            for s in sentences:
                s_words = set(re.findall(r'[a-zA-Z]+', s.lower())) - stop_words
                sent_content_sets.append(s_words)
            
            high_overlap_count = 0
            pair_count = 0
            for i in range(len(sent_content_sets)):
                for j in range(i + 1, len(sent_content_sets)):
                    if sent_content_sets[i] and sent_content_sets[j]:
                        intersection = len(sent_content_sets[i] & sent_content_sets[j])
                        smaller = min(len(sent_content_sets[i]), len(sent_content_sets[j]))
                        if smaller > 0:
                            overlap = intersection / smaller
                            if overlap > 0.7:
                                high_overlap_count += 1
                        pair_count += 1
            
            if pair_count > 0:
                sent_repetition_penalty = min(high_overlap_count / pair_count, 1.0) * 0.4
        
        # --- Combine all features ---
        score = (
            entropy_score * 10 +          # 0-10: information richness
            sentence_length_score * 12 +   # 0-12: sentence structure
            variety_score * 8 +            # 0-8: sentence variety
            density_score * 10 +           # 0-10: content density
            content_uniqueness_score * 15 + # 0-15: vocabulary freshness
            relevance_score * 15 +         # 0-15: query relevance
            length_score * 15 +            # 0-15: adequate length
            - repetition_penalty * 25 +    # 0-25: trigram/5gram repetition
            - filler_penalty * 10 +        # 0-5: filler phrases
            - consec_penalty * 10 +        # 0-5: consecutive duplicates
            - sent_repetition_penalty * 15 # 0-6: sentence-level repetition
        )
        
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        # Fallback: return a middling score
        try:
            if response and response.strip():
                return 30.0
            return 0.0
        except Exception:
            return 0.0