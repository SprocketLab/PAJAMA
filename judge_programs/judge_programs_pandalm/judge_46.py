def judging_function(query, response):
    """
    Evaluate clarity and conciseness using a structural coherence approach.
    
    This variant focuses on:
    1. Compression ratio (response length relative to query complexity)
    2. Unique information density (ratio of unique content words to total words)
    3. Repetition detection at character/substring level (not word/n-gram overlap)
    4. Syntactic complexity via punctuation patterns and clause structure
    5. Directness scoring (how quickly the response addresses the query topic)
    6. Filler phrase detection with weighted penalties
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # Tokenize
        resp_words = re.findall(r'[a-zA-Z]+', response.lower())
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        
        if len(resp_words) == 0:
            return 1.0
        
        total_words = len(resp_words)
        
        # === 1. Compression Ratio Score ===
        # Reward responses that are substantive but not bloated
        # Ideal length depends on query complexity
        query_complexity = max(len(query_words), 1)
        # Ideal response is roughly 3-8x the query length
        ratio = total_words / query_complexity
        if ratio < 1:
            compression_score = 3.0  # Too short
        elif ratio <= 3:
            compression_score = 5.0 + (ratio - 1) * 2.0  # Building up
        elif ratio <= 10:
            compression_score = 9.0  # Sweet spot
        elif ratio <= 20:
            compression_score = 9.0 - (ratio - 10) * 0.3  # Getting verbose
        else:
            compression_score = max(2.0, 6.0 - (ratio - 20) * 0.15)
        compression_score = max(0, min(10, compression_score))
        
        # === 2. Unique Information Density ===
        # Ratio of unique words to total words (type-token ratio)
        # But adjusted for text length using Guiraud's index: unique / sqrt(total)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her',
            'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'also',
            'about', 'up', 'there', 'here', 'while'
        }
        
        content_words = [w for w in resp_words if w not in stop_words and len(w) > 2]
        if len(content_words) > 0:
            unique_content = set(content_words)
            # Guiraud's index normalized
            guiraud = len(unique_content) / math.sqrt(len(content_words))
            # Typical range 2-8, normalize to 0-10
            density_score = min(10, guiraud * 1.5)
        else:
            density_score = 2.0
        
        # === 3. Substring Repetition Detection ===
        # Use longest repeated substring approach via suffix analysis
        # Check for repeated phrases (4+ word sequences)
        def get_phrase_repetitions(words, min_len=3, max_len=8):
            """Count repeated phrases of various lengths."""
            total_repeated_words = 0
            seen_phrases = set()
            for plen in range(min_len, min(max_len + 1, len(words))):
                phrase_counts = Counter()
                for i in range(len(words) - plen + 1):
                    phrase = tuple(words[i:i+plen])
                    phrase_counts[phrase] += 1
                for phrase, count in phrase_counts.items():
                    if count > 1:
                        phrase_str = ' '.join(phrase)
                        if phrase_str not in seen_phrases:
                            seen_phrases.add(phrase_str)
                            # Each extra occurrence contributes repeated words
                            total_repeated_words += (count - 1) * plen
            return total_repeated_words
        
        repeated_words = get_phrase_repetitions(resp_words)
        repetition_ratio = repeated_words / max(total_words, 1)
        # Penalize heavily for high repetition
        repetition_score = max(0, 10 - repetition_ratio * 25)
        
        # Also check character-level repetition (e.g., "Reduced-Sized Reduced-Sized...")
        # Look for repeated character sequences
        char_rep_penalty = 0
        resp_lower = response.lower()
        for chunk_size in [20, 30, 50]:
            if len(resp_lower) > chunk_size * 2:
                chunks = []
                for i in range(0, len(resp_lower) - chunk_size + 1, chunk_size // 2):
                    chunks.append(resp_lower[i:i+chunk_size])
                chunk_counts = Counter(chunks)
                for chunk, cnt in chunk_counts.items():
                    if cnt > 2:
                        char_rep_penalty += (cnt - 2) * 0.5
        
        repetition_score = max(0, repetition_score - char_rep_penalty)
        
        # === 4. Syntactic Clarity via Clause Structure ===
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Average words per sentence
        sent_lengths = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            if s_words:
                sent_lengths.append(len(s_words))
        
        if sent_lengths:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            # Variance in sentence length (some variety is good, too much is bad)
            if len(sent_lengths) > 1:
                variance = sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / max(avg_sent_len, 1)  # coefficient of variation
            else:
                cv = 0
            
            # Ideal avg sentence length: 10-25 words
            if avg_sent_len < 5:
                length_clarity = 4.0
            elif avg_sent_len <= 10:
                length_clarity = 5.0 + (avg_sent_len - 5) * 0.8
            elif avg_sent_len <= 25:
                length_clarity = 9.0
            elif avg_sent_len <= 40:
                length_clarity = 9.0 - (avg_sent_len - 25) * 0.2
            else:
                length_clarity = max(2.0, 6.0 - (avg_sent_len - 40) * 0.1)
            
            # Moderate CV is good (0.1-0.5), too high or zero is bad
            if cv < 0.05:
                variety_bonus = 0
            elif cv <= 0.5:
                variety_bonus = 1.0
            elif cv <= 1.0:
                variety_bonus = 0.5
            else:
                variety_bonus = -0.5
            
            syntax_score = min(10, length_clarity + variety_bonus)
        else:
            syntax_score = 3.0
        
        # === 5. Directness / Topic Addressing ===
        # How quickly does the response use words from the query?
        # Check first sentence overlap with query
        query_content = set(w for w in query_words if w not in stop_words and len(w) > 2)
        
        if sentences and query_content:
            first_sent_words = set(re.findall(r'[a-zA-Z]+', sentences[0].lower()))
            first_overlap = len(query_content & first_sent_words) / max(len(query_content), 1)
            directness_score = 4.0 + first_overlap * 6.0  # 4-10 range
        else:
            directness_score = 5.0
        
        # === 6. Filler and Vagueness Penalty ===
        filler_patterns = [
            (r'\bin general\b', 0.3),
            (r'\bit is worth noting that\b', 0.5),
            (r'\bit should be noted that\b', 0.5),
            (r'\bas a matter of fact\b', 0.4),
            (r'\bbasically\b', 0.2),
            (r'\bessentially\b', 0.2),
            (r'\bactually\b', 0.15),
            (r'\breally\b', 0.15),
            (r'\bvery\b', 0.1),
            (r'\bjust\b', 0.1),
            (r'\bquite\b', 0.15),
            (r'\bsomewhat\b', 0.2),
            (r'\bin other words\b', 0.4),
            (r'\bthat is to say\b', 0.4),
            (r'\bas mentioned\b', 0.3),
            (r'\bas stated\b', 0.3),
            (r'\bit goes without saying\b', 0.5),
            (r'\bneedless to say\b', 0.4),
            (r'\bin conclusion\b', 0.2),
            (r'\bin summary\b', 0.2),
            (r'\ball in all\b', 0.3),
            (r'\bat the end of the day\b', 0.4),
            (r'\bkind of\b', 0.2),
            (r'\bsort of\b', 0.2),
            (r'\bmore or less\b', 0.3),
            (r'\bto be honest\b', 0.3),
            (r'\bfrankly speaking\b', 0.3),
        ]
        
        filler_penalty = 0
        for pattern, weight in filler_patterns:
            matches = re.findall(pattern, resp_lower)
            filler_penalty += len(matches) * weight
        
        # Normalize filler penalty relative to response length
        filler_penalty_normalized = filler_penalty / max(total_words / 50, 1)
        filler_score = max(0, 10 - filler_penalty_normalized * 3)
        
        # === 7. Emptiness / Substance Check ===
        # Penalize responses that are just echoing the query or near-empty
        if total_words < 3:
            substance_score = 2.0
        elif total_words < 5:
            substance_score = 4.0
        else:
            # Check if response is just parroting query
            resp_content = set(w for w in resp_words if w not in stop_words and len(w) > 2)
            if query_content and resp_content:
                only_in_resp = resp_content - query_content
                novelty = len(only_in_resp) / max(len(resp_content), 1)
                substance_score = 3.0 + novelty * 7.0
            else:
                substance_score = 5.0
        
        # === 8. Structural markers (lists, formatting) ===
        has_list = bool(re.search(r'(?:^|\n)\s*[\-\*\d]+[\.\)]\s', response))
        has_colon_structure = bool(re.search(r':\s', response))
        structure_bonus = 0
        if has_list:
            structure_bonus += 0.5
        if has_colon_structure:
            structure_bonus += 0.3
        
        # === Combine scores with weights ===
        weights = {
            'compression': 0.10,
            'density': 0.15,
            'repetition': 0.25,
            'syntax': 0.15,
            'directness': 0.10,
            'filler': 0.10,
            'substance': 0.15,
        }
        
        final_score = (
            weights['compression'] * compression_score +
            weights['density'] * density_score +
            weights['repetition'] * repetition_score +
            weights['syntax'] * syntax_score +
            weights['directness'] * directness_score +
            weights['filler'] * filler_score +
            weights['substance'] * substance_score +
            structure_bonus
        )
        
        # Scale to 0-100 range
        final_score = max(0, min(100, final_score * 10))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 30.0
            return 0.0
        except Exception:
            return 0.0