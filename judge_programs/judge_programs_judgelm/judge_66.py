def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of a response.
    
    This variant focuses on:
    - Information density and signal-to-noise ratio
    - Sentence structure variety and rhythm
    - Repetition detection (penalize repeated content)
    - Response completeness (not cut off)
    - Appropriate response length relative to query complexity
    - Coherent block structure (detecting logical segments)
    - Punctuation quality as a proxy for structure
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        resp = response.strip()
        query_clean = query.strip() if query else ""
        
        # === Feature 1: Response length adequacy (0-10) ===
        resp_len = len(resp)
        word_count = len(resp.split())
        
        if word_count <= 1:
            return 0.5
        
        if word_count <= 3:
            # Very short responses - could be appropriate for simple queries
            # but generally penalized for structural organization
            length_score = 2.0
        elif word_count <= 10:
            length_score = 4.0
        elif word_count <= 30:
            length_score = 6.0
        elif word_count <= 100:
            length_score = 8.0
        elif word_count <= 300:
            length_score = 7.5
        else:
            length_score = 6.5  # Very long might be rambling
        
        # === Feature 2: Sentence structure analysis (0-10) ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # Sentence length variety (standard deviation of sentence lengths)
        if num_sentences >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            # Some variety is good, too much is chaotic
            if 2 <= std_sl <= 15:
                variety_score = 8.0
            elif std_sl < 2:
                variety_score = 5.0  # Monotonous
            else:
                variety_score = 4.0  # Too chaotic
        else:
            variety_score = 3.0
        
        # === Feature 3: Repetition detection (0-10, higher = less repetition = better) ===
        words_lower = resp.lower().split()
        
        # N-gram repetition (trigrams)
        if len(words_lower) >= 6:
            trigrams = [tuple(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0
            
            # Also check for repeated lines
            lines = [l.strip() for l in resp.split('\n') if l.strip()]
            if len(lines) >= 2:
                line_counts = Counter(lines)
                repeated_lines = sum(c - 1 for c in line_counts.values() if c > 1)
                line_rep_ratio = repeated_lines / len(lines)
            else:
                line_rep_ratio = 0
            
            combined_rep = repetition_ratio * 0.6 + line_rep_ratio * 0.4
            repetition_score = max(0, 10 - combined_rep * 25)
        else:
            repetition_score = 6.0
        
        # === Feature 4: Punctuation quality (0-10) ===
        # Good responses use proper punctuation
        period_count = resp.count('.')
        comma_count = resp.count(',')
        colon_count = resp.count(':')
        semicolon_count = resp.count(';')
        
        punct_density = (period_count + comma_count + colon_count + semicolon_count) / max(word_count, 1)
        
        if 0.03 <= punct_density <= 0.25:
            punct_score = 8.0
        elif punct_density < 0.03:
            punct_score = 3.0  # Barely any punctuation
        else:
            punct_score = 5.0  # Over-punctuated
        
        # Bonus for ending with proper punctuation
        if resp[-1] in '.!?"\'':
            punct_score = min(10, punct_score + 1)
        
        # === Feature 5: Block/segment structure (0-10) ===
        lines = resp.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        empty_line_count = sum(1 for l in lines if not l.strip())
        
        # Detect structured elements (different from variant 1's approach)
        # Look for labeled/keyed patterns like "Key: Value"
        kv_pattern = re.findall(r'^[A-Z][^:]{0,40}:\s', resp, re.MULTILINE)
        
        # Detect numbered items with various formats
        numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[.)]\s|[a-z][.)]\s|[A-Z][.)]\s|Step\s+\d|Phase\s+\d)', resp)
        
        # Detect dash/arrow based lists
        dash_items = re.findall(r'(?:^|\n)\s*[-–—→►•]\s', resp)
        
        # Detect "Output:" style prefixes (common in examples)
        output_prefixes = re.findall(r'(?:^|\n)\s*(?:Output|Input|Answer|Question|Note|Example)\s*:', resp, re.IGNORECASE)
        
        structural_elements = len(kv_pattern) + len(numbered_pattern) + len(dash_items) + len(output_prefixes)
        
        if structural_elements >= 3:
            block_score = 9.0
        elif structural_elements >= 1:
            block_score = 7.0
        elif num_sentences >= 3 and empty_line_count >= 1:
            block_score = 6.5  # Paragraph breaks
        elif num_sentences >= 2:
            block_score = 5.0
        else:
            block_score = 3.0
        
        # === Feature 6: Completeness / not truncated (0-10) ===
        completeness_score = 8.0
        
        # Check if response appears cut off
        if resp[-1] not in '.!?"\')]}:;' and word_count > 15:
            completeness_score -= 3.0
        
        # Check for incomplete sentences at end
        last_sentence = sentences[-1] if sentences else ""
        last_words = last_sentence.split()
        if last_words and last_words[-1].lower() in ('the', 'a', 'an', 'to', 'of', 'in', 'for', 'and', 'or', 'but', 'is', 'was', 'are', 'were', 'has', 'have', 'with'):
            completeness_score -= 2.0
        
        # Check for artifacts like repeated fragments
        if re.search(r'(.{20,})\1', resp):
            completeness_score -= 3.0
        
        completeness_score = max(0, completeness_score)
        
        # === Feature 7: Signal-to-noise ratio (0-10) ===
        # Penalize responses with lots of code/HTML when not asked for it
        query_asks_code = bool(re.search(r'\b(code|html|program|script|function|tag)\b', query_clean, re.IGNORECASE))
        
        code_chars = len(re.findall(r'[{}()<>\[\]=/\\;]', resp))
        code_ratio = code_chars / max(resp_len, 1)
        
        if not query_asks_code and code_ratio > 0.1:
            noise_score = max(0, 7 - code_ratio * 30)
        else:
            noise_score = 8.0
        
        # Penalize responses that seem to hallucinate extra Q&A pairs
        extra_qa = len(re.findall(r'(?:Question|Answer|Input|Output)\s*:', resp, re.IGNORECASE))
        if extra_qa > 4:
            noise_score = max(0, noise_score - 3)
        
        # === Feature 8: Query-response proportionality ===
        query_words = len(query_clean.split()) if query_clean else 5
        # Estimate query complexity
        query_complexity = min(query_words / 5, 3.0)  # 0 to 3 scale
        
        # Simple queries should get concise answers
        # Complex queries should get detailed answers
        expected_min_words = max(3, query_complexity * 10)
        expected_max_words = max(50, query_complexity * 150)
        
        if expected_min_words <= word_count <= expected_max_words:
            proportionality_score = 8.0
        elif word_count < expected_min_words:
            proportionality_score = max(2, 8 - (expected_min_words - word_count) * 0.5)
        else:
            proportionality_score = max(4, 8 - (word_count - expected_max_words) / 50)
        
        # === Feature 9: Opening quality ===
        # Good responses start with a clear, direct opening
        opening_score = 6.0
        first_line = non_empty_lines[0] if non_empty_lines else ""
        
        # Starts with a capital letter
        if first_line and first_line[0].isupper():
            opening_score += 1.0
        
        # Doesn't start with just a fragment
        first_line_words = first_line.split()
        if len(first_line_words) >= 3:
            opening_score += 1.5
        elif len(first_line_words) <= 1:
            opening_score -= 2.0
        
        # Starts with a lowercase word (likely a fragment)
        if first_line and first_line[0].islower():
            opening_score -= 2.0
        
        opening_score = max(0, min(10, opening_score))
        
        # === Combine all features with weights ===
        weights = {
            'length': 0.10,
            'variety': 0.10,
            'repetition': 0.18,
            'punctuation': 0.10,
            'block': 0.15,
            'completeness': 0.12,
            'noise': 0.10,
            'proportionality': 0.08,
            'opening': 0.07,
        }
        
        scores = {
            'length': length_score,
            'variety': variety_score,
            'repetition': repetition_score,
            'punctuation': punct_score,
            'block': block_score,
            'completeness': completeness_score,
            'opening': opening_score,
            'noise': noise_score,
            'proportionality': proportionality_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # === Apply severe penalties for clearly bad responses ===
        # Single character or nearly empty
        if word_count <= 2:
            final_score = min(final_score, 1.5)
        
        # Response is just punctuation or noise
        alpha_ratio = sum(1 for c in resp if c.isalpha()) / max(resp_len, 1)
        if alpha_ratio < 0.3:
            final_score *= 0.4
        
        # Massive repetition detected via simple check
        if word_count > 10:
            unique_words = len(set(words_lower))
            unique_ratio = unique_words / word_count
            if unique_ratio < 0.25:
                final_score *= 0.5
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except Exception:
            return 2.0