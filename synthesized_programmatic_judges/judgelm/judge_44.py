def judging_function(query, response):
    """
    Evaluates clarity and conciseness using information density, structural coherence,
    and signal-to-noise ratio analysis.
    
    This variant focuses on:
    1. Information density (unique content words per total words)
    2. Structural coherence (sentence-level flow and completeness)
    3. Signal-to-noise ratio (meaningful content vs filler/garbage)
    4. Response-query alignment (topical relevance without bloat)
    5. Formatting cleanliness (absence of artifacts, broken text, code dumps)
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
        
        # === FEATURE 1: Response length adequacy ===
        # Too short responses are usually bad, too long can be bloated
        resp_len = len(response)
        word_tokens = response.split()
        num_words = len(word_tokens)
        
        if num_words <= 1:
            return 0.5
        
        # Minimum viable response length based on query complexity
        query_words = query.split()
        query_len = len(query_words)
        
        # Length adequacy score: penalize extremely short or extremely long
        # Sweet spot depends on query but generally 10-300 words
        if num_words < 3:
            length_score = 0.15
        elif num_words < 6:
            length_score = 0.4
        elif num_words < 15:
            length_score = 0.7
        elif num_words <= 200:
            length_score = 1.0
        elif num_words <= 400:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # === FEATURE 2: Information Density ===
        # Ratio of unique meaningful words to total words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'while',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom', 'also', 'about', 'up'
        }
        
        words_lower = [w.lower().strip('.,!?;:()[]{}"\'-') for w in word_tokens]
        words_lower = [w for w in words_lower if w]
        
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 1]
        unique_content = set(content_words)
        
        if num_words > 0:
            # Information density: unique content words / total words
            info_density = len(unique_content) / num_words if num_words > 0 else 0
            # Repetition ratio: unique content / all content words
            repetition_ratio = len(unique_content) / len(content_words) if content_words else 0
        else:
            info_density = 0
            repetition_ratio = 0
        
        # Score info density (ideal range: 0.25-0.55)
        if info_density < 0.1:
            density_score = 0.2
        elif info_density < 0.2:
            density_score = 0.5
        elif info_density <= 0.6:
            density_score = 1.0
        else:
            density_score = 0.7  # too many unique words might mean incoherent
        
        # Score repetition (higher unique ratio = less repetition = better)
        if repetition_ratio > 0.85:
            repetition_score = 1.0
        elif repetition_ratio > 0.65:
            repetition_score = 0.8
        elif repetition_ratio > 0.45:
            repetition_score = 0.55
        elif repetition_ratio > 0.3:
            repetition_score = 0.3
        else:
            repetition_score = 0.15
        
        # === FEATURE 3: Sentence Completeness and Structure ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Check for incomplete sentences (ending mid-word or with weird patterns)
        incomplete_penalty = 0.0
        if response.rstrip()[-1] not in '.!?"\')]}' and num_words > 10:
            # Response seems cut off
            incomplete_penalty = 0.15
        
        # Average sentence length (ideal: 8-25 words)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        if avg_sent_len < 3:
            sent_len_score = 0.3
        elif avg_sent_len < 8:
            sent_len_score = 0.7
        elif avg_sent_len <= 25:
            sent_len_score = 1.0
        elif avg_sent_len <= 40:
            sent_len_score = 0.7
        else:
            sent_len_score = 0.4
        
        # Sentence length variance (moderate variance = good writing)
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_sl if mean_sl > 0 else 0
            # Coefficient of variation: 0.2-0.6 is ideal
            if 0.15 <= cv <= 0.7:
                variance_score = 1.0
            elif cv < 0.15:
                variance_score = 0.7  # too uniform
            else:
                variance_score = 0.5  # too variable
        else:
            variance_score = 0.7
        
        # === FEATURE 4: Signal-to-Noise Ratio ===
        # Detect garbage/artifacts
        noise_indicators = 0
        
        # Repeated lines or phrases
        lines = response.split('\n')
        lines_stripped = [l.strip() for l in lines if l.strip()]
        if lines_stripped:
            line_counter = Counter(lines_stripped)
            repeated_lines = sum(1 for count in line_counter.values() if count > 1)
            if repeated_lines > 0:
                noise_indicators += min(repeated_lines * 0.15, 0.5)
        
        # HTML/code artifacts when not asked for code
        query_lower = query.lower()
        is_code_query = any(kw in query_lower for kw in ['html', 'code', 'program', 'function', 'script', 'tag', 'css'])
        
        if not is_code_query:
            html_tags = len(re.findall(r'<[^>]+>', response))
            if html_tags > 2:
                noise_indicators += min(html_tags * 0.08, 0.4)
            
            code_patterns = len(re.findall(r'(import |def |class |print\(|return )', response))
            if code_patterns > 2:
                noise_indicators += min(code_patterns * 0.1, 0.4)
        
        # "Input:" / "Output:" repetition patterns (common in bad responses)
        io_patterns = len(re.findall(r'(?:Input|Output|Question|Answer):', response))
        if io_patterns > 3:
            noise_indicators += min((io_patterns - 3) * 0.1, 0.3)
        
        # Excessive special characters
        special_chars = len(re.findall(r'[#\*\[\]\{\}\|\\`~]', response))
        special_ratio = special_chars / resp_len if resp_len > 0 else 0
        if special_ratio > 0.05 and not is_code_query:
            noise_indicators += min(special_ratio * 5, 0.3)
        
        # Empty or near-empty content (just punctuation/whitespace)
        alpha_chars = sum(1 for c in response if c.isalpha())
        alpha_ratio = alpha_chars / resp_len if resp_len > 0 else 0
        if alpha_ratio < 0.3:
            noise_indicators += 0.3
        
        noise_score = max(0, 1.0 - noise_indicators)
        
        # === FEATURE 5: Query-Response Topical Alignment ===
        # Check if the response addresses the query topic
        query_content = set(w.lower().strip('.,!?;:()[]{}"\'-') for w in query_words 
                          if w.lower().strip('.,!?;:()[]{}"\'-') not in stop_words and len(w) > 2)
        
        if query_content and content_words:
            # How many query content words appear in response
            overlap = len(query_content.intersection(set(content_words)))
            alignment = overlap / len(query_content) if query_content else 0
            alignment_score = min(1.0, alignment * 1.5 + 0.2)  # generous baseline
        else:
            alignment_score = 0.5  # neutral
        
        # === FEATURE 6: Consecutive duplicate detection ===
        # Detect when same phrase is repeated consecutively
        bigrams = []
        for i in range(len(words_lower) - 1):
            bigrams.append(words_lower[i] + ' ' + words_lower[i+1])
        
        if bigrams:
            consecutive_repeats = 0
            for i in range(1, len(bigrams)):
                if bigrams[i] == bigrams[i-1]:
                    consecutive_repeats += 1
            consec_ratio = consecutive_repeats / len(bigrams)
            consec_score = max(0, 1.0 - consec_ratio * 5)
        else:
            consec_score = 1.0
        
        # === FEATURE 7: Paragraph-level redundancy ===
        # Check if paragraphs repeat the same ideas using trigram overlap
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip() and len(p.strip()) > 20]
        
        if len(paragraphs) > 1:
            para_trigrams = []
            for p in paragraphs:
                p_words = [w.lower().strip('.,!?;:()[]{}"\'-') for w in p.split()]
                p_words = [w for w in p_words if w]
                trigrams = set()
                for i in range(len(p_words) - 2):
                    trigrams.add((p_words[i], p_words[i+1], p_words[i+2]))
                para_trigrams.append(trigrams)
            
            # Measure pairwise trigram overlap between paragraphs
            total_overlap = 0
            pairs = 0
            for i in range(len(para_trigrams)):
                for j in range(i+1, len(para_trigrams)):
                    if para_trigrams[i] and para_trigrams[j]:
                        intersection = len(para_trigrams[i] & para_trigrams[j])
                        smaller = min(len(para_trigrams[i]), len(para_trigrams[j]))
                        if smaller > 0:
                            total_overlap += intersection / smaller
                        pairs += 1
            
            avg_overlap = total_overlap / pairs if pairs > 0 else 0
            para_redundancy_score = max(0, 1.0 - avg_overlap * 2)
        else:
            para_redundancy_score = 1.0
        
        # === FEATURE 8: Filler phrase density ===
        filler_patterns = [
            r'\bit is worth noting that\b',
            r'\bit should be noted that\b',
            r'\bas a matter of fact\b',
            r'\bin order to\b',
            r'\bdue to the fact that\b',
            r'\bat the end of the day\b',
            r'\bfor what it\'s worth\b',
            r'\bneedless to say\b',
            r'\bto be honest\b',
            r'\bbasically\b',
            r'\bessentially\b',
            r'\bactually\b',
            r'\bliterally\b',
            r'\bobviously\b',
            r'\bclearly\b',
        ]
        
        resp_lower = response.lower()
        filler_count = 0
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, resp_lower))
        
        filler_ratio = filler_count / num_sentences if num_sentences > 0 else 0
        filler_score = max(0, 1.0 - filler_ratio * 0.5)
        
        # === COMBINE SCORES ===
        # Weighted combination
        raw_score = (
            length_score * 0.15 +
            density_score * 0.12 +
            repetition_score * 0.15 +
            sent_len_score * 0.08 +
            variance_score * 0.05 +
            noise_score * 0.18 +
            alignment_score * 0.10 +
            consec_score * 0.07 +
            para_redundancy_score * 0.05 +
            filler_score * 0.05
        )
        
        # Apply incomplete penalty
        raw_score = raw_score - incomplete_penalty
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, raw_score * 10.0))
        
        # Apply floor/ceiling adjustments for extreme cases
        # Very short non-informative responses
        if num_words <= 2 and len(content_words) <= 1:
            final_score = min(final_score, 1.5)
        
        # Responses that are just punctuation or single characters
        if alpha_ratio < 0.15:
            final_score = min(final_score, 0.5)
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 5:
                return 4.0
            return 1.0
        except Exception:
            return 3.0