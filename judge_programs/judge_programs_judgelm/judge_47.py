def judging_function(query, response):
    """
    Evaluates clarity and conciseness using a structural coherence approach:
    - Signal-to-noise ratio (meaningful content vs filler)
    - Structural completeness (proper sentences, no fragments/artifacts)
    - Repetition detection via sliding window content similarity
    - Response appropriateness relative to query complexity
    - Penalize HTML/code artifacts, broken text, single-word non-answers
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            query = ""
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        # === Feature 1: Structural Integrity Score (0-1) ===
        # Check for broken/garbled text indicators
        structural_score = 1.0
        
        # Detect HTML tags (not requested in output)
        html_tags = re.findall(r'<[^>]+>', response_clean)
        # If query asks for HTML, don't penalize
        query_lower = query_clean.lower()
        if 'html' not in query_lower and 'tag' not in query_lower:
            if len(html_tags) > 2:
                structural_score -= 0.3
        
        # Detect code blocks when not asked for code
        code_indicators = ['import ', 'def ', 'class ', 'return ', 'if __name__']
        if 'code' not in query_lower and 'python' not in query_lower and 'program' not in query_lower:
            code_count = sum(1 for ci in code_indicators if ci in response_clean)
            if code_count >= 2:
                structural_score -= 0.25
        
        # Detect truncated responses (ending mid-word or mid-sentence without punctuation)
        truncation_penalty = 0.0
        if len(response_clean) > 50:
            last_char = response_clean[-1]
            if last_char not in '.!?"\')]}:;':
                # Might be truncated - mild penalty
                truncation_penalty = 0.05
        
        structural_score -= truncation_penalty
        structural_score = max(0.0, structural_score)
        
        # === Feature 2: Substantiveness Score (0-1) ===
        # Very short responses that don't answer meaningfully
        words = response_clean.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Check if response is substantive enough given the query
        query_words = query_clean.split()
        query_word_count = len(query_words)
        
        substantive_score = 1.0
        if word_count <= 2:
            # Very short - could be perfect (like "Tokyo") or terrible (like "no" or ".")
            # Check if it seems like a valid concise answer
            response_lower = response_clean.lower().strip('.')
            non_answers = {'no', 'yes', 'ok', 'sure', 'maybe', 'idk', '.', ''}
            if response_lower in non_answers and query_word_count > 5:
                substantive_score = 0.1
            else:
                # Could be a valid short answer - moderate score
                substantive_score = 0.6
        elif word_count < 8:
            substantive_score = 0.7
        else:
            substantive_score = 1.0
        
        # === Feature 3: Repetition Detection via Sliding Windows (0-1) ===
        # Use character-level n-gram windows to detect repeated chunks
        repetition_score = 1.0
        
        if len(response_clean) > 40:
            # Split into sentences
            sentences = re.split(r'[.!?]+', response_clean)
            sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 5]
            
            if len(sentences) >= 2:
                # Compare each pair of sentences using character trigrams
                def char_trigrams(text):
                    text = text.lower()
                    return Counter(text[i:i+3] for i in range(len(text)-2))
                
                def cosine_sim(c1, c2):
                    if not c1 or not c2:
                        return 0.0
                    common = set(c1.keys()) & set(c2.keys())
                    dot = sum(c1[k] * c2[k] for k in common)
                    mag1 = math.sqrt(sum(v*v for v in c1.values()))
                    mag2 = math.sqrt(sum(v*v for v in c2.values()))
                    if mag1 == 0 or mag2 == 0:
                        return 0.0
                    return dot / (mag1 * mag2)
                
                high_sim_count = 0
                total_pairs = 0
                trigrams_list = [char_trigrams(s) for s in sentences]
                
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        sim = cosine_sim(trigrams_list[i], trigrams_list[j])
                        total_pairs += 1
                        if sim > 0.75:
                            high_sim_count += 1
                
                if total_pairs > 0:
                    rep_ratio = high_sim_count / total_pairs
                    repetition_score = max(0.0, 1.0 - rep_ratio * 1.5)
            
            # Also check for repeated substrings (exact or near-exact)
            # Sliding window of ~30 chars
            window_size = min(30, len(response_clean) // 3)
            if window_size >= 15:
                windows = []
                text_lower = response_clean.lower()
                for i in range(0, len(text_lower) - window_size, 5):
                    windows.append(text_lower[i:i+window_size])
                
                if len(windows) > 1:
                    seen = Counter(windows)
                    repeated_windows = sum(1 for v in seen.values() if v > 1)
                    if len(seen) > 0:
                        dup_ratio = repeated_windows / len(seen)
                        if dup_ratio > 0.1:
                            repetition_score *= max(0.3, 1.0 - dup_ratio)
        
        # === Feature 4: Noise Detection (0-1) ===
        # Detect filler patterns, "Question:", "Answer:", "Input:", "Output:" artifacts
        noise_score = 1.0
        
        # Count meta-artifacts
        artifacts = re.findall(r'(?:Question|Answer|Input|Output|Comment)\s*:', response_clean)
        if len(artifacts) > 2:
            noise_score -= min(0.4, len(artifacts) * 0.08)
        
        # Detect "prompt leaking" patterns
        prompt_leak_patterns = [
            r'(?:Determine|Identify|Create|Write|Rewrite)\s+(?:the|a|an|if|which)',
        ]
        if not any(kw in query_lower for kw in ['determine', 'identify', 'create', 'write', 'rewrite']):
            for pat in prompt_leak_patterns:
                matches = re.findall(pat, response_clean)
                if len(matches) > 1:
                    noise_score -= 0.2
        
        # Detect excessive whitespace/newlines as noise
        newline_count = response_clean.count('\n')
        if word_count > 0:
            newline_ratio = newline_count / word_count
            if newline_ratio > 0.5:
                noise_score -= 0.15
        
        noise_score = max(0.0, noise_score)
        
        # === Feature 5: Clarity via Sentence Quality (0-1) ===
        clarity_score = 1.0
        
        if word_count > 0:
            # Average word length (extremely long avg = jargon heavy, extremely short = maybe gibberish)
            avg_word_len = sum(len(w) for w in words) / word_count
            if avg_word_len > 10:
                clarity_score -= 0.15
            elif avg_word_len < 2.5:
                clarity_score -= 0.1
            
            # Sentence count and average sentence length
            sentences_raw = re.split(r'[.!?]+', response_clean)
            sentences_raw = [s.strip() for s in sentences_raw if s.strip()]
            
            if sentences_raw:
                sent_lengths = [len(s.split()) for s in sentences_raw]
                avg_sent_len = sum(sent_lengths) / len(sent_lengths)
                
                # Very long sentences reduce clarity
                if avg_sent_len > 35:
                    clarity_score -= 0.15
                elif avg_sent_len > 25:
                    clarity_score -= 0.05
                
                # High variance in sentence length can indicate poor structure
                if len(sent_lengths) > 1:
                    mean_sl = sum(sent_lengths) / len(sent_lengths)
                    variance = sum((sl - mean_sl)**2 for sl in sent_lengths) / len(sent_lengths)
                    std_dev = math.sqrt(variance)
                    cv = std_dev / max(mean_sl, 1)
                    if cv > 1.5:
                        clarity_score -= 0.1
        
        clarity_score = max(0.0, clarity_score)
        
        # === Feature 6: Conciseness - Penalize bloat relative to query ===
        conciseness_score = 1.0
        
        # If response is extremely long relative to what seems needed
        if word_count > 300 and query_word_count < 20:
            conciseness_score -= 0.1
        
        # Check for "padding" phrases
        padding_phrases = [
            r'\bin conclusion\b', r'\bas mentioned (?:above|earlier|before)\b',
            r'\bit is (?:important|worth) (?:to note|noting|mentioning)\b',
            r'\bneedless to say\b', r'\bit goes without saying\b',
            r'\bas (?:we|you) (?:all )?know\b', r'\bin other words\b',
            r'\bto put it (?:simply|another way)\b', r'\bthat being said\b',
            r'\bhaving said that\b', r'\bwith that (?:being )?said\b',
            r'\ball in all\b', r'\bat the end of the day\b',
        ]
        
        response_lower = response_clean.lower()
        padding_count = 0
        for pp in padding_phrases:
            padding_count += len(re.findall(pp, response_lower))
        
        if padding_count > 0:
            conciseness_score -= min(0.2, padding_count * 0.05)
        
        conciseness_score = max(0.0, conciseness_score)
        
        # === Feature 7: Content Relevance Signal (0-1) ===
        # Light check: does the response share meaningful words with the query?
        relevance_score = 1.0
        
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'down', 'it', 'its', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'much', 'many'
        }
        
        query_content_words = set(
            w.lower().strip('.,!?;:\'"()[]{}') for w in query_words
        ) - stop_words
        
        response_content_words = set(
            w.lower().strip('.,!?;:\'"()[]{}') for w in words
        ) - stop_words
        
        if query_content_words and response_content_words:
            overlap = query_content_words & response_content_words
            coverage = len(overlap) / max(len(query_content_words), 1)
            if coverage < 0.1 and word_count > 5:
                relevance_score = 0.5
            elif coverage < 0.2:
                relevance_score = 0.7
        elif query_content_words and not response_content_words:
            relevance_score = 0.3
        
        # === Combine all features with weights ===
        # Weights chosen to emphasize clarity/conciseness dimensions
        weights = {
            'structural': 0.15,
            'substantive': 0.20,
            'repetition': 0.20,
            'noise': 0.15,
            'clarity': 0.10,
            'conciseness': 0.10,
            'relevance': 0.10,
        }
        
        raw_score = (
            weights['structural'] * structural_score +
            weights['substantive'] * substantive_score +
            weights['repetition'] * repetition_score +
            weights['noise'] * noise_score +
            weights['clarity'] * clarity_score +
            weights['conciseness'] * conciseness_score +
            weights['relevance'] * relevance_score
        )
        
        # Scale to 0-10
        final_score = raw_score * 10.0
        
        # Apply a small boost for responses that are well-formed and moderate length
        if 10 <= word_count <= 200 and repetition_score > 0.8 and noise_score > 0.8:
            final_score = min(10.0, final_score + 0.5)
        
        # Hard floor for truly terrible responses
        if word_count <= 1 and response_clean.strip('.!?, ') == '':
            final_score = 0.0
        
        return round(max(0.0, min(10.0, final_score)), 2)
    
    except Exception:
        # Fallback: return a middle-of-the-road score
        try:
            if response and len(response.strip()) > 10:
                return 5.0
            return 2.0
        except Exception:
            return 3.0