def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, structural coherence,
    and signal-to-noise ratio analysis.
    
    This variant focuses on:
    1. Information density (unique content words / total words ratio)
    2. Structural coherence (logical flow via transition analysis)
    3. Signal-to-noise ratio (meaningful content vs filler/noise)
    4. Response appropriateness (length relative to query complexity)
    5. Entropy-based redundancy detection
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # === Basic tokenization ===
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+', text.lower())
        
        response_tokens = tokenize(response)
        query_tokens = tokenize(query)
        total_words = len(response_tokens)
        
        if total_words == 0:
            return 0.5
        
        # Very short responses (1-2 words) are usually bad unless query is very simple
        if total_words <= 2:
            # Check if query seems to need a short answer
            query_lower = query.lower().strip()
            short_answer_patterns = [r'^identify\b', r'^name\b', r'^what is the\b.*\bcalled\b']
            is_short_ok = any(re.search(p, query_lower) for p in short_answer_patterns)
            if is_short_ok:
                return 4.0
            return 1.0
        
        # === 1. Information Density ===
        # Ratio of unique content words to total words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'that',
            'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'what', 'about', 'up', 'also', 'any',
        }
        
        content_words = [w for w in response_tokens if w not in stop_words and len(w) > 1]
        unique_content = set(content_words)
        
        if total_words > 0:
            info_density = len(unique_content) / total_words
        else:
            info_density = 0
        
        # Clamp and scale info_density (typical range 0.1 - 0.6)
        info_density_score = min(max((info_density - 0.05) / 0.50, 0), 1.0)
        
        # === 2. Character-level entropy for redundancy detection ===
        def char_entropy(text):
            if not text:
                return 0
            freq = Counter(text.lower())
            total = len(text)
            ent = 0
            for count in freq.values():
                p = count / total
                if p > 0:
                    ent -= p * math.log2(p)
            return ent
        
        resp_entropy = char_entropy(response_stripped)
        # Good English text typically has entropy around 4.0-4.5
        # Very repetitive text has lower entropy
        # Normalize: entropy of 4.0+ is good, below 3.0 is bad
        entropy_score = min(max((resp_entropy - 2.0) / 2.5, 0), 1.0)
        
        # === 3. Sentence-level redundancy via Jaccard similarity between sentences ===
        sentences = re.split(r'[.!?\n]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        redundancy_penalty = 0.0
        if len(sentences) >= 2:
            sentence_word_sets = [set(tokenize(s)) - stop_words for s in sentences]
            pair_count = 0
            high_overlap_count = 0
            for i in range(len(sentence_word_sets)):
                for j in range(i + 1, min(i + 5, len(sentence_word_sets))):
                    s1, s2 = sentence_word_sets[i], sentence_word_sets[j]
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    intersection = len(s1 & s2)
                    union = len(s1 | s2)
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.6:
                            high_overlap_count += 1
                    pair_count += 1
            
            if pair_count > 0:
                redundancy_penalty = min(high_overlap_count / max(pair_count, 1), 1.0)
        
        redundancy_score = 1.0 - redundancy_penalty
        
        # === 4. Noise detection ===
        noise_score = 1.0
        noise_penalties = 0.0
        
        # Check for HTML/code artifacts
        html_tags = len(re.findall(r'<[^>]+>', response))
        if html_tags > 2:
            noise_penalties += 0.3
        
        # Check for code blocks when not expected
        code_indicators = len(re.findall(r'(import |def |class |print\(|function\(|var |let |const )', response))
        query_asks_code = bool(re.search(r'(code|program|script|function|implement|html|css|python)', query.lower()))
        if code_indicators > 2 and not query_asks_code:
            noise_penalties += 0.3
        
        # Check for repeated patterns (like "Question:" "Answer:" loops)
        repeated_patterns = re.findall(r'(Question:|Answer:|Input:|Output:|#)', response)
        if len(repeated_patterns) > 3:
            noise_penalties += 0.25
        
        # Check for garbled/broken text (repeated characters or fragments)
        broken_patterns = len(re.findall(r'(\b\w{1,3}\b)\s+\1', response))
        if broken_patterns > 2:
            noise_penalties += 0.2
        
        # Check for incomplete words/fragments at end (truncation is less bad than noise)
        # But random fragments mid-text are bad
        fragment_pattern = len(re.findall(r'\b[a-z]{1,2}\b', response_stripped))
        fragment_ratio = fragment_pattern / max(total_words, 1)
        if fragment_ratio > 0.3:
            noise_penalties += 0.15
        
        noise_score = max(1.0 - noise_penalties, 0.0)
        
        # === 5. Filler and hedge word density ===
        filler_words = [
            'basically', 'actually', 'literally', 'honestly', 'frankly',
            'really', 'quite', 'rather', 'somewhat', 'perhaps', 'maybe',
            'possibly', 'essentially', 'virtually', 'practically',
            'obviously', 'clearly', 'certainly', 'definitely', 'absolutely',
            'simply', 'merely', 'just', 'kind', 'sort', 'stuff', 'things',
            'like', 'well', 'anyway', 'anyways', 'overall',
        ]
        filler_count = sum(1 for w in response_tokens if w in filler_words)
        filler_ratio = filler_count / max(total_words, 1)
        filler_score = max(1.0 - filler_ratio * 5, 0.0)  # penalize heavily
        
        # === 6. Response-query relevance (lightweight) ===
        query_content = set(query_tokens) - stop_words
        response_content = set(content_words)
        
        if len(query_content) > 0:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5  # neutral if query has no content words
        
        relevance_score = min(relevance * 1.5, 1.0)  # scale up, cap at 1
        
        # === 7. Length appropriateness ===
        # Estimate query complexity by word count and question indicators
        query_word_count = len(query_tokens)
        has_question = bool(re.search(r'\?', query))
        
        # Simple heuristic: expected response length
        if query_word_count <= 5:
            ideal_length = 30  # short queries might need moderate answers
        elif query_word_count <= 15:
            ideal_length = 60
        else:
            ideal_length = 100
        
        # Check if query asks for brevity
        asks_short = bool(re.search(r'(short|brief|concise|one word|one sentence|shorter)', query.lower()))
        if asks_short:
            ideal_length = max(ideal_length // 3, 10)
        
        # Length ratio penalty - too long is worse than slightly short for conciseness
        length_ratio = total_words / max(ideal_length, 1)
        if length_ratio < 0.1:
            length_score = 0.2  # way too short
        elif length_ratio < 0.3:
            length_score = 0.5
        elif length_ratio <= 2.0:
            length_score = 1.0  # good range
        elif length_ratio <= 4.0:
            length_score = 0.7  # somewhat verbose
        else:
            length_score = 0.4  # very verbose
        
        # === 8. Structural quality - check for proper sentence structure ===
        # Sentences that start with capitals, end with punctuation
        well_formed = 0
        for s in sentences:
            s = s.strip()
            if len(s) > 3 and s[0].isupper():
                well_formed += 1
        
        if len(sentences) > 0:
            structure_score = min(well_formed / len(sentences), 1.0)
        else:
            structure_score = 0.3
        
        # === 9. Detect pure repetition (exact or near-exact repeated lines) ===
        lines = [l.strip() for l in response_stripped.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counter = Counter(lines)
            most_common_count = line_counter.most_common(1)[0][1]
            if most_common_count > 2:
                exact_repeat_penalty = min((most_common_count - 1) * 0.15, 0.6)
            else:
                exact_repeat_penalty = 0.0
        else:
            exact_repeat_penalty = 0.0
        
        repeat_score = 1.0 - exact_repeat_penalty
        
        # === Combine scores with weights ===
        weights = {
            'info_density': 0.15,
            'entropy': 0.10,
            'redundancy': 0.15,
            'noise': 0.20,
            'filler': 0.05,
            'relevance': 0.15,
            'length': 0.08,
            'structure': 0.05,
            'repeat': 0.07,
        }
        
        scores = {
            'info_density': info_density_score,
            'entropy': entropy_score,
            'redundancy': redundancy_score,
            'noise': noise_score,
            'filler': filler_score,
            'relevance': relevance_score,
            'length': length_score,
            'structure': structure_score,
            'repeat': repeat_score,
        }
        
        combined = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10
        final_score = combined * 10.0
        
        # Apply floor for responses that have actual content
        if total_words >= 5 and noise_score > 0.5:
            final_score = max(final_score, 2.0)
        
        # Hard penalty for clearly broken/nonsensical responses
        if total_words <= 2 and not query_asks_code:
            final_score = min(final_score, 2.0)
        
        # Ensure range
        final_score = max(0.0, min(10.0, round(final_score, 2)))
        
        return final_score
        
    except Exception:
        # Fallback: return a middle score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except Exception:
            return 3.0