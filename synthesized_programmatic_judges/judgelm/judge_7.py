def judging_function(query, response):
    """
    Evaluates response relevance using a BM25-inspired term weighting approach
    combined with query intent coverage analysis and response coherence signals.
    
    This variant uses:
    - BM25-like term importance weighting
    - Query intent decomposition (question words, key entities, action verbs)
    - Response structural quality signals (length adequacy, repetition penalty, noise detection)
    - Proportional coverage of query components
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if not response or len(response) < 2:
            return 0.0
        
        # --- Tokenization ---
        def tokenize(text):
            text = text.lower()
            tokens = re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text)
            return tokens
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'its', 'it', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'am', 'any', 'down', 'get', 'got', 'much', 'now',
            'please', 'make', 'let'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        query_content = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        response_content = [t for t in response_tokens if t not in stopwords and len(t) > 1]
        
        if not query_content:
            query_content = query_tokens[:5]
        
        # --- BM25-inspired term weighting ---
        # Treat query terms with IDF-like weights based on term specificity (length as proxy)
        # and compute BM25-style relevance score
        
        resp_counter = Counter(response_content)
        resp_len = len(response_content)
        
        # Average document length estimate
        avg_dl = max(50, len(query_content) * 8)
        k1 = 1.5
        b = 0.75
        
        # Compute IDF-like weight for each query term
        # Longer, rarer-looking terms get higher weight
        def term_weight(term):
            base = 1.0
            if len(term) >= 6:
                base += 0.5
            if len(term) >= 8:
                base += 0.5
            # Terms with digits (specific identifiers) get bonus
            if any(c.isdigit() for c in term):
                base += 0.3
            return base
        
        # Unique query content terms with weights
        query_term_weights = {}
        for t in query_content:
            if t not in query_term_weights:
                query_term_weights[t] = term_weight(t)
        
        # BM25 score
        bm25_score = 0.0
        total_weight = sum(query_term_weights.values())
        
        if total_weight > 0:
            for term, w in query_term_weights.items():
                tf = resp_counter.get(term, 0)
                # BM25 TF normalization
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * resp_len / avg_dl)
                if denominator > 0:
                    bm25_score += w * (numerator / denominator)
            
            bm25_score = bm25_score / total_weight
        
        # Normalize BM25 to 0-1 range with saturation
        bm25_normalized = min(1.0, bm25_score / 2.0)
        
        # --- Query Intent Coverage ---
        # What fraction of unique query content words appear in the response?
        query_unique = set(query_content)
        response_set = set(response_content)
        
        if query_unique:
            coverage = len(query_unique & response_set) / len(query_unique)
        else:
            coverage = 0.5
        
        # --- Semantic Field Expansion ---
        # Check if response contains words that share common stems with query words
        def crude_stem(word):
            """Very crude stemmer - strips common suffixes"""
            if len(word) <= 3:
                return word
            for suffix in ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful', 
                          'less', 'ous', 'ive', 'ing', 'ated', 'ize', 'ise',
                          'ity', 'ally', 'ical', 'ence', 'ance', 'ers', 'est',
                          'ed', 'ly', 'er', 'es', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        query_stems = set(crude_stem(t) for t in query_unique if len(t) > 2)
        response_stems = set(crude_stem(t) for t in response_set if len(t) > 2)
        
        if query_stems:
            stem_coverage = len(query_stems & response_stems) / len(query_stems)
        else:
            stem_coverage = 0.5
        
        # --- Response Quality Signals ---
        
        # 1. Length adequacy
        query_len = len(query)
        resp_len_chars = len(response)
        
        # Responses that are too short relative to query are penalized
        if resp_len_chars < 5:
            length_score = 0.05
        elif resp_len_chars < 15:
            length_score = 0.2
        elif resp_len_chars < 30:
            length_score = 0.4
        elif resp_len_chars < 80:
            length_score = 0.7
        elif resp_len_chars < 500:
            length_score = 1.0
        elif resp_len_chars < 1000:
            length_score = 0.95
        else:
            # Very long responses might be rambling
            length_score = 0.85
        
        # 2. Repetition penalty
        if len(response_tokens) > 5:
            # Check for repeated phrases (3-grams)
            trigrams = []
            for i in range(len(response_tokens) - 2):
                trigrams.append(tuple(response_tokens[i:i+3]))
            
            if trigrams:
                trigram_counter = Counter(trigrams)
                most_common_count = trigram_counter.most_common(1)[0][1]
                unique_ratio = len(set(trigrams)) / len(trigrams)
                
                if most_common_count > 3 or unique_ratio < 0.3:
                    repetition_penalty = 0.4
                elif most_common_count > 2 or unique_ratio < 0.5:
                    repetition_penalty = 0.7
                else:
                    repetition_penalty = 1.0
            else:
                repetition_penalty = 1.0
        else:
            repetition_penalty = 0.9 if len(response_tokens) >= 2 else 0.5
        
        # 3. Noise/garbage detection
        noise_penalty = 1.0
        
        # Check for HTML/code artifacts when query doesn't ask for them
        code_indicators = ['<h1>', '<p>', '<div>', 'import ', 'def ', 'class ', '```', 
                          '<blockquote>', '</p>', 'input:', 'output:']
        query_lower = query.lower()
        asks_for_code = any(w in query_lower for w in ['code', 'html', 'program', 'script', 'tag', 'function'])
        
        if not asks_for_code:
            code_count = sum(1 for indicator in code_indicators if indicator.lower() in response.lower())
            if code_count >= 3:
                noise_penalty *= 0.5
            elif code_count >= 1:
                noise_penalty *= 0.8
        
        # Check for question echoing (response just repeats variations of query)
        query_in_response = query.lower().strip().rstrip('?').strip()
        if len(query_in_response) > 20 and query_in_response in response.lower():
            # Some echoing is okay for context, heavy echoing is bad
            echo_ratio = len(query_in_response) / max(len(response), 1)
            if echo_ratio > 0.7:
                noise_penalty *= 0.6
        
        # Check for "Question:" / "Answer:" pattern repetition (quiz-like noise)
        qa_patterns = len(re.findall(r'(?:question|answer)\s*:', response.lower()))
        if qa_patterns > 2:
            noise_penalty *= 0.6
        
        # 4. Direct address detection
        # Does the response seem to directly answer/address the query?
        direct_address_bonus = 0.0
        
        # Question-type detection
        question_words = {'what', 'where', 'when', 'who', 'why', 'how', 'which', 'can', 'is', 'are', 'do', 'does'}
        query_first_word = query_tokens[0] if query_tokens else ''
        is_question = query_first_word in question_words or '?' in query
        
        if is_question:
            # Check if response starts with a substantive statement (not another question)
            first_resp_tokens = response_tokens[:5] if len(response_tokens) >= 5 else response_tokens
            starts_with_question = first_resp_tokens[0] in question_words if first_resp_tokens else False
            
            if not starts_with_question and len(response_tokens) > 3:
                direct_address_bonus = 0.1
        
        # Check for imperative query (commands like "Rewrite", "Create", "Identify")
        imperative_words = {'rewrite', 'create', 'write', 'identify', 'list', 'explain',
                           'describe', 'summarize', 'translate', 'convert', 'generate',
                           'find', 'calculate', 'determine', 'compare', 'analyze'}
        
        is_imperative = query_tokens[0] in imperative_words if query_tokens else False
        
        if is_imperative:
            # Response should contain substantive content, not just echo the command
            if len(response_content) > len(query_content):
                direct_address_bonus += 0.1
        
        # 5. Entity/proper noun matching
        # Extract capitalized words as potential entities
        def extract_entities(text):
            # Find capitalized words that aren't at sentence start
            words = text.split()
            entities = set()
            for i, w in enumerate(words):
                cleaned = re.sub(r'[^a-zA-Z0-9]', '', w)
                if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                    entities.add(cleaned.lower())
            return entities
        
        query_entities = extract_entities(query)
        response_entities = extract_entities(response)
        
        if query_entities:
            entity_coverage = len(query_entities & response_entities) / len(query_entities)
        else:
            entity_coverage = 0.5  # neutral
        
        # --- Combine Scores ---
        # Weighted combination
        raw_score = (
            bm25_normalized * 2.5 +      # BM25 relevance (0-2.5)
            coverage * 2.0 +               # Direct word coverage (0-2.0)
            stem_coverage * 1.0 +          # Stem-based coverage (0-1.0)
            entity_coverage * 1.5 +        # Entity matching (0-1.5)
            length_score * 1.5 +           # Length adequacy (0-1.5)
            direct_address_bonus * 1.0     # Direct address bonus (0-0.2)
        )
        
        # Apply penalties
        raw_score *= repetition_penalty
        raw_score *= noise_penalty
        
        # Maximum possible raw score ~ 2.5 + 2.0 + 1.0 + 1.5 + 1.5 + 0.2 = 8.7
        # Scale to 0-10
        max_raw = 8.7
        final_score = (raw_score / max_raw) * 10.0
        
        # Ensure bounds
        final_score = max(0.0, min(10.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception:
        return 3.0