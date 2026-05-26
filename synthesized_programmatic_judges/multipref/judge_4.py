def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query term importance weighting, answer directness detection, and 
    semantic coverage analysis through co-occurrence patterns.
    
    This variant focuses on:
    1. Query term importance (IDF-like weighting within the response)
    2. Early mention bonus (relevant terms appearing early = more direct)
    3. Question type detection and answer pattern matching
    4. Query phrase coverage (multi-word subsequences)
    5. Response focus ratio (relevant vs irrelevant content proportion)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 5:
            return 0.0
        
        # --- Tokenization and preprocessing ---
        STOPWORDS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
            'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'am', 'about', 'up', 'down', 'also', 'any', 'much',
            'many', 'get', 'got', 'like', 'well', 'really', 'even', 'still',
            'let', 'make', 'made', 'know', 'think', 'see', 'come', 'go', 'going',
            'take', 'say', 'said', 'tell', 'told', 'give', 'given', 'im', 'dont',
            'cant', 'wont', 'youre', 'theyre', 'ive', 'youve', 'weve', 'theyve',
            'id', 'youd', 'hed', 'shed', 'wed', 'theyd', 'ill', 'youll', 'hell',
            'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt', 'werent',
            'hasnt', 'havent', 'hadnt', 'doesnt', 'didnt', 'wont', 'wouldnt',
            'couldnt', 'shouldnt', 'mustnt', 'neednt', 'daren', 'needn', 'haven',
            'bit', 'quite', 'rather', 'something', 'anything', 'everything',
            'nothing', 'someone', 'anyone', 'everyone', 'one', 'two', 'three',
        }
        
        def tokenize(text):
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def content_tokens(tokens):
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = content_tokens(query_tokens)
        response_content = content_tokens(response_tokens)
        
        if not query_content or not response_content:
            # Fallback: check raw overlap
            q_set = set(query_tokens)
            r_set = set(response_tokens)
            overlap = len(q_set & r_set)
            return min(overlap * 2.0, 10.0)
        
        # --- 1. Query Term Importance Weighting (IDF-like) ---
        # Rarer query terms (less common in English) get higher weight
        # Approximate by term length and specificity heuristics
        def term_importance(term):
            """Longer, more specific terms are more important for relevance."""
            base = 1.0
            if len(term) >= 8:
                base = 2.5
            elif len(term) >= 6:
                base = 2.0
            elif len(term) >= 4:
                base = 1.5
            # Numbers/technical terms get boost
            if any(c.isdigit() for c in term):
                base *= 1.5
            return base
        
        query_content_unique = list(set(query_content))
        term_weights = {t: term_importance(t) for t in query_content_unique}
        total_weight = sum(term_weights.values())
        
        # --- 2. Weighted Query Term Coverage in Response ---
        response_content_set = set(response_content)
        response_token_set = set(response_tokens)
        
        covered_weight = 0.0
        for term, weight in term_weights.items():
            if term in response_content_set:
                covered_weight += weight
            # Partial match (substring) gets partial credit
            elif any(term in rt or rt in term for rt in response_content_set if len(rt) > 3):
                covered_weight += weight * 0.4
        
        coverage_score = covered_weight / total_weight if total_weight > 0 else 0.0
        
        # --- 3. Early Mention Bonus ---
        # Query terms appearing in the first 25% of response get a bonus
        early_cutoff = max(len(response_tokens) // 4, 10)
        early_tokens = set(response_tokens[:early_cutoff])
        early_content = set(content_tokens(list(early_tokens)))
        
        early_hits = sum(1 for t in query_content_unique if t in early_content)
        early_ratio = early_hits / len(query_content_unique) if query_content_unique else 0
        early_bonus = early_ratio * 1.5  # up to 1.5 points
        
        # --- 4. Question Type Detection and Answer Pattern Matching ---
        query_lower = query.lower().strip()
        question_type_score = 0.0
        
        # Detect question type
        response_lower = response.lower()
        
        if query_lower.startswith(('how can', 'how do', 'how to', 'how should', 'how would')):
            # Expects instructional content
            # Look for step indicators, action verbs, numbered lists
            step_patterns = len(re.findall(r'(?:step\s*\d|first|second|third|\d+\.\s|next|then|finally|start\s+by|begin\s+with)', response_lower))
            question_type_score = min(step_patterns * 0.3, 2.0)
        
        elif query_lower.startswith(('what is', 'what are', 'what\'s', 'whats')):
            # Expects definitional content
            definition_patterns = len(re.findall(r'(?:is\s+a|are\s+|refers?\s+to|means?\s|defined?\s+as|known\s+as|type\s+of)', response_lower))
            question_type_score = min(definition_patterns * 0.4, 2.0)
        
        elif query_lower.startswith(('why ', 'why\'')):
            # Expects explanatory content
            explanation_patterns = len(re.findall(r'(?:because|reason|due\s+to|caused?\s+by|result\s+of|therefore|thus|since|leads?\s+to|explanation)', response_lower))
            question_type_score = min(explanation_patterns * 0.4, 2.0)
        
        elif query_lower.startswith(('can ', 'could ', 'should ', 'do you', 'is it', 'are there')):
            # Yes/no or opinion questions - look for direct stance
            stance_patterns = len(re.findall(r'(?:^|\.\s*)(yes|no|i\s+(?:do|don\'t|believe|think)|absolutely|certainly|definitely|not\s+(?:recommend|believe|think))', response_lower))
            question_type_score = min(stance_patterns * 0.5, 2.0)
        
        elif any(query_lower.startswith(w) for w in ('i want', 'i wanna', 'i need', 'i\'m', 'im ')):
            # Request for help - look for helpful response patterns
            help_patterns = len(re.findall(r'(?:here\s+(?:are|is)|you\s+(?:can|should|could|might|need)|steps?\s+|tip|suggest|recommend|try\s+|consider)', response_lower))
            question_type_score = min(help_patterns * 0.3, 2.0)
        
        # --- 5. Query Phrase Coverage (contiguous subsequences) ---
        def get_phrases(tokens, min_len=2, max_len=4):
            phrases = []
            for n in range(min_len, min(max_len + 1, len(tokens) + 1)):
                for i in range(len(tokens) - n + 1):
                    phrases.append(' '.join(tokens[i:i+n]))
            return phrases
        
        query_phrases = get_phrases(query_content, 2, 3)
        response_text_lower = ' '.join(response_content)
        
        phrase_hits = 0
        for phrase in query_phrases:
            if phrase in response_text_lower:
                phrase_hits += 1
        
        phrase_score = 0.0
        if query_phrases:
            phrase_score = (phrase_hits / len(query_phrases)) * 2.0  # up to 2.0
        
        # --- 6. Response Focus Ratio ---
        # What proportion of response content words are related to query?
        # Use a broader notion: words that share stems with query terms
        def simple_stem(word):
            """Very simple stemmer - strip common suffixes."""
            for suffix in ['tion', 'sion', 'ment', 'ness', 'ity', 'ies', 'ing', 'ous', 
                          'ive', 'ful', 'less', 'able', 'ible', 'ally', 'edly',
                          'ated', 'ting', 'ted', 'ers', 'est', 'ize', 'ise',
                          'ely', 'ary', 'ory', 'al', 'ed', 'er', 'ly', 'es', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        query_stems = set(simple_stem(t) for t in query_content_unique)
        
        # Also add the original terms
        query_related = query_stems | set(query_content_unique)
        
        # Count response tokens that are related to query
        related_count = 0
        for t in response_content:
            stem = simple_stem(t)
            if t in query_related or stem in query_stems:
                related_count += 1
            # Check if any query stem is a prefix of this token's stem or vice versa
            elif any(stem.startswith(qs) or qs.startswith(stem) for qs in query_stems if len(qs) >= 3 and len(stem) >= 3):
                related_count += 0.5
        
        focus_ratio = related_count / len(response_content) if response_content else 0
        # Don't expect too high focus - 10-30% is good for most responses
        focus_score = min(focus_ratio * 8.0, 2.0)  # up to 2.0
        
        # --- 7. Contextual Coherence: Response echoes query structure ---
        # Check if response references the specific entities/concepts from query
        # Extract potential named entities or specific terms (capitalized, numbers, quoted)
        specific_terms = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        specific_terms += re.findall(r'\d+\.?\d*', query)
        specific_terms = [t.lower() for t in specific_terms if len(t) > 1]
        
        specificity_score = 0.0
        if specific_terms:
            specific_hits = sum(1 for t in specific_terms if t in response_lower)
            specificity_score = (specific_hits / len(specific_terms)) * 1.5  # up to 1.5
        
        # --- 8. Response Quality Indicators (that correlate with relevance) ---
        quality_score = 0.0
        
        # Direct address of the query topic in opening sentence
        first_sentence = response.split('.')[0] if '.' in response else response[:100]
        first_sentence_lower = first_sentence.lower()
        first_sent_tokens = set(content_tokens(tokenize(first_sentence_lower)))
        query_in_first = len(set(query_content_unique) & first_sent_tokens)
        if query_in_first >= 2:
            quality_score += 0.8
        elif query_in_first >= 1:
            quality_score += 0.4
        
        # Structured response (numbered steps, bullet points, headers)
        has_structure = bool(re.search(r'(?:\d+\.\s|\*\s|-\s|#{1,3}\s|step\s+\d)', response_lower))
        if has_structure:
            quality_score += 0.3
        
        # Response length adequacy (not too short for complex queries)
        query_complexity = len(query_content_unique)
        response_len = len(response_tokens)
        if query_complexity > 5 and response_len > 50:
            quality_score += 0.3
        elif query_complexity <= 5 and response_len > 20:
            quality_score += 0.2
        
        # --- 9. Semantic Field Overlap ---
        # Build a simple semantic field from query by looking at word co-occurrence
        # Use the idea that words appearing near query terms in the response are topically relevant
        # Count unique response content words that appear within window of query terms
        window_size = 5
        response_positions = {}
        for i, token in enumerate(response_tokens):
            if token not in response_positions:
                response_positions[token] = []
            response_positions[token].append(i)
        
        query_term_positions = []
        for qt in set(query_tokens):
            if qt in response_positions:
                query_term_positions.extend(response_positions[qt])
        
        if query_term_positions:
            nearby_words = set()
            for pos in query_term_positions:
                for offset in range(-window_size, window_size + 1):
                    idx = pos + offset
                    if 0 <= idx < len(response_tokens):
                        word = response_tokens[idx]
                        if word not in STOPWORDS and len(word) > 2:
                            nearby_words.add(word)
            
            # The density of unique content words near query terms
            semantic_density = len(nearby_words) / (len(set(response_content)) + 1)
            semantic_score = min(semantic_density * 1.5, 1.5)
        else:
            semantic_score = 0.0
        
        # --- 10. Penalize off-topic signals ---
        penalty = 0.0
        
        # If response starts with disclaimers or off-topic preamble
        disclaimer_patterns = re.findall(r'^(?:i\'m not sure|i don\'t know|i cannot|as an ai|i\'m an ai|disclaimer)', response_lower[:200])
        if disclaimer_patterns:
            penalty += 0.5
        
        # If response mentions being unable to help without then helping
        if re.search(r'i\'m not aware|i don\'t have|i cannot|unable to', response_lower[:150]):
            # Check if it then provides useful content
            if len(response_tokens) < 40:
                penalty += 0.8
        
        # --- Combine all scores ---
        # Coverage: 0-1 (weight: 3.0) -> 0-3.0
        # Early bonus: 0-1.5
        # Question type: 0-2.0
        # Phrase score: 0-2.0
        # Focus score: 0-2.0
        # Specificity: 0-1.5
        # Quality: 0-1.4
        # Semantic: 0-1.5
        # Penalty: 0-1.3
        # Total theoretical max ~15, normalize to 0-10
        
        raw_score = (
            coverage_score * 3.0 +
            early_bonus +
            question_type_score +
            phrase_score +
            focus_score +
            specificity_score +
            quality_score +
            semantic_score -
            penalty
        )
        
        # Normalize to 0-10 range
        # Theoretical max around 15, practical max around 12
        final_score = max(0.0, min(10.0, raw_score * (10.0 / 13.0)))
        
        return round(final_score, 4)
    
    except Exception:
        return 0.0