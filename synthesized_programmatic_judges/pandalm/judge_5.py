def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query decomposition into intent components and coverage analysis.
    
    This variant focuses on:
    1. Query intent decomposition (extracting key concepts/verbs/objects)
    2. Weighted term importance (IDF-like weighting against common English words)
    3. Response coverage of query components
    4. Penalizing repetition, emptiness, and off-topic verbosity
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
        
        # Common English stopwords - these get low weight
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
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'any', 'also', 'get', 'got', 'like', 'make', 'made',
        }
        
        # Semi-common words that carry some weight
        semi_common = {
            'give', 'take', 'come', 'go', 'see', 'know', 'think', 'say', 'tell',
            'find', 'use', 'way', 'many', 'well', 'back', 'much', 'good', 'new',
            'first', 'last', 'long', 'great', 'little', 'right', 'old', 'big',
            'high', 'small', 'large', 'next', 'early', 'young', 'important',
            'different', 'following', 'given', 'input', 'output', 'example',
        }
        
        def tokenize(text):
            """Tokenize text into lowercase words."""
            return re.findall(r'[a-z]+', text.lower())
        
        def get_stems(word):
            """Very simple stemming - returns a set of possible stems."""
            stems = {word}
            # Remove common suffixes
            for suffix in ['ing', 'tion', 'sion', 'ment', 'ness', 'able', 'ible',
                          'ful', 'less', 'ous', 'ive', 'al', 'ly', 'er', 'est',
                          'ed', 'es', 's', 'ize', 'ise', 'ity', 'ence', 'ance']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    stems.add(word[:-len(suffix)])
            # Also add the word with common suffixes for matching
            return stems
        
        def term_weight(word):
            """Assign weight to a term based on how informative it is."""
            if word in stopwords:
                return 0.05
            if word in semi_common:
                return 0.3
            if len(word) <= 2:
                return 0.1
            if len(word) <= 3:
                return 0.2
            # Longer, less common words are more informative
            return min(1.0, 0.4 + len(word) * 0.08)
        
        # --- Query Analysis ---
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # Extract action verbs from query (typically instructions)
        action_verbs = {
            'compare', 'contrast', 'describe', 'explain', 'generate', 'create',
            'write', 'rewrite', 'provide', 'list', 'summarize', 'analyze',
            'evaluate', 'discuss', 'define', 'identify', 'classify', 'suggest',
            'recommend', 'propose', 'develop', 'design', 'implement', 'calculate',
            'convert', 'translate', 'compose', 'construct', 'elaborate', 'outline',
            'illustrate', 'demonstrate', 'crop', 'reduce', 'add', 'show', 'come',
        }
        
        query_actions = [t for t in query_tokens if t in action_verbs]
        
        # Key content words from query (non-stopword, non-action)
        query_content = []
        for t in query_tokens:
            if t not in stopwords and len(t) > 2:
                query_content.append(t)
        
        # Build stem mapping for flexible matching
        response_stems = {}
        for token in response_tokens:
            for stem in get_stems(token):
                if stem not in response_stems:
                    response_stems[stem] = set()
                response_stems[stem].add(token)
        
        response_token_set = set(response_tokens)
        
        def word_found_in_response(word):
            """Check if a word (or its stem variants) appears in the response."""
            if word in response_token_set:
                return True
            word_stems = get_stems(word)
            for stem in word_stems:
                if stem in response_stems:
                    return True
                # Check if any response stem starts with this stem
                for rs in response_stems:
                    if len(stem) >= 4 and (rs.startswith(stem) or stem.startswith(rs)):
                        return True
            return False
        
        # --- Score Component 1: Weighted Query Content Coverage ---
        total_weight = 0.0
        covered_weight = 0.0
        
        seen_content = set()
        for word in query_content:
            if word in seen_content:
                continue
            seen_content.add(word)
            w = term_weight(word)
            total_weight += w
            if word_found_in_response(word):
                covered_weight += w
        
        coverage_score = (covered_weight / total_weight) if total_weight > 0 else 0.5
        
        # --- Score Component 2: Response Informativeness ---
        # Unique content words in response
        response_content = [t for t in response_tokens if t not in stopwords and len(t) > 2]
        response_unique = set(response_content)
        
        # Repetition penalty
        if len(response_content) > 0:
            uniqueness_ratio = len(response_unique) / len(response_content)
        else:
            uniqueness_ratio = 0.0
        
        # Severe repetition detection (same word repeated many times)
        response_counter = Counter(response_tokens)
        max_repeat = max(response_counter.values()) if response_counter else 0
        total_tokens = len(response_tokens)
        
        repetition_penalty = 0.0
        if total_tokens > 5 and max_repeat > 3:
            dominant_ratio = max_repeat / total_tokens
            if dominant_ratio > 0.3:
                repetition_penalty = min(0.4, (dominant_ratio - 0.3) * 2)
        
        # --- Score Component 3: Response Length Adequacy ---
        # Not too short, not excessively long relative to query complexity
        query_complexity = len(seen_content)  # number of unique content words
        
        response_len = len(response_tokens)
        
        # Very short responses are usually worse
        if response_len < 5:
            length_score = 0.2
        elif response_len < 10:
            length_score = 0.5
        elif response_len < 20:
            length_score = 0.7
        elif response_len < 80:
            length_score = 1.0
        elif response_len < 150:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # --- Score Component 4: Topical Coherence ---
        # How much of the response content relates back to query topics
        if len(response_unique) > 0:
            query_stems = {}
            for token in query_tokens:
                for stem in get_stems(token):
                    if stem not in query_stems:
                        query_stems[stem] = set()
                    query_stems[stem].add(token)
            
            query_token_set = set(query_tokens)
            
            related_count = 0
            for rword in response_unique:
                if rword in query_token_set:
                    related_count += 1
                    continue
                r_stems = get_stems(rword)
                found = False
                for rs in r_stems:
                    if rs in query_stems:
                        found = True
                        break
                    for qs in query_stems:
                        if len(rs) >= 4 and (rs.startswith(qs) or qs.startswith(rs)):
                            found = True
                            break
                    if found:
                        break
                if found:
                    related_count += 1
            
            topical_coherence = related_count / len(response_unique)
        else:
            topical_coherence = 0.0
        
        # --- Score Component 5: Structural Quality ---
        # Check if response actually forms sentences (not just garbage)
        sentence_count = len(re.split(r'[.!?]+', response.strip()))
        has_structure = 1.0 if sentence_count >= 1 and len(response) > 10 else 0.5
        
        # Check for actual explanation/elaboration beyond just echoing
        # Responses that add new relevant content beyond query words
        new_content_words = response_unique - set(query_tokens) - stopwords
        elaboration_score = min(1.0, len(new_content_words) / max(5, query_complexity * 2))
        
        # --- Score Component 6: Direct Address ---
        # Does the response directly address the query type?
        direct_address = 0.0
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for query-type specific patterns
        if 'compare' in query_lower or 'contrast' in query_lower:
            comparison_words = ['both', 'while', 'whereas', 'however', 'differ', 'similar',
                              'unlike', 'but', 'on the other hand', 'in contrast']
            matches = sum(1 for w in comparison_words if w in response_lower)
            direct_address = min(1.0, matches * 0.3)
        
        elif 'explain' in query_lower or 'describe' in query_lower or 'what' in query_lower:
            explanation_words = ['means', 'because', 'therefore', 'this', 'refers',
                               'suggests', 'indicates', 'is used', 'describes']
            matches = sum(1 for w in explanation_words if w in response_lower)
            direct_address = min(1.0, matches * 0.25)
        
        elif 'provide' in query_lower or 'list' in query_lower or 'give' in query_lower:
            # Check for multiple items
            comma_count = response.count(',')
            direct_address = min(1.0, comma_count * 0.2)
        
        elif 'rewrite' in query_lower or 'rephrase' in query_lower:
            # Should use different words
            direct_address = min(1.0, len(new_content_words) * 0.15)
        
        elif 'generate' in query_lower or 'create' in query_lower or 'write' in query_lower:
            direct_address = min(1.0, 0.3 + elaboration_score * 0.7)
        
        else:
            direct_address = 0.5  # neutral
        
        # --- Combine Scores ---
        # Weighted combination
        final_score = (
            coverage_score * 30.0 +          # Query content coverage (0-30)
            topical_coherence * 15.0 +        # Topical coherence (0-15)
            length_score * 15.0 +             # Length adequacy (0-15)
            uniqueness_ratio * 10.0 +         # Non-repetitiveness (0-10)
            elaboration_score * 10.0 +        # Elaboration beyond echo (0-10)
            has_structure * 5.0 +             # Structural quality (0-5)
            direct_address * 15.0             # Direct address of query type (0-15)
        )
        
        # Apply repetition penalty
        final_score *= (1.0 - repetition_penalty)
        
        # Penalty for extremely short responses
        if len(response.strip()) < 15:
            final_score *= 0.4
        elif len(response.strip()) < 30:
            final_score *= 0.6
        
        # Penalty for responses that are just the query echoed back
        if response.strip().lower() == query.strip().lower():
            final_score *= 0.2
        
        # Penalty for empty/placeholder responses
        if response.strip().lower() in ['<noinput>', 'n/a', 'none', '']:
            return 0.5
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(str(response).strip()) > 10:
                return 25.0
            return 5.0
        except Exception:
            return 5.0