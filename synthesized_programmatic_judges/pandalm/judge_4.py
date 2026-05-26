def judging_function(query, response):
    """
    Evaluates response relevance using a TF-IDF inspired approach with
    query decomposition into intent components and coverage analysis.
    
    This variant focuses on:
    1. Query intent decomposition (extracting key question components)
    2. TF-IDF-like term weighting (not raw overlap)
    3. Response coherence and information density
    4. Penalty for repetition and low-information content
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
        
        # --- Tokenization helpers ---
        def tokenize(text):
            text = text.lower()
            tokens = re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
            return tokens
        
        # Common English stopwords
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
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'what',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'about', 'up',
        }
        
        def content_tokens(tokens):
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = content_tokens(query_tokens)
        response_content = content_tokens(response_tokens)
        
        if not response_tokens:
            return 0.0
        
        # --- 1. Query Intent Decomposition ---
        # Extract different types of query components
        
        # Action verbs from query (what the query asks to do)
        action_words = set()
        action_verbs = {
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'generate', 'create', 'write', 'rewrite', 'summarize', 'analyze',
            'evaluate', 'discuss', 'define', 'identify', 'suggest', 'recommend',
            'show', 'demonstrate', 'illustrate', 'outline', 'classify', 'crop',
            'reduce', 'add', 'come', 'give', 'make', 'find', 'tell', 'name'
        }
        for t in query_tokens:
            if t in action_verbs:
                action_words.add(t)
        
        # Subject/topic words (content words that aren't action verbs)
        topic_words = [t for t in query_content if t not in action_verbs]
        
        # --- 2. TF-IDF-like Term Importance Weighting ---
        # Words that appear in the query but are relatively uncommon get higher weight
        # Simulate IDF using word length and frequency heuristics
        
        def term_importance(word):
            """Estimate importance of a term - longer, less common words matter more."""
            base = 1.0
            # Length bonus (longer words tend to be more specific)
            if len(word) >= 8:
                base += 1.5
            elif len(word) >= 6:
                base += 1.0
            elif len(word) >= 4:
                base += 0.5
            # Penalize very short content words
            if len(word) <= 2:
                base *= 0.3
            return base
        
        # Build weighted query profile
        query_term_weights = {}
        for t in query_content:
            w = term_importance(t)
            query_term_weights[t] = max(query_term_weights.get(t, 0), w)
        
        # Also weight topic words higher than action words
        for t in topic_words:
            if t in query_term_weights:
                query_term_weights[t] *= 1.5
        
        # --- 3. Weighted Coverage Score ---
        # How much of the weighted query content is covered by the response
        response_content_set = set(response_content)
        
        if query_term_weights:
            total_weight = sum(query_term_weights.values())
            covered_weight = sum(
                query_term_weights[t] for t in query_term_weights
                if t in response_content_set
            )
            weighted_coverage = covered_weight / total_weight if total_weight > 0 else 0
        else:
            weighted_coverage = 0.5  # neutral if query has no content words
        
        # --- 4. Semantic Field Expansion ---
        # Check if response uses words from the same semantic field
        # Use character-level similarity (shared prefixes/stems) as a proxy
        
        def stem_match(w1, w2):
            """Check if two words share a common stem (simple prefix matching)."""
            if w1 == w2:
                return True
            min_len = min(len(w1), len(w2))
            if min_len < 3:
                return False
            prefix_len = 0
            for c1, c2 in zip(w1, w2):
                if c1 == c2:
                    prefix_len += 1
                else:
                    break
            # Need at least 60% of shorter word or 4 chars as shared prefix
            return prefix_len >= max(4, int(min_len * 0.6))
        
        # Count stem-level matches for query content not exactly in response
        unmatched_query = [t for t in query_term_weights if t not in response_content_set]
        stem_matches = 0
        for qt in unmatched_query:
            for rt in response_content_set:
                if stem_match(qt, rt):
                    stem_matches += 1
                    break
        
        total_query_terms = len(query_term_weights) if query_term_weights else 1
        stem_coverage_bonus = (stem_matches / total_query_terms) * 0.3
        
        # --- 5. Response Information Density ---
        response_counter = Counter(response_content)
        total_content = len(response_content)
        unique_content = len(response_counter)
        
        if total_content > 0:
            # Type-token ratio for content words
            ttr = unique_content / total_content
        else:
            ttr = 0
        
        # --- 6. Repetition Penalty ---
        # Detect excessive repetition (like "miserable, miserable, miserable")
        if response_tokens:
            token_counter = Counter(response_tokens)
            max_freq = max(token_counter.values())
            total_tokens = len(response_tokens)
            
            # Ratio of most frequent token
            max_freq_ratio = max_freq / total_tokens
            
            # Count tokens that appear more than 3 times
            highly_repeated = sum(1 for t, c in token_counter.items() 
                                  if c > 3 and t not in STOPWORDS)
            
            repetition_penalty = 0
            if max_freq_ratio > 0.15 and max_freq > 5:
                repetition_penalty += (max_freq_ratio - 0.15) * 2
            repetition_penalty += highly_repeated * 0.05
            repetition_penalty = min(repetition_penalty, 0.5)
        else:
            repetition_penalty = 0
        
        # --- 7. Response Length Adequacy ---
        query_len = len(query_tokens)
        resp_len = len(response_tokens)
        
        # Ideal response should be longer than query but not excessively
        if resp_len == 0:
            length_score = 0
        elif resp_len < 5:
            length_score = 0.2
        elif resp_len < 10:
            length_score = 0.5
        elif resp_len < 20:
            length_score = 0.7
        elif resp_len < 80:
            length_score = 1.0
        elif resp_len < 150:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # --- 8. Sentence Structure Analysis ---
        # Good responses typically have multiple well-formed sentences
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            structure_score = 0.1
        elif num_sentences == 1:
            structure_score = 0.5
        elif num_sentences <= 5:
            structure_score = 0.9
        else:
            structure_score = 1.0
        
        # Check if sentences have reasonable length variation (not all identical)
        if num_sentences > 1:
            sent_lengths = [len(tokenize(s)) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            if avg_sent_len > 0:
                variance = sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Some variation is good
                variation_score = min(std_dev / (avg_sent_len + 1), 1.0) * 0.3
            else:
                variation_score = 0
        else:
            variation_score = 0
        
        # --- 9. Direct Address Detection ---
        # Check if response directly addresses the query type
        query_lower = query.lower()
        response_lower = response.lower()
        
        direct_address_bonus = 0
        
        # If query asks "what", response should contain explanatory language
        if 'what' in query_lower[:20]:
            explain_markers = ['means', 'refers', 'is', 'describes', 'indicates', 'suggests']
            if any(m in response_lower for m in explain_markers):
                direct_address_bonus += 0.15
        
        # If query asks to "compare", response should have comparison language
        if 'compare' in query_lower or 'contrast' in query_lower:
            compare_markers = ['while', 'whereas', 'but', 'however', 'unlike', 'differ',
                             'similar', 'both', 'on the other hand', 'in contrast']
            matches = sum(1 for m in compare_markers if m in response_lower)
            direct_address_bonus += min(matches * 0.05, 0.2)
        
        # If query asks to "explain", check for explanatory depth
        if 'explain' in query_lower:
            explain_markers = ['because', 'this means', 'in other words', 'suggests',
                             'implies', 'refers to', 'meaning']
            matches = sum(1 for m in explain_markers if m in response_lower)
            direct_address_bonus += min(matches * 0.05, 0.15)
        
        # If query asks to "describe", check for descriptive language
        if 'describe' in query_lower:
            if num_sentences >= 2:
                direct_address_bonus += 0.1
        
        # If query asks for examples/list, check for multiple items
        if any(w in query_lower for w in ['example', 'list', 'provide', 'name']):
            # Count comma-separated items or list items
            comma_items = response.count(',')
            if comma_items >= 2:
                direct_address_bonus += 0.1
        
        # --- 10. Completeness heuristic ---
        # Check if response seems cut off
        cutoff_penalty = 0
        if response.rstrip()[-1:] not in '.!?")\']' and len(response) > 50:
            cutoff_penalty = 0.15
        
        # --- 11. Non-response detection ---
        non_response_patterns = ['<noinput>', 'n/a', 'no input', 'i cannot', 'i can\'t']
        is_non_response = any(p in response_lower for p in non_response_patterns)
        if is_non_response and len(response_tokens) < 10:
            return 1.0
        
        # --- Combine all signals ---
        # Weighted combination
        score = (
            weighted_coverage * 30 +          # Primary: query term coverage (0-30)
            stem_coverage_bonus * 10 +         # Stem-level matches (0-3)
            length_score * 15 +                # Length adequacy (0-15)
            structure_score * 10 +             # Sentence structure (0-10)
            ttr * 10 +                         # Vocabulary diversity (0-10)
            variation_score * 5 +              # Sentence variation (0-1.5)
            direct_address_bonus * 30 +        # Direct query address (0-6)
            - repetition_penalty * 20 +        # Repetition penalty (0 to -10)
            - cutoff_penalty * 15              # Cutoff penalty (0 to -2.25)
        )
        
        # Ensure score is in reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
        
    except Exception:
        return 5.0