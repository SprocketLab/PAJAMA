def judging_function(query, response):
    """
    Evaluate response relevance using a query-focused information retrieval approach.
    
    This variant uses:
    - TF-IDF inspired term weighting (not raw overlap)
    - Query term coverage and distribution analysis
    - Semantic field expansion via co-occurrence patterns
    - Response coherence relative to query intent
    - Positional relevance (early mention of key terms scores higher)
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
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'that',
            'this', 'these', 'those', 'am', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom', 'also', 'any', 'much', 'many',
            'well', 'get', 'got', 'like', 'make', 'made', 'know', 'think', 'see',
            'come', 'go', 'take', 'give', 'say', 'said', 'tell', 'told', 'one',
            'two', 'first', 'new', 'now', 'way', 'even', 'back', 'still', 'let',
            'us', 'thing', 'things', 'really', 'quite', 'since', 'however', 'yet',
            'though', 'although', 'whether', 'either', 'neither', 'else', 'rather',
            'im', 'dont', 'doesnt', 'cant', 'wont', 'isnt', 'arent', 'wasnt',
            'werent', 'havent', 'hasnt', 'hadnt', 'wouldnt', 'couldnt', 'shouldnt',
        }
        
        def tokenize(text):
            """Extract lowercase word tokens."""
            return re.findall(r'[a-z][a-z\']*[a-z]|[a-z]', text.lower())
        
        def get_content_words(tokens):
            """Filter to content words only."""
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = get_content_words(query_tokens)
        response_content = get_content_words(response_tokens)
        
        if not query_content or not response_content:
            # Fallback: basic token overlap
            q_set = set(query_tokens)
            r_set = set(response_tokens)
            if not q_set:
                return 5.0
            overlap = len(q_set & r_set) / len(q_set)
            return round(overlap * 10, 2)
        
        # --- 1. IDF-weighted query term coverage ---
        # Simulate IDF: rarer query terms (appearing less in a "general" sense) get higher weight
        # We approximate by term length and character complexity as a proxy
        def pseudo_idf(term):
            """Longer, more specific terms are likely rarer and more important."""
            base = math.log(2 + len(term))
            # Boost terms with numbers or unusual patterns
            if any(c.isdigit() for c in term):
                base *= 1.5
            return base
        
        query_content_unique = list(set(query_content))
        query_term_weights = {}
        for term in query_content_unique:
            query_term_weights[term] = pseudo_idf(term)
        
        # Normalize weights
        total_weight = sum(query_term_weights.values())
        if total_weight > 0:
            for term in query_term_weights:
                query_term_weights[term] /= total_weight
        
        response_content_set = set(response_content)
        response_token_set = set(response_tokens)
        
        # Weighted coverage score
        weighted_coverage = 0.0
        for term, weight in query_term_weights.items():
            if term in response_content_set:
                weighted_coverage += weight
            # Partial credit for substring matches
            elif any(term in rt or rt in term for rt in response_content_set if len(rt) > 3):
                weighted_coverage += weight * 0.5
        
        # --- 2. Positional relevance: where do query terms first appear? ---
        # Earlier mentions of query terms suggest more direct relevance
        positional_score = 0.0
        response_lower = response.lower()
        n_query_terms_found = 0
        
        for term in query_content_unique:
            pos = response_lower.find(term)
            if pos >= 0:
                n_query_terms_found += 1
                # Score inversely proportional to position
                # Normalize by response length
                rel_pos = pos / max(len(response_lower), 1)
                # Early appearance gets higher score (exponential decay)
                positional_score += math.exp(-2.0 * rel_pos) * query_term_weights.get(term, 0.1)
        
        # Normalize positional score
        if query_content_unique:
            positional_score = positional_score / max(sum(query_term_weights.values()), 0.01)
        
        # --- 3. Query term density and distribution ---
        # How evenly are query terms distributed throughout the response?
        response_len = len(response_tokens)
        if response_len > 0:
            # Split response into chunks
            n_chunks = min(5, max(1, response_len // 10))
            chunk_size = response_len // max(n_chunks, 1)
            
            query_content_set = set(query_content)
            chunks_with_query_terms = 0
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < n_chunks - 1 else response_len
                chunk_tokens = set(response_tokens[start:end])
                if chunk_tokens & query_content_set:
                    chunks_with_query_terms += 1
            
            distribution_score = chunks_with_query_terms / max(n_chunks, 1)
        else:
            distribution_score = 0.0
        
        # --- 4. Semantic field analysis ---
        # Build semantic fields: groups of related terms that should co-occur
        # Use character n-gram overlap as a proxy for semantic relatedness
        def char_ngrams(word, n=3):
            """Generate character n-grams."""
            if len(word) < n:
                return {word}
            return {word[i:i+n] for i in range(len(word) - n + 1)}
        
        def char_similarity(w1, w2):
            """Character n-gram Jaccard similarity."""
            ng1 = char_ngrams(w1)
            ng2 = char_ngrams(w2)
            if not ng1 or not ng2:
                return 0.0
            return len(ng1 & ng2) / len(ng1 | ng2)
        
        # Find response terms that are morphologically similar to query terms
        semantic_matches = 0
        for qt in query_content_unique:
            for rt in response_content_set:
                if qt == rt:
                    continue
                sim = char_similarity(qt, rt)
                if sim > 0.45:  # Threshold for morphological similarity
                    semantic_matches += 1
                    break
        
        semantic_score = semantic_matches / max(len(query_content_unique), 1)
        
        # --- 5. Intent alignment ---
        # Detect query type and check if response matches expected pattern
        query_lower = query.lower().strip()
        response_lower_strip = response.lower().strip()
        
        intent_bonus = 0.0
        
        # Question detection
        is_question = any(query_lower.startswith(w) for w in [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'do',
            'does', 'is', 'are', 'should', 'could', 'would', 'will'
        ]) or query.strip().endswith('?')
        
        # Request/command detection
        is_request = any(query_lower.startswith(w) for w in [
            'i need', 'i want', 'i wanna', 'help me', 'please', 'give me',
            'tell me', 'show me', 'find', 'list', 'suggest', 'recommend',
            'i\'m preparing', 'i\'m looking', 'i love'
        ])
        
        if is_question:
            # Check if response attempts to answer (not just restate the question)
            # Look for answer indicators
            answer_indicators = [
                'yes', 'no', 'because', 'the reason', 'this is', 'here are',
                'there are', 'it is', 'they are', 'you can', 'you should',
                'the answer', 'to answer', 'in short', 'specifically',
                'essentially', 'basically', 'primarily'
            ]
            if any(ind in response_lower_strip for ind in answer_indicators):
                intent_bonus += 0.3
            
            # Penalize if response just asks more questions without answering
            response_questions = response.count('?')
            if response_questions > 3 and len(response) < 500:
                intent_bonus -= 0.2
        
        if is_request:
            # Check for actionable content
            action_indicators = [
                'here', 'step', 'first', 'start', 'begin', 'try', 'consider',
                'option', 'approach', 'method', 'way', 'tip', 'suggestion',
                'recipe', 'idea', 'plan'
            ]
            if any(ind in response_lower_strip for ind in action_indicators):
                intent_bonus += 0.3
        
        # --- 6. Topic coherence score ---
        # Measure how much the response stays on topic by checking
        # the ratio of response content words that relate to query topics
        query_content_counter = Counter(query_content)
        response_content_counter = Counter(response_content)
        
        # Get top query terms (most frequent or important)
        top_query_terms = set(query_content_unique)
        
        # Count response words that are query-related (exact or morphologically similar)
        related_response_words = 0
        total_response_content = len(response_content)
        
        for rword in response_content:
            if rword in top_query_terms:
                related_response_words += 1
            else:
                for qt in top_query_terms:
                    if char_similarity(rword, qt) > 0.5:
                        related_response_words += 0.5
                        break
        
        topic_coherence = related_response_words / max(total_response_content, 1)
        # Cap it - we don't want 100% overlap, that would be parroting
        topic_coherence = min(topic_coherence, 0.5) * 2  # Scale to 0-1
        
        # --- 7. Response quality signals ---
        quality_bonus = 0.0
        
        # Structured response (numbered lists, bullet points, headers)
        has_structure = bool(re.search(r'(\d+[\.\)]\s|\*\*|###|^\s*[-•])', response, re.MULTILINE))
        if has_structure and len(response) > 100:
            quality_bonus += 0.15
        
        # Response length adequacy
        # Very short responses for complex queries are usually worse
        query_complexity = len(query_content_unique)
        response_length = len(response)
        
        if query_complexity > 5 and response_length < 100:
            quality_bonus -= 0.2
        elif response_length > 200:
            quality_bonus += 0.1
        
        # Check for direct address of query subject
        # Extract potential key entities (capitalized words, quoted terms, specific nouns)
        query_entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        query_entities = [e.lower() for e in query_entities if len(e) > 2]
        
        entity_coverage = 0.0
        if query_entities:
            found = sum(1 for e in query_entities if e in response_lower_strip)
            entity_coverage = found / len(query_entities)
        else:
            entity_coverage = 0.5  # Neutral if no entities detected
        
        # --- 8. Engagement and directness ---
        # Does the response acknowledge the query directly?
        directness_score = 0.0
        
        # Check first sentence/paragraph for query term presence
        first_para = response.split('\n')[0] if '\n' in response else response[:min(200, len(response))]
        first_para_tokens = set(tokenize(first_para))
        first_para_overlap = len(first_para_tokens & set(query_content)) / max(len(set(query_content)), 1)
        directness_score = first_para_overlap
        
        # --- Combine all scores ---
        # Weighted combination
        final_score = (
            weighted_coverage * 25.0 +      # 0-25: query term coverage (IDF-weighted)
            positional_score * 10.0 +         # 0-10: early mention of key terms
            distribution_score * 10.0 +       # 0-10: even distribution of query terms
            semantic_score * 10.0 +            # 0-10: morphological/semantic matches
            intent_bonus * 10.0 +              # -2 to 3: intent alignment
            topic_coherence * 10.0 +           # 0-10: staying on topic
            entity_coverage * 10.0 +           # 0-10: entity coverage
            directness_score * 10.0 +          # 0-10: direct first-paragraph relevance
            quality_bonus * 10.0               # -2 to 2.5: quality signals
        )
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Never crash - return a neutral score
        return 25.0