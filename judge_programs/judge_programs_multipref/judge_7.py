def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query term importance weighting, semantic field coverage, and response
    coherence analysis based on sentence-level relevance distribution.
    
    This variant focuses on:
    1. Query decomposition into weighted terms (IDF-like weighting)
    2. Sentence-level relevance scoring (each sentence scored against query)
    3. Relevance distribution analysis (are relevant parts spread throughout?)
    4. Question-type detection and answer-pattern matching
    5. Transition/cohesion signals indicating structured addressing of query
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
        
        # --- Utility functions ---
        
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
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'down',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'myself', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'am', 'also', 'any', 'much', 'many', 'like', 'get', 'got',
        }
        
        def tokenize(text):
            """Tokenize text into lowercase words."""
            return re.findall(r'[a-z][a-z\']*[a-z]|[a-z]', text.lower())
        
        def content_words(tokens):
            """Filter to content words only."""
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        def split_sentences(text):
            """Split text into sentences."""
            # Split on sentence-ending punctuation, bullet points, numbered items
            sents = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*(?=\d+[\.\)]|\*|\-\s)', text)
            # Also split on newlines that separate paragraphs
            result = []
            for s in sents:
                parts = re.split(r'\n\s*\n', s)
                result.extend(parts)
            return [s.strip() for s in result if s.strip() and len(s.strip()) > 3]
        
        # --- 1. Query Analysis & Term Weighting ---
        
        query_tokens = tokenize(query)
        query_content = content_words(query_tokens)
        
        if not query_content:
            query_content = query_tokens[:10] if query_tokens else ['']
        
        response_tokens = tokenize(response)
        response_content = content_words(response_tokens)
        
        # Compute term frequency in response
        resp_tf = Counter(response_content)
        total_resp_words = max(len(response_content), 1)
        
        # Weight query terms by rarity (inverse document frequency approximation)
        # Rarer query terms are more important for relevance
        query_term_counts = Counter(query_content)
        
        # Approximate IDF: shorter, less common words get higher weight
        # We use response length normalization
        def term_importance(term):
            """Higher importance for terms that are specific/rare."""
            base = 1.0
            # Longer words tend to be more specific
            if len(term) >= 6:
                base += 0.5
            if len(term) >= 8:
                base += 0.5
            # Terms appearing multiple times in query are more central
            base += 0.3 * (query_term_counts[term] - 1)
            return base
        
        query_term_weights = {}
        for t in set(query_content):
            query_term_weights[t] = term_importance(t)
        
        # Normalize weights
        total_weight = sum(query_term_weights.values())
        if total_weight > 0:
            for t in query_term_weights:
                query_term_weights[t] /= total_weight
        
        # --- 2. Weighted Query Term Coverage ---
        
        weighted_coverage = 0.0
        for term, weight in query_term_weights.items():
            if term in resp_tf:
                # Diminishing returns for repeated mentions
                mention_score = min(1.0, resp_tf[term] / max(1, 2))
                weighted_coverage += weight * mention_score
        
        # Also check for partial matches (stemming approximation)
        partial_bonus = 0.0
        uncovered_terms = [t for t in query_term_weights if t not in resp_tf]
        for qt in uncovered_terms:
            best_match = 0.0
            qt_prefix = qt[:max(4, len(qt) - 2)]
            for rt in set(response_content):
                rt_prefix = rt[:max(4, len(rt) - 2)]
                if qt_prefix == rt_prefix and len(qt) > 3:
                    match_score = len(qt_prefix) / max(len(qt), len(rt))
                    best_match = max(best_match, match_score)
            partial_bonus += query_term_weights[qt] * best_match * 0.6
        
        coverage_score = min(1.0, weighted_coverage + partial_bonus)
        
        # --- 3. Sentence-Level Relevance Distribution ---
        
        sentences = split_sentences(response)
        if not sentences:
            sentences = [response]
        
        query_content_set = set(query_content)
        
        # Expand query set with morphological variants
        expanded_query = set(query_content_set)
        for q in query_content_set:
            if q.endswith('s') and len(q) > 3:
                expanded_query.add(q[:-1])
            elif q.endswith('ing') and len(q) > 5:
                expanded_query.add(q[:-3])
                expanded_query.add(q[:-3] + 'e')
            elif q.endswith('ed') and len(q) > 4:
                expanded_query.add(q[:-2])
                expanded_query.add(q[:-1])
            elif q.endswith('tion') and len(q) > 5:
                expanded_query.add(q[:-4] + 't')
                expanded_query.add(q[:-4] + 'te')
            # Add the word plus common suffixes
            expanded_query.add(q + 's')
            expanded_query.add(q + 'ing')
            expanded_query.add(q + 'ed')
        
        sentence_relevance_scores = []
        for sent in sentences:
            sent_tokens = set(content_words(tokenize(sent)))
            if not sent_tokens:
                sentence_relevance_scores.append(0.0)
                continue
            
            overlap = sent_tokens & expanded_query
            # Relevance = proportion of query terms found in this sentence
            sent_rel = len(overlap) / max(len(query_content_set), 1)
            # Also consider what fraction of the sentence is relevant
            density = len(overlap) / max(len(sent_tokens), 1)
            
            sentence_relevance_scores.append(0.6 * sent_rel + 0.4 * density)
        
        if sentence_relevance_scores:
            # Average sentence relevance
            avg_sent_rel = sum(sentence_relevance_scores) / len(sentence_relevance_scores)
            
            # Max sentence relevance (at least one highly relevant sentence)
            max_sent_rel = max(sentence_relevance_scores)
            
            # Relevance spread: what fraction of sentences have some relevance?
            relevant_sent_count = sum(1 for s in sentence_relevance_scores if s > 0.05)
            relevance_spread = relevant_sent_count / max(len(sentence_relevance_scores), 1)
            
            # Check if first sentence is relevant (important for direct addressing)
            first_sent_rel = sentence_relevance_scores[0] if sentence_relevance_scores else 0
            
            distribution_score = (
                0.3 * avg_sent_rel +
                0.25 * max_sent_rel +
                0.25 * relevance_spread +
                0.2 * min(1.0, first_sent_rel * 2)
            )
        else:
            distribution_score = 0.0
        
        # --- 4. Question Type Detection & Answer Pattern Matching ---
        
        question_type_score = 0.0
        query_lower = query.lower().strip()
        
        # Detect question type
        is_how = bool(re.search(r'\bhow\b', query_lower))
        is_what = bool(re.search(r'\bwhat\b', query_lower))
        is_why = bool(re.search(r'\bwhy\b', query_lower))
        is_can_should = bool(re.search(r'\b(can|should|could|would|do you think|is it)\b', query_lower))
        is_list_request = bool(re.search(r'\b(ideas?|suggestions?|tips?|ways?|steps?|list|examples?)\b', query_lower))
        is_help_request = bool(re.search(r'\b(help|assist|guide|teach|learn|figure out|need to)\b', query_lower))
        is_explain = bool(re.search(r'\b(explain|describe|tell me about|what happens)\b', query_lower))
        
        resp_lower = response.lower()
        
        # Check for appropriate answer patterns
        if is_how:
            # "How" questions should have procedural/explanatory content
            has_steps = bool(re.search(r'(\d+[\.\)]\s|step\s*\d|first|second|third|then|next|finally)', resp_lower))
            has_explanation = bool(re.search(r'(because|by\s+\w+ing|you can|you need|this\s+(is|works|means))', resp_lower))
            question_type_score = 0.7 if has_steps else (0.4 if has_explanation else 0.1)
        
        elif is_why:
            # "Why" questions should have causal explanations
            has_because = bool(re.search(r'(because|reason|due to|since|as a result|this is why|cause)', resp_lower))
            question_type_score = 0.7 if has_because else 0.2
        
        elif is_can_should:
            # Opinion/possibility questions should have a stance
            has_stance = bool(re.search(r'(yes|no|i (think|believe|don\'t)|it (is|isn\'t|should|shouldn\'t)|absolutely|definitely|not\s+\w+\s+(to|for))', resp_lower))
            question_type_score = 0.6 if has_stance else 0.2
        
        elif is_list_request:
            # List requests should have enumerated items
            list_items = len(re.findall(r'(\d+[\.\)]\s|\*\s|\-\s|•)', response))
            question_type_score = min(1.0, list_items / 4) if list_items > 0 else 0.15
        
        elif is_help_request:
            # Help requests should have actionable advice
            has_advice = bool(re.search(r'(you can|you should|try|consider|start|make sure|here\'s|here are)', resp_lower))
            question_type_score = 0.6 if has_advice else 0.2
        
        elif is_what:
            # "What" questions should have definitional/informational content
            has_info = bool(re.search(r'(is a|are |refers to|means|involves|includes|consists)', resp_lower))
            question_type_score = 0.5 if has_info else 0.2
        
        else:
            # Generic: check if response seems to engage with the topic
            question_type_score = 0.3
        
        # --- 5. Topical Coherence via Shared Semantic Field ---
        
        # Build "semantic field" from query: extract key noun phrases and check response
        # Extract bigrams from query as potential key phrases
        query_bigrams = set()
        for i in range(len(query_tokens) - 1):
            if query_tokens[i] not in STOPWORDS or query_tokens[i+1] not in STOPWORDS:
                query_bigrams.add((query_tokens[i], query_tokens[i+1]))
        
        resp_bigrams = set()
        for i in range(len(response_tokens) - 1):
            resp_bigrams.add((response_tokens[i], response_tokens[i+1]))
        
        bigram_overlap = len(query_bigrams & resp_bigrams)
        bigram_score = min(1.0, bigram_overlap / max(len(query_bigrams), 1)) if query_bigrams else 0.3
        
        # --- 6. Response Opening Relevance ---
        # Check if the response starts by addressing the query directly
        
        opening = response[:min(200, len(response))].lower()
        opening_tokens = set(content_words(tokenize(opening)))
        opening_overlap = len(opening_tokens & expanded_query)
        opening_score = min(1.0, opening_overlap / max(len(query_content_set) * 0.5, 1))
        
        # Check for direct acknowledgment patterns
        ack_patterns = [
            r'^(yes|no|certainly|absolutely|sure|great|of course|definitely)',
            r'^(here|let|i\'ll|i can|i\'d)',
            r'(that\'s a|good question|great idea)',
        ]
        has_acknowledgment = any(re.search(p, resp_lower[:100]) for p in ack_patterns)
        if has_acknowledgment:
            opening_score = min(1.0, opening_score + 0.15)
        
        # --- 7. Penalize tangential content ---
        
        # If response has many content words not related to query at all
        response_unique = set(response_content)
        query_field = expanded_query.copy()
        
        # Build extended query field from query context
        # (words that commonly co-occur with query terms)
        on_topic_words = response_unique & query_field
        potentially_off_topic = response_unique - query_field
        
        # Don't penalize too harshly - responses naturally expand on topics
        if len(response_unique) > 0:
            on_topic_ratio = len(on_topic_words) / len(response_unique)
        else:
            on_topic_ratio = 0.0
        
        # Mild tangent penalty (only if very low overlap)
        tangent_penalty = max(0, 0.1 - on_topic_ratio) * 2  # penalty up to 0.2
        
        # --- 8. Response Engagement Quality ---
        
        # Check for transition words and structured flow (indicates organized response to query)
        transition_words = [
            'however', 'furthermore', 'additionally', 'moreover', 'therefore',
            'consequently', 'in addition', 'on the other hand', 'for example',
            'for instance', 'in particular', 'specifically', 'in contrast',
            'meanwhile', 'nevertheless', 'as a result', 'in conclusion',
            'to summarize', 'first', 'second', 'third', 'finally', 'next',
            'also', 'another', 'importantly'
        ]
        transition_count = sum(1 for tw in transition_words if tw in resp_lower)
        flow_score = min(1.0, transition_count / 4)
        
        # --- 9. Length appropriateness ---
        
        resp_len = len(response)
        query_len = len(query)
        
        # Very short responses to complex queries are bad
        if query_len > 50 and resp_len < 50:
            length_penalty = 0.3
        elif resp_len < 20:
            length_penalty = 0.5
        else:
            length_penalty = 0.0
        
        # --- FINAL SCORING ---
        
        # Combine all components with weights
        final_score = (
            coverage_score * 25 +          # Query term coverage (0-25)
            distribution_score * 20 +       # Relevance distribution across sentences (0-20)
            question_type_score * 15 +      # Answer pattern matching (0-15)
            bigram_score * 10 +             # Bigram/phrase overlap (0-10)
            opening_score * 15 +            # Opening relevance (0-15)
            flow_score * 8 +                # Structural flow (0-8)
            on_topic_ratio * 7 -            # On-topic ratio (0-7)
            tangent_penalty * 10 -          # Tangent penalty (0-2)
            length_penalty * 10             # Length penalty (0-5)
        )
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Never crash - return a neutral score
        return 25.0