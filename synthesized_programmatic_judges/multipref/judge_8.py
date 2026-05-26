def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query term importance weighting, semantic field expansion, and 
    response coherence analysis based on sentence-level relevance distribution.
    
    This variant focuses on:
    1. Query decomposition into weighted terms (IDF-like weighting)
    2. Sentence-level relevance scoring across the response
    3. Coverage uniformity (relevant content spread throughout response)
    4. Query intent detection and fulfillment signals
    5. Penalization for filler/generic content ratio
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
        def tokenize(text):
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def get_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text)
            return [s.strip() for s in sents if len(s.strip()) > 10]
        
        # Common English stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'up',
            'about', 'also', 'well', 'back', 'even', 'still', 'new', 'any',
            'many', 'much', 'get', 'got', 'go', 'going', 'make', 'like',
            'think', 'know', 'see', 'come', 'take', 'want', 'give', 'tell',
            'say', 'said', 'one', 'two', 'first', 'way', 'thing', 'things',
            'let', 'sure', 'really', 'right', 'now', 'something', 'down',
            'im', 'dont', 'ive', 'youre', 'thats', 'wont', 'cant', 'doesnt',
            'am', 'however', 'also', 'another', 'since', 'until', 'whether',
        }
        
        # Generic filler phrases that indicate low-content responses
        filler_phrases = [
            'here are some', 'here is a', 'i hope this helps', 'feel free to',
            'let me know if', 'happy to help', 'great question', 'good question',
            'that\'s a great', 'that\'s a good', 'absolutely', 'of course',
            'no problem', 'you\'re welcome', 'glad you asked',
        ]
        
        # --- 1. Query Analysis: Extract and weight query terms ---
        query_tokens = tokenize(query)
        query_content_tokens = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        
        if not query_content_tokens:
            query_content_tokens = [t for t in query_tokens if len(t) > 1]
        
        if not query_content_tokens:
            return 5.0  # Can't evaluate relevance without query terms
        
        # Approximate IDF-like weights: rarer/longer words get higher weight
        # Also boost words that appear to be domain-specific
        def term_importance(term):
            """Estimate importance of a query term."""
            base = 1.0
            # Length bonus (longer words tend to be more specific)
            if len(term) >= 8:
                base += 0.5
            elif len(term) >= 5:
                base += 0.25
            # Numeric terms might be important (specific values)
            if any(c.isdigit() for c in term):
                base += 0.3
            # Capitalized in original query suggests proper noun/importance
            if term.capitalize() in query or term.upper() in query:
                base += 0.2
            return base
        
        query_term_weights = {}
        for t in query_content_tokens:
            query_term_weights[t] = query_term_weights.get(t, 0) + term_importance(t)
        
        # Normalize weights
        total_weight = sum(query_term_weights.values())
        if total_weight > 0:
            query_term_weights = {k: v / total_weight for k, v in query_term_weights.items()}
        
        response_tokens = tokenize(response)
        response_content_tokens = [t for t in response_tokens if t not in stopwords and len(t) > 1]
        response_token_set = set(response_tokens)
        response_content_set = set(response_content_tokens)
        response_token_counts = Counter(response_tokens)
        
        # --- 2. Weighted Term Coverage ---
        # How much of the query's weighted importance is covered by the response?
        covered_weight = 0.0
        for term, weight in query_term_weights.items():
            if term in response_token_set:
                covered_weight += weight
            else:
                # Check for partial/stem matches
                for rt in response_content_set:
                    if len(term) >= 4 and len(rt) >= 4:
                        # Shared prefix of length >= 4
                        shared = 0
                        for c1, c2 in zip(term, rt):
                            if c1 == c2:
                                shared += 1
                            else:
                                break
                        if shared >= min(len(term), 4):
                            covered_weight += weight * 0.6
                            break
        
        term_coverage_score = covered_weight  # 0 to 1
        
        # --- 3. Sentence-Level Relevance Distribution ---
        sentences = get_sentences(response)
        if not sentences:
            sentences = [response]
        
        query_content_set = set(query_content_tokens)
        
        def sentence_relevance(sent):
            """Score how relevant a single sentence is to the query."""
            sent_tokens = set(tokenize(sent))
            if not sent_tokens:
                return 0.0
            
            # Weighted overlap
            overlap_score = 0.0
            for term, weight in query_term_weights.items():
                if term in sent_tokens:
                    overlap_score += weight
                else:
                    # Stem match
                    for st in sent_tokens:
                        if len(term) >= 4 and len(st) >= 4:
                            shared = 0
                            for c1, c2 in zip(term, st):
                                if c1 == c2:
                                    shared += 1
                                else:
                                    break
                            if shared >= min(len(term), 4):
                                overlap_score += weight * 0.4
                                break
            return overlap_score
        
        sent_scores = [sentence_relevance(s) for s in sentences]
        
        # Average sentence relevance
        avg_sent_relevance = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
        
        # Fraction of sentences that are at least somewhat relevant
        relevant_sent_fraction = sum(1 for s in sent_scores if s > 0.05) / len(sent_scores) if sent_scores else 0.0
        
        # --- 4. Relevance Distribution Uniformity ---
        # Good responses maintain relevance throughout, not just at the start
        if len(sent_scores) >= 3:
            third = len(sent_scores) // 3
            first_third = sum(sent_scores[:third]) / max(third, 1)
            last_third = sum(sent_scores[-third:]) / max(third, 1)
            mid_third = sum(sent_scores[third:2*third]) / max(third, 1)
            
            # Penalize if relevance drops off significantly
            if first_third > 0:
                distribution_score = min(1.0, (last_third + mid_third) / (2 * first_third + 0.001))
            else:
                distribution_score = 0.5
        else:
            distribution_score = 0.7  # Short responses get neutral score
        
        # --- 5. Query Intent Fulfillment ---
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Detect question type and check for appropriate response patterns
        intent_score = 0.5  # baseline
        
        # How/What/Why questions should have explanatory content
        if re.search(r'\b(how|what|why|explain|describe)\b', query_lower):
            # Check for explanatory markers
            explanatory_markers = [
                r'\bbecause\b', r'\bdue to\b', r'\breason\b', r'\bthis means\b',
                r'\bin order to\b', r'\bby\b', r'\bthrough\b', r'\bprocess\b',
                r'\bstep\b', r'\bfirst\b', r'\bthen\b', r'\bnext\b',
                r'\bresult\b', r'\bcause\b', r'\bmethod\b', r'\bway\b',
            ]
            marker_count = sum(1 for m in explanatory_markers if re.search(m, response_lower))
            intent_score = min(1.0, 0.3 + marker_count * 0.1)
        
        # Yes/No questions should have a clear stance
        if re.search(r'\b(should|would|could|is it|are they|do you|can you|does)\b', query_lower) and query.strip().endswith('?'):
            if re.search(r'\b(yes|no|i do|i don\'t|i believe|i think|absolutely|certainly|not)\b', response_lower):
                intent_score = max(intent_score, 0.7)
        
        # List/suggestion requests
        if re.search(r'\b(suggest|ideas|tips|ways|examples|list|recommend)\b', query_lower):
            # Check for enumeration
            list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|\*|-|•)', response))
            if list_items >= 3:
                intent_score = max(intent_score, 0.8)
            elif list_items >= 1:
                intent_score = max(intent_score, 0.6)
        
        # --- 6. Filler/Generic Content Penalty ---
        filler_count = sum(1 for fp in filler_phrases if fp in response_lower)
        filler_penalty = min(0.15, filler_count * 0.03)
        
        # --- 7. Semantic Field Coherence ---
        # Build a simple "semantic field" from query terms by looking at co-occurring
        # content words in the response that relate to query terms
        # Check if response introduces topically related vocabulary
        
        # Use character trigram overlap between query and response as a proxy
        # for semantic relatedness beyond exact word matching
        def char_trigrams(text):
            text = ' '.join(tokenize(text))
            return set(text[i:i+3] for i in range(len(text)-2))
        
        query_trigrams = char_trigrams(query)
        response_trigrams = char_trigrams(response[:1000])  # Limit for performance
        
        if query_trigrams:
            trigram_overlap = len(query_trigrams & response_trigrams) / len(query_trigrams)
        else:
            trigram_overlap = 0.0
        
        # --- 8. Response Directness ---
        # Check if the response addresses the query early (first sentence/paragraph)
        first_sentence = sentences[0] if sentences else response[:200]
        first_sent_relevance = sentence_relevance(first_sentence)
        
        # Bonus for being direct (addressing query in first sentence)
        directness_bonus = min(0.15, first_sent_relevance * 0.3)
        
        # --- 9. Specificity Score ---
        # Responses with more specific/unique content words score higher
        if response_content_tokens:
            unique_ratio = len(response_content_set) / len(response_content_tokens)
            # Very repetitive responses are penalized
            specificity = min(1.0, unique_ratio * 1.5)
        else:
            specificity = 0.0
        
        # --- 10. Response Length Appropriateness ---
        # Not too short, not just padding
        resp_len = len(response_tokens)
        if resp_len < 20:
            length_factor = 0.6
        elif resp_len < 50:
            length_factor = 0.8
        elif resp_len < 300:
            length_factor = 1.0
        else:
            length_factor = 0.95  # Slight penalty for very long (might be padded)
        
        # --- Combine Scores ---
        # Weighted combination of all factors
        final_score = (
            term_coverage_score * 25.0 +       # 0-25: core term coverage
            avg_sent_relevance * 15.0 +          # 0-15: average sentence relevance
            relevant_sent_fraction * 10.0 +      # 0-10: fraction of relevant sentences
            distribution_score * 8.0 +           # 0-8: relevance distribution
            intent_score * 12.0 +                # 0-12: intent fulfillment
            trigram_overlap * 10.0 +             # 0-10: character-level similarity
            directness_bonus * 40.0 +            # 0-6: directness bonus
            specificity * 8.0 +                  # 0-8: vocabulary specificity
            length_factor * 6.0 -                # 0-6: length appropriateness
            filler_penalty * 40.0                # penalty for filler
        )
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 5.0