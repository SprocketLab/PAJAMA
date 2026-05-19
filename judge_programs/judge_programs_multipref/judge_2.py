def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query term importance weighting, bigram matching, and discourse coherence signals.
    
    This variant focuses on:
    1. Query term IDF-weighted coverage (not simple overlap)
    2. Bigram/trigram phrase matching between query and response
    3. Query intent detection and response alignment
    4. Discourse markers indicating direct addressing
    5. Information density relative to query complexity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 5:
            return 0.0
        
        # --- Tokenization ---
        def tokenize(text):
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Common stopwords - more extensive list
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
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'it',
            'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'this', 'that', 'these', 'those', 'what',
            'which', 'who', 'whom', 'am', 'im', 'ive', 'dont', 'doesnt', 'also',
            'any', 'get', 'got', 'much', 'many', 'like', 'make', 'made'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # Content words (non-stopwords)
        query_content = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        response_content = [t for t in response_tokens if t not in stopwords and len(t) > 1]
        
        if not query_content:
            query_content = query_tokens[:5]  # fallback
        
        # --- Feature 1: IDF-weighted query term coverage ---
        # Simulate IDF: shorter, rarer-looking words get moderate weight;
        # longer, more specific words get higher weight
        def pseudo_idf(word):
            """Longer words and words with specific patterns are more informative."""
            base = math.log(2 + len(word))
            # Words with numbers are often very specific
            if any(c.isdigit() for c in word):
                base *= 1.5
            # Very short words are less informative
            if len(word) <= 3:
                base *= 0.6
            return base
        
        response_content_set = set(response_content)
        response_token_set = set(response_tokens)
        
        # Also check for partial/stem matching
        def fuzzy_match(query_word, response_set):
            """Check if query word appears in response, including stem-like matching."""
            if query_word in response_set:
                return 1.0
            # Check if query word is a substring of any response word or vice versa (min 4 chars)
            if len(query_word) >= 4:
                for rw in response_set:
                    if len(rw) >= 4:
                        # Shared prefix of at least 4 characters
                        min_len = min(len(query_word), len(rw))
                        shared = 0
                        for i in range(min_len):
                            if query_word[i] == rw[i]:
                                shared += 1
                            else:
                                break
                        if shared >= 4 and shared >= 0.7 * min(len(query_word), len(rw)):
                            return 0.8
            return 0.0
        
        total_idf = 0.0
        matched_idf = 0.0
        for qw in query_content:
            idf = pseudo_idf(qw)
            total_idf += idf
            match_score = fuzzy_match(qw, response_token_set)
            matched_idf += idf * match_score
        
        idf_coverage = matched_idf / total_idf if total_idf > 0 else 0.0
        
        # --- Feature 2: Bigram and trigram phrase matching ---
        query_bigrams = get_ngrams(query_tokens, 2)
        response_bigrams = set(get_ngrams(response_tokens, 2))
        query_trigrams = get_ngrams(query_tokens, 3)
        response_trigrams = set(get_ngrams(response_tokens, 3))
        
        bigram_matches = sum(1 for bg in query_bigrams if bg in response_bigrams)
        bigram_score = bigram_matches / max(len(query_bigrams), 1)
        
        trigram_matches = sum(1 for tg in query_trigrams if tg in response_trigrams)
        trigram_score = trigram_matches / max(len(query_trigrams), 1)
        
        phrase_score = 0.6 * bigram_score + 0.4 * trigram_score
        
        # --- Feature 3: Query intent detection and response alignment ---
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Detect query type
        intent_score = 0.0
        
        # Question type detection
        is_how = query_lower.startswith('how') or ' how ' in query_lower
        is_what = query_lower.startswith('what') or ' what ' in query_lower
        is_why = query_lower.startswith('why') or ' why ' in query_lower
        is_opinion = any(w in query_lower for w in ['do you think', 'should', 'opinion', 'believe', 'feel about'])
        is_instruction = any(w in query_lower for w in ['i want to', 'i wanna', 'i need to', 'help me', 'can you help', 'how can i', 'how do i', 'how to'])
        is_factual = is_what or any(w in query_lower for w in ["what's happening", 'tell me about', 'explain', 'describe'])
        is_suggestion = any(w in query_lower for w in ['suggest', 'ideas', 'recommendation', 'any suggestions', 'tips'])
        
        # Check if response aligns with intent
        if is_opinion:
            # Opinion questions: response should take a stance or discuss perspectives
            opinion_markers = ['i believe', 'i think', 'i do not', "i don't", 'no,', 'yes,', 
                             'should not', 'should be', 'important', 'crucial']
            if any(m in response_lower for m in opinion_markers):
                intent_score += 0.7
            # Should also address the subject
            intent_score += 0.3
            
        if is_instruction or is_how:
            # Instructional: response should have steps, procedures
            step_markers = ['step', 'first', 'start by', 'begin', 'you can', "you'll need",
                          'here are', 'here\'s how', 'to get started', '1.', '1)', 'gather']
            matches = sum(1 for m in step_markers if m in response_lower)
            intent_score += min(matches * 0.15, 0.8)
            
        if is_suggestion:
            suggestion_markers = ['here are', 'you might', 'consider', 'try', 'option',
                                'idea', 'suggest', 'recommend', 'could']
            matches = sum(1 for m in suggestion_markers if m in response_lower)
            intent_score += min(matches * 0.15, 0.7)
            
        if is_factual:
            # Should provide information directly
            if len(response_content) > 15:
                intent_score += 0.3
            # Check for explanatory language
            explain_markers = ['because', 'this is', 'the reason', 'due to', 'since',
                             'it is', 'they are', 'which means', 'refers to']
            matches = sum(1 for m in explain_markers if m in response_lower)
            intent_score += min(matches * 0.1, 0.5)
        
        intent_score = min(intent_score, 1.0)
        
        # --- Feature 4: Direct addressing signals ---
        # Does the response directly engage with the query topic?
        direct_score = 0.0
        
        # Check if response echoes key query phrases early (first 30% of response)
        response_start = response_lower[:max(len(response_lower) // 3, 100)]
        
        query_content_unique = list(set(query_content))
        early_matches = sum(1 for qw in query_content_unique if qw in response_start)
        early_ratio = early_matches / max(len(query_content_unique), 1)
        direct_score += early_ratio * 0.5
        
        # Check for direct engagement phrases
        engagement_phrases = [
            'certainly', 'great question', "that's a great", 'absolutely',
            'here are', "let's", 'i can help', 'sure', 'of course'
        ]
        if any(response_lower.startswith(p) or response_lower[:50].find(p) >= 0 for p in engagement_phrases):
            direct_score += 0.2
        
        # Check if the response rephrases or echoes the query
        # (presence of query noun phrases in response)
        query_nouns = [t for t in query_content if len(t) >= 4]
        if query_nouns:
            noun_coverage = sum(1 for n in query_nouns if n in response_token_set) / len(query_nouns)
            direct_score += noun_coverage * 0.3
        
        direct_score = min(direct_score, 1.0)
        
        # --- Feature 5: Information density and structure ---
        structure_score = 0.0
        
        # Sentence count
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Good responses typically have multiple sentences
        if num_sentences >= 3:
            structure_score += 0.2
        if num_sentences >= 5:
            structure_score += 0.1
        
        # Check for structured formatting (numbered lists, headers)
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,3}\s', response)) or bool(re.search(r'\*\*[^*]+\*\*', response))
        
        if has_numbering:
            structure_score += 0.2
        if has_headers:
            structure_score += 0.15
        
        # Response length relative to query complexity
        query_complexity = len(query_content)
        response_length = len(response_content)
        
        # Adequate length for the query
        if response_length >= query_complexity * 2:
            structure_score += 0.2
        elif response_length >= query_complexity:
            structure_score += 0.1
        
        # Penalize very short responses for complex queries
        if query_complexity > 5 and response_length < 20:
            structure_score -= 0.2
        
        structure_score = max(0.0, min(structure_score, 1.0))
        
        # --- Feature 6: Topic coherence via content word distribution ---
        # Check that response content words cluster around query topics
        # rather than diverging to unrelated topics
        response_content_counter = Counter(response_content)
        
        # What fraction of the most common response words relate to query terms?
        top_response_words = [w for w, _ in response_content_counter.most_common(20)]
        
        # Build a simple topic set from query content words + related words
        query_content_set = set(query_content)
        
        topic_aligned = 0
        for rw in top_response_words:
            if rw in query_content_set:
                topic_aligned += 2
            elif any(rw.startswith(qw[:4]) or qw.startswith(rw[:4]) 
                     for qw in query_content_set if len(qw) >= 4 and len(rw) >= 4):
                topic_aligned += 1
        
        topic_coherence = min(topic_aligned / max(len(top_response_words), 1), 1.0)
        
        # --- Feature 7: Specificity bonus ---
        # Responses that use specific terms from the query domain score higher
        specificity_score = 0.0
        
        # Check for domain-specific terms (words that appear in query but are uncommon)
        specific_query_terms = [t for t in query_content if len(t) >= 5 and t not in stopwords]
        if specific_query_terms:
            specific_matches = sum(1 for t in specific_query_terms if t in response_token_set)
            specificity_score = specific_matches / len(specific_query_terms)
        
        # --- Combine all features with weights ---
        # Weights tuned for relevance assessment
        final_score = (
            3.0 * idf_coverage +        # Core: query terms covered (IDF-weighted)
            2.0 * phrase_score +          # Phrase-level matching
            1.5 * intent_score +          # Query intent alignment
            1.5 * direct_score +          # Direct addressing
            1.0 * structure_score +       # Response structure
            1.0 * topic_coherence +       # Topic coherence
            1.0 * specificity_score       # Specificity
        )
        
        # Normalize to 0-10 range
        max_possible = 3.0 + 2.0 + 1.5 + 1.5 + 1.0 + 1.0 + 1.0  # = 11.0
        normalized = (final_score / max_possible) * 10.0
        
        # Apply slight sigmoid-like transformation for better discrimination
        # This spreads out the middle range
        midpoint = 5.0
        steepness = 0.5
        transformed = 10.0 / (1.0 + math.exp(-steepness * (normalized - midpoint)))
        
        return round(transformed, 4)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            # Last resort: simple word overlap
            q_words = set(query.lower().split())
            r_words = set(response.lower().split())
            overlap = len(q_words & r_words) / max(len(q_words), 1)
            return round(overlap * 5.0, 4)
        except:
            return 3.0