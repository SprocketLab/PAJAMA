def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a semantic coverage approach
    based on query decomposition, topic modeling via co-occurrence clusters, and
    information density analysis. This is fundamentally different from overlap/Jaccard,
    TF-IDF, cosine similarity, n-gram, or word length approaches.
    
    Strategy: 
    1. Extract query "information needs" (question words, key noun phrases, entities)
    2. Build a topic signature from the query using word associations and context windows
    3. Score response on how well it addresses each identified information need
    4. Measure discourse coherence (does the response flow as an answer?)
    5. Penalize generic/boilerplate content and reward specific, on-topic elaboration
    """
    try:
        import re
        import math
        from collections import Counter, defaultdict
        
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        # --- Preprocessing ---
        def tokenize(text):
            return re.findall(r"[a-zA-Z0-9']+", text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 5]
        
        STOP_WORDS = {
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
            'it', 'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'like', 'get', 'got', 'much', 'many',
            'really', 'well', 'even', 'still', 'already', 'yet', 'though',
            'although', 'however', 'since', 'unless', 'whether', 'am', 'im',
            'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
            'ive', 'youre', 'theyre', 'weve', 'theyve', 'ill', 'youll', 'hell',
            'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt', 'werent',
            'hasnt', 'havent', 'hadnt', 'cant', 'been', 'being', 'having',
            'doing', 'going', 'make', 'made', 'thing', 'things', 'something',
            'anything', 'everything', 'nothing', 'one', 'two', 'three',
        }
        
        # Generic/boilerplate phrases that indicate low-quality responses
        BOILERPLATE_PATTERNS = [
            r'welcome to\s+/r/',
            r'please read our rules',
            r'your comments will be removed',
            r'while we do not require',
            r'this (post|thread|comment) has been',
            r'i am a bot',
            r'automod',
        ]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 0.0
        
        query_content = [w for w in query_tokens if w not in STOP_WORDS and len(w) > 1]
        response_content = [w for w in response_tokens if w not in STOP_WORDS and len(w) > 1]
        
        if not query_content:
            query_content = query_tokens[:10]
        
        # --- 1. Boilerplate Detection ---
        boilerplate_score = 0.0
        response_lower = response.lower()
        for pattern in BOILERPLATE_PATTERNS:
            if re.search(pattern, response_lower):
                boilerplate_score += 0.3
        boilerplate_score = min(boilerplate_score, 1.0)
        
        # --- 2. Query Intent Decomposition ---
        # Identify what type of question and what entities/concepts are being asked about
        
        # Detect question type
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'could', 'should', 'would', 'will'}
        query_lower = query.lower()
        
        # Extract key phrases using adjacency (bigrams of content words)
        def extract_key_phrases(tokens, content_set):
            """Extract meaningful adjacent content word pairs."""
            phrases = []
            for i in range(len(tokens) - 1):
                if tokens[i] in content_set and tokens[i+1] in content_set:
                    phrases.append(tokens[i] + '_' + tokens[i+1])
            return phrases
        
        query_content_set = set(query_content)
        query_phrases = extract_key_phrases(query_tokens, query_content_set)
        response_phrases = extract_key_phrases(response_tokens, set(response_content))
        
        # --- 3. Query Concept Coverage ---
        # Weight query terms by position and frequency (earlier = more important for topic)
        query_term_weights = {}
        for i, term in enumerate(query_content):
            # Earlier terms get slightly more weight, rare terms get more weight
            position_weight = 1.0 / (1.0 + 0.02 * i)
            freq = query_content.count(term)
            rarity_weight = 1.0 / math.sqrt(freq)
            weight = position_weight * rarity_weight
            if term not in query_term_weights:
                query_term_weights[term] = weight
            else:
                query_term_weights[term] = max(query_term_weights[term], weight)
        
        # Normalize weights
        total_weight = sum(query_term_weights.values())
        if total_weight > 0:
            query_term_weights = {k: v / total_weight for k, v in query_term_weights.items()}
        
        # Check coverage: what fraction of weighted query concepts appear in response?
        response_content_set = set(response_content)
        coverage_score = 0.0
        for term, weight in query_term_weights.items():
            if term in response_content_set:
                coverage_score += weight
            else:
                # Check for partial matches (stemming approximation)
                for rterm in response_content_set:
                    if len(term) >= 4 and len(rterm) >= 4:
                        # Check if they share a common prefix of length >= 4
                        common_len = 0
                        for c1, c2 in zip(term, rterm):
                            if c1 == c2:
                                common_len += 1
                            else:
                                break
                        if common_len >= min(len(term), len(rterm)) * 0.7 and common_len >= 4:
                            coverage_score += weight * 0.7
                            break
        
        # --- 4. Phrase-level Topic Alignment ---
        query_phrase_set = set(query_phrases)
        response_phrase_set = set(response_phrases)
        if query_phrase_set:
            phrase_overlap = len(query_phrase_set & response_phrase_set) / len(query_phrase_set)
        else:
            phrase_overlap = 0.0
        
        # --- 5. Contextual Window Analysis ---
        # For each key query concept, check if it appears in the response surrounded by
        # related context (not just mentioned in passing)
        def get_context_windows(tokens, target_terms, window_size=5):
            """Get context windows around target terms."""
            windows = []
            for i, token in enumerate(tokens):
                if token in target_terms:
                    start = max(0, i - window_size)
                    end = min(len(tokens), i + window_size + 1)
                    context = set(tokens[start:end]) - STOP_WORDS - {token}
                    windows.append((token, context))
            return windows
        
        # Get top query terms
        top_query_terms = sorted(query_term_weights.items(), key=lambda x: x[1], reverse=True)[:15]
        top_query_term_set = {t[0] for t in top_query_terms}
        
        query_windows = get_context_windows(query_tokens, top_query_term_set)
        response_windows = get_context_windows(response_tokens, top_query_term_set)
        
        # Build context profiles
        query_context = defaultdict(set)
        for term, ctx in query_windows:
            query_context[term].update(ctx)
        
        response_context = defaultdict(set)
        for term, ctx in response_windows:
            response_context[term].update(ctx)
        
        # Measure contextual alignment
        context_alignment = 0.0
        context_count = 0
        for term in top_query_term_set:
            if term in response_context and term in query_context:
                q_ctx = query_context[term]
                r_ctx = response_context[term]
                if q_ctx and r_ctx:
                    overlap = len(q_ctx & r_ctx)
                    union = len(q_ctx | r_ctx)
                    if union > 0:
                        context_alignment += overlap / union
                    context_count += 1
                elif r_ctx:
                    # Term found in response with some context, partial credit
                    context_alignment += 0.2
                    context_count += 1
            elif term in response_context:
                context_alignment += 0.1
                context_count += 1
        
        if context_count > 0:
            context_alignment /= context_count
        
        # --- 6. Response Specificity & Information Density ---
        # Responses with more specific/rare content words are generally better
        response_word_lengths = [len(w) for w in response_content]
        avg_word_length = sum(response_word_lengths) / max(len(response_word_lengths), 1)
        
        # Unique content word ratio (diversity)
        if response_content:
            vocab_richness = len(set(response_content)) / len(response_content)
        else:
            vocab_richness = 0.0
        
        # Response length factor (moderate length preferred)
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Ideal response is at least as long as query, but not excessively short
        if resp_len < 10:
            length_factor = resp_len / 10.0
        elif resp_len < 30:
            length_factor = 0.6 + 0.4 * (resp_len - 10) / 20
        else:
            length_factor = min(1.0, 0.8 + 0.2 * min(resp_len, 200) / 200)
        
        # --- 7. Sentence-level Relevance Distribution ---
        # Check if multiple sentences in the response are relevant (not just one mention)
        response_sentences = get_sentences(response)
        query_content_lower = set(query_content)
        
        relevant_sentence_count = 0
        total_sentence_relevance = 0.0
        
        for sent in response_sentences:
            sent_tokens = set(tokenize(sent)) - STOP_WORDS
            if sent_tokens:
                sent_relevance = len(sent_tokens & query_content_lower) / max(len(query_content_lower), 1)
                # Also check for approximate matches
                approx_matches = 0
                for qt in query_content_lower:
                    for st in sent_tokens:
                        if len(qt) >= 4 and len(st) >= 4:
                            common = 0
                            for c1, c2 in zip(qt, st):
                                if c1 == c2:
                                    common += 1
                                else:
                                    break
                            if common >= 4 and common >= min(len(qt), len(st)) * 0.7:
                                approx_matches += 1
                                break
                
                total_relevance = sent_relevance + approx_matches / max(len(query_content_lower), 1) * 0.5
                if total_relevance > 0.05:
                    relevant_sentence_count += 1
                total_sentence_relevance += total_relevance
        
        if response_sentences:
            sentence_coverage_ratio = relevant_sentence_count / len(response_sentences)
            avg_sentence_relevance = total_sentence_relevance / len(response_sentences)
        else:
            sentence_coverage_ratio = 0.0
            avg_sentence_relevance = 0.0
        
        # --- 8. Direct Address Detection ---
        # Does the response directly engage with the question rather than deflecting?
        direct_address_indicators = [
            r'\b(yes|no|essentially|basically|actually|specifically)\b',
            r'\b(the answer|to answer|in short|in summary)\b',
            r'\b(because|since|the reason|this is due to)\b',
            r'\b(for example|for instance|such as|e\.g\.)\b',
            r'\b(first|second|third|additionally|furthermore|moreover)\b',
        ]
        
        deflection_indicators = [
            r'\b(i (don\'?t|cant|cannot) (help|answer|respond))\b',
            r'\b(not sure|no idea|can\'?t say)\b',
            r'\b(please read|see the rules|check the sidebar)\b',
            r'\b(this (has been|was) (removed|deleted))\b',
        ]
        
        direct_score = 0.0
        for pattern in direct_address_indicators:
            if re.search(pattern, response_lower):
                direct_score += 0.15
        direct_score = min(direct_score, 0.6)
        
        deflection_score = 0.0
        for pattern in deflection_indicators:
            if re.search(pattern, response_lower):
                deflection_score += 0.25
        deflection_score = min(deflection_score, 0.8)
        
        # --- 9. Topic Expansion Score ---
        # Good responses often introduce new but related terms not in the query
        # Check if response content words are topically coherent with query
        response_only_terms = response_content_set - query_content_set
        
        # For response-only terms, check if they co-occur with query terms in the response
        expansion_coherence = 0.0
        expansion_count = 0
        
        response_token_list = response_tokens
        for i, token in enumerate(response_token_list):
            if token in response_only_terms and token not in STOP_WORDS and len(token) > 2:
                # Check nearby tokens for query terms
                start = max(0, i - 8)
                end = min(len(response_token_list), i + 9)
                nearby = set(response_token_list[start:end])
                if nearby & top_query_term_set:
                    expansion_coherence += 1
                expansion_count += 1
        
        if expansion_count > 0:
            expansion_ratio = expansion_coherence / expansion_count
        else:
            expansion_ratio = 0.0
        
        # --- 10. Combine All Signals ---
        # Weighted combination with emphasis on coverage and context
        
        score = (
            coverage_score * 25.0 +              # How much of query is addressed (0-25)
            phrase_overlap * 10.0 +                # Phrase-level alignment (0-10)
            context_alignment * 12.0 +             # Contextual depth (0-12)
            avg_sentence_relevance * 10.0 +        # Per-sentence relevance (0-10)
            sentence_coverage_ratio * 5.0 +        # Distribution of relevance (0-5)
            direct_score * 8.0 +                   # Direct engagement (0-4.8)
            expansion_ratio * 8.0 +                # Coherent topic expansion (0-8)
            length_factor * 5.0 +                  # Adequate length (0-5)
            vocab_richness * 5.0 +                 # Information density (0-5)
            min(avg_word_length / 8.0, 1.0) * 3.0  # Word sophistication (0-3)
        )
        
        # Apply penalties
        score *= (1.0 - boilerplate_score * 0.7)   # Boilerplate penalty
        score *= (1.0 - deflection_score * 0.5)     # Deflection penalty
        
        # Very short responses get additional penalty
        if resp_len < 15:
            score *= (resp_len / 15.0)
        
        # Clamp to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
        
    except Exception:
        return 0.0