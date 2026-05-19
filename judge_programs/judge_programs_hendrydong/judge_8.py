def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a semantic frame / topic modeling
    approach based on co-occurrence patterns, query decomposition into sub-questions,
    and response coverage analysis.
    
    This variant uses:
    - Query intent decomposition (question words, key phrases)
    - Semantic field expansion via co-occurrence windows
    - Response structural analysis (does it answer vs deflect)
    - Topic coherence via PMI-inspired co-occurrence scoring
    - Coverage of query's semantic "slots"
    """
    import re
    import math
    from collections import Counter, defaultdict
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        # --- Utility functions ---
        def clean_and_tokenize(text):
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
            tokens = text.split()
            return [t for t in tokens if len(t) > 1]
        
        def get_stopwords():
            return set(
                'the a an is are was were be been being have has had do does did will would '
                'shall should may might can could of in to for on with at by from as into '
                'through during before after above below between out off over under again '
                'further then once here there when where why how all both each few more most '
                'other some such no nor not only own same so than too very just don doesnt '
                'its it he she they them their his her this that these those am im ive youre '
                'weve theyre isnt arent wasnt werent wont dont didnt also about up but and or '
                'if because until while which what who whom whose my your our me us i you we '
                'really very much many get got like one two three even still already yet '
                'would could should might must need want think know see go come make take '
                'give find tell say use try keep let begin seem help show hear play run move '
                'live believe hold bring happen write provide sit stand lose pay meet include '
                'continue set learn change lead understand watch follow stop create speak read '
                'allow add spend grow open walk win offer remember love consider appear buy '
                'wait serve die send expect build stay fall cut reach kill remain suggest '
                'raise pass sell require report decide pull develop'.split()
            )
        
        stopwords = get_stopwords()
        
        def content_tokens(text):
            tokens = clean_and_tokenize(text)
            return [t for t in tokens if t not in stopwords and len(t) > 2]
        
        def get_ngrams(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # --- 1. Query Intent Decomposition ---
        # Extract what the query is really asking about
        query_tokens = clean_and_tokenize(query)
        query_content = content_tokens(query)
        resp_tokens = clean_and_tokenize(response)
        resp_content = content_tokens(response)
        
        if not query_content or not resp_content:
            return 1.0
        
        # Identify question type and key intent words
        question_patterns = {
            'how': 'process',
            'why': 'reason',
            'what': 'definition',
            'when': 'temporal',
            'where': 'location',
            'who': 'person',
            'which': 'selection',
            'is': 'confirmation',
            'are': 'confirmation',
            'can': 'ability',
            'does': 'confirmation',
            'do': 'confirmation',
            'should': 'advice',
            'would': 'hypothetical',
            'could': 'possibility',
            'imagine': 'creative',
            'create': 'creative',
            'generate': 'creative',
            'write': 'creative',
        }
        
        query_intent = 'general'
        for word, intent in question_patterns.items():
            if word in query_tokens[:10]:
                query_intent = intent
                break
        
        # --- 2. Extract Key Phrases from Query ---
        # Use positional weighting - words near question words and at start matter more
        query_word_weights = {}
        for i, token in enumerate(query_content):
            # Position weight: earlier words in query tend to be more important
            pos_weight = 1.0 / (1.0 + 0.05 * i)
            # Frequency weight in query
            freq_weight = query_content.count(token) 
            # Length weight: longer words tend to be more specific/important
            len_weight = min(len(token) / 5.0, 2.0)
            
            weight = pos_weight * (0.5 + 0.5 * len_weight) * (0.8 + 0.2 * freq_weight)
            if token in query_word_weights:
                query_word_weights[token] = max(query_word_weights[token], weight)
            else:
                query_word_weights[token] = weight
        
        # --- 3. Semantic Field Expansion via Co-occurrence Windows ---
        # Build co-occurrence graph within response to find topic clusters
        window_size = 5
        cooccurrence = defaultdict(lambda: defaultdict(int))
        for i, token in enumerate(resp_content):
            for j in range(max(0, i - window_size), min(len(resp_content), i + window_size + 1)):
                if i != j:
                    cooccurrence[token][resp_content[j]] += 1
        
        # Find which response words are semantically connected to query words
        query_content_set = set(query_content)
        resp_content_set = set(resp_content)
        
        # Direct matches
        direct_matches = query_content_set & resp_content_set
        
        # Words in response that co-occur frequently with direct matches (semantic neighbors)
        semantic_neighbors = set()
        for match_word in direct_matches:
            if match_word in cooccurrence:
                neighbors = sorted(cooccurrence[match_word].items(), key=lambda x: -x[1])
                for neighbor, count in neighbors[:5]:
                    if count >= 1 and neighbor not in stopwords:
                        semantic_neighbors.add(neighbor)
        
        # --- 4. Weighted Coverage Score ---
        total_weight = sum(query_word_weights.values())
        if total_weight == 0:
            return 1.0
        
        covered_weight = 0.0
        for word, weight in query_word_weights.items():
            if word in resp_content_set:
                covered_weight += weight
            else:
                # Check for partial/substring matches
                for rword in resp_content_set:
                    if len(word) > 3 and len(rword) > 3:
                        # Stem-like matching: shared prefix of length >= 4
                        shared_prefix_len = 0
                        for c1, c2 in zip(word, rword):
                            if c1 == c2:
                                shared_prefix_len += 1
                            else:
                                break
                        if shared_prefix_len >= min(4, min(len(word), len(rword)) - 1):
                            covered_weight += weight * 0.7
                            break
        
        coverage_score = covered_weight / total_weight
        
        # --- 5. Topic Coherence Score ---
        # Measure how coherent the response's topic is with the query's topic
        # Using PMI-inspired metric on bigram patterns
        query_bigrams = set(get_ngrams(query_content, 2))
        resp_bigrams = set(get_ngrams(resp_content, 2))
        
        bigram_overlap = len(query_bigrams & resp_bigrams)
        bigram_score = bigram_overlap / (1.0 + len(query_bigrams)) if query_bigrams else 0
        
        # Trigram overlap for even stronger topical alignment
        query_trigrams = set(get_ngrams(query_content, 3))
        resp_trigrams = set(get_ngrams(resp_content, 3))
        trigram_overlap = len(query_trigrams & resp_trigrams)
        trigram_score = trigram_overlap / (1.0 + len(query_trigrams)) if query_trigrams else 0
        
        # --- 6. Response Structural Quality ---
        # Does the response actually engage with the question or deflect?
        
        # Deflection indicators
        deflection_phrases = [
            'welcome to', 'please read', 'your post', 'this thread',
            'removed', 'moderator', 'rule', 'subreddit', 'flair',
            'i don\'t know', 'not sure', 'can\'t help',
            'while you wait', 'you might be interested in this'
        ]
        resp_lower = response.lower()
        deflection_count = sum(1 for phrase in deflection_phrases if phrase in resp_lower)
        deflection_penalty = min(deflection_count * 0.15, 0.5)
        
        # Engagement indicators - response uses first person, addresses the question
        engagement_phrases = [
            'because', 'therefore', 'essentially', 'specifically',
            'for example', 'in other words', 'the reason', 'this means',
            'you can', 'you should', 'i would', 'i think',
            'the answer', 'to answer', 'in short', 'basically'
        ]
        engagement_count = sum(1 for phrase in engagement_phrases if phrase in resp_lower)
        engagement_bonus = min(engagement_count * 0.05, 0.25)
        
        # --- 7. Response Depth / Substantiveness ---
        resp_word_count = len(resp_tokens)
        # Moderate length is good, very short may be insufficient
        if resp_word_count < 10:
            length_factor = 0.4
        elif resp_word_count < 25:
            length_factor = 0.6 + 0.016 * (resp_word_count - 10)
        elif resp_word_count < 80:
            length_factor = 0.85 + 0.003 * (resp_word_count - 25)
        else:
            length_factor = 1.0
        
        # --- 8. Specificity Score ---
        # Response should contain specific/concrete words relevant to the domain
        # Measure by looking at word rarity (longer, less common words = more specific)
        resp_specificity = 0
        for word in resp_content:
            if len(word) >= 6:
                resp_specificity += 1
            if len(word) >= 8:
                resp_specificity += 1
            if len(word) >= 10:
                resp_specificity += 1
        
        if len(resp_content) > 0:
            specificity_ratio = resp_specificity / len(resp_content)
        else:
            specificity_ratio = 0
        
        # Normalize specificity
        specificity_score = min(specificity_ratio / 1.5, 1.0)
        
        # --- 9. Query-Response Semantic Distance ---
        # Build term frequency vectors and compute a distance metric
        # that's different from cosine - use Jensen-Shannon-inspired divergence
        all_terms = list(query_content_set | resp_content_set)
        if not all_terms:
            return 1.0
        
        query_freq = Counter(query_content)
        resp_freq = Counter(resp_content)
        
        query_total = sum(query_freq.values())
        resp_total = sum(resp_freq.values())
        
        # Compute distributions
        kl_sum = 0.0
        shared_mass = 0.0
        for term in all_terms:
            p = (query_freq.get(term, 0) + 0.001) / (query_total + 0.001 * len(all_terms))
            q = (resp_freq.get(term, 0) + 0.001) / (resp_total + 0.001 * len(all_terms))
            m = 0.5 * (p + q)
            
            if p > 0 and m > 0:
                kl_sum += p * math.log(p / m)
            if q > 0 and m > 0:
                kl_sum += q * math.log(q / m)
            
            # Shared probability mass
            shared_mass += min(p, q)
        
        # JS divergence (0 = identical, ln(2) = maximally different)
        js_div = kl_sum / 2.0
        js_similarity = max(0, 1.0 - js_div / math.log(2))
        
        # --- 10. Semantic Slot Filling ---
        # For different query intents, check if response fills expected slots
        slot_bonus = 0.0
        
        if query_intent == 'process':
            # "How" questions: look for process/step words
            process_indicators = ['first', 'then', 'next', 'step', 'start', 'begin',
                                  'after', 'before', 'finally', 'by', 'through', 'using',
                                  'method', 'way', 'approach', 'technique', 'process']
            slot_bonus = min(sum(1 for w in process_indicators if w in resp_lower) * 0.04, 0.2)
        
        elif query_intent == 'reason':
            # "Why" questions: look for causal language
            causal_indicators = ['because', 'since', 'due', 'reason', 'cause', 'result',
                                 'therefore', 'consequently', 'thus', 'hence', 'led',
                                 'stems', 'originates', 'factor', 'explains']
            slot_bonus = min(sum(1 for w in causal_indicators if w in resp_lower) * 0.04, 0.2)
        
        elif query_intent == 'creative':
            # Creative tasks: look for narrative/creative elements
            creative_indicators = ['*', '"', 'said', 'replied', 'looked', 'felt',
                                   'walked', 'turned', 'voice', 'eyes', 'slowly',
                                   'suddenly', 'moment', 'figure', 'shadow']
            slot_bonus = min(sum(1 for w in creative_indicators if w in resp_lower) * 0.04, 0.25)
        
        elif query_intent == 'advice':
            advice_indicators = ['recommend', 'suggest', 'advise', 'best', 'better',
                                 'option', 'consider', 'worth', 'benefit', 'advantage',
                                 'experience', 'personally', 'career', 'helped']
            slot_bonus = min(sum(1 for w in advice_indicators if w in resp_lower) * 0.04, 0.2)
        
        elif query_intent in ('definition', 'confirmation'):
            definition_indicators = ['means', 'defined', 'refers', 'essentially',
                                     'basically', 'concept', 'idea', 'term',
                                     'yes', 'no', 'correct', 'actually', 'indeed']
            slot_bonus = min(sum(1 for w in definition_indicators if w in resp_lower) * 0.04, 0.2)
        
        # --- 11. Semantic Neighbor Bonus ---
        # Words in response that are topically related (co-occur with query words)
        neighbor_bonus = min(len(semantic_neighbors) * 0.02, 0.15)
        
        # --- 12. Final Composite Score ---
        # Weight the components
        raw_score = (
            coverage_score * 3.0 +          # How much of the query's content is addressed
            js_similarity * 2.0 +            # Distribution similarity
            bigram_score * 2.5 +             # Phrase-level overlap
            trigram_score * 3.0 +            # Strong topical alignment
            length_factor * 0.8 +            # Adequate length
            specificity_score * 0.7 +        # Domain-specific vocabulary
            engagement_bonus +               # Directly engages with question
            slot_bonus +                     # Fills expected answer slots
            neighbor_bonus -                 # Semantic neighborhood
            deflection_penalty               # Penalty for deflection
        )
        
        # Normalize to 0-10 range
        # Max theoretical: ~3 + 2 + 2.5 + 3 + 0.8 + 0.7 + 0.25 + 0.25 + 0.15 = ~12.65
        # Typical good: ~6-8, typical bad: ~2-4
        normalized = (raw_score / 10.0) * 10.0  # Keep in roughly 0-10
        
        # Apply sigmoid-like transformation for better discrimination
        # Center around 5, spread out differences
        centered = normalized - 4.5
        discriminated = 5.0 + 5.0 * (2.0 / (1.0 + math.exp(-0.6 * centered)) - 1.0)
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, discriminated))
        
        return round(final_score, 3)
    
    except Exception:
        return 2.0