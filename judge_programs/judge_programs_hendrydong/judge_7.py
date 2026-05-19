def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a semantic field coverage approach.
    
    Algorithm: Query Decomposition + Semantic Field Coverage + Discourse Quality
    
    This variant uses a fundamentally different approach:
    1. Extracts "semantic fields" from the query by identifying key noun phrases and verb phrases
    2. Builds a query intent model (question type detection + topic extraction)
    3. Measures how many distinct semantic fields from the query are addressed in the response
    4. Evaluates discourse coherence via sentence-level topic threading
    5. Penalizes generic/boilerplate responses using an entropy-based informativeness measure
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
        if len(query) < 3:
            return 5.0
        
        # --- Helper functions ---
        
        def tokenize(text):
            """Lowercase tokenization, keeping alphanumeric and hyphens."""
            return re.findall(r"[a-z0-9](?:[a-z0-9'-]*[a-z0-9])?", text.lower())
        
        def get_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 3]
        
        STOP_WORDS = set(
            "i me my myself we our ours ourselves you your yours yourself yourselves "
            "he him his himself she her hers herself it its itself they them their theirs "
            "themselves what which who whom this that these those am is are was were be "
            "been being have has had having do does did doing a an the and but if or "
            "because as until while of at by for with about against between through during "
            "before after above below to from up down in out on off over under again further "
            "then once here there when where why how all both each few more most other some "
            "such no nor not only own same so than too very s t can will just don should now "
            "d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn "
            "needn shan shouldn wasn weren won wouldn would could might shall may also even "
            "still already yet much many get got like really quite well thing things way "
            "something anything everything nothing going know think want need make made "
            "one two said say go come take see look use used using however therefore thus "
            "please thank thanks sure yes no okay ok hi hello hey dear sir madam "
            "able let put try new old first last long great little just right back good".split()
        )
        
        def content_words(tokens):
            return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        
        def extract_bigrams(tokens):
            """Extract content word bigrams as pseudo-phrases."""
            ct = content_words(tokens)
            bigrams = set()
            for i in range(len(ct) - 1):
                bigrams.add((ct[i], ct[i+1]))
            return bigrams
        
        def extract_trigrams(tokens):
            ct = content_words(tokens)
            trigrams = set()
            for i in range(len(ct) - 2):
                trigrams.add((ct[i], ct[i+1], ct[i+2]))
            return trigrams
        
        # --- 1. Query Intent Analysis ---
        
        query_lower = query.lower()
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        if not query_content:
            return 5.0
        
        # Detect question type
        question_types = {
            'how': ['how', 'method', 'process', 'steps', 'way'],
            'why': ['why', 'reason', 'cause', 'because', 'explain'],
            'what': ['what', 'define', 'definition', 'meaning'],
            'when': ['when', 'time', 'date', 'year', 'period'],
            'where': ['where', 'location', 'place'],
            'who': ['who', 'person', 'people'],
            'which': ['which', 'choose', 'select', 'compare'],
            'opinion': ['think', 'opinion', 'feel', 'believe', 'argument'],
            'experience': ['experience', 'story', 'personal', 'your'],
            'action': ['imagine', 'create', 'write', 'generate', 'build', 'make'],
        }
        
        detected_intents = set()
        for intent, keywords in question_types.items():
            for kw in keywords:
                if kw in query_lower:
                    detected_intents.add(intent)
                    break
        
        # --- 2. Semantic Field Extraction ---
        # Extract "semantic fields" = clusters of related content words from query
        
        # Weight query words by position (earlier = more important for topic)
        query_word_weights = {}
        for i, w in enumerate(query_content):
            # Words appearing earlier get slightly higher weight
            position_weight = 1.0 + 0.5 * (1.0 - i / max(len(query_content), 1))
            if w not in query_word_weights:
                query_word_weights[w] = position_weight
            else:
                query_word_weights[w] += position_weight * 0.5  # repeated = important
        
        # Boost words that appear in the first sentence of query (likely the core question)
        query_sents = get_sentences(query)
        if query_sents:
            first_sent_words = set(content_words(tokenize(query_sents[0])))
            for w in first_sent_words:
                if w in query_word_weights:
                    query_word_weights[w] *= 1.5
        
        # Identify rare/specific words (longer words tend to be more specific)
        for w in query_word_weights:
            if len(w) >= 7:
                query_word_weights[w] *= 1.3
            if len(w) >= 10:
                query_word_weights[w] *= 1.2
        
        # --- 3. Semantic Field Coverage Score ---
        
        response_word_set = set(response_content)
        response_token_set = set(response_tokens)
        
        # For each query content word, check if it or a related form appears in response
        def word_match_score(query_word, resp_set):
            """Check for exact match, prefix match, or substring match."""
            if query_word in resp_set:
                return 1.0
            # Prefix/stem matching (crude but effective)
            stem = query_word[:max(4, len(query_word) - 2)]
            for rw in resp_set:
                if rw.startswith(stem) and len(rw) >= len(stem):
                    return 0.8
                if stem in rw and len(stem) >= 4:
                    return 0.6
            return 0.0
        
        total_weight = sum(query_word_weights.values())
        if total_weight == 0:
            return 5.0
        
        covered_weight = 0.0
        for w, weight in query_word_weights.items():
            match = word_match_score(w, response_word_set)
            covered_weight += weight * match
        
        field_coverage = covered_weight / total_weight  # 0 to 1
        
        # --- 4. Phrase-level Coverage (bigrams from query found in response) ---
        
        query_bigrams = extract_bigrams(query_tokens)
        response_bigrams = extract_bigrams(response_tokens)
        
        if query_bigrams:
            # Check both exact bigram match and reversed/partial
            bigram_matches = 0
            for qb in query_bigrams:
                if qb in response_bigrams:
                    bigram_matches += 1
                else:
                    # Check if both words appear close together in response
                    w1, w2 = qb
                    if w1 in response_word_set and w2 in response_word_set:
                        # Check proximity
                        positions_w1 = [i for i, t in enumerate(response_content) if t == w1]
                        positions_w2 = [i for i, t in enumerate(response_content) if t == w2]
                        if positions_w1 and positions_w2:
                            min_dist = min(abs(p1 - p2) for p1 in positions_w1 for p2 in positions_w2)
                            if min_dist <= 5:
                                bigram_matches += 0.5
            phrase_coverage = bigram_matches / len(query_bigrams)
        else:
            phrase_coverage = field_coverage  # fallback
        
        # --- 5. Response Informativeness (entropy-based) ---
        
        if response_content:
            word_counts = Counter(response_content)
            total = len(response_content)
            entropy = 0.0
            for count in word_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalize entropy (max entropy for N unique words)
            max_entropy = math.log2(max(len(word_counts), 1)) if len(word_counts) > 1 else 1.0
            normalized_entropy = entropy / max(max_entropy, 0.001)
        else:
            normalized_entropy = 0.0
        
        # --- 6. Boilerplate / Generic Response Detection ---
        
        boilerplate_phrases = [
            "welcome to", "please read our rules", "your comments will be removed",
            "can you please describe", "i can help you", "here to assist",
            "i'm an ai", "as an ai language model", "i cannot", "i don't have access",
            "do not fear", "is there something i can help",
        ]
        
        response_lower = response.lower()
        boilerplate_penalty = 0.0
        for bp in boilerplate_phrases:
            if bp in response_lower:
                boilerplate_penalty += 0.15
        boilerplate_penalty = min(boilerplate_penalty, 0.5)
        
        # --- 7. Discourse Threading Score ---
        # Check if multiple sentences in the response each connect to query topics
        
        resp_sents = get_sentences(response)
        if len(resp_sents) > 1:
            sents_on_topic = 0
            for sent in resp_sents:
                sent_content = set(content_words(tokenize(sent)))
                # A sentence is "on topic" if it shares content words with query
                overlap = sent_content & set(query_content)
                if overlap:
                    sents_on_topic += 1
                else:
                    # Check stem-level overlap
                    query_stems = {w[:max(4, len(w)-2)] for w in query_content}
                    sent_stems = {w[:max(4, len(w)-2)] for w in sent_content}
                    if query_stems & sent_stems:
                        sents_on_topic += 0.5
            
            threading_score = sents_on_topic / len(resp_sents)
        else:
            threading_score = field_coverage  # single sentence, use field coverage
        
        # --- 8. Response Substantiveness ---
        
        # Reward responses that have enough substance
        response_length = len(response_content)
        if response_length < 5:
            length_factor = 0.4
        elif response_length < 15:
            length_factor = 0.6 + 0.4 * ((response_length - 5) / 10)
        elif response_length < 50:
            length_factor = 1.0
        elif response_length < 150:
            length_factor = 1.0
        else:
            # Very long responses: slight diminishing returns
            length_factor = 0.95
        
        # --- 9. Direct Address Detection ---
        # Check if response directly engages with the specific subject matter
        
        # Extract proper nouns / capitalized terms from query (likely key entities)
        query_entities = set(re.findall(r'\b[A-Z][a-z]{2,}\b', query))
        # Also check for quoted/special terms
        query_special = set(re.findall(r'`([^`]+)`', query))
        
        entity_coverage = 0.0
        if query_entities:
            found = sum(1 for e in query_entities if e.lower() in response_token_set)
            entity_coverage = found / len(query_entities)
        
        special_coverage = 0.0
        if query_special:
            found = sum(1 for s in query_special if s.lower() in response_lower)
            special_coverage = found / len(query_special)
        
        # --- 10. Intent Fulfillment ---
        
        intent_score = 0.5  # baseline
        
        if 'experience' in detected_intents:
            # Personal experience questions: look for first-person narrative
            personal_markers = ['i ', 'my ', 'me ', "i've", "i'm", 'got me', 'helped me']
            personal_count = sum(1 for m in personal_markers if m in response_lower)
            intent_score = min(1.0, 0.3 + personal_count * 0.15)
        
        if 'action' in detected_intents:
            # Creative/action queries: look for engagement with the scenario
            action_markers = ['*', ':', '"', 'stepping', 'feeling', 'walking']
            action_count = sum(1 for m in action_markers if m in response_lower)
            intent_score = min(1.0, 0.3 + action_count * 0.15)
        
        if 'opinion' in detected_intents or 'why' in detected_intents:
            # Argumentative queries: look for reasoning markers
            reasoning_markers = ['because', 'therefore', 'since', 'thus', 'argument',
                                'reason', 'evidence', 'suggests', 'implies', 'consider',
                                'example', 'for instance', 'perspective']
            reasoning_count = sum(1 for m in reasoning_markers if m in response_lower)
            intent_score = min(1.0, 0.3 + reasoning_count * 0.1)
        
        # --- 11. Combine all signals ---
        
        # Weighted combination
        score = (
            field_coverage * 3.0 +          # Core semantic coverage (0-3)
            phrase_coverage * 1.5 +          # Phrase-level precision (0-1.5)
            threading_score * 1.5 +          # Discourse coherence (0-1.5)
            normalized_entropy * 0.8 +       # Informativeness (0-0.8)
            entity_coverage * 1.0 +          # Entity coverage (0-1)
            special_coverage * 0.5 +         # Special term coverage (0-0.5)
            intent_score * 1.0 +             # Intent fulfillment (0-1)
            length_factor * 0.7              # Substantiveness (0-0.7)
        )
        
        # Apply boilerplate penalty
        score *= (1.0 - boilerplate_penalty)
        
        # Scale to 0-10 range
        max_possible = 3.0 + 1.5 + 1.5 + 0.8 + 1.0 + 0.5 + 1.0 + 0.7  # = 10.0
        score = (score / max_possible) * 10.0
        
        # Clamp
        score = max(0.0, min(10.0, score))
        
        return round(score, 4)
    
    except Exception:
        return 3.0