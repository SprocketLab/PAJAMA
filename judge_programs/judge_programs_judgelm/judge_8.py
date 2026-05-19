def judging_function(query, response):
    """
    Evaluates response relevance using a dependency-chain approach:
    - Extracts key question components (interrogative type, subject entities, action verbs)
    - Builds a "query intent profile" and checks how well the response satisfies each component
    - Uses sentence-level coverage analysis rather than simple word overlap
    - Penalizes repetition, off-topic content, and incoherence
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
            return 0.5
        
        # --- Utility functions ---
        def tokenize(text):
            return re.findall(r'[a-z0-9]+', text.lower())
        
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
            'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'up', 'about', 'also', 'well', 'back', 'even', 'still', 'new',
            'want', 'know', 'make', 'like', 'get', 'go', 'see', 'come', 'take',
            'find', 'give', 'tell', 'say', 'think', 'use', 'work', 'call',
            'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean',
            'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play',
            'run', 'move', 'live', 'believe', 'bring', 'happen', 'write',
            'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
            'continue', 'set', 'learn', 'change', 'lead', 'understand',
            'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow',
            'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember',
            'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send',
            'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain',
            'am', 'much', 'many', 'any', 'don', 't', 's', 're', 've', 'll',
            'input', 'output', 'please', 'ok', 'yes', 'sure'
        }
        
        def content_words(tokens):
            return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 3]
        
        query_tokens = tokenize(query)
        resp_tokens = tokenize(response)
        
        query_content = content_words(query_tokens)
        resp_content = content_words(resp_tokens)
        
        if not query_content:
            query_content = query_tokens[:10]
        if not resp_tokens:
            return 0.5
        
        # === COMPONENT 1: Query Intent Decomposition ===
        # Extract key entities and concepts from query (capitalized words, nouns, named entities)
        query_lower = query.lower()
        
        # Find potential named entities (capitalized words in original query)
        named_entities = set()
        for word in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query):
            named_entities.add(word.lower())
        # Also catch all-caps
        for word in re.findall(r'\b[A-Z]{2,}\b', query):
            named_entities.add(word.lower())
        
        # Find numbers and specific values mentioned in query
        query_numbers = set(re.findall(r'\b\d+\b', query))
        
        # Identify query type
        query_type = 'statement'
        if re.search(r'\?', query):
            query_type = 'question'
        if re.search(r'^(create|write|make|generate|build|design|rewrite|identify|list|name|explain|describe|tell|find|show|give|provide|suggest|recommend|compare|translate|convert|summarize|classify|categorize|determine|calculate|compute|solve|evaluate|analyze|assess|review|check|verify|validate)\b', query_lower):
            query_type = 'imperative'
        if re.search(r'\b(rewrite|regenerate|rephrase|modify|change|update|edit|revise|shorten|expand|simplify)\b', query_lower):
            query_type = 'transformation'
        
        # === COMPONENT 2: Topic-Keyword Coverage (weighted) ===
        # Weight query content words by position and rarity
        query_content_counter = Counter(query_content)
        total_query_content = len(query_content) if query_content else 1
        
        # Give higher weight to words that appear less frequently (more specific)
        resp_content_set = set(resp_content)
        
        weighted_coverage = 0.0
        total_weight = 0.0
        for word, count in query_content_counter.items():
            # Rarer words in general get higher weight
            word_weight = 1.0
            if len(word) > 6:
                word_weight += 0.5  # longer words tend to be more specific
            if word in named_entities or any(word in ne for ne in named_entities):
                word_weight += 1.5  # named entities are critical
            if word in {'not', 'never', 'without', 'except', 'only'}:
                word_weight += 0.5  # negation/restriction words matter
            
            total_weight += word_weight
            if word in resp_content_set:
                weighted_coverage += word_weight
        
        if total_weight > 0:
            topic_coverage_score = weighted_coverage / total_weight
        else:
            topic_coverage_score = 0.3  # neutral if no content words
        
        # === COMPONENT 3: Sentence-level Relevance Distribution ===
        # Check if relevance is spread across the response or concentrated
        resp_sentences = get_sentences(response)
        query_content_set = set(query_content)
        
        sentence_relevance_scores = []
        for sent in resp_sentences:
            sent_tokens = content_words(tokenize(sent))
            if not sent_tokens:
                sentence_relevance_scores.append(0.0)
                continue
            overlap = len(set(sent_tokens) & query_content_set)
            # Also check for semantically related content (shared rare words)
            sent_score = overlap / max(len(query_content_set), 1)
            sentence_relevance_scores.append(min(sent_score, 1.0))
        
        if sentence_relevance_scores:
            # We want at least some sentences to be relevant
            max_sent_rel = max(sentence_relevance_scores)
            avg_sent_rel = sum(sentence_relevance_scores) / len(sentence_relevance_scores)
            
            # Fraction of sentences that have ANY relevance
            relevant_sent_fraction = sum(1 for s in sentence_relevance_scores if s > 0.05) / len(sentence_relevance_scores)
        else:
            max_sent_rel = 0.0
            avg_sent_rel = 0.0
            relevant_sent_fraction = 0.0
        
        # === COMPONENT 4: Response Coherence & Quality Signals ===
        
        # Repetition penalty: detect repeated phrases
        resp_bigrams = []
        for i in range(len(resp_tokens) - 1):
            resp_bigrams.append((resp_tokens[i], resp_tokens[i+1]))
        
        bigram_counter = Counter(resp_bigrams)
        if resp_bigrams:
            max_bigram_freq = max(bigram_counter.values())
            unique_bigram_ratio = len(bigram_counter) / len(resp_bigrams)
        else:
            max_bigram_freq = 1
            unique_bigram_ratio = 1.0
        
        repetition_penalty = 0.0
        if unique_bigram_ratio < 0.3:
            repetition_penalty = 0.4
        elif unique_bigram_ratio < 0.5:
            repetition_penalty = 0.2
        elif unique_bigram_ratio < 0.7:
            repetition_penalty = 0.1
        
        # Check for large repeated blocks (copy-paste style)
        if len(resp_sentences) > 2:
            sent_texts = [' '.join(tokenize(s)) for s in resp_sentences]
            sent_text_counter = Counter(sent_texts)
            max_sent_repeat = max(sent_text_counter.values())
            if max_sent_repeat > 2:
                repetition_penalty += 0.3
        
        # === COMPONENT 5: Response Length Appropriateness ===
        query_len = len(query_tokens)
        resp_len = len(resp_tokens)
        
        # Very short responses to complex queries are suspicious
        length_score = 1.0
        if resp_len < 3:
            length_score = 0.2
        elif resp_len < 5:
            length_score = 0.4
        elif resp_len < 10 and query_len > 15:
            length_score = 0.6
        elif resp_len > 200 and query_len < 20:
            # Very long response to short query - might be rambling
            length_score = 0.85
        
        # === COMPONENT 6: Off-topic / Tangent Detection ===
        # If response introduces many words completely unrelated to query
        resp_unique_content = set(resp_content)
        query_extended = set(query_content)
        
        if resp_unique_content:
            novel_words = resp_unique_content - query_extended
            novel_ratio = len(novel_words) / len(resp_unique_content)
        else:
            novel_ratio = 1.0
        
        # High novel ratio isn't always bad (good answers add info)
        # But combined with low topic coverage, it signals off-topic
        off_topic_signal = 0.0
        if topic_coverage_score < 0.2 and novel_ratio > 0.9:
            off_topic_signal = 0.3
        elif topic_coverage_score < 0.3 and novel_ratio > 0.85:
            off_topic_signal = 0.15
        
        # === COMPONENT 7: Format/Structure Quality ===
        # Check for HTML/code when not asked for
        has_code = bool(re.search(r'(def |import |class |<[a-z]+>|function\s*\(|var\s+\w)', response))
        query_asks_code = bool(re.search(r'\b(code|html|program|script|function|tag|css|javascript|python)\b', query_lower))
        
        format_penalty = 0.0
        if has_code and not query_asks_code:
            format_penalty = 0.2
        
        # Check for "Output:" prefix repetition (seen in examples)
        output_prefix_count = len(re.findall(r'\bOutput:', response))
        if query_type == 'transformation' and output_prefix_count > 0:
            format_penalty = 0.0  # This is fine for transformation tasks
        
        # === COMPONENT 8: Direct Address Detection ===
        # Does the response directly address the question's core?
        # Extract the core question words
        question_words = re.findall(r'\b(who|what|where|when|why|how|which|whose|whom)\b', query_lower)
        
        direct_address_bonus = 0.0
        if question_words:
            qw = question_words[0]
            resp_lower = response.lower()
            
            # Check if response pattern matches question type
            if qw == 'where' and re.search(r'\b(in|at|on|from|near|visit|located|place|location|site)\b', resp_lower):
                direct_address_bonus = 0.1
            elif qw == 'who' and re.search(r'\b[A-Z][a-z]+\b', response):
                direct_address_bonus = 0.1
            elif qw == 'when' and re.search(r'\b(\d{4}|\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)|century|year|decade|ago)\b', resp_lower):
                direct_address_bonus = 0.1
            elif qw == 'how' and re.search(r'\b(by|using|through|via|step|first|then|method|way|approach)\b', resp_lower):
                direct_address_bonus = 0.1
            elif qw in ('what', 'which'):
                # Just needs to provide some substantive content
                if len(resp_content) > 3:
                    direct_address_bonus = 0.05
            elif qw == 'why' and re.search(r'\b(because|since|due|reason|cause|result|therefore|so)\b', resp_lower):
                direct_address_bonus = 0.1
        
        # For imperative queries, check if the task was attempted
        if query_type == 'imperative':
            # The response should contain some transformation/creation
            if len(resp_content) > 3:
                direct_address_bonus += 0.1
        
        # === COMPONENT 9: Named Entity Recall ===
        entity_recall = 0.0
        if named_entities:
            resp_lower_text = response.lower()
            found = sum(1 for ne in named_entities if ne in resp_lower_text)
            entity_recall = found / len(named_entities)
        else:
            entity_recall = 0.5  # neutral
        
        # === COMPONENT 10: Semantic Density ===
        # Ratio of content words to total words in response
        if resp_tokens:
            content_density = len(resp_content) / len(resp_tokens)
        else:
            content_density = 0.0
        
        # Very low density means lots of filler
        density_factor = min(content_density / 0.4, 1.0)  # normalize around 0.4 being good
        
        # === FINAL SCORING ===
        # Weighted combination of all components
        score = (
            topic_coverage_score * 2.5 +      # 0-2.5: core keyword coverage
            relevant_sent_fraction * 1.0 +     # 0-1.0: sentence-level relevance spread
            max_sent_rel * 0.5 +               # 0-0.5: best sentence relevance
            avg_sent_rel * 0.5 +               # 0-0.5: average sentence relevance
            length_score * 1.5 +               # 0-1.5: appropriate length
            direct_address_bonus * 10.0 +      # 0-1.0 (scaled): directly addresses question
            entity_recall * 1.5 +              # 0-1.5: named entity recall
            density_factor * 0.5 +             # 0-0.5: content density
            (1.0 - repetition_penalty) * 0.5 - # 0-0.5: non-repetitive
            off_topic_signal * 5.0 -           # penalty for off-topic
            format_penalty * 3.0               # penalty for wrong format
        )
        
        # Normalize to 0-10 range
        # Max theoretical: 2.5 + 1.0 + 0.5 + 0.5 + 1.5 + 2.0 + 1.5 + 0.5 + 0.5 = 10.5
        # Min theoretical: 0 - 1.5 - 0.6 = -2.1
        score = max(0.0, min(10.0, score))
        
        # Apply a slight sigmoid-like transformation for better discrimination
        # Map [0,10] through a curve that spreads the middle
        normalized = score / 10.0
        # Slight power curve to spread scores
        adjusted = normalized ** 0.85
        final_score = adjusted * 10.0
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0