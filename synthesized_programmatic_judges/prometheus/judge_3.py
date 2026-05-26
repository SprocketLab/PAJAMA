def judging_function(query, response):
    """
    Evaluates response relevance using a query-response semantic alignment approach
    based on:
    1. N-gram coverage analysis (bigrams and trigrams from query found in response)
    2. Query intent keyword extraction and fulfillment scoring
    3. Discourse coherence markers and engagement patterns
    4. Semantic field overlap using co-occurrence word clusters
    5. Response directness (how quickly the response addresses query terms)
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
        
        # --- Preprocessing ---
        def tokenize(text):
            """Lowercase tokenization, keeping only alphanumeric."""
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def get_ngrams(tokens, n):
            """Generate n-grams from token list."""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Common stop words to filter for content analysis
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'that',
            'which', 'who', 'whom', 'this', 'these', 'those', 'am', 'it', 'its',
            'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'my', 'me', 'i', 'what', 'about', 'up', 'down',
            'also', 'any', 'much', 'many', 'well', 'get', 'got', 'like', 'make',
            'made', 'even', 'still', 'way', 'take', 'come', 'go', 'know', 'say',
            'said', 'one', 'two', 'first', 'new', 'now', 'look', 'think', 'see',
            'time', 'thing', 'things', 'people', 'person', 'really', 'want',
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = [t for t in query_tokens if t not in stop_words and len(t) > 2]
        response_content = [t for t in response_tokens if t not in stop_words and len(t) > 2]
        
        if not query_content or not response_content:
            return 2.0
        
        # --- Feature 1: Weighted Query Term Coverage ---
        # Weight query terms by their specificity (inverse frequency approximation via length)
        query_term_counts = Counter(query_content)
        response_content_set = set(response_content)
        response_content_counter = Counter(response_content)
        
        # Calculate term importance: longer and rarer words in query are more important
        total_query_terms = len(query_term_counts)
        covered_weight = 0.0
        total_weight = 0.0
        
        for term, count in query_term_counts.items():
            # Weight by length (longer words are more specific) and frequency in query
            weight = (1 + math.log(1 + len(term) - 2)) * math.sqrt(count)
            total_weight += weight
            if term in response_content_set:
                # Bonus for multiple mentions in response (shows emphasis)
                mention_factor = min(1.0 + 0.2 * math.log(1 + response_content_counter[term]), 1.5)
                covered_weight += weight * mention_factor
        
        coverage_score = covered_weight / max(total_weight, 1e-9)
        coverage_score = min(coverage_score, 1.0)
        
        # --- Feature 2: Bigram and Trigram Overlap ---
        # Shared n-grams indicate topical alignment beyond single words
        query_bigrams = set(get_ngrams(query_tokens, 2))
        response_bigrams = set(get_ngrams(response_tokens, 2))
        query_trigrams = set(get_ngrams(query_tokens, 3))
        response_trigrams = set(get_ngrams(response_tokens, 3))
        
        bigram_overlap = len(query_bigrams & response_bigrams) / max(len(query_bigrams), 1)
        trigram_overlap = len(query_trigrams & response_trigrams) / max(len(query_trigrams), 1)
        
        ngram_score = 0.6 * min(bigram_overlap * 5, 1.0) + 0.4 * min(trigram_overlap * 8, 1.0)
        
        # --- Feature 3: Query Intent Fulfillment ---
        # Detect query type and check if response matches expected patterns
        query_lower = query.lower()
        response_lower = response.lower()
        
        intent_score = 0.5  # baseline
        
        # Check for question patterns and matching response patterns
        is_how_to = bool(re.search(r'\bhow\s+(to|would|can|could|should|do)\b', query_lower))
        is_explain = bool(re.search(r'\b(explain|describe|what\s+is|understand|concept)\b', query_lower))
        is_seeking_help = bool(re.search(r'\b(help|assist|advice|comfort|support|seeking|need)\b', query_lower))
        is_emotional = bool(re.search(r'\b(feeling|feel|emotion|stress|sad|frustrat|heartbr|loneli|despair|exhaust)\b', query_lower))
        is_manage = bool(re.search(r'\b(manage|handle|cope|deal\s+with|address)\b', query_lower))
        
        # Check response alignment with intent
        if is_how_to:
            # Response should contain instructional language
            has_steps = bool(re.search(r'\b(first|step|start|begin|then|next|after|finally)\b', response_lower))
            has_instructions = bool(re.search(r'\b(you\s+(can|should|could|need|might|will)|try|consider|make sure)\b', response_lower))
            if has_steps or has_instructions:
                intent_score += 0.3
        
        if is_explain:
            # Response should contain explanatory language
            has_explanation = bool(re.search(r'\b(means|refers|is\s+a|defined|essentially|basically|imagine|think\s+of)\b', response_lower))
            has_examples = bool(re.search(r'\b(for\s+example|for\s+instance|such\s+as|like\s+a|imagine|consider)\b', response_lower))
            if has_explanation:
                intent_score += 0.2
            if has_examples:
                intent_score += 0.15
        
        if is_seeking_help or is_emotional:
            # Response should show empathy and engagement
            has_empathy = bool(re.search(r'\b(understand|sorry|hear|feel|completely|absolutely|natural|okay|valid)\b', response_lower))
            has_acknowledgment = bool(re.search(r'\b(i\s+can\s+see|i\s+hear|it\'s\s+(okay|understandable|natural|completely)|that\'s\s+(tough|hard|difficult))\b', response_lower))
            has_advice = bool(re.search(r'\b(try|consider|remember|don\'t\s+forget|keep|take|break|reach\s+out)\b', response_lower))
            if has_empathy:
                intent_score += 0.2
            if has_acknowledgment:
                intent_score += 0.15
            if has_advice:
                intent_score += 0.1
        
        if is_manage:
            has_strategies = bool(re.search(r'\b(strategy|approach|method|technique|way|tip|suggest|recommend)\b', response_lower))
            if has_strategies:
                intent_score += 0.2
        
        intent_score = min(intent_score, 1.0)
        
        # --- Feature 4: Response Directness / Early Relevance ---
        # Check how quickly the response addresses query content words
        # Responses that address the topic early are usually more relevant
        early_window = response_tokens[:min(30, len(response_tokens))]
        early_content = set(t for t in early_window if t not in stop_words and len(t) > 2)
        query_content_set = set(query_content)
        
        early_overlap = len(early_content & query_content_set) / max(len(query_content_set), 1)
        directness_score = min(early_overlap * 3, 1.0)
        
        # --- Feature 5: Semantic Field Clustering ---
        # Build semantic clusters: group words that tend to co-occur in the same domain
        # Use character n-gram overlap as a proxy for semantic similarity
        def char_ngrams(word, n=3):
            """Get character n-grams of a word."""
            padded = f"_{word}_"
            return set(padded[i:i+n] for i in range(len(padded) - n + 1))
        
        # For each query content word, find response words with high char-ngram overlap
        # This captures morphological variants and related terms
        semantic_matches = 0
        for q_word in query_content_set:
            q_chars = char_ngrams(q_word)
            if not q_chars:
                continue
            best_sim = 0
            for r_word in set(response_content):
                if r_word == q_word:
                    best_sim = 1.0
                    break
                r_chars = char_ngrams(r_word)
                if not r_chars:
                    continue
                sim = len(q_chars & r_chars) / max(len(q_chars | r_chars), 1)
                if sim > best_sim:
                    best_sim = sim
            if best_sim > 0.35:  # threshold for "related"
                semantic_matches += best_sim
        
        semantic_score = semantic_matches / max(len(query_content_set), 1)
        semantic_score = min(semantic_score, 1.0)
        
        # --- Feature 6: Response Quality Indicators ---
        # Check for signs of a well-structured, substantive response
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Adequate length (not too short, not rambling)
        response_len = len(response_tokens)
        length_score = 0.0
        if response_len < 15:
            length_score = 0.2
        elif response_len < 30:
            length_score = 0.5
        elif response_len < 200:
            length_score = 0.8 + 0.2 * min((response_len - 30) / 100, 1.0)
        else:
            length_score = 0.9
        
        # Check for dismissive or unhelpful patterns
        dismissive_patterns = [
            r'\bjust\s+(do|get|try|buy|move)\b',
            r'\bget\s+over\s+it\b',
            r'\bstop\s+(being|feeling|worrying)\b',
            r'\bit\'s\s+not\s+a\s+big\s+deal\b',
            r'\byou\s+should\s+be\s+able\b',
            r'\bmaybe\s+you\'re\s+just\b',
            r'\bmight\s+not\b.*\bable\b',
            r'\bprobably\s+won\'t\b',
        ]
        
        dismissive_count = sum(1 for p in dismissive_patterns if re.search(p, response_lower))
        dismissive_penalty = min(dismissive_count * 0.1, 0.3)
        
        # Check for hedging/uncertainty that undermines helpfulness
        negative_capability = 0
        neg_patterns = [r'\bcannot\b', r'\bcan\'t\b', r'\bnot\s+able\b', r'\bwon\'t\b', r'\bmay\s+not\b']
        for p in neg_patterns:
            if re.search(p, response_lower):
                negative_capability += 1
        negative_penalty = min(negative_capability * 0.05, 0.15)
        
        # --- Feature 7: Topic-Specific Vocabulary Presence ---
        # Extract likely topic from query and check response has domain vocabulary
        # Use word frequency distribution comparison
        query_word_freq = Counter(query_content)
        response_word_freq = Counter(response_content)
        
        # Get the most distinctive query words (highest frequency, longest)
        key_terms = sorted(query_word_freq.keys(), 
                          key=lambda w: query_word_freq[w] * len(w), reverse=True)[:10]
        
        key_term_presence = 0
        for term in key_terms:
            if term in response_word_freq:
                key_term_presence += 1
            else:
                # Check for partial matches (stems)
                stem = term[:max(4, len(term) - 2)]
                for r_word in response_word_freq:
                    if r_word.startswith(stem) or stem in r_word:
                        key_term_presence += 0.6
                        break
        
        key_term_score = key_term_presence / max(len(key_terms), 1)
        key_term_score = min(key_term_score, 1.0)
        
        # --- Feature 8: Pronoun and Reference Alignment ---
        # If query uses "I", "my", response should use "you", "your" (and vice versa)
        query_first_person = len(re.findall(r'\b(i|my|me|i\'m|i\'ve)\b', query_lower))
        response_second_person = len(re.findall(r'\b(you|your|you\'re|you\'ve)\b', response_lower))
        
        pronoun_alignment = 0.5  # neutral
        if query_first_person > 0:
            if response_second_person > 0:
                pronoun_alignment = 0.8
            else:
                pronoun_alignment = 0.3
        
        # --- Combine all features with weights ---
        # Weights tuned to emphasize relevance dimensions
        final_score = (
            0.22 * coverage_score +
            0.12 * ngram_score +
            0.18 * intent_score +
            0.12 * directness_score +
            0.10 * semantic_score +
            0.06 * length_score +
            0.12 * key_term_score +
            0.08 * pronoun_alignment
        )
        
        # Apply penalties
        final_score = final_score - dismissive_penalty - negative_penalty
        final_score = max(final_score, 0.0)
        
        # Scale to 1-5 range
        scaled = 1.0 + 4.0 * final_score
        
        # Apply slight non-linearity to spread out scores
        scaled = 1.0 + 4.0 * (((scaled - 1.0) / 4.0) ** 0.85)
        
        return round(min(max(scaled, 1.0), 5.0), 2)
    
    except Exception:
        return 2.5