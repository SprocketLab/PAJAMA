def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a combination of:
    - Query keyword extraction and coverage analysis
    - Sentence-level relevance scoring (what fraction of response sentences are on-topic)
    - Query intent detection and response alignment
    - Information density and directness measurement
    - Penalization for hedging/deflection patterns
    
    This variant focuses on sentence-level relevance decomposition and intent alignment,
    which is substantially different from word overlap, cosine similarity, n-gram, 
    bullet/header detection, or paragraph analysis approaches.
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
        
        # --- Utility functions ---
        
        STOP_WORDS = {
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
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'up', 'about', 'against',
            'am', 'an', 'any', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn',
            'hasn', 'haven', 'isn', 'let', 'mustn', 'shan', 'shouldn', 'wasn',
            'weren', 'won', 'wouldn', 'also', 'like', 'get', 'got', 'much',
            'many', 'well', 'back', 'even', 'still', 'way', 'take', 'make',
            'come', 'go', 'know', 'say', 'see', 'look', 'think', 'thing',
            'really', 'quite', 'down', 'something', 'anything', 'everything',
        }
        
        def tokenize(text):
            """Tokenize text into lowercase words."""
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def get_content_words(text):
            """Get content words (non-stopwords) from text."""
            tokens = tokenize(text)
            return [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
        
        def split_sentences(text):
            """Split text into sentences."""
            # Handle markdown/formatting
            clean = re.sub(r'#{1,6}\s*', '', text)
            clean = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', clean)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            # Split on sentence boundaries
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\d+[\.\)])|(?<=\n)\s*(?=[-•*])', clean)
            # Also split on newlines that seem like list items
            result = []
            for s in sents:
                sub = re.split(r'\n+', s)
                result.extend(sub)
            return [s.strip() for s in result if len(s.strip()) > 3]
        
        def get_query_intent_words(q):
            """Extract the most important words from the query that signal intent."""
            tokens = tokenize(q)
            content = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
            # Weight words by position (earlier words in query often more important)
            weighted = {}
            for i, w in enumerate(content):
                pos_weight = 1.0 + 0.5 * (1.0 - i / max(len(content), 1))
                # Longer words tend to be more specific/important
                len_weight = min(len(w) / 4.0, 1.5)
                weighted[w] = max(weighted.get(w, 0), pos_weight * len_weight)
            return weighted
        
        # --- Feature 1: Sentence-level relevance decomposition ---
        
        query_content = get_content_words(query)
        query_content_set = set(query_content)
        query_intent = get_query_intent_words(query)
        
        response_sentences = split_sentences(response)
        if not response_sentences:
            return 1.0
        
        # For each sentence, compute how relevant it is to the query
        sentence_relevance_scores = []
        for sent in response_sentences:
            sent_words = set(get_content_words(sent))
            if not sent_words:
                sentence_relevance_scores.append(0.0)
                continue
            
            # Direct word overlap with query content
            overlap = sent_words & query_content_set
            overlap_ratio = len(overlap) / max(len(query_content_set), 1)
            
            # Weighted overlap using intent weights
            weighted_overlap = sum(query_intent.get(w, 0) for w in overlap)
            max_weighted = sum(query_intent.values()) if query_intent else 1
            weighted_ratio = weighted_overlap / max(max_weighted, 1)
            
            # Check for semantic proximity: words that share stems/prefixes with query words
            stem_matches = 0
            for sw in sent_words:
                for qw in query_content_set:
                    # Check if words share a common prefix of length >= 4
                    min_len = min(len(sw), len(qw))
                    if min_len >= 4:
                        prefix_len = 0
                        for c1, c2 in zip(sw, qw):
                            if c1 == c2:
                                prefix_len += 1
                            else:
                                break
                        if prefix_len >= 4 and prefix_len >= min_len * 0.6:
                            stem_matches += 1
                            break
            stem_ratio = stem_matches / max(len(sent_words), 1)
            
            rel_score = 0.4 * overlap_ratio + 0.4 * weighted_ratio + 0.2 * stem_ratio
            sentence_relevance_scores.append(min(rel_score, 1.0))
        
        # Compute aggregate sentence relevance metrics
        if sentence_relevance_scores:
            avg_sent_relevance = sum(sentence_relevance_scores) / len(sentence_relevance_scores)
            # Fraction of sentences that are at least somewhat relevant
            relevant_sent_fraction = sum(1 for s in sentence_relevance_scores if s > 0.05) / len(sentence_relevance_scores)
            # First sentence relevance (important for directness)
            first_sent_relevance = sentence_relevance_scores[0]
            # Max sentence relevance
            max_sent_relevance = max(sentence_relevance_scores)
        else:
            avg_sent_relevance = 0.0
            relevant_sent_fraction = 0.0
            first_sent_relevance = 0.0
            max_sent_relevance = 0.0
        
        # --- Feature 2: Query coverage analysis ---
        # How many of the query's key concepts are addressed in the response?
        
        response_content = set(get_content_words(response))
        
        # Check coverage of query intent words
        covered_intent = 0
        total_intent_weight = 0
        for word, weight in query_intent.items():
            total_intent_weight += weight
            if word in response_content:
                covered_intent += weight
            else:
                # Check stem match
                for rw in response_content:
                    min_len = min(len(word), len(rw))
                    if min_len >= 4:
                        prefix_len = 0
                        for c1, c2 in zip(word, rw):
                            if c1 == c2:
                                prefix_len += 1
                            else:
                                break
                        if prefix_len >= 4 and prefix_len >= min_len * 0.6:
                            covered_intent += weight * 0.7
                            break
        
        query_coverage = covered_intent / max(total_intent_weight, 1)
        
        # --- Feature 3: Directness / how quickly the response addresses the query ---
        
        response_tokens = tokenize(response)
        response_content_list = get_content_words(response)
        
        # Find position of first query keyword in response
        first_keyword_pos = len(response_tokens)  # default to end
        for i, token in enumerate(response_tokens):
            if token in query_content_set:
                first_keyword_pos = i
                break
        
        # Normalize: earlier is better
        directness_score = max(0, 1.0 - first_keyword_pos / max(len(response_tokens), 1))
        
        # Also check if the first sentence directly engages with the query
        # (e.g., restates it or answers it)
        first_sent = response_sentences[0] if response_sentences else ""
        first_sent_content = set(get_content_words(first_sent))
        first_sent_overlap = len(first_sent_content & query_content_set) / max(len(query_content_set), 1)
        
        engagement_score = 0.5 * directness_score + 0.5 * first_sent_overlap
        
        # --- Feature 4: Detect hedging / deflection ---
        
        hedging_patterns = [
            r"i'?m not (?:sure|certain|aware)",
            r"i don'?t (?:know|have|think i can)",
            r"i cannot (?:help|assist|provide)",
            r"beyond (?:my|the) (?:scope|knowledge|ability)",
            r"i'?m (?:just )?an? (?:ai|language model|chatbot)",
            r"as an ai",
            r"i don'?t have (?:access|information)",
            r"(?:however|but),?\s*(?:i (?:can|could|would) (?:suggest|recommend|provide))",
        ]
        
        response_lower = response.lower()
        hedging_count = 0
        for pattern in hedging_patterns:
            if re.search(pattern, response_lower):
                hedging_count += 1
        
        hedging_penalty = min(hedging_count * 0.08, 0.3)
        
        # --- Feature 5: Topic vocabulary density ---
        # Measure how much of the response uses vocabulary related to the query topic
        
        # Build an expanded topic vocabulary from query using co-occurrence hints
        # For this, we look at words that appear near query words in the response
        query_word_positions = {}
        for i, token in enumerate(response_tokens):
            if token in query_content_set:
                query_word_positions.setdefault(token, []).append(i)
        
        # Words within a window of query words are likely topically relevant
        topic_adjacent_words = set()
        window = 5
        for positions in query_word_positions.values():
            for pos in positions:
                for j in range(max(0, pos - window), min(len(response_tokens), pos + window + 1)):
                    w = response_tokens[j]
                    if w not in STOP_WORDS and len(w) > 2:
                        topic_adjacent_words.add(w)
        
        # Topic density: fraction of content words that are either query words or adjacent to them
        if response_content_list:
            topic_words_in_response = sum(1 for w in response_content_list 
                                          if w in query_content_set or w in topic_adjacent_words)
            topic_density = topic_words_in_response / len(response_content_list)
        else:
            topic_density = 0.0
        
        # --- Feature 6: Response structure quality (brief, focused assessment) ---
        
        # Penalize very short responses that might not fully address the query
        length_tokens = len(response_tokens)
        if length_tokens < 10:
            length_factor = 0.3
        elif length_tokens < 30:
            length_factor = 0.6
        elif length_tokens < 50:
            length_factor = 0.8
        else:
            length_factor = 1.0
        
        # Slight bonus for responses that seem to provide structured/organized answers
        has_enumeration = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]', response))
        has_steps = bool(re.search(r'(?:step|first|second|third|finally|next|then)', response_lower))
        
        # Check if query is asking for a process/list/explanation
        query_lower = query.lower()
        asks_how = bool(re.search(r'\b(?:how|steps|process|way|method|procedure)\b', query_lower))
        asks_what = bool(re.search(r'\b(?:what|explain|describe|tell|define)\b', query_lower))
        asks_list = bool(re.search(r'\b(?:ideas|suggestions|tips|ways|reasons|examples|list)\b', query_lower))
        asks_opinion = bool(re.search(r'\b(?:think|believe|opinion|should|feel)\b', query_lower))
        
        structure_bonus = 0.0
        if (asks_how or asks_list) and (has_enumeration or has_steps):
            structure_bonus = 0.05
        
        # --- Feature 7: Question-type alignment ---
        # Check if response type matches query type
        
        is_yes_no_query = bool(re.search(r'^(?:do|does|did|is|are|was|were|can|could|should|would|will|has|have)\b', query_lower))
        starts_with_answer = bool(re.search(r'^(?:yes|no|absolutely|certainly|definitely|indeed|of course|i (?:do|don\'t|think|believe))', response_lower))
        
        alignment_bonus = 0.0
        if is_yes_no_query and starts_with_answer:
            alignment_bonus = 0.05
        
        # --- Feature 8: Unique query bigram coverage ---
        # Check if important bigrams from the query appear in the response
        
        query_tokens_clean = [w for w in tokenize(query) if w not in STOP_WORDS and len(w) > 1]
        query_bigrams = set()
        for i in range(len(query_tokens_clean) - 1):
            query_bigrams.add((query_tokens_clean[i], query_tokens_clean[i + 1]))
        
        response_tokens_clean = [w for w in tokenize(response) if w not in STOP_WORDS and len(w) > 1]
        response_bigrams = set()
        for i in range(len(response_tokens_clean) - 1):
            response_bigrams.add((response_tokens_clean[i], response_tokens_clean[i + 1]))
        
        if query_bigrams:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.0
        
        # --- Combine all features into final score ---
        
        # Weights for each component
        score = (
            0.20 * avg_sent_relevance * 10 +          # 0-2.0: average sentence relevance
            0.10 * relevant_sent_fraction * 10 +       # 0-1.0: fraction of relevant sentences
            0.10 * first_sent_relevance * 10 +         # 0-1.0: first sentence relevance
            0.15 * query_coverage * 10 +               # 0-1.5: query keyword coverage
            0.10 * engagement_score * 10 +             # 0-1.0: directness of engagement
            0.10 * topic_density * 10 +                # 0-1.0: topic vocabulary density
            0.10 * bigram_coverage * 10 +              # 0-1.0: bigram coverage
            0.05 * max_sent_relevance * 10 +           # 0-0.5: peak sentence relevance
            structure_bonus * 10 +                      # 0-0.5: structure bonus
            alignment_bonus * 10 +                      # 0-0.5: alignment bonus
            - hedging_penalty * 10                      # penalty for hedging
        )
        
        # Apply length factor
        score = score * (0.7 + 0.3 * length_factor)
        
        # Clamp to 0-10
        score = max(0.0, min(10.0, score))
        
        return round(score, 4)
    
    except Exception:
        return 0.0