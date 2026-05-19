def judging_function(query, response):
    """
    Evaluates response relevance using a TF-IDF-inspired approach with
    information-theoretic measures: mutual information between query and response,
    query coverage analysis, and response focus/coherence scoring.
    
    This variant uses:
    - TF-IDF-like term weighting (not raw overlap)
    - Query intent extraction via question word analysis
    - Information density and redundancy penalties
    - Sentence-level relevance distribution analysis
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        # --- Text preprocessing ---
        def tokenize(text):
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return [w for w in text.split() if len(w) > 0]
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'about', 'up', 'out', 'then', 'there', 'here', 'also', 'over', 'under',
            'again', 'once', 'while', 'since', 'until', 'although', 'though'
        }
        
        def content_words(tokens):
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not response_tokens:
            return 0.0
        
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        if not query_content:
            query_content = query_tokens[:10]
        
        # --- 1. TF-IDF-inspired term importance ---
        # Treat query terms as "document frequency" proxy: rarer words in a 
        # general sense get higher weight using word length as a crude proxy
        # for specificity, plus position-based weighting
        
        def term_importance(word):
            """Estimate importance: longer, less common words matter more."""
            base = min(len(word) / 4.0, 2.0)  # length factor
            # Common short words get lower weight
            if len(word) <= 3:
                base *= 0.5
            return max(base, 0.3)
        
        query_term_weights = {}
        for i, w in enumerate(query_content):
            # Position weight: earlier query terms slightly more important
            pos_weight = 1.0 / (1.0 + 0.1 * i)
            importance = term_importance(w) * pos_weight
            if w in query_term_weights:
                query_term_weights[w] = max(query_term_weights[w], importance)
            else:
                query_term_weights[w] = importance
        
        # --- 2. Weighted query coverage ---
        response_content_set = set(response_content)
        response_token_set = set(response_tokens)
        
        covered_weight = 0.0
        total_weight = 0.0
        for term, weight in query_term_weights.items():
            total_weight += weight
            if term in response_content_set or term in response_token_set:
                covered_weight += weight
            else:
                # Check for partial/stem matches
                for rt in response_content_set:
                    if len(term) >= 4 and len(rt) >= 4:
                        # Shared prefix of at least 4 chars
                        common_len = 0
                        for c1, c2 in zip(term, rt):
                            if c1 == c2:
                                common_len += 1
                            else:
                                break
                        if common_len >= min(len(term), 4):
                            covered_weight += weight * 0.7
                            break
        
        coverage_score = covered_weight / total_weight if total_weight > 0 else 0.0
        
        # --- 3. Sentence-level relevance distribution ---
        # Check how many sentences in the response are relevant to the query
        response_sentences = get_sentences(response)
        
        if not response_sentences:
            response_sentences = [response]
        
        query_content_set = set(query_content)
        sentence_relevance_scores = []
        
        for sent in response_sentences:
            sent_tokens = content_words(tokenize(sent))
            if not sent_tokens:
                sentence_relevance_scores.append(0.0)
                continue
            
            sent_set = set(sent_tokens)
            # How much of this sentence relates to query topics
            relevant_count = sum(1 for t in sent_tokens if t in query_content_set or
                               any(len(t) >= 4 and len(q) >= 4 and 
                                   (t.startswith(q[:4]) or q.startswith(t[:4]))
                                   for q in query_content_set))
            sent_relevance = relevant_count / len(sent_tokens) if sent_tokens else 0
            sentence_relevance_scores.append(min(sent_relevance, 1.0))
        
        # Average sentence relevance
        avg_sent_relevance = sum(sentence_relevance_scores) / len(sentence_relevance_scores) if sentence_relevance_scores else 0
        
        # Proportion of sentences that have ANY relevance
        relevant_sent_ratio = sum(1 for s in sentence_relevance_scores if s > 0.05) / len(sentence_relevance_scores) if sentence_relevance_scores else 0
        
        # --- 4. Response information density and quality ---
        response_counter = Counter(response_content)
        total_content = len(response_content)
        unique_content = len(set(response_content))
        
        # Lexical diversity (penalize repetition)
        if total_content > 0:
            diversity = unique_content / total_content
        else:
            diversity = 0.0
        
        # Repetition penalty: check for repeated phrases (3-grams)
        if len(response_tokens) >= 3:
            trigrams = [tuple(response_tokens[i:i+3]) for i in range(len(response_tokens)-2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0
        else:
            repetition_ratio = 0.0
        
        repetition_penalty = max(0, 1.0 - repetition_ratio * 2.0)
        
        # --- 5. Response length appropriateness ---
        query_len = len(query_tokens)
        resp_len = len(response_tokens)
        
        # Very short responses are usually worse
        if resp_len < 5:
            length_score = 0.2
        elif resp_len < 10:
            length_score = 0.5
        elif resp_len < 20:
            length_score = 0.75
        elif resp_len <= 150:
            length_score = 1.0
        elif resp_len <= 300:
            length_score = 0.95
        else:
            # Very long responses might have padding
            length_score = 0.85
        
        # --- 6. Query intent matching ---
        query_lower = query.lower()
        
        # Detect query type and check if response matches expected pattern
        intent_bonus = 0.0
        
        # Action verbs in query
        action_patterns = {
            'explain': ['means', 'refers', 'implies', 'suggest', 'describe', 'definition', 'concept'],
            'describe': ['is', 'are', 'involves', 'includes', 'consists', 'features'],
            'compare': ['both', 'while', 'whereas', 'however', 'unlike', 'similar', 'different', 'contrast'],
            'list': ['first', 'second', 'third', 'also', 'another', 'additionally'],
            'provide': ['example', 'instance', 'such as', 'including'],
            'rewrite': [],  # just needs to be different from input
            'generate': [],
            'create': [],
            'write': [],
        }
        
        resp_lower = response.lower()
        for action, indicators in action_patterns.items():
            if action in query_lower:
                if indicators:
                    matches = sum(1 for ind in indicators if ind in resp_lower)
                    intent_bonus = min(matches * 0.08, 0.3)
                break
        
        # Question word handling
        if any(qw in query_lower for qw in ['what', 'why', 'how', 'when', 'where', 'who']):
            # Response should not just echo the question
            if '?' not in response or len(response_sentences) > 1:
                intent_bonus += 0.05
        
        # --- 7. Semantic field overlap using co-occurrence ---
        # Build simple "semantic fields" from shared character 4-grams between words
        def char_ngrams(word, n=4):
            if len(word) < n:
                return {word}
            return {word[i:i+n] for i in range(len(word)-n+1)}
        
        query_char_ngrams = set()
        for w in query_content:
            query_char_ngrams.update(char_ngrams(w))
        
        response_char_ngrams = set()
        for w in response_content:
            response_char_ngrams.update(char_ngrams(w))
        
        if query_char_ngrams and response_char_ngrams:
            char_overlap = len(query_char_ngrams & response_char_ngrams) / len(query_char_ngrams)
        else:
            char_overlap = 0.0
        
        # --- 8. Penalize empty/nonsensical responses ---
        nonsense_penalty = 1.0
        
        # Check for excessive repetition of single words
        if response_counter:
            max_word_freq = max(response_counter.values())
            if total_content > 5 and max_word_freq / total_content > 0.4:
                nonsense_penalty *= 0.6
        
        # Check if response is just echoing the query
        if response.strip().lower() == query.strip().lower():
            nonsense_penalty *= 0.3
        
        # Check for truncated responses (ending mid-word/sentence)
        if response.rstrip()[-1:] not in '.!?")\']' and len(response) > 50:
            # Might be truncated - small penalty
            nonsense_penalty *= 0.9
        
        # Check for <noinput> or similar non-responses
        if re.match(r'^\s*<\s*noinput\s*>\s*$', response, re.IGNORECASE):
            return 0.5
        
        # --- 9. Mutual information approximation ---
        # How much knowing the query reduces uncertainty about the response
        query_bigrams = set()
        for i in range(len(query_content) - 1):
            query_bigrams.add((query_content[i], query_content[i+1]))
        
        response_bigrams = set()
        for i in range(len(response_content) - 1):
            response_bigrams.add((response_content[i], response_content[i+1]))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        # --- Combine all signals ---
        # Weights chosen to emphasize coverage and sentence relevance
        score = (
            coverage_score * 30.0 +           # 0-30: weighted query term coverage
            avg_sent_relevance * 10.0 +        # 0-10: sentence-level relevance
            relevant_sent_ratio * 8.0 +        # 0-8: proportion of relevant sentences
            char_overlap * 10.0 +              # 0-10: character n-gram semantic overlap
            bigram_overlap * 8.0 +             # 0-8: bigram overlap (phrase matching)
            length_score * 8.0 +               # 0-8: appropriate length
            diversity * 6.0 +                  # 0-6: lexical diversity
            intent_bonus * 10.0 +              # 0-3: intent matching bonus
            repetition_penalty * 5.0           # 0-5: repetition penalty
        )
        
        # Apply nonsense penalty
        score *= nonsense_penalty
        
        # Normalize to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
        
    except Exception:
        # Fallback: return a neutral score
        try:
            if response and response.strip():
                return 25.0
            return 0.0
        except Exception:
            return 0.0