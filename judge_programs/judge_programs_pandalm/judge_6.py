def judging_function(query, response):
    """
    Evaluates response relevance using a TF-IDF-inspired weighting scheme with
    query decomposition into intent components, coverage analysis, and 
    information density scoring.
    
    This variant focuses on:
    1. Query intent decomposition (verb phrases, noun phrases, modifiers)
    2. TF-IDF-inspired term importance weighting
    3. Response information density and coverage breadth
    4. Penalization for repetition, emptiness, and off-topic drift
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
            return 0.0
        
        # --- Tokenization helpers ---
        def tokenize(text):
            """Lowercase tokenization, removing punctuation."""
            text = text.lower()
            tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
            return tokens
        
        def get_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        # Common English stopwords
        stopwords = {
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
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'what',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'about', 'up',
        }
        
        # Document frequency approximation: words that appear in many contexts
        # get lower weight (simulating IDF)
        very_common = {
            'also', 'like', 'well', 'get', 'make', 'know', 'think', 'take',
            'come', 'go', 'see', 'look', 'find', 'give', 'tell', 'say',
            'thing', 'way', 'people', 'time', 'good', 'new', 'first', 'last',
            'long', 'great', 'little', 'right', 'old', 'big', 'high', 'small',
            'large', 'next', 'early', 'young', 'important', 'public', 'bad',
            'much', 'many', 'example', 'one', 'two', 'three',
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # --- 1. Query Intent Decomposition ---
        # Extract content words from query with importance weights
        query_content = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        
        # Assign TF-IDF-inspired weights to query terms
        query_term_weights = {}
        query_freq = Counter(query_content)
        for term, freq in query_freq.items():
            # TF component: sublinear
            tf = 1 + math.log(freq) if freq > 0 else 0
            # IDF-inspired: penalize very common words
            if term in very_common:
                idf = 0.5
            elif len(term) <= 3:
                idf = 0.7
            else:
                idf = 1.0
            # Boost terms that appear to be key nouns/concepts (longer words)
            length_boost = min(1.0 + (len(term) - 4) * 0.1, 1.5) if len(term) > 4 else 1.0
            query_term_weights[term] = tf * idf * length_boost
        
        # Identify action verbs in query (intent signals)
        action_verbs = {
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'generate', 'create', 'write', 'rewrite', 'summarize', 'analyze',
            'evaluate', 'discuss', 'define', 'identify', 'classify', 'suggest',
            'recommend', 'show', 'demonstrate', 'illustrate', 'outline', 'crop',
            'reduce', 'add', 'come', 'give', 'name', 'mention', 'state',
        }
        
        query_actions = [t for t in query_tokens if t in action_verbs]
        
        # --- 2. Weighted Coverage Score ---
        response_content = [t for t in response_tokens if t not in stopwords and len(t) > 1]
        response_content_set = set(response_content)
        
        if not query_term_weights:
            # If query has no content words, basic length check
            coverage_score = min(len(response_tokens) / 20.0, 1.0)
        else:
            total_weight = sum(query_term_weights.values())
            covered_weight = sum(
                query_term_weights[term] for term in query_term_weights
                if term in response_content_set
            )
            coverage_score = covered_weight / total_weight if total_weight > 0 else 0.0
        
        # --- 3. Semantic Field Expansion ---
        # Check if response contains words that are morphologically related to query terms
        # (stemming approximation via prefix matching)
        def get_stem(word):
            """Very rough stemming: strip common suffixes."""
            for suffix in ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful',
                          'less', 'ous', 'ive', 'ing', 'ated', 'ize', 'ise',
                          'ally', 'ity', 'ence', 'ance', 'ers', 'est', 'ed', 'ly',
                          'er', 'es', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        query_stems = set(get_stem(t) for t in query_content)
        response_stems = set(get_stem(t) for t in response_content)
        
        if query_stems:
            stem_overlap = len(query_stems & response_stems) / len(query_stems)
        else:
            stem_overlap = 0.0
        
        # --- 4. Information Density Score ---
        # Unique content words / total words ratio (penalizes repetition)
        response_unique_content = set(response_content)
        if len(response_tokens) > 0:
            density = len(response_unique_content) / len(response_tokens)
        else:
            density = 0.0
        
        # --- 5. Repetition Penalty ---
        response_content_freq = Counter(response_content)
        if response_content_freq:
            max_freq = max(response_content_freq.values())
            total_content = len(response_content)
            # High repetition of any single word is bad
            repetition_ratio = max_freq / total_content if total_content > 0 else 0
            repetition_penalty = max(0, (repetition_ratio - 0.15) * 2.0)  # penalize if any word > 15% of content
        else:
            repetition_penalty = 0.0
        
        # Also check for repeated phrases (3-grams)
        if len(response_tokens) >= 3:
            trigrams = [tuple(response_tokens[i:i+3]) for i in range(len(response_tokens) - 2)]
            trigram_freq = Counter(trigrams)
            if trigrams:
                max_tri = max(trigram_freq.values())
                tri_ratio = max_tri / len(trigrams)
                if tri_ratio > 0.1 and max_tri > 2:
                    repetition_penalty += tri_ratio * 2.0
        
        # --- 6. Response Length Adequacy ---
        query_len = len(query_tokens)
        response_len = len(response_tokens)
        
        # Expect response to be reasonably longer than query for most tasks
        if response_len < 5:
            length_score = 0.2
        elif response_len < 10:
            length_score = 0.5
        elif response_len < 20:
            length_score = 0.75
        elif response_len < 100:
            length_score = 1.0
        else:
            # Slight diminishing returns for very long responses
            length_score = 1.0 - min(0.2, (response_len - 100) / 1000.0)
        
        # --- 7. Sentence-level Relevance Distribution ---
        # Check that relevance is distributed across the response, not just in one sentence
        response_sents = get_sentences(response)
        if len(response_sents) > 1 and query_content:
            query_content_set = set(query_content)
            sent_relevance_scores = []
            for sent in response_sents:
                sent_tokens = set(tokenize(sent))
                if sent_tokens:
                    overlap = len(sent_tokens & query_content_set) + len(
                        set(get_stem(t) for t in sent_tokens if t not in stopwords) & query_stems
                    )
                    sent_relevance_scores.append(min(overlap / max(len(query_content_set), 1), 1.0))
                else:
                    sent_relevance_scores.append(0.0)
            
            # Reward responses where relevance is spread across sentences
            if sent_relevance_scores:
                avg_sent_relevance = sum(sent_relevance_scores) / len(sent_relevance_scores)
                # Also count how many sentences have at least some relevance
                relevant_sent_ratio = sum(1 for s in sent_relevance_scores if s > 0.05) / len(sent_relevance_scores)
                distribution_score = 0.5 * avg_sent_relevance + 0.5 * relevant_sent_ratio
            else:
                distribution_score = 0.0
        else:
            distribution_score = coverage_score  # single sentence, use coverage
        
        # --- 8. Topic Drift Detection ---
        # If response has many content words NOT related to query at all, penalize
        if response_unique_content and query_stems:
            response_related = sum(
                1 for w in response_unique_content
                if w in set(query_content) or get_stem(w) in query_stems
            )
            on_topic_ratio = response_related / len(response_unique_content) if response_unique_content else 0
        else:
            on_topic_ratio = 0.5  # neutral when we can't assess
        
        # --- 9. Query Type Awareness ---
        # Detect query type and check if response format matches
        query_lower = query.lower()
        format_bonus = 0.0
        
        if any(w in query_lower for w in ['compare', 'contrast', 'difference', 'similar']):
            # Should mention both sides
            comparison_words = {'but', 'while', 'whereas', 'however', 'unlike', 'differ', 'both', 'similar', 'different'}
            if any(w in set(response_tokens) for w in comparison_words):
                format_bonus = 0.3
        
        if any(w in query_lower for w in ['explain', 'describe', 'what']):
            # Should have explanatory content
            if len(response_sents) >= 2 and response_len >= 15:
                format_bonus = 0.2
        
        if any(w in query_lower for w in ['list', 'provide examples', 'name', 'examples']):
            # Should have multiple items
            comma_count = response.count(',')
            if comma_count >= 2:
                format_bonus = 0.2
        
        if any(w in query_lower for w in ['rewrite', 'rephrase', 'paraphrase']):
            # Should be different from input but maintain meaning
            format_bonus = 0.1
        
        # --- 10. Composite Score ---
        # Weight the components
        score = (
            coverage_score * 25.0 +          # How well query terms are covered
            stem_overlap * 15.0 +             # Morphological coverage
            distribution_score * 15.0 +       # Relevance spread across response
            density * 10.0 +                  # Information density
            length_score * 15.0 +             # Appropriate length
            on_topic_ratio * 10.0 +           # Staying on topic
            format_bonus * 10.0               # Format appropriateness
        )
        
        # Apply penalties
        score -= repetition_penalty * 15.0
        
        # Penalty for very short responses
        if response_len < 5:
            score *= 0.4
        elif response_len < 10:
            score *= 0.7
        
        # Penalty for responses that are just echoing the query
        if response_len > 0:
            query_set = set(query_tokens)
            response_set = set(response_tokens)
            if response_set and query_set:
                echo_ratio = len(response_set & query_set) / len(response_set)
                if echo_ratio > 0.8 and response_len <= len(query_tokens) * 1.5:
                    score *= 0.6  # Mostly just repeating the query
        
        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
        
    except Exception:
        try:
            # Absolute fallback
            if response and len(response.strip()) > 10:
                return 25.0
            return 5.0
        except Exception:
            return 5.0