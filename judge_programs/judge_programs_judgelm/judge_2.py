def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using TF-IDF inspired weighting,
    query intent coverage analysis, and response quality signals.
    
    This variant uses:
    - IDF-weighted term matching (not simple overlap or Jaccard)
    - Query intent decomposition (question words, entities, action verbs)
    - Response coherence signals (sentence structure, repetition penalty)
    - Length-appropriateness scoring
    - N-gram coverage (bigrams/trigrams from query found in response)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 2:
            return 0.5
        
        # --- Tokenization ---
        def tokenize(text):
            text = text.lower()
            tokens = re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text)
            return tokens
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        # Stop words (common English)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'also', 'am'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        query_content = [t for t in query_tokens if t not in stop_words and len(t) > 1]
        response_content = [t for t in response_tokens if t not in stop_words and len(t) > 1]
        
        # --- 1. IDF-Weighted Term Coverage ---
        # Simulate IDF: rarer words in a "pseudo-corpus" of query+response get higher weight
        # We use inverse document frequency across response sentences
        response_sents = get_sentences(response)
        
        # Calculate pseudo-IDF for query content words based on how common they are generally
        # Use word length and character diversity as proxy for specificity
        def word_specificity(word):
            """Higher for more specific/unusual words"""
            base = min(len(word) / 4.0, 2.0)  # longer words tend to be more specific
            # Words with numbers are often specific
            if any(c.isdigit() for c in word):
                base += 0.5
            # Capitalize bonus (proper nouns - check original)
            return max(base, 0.5)
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        response_content_counter = Counter(response_content)
        
        if not query_content_set:
            # Query is all stop words; use all tokens
            query_content = [t for t in query_tokens if len(t) > 1]
            query_content_set = set(query_content)
        
        if not query_content_set:
            query_content_set = set(query_tokens)
            query_content = query_tokens
        
        # Weighted coverage
        total_weight = 0.0
        matched_weight = 0.0
        for word in query_content_set:
            w = word_specificity(word)
            total_weight += w
            if word in response_content_set:
                matched_weight += w
            else:
                # Check for partial matches (stemming approximation)
                prefix = word[:max(4, len(word) - 2)]
                for rw in response_content_set:
                    if rw.startswith(prefix) and abs(len(rw) - len(word)) <= 3:
                        matched_weight += w * 0.6
                        break
        
        idf_coverage = matched_weight / total_weight if total_weight > 0 else 0.0
        
        # --- 2. Bigram/Trigram Coverage from Query in Response ---
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        query_bigrams = set(get_ngrams(query_tokens, 2))
        response_bigrams = set(get_ngrams(response_tokens, 2))
        query_trigrams = set(get_ngrams(query_tokens, 3))
        response_trigrams = set(get_ngrams(response_tokens, 3))
        
        bigram_overlap = len(query_bigrams & response_bigrams) / max(len(query_bigrams), 1)
        trigram_overlap = len(query_trigrams & response_trigrams) / max(len(query_trigrams), 1)
        
        ngram_score = bigram_overlap * 0.6 + trigram_overlap * 0.4
        
        # --- 3. Topic/Domain Alignment ---
        # Extract "topic words" - nouns and key entities (heuristic: capitalized words, longer words)
        def extract_key_terms(text):
            """Extract likely important terms from text"""
            # Find capitalized words (potential proper nouns)
            caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
            # Find quoted terms
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
            quoted_flat = [q for pair in quoted for q in pair if q]
            # Numbers
            numbers = re.findall(r'\b\d+\b', text)
            return [t.lower() for t in caps + quoted_flat + numbers]
        
        query_key_terms = set(extract_key_terms(query))
        response_key_terms = set(extract_key_terms(response))
        response_lower = response.lower()
        
        key_term_hits = 0
        for term in query_key_terms:
            if term.lower() in response_lower:
                key_term_hits += 1
        
        key_term_score = key_term_hits / max(len(query_key_terms), 1) if query_key_terms else 0.5
        
        # --- 4. Query Intent Analysis ---
        query_lower = query.lower().strip()
        
        # Detect question type
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whom',
                         'whose', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does'}
        
        is_question = query.rstrip().endswith('?') or any(
            query_lower.startswith(qw + ' ') for qw in question_words
        )
        
        # Detect imperative/instruction
        imperative_verbs = {'create', 'write', 'make', 'list', 'describe', 'explain', 'identify',
                           'find', 'tell', 'show', 'give', 'provide', 'generate', 'rewrite',
                           'translate', 'summarize', 'compare', 'name', 'define', 'calculate',
                           'convert', 'classify', 'categorize', 'determine', 'analyze', 'remove'}
        
        is_imperative = any(query_lower.startswith(v + ' ') or query_lower.startswith(v + '\n')
                          for v in imperative_verbs)
        
        # Check if response seems to address the intent
        intent_score = 0.5  # neutral default
        
        if is_question:
            # Questions should get substantive answers
            if len(response_content) >= 3:
                intent_score = 0.7
            if len(response_sents) >= 1 and len(response_tokens) >= 10:
                intent_score = 0.8
        
        if is_imperative:
            # Check if the response actually performs the task
            if len(response_content) >= 5:
                intent_score = 0.7
            if len(response_sents) >= 1 and len(response_tokens) >= 8:
                intent_score = 0.8
        
        # --- 5. Response Quality Signals ---
        
        # 5a. Length appropriateness
        q_len = len(query_tokens)
        r_len = len(response_tokens)
        
        # Very short responses are usually bad (unless query is very simple)
        length_score = 1.0
        if r_len < 3:
            length_score = 0.15
        elif r_len < 5:
            length_score = 0.3
        elif r_len < 10:
            length_score = 0.6
        elif r_len < 20:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Very long responses might have padding/repetition
        if r_len > 200:
            length_score *= 0.95
        
        # 5b. Repetition penalty
        if len(response_sents) >= 2:
            sent_set = set()
            repeated = 0
            for s in response_sents:
                s_norm = ' '.join(tokenize(s))
                if s_norm in sent_set:
                    repeated += 1
                sent_set.add(s_norm)
            repetition_penalty = repeated / len(response_sents)
        else:
            repetition_penalty = 0.0
        
        # 5c. Check for garbage/code when not expected
        garbage_score = 0.0
        code_indicators = response.count('import ') + response.count('def ') + response.count('class ')
        query_expects_code = any(kw in query_lower for kw in ['code', 'python', 'function', 'program', 'script', 'html', 'tag'])
        
        if code_indicators >= 2 and not query_expects_code:
            garbage_score = 0.3
        
        # Check for excessive HTML when not expected
        html_tags = len(re.findall(r'<[^>]+>', response))
        query_expects_html = any(kw in query_lower for kw in ['html', 'tag', 'web', 'page'])
        if html_tags > 3 and not query_expects_html:
            garbage_score = max(garbage_score, 0.2)
        
        # 5d. Off-topic detection: response introduces many terms not related to query
        if len(response_content) > 5:
            response_unique = set(response_content)
            query_expanded = set(query_content)
            # How many response words have no relation to query words
            unrelated = 0
            for rw in response_unique:
                related = False
                for qw in query_expanded:
                    # Check prefix match or exact match
                    if rw == qw or rw.startswith(qw[:3]) or qw.startswith(rw[:3]):
                        related = True
                        break
                if not related:
                    unrelated += 1
            
            # Some unrelated words are fine (response should add info)
            # But if almost everything is unrelated, it's off-topic
            unrelated_ratio = unrelated / max(len(response_unique), 1)
            # Don't penalize too much - responses naturally introduce new terms
            off_topic_penalty = max(0, (unrelated_ratio - 0.7) * 1.5)  # only penalize if >70% unrelated
        else:
            off_topic_penalty = 0.0
        
        # 5e. Check if response is just echoing the query
        echo_ratio = 0.0
        if len(response_tokens) > 0:
            query_token_set = set(query_tokens)
            echo_count = sum(1 for t in response_tokens if t in query_token_set)
            echo_ratio = echo_count / len(response_tokens)
        
        # High echo with short response = bad (just repeating)
        echo_penalty = 0.0
        if echo_ratio > 0.8 and r_len < q_len * 1.2:
            echo_penalty = 0.2
        
        # --- 6. Semantic Field Overlap ---
        # Build a simple semantic field by looking at character trigrams shared between query and response
        def char_trigrams(text):
            text = text.lower()
            return set(text[i:i+3] for i in range(len(text) - 2) if text[i:i+3].strip())
        
        q_ctri = char_trigrams(query)
        r_ctri = char_trigrams(response[:500])  # limit for efficiency
        
        if q_ctri and r_ctri:
            char_tri_overlap = len(q_ctri & r_ctri) / max(len(q_ctri), 1)
        else:
            char_tri_overlap = 0.0
        
        # --- Combine all signals ---
        # Weights for each component
        score = (
            idf_coverage * 2.5 +          # 0-2.5: weighted term coverage
            ngram_score * 1.0 +            # 0-1.0: phrase-level matching
            key_term_score * 1.5 +         # 0-1.5: key entity coverage
            intent_score * 1.5 +           # 0-1.5: intent addressing
            length_score * 1.5 +           # 0-1.5: length appropriateness
            char_tri_overlap * 1.0 +       # 0-1.0: character-level semantic field
            (1.0 - repetition_penalty) * 0.5 +  # 0-0.5: repetition
            (1.0 - garbage_score) * 0.5 +  # 0-0.5: no garbage
            (1.0 - off_topic_penalty) * 0.3 +  # 0-0.3: on-topic
            (1.0 - echo_penalty) * 0.2     # 0-0.2: not just echo
        )
        # Max possible ~ 10.5
        
        # Normalize to 0-10 range
        score = score * 10.0 / 10.5
        
        # Clamp
        score = max(0.5, min(10.0, score))
        
        # Apply a floor penalty for truly minimal responses
        if r_len <= 2:
            score = min(score, 1.5)
        elif r_len <= 4:
            score = min(score, 3.0)
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except:
            return 2.0