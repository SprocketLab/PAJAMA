def judging_function(query, response):
    """
    Evaluates response relevance using a dependency-graph inspired approach:
    - Extracts key "question words" and "topic phrases" from the query
    - Builds a weighted term importance model based on query structure
    - Scores response on: topic coverage, response coherence, question-type alignment,
      and penalizes repetition/off-topic content
    
    This variant focuses on: query decomposition into intent + topic slots,
    then measuring how well the response fills those slots.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if not query or not response:
            return 0.0
        
        # --- Tokenization helpers ---
        def tokenize(text):
            return re.findall(r'[a-zA-Z0-9]+', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        STOP_WORDS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'am', 'any', 'get', 'got', 'like',
            'make', 'much', 'well', 'back', 'even', 'give', 'go', 'know', 'take',
            'come', 'think', 'say', 'see', 'want', 'look', 'use', 'find', 'tell',
            'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let',
            'begin', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
            'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay',
            'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead',
            'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read',
            'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer',
            'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve',
            'die', 'send', 'expect', 'build', 'stay', 'fall', 'oh', 'yeah',
            'ok', 'okay', 'please', 'thank', 'thanks', 'yes', 'sure', 'right',
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # --- 1. Query Decomposition: Intent Detection ---
        query_lower = query.lower()
        
        # Detect question type / intent
        intent_type = 'general'
        if re.search(r'\b(how many|how much|count|number of)\b', query_lower):
            intent_type = 'quantitative'
        elif re.search(r'\b(who|whom|whose)\b', query_lower):
            intent_type = 'person'
        elif re.search(r'\b(where|location|place)\b', query_lower):
            intent_type = 'location'
        elif re.search(r'\b(when|date|year|time)\b', query_lower):
            intent_type = 'temporal'
        elif re.search(r'\b(why|reason|cause)\b', query_lower):
            intent_type = 'causal'
        elif re.search(r'\b(how|explain|describe|steps)\b', query_lower):
            intent_type = 'procedural'
        elif re.search(r'\b(what|which|identify|name|list)\b', query_lower):
            intent_type = 'factual'
        elif re.search(r'\b(rewrite|create|write|generate|make|compose)\b', query_lower):
            intent_type = 'creative'
        elif re.search(r'\b(is it|can i|should i|do you|are there)\b', query_lower):
            intent_type = 'yes_no'
        
        # --- 2. Extract Topic Slots from Query ---
        # Content words from query (non-stop words) with positional weighting
        query_content_words = []
        for i, token in enumerate(query_tokens):
            if token not in STOP_WORDS and len(token) > 1:
                query_content_words.append(token)
        
        # Build importance weights: words appearing in quotes, capitalized, 
        # or as named entities get higher weight
        word_importance = {}
        for w in query_content_words:
            word_importance[w] = word_importance.get(w, 0) + 1.0
        
        # Boost words in quotes
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
        for phrase in quoted:
            for w in tokenize(phrase):
                if w in word_importance:
                    word_importance[w] += 2.0
                else:
                    word_importance[w] = 3.0
        
        # Boost proper nouns (capitalized words in original query)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        for pn in proper_nouns:
            w = pn.lower()
            if w not in STOP_WORDS:
                if w in word_importance:
                    word_importance[w] += 1.5
                else:
                    word_importance[w] = 2.5
        
        # Normalize importance
        if word_importance:
            max_imp = max(word_importance.values())
            if max_imp > 0:
                for w in word_importance:
                    word_importance[w] /= max_imp
        
        # --- 3. Topic Slot Coverage Score ---
        response_word_set = set(response_tokens)
        
        if word_importance:
            covered_weight = 0.0
            total_weight = 0.0
            for w, imp in word_importance.items():
                total_weight += imp
                if w in response_word_set:
                    covered_weight += imp
            topic_coverage = covered_weight / total_weight if total_weight > 0 else 0.0
        else:
            # Fallback: simple overlap
            q_set = set(query_tokens)
            r_set = set(response_tokens)
            common = q_set & r_set
            topic_coverage = len(common) / max(len(q_set), 1)
        
        # --- 4. Semantic Density: ratio of response that is "on-topic" ---
        response_content = [t for t in response_tokens if t not in STOP_WORDS and len(t) > 1]
        query_content_set = set(query_content_words)
        
        if response_content:
            on_topic_count = sum(1 for t in response_content if t in query_content_set)
            # Also count semantically related (shared stems via prefix matching)
            for t in response_content:
                if t not in query_content_set:
                    for qw in query_content_set:
                        # Crude stem matching: share prefix of length >= 4
                        prefix_len = min(len(t), len(qw), max(4, min(len(t), len(qw)) - 2))
                        if prefix_len >= 4 and t[:prefix_len] == qw[:prefix_len]:
                            on_topic_count += 0.5
                            break
            
            semantic_density = min(on_topic_count / len(response_content), 1.0)
        else:
            semantic_density = 0.0
        
        # --- 5. Response Substance Score ---
        response_sentences = get_sentences(response)
        num_sentences = len(response_sentences)
        
        # Penalize very short responses
        response_len = len(response_tokens)
        if response_len <= 1:
            length_score = 0.05
        elif response_len <= 3:
            length_score = 0.2
        elif response_len <= 8:
            length_score = 0.5
        elif response_len <= 15:
            length_score = 0.7
        elif response_len <= 200:
            length_score = 1.0
        else:
            # Slight penalty for extremely long responses (might be off-topic rambling)
            length_score = max(0.6, 1.0 - (response_len - 200) / 2000)
        
        # --- 6. Intent Alignment Score ---
        resp_lower = response.lower()
        intent_score = 0.5  # baseline
        
        if intent_type == 'quantitative':
            # Should contain numbers
            if re.search(r'\d+', response):
                intent_score = 0.9
            else:
                intent_score = 0.3
        elif intent_type == 'person':
            # Should contain proper names or person references
            if re.search(r'\b[A-Z][a-z]+\b', response):
                intent_score = 0.8
            else:
                intent_score = 0.35
        elif intent_type == 'location':
            if re.search(r'\b[A-Z][a-z]+\b', response) or re.search(r'\b(city|country|state|town|place|location|street|building)\b', resp_lower):
                intent_score = 0.8
            else:
                intent_score = 0.35
        elif intent_type == 'temporal':
            if re.search(r'\b(\d{4}|\d{1,2}/\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december|year|century|decade)\b', resp_lower):
                intent_score = 0.85
            else:
                intent_score = 0.35
        elif intent_type == 'causal':
            if re.search(r'\b(because|reason|due to|caused|result|since|therefore|thus)\b', resp_lower):
                intent_score = 0.85
            else:
                intent_score = 0.4
        elif intent_type == 'procedural':
            if re.search(r'\b(step|first|then|next|finally|process|method|way)\b', resp_lower) or num_sentences >= 2:
                intent_score = 0.8
            else:
                intent_score = 0.4
        elif intent_type == 'creative':
            # For creative tasks, response should be substantially different from query
            if response_len > 5 and num_sentences >= 1:
                intent_score = 0.8
            else:
                intent_score = 0.3
        elif intent_type == 'yes_no':
            if re.search(r'\b(yes|no|absolutely|certainly|definitely|of course|sure|it is|it can|you can|you should)\b', resp_lower):
                intent_score = 0.75
                # Bonus if explanation follows
                if num_sentences >= 2 or response_len > 15:
                    intent_score = 0.9
            else:
                intent_score = 0.4
        elif intent_type == 'factual':
            if response_len >= 3:
                intent_score = 0.7
            else:
                intent_score = 0.3
        else:
            # General: reward substantive responses
            if num_sentences >= 2 and response_len >= 10:
                intent_score = 0.75
            elif response_len >= 5:
                intent_score = 0.6
            else:
                intent_score = 0.35
        
        # --- 7. Repetition Penalty ---
        if len(response_tokens) > 5:
            token_counts = Counter(response_tokens)
            content_tokens_in_resp = [t for t in response_tokens if t not in STOP_WORDS and len(t) > 1]
            if content_tokens_in_resp:
                content_counts = Counter(content_tokens_in_resp)
                max_repeat = max(content_counts.values())
                avg_repeat = sum(content_counts.values()) / len(content_counts)
                
                # High repetition is bad
                if max_repeat > 5 and max_repeat / len(content_tokens_in_resp) > 0.15:
                    repetition_penalty = 0.5
                elif avg_repeat > 3:
                    repetition_penalty = 0.7
                else:
                    repetition_penalty = 1.0
            else:
                repetition_penalty = 0.8
        else:
            repetition_penalty = 1.0
        
        # Check for sentence-level repetition
        if len(response_sentences) >= 3:
            unique_sents = set()
            dup_count = 0
            for s in response_sentences:
                s_normalized = ' '.join(tokenize(s))
                if s_normalized in unique_sents:
                    dup_count += 1
                unique_sents.add(s_normalized)
            if dup_count > 0:
                sent_rep_penalty = max(0.4, 1.0 - dup_count * 0.2)
                repetition_penalty *= sent_rep_penalty
        
        # --- 8. Off-topic / Noise Penalty ---
        noise_penalty = 1.0
        
        # Check for HTML/code when not asked for
        has_code_query = bool(re.search(r'\b(code|html|program|script|function|tag)\b', query_lower))
        has_code_response = bool(re.search(r'<[a-z]+>|import |def |class |function\(|var |const |let ', resp_lower))
        
        if has_code_response and not has_code_query:
            noise_penalty *= 0.5
        
        # Check for "Question:" / "Input:" / "Output:" patterns suggesting template confusion
        template_matches = len(re.findall(r'\b(Question|Input|Output|Answer):', response))
        if template_matches > 2:
            noise_penalty *= max(0.3, 1.0 - template_matches * 0.1)
        
        # --- 9. Response directly addresses query (first-sentence relevance) ---
        first_sentence_score = 0.5
        if response_sentences:
            first_sent_tokens = set(tokenize(response_sentences[0]))
            first_sent_content = first_sent_tokens - STOP_WORDS
            query_content_overlap = len(first_sent_content & query_content_set)
            if query_content_set:
                first_sentence_score = min(1.0, 0.3 + 0.7 * query_content_overlap / max(len(query_content_set), 1))
            else:
                first_sentence_score = 0.5
        
        # --- 10. Bigram topic alignment ---
        def get_bigrams(tokens):
            return [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]
        
        query_content_tokens = [t for t in query_tokens if t not in STOP_WORDS]
        resp_content_tokens = [t for t in response_tokens if t not in STOP_WORDS]
        
        bigram_score = 0.0
        if len(query_content_tokens) >= 2 and len(resp_content_tokens) >= 2:
            q_bigrams = set(get_bigrams(query_content_tokens))
            r_bigrams = set(get_bigrams(resp_content_tokens))
            if q_bigrams:
                bigram_overlap = len(q_bigrams & r_bigrams)
                bigram_score = min(1.0, bigram_overlap / max(len(q_bigrams), 1))
        
        # --- Combine scores with weighted formula ---
        # Weights chosen to emphasize topic coverage and intent alignment
        final_score = (
            topic_coverage * 2.5 +          # How well topic slots are covered
            semantic_density * 1.5 +         # How much of response is on-topic
            intent_score * 2.0 +             # Does response match question type
            length_score * 1.0 +             # Adequate length
            first_sentence_score * 1.5 +     # First sentence relevance
            bigram_score * 1.5               # Phrase-level alignment
        )
        
        # Apply penalties
        final_score *= repetition_penalty
        final_score *= noise_penalty
        
        # Normalize to 0-10 scale
        max_possible = 2.5 + 1.5 + 2.0 + 1.0 + 1.5 + 1.5  # = 10.0
        final_score = (final_score / max_possible) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0