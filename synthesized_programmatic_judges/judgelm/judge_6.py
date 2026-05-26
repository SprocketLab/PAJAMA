def judging_function(query, response):
    """
    Evaluates response relevance using a dependency/structural analysis approach:
    - Query intent classification and keyword extraction with TF weighting
    - Response structure quality (sentence completeness, coherence flow)
    - Semantic field matching using word co-occurrence neighborhoods
    - Penalization for repetition, off-topic content, and non-responsiveness
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 2:
            return 0.5
        
        # === Helper functions ===
        def tokenize(text):
            return re.findall(r'[a-z0-9]+', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 2]
        
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
            'if', 'while', 'about', 'up', 'its', 'it', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'am', 'an', 'any', 'dont', 'get', 'got', 'much',
            'make', 'made', 'please', 'well', 'us', 'like'
        }
        
        def content_words(tokens):
            return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        
        # === Tokenize ===
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # === 1. Query Intent Detection ===
        # Detect query type to understand what kind of response is expected
        query_lower = query.lower()
        
        intent_patterns = {
            'identify': r'\b(identify|classify|categorize|determine|which)\b',
            'list': r'\b(list|enumerate|name|give me)\b',
            'explain': r'\b(explain|describe|how|why|what is|what are|what was)\b',
            'create': r'\b(create|write|generate|make|compose|rewrite|produce)\b',
            'find': r'\b(find|where|locate|search)\b',
            'count': r'\b(how many|count|number of)\b',
            'opinion': r'\b(is it ok|should i|can i|do you think|is it)\b',
            'shorten': r'\b(shorter|shorten|summarize|brief|concise|regenerate)\b',
        }
        
        detected_intents = []
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query_lower):
                detected_intents.append(intent)
        
        # === 2. Keyword Importance Weighting ===
        # Use position-based and frequency-based weighting for query keywords
        query_content_set = set(query_content)
        
        # Words appearing later or in key positions get higher weight
        keyword_weights = {}
        for i, word in enumerate(query_content):
            # Words at the end of query often carry more semantic weight
            position_weight = 0.5 + 0.5 * (i / max(len(query_content), 1))
            # Rare words in query are more important
            freq = query_content.count(word)
            rarity_weight = 1.0 / math.sqrt(freq)
            keyword_weights[word] = max(keyword_weights.get(word, 0), position_weight * rarity_weight)
        
        # === 3. Weighted Keyword Coverage ===
        response_content_set = set(response_content)
        
        if keyword_weights:
            total_weight = sum(keyword_weights.values())
            covered_weight = sum(w for k, w in keyword_weights.items() if k in response_content_set)
            weighted_coverage = covered_weight / total_weight if total_weight > 0 else 0
        else:
            weighted_coverage = 0.5  # neutral if no content words
        
        # === 4. Semantic Field Expansion ===
        # Build simple semantic neighborhoods using character-level similarity (edit distance proxy)
        def char_trigrams(word):
            if len(word) < 3:
                return {word}
            return {word[i:i+3] for i in range(len(word) - 2)}
        
        def fuzzy_match_score(w1, w2):
            """Trigram-based similarity"""
            t1 = char_trigrams(w1)
            t2 = char_trigrams(w2)
            if not t1 or not t2:
                return 0
            intersection = len(t1 & t2)
            union = len(t1 | t2)
            return intersection / union if union > 0 else 0
        
        # Check for fuzzy matches between query content words and response
        fuzzy_matches = 0
        for qw in query_content_set:
            best_match = 0
            for rw in response_content_set:
                if qw == rw:
                    best_match = 1.0
                    break
                # Check stems (prefix matching)
                min_len = min(len(qw), len(rw))
                if min_len >= 4 and qw[:min_len-1] == rw[:min_len-1]:
                    best_match = max(best_match, 0.85)
                else:
                    score = fuzzy_match_score(qw, rw)
                    if score > 0.5:
                        best_match = max(best_match, score * 0.7)
            fuzzy_matches += best_match
        
        fuzzy_coverage = fuzzy_matches / len(query_content_set) if query_content_set else 0.5
        
        # === 5. Response Structure Quality ===
        response_sentences = get_sentences(response)
        num_sentences = len(response_sentences)
        
        # Sentence completeness: check if sentences seem complete
        complete_sentences = 0
        for sent in response_sentences:
            words = tokenize(sent)
            if len(words) >= 3:
                complete_sentences += 1
            elif len(words) >= 1:
                complete_sentences += 0.5
        
        sentence_quality = complete_sentences / max(num_sentences, 1)
        
        # === 6. Repetition Penalty ===
        # Detect repeated phrases/sentences
        if len(response_sentences) > 1:
            unique_sents = set()
            repeated = 0
            for sent in response_sentences:
                normalized = ' '.join(tokenize(sent))
                if normalized in unique_sents:
                    repeated += 1
                unique_sents.add(normalized)
            repetition_ratio = repeated / len(response_sentences)
        else:
            repetition_ratio = 0
        
        # Word-level repetition (beyond normal)
        if response_content:
            word_counts = Counter(response_content)
            total_content = len(response_content)
            unique_content = len(word_counts)
            # Type-token ratio for content words
            ttr = unique_content / total_content if total_content > 0 else 1
            # Very low TTR suggests excessive repetition
            word_repetition_penalty = max(0, 1 - ttr) * 0.5 if ttr < 0.3 else 0
        else:
            word_repetition_penalty = 0
        
        repetition_penalty = repetition_ratio * 0.6 + word_repetition_penalty
        
        # === 7. Off-topic Detection ===
        # Check if response contains significant content unrelated to query
        if response_content and query_content_set:
            related_words = 0
            for rw in response_content:
                if rw in query_content_set:
                    related_words += 1
                else:
                    # Check if it's in a plausible semantic field
                    for qw in query_content_set:
                        if fuzzy_match_score(qw, rw) > 0.4:
                            related_words += 0.5
                            break
            topic_alignment = related_words / len(response_content) if response_content else 0
        else:
            topic_alignment = 0.3
        
        # === 8. Response Length Appropriateness ===
        query_len = len(query_tokens)
        response_len = len(response_tokens)
        
        # Very short responses to substantive queries are usually bad
        if response_len < 3 and query_len > 5:
            length_score = 0.15
        elif response_len < 5 and query_len > 10:
            length_score = 0.25
        elif response_len > query_len * 15:
            # Excessively long might indicate rambling
            length_score = 0.6
        else:
            # Sweet spot
            ratio = response_len / max(query_len, 1)
            if ratio < 0.3:
                length_score = 0.3
            elif ratio < 1:
                length_score = 0.5 + 0.3 * ratio
            elif ratio <= 8:
                length_score = 0.9
            else:
                length_score = max(0.5, 0.9 - (ratio - 8) * 0.03)
        
        # === 9. Direct Address Detection ===
        # Check if response directly addresses the query structure
        # E.g., if query asks "What is X?", response should mention X
        
        # Extract potential named entities / key phrases from query
        # (capitalized words, quoted terms, proper nouns)
        proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query)
        quoted_terms = re.findall(r"['\"]([^'\"]+)['\"]", query)
        
        key_entities = set()
        for pn in proper_nouns:
            key_entities.update(tokenize(pn))
        for qt in quoted_terms:
            key_entities.update(tokenize(qt))
        # Remove stop words from entities
        key_entities = {e for e in key_entities if e not in STOP_WORDS and len(e) > 1}
        
        if key_entities:
            entity_coverage = len(key_entities & response_content_set) / len(key_entities)
        else:
            entity_coverage = 0.5  # neutral
        
        # === 10. HTML/Code Noise Detection ===
        # Penalize responses that contain lots of code/HTML when not asked for
        code_query = bool(re.search(r'\b(code|html|program|script|function|tag)\b', query_lower))
        code_chars = len(re.findall(r'[<>{}\[\]();=]', response))
        code_ratio = code_chars / max(len(response), 1)
        
        if not code_query and code_ratio > 0.05:
            noise_penalty = min(0.4, code_ratio * 3)
        else:
            noise_penalty = 0
        
        # === 11. Question Echo Detection ===
        # Penalize if response just repeats the query without adding info
        if query_content and response_content:
            query_bigrams = set()
            for i in range(len(query_content) - 1):
                query_bigrams.add((query_content[i], query_content[i+1]))
            
            response_bigrams = set()
            for i in range(len(response_content) - 1):
                response_bigrams.add((response_content[i], response_content[i+1]))
            
            if query_bigrams and response_bigrams:
                echo_ratio = len(query_bigrams & response_bigrams) / len(response_bigrams) if response_bigrams else 0
            else:
                echo_ratio = 0
            
            # Some echo is fine (shows relevance), too much means just repeating
            if echo_ratio > 0.7 and len(response_content) <= len(query_content) * 1.2:
                echo_penalty = 0.2
            else:
                echo_penalty = 0
        else:
            echo_penalty = 0
        
        # === 12. Substantiveness Score ===
        # Does the response provide actual information/content?
        if response_content:
            # Count unique content words not in query
            new_content = response_content_set - query_content_set
            substantiveness = min(1.0, len(new_content) / max(5, len(query_content_set)))
        else:
            substantiveness = 0.1
        
        # === Combine all signals ===
        # Weighted combination
        raw_score = (
            weighted_coverage * 2.5 +      # Core: do response words match query keywords?
            fuzzy_coverage * 1.5 +          # Fuzzy/stem matching
            entity_coverage * 1.5 +         # Named entity coverage
            topic_alignment * 0.8 +         # General topic staying
            sentence_quality * 0.8 +        # Well-formed sentences
            length_score * 1.2 +            # Appropriate length
            substantiveness * 1.0 +         # Provides new information
            - repetition_penalty * 2.0 +    # Penalize repetition
            - noise_penalty * 2.5 +         # Penalize code noise
            - echo_penalty * 1.5            # Penalize pure echo
        )
        
        # Normalize: theoretical max is about 2.5+1.5+1.5+0.8+0.8+1.2+1.0 = 9.3
        # Theoretical min is about -2.0-2.5-1.5 = -6.0
        # Map to 0-10 range
        max_possible = 9.3
        min_possible = -2.0
        
        normalized = (raw_score - min_possible) / (max_possible - min_possible) * 10
        
        # Clamp
        final_score = max(0.5, min(10.0, normalized))
        
        # === Special case adjustments ===
        
        # Very short, dismissive responses
        if len(response.strip()) < 10 and len(query.strip()) > 20:
            final_score = min(final_score, 2.5)
        
        # Single word/very terse responses
        if len(response_tokens) <= 2:
            final_score = min(final_score, 3.0)
        
        # Response is just punctuation or whitespace
        if len(response_content) == 0:
            final_score = min(final_score, 1.0)
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0