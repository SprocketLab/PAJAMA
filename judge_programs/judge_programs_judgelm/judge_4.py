def judging_function(query, response):
    """
    Evaluates response relevance using a topic-modeling inspired approach:
    - Builds "topic vectors" from query using keyword extraction with TF-IDF-like weighting
    - Measures how well the response covers query topics via sentence-level analysis
    - Penalizes responses that drift off-topic by measuring response coherence
    - Uses edit-distance based fuzzy matching for topic terms
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
        
        # --- Tokenization and preprocessing ---
        def tokenize(text):
            """Tokenize and normalize text."""
            text = text.lower()
            tokens = re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
            return tokens
        
        # Common English stopwords
        STOPWORDS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'although',
            'though', 'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'whose', 'up', 'down',
            'about', 'also', 'am', 'an', 'any', 'get', 'got', 'like', 'make',
            'made', 'much', 'many', 'well', 'back', 'even', 'still', 'way',
            'take', 'since', 'another', 'know', 'help', 'tell', 'give', 'us',
            'etc', 'really', 'please', 'also', 'however', 'yet', 'already'
        }
        
        # Imperative/question words that signal query intent but aren't content
        QUERY_SIGNAL_WORDS = {
            'identify', 'explain', 'describe', 'list', 'create', 'write',
            'rewrite', 'generate', 'find', 'show', 'provide', 'name',
            'determine', 'compare', 'summarize', 'translate', 'convert',
            'calculate', 'define', 'discuss', 'elaborate', 'suggest',
            'recommend', 'analyze', 'evaluate', 'classify', 'categorize',
            'regenerate', 'remove', 'shorter', 'longer', 'last', 'response',
            'make', 'want', 'can', 'could', 'please', 'ok', 'okay'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not response_tokens:
            return 0.0
        
        # --- Extract query topic terms with importance weights ---
        def extract_topic_terms(tokens):
            """Extract content words with importance weighting."""
            content_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
            freq = Counter(content_tokens)
            # Weight by inverse frequency in general English (approximated by length)
            # and by position (earlier words in query tend to be more important for topic)
            weighted = {}
            for i, token in enumerate(tokens):
                if token in STOPWORDS or len(token) <= 1:
                    continue
                if token in QUERY_SIGNAL_WORDS:
                    weight = 0.3  # Lower weight for instruction words
                else:
                    # Longer, less common words get higher weight
                    length_bonus = min(len(token) / 5.0, 1.5)
                    # Rarity bonus (words appearing once are likely more specific)
                    rarity = 1.0 / (freq[token] ** 0.5)
                    weight = length_bonus * rarity
                if token in weighted:
                    weighted[token] = max(weighted[token], weight)
                else:
                    weighted[token] = weight
            return weighted
        
        query_topics = extract_topic_terms(query_tokens)
        
        if not query_topics:
            # Fallback: use all non-stopword tokens with equal weight
            query_topics = {t: 1.0 for t in query_tokens if t not in STOPWORDS and len(t) > 1}
        
        if not query_topics:
            # Ultra fallback
            query_topics = {t: 1.0 for t in query_tokens if len(t) > 0}
        
        # --- Fuzzy matching function ---
        def edit_distance_ratio(s1, s2):
            """Return similarity ratio based on edit distance (0 to 1)."""
            if s1 == s2:
                return 1.0
            len1, len2 = len(s1), len(s2)
            if max(len1, len2) == 0:
                return 1.0
            # Quick reject
            if abs(len1 - len2) > max(len1, len2) * 0.4:
                return 0.0
            # Simple Levenshtein for short words
            if len1 > 12 or len2 > 12:
                # For long words, use prefix/suffix matching
                common_prefix = 0
                for i in range(min(len1, len2)):
                    if s1[i] == s2[i]:
                        common_prefix += 1
                    else:
                        break
                common_suffix = 0
                for i in range(1, min(len1, len2) - common_prefix + 1):
                    if s1[-i] == s2[-i]:
                        common_suffix += 1
                    else:
                        break
                return (common_prefix + common_suffix) / max(len1, len2)
            
            # Full edit distance
            dp = list(range(len2 + 1))
            for i in range(1, len1 + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, len2 + 1):
                    temp = dp[j]
                    if s1[i-1] == s2[j-1]:
                        dp[j] = prev
                    else:
                        dp[j] = 1 + min(prev, dp[j], dp[j-1])
                    prev = temp
            
            distance = dp[len2]
            return 1.0 - distance / max(len1, len2)
        
        # --- Sentence-level analysis of response ---
        def split_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 2]
        
        response_sentences = split_sentences(response)
        if not response_sentences:
            response_sentences = [response]
        
        # --- Score 1: Topic Coverage (how many query topics are addressed) ---
        response_token_set = set(response_tokens)
        
        topic_coverage_score = 0.0
        total_topic_weight = sum(query_topics.values())
        
        if total_topic_weight == 0:
            topic_coverage_score = 5.0
        else:
            covered_weight = 0.0
            for term, weight in query_topics.items():
                # Exact match
                if term in response_token_set:
                    covered_weight += weight
                    continue
                # Fuzzy match
                best_match = 0.0
                for rt in response_token_set:
                    if abs(len(rt) - len(term)) <= 2:
                        sim = edit_distance_ratio(term, rt)
                        if sim > best_match:
                            best_match = sim
                        if best_match > 0.85:
                            break
                if best_match > 0.75:
                    covered_weight += weight * best_match
                # Check for substring containment (e.g., "baseball" in "baseballs")
                elif any(term in rt or rt in term for rt in response_token_set if len(rt) > 3):
                    covered_weight += weight * 0.8
            
            topic_coverage_score = (covered_weight / total_topic_weight) * 10.0
        
        # --- Score 2: Response Coherence / On-topic ratio ---
        response_content = [t for t in response_tokens if t not in STOPWORDS and len(t) > 1]
        response_content_set = set(response_content)
        
        # Build an expanded topic set from query (including related terms via co-occurrence)
        query_content = set(t for t in query_tokens if t not in STOPWORDS and len(t) > 1)
        
        if response_content:
            on_topic_count = 0
            for token in response_content:
                if token in query_content:
                    on_topic_count += 1
                    continue
                # Check fuzzy match to any query term
                for qt in query_content:
                    if edit_distance_ratio(token, qt) > 0.75:
                        on_topic_count += 0.8
                        break
            
            on_topic_ratio = on_topic_count / len(response_content)
        else:
            on_topic_ratio = 0.0
        
        # --- Score 3: Response substance and completeness ---
        response_len = len(response_tokens)
        
        # Very short responses are suspicious but not always bad
        # Very long responses with repetition are bad
        
        # Length adequacy score
        if response_len < 3:
            length_score = 0.2
        elif response_len < 8:
            length_score = 0.5
        elif response_len < 15:
            length_score = 0.7
        elif response_len < 200:
            length_score = 1.0
        else:
            length_score = 0.9  # Slight penalty for very long
        
        # --- Score 4: Repetition detection ---
        if len(response_sentences) > 1:
            unique_sents = set()
            for s in response_sentences:
                # Normalize
                normalized = re.sub(r'\s+', ' ', s.lower().strip())
                unique_sents.add(normalized)
            repetition_ratio = len(unique_sents) / len(response_sentences)
        else:
            repetition_ratio = 1.0
        
        # Also check token-level repetition
        if response_content:
            unique_content_ratio = len(set(response_content)) / len(response_content)
        else:
            unique_content_ratio = 0.0
        
        repetition_score = (repetition_ratio * 0.6 + unique_content_ratio * 0.4)
        
        # --- Score 5: Drift detection ---
        # Check if response starts on-topic but drifts
        # Analyze first vs second half
        if len(response_sentences) >= 4:
            mid = len(response_sentences) // 2
            first_half = ' '.join(response_sentences[:mid])
            second_half = ' '.join(response_sentences[mid:])
            
            first_tokens = set(tokenize(first_half)) - STOPWORDS
            second_tokens = set(tokenize(second_half)) - STOPWORDS
            
            first_relevance = len(first_tokens & query_content) / max(len(first_tokens), 1)
            second_relevance = len(second_tokens & query_content) / max(len(second_tokens), 1)
            
            # If second half is much less relevant, penalize
            if first_relevance > 0 and second_relevance < first_relevance * 0.3:
                drift_penalty = 0.7
            else:
                drift_penalty = 1.0
        else:
            drift_penalty = 1.0
        
        # --- Score 6: Response contains actual content (not just code/HTML noise) ---
        # Detect if response is mostly code or markup when query doesn't ask for it
        code_indicators = len(re.findall(r'[{}<>=/;]', response))
        code_ratio = code_indicators / max(len(response), 1)
        
        query_asks_code = any(w in query.lower() for w in ['code', 'html', 'python', 'program', 'script', 'tag', 'css', 'javascript'])
        
        if code_ratio > 0.1 and not query_asks_code:
            code_penalty = max(0.3, 1.0 - code_ratio * 3)
        else:
            code_penalty = 1.0
        
        # --- Score 7: Direct address detection ---
        # Check if response seems to directly answer vs deflect
        deflection_phrases = [
            'you can tell us', 'let us know', 'comment below',
            'i don\'t know', 'no idea', 'not sure'
        ]
        response_lower = response.lower()
        deflection_penalty = 1.0
        for phrase in deflection_phrases:
            if phrase in response_lower:
                deflection_penalty = 0.5
                break
        
        # Check for extremely minimal responses (single word/period)
        stripped = response.strip().strip('.')
        if len(stripped) < 3:
            return 0.5
        
        # --- Score 8: Sentence-level topic alignment ---
        # For each sentence, compute how many query topics it touches
        sentence_topic_scores = []
        for sent in response_sentences[:10]:  # Cap at 10 sentences
            sent_tokens = set(tokenize(sent))
            sent_content = sent_tokens - STOPWORDS
            if not sent_content:
                sentence_topic_scores.append(0.0)
                continue
            
            hits = 0
            for term in query_topics:
                if term in sent_content:
                    hits += 1
                elif any(edit_distance_ratio(term, st) > 0.8 for st in sent_content if abs(len(st) - len(term)) <= 2):
                    hits += 0.7
            
            score = hits / max(len(query_topics), 1)
            sentence_topic_scores.append(min(score, 1.0))
        
        if sentence_topic_scores:
            # Weight first sentences more heavily
            weighted_sent_score = 0.0
            total_weight = 0.0
            for i, s in enumerate(sentence_topic_scores):
                w = 1.0 / (1.0 + i * 0.3)  # Decaying weight
                weighted_sent_score += s * w
                total_weight += w
            avg_sentence_topic = weighted_sent_score / total_weight
        else:
            avg_sentence_topic = 0.0
        
        # --- Combine all scores ---
        # Topic coverage is most important
        # Then sentence-level alignment, then coherence
        
        raw_score = (
            topic_coverage_score * 0.35 +          # 0-10 range
            avg_sentence_topic * 10.0 * 0.20 +     # 0-10 range
            on_topic_ratio * 10.0 * 0.10 +          # 0-10 range
            length_score * 10.0 * 0.10 +             # 0-10 range
            repetition_score * 10.0 * 0.10 +         # 0-10 range
            5.0 * 0.15                                # Base score for attempting a response
        )
        
        # Apply penalties
        raw_score *= drift_penalty
        raw_score *= code_penalty
        raw_score *= deflection_penalty
        
        # Boost for responses that are substantive and on-topic
        if topic_coverage_score > 6 and length_score > 0.7 and repetition_score > 0.7:
            raw_score *= 1.1
        
        # Penalize heavily if almost no topic coverage
        if topic_coverage_score < 2.0:
            raw_score *= 0.6
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0