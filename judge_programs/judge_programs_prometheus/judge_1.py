def judging_function(query, response):
    """
    Evaluates the relevance of an LLM response to a given query.
    Uses word overlap, topic alignment, semantic similarity heuristics,
    and structural quality indicators.
    
    Returns a score where HIGHER = BETTER quality (range roughly 0-10).
    """
    try:
        import re
        import math
        from collections import Counter
        
        # Handle edge cases
        if not query or not response:
            return 0.0
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        query = query.strip()
        response = response.strip()
        
        if len(query) == 0 or len(response) == 0:
            return 0.0
        
        # --- Preprocessing ---
        def tokenize(text):
            """Lowercase and extract alphanumeric tokens."""
            return re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text.lower())
        
        def get_ngrams(tokens, n):
            """Generate n-grams from token list."""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Stopwords - common English words to filter for content analysis
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
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
            'themselves', 'what', 'which', 'who', 'whom', 'whose', 'about', 'up',
            'down', 'also', 'like', 'get', 'got', 'make', 'made', 'go', 'going',
            'come', 'take', 'know', 'see', 'think', 'want', 'give', 'say', 'said',
            'tell', 'told', 'ask', 'asked', 'one', 'two', 'first', 'new', 'way',
            'well', 'also', 'back', 'much', 'even', 'still', 'right', 'now',
            'thing', 'things', 'let', 'put', 'keep', 'try', 'any', 'many',
            'however', 'really', 'already', 'yet', 'since', 'until', 'whether',
            'am', 'been', 'don', 't', 's', 've', 're', 'll', 'd', 'm',
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if len(query_tokens) == 0 or len(response_tokens) == 0:
            return 0.5
        
        # Content words (non-stopwords)
        query_content = [w for w in query_tokens if w not in stopwords and len(w) > 2]
        response_content = [w for w in response_tokens if w not in stopwords and len(w) > 2]
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        # --- Feature 1: Content Word Overlap (Jaccard-like) ---
        if len(query_content_set) > 0:
            # What fraction of query content words appear in response
            query_coverage = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            query_coverage = 0.5  # neutral if query has no content words
        
        # --- Feature 2: Weighted Term Overlap using IDF-like weighting ---
        # Longer, rarer words in the query are more important
        def word_importance(word):
            """Heuristic importance: longer words and less common words matter more."""
            base = min(len(word) / 4.0, 2.0)  # length factor, capped
            return base
        
        if query_content:
            weighted_overlap = 0.0
            total_weight = 0.0
            for word in set(query_content):
                w = word_importance(word)
                total_weight += w
                if word in response_content_set:
                    weighted_overlap += w
            weighted_coverage = weighted_overlap / total_weight if total_weight > 0 else 0.0
        else:
            weighted_coverage = 0.5
        
        # --- Feature 3: Bigram overlap for phrase-level relevance ---
        query_bigrams = set(get_ngrams(query_content, 2))
        response_bigrams = set(get_ngrams(response_content, 2))
        
        if len(query_bigrams) > 0:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        # --- Feature 4: Topic/Theme Alignment ---
        # Build topic clusters from the query and check response alignment
        # Use TF-based cosine similarity between query and response
        query_tf = Counter(query_content)
        response_tf = Counter(response_content)
        
        all_words = set(query_tf.keys()) | set(response_tf.keys())
        
        if len(all_words) > 0:
            dot_product = sum(query_tf.get(w, 0) * response_tf.get(w, 0) for w in all_words)
            mag_q = math.sqrt(sum(v**2 for v in query_tf.values()))
            mag_r = math.sqrt(sum(v**2 for v in response_tf.values()))
            
            if mag_q > 0 and mag_r > 0:
                cosine_sim = dot_product / (mag_q * mag_r)
            else:
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0
        
        # --- Feature 5: Response addresses the query's intent ---
        # Check for question-type words and whether response has answer-type patterns
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Detect query intent type
        is_how_query = any(w in query_lower for w in ['how to', 'how would', 'how can', 'how do', 'how should'])
        is_what_query = any(w in query_lower for w in ['what is', 'what are', 'what does', 'what would'])
        is_explain_query = any(w in query_lower for w in ['explain', 'describe', 'elaborate', 'understand', 'concept'])
        is_help_query = any(w in query_lower for w in ['help', 'assist', 'advice', 'comfort', 'support', 'seeking'])
        is_manage_query = any(w in query_lower for w in ['manage', 'handle', 'cope', 'deal with', 'address'])
        
        intent_score = 0.0
        intent_checks = 0
        
        if is_how_query or is_explain_query:
            intent_checks += 1
            # Response should have explanatory patterns
            explanatory_patterns = [
                r'\bfirst\b', r'\bstep\b', r'\bstart\b', r'\bbegin\b',
                r'\bthen\b', r'\bnext\b', r'\bfinally\b', r'\bprocess\b',
                r'\bmethod\b', r'\bapproach\b', r'\bimagine\b', r'\bthink of\b',
                r'\bbasically\b', r'\bessentially\b', r'\bmeans\b', r'\bworks\b',
                r'\b1\b', r'\b2\b', r'\b3\b',
            ]
            matches = sum(1 for p in explanatory_patterns if re.search(p, response_lower))
            intent_score += min(matches / 3.0, 1.0)
        
        if is_help_query or is_manage_query:
            intent_checks += 1
            # Response should show empathy and provide actionable advice
            empathy_patterns = [
                r'\bunderstand\b', r'\bfeel\b', r'\bsorry\b', r'\bhear\b',
                r'\bokay\b', r'\bnatural\b', r'\bnormal\b', r'\bvalid\b',
                r'\bcompletely\b', r'\bgenuinely\b', r'\btruly\b',
            ]
            advice_patterns = [
                r'\btry\b', r'\bconsider\b', r'\bremember\b', r'\bhelp\b',
                r'\bsuggest\b', r'\brecommend\b', r'\bcould\b', r'\bmight\b',
                r'\bshould\b', r'\bimportant\b', r'\bdon\'t hesitate\b',
            ]
            empathy_matches = sum(1 for p in empathy_patterns if re.search(p, response_lower))
            advice_matches = sum(1 for p in advice_patterns if re.search(p, response_lower))
            intent_score += min(empathy_matches / 2.0, 1.0) * 0.5 + min(advice_matches / 2.0, 1.0) * 0.5
        
        if intent_checks > 0:
            intent_alignment = intent_score / intent_checks
        else:
            intent_alignment = 0.5  # neutral
        
        # --- Feature 6: Response Depth and Substance ---
        # Longer, more detailed responses that stay on topic tend to be better
        response_length = len(response_tokens)
        
        # Penalize very short responses, reward moderate length
        if response_length < 10:
            length_score = 0.2
        elif response_length < 30:
            length_score = 0.5
        elif response_length < 60:
            length_score = 0.7
        elif response_length < 150:
            length_score = 0.9
        elif response_length < 300:
            length_score = 1.0
        else:
            length_score = 0.95  # very long might be slightly verbose
        
        # --- Feature 7: Structural Quality Indicators ---
        # Check for structured responses (lists, steps, paragraphs)
        has_numbered_list = bool(re.search(r'\b[1-9]\.\s', response))
        has_bullet_points = bool(re.search(r'[-•]\s', response))
        has_paragraphs = response.count('\n\n') >= 1
        has_colons = response.count(':') >= 1
        
        structure_score = 0.0
        structure_score += 0.3 if has_numbered_list else 0.0
        structure_score += 0.2 if has_bullet_points else 0.0
        structure_score += 0.3 if has_paragraphs else 0.0
        structure_score += 0.2 if has_colons else 0.0
        structure_score = min(structure_score, 1.0)
        
        # --- Feature 8: Negative indicators (off-topic, dismissive) ---
        dismissive_patterns = [
            r'\bjust\s+(?:do|get|buy|move)\b',
            r'\bget over it\b', r'\bget yourself together\b',
            r'\bnot\s+(?:able|capable)\b.*\bmodel\b',
            r'\bmight not\b.*\bable\b',
            r'\bprobably won\'t\b',
            r'\bcan\'t\b.*\brecognize\b',
        ]
        
        negative_count = sum(1 for p in dismissive_patterns if re.search(p, response_lower))
        negativity_penalty = min(negative_count * 0.15, 0.6)
        
        # Check if response contradicts what the query asks for
        # e.g., query asks to adapt but response says to continue same way
        contradiction_penalty = 0.0
        if 'adapt' in query_lower or 'mirror' in query_lower or 'match' in query_lower:
            if 'continue' in response_lower and ('same' in response_lower or 'before' in response_lower):
                if 'disregard' in response_lower or 'ignore' in response_lower or 'persist' in response_lower:
                    contradiction_penalty = 0.5
        
        # --- Feature 9: Engagement and Tone Appropriateness ---
        # Check if response acknowledges the user's situation
        acknowledgment_patterns = [
            r'\bi (?:can |)(?:see|hear|understand|sense)\b',
            r'\bit(?:\'s| is) (?:completely |totally |absolutely |perfectly )?(?:understandable|okay|normal|natural|fine)\b',
            r'\bi\'m (?:genuinely |truly |really |so )?sorry\b',
            r'\bthat\'s (?:completely |totally )?understandable\b',
        ]
        
        acknowledgment_score = 0.0
        for p in acknowledgment_patterns:
            if re.search(p, response_lower):
                acknowledgment_score += 0.25
        acknowledgment_score = min(acknowledgment_score, 1.0)
        
        # Only factor in acknowledgment if query seems to require emotional support
        emotional_query = any(w in query_lower for w in [
            'feeling', 'frustrated', 'stress', 'sad', 'lonely', 'heartbroken',
            'devastated', 'exhausted', 'comfort', 'emotional', 'breakup',
            'passed away', 'regret', 'tether', 'down'
        ])
        
        if not emotional_query:
            acknowledgment_score = 0.5  # neutral for non-emotional queries
        
        # --- Feature 10: Query-specific keyword density in response ---
        # How concentrated are query keywords in the response
        if query_content and response_content:
            query_keyword_count_in_response = sum(
                response_content.count(w) for w in set(query_content)
            )
            keyword_density = query_keyword_count_in_response / len(response_content)
            # Normalize: good density is around 0.05-0.15
            keyword_density_score = min(keyword_density / 0.10, 1.0)
        else:
            keyword_density_score = 0.0
        
        # --- Combine Features into Final Score ---
        # Weights tuned to prioritize relevance dimensions
        weights = {
            'query_coverage': 1.5,
            'weighted_coverage': 1.8,
            'bigram_overlap': 1.2,
            'cosine_sim': 2.0,
            'intent_alignment': 1.5,
            'length_score': 0.6,
            'structure_score': 0.5,
            'acknowledgment': 0.8,
            'keyword_density': 1.0,
        }
        
        raw_score = (
            weights['query_coverage'] * query_coverage +
            weights['weighted_coverage'] * weighted_coverage +
            weights['bigram_overlap'] * bigram_overlap +
            weights['cosine_sim'] * cosine_sim +
            weights['intent_alignment'] * intent_alignment +
            weights['length_score'] * length_score +
            weights['structure_score'] * structure_score +
            weights['acknowledgment'] * acknowledgment_score +
            weights['keyword_density'] * keyword_density_score
        )
        
        total_weight = sum(weights.values())
        normalized_score = raw_score / total_weight  # 0 to 1 range
        
        # Apply penalties
        normalized_score -= negativity_penalty
        normalized_score -= contradiction_penalty
        
        # Clamp to [0, 1]
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        # Scale to 1-5 range to match the examples
        final_score = 1.0 + normalized_score * 4.0
        
        # Round to 2 decimal places
        final_score = round(final_score, 2)
        
        return final_score
        
    except Exception as e:
        # Never crash - return a neutral score
        return 3.0