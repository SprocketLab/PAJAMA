def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question decomposition and topic coverage analysis
    - Information density (unique concepts per unit length)
    - Structural diversity (different explanation strategies)
    - Specificity scoring (named entities, numbers, examples)
    - Response-to-query proportionality
    - Discourse connective analysis for logical flow/depth
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not query:
            return 0.0

        query = str(query)
        response = str(response)

        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.5

        # =============================================
        # 1. QUERY DECOMPOSITION & TOPIC COVERAGE
        # =============================================
        # Extract question words and key topics from query
        query_lower = query.lower()
        response_lower = response.lower()

        # Count how many question aspects the query has
        question_markers = re.findall(r'\b(what|how|why|when|where|who|which|does|did|is|are|can|could|should|would|has|have)\b', query_lower)
        num_question_aspects = max(len(set(question_markers)), 1)

        # Extract meaningful content words from query (4+ chars, not stopwords)
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                     'their', 'would', 'could', 'should', 'about', 'which', 'there',
                     'what', 'when', 'where', 'your', 'more', 'some', 'than', 'them',
                     'other', 'into', 'just', 'also', 'most', 'very', 'much', 'like',
                     'does', 'will', 'each', 'make', 'over', 'such', 'even', 'after',
                     'know', 'because', 'good', 'think', 'well', 'back', 'only', 'come',
                     'made', 'find', 'here', 'thing', 'many', 'then', 'those', 'being',
                     'same', 'want', 'give', 'take', 'long', 'look', 'need', 'still'}

        def extract_content_words(text):
            words = re.findall(r'[a-z]{3,}', text.lower())
            return [w for w in words if w not in stopwords and len(w) >= 4]

        query_content_words = extract_content_words(query)
        response_content_words = extract_content_words(response)

        # Topic coverage: what fraction of query content words appear in response
        query_word_set = set(query_content_words)
        response_word_set = set(response_content_words)

        if query_word_set:
            topic_coverage = len(query_word_set & response_word_set) / len(query_word_set)
        else:
            topic_coverage = 0.5

        # =============================================
        # 2. INFORMATION DENSITY & CONCEPT RICHNESS
        # =============================================
        response_words = response.split()
        response_len = len(response_words)

        # Unique content words as measure of concept breadth
        unique_content = set(response_content_words)
        num_unique_content = len(unique_content)

        # Information density: unique concepts normalized by length
        # Use log to prevent very long responses from dominating
        if response_len > 0:
            info_density = num_unique_content / (math.log2(response_len + 1) * 5)
        else:
            info_density = 0

        # Cap at reasonable level
        info_density = min(info_density, 5.0)

        # =============================================
        # 3. SPECIFICITY SCORING
        # =============================================
        # Named entities (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]+', response)
        named_entities = 0
        for sent in sentences:
            words_in_sent = sent.strip().split()
            if len(words_in_sent) > 1:
                for w in words_in_sent[1:]:
                    if w and w[0].isupper() and len(w) > 1 and w.isalpha():
                        named_entities += 1

        # Numbers and quantitative info
        numbers = re.findall(r'\b\d+[\d,.]*\b', response)
        num_count = len(numbers)

        # Quoted terms, technical terms (words with special chars or very specific)
        quoted_terms = len(re.findall(r'["\*\'`][\w\s]+["\*\'`]', response))

        # Example indicators
        example_phrases = re.findall(r'\b(for example|for instance|such as|e\.g\.|i\.e\.|specifically|in particular|consider|imagine|suppose|like when|case of)\b', response_lower)
        num_examples = len(example_phrases)

        specificity_score = (
            min(named_entities, 10) * 0.3 +
            min(num_count, 8) * 0.4 +
            min(quoted_terms, 5) * 0.3 +
            min(num_examples, 5) * 0.5
        )
        specificity_score = min(specificity_score, 8.0)

        # =============================================
        # 4. DISCOURSE & REASONING DEPTH
        # =============================================
        # Causal/explanatory connectives indicate deeper coverage
        causal_connectives = re.findall(
            r'\b(because|therefore|thus|hence|consequently|as a result|due to|since|'
            r'so that|in order to|leads to|causes|means that|implies|suggests)\b',
            response_lower
        )

        # Contrastive connectives indicate nuanced coverage
        contrast_connectives = re.findall(
            r'\b(however|although|but|whereas|while|on the other hand|nevertheless|'
            r'despite|yet|instead|rather|conversely|alternatively|though)\b',
            response_lower
        )

        # Additive/elaborative connectives indicate breadth
        additive_connectives = re.findall(
            r'\b(additionally|furthermore|moreover|also|in addition|another|'
            r'besides|not only|as well|plus|equally|similarly)\b',
            response_lower
        )

        # Conditional/hypothetical reasoning
        conditional_markers = re.findall(
            r'\b(if|unless|assuming|provided|depending|in case|whether|suppose)\b',
            response_lower
        )

        discourse_score = (
            min(len(causal_connectives), 6) * 0.6 +
            min(len(contrast_connectives), 5) * 0.7 +
            min(len(additive_connectives), 5) * 0.5 +
            min(len(conditional_markers), 4) * 0.4
        )
        discourse_score = min(discourse_score, 8.0)

        # =============================================
        # 5. STRUCTURAL DIVERSITY (explanation strategies)
        # =============================================
        strategies_used = 0

        # Check for definitions
        if re.search(r'\b(is defined as|means|refers to|is a|is the)\b', response_lower):
            strategies_used += 1

        # Check for comparisons
        if re.search(r'\b(compared to|unlike|similar to|difference between|versus|vs)\b', response_lower):
            strategies_used += 1

        # Check for temporal/process description
        if re.search(r'\b(first|then|next|after|before|finally|step|process|begin|start)\b', response_lower):
            strategies_used += 1

        # Check for evidence/authority citation
        if re.search(r'\b(according to|research|study|evidence|data|source|found that|shows that)\b', response_lower):
            strategies_used += 1

        # Check for personal experience/anecdote
        if re.search(r'\b(i have|i\'ve|in my experience|personally|i found|i think|i would)\b', response_lower):
            strategies_used += 1

        # Check for code blocks
        if '```' in response or re.search(r'^\s{4,}\S', response, re.MULTILINE):
            strategies_used += 1

        # Check for qualification/nuance
        if re.search(r'\b(it depends|nuance|caveat|exception|note that|keep in mind|important to|worth noting)\b', response_lower):
            strategies_used += 1

        structural_diversity = min(strategies_used, 5) * 1.2

        # =============================================
        # 6. RESPONSE LENGTH PROPORTIONALITY
        # =============================================
        # Longer queries with more sub-questions deserve longer responses
        query_len = len(query.split())

        # Base expected length scales with query complexity
        expected_min_len = max(30, query_len * 1.5)

        if response_len < expected_min_len * 0.3:
            length_penalty = 0.4
        elif response_len < expected_min_len * 0.6:
            length_penalty = 0.7
        elif response_len < expected_min_len:
            length_penalty = 0.85
        else:
            length_penalty = 1.0

        # Bonus for substantial responses (diminishing returns)
        length_bonus = min(math.log2(max(response_len, 1) + 1) / 3.0, 3.0)

        # =============================================
        # 7. SENTENCE-LEVEL COMPLETENESS
        # =============================================
        # Check if response ends mid-sentence (truncation)
        truncation_penalty = 0
        if response_stripped and response_stripped[-1] not in '.!?"\')]}':
            # Likely truncated
            truncation_penalty = -1.5
            # Partial credit if it's still long
            if response_len > 100:
                truncation_penalty = -0.5

        # Count complete sentences
        complete_sentences = len([s for s in sentences if len(s.strip().split()) >= 3])

        # Multi-sentence responses cover more ground
        sentence_coverage = min(complete_sentences / max(num_question_aspects, 1), 5.0)

        # =============================================
        # 8. N-GRAM DIVERSITY (avoid repetitive responses)
        # =============================================
        if response_len >= 6:
            trigrams = [' '.join(response_words[i:i+3]).lower() for i in range(len(response_words)-2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                unique_ratio = len(trigram_counts) / len(trigrams)
                repetition_penalty = max(0, (1.0 - unique_ratio) * -3.0)
            else:
                repetition_penalty = 0
        else:
            repetition_penalty = 0

        # =============================================
        # 9. QUERY-SPECIFIC KEYWORD CLUSTERS
        # =============================================
        # Extract 2-word phrases from query and check coverage
        query_words_list = re.findall(r'[a-z]+', query_lower)
        query_bigrams = set()
        for i in range(len(query_words_list) - 1):
            if query_words_list[i] not in stopwords or query_words_list[i+1] not in stopwords:
                query_bigrams.add((query_words_list[i], query_words_list[i+1]))

        response_words_list = re.findall(r'[a-z]+', response_lower)
        response_bigrams = set()
        for i in range(len(response_words_list) - 1):
            response_bigrams.add((response_words_list[i], response_words_list[i+1]))

        if query_bigrams:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.3

        # =============================================
        # FINAL SCORE COMPOSITION
        # =============================================
        score = (
            topic_coverage * 12.0 +          # 0-12: query topic coverage
            info_density * 3.5 +              # 0-17.5: concept richness
            specificity_score * 2.0 +         # 0-16: concrete details
            discourse_score * 2.5 +           # 0-20: reasoning depth
            structural_diversity * 2.0 +      # 0-12: explanation strategies
            length_bonus * 3.0 +              # 0-9: substantial response
            sentence_coverage * 2.0 +         # 0-10: multi-aspect coverage
            bigram_coverage * 5.0 +           # 0-5: phrase-level relevance
            truncation_penalty +              # -1.5 to 0
            repetition_penalty                # -3 to 0
        ) * length_penalty

        # Normalize to 0-10 range
        # Theoretical max ~100, practical high ~60-70
        score = max(0.0, score)
        score = (score / 55.0) * 10.0
        score = min(10.0, max(0.0, score))

        return round(score, 3)

    except Exception:
        try:
            # Fallback: simple length-based score
            return min(len(str(response).split()) / 30.0, 5.0)
        except Exception:
            return 2.0