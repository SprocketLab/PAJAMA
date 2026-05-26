def judging_function(query, response):
    """
    Evaluates completeness and coverage using a topic-modeling / semantic coverage approach.
    
    Algorithm: Extract "information needs" from the query (question words, key entities,
    sub-questions), then measure how thoroughly the response addresses each need through
    a combination of:
    1. Query decomposition into information facets
    2. Response information density (unique concepts per sentence)
    3. Coherence and on-topic ratio
    4. Depth signals (explanations, reasoning chains, examples)
    5. Penalization for repetition, off-topic content, and truncation
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        if not query or not query.strip():
            return 5.0
        
        query_clean = query.strip()
        response_clean = response.strip()
        
        # ---- Stopwords ----
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'also', 'am', 'an', 'any', 'de', 'en', 'la', 'le'
        }
        
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def content_words(tokens):
            return [t for t in tokens if t not in stopwords and len(t) > 1]
        
        query_tokens = tokenize(query_clean)
        response_tokens = tokenize(response_clean)
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        if len(response_tokens) < 1:
            return 0.5
        
        # ========================================
        # 1. QUERY FACET EXTRACTION & COVERAGE
        # ========================================
        # Extract distinct "facets" from the query - these represent information needs
        
        # Split query into sub-questions (by ?, newline, or semicolons)
        sub_questions = re.split(r'[?\n;]+', query_clean)
        sub_questions = [sq.strip() for sq in sub_questions if len(sq.strip()) > 3]
        num_sub_questions = max(len(sub_questions), 1)
        
        # Extract key query concepts (content words, bigrams)
        query_content_set = set(query_content)
        
        # Build query bigrams
        query_bigrams = set()
        for i in range(len(query_content) - 1):
            query_bigrams.add((query_content[i], query_content[i+1]))
        
        # Measure how many query content words appear in response
        response_content_set = set(response_content)
        if query_content_set:
            word_coverage = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            word_coverage = 0.5
        
        # Measure query bigram coverage in response
        response_content_list = response_content
        response_bigrams = set()
        for i in range(len(response_content_list) - 1):
            response_bigrams.add((response_content_list[i], response_content_list[i+1]))
        
        if query_bigrams:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        # Sub-question coverage: for each sub-question, check if response has relevant content
        sub_q_covered = 0
        for sq in sub_questions:
            sq_content = content_words(tokenize(sq))
            if not sq_content:
                sub_q_covered += 1
                continue
            sq_set = set(sq_content)
            overlap = len(sq_set & response_content_set)
            if overlap >= max(1, len(sq_set) * 0.3):
                sub_q_covered += 1
        
        sub_q_coverage = sub_q_covered / num_sub_questions if num_sub_questions > 0 else 0.5
        
        facet_score = 0.4 * word_coverage + 0.2 * bigram_coverage + 0.4 * sub_q_coverage
        
        # ========================================
        # 2. INFORMATION DENSITY & RICHNESS
        # ========================================
        # Split response into sentences
        sentences = re.split(r'[.!?\n]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Unique concepts per sentence (information density)
        all_response_concepts = set()
        sentence_concept_counts = []
        for sent in sentences:
            sent_content = content_words(tokenize(sent))
            unique_in_sent = set(sent_content)
            new_concepts = unique_in_sent - all_response_concepts
            sentence_concept_counts.append(len(new_concepts))
            all_response_concepts.update(unique_in_sent)
        
        # Average new information per sentence
        avg_new_info = sum(sentence_concept_counts) / num_sentences if num_sentences > 0 else 0
        # Normalize: ~5 new concepts per sentence is good
        info_density_score = min(avg_new_info / 5.0, 1.0)
        
        # Total unique concepts (vocabulary richness)
        total_unique_concepts = len(all_response_concepts)
        # Type-token ratio for content words
        if len(response_content) > 0:
            ttr = total_unique_concepts / len(response_content)
        else:
            ttr = 0
        
        richness_score = min(total_unique_concepts / 30.0, 1.0) * 0.6 + min(ttr / 0.7, 1.0) * 0.4
        
        # ========================================
        # 3. DEPTH SIGNALS
        # ========================================
        depth_indicators = 0
        
        # Causal/explanatory connectors
        explanation_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
            r'\bas a result\b', r'\bdue to\b', r'\bsince\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bincluding\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bfirst\b', r'\bsecond\b', r'\bthird\b',
            r'\bfinally\b', r'\bin conclusion\b', r'\bto summarize\b',
            r'\bnamely\b', r'\bthat is\b', r'\bin other words\b'
        ]
        
        resp_lower = response_clean.lower()
        for pattern in explanation_patterns:
            if re.search(pattern, resp_lower):
                depth_indicators += 1
        
        # Presence of numbers/data
        if re.search(r'\d+', response_clean):
            depth_indicators += 1
        
        # Presence of proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[.!?]\s)[a-z].*?\b([A-Z][a-z]+)\b', response_clean)
        mid_caps = re.findall(r'(?<=\s)([A-Z][a-z]{2,})', response_clean)
        if len(mid_caps) > 2:
            depth_indicators += 1
        
        # Normalize depth score
        depth_score = min(depth_indicators / 8.0, 1.0)
        
        # ========================================
        # 4. RESPONSE LENGTH ADEQUACY
        # ========================================
        resp_word_count = len(response_tokens)
        query_word_count = len(query_tokens)
        
        # Detect if query asks for something simple vs complex
        is_simple_query = query_word_count < 12 and num_sub_questions <= 1
        
        # For simple queries, shorter responses can be fine
        # For complex queries, we expect more
        if is_simple_query:
            ideal_min_words = 8
            ideal_max_words = 200
        else:
            ideal_min_words = 20
            ideal_max_words = 500
        
        if resp_word_count < 3:
            length_score = 0.05
        elif resp_word_count < ideal_min_words:
            length_score = 0.2 + 0.3 * (resp_word_count / ideal_min_words)
        elif resp_word_count <= ideal_max_words:
            length_score = 0.7 + 0.3 * min(resp_word_count / max(ideal_min_words * 3, 30), 1.0)
        else:
            # Very long - could be good or could be rambling
            length_score = 0.85
        
        # ========================================
        # 5. PENALTIES
        # ========================================
        penalty = 0.0
        
        # Repetition penalty: detect repeated phrases/sentences
        if num_sentences >= 2:
            seen_sents = set()
            repeated = 0
            for sent in sentences:
                normalized = ' '.join(tokenize(sent))
                if normalized in seen_sents:
                    repeated += 1
                seen_sents.add(normalized)
            repetition_ratio = repeated / num_sentences
            penalty += repetition_ratio * 0.3
        
        # Repeated n-grams penalty (3-grams)
        if len(response_tokens) >= 6:
            trigrams = []
            for i in range(len(response_tokens) - 2):
                trigrams.append(tuple(response_tokens[i:i+3]))
            trigram_counts = Counter(trigrams)
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                trigram_rep_ratio = repeated_trigrams / max(len(set(trigrams)), 1)
                penalty += min(trigram_rep_ratio * 0.5, 0.25)
        
        # Off-topic penalty: if response has lots of content unrelated to query
        # Measure via how much of response content is "new" vs query-related
        if response_content_set and query_content_set:
            on_topic_words = response_content_set & query_content_set
            # Also consider words semantically near query (same sentence as query words)
            # Simple proxy: words that co-occur with query words in same sentence
            extended_on_topic = set(on_topic_words)
            for sent in sentences:
                sent_tokens = content_words(tokenize(sent))
                sent_set = set(sent_tokens)
                if sent_set & query_content_set:
                    extended_on_topic.update(sent_set)
            
            on_topic_ratio = len(extended_on_topic) / max(len(response_content_set), 1)
            if on_topic_ratio < 0.3:
                penalty += 0.15
        
        # Truncation penalty
        if response_clean and response_clean[-1] not in '.!?"\')]}':
            # Check if it looks truncated (ends mid-sentence)
            last_chars = response_clean[-20:]
            if not re.search(r'[.!?]\s*$', last_chars):
                # Mild penalty for truncation
                penalty += 0.05
        
        # Gibberish/code penalty when query doesn't ask for code
        code_query = bool(re.search(r'\b(code|program|function|html|css|python|script|implement)\b', query_clean.lower()))
        if not code_query:
            code_lines = len(re.findall(r'(?:def |import |class |if __name__|print\(|return |var |function\()', response_clean))
            if code_lines > 2:
                penalty += 0.2
        
        # Very short response penalty for complex queries
        if not is_simple_query and resp_word_count < 5:
            penalty += 0.3
        
        # Single word/phrase response penalty
        if resp_word_count <= 2:
            penalty += 0.3
        
        # Penalty for responses that just echo the query
        if query_content:
            response_only = response_content_set - query_content_set
            if len(response_only) < 2 and resp_word_count < query_word_count * 1.5:
                penalty += 0.15
        
        penalty = min(penalty, 0.8)  # Cap total penalty
        
        # ========================================
        # 6. TASK-SPECIFIC BONUS
        # ========================================
        bonus = 0.0
        
        # If query asks "how many" or "what is" - check if response has a direct answer
        if re.search(r'\bhow many\b', query_clean.lower()):
            if re.search(r'\d+', response_clean):
                bonus += 0.05
        
        # If query asks for a list/multiple items
        list_request = bool(re.search(r'\b(list|identify|name|three|four|five|ways|types|examples|different)\b', query_clean.lower()))
        if list_request:
            # Count distinct items in response (by bullets, numbers, or comma-separated)
            bullet_items = len(re.findall(r'(?:^|\n)\s*(?:[-•*]|\d+[.)]\s)', response_clean))
            comma_items = len(re.findall(r',', response_clean)) + 1
            items_found = max(bullet_items, min(comma_items, 10))
            if items_found >= 3:
                bonus += 0.1
        
        # If query asks for rewriting/alternatives
        if re.search(r'\b(rewrite|rephrase|different ways?|alternatives?)\b', query_clean.lower()):
            # Count distinct outputs/versions
            output_markers = len(re.findall(r'(?:output|version|\d+[.):])', resp_lower))
            distinct_sentences_content = set()
            for sent in sentences:
                sc = frozenset(content_words(tokenize(sent)))
                if len(sc) > 2:
                    distinct_sentences_content.add(sc)
            if len(distinct_sentences_content) >= 3 or output_markers >= 2:
                bonus += 0.1
        
        bonus = min(bonus, 0.2)
        
        # ========================================
        # FINAL SCORE COMPOSITION
        # ========================================
        # Weighted combination
        raw_score = (
            0.30 * facet_score +
            0.20 * richness_score +
            0.15 * depth_score +
            0.20 * length_score +
            0.15 * info_density_score
        )
        
        # Apply penalty and bonus
        final = raw_score * (1.0 - penalty) + bonus
        
        # Scale to 0-10
        final_score = final * 10.0
        
        # Clamp
        final_score = max(0.5, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 3.0