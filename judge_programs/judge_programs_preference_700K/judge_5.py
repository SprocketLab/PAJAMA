def judging_function(query, response):
    """
    Evaluates response relevance using a semantic coverage approach based on:
    - Query intent decomposition (extracting key question components)
    - Response coverage of query entities and concepts
    - Information density and directness metrics
    - Penalization for hedging, deflection, and off-topic signals
    
    This variant focuses on structural/pragmatic analysis rather than
    simple word overlap, TF-IDF, cosine similarity, or n-gram approaches.
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
        
        # ---- Helper functions ----
        def tokenize(text):
            return re.findall(r'[a-zA-Z0-9]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 3]
        
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'that', 'this', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'these', 'those', 'am', 'im', 'also', 'like', 'get', 'got', 'much',
            'many', 'any', 'dont', 'ive', 'thats', 'well', 'really', 'think',
            'know', 'see', 'way', 'make', 'one', 'two', 'even', 'still'
        }
        
        def content_words(tokens):
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        query_tokens = tokenize(query)
        resp_tokens = tokenize(response)
        
        query_content = content_words(query_tokens)
        resp_content = content_words(resp_tokens)
        
        if not query_content or not resp_content:
            return 1.0
        
        # ============================================================
        # 1. QUERY CONCEPT COVERAGE (weighted by rarity in query)
        # ============================================================
        # Extract key concepts from query - words that appear meaningful
        # Weight rarer/longer words more heavily as they carry more semantic load
        
        query_content_counter = Counter(query_content)
        resp_content_set = set(resp_content)
        
        # Weight each query concept by its "importance" (length as proxy for specificity)
        total_weight = 0.0
        covered_weight = 0.0
        
        for word, count in query_content_counter.items():
            # Longer words and less common English words are more topically important
            word_weight = math.log2(max(len(word), 2)) * count
            total_weight += word_weight
            if word in resp_content_set:
                covered_weight += word_weight
        
        concept_coverage = covered_weight / total_weight if total_weight > 0 else 0.0
        
        # ============================================================
        # 2. QUERY ENTITY/PROPER NOUN COVERAGE
        # ============================================================
        # Extract capitalized words and special terms from original query
        entities_in_query = set()
        cap_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query)
        for w in cap_words:
            wl = w.lower()
            if wl not in STOPWORDS:
                entities_in_query.add(wl)
        
        # Also extract quoted terms, technical terms (with special chars)
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        for q in quoted:
            for t in tokenize(q):
                if t not in STOPWORDS and len(t) > 2:
                    entities_in_query.add(t)
        
        # Check coverage of entities in response
        if entities_in_query:
            entity_hits = sum(1 for e in entities_in_query if e in resp_content_set)
            entity_coverage = entity_hits / len(entities_in_query)
        else:
            entity_coverage = concept_coverage  # fallback
        
        # ============================================================
        # 3. QUESTION TYPE DETECTION AND ANSWER PATTERN MATCHING
        # ============================================================
        query_lower = query.lower()
        
        # Detect question type
        is_how = bool(re.search(r'\bhow\b', query_lower))
        is_what = bool(re.search(r'\bwhat\b', query_lower))
        is_why = bool(re.search(r'\bwhy\b', query_lower))
        is_when = bool(re.search(r'\bwhen\b', query_lower))
        is_where = bool(re.search(r'\bwhere\b', query_lower))
        is_who = bool(re.search(r'\bwho\b', query_lower))
        is_yes_no = bool(re.search(r'^(is|are|do|does|did|can|could|would|should|has|have|will)\b', query_lower))
        is_imperative = bool(re.search(r'^(imagine|create|write|generate|explain|describe|list|tell|give|show|make|build|design)', query_lower))
        
        resp_lower = response.lower()
        resp_sentences = get_sentences(response)
        
        # Check if response has answer-like patterns for the question type
        answer_pattern_score = 0.5  # neutral default
        
        if is_why:
            # Look for causal language
            causal = re.findall(r'\b(because|reason|due to|caused|result|since|therefore|thus|hence|leads? to|stems? from)\b', resp_lower)
            answer_pattern_score = min(1.0, 0.3 + len(causal) * 0.15)
        
        elif is_how:
            # Look for procedural/explanatory language
            proc = re.findall(r'\b(step|first|then|next|by|through|using|method|way|process|essentially|basically)\b', resp_lower)
            answer_pattern_score = min(1.0, 0.3 + len(proc) * 0.1)
        
        elif is_what:
            # Look for definitional/descriptive language
            defn = re.findall(r'\b(is|means|refers|defined|called|known|type|kind|form|involves)\b', resp_lower)
            answer_pattern_score = min(1.0, 0.3 + len(defn) * 0.08)
        
        elif is_yes_no:
            # Look for affirmative/negative stance
            stance = re.findall(r'\b(yes|no|absolutely|definitely|certainly|indeed|not really|actually)\b', resp_lower)
            answer_pattern_score = min(1.0, 0.3 + len(stance) * 0.2)
        
        elif is_imperative:
            # Check if response actually attempts the task
            if len(resp_tokens) > 15:
                answer_pattern_score = 0.7
            if len(resp_tokens) > 40:
                answer_pattern_score = 0.85
        
        # ============================================================
        # 4. RESPONSE DIRECTNESS - does it address query early?
        # ============================================================
        # Check if first 1-2 sentences contain query concepts
        if resp_sentences:
            first_portion = ' '.join(resp_sentences[:min(2, len(resp_sentences))])
            first_tokens = set(content_words(tokenize(first_portion)))
            query_content_set = set(query_content)
            
            early_overlap = len(first_tokens & query_content_set)
            directness = min(1.0, early_overlap / max(1, min(len(query_content_set), 5)))
        else:
            directness = 0.2
        
        # ============================================================
        # 5. TOPICAL COHERENCE - response stays on topic throughout
        # ============================================================
        # Split response into chunks and check each chunk's relevance
        query_content_set = set(query_content)
        
        if len(resp_sentences) >= 3:
            chunk_size = max(1, len(resp_sentences) // 3)
            chunks = []
            for i in range(0, len(resp_sentences), chunk_size):
                chunks.append(' '.join(resp_sentences[i:i+chunk_size]))
            
            chunk_relevances = []
            for chunk in chunks[:4]:  # max 4 chunks
                chunk_tokens = set(content_words(tokenize(chunk)))
                if chunk_tokens:
                    overlap = len(chunk_tokens & query_content_set)
                    rel = overlap / max(1, min(len(query_content_set), 8))
                    chunk_relevances.append(min(1.0, rel))
                else:
                    chunk_relevances.append(0.0)
            
            if chunk_relevances:
                coherence = sum(chunk_relevances) / len(chunk_relevances)
                # Penalize if later chunks drift off topic
                if len(chunk_relevances) >= 2:
                    drift_penalty = max(0, chunk_relevances[0] - chunk_relevances[-1]) * 0.2
                    coherence -= drift_penalty
            else:
                coherence = 0.3
        else:
            coherence = concept_coverage  # fallback for short responses
        
        # ============================================================
        # 6. DEFLECTION / META-RESPONSE DETECTION
        # ============================================================
        deflection_patterns = [
            r'\bwelcome to\b',
            r'\bplease read\b',
            r'\brules before\b',
            r'\bremoved if\b',
            r'\bi don\'?t (know|understand)\b',
            r'\bi\'?m not sure\b',
            r'\bi can\'?t (help|answer|assist)\b',
            r'\byou (should|might|could) (ask|try|look|google|search)\b',
            r'\bwhile you wait\b',
            r'\byou might be interested in\b',
            r'\banswer.*by u/',
        ]
        
        deflection_count = 0
        for pattern in deflection_patterns:
            if re.search(pattern, resp_lower):
                deflection_count += 1
        
        deflection_penalty = min(0.5, deflection_count * 0.2)
        
        # ============================================================
        # 7. SUBSTANTIVENESS - does the response provide real content?
        # ============================================================
        # Measure information density: ratio of content words to total words
        if resp_tokens:
            info_density = len(resp_content) / len(resp_tokens)
        else:
            info_density = 0.0
        
        # Check for specific/concrete language (numbers, examples, names)
        specifics = len(re.findall(r'\b\d+\b', response))
        specifics += len(re.findall(r'\b[A-Z][a-z]+\b', response)) * 0.3
        specifics += len(re.findall(r'\b(example|instance|case|specifically|particular|e\.g\.|i\.e\.)\b', resp_lower))
        
        specificity_score = min(1.0, specifics / max(1, len(resp_sentences) * 2))
        
        # Length appropriateness - not too short, diminishing returns for very long
        word_count = len(resp_tokens)
        if word_count < 10:
            length_score = 0.2
        elif word_count < 25:
            length_score = 0.4
        elif word_count < 50:
            length_score = 0.65
        elif word_count < 100:
            length_score = 0.8
        elif word_count < 200:
            length_score = 0.9
        else:
            length_score = 0.95
        
        # ============================================================
        # 8. SEMANTIC FIELD MATCHING via word co-occurrence proxy
        # ============================================================
        # Build "semantic neighborhoods" by looking at words that tend to
        # co-occur with query terms within response sentences
        
        query_bigrams = set()
        for i in range(len(query_content) - 1):
            query_bigrams.add((query_content[i], query_content[i+1]))
        
        resp_content_list = resp_content
        resp_bigrams = set()
        for i in range(len(resp_content_list) - 1):
            resp_bigrams.add((resp_content_list[i], resp_content_list[i+1]))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & resp_bigrams)
            semantic_field_score = min(1.0, bigram_overlap / max(1, len(query_bigrams)) * 2)
        else:
            semantic_field_score = concept_coverage
        
        # ============================================================
        # 9. RESPONSE ENGAGEMENT - does it engage with the query's framing?
        # ============================================================
        # Check if response acknowledges/references specific aspects of the query
        
        # Extract phrases of 3+ words from query (sliding window on content words)
        query_phrases = set()
        for i in range(len(query_content) - 2):
            phrase = tuple(query_content[i:i+3])
            query_phrases.add(phrase)
        
        resp_phrases = set()
        for i in range(len(resp_content_list) - 2):
            phrase = tuple(resp_content_list[i:i+3])
            resp_phrases.add(phrase)
        
        if query_phrases:
            phrase_hits = len(query_phrases & resp_phrases)
            engagement_score = min(1.0, phrase_hits / max(1, len(query_phrases)) * 3)
        else:
            engagement_score = concept_coverage
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        # Weighted combination emphasizing different aspects
        
        raw_score = (
            concept_coverage * 2.5 +      # Core topic coverage
            entity_coverage * 2.0 +        # Named entity coverage  
            answer_pattern_score * 1.5 +   # Appropriate answer structure
            directness * 1.5 +             # Gets to the point
            coherence * 1.0 +              # Stays on topic
            info_density * 0.8 +           # Information richness
            specificity_score * 0.8 +      # Concrete details
            length_score * 0.7 +           # Appropriate length
            semantic_field_score * 1.2 +   # Semantic field alignment
            engagement_score * 1.0         # Engages with query framing
        )
        
        max_possible = 2.5 + 2.0 + 1.5 + 1.5 + 1.0 + 0.8 + 0.8 + 0.7 + 1.2 + 1.0  # = 13.0
        
        normalized = (raw_score / max_possible) * 10.0
        
        # Apply deflection penalty
        normalized -= deflection_penalty * 10.0
        
        # Bonus for responses that show deep engagement (multiple signals align)
        high_signals = sum(1 for s in [concept_coverage, entity_coverage, directness, coherence] if s > 0.5)
        if high_signals >= 3:
            normalized += 0.5
        
        # Ensure bounds
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Ultra-fallback: simple length-based score
            return min(5.0, len(str(response).split()) / 20.0)
        except Exception:
            return 2.0