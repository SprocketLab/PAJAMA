def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a semantic coverage approach
    based on query decomposition, entity/concept matching, and response structure analysis.
    
    This variant focuses on:
    1. Query intent decomposition (question words, key entities, action verbs)
    2. Concept coverage scoring with proximity weighting
    3. Response engagement patterns (direct address vs tangential)
    4. Information density and specificity metrics
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
        
        # --- Helper functions ---
        def tokenize(text):
            return re.findall(r'[a-zA-Z0-9]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 3]
        
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'this',
            'that', 'these', 'those', 'am', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom', 'also', 'like', 'get', 'got',
            'much', 'many', 'really', 'think', 'know', 'make', 'even', 'well',
            'back', 'still', 'way', 'take', 'come', 'go', 'see', 'look', 'thing',
            'things', 'say', 'said', 'one', 'two', 'don', 't', 's', 've', 're',
            'll', 'd', 'm', 'im', 'dont', 'ive', 'been', 'going'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # --- 1. Query Concept Extraction ---
        # Extract meaningful concepts from query (non-stopword tokens)
        query_concepts = [t for t in query_tokens if t not in STOP_WORDS and len(t) > 1]
        
        # Extract multi-word phrases (bigrams) from query for concept matching
        query_bigrams = set()
        for i in range(len(query_tokens) - 1):
            if query_tokens[i] not in STOP_WORDS or query_tokens[i+1] not in STOP_WORDS:
                query_bigrams.add((query_tokens[i], query_tokens[i+1]))
        
        # Weight query concepts by position (earlier = more important for topic)
        # and by rarity (less common words are more topically important)
        concept_weights = {}
        total_query_len = len(query_concepts) if query_concepts else 1
        for idx, concept in enumerate(query_concepts):
            # Position weight: concepts earlier in query get slight boost
            pos_weight = 1.0 + 0.3 * (1.0 - idx / total_query_len)
            # Length weight: longer words tend to be more specific
            len_weight = 1.0 + 0.1 * min(len(concept) - 2, 5)
            # Capitalized words in original might be entities
            weight = pos_weight * len_weight
            if concept in concept_weights:
                concept_weights[concept] = max(concept_weights[concept], weight)
            else:
                concept_weights[concept] = weight
        
        # --- 2. Concept Coverage Score ---
        # How many query concepts appear in the response?
        response_token_set = set(response_tokens)
        response_token_counter = Counter(response_tokens)
        
        if not concept_weights:
            concept_coverage = 0.5
        else:
            covered_weight = 0.0
            total_weight = 0.0
            for concept, weight in concept_weights.items():
                total_weight += weight
                if concept in response_token_set:
                    # Boost if concept appears multiple times (shows focus)
                    freq_bonus = min(1.0 + 0.15 * (response_token_counter[concept] - 1), 1.5)
                    covered_weight += weight * freq_bonus
                else:
                    # Check for partial/stem matches
                    stem = concept[:max(4, len(concept) - 2)]
                    partial_match = any(t.startswith(stem) for t in response_token_set if len(t) > 3)
                    if partial_match:
                        covered_weight += weight * 0.5
            
            concept_coverage = covered_weight / total_weight if total_weight > 0 else 0.0
        
        # --- 3. Bigram concept coverage ---
        response_bigrams = set()
        for i in range(len(response_tokens) - 1):
            response_bigrams.add((response_tokens[i], response_tokens[i+1]))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams)
            bigram_coverage = bigram_overlap / len(query_bigrams)
        else:
            bigram_coverage = 0.0
        
        # --- 4. Query Intent Analysis ---
        # Detect what kind of answer is expected
        query_lower = query.lower()
        
        # Question type detection
        is_how = bool(re.search(r'\bhow\b', query_lower))
        is_what = bool(re.search(r'\bwhat\b', query_lower))
        is_why = bool(re.search(r'\bwhy\b', query_lower))
        is_when = bool(re.search(r'\bwhen\b', query_lower))
        is_where = bool(re.search(r'\bwhere\b', query_lower))
        is_who = bool(re.search(r'\bwho\b', query_lower))
        is_question = any([is_how, is_what, is_why, is_when, is_where, is_who]) or '?' in query
        is_imperative = bool(re.search(r'^(create|write|generate|imagine|build|design|make|list|explain|describe|tell|show|give|find|calculate)\b', query_lower))
        
        # --- 5. Response Directness Score ---
        # Does the response start by addressing the query directly?
        response_sentences = get_sentences(response)
        
        directness_score = 0.0
        if response_sentences:
            first_sent_tokens = set(tokenize(response_sentences[0]))
            first_sent_concepts = first_sent_tokens - STOP_WORDS
            query_concept_set = set(query_concepts)
            
            # How many query concepts appear in the first sentence?
            if query_concept_set:
                first_sent_overlap = len(first_sent_concepts & query_concept_set) / len(query_concept_set)
            else:
                first_sent_overlap = 0.0
            
            directness_score = first_sent_overlap
            
            # Check if response begins with engagement patterns
            resp_lower = response.lower().strip()
            engagement_patterns = [
                r'^(yes|no|both|essentially|actually|absolutely|definitely|certainly)',
                r'^(the\s+(answer|reason|key|issue|problem|solution|idea))',
                r'^(in\s+(terms|short|general|my\s+experience))',
                r'^(as\s+(a|an|you|someone))',
                r'^(if\s+you)',
                r'^(for\s+(example|instance|context))',
                r'^(being\s+an?)',
                r'^(i\s+(think|believe|would|have|am|was|worked|got|found))',
                r'^(a\s+lot)',
                r'^(me:\s*\*)',  # roleplay format
                r'^(sure|okay|here)',
            ]
            for pattern in engagement_patterns:
                if re.search(pattern, resp_lower):
                    directness_score += 0.15
                    break
        
        directness_score = min(directness_score, 1.0)
        
        # --- 6. Topic Coherence via Sentence-level Relevance ---
        # What fraction of response sentences contain query concepts?
        if response_sentences and query_concepts:
            query_concept_set = set(query_concepts)
            relevant_sentences = 0
            for sent in response_sentences:
                sent_tokens = set(tokenize(sent))
                sent_concepts = sent_tokens - STOP_WORDS
                if sent_concepts & query_concept_set:
                    relevant_sentences += 1
            
            sentence_relevance = relevant_sentences / len(response_sentences)
        else:
            sentence_relevance = 0.3
        
        # --- 7. Specificity and Information Density ---
        # Responses with more specific/unique content words tend to be more informative
        response_content = [t for t in response_tokens if t not in STOP_WORDS and len(t) > 2]
        
        if response_content:
            # Type-token ratio as proxy for information density
            unique_content = set(response_content)
            ttr = len(unique_content) / len(response_content) if response_content else 0
            # Normalize: very high TTR in short responses isn't necessarily good
            info_density = min(ttr, 0.9)
            
            # Average word length of content words (longer = more specific)
            avg_word_len = sum(len(w) for w in response_content) / len(response_content)
            specificity = min((avg_word_len - 3) / 5, 1.0)  # normalize around 3-8 char range
            specificity = max(specificity, 0.0)
        else:
            info_density = 0.0
            specificity = 0.0
        
        # --- 8. Response Length Appropriateness ---
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Generally, longer responses that stay on topic score better
        # But extremely short responses to complex queries are penalized
        if query_len > 30:  # complex query
            if resp_len < 20:
                length_score = 0.3
            elif resp_len < 50:
                length_score = 0.6
            else:
                length_score = min(0.7 + 0.3 * (resp_len / 200), 1.0)
        else:
            if resp_len < 10:
                length_score = 0.4
            else:
                length_score = min(0.6 + 0.4 * (resp_len / 150), 1.0)
        
        # --- 9. Substantive Content Detection ---
        # Detect if response is mostly meta-commentary rather than substantive
        meta_patterns = [
            r'welcome to',
            r'please read our rules',
            r'your (comment|post) (will be|has been)',
            r'this (sub|subreddit|thread)',
            r'removed if',
            r'while (we|you) (wait|do not)',
            r'you might be interested in this',
            r'see (this|the) (link|response|answer|thread)',
        ]
        
        meta_score = 0.0
        for pattern in meta_patterns:
            if re.search(pattern, response.lower()):
                meta_score += 0.25
        meta_score = min(meta_score, 1.0)
        meta_penalty = 1.0 - (meta_score * 0.6)
        
        # --- 10. Response introduces relevant new information ---
        # Good responses add information beyond just echoing the query
        response_unique_concepts = set(response_content) - set(query_concepts) - STOP_WORDS
        if response_content:
            new_info_ratio = len(response_unique_concepts) / len(set(response_content))
        else:
            new_info_ratio = 0.0
        
        # Balance: too much new info with no query overlap = off topic
        # Sweet spot is moderate new info with good query coverage
        if concept_coverage > 0.3:
            new_info_bonus = min(new_info_ratio * 0.3, 0.2)
        else:
            new_info_bonus = -0.1 * new_info_ratio  # penalty for off-topic new info
        
        # --- 11. Proximity-based relevance ---
        # Check if query concepts appear near each other in response (topical clustering)
        if len(query_concepts) >= 2:
            concept_positions = {}
            for idx, token in enumerate(response_tokens):
                if token in set(query_concepts):
                    if token not in concept_positions:
                        concept_positions[token] = []
                    concept_positions[token].append(idx)
            
            if len(concept_positions) >= 2:
                # Calculate average minimum distance between different query concepts in response
                concepts_list = list(concept_positions.keys())
                distances = []
                for i in range(len(concepts_list)):
                    for j in range(i + 1, len(concepts_list)):
                        min_dist = float('inf')
                        for p1 in concept_positions[concepts_list[i]]:
                            for p2 in concept_positions[concepts_list[j]]:
                                min_dist = min(min_dist, abs(p1 - p2))
                        if min_dist < float('inf'):
                            distances.append(min_dist)
                
                if distances:
                    avg_dist = sum(distances) / len(distances)
                    # Closer concepts = more topically focused
                    proximity_score = max(0, 1.0 - avg_dist / (len(response_tokens) * 0.5))
                else:
                    proximity_score = 0.0
            else:
                proximity_score = 0.2
        else:
            proximity_score = 0.5  # neutral for very simple queries
        
        # --- Combine all signals ---
        # Weighted combination
        score = (
            concept_coverage * 2.8 +       # Primary: do query concepts appear in response?
            bigram_coverage * 1.2 +          # Phrase-level matching
            directness_score * 1.5 +         # Does response address query directly?
            sentence_relevance * 1.5 +       # Are response sentences on-topic?
            info_density * 0.6 +             # Information richness
            specificity * 0.5 +              # Word specificity
            length_score * 0.8 +             # Appropriate length
            proximity_score * 0.8 +          # Topical clustering
            new_info_bonus                   # Bonus/penalty for new information
        )
        
        # Apply meta-commentary penalty
        score *= meta_penalty
        
        # Normalize to 0-10 range
        # Max theoretical raw score is roughly: 2.8 + 1.2 + 1.5 + 1.5 + 0.6 + 0.5 + 0.8 + 0.8 + 0.2 = 9.9
        score = max(0.0, min(score, 10.0))
        
        # Apply slight sigmoid-like transformation to spread scores
        # This helps discriminate between mediocre and good responses
        midpoint = 5.0
        steepness = 0.5
        normalized = 10.0 / (1.0 + math.exp(-steepness * (score - midpoint)))
        
        return round(normalized, 3)
    
    except Exception:
        return 2.0