def judging_function(query, response):
    """
    Evaluates response relevance using a query-intent matching approach based on:
    - Semantic field coverage (how many query topic clusters are addressed)
    - Question-type detection and response appropriateness
    - Weighted keyword proximity analysis
    - Response engagement and directness signals
    - Penalty for evasion, negativity, or off-topic drift
    
    This variant focuses on intent decomposition and coverage analysis,
    distinct from overlap/similarity/n-gram/paragraph approaches.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 1.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 1.0
        
        # --- Utility functions ---
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
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
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
            'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'about', 'up', 'down', 'also', 'much', 'get', 'got',
            'like', 'make', 'made', 'know', 'think', 'say', 'said', 'well',
            'even', 'back', 'still', 'way', 'take', 'come', 'go', 'see', 'thing',
            'things', 'really', 'right', 'now', 'new', 'one', 'two', 'first',
            'time', 'long', 'been', 'people', 'person', 'individual', 'must',
            'model', 'ai', 'response', 'provide', 'following'
        }
        
        def content_words(text):
            tokens = tokenize(text)
            return [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
        
        def get_phrases(text, n=2):
            """Extract meaningful consecutive word pairs from content words in context."""
            tokens = tokenize(text)
            phrases = set()
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i+n])
                if any(w not in STOP_WORDS and len(w) > 2 for w in gram):
                    phrases.add(gram)
            return phrases
        
        # --- 1. Intent Decomposition ---
        # Extract the core intent topics from the query by identifying topic clusters
        query_content = content_words(query)
        response_content = content_words(response)
        
        if not query_content:
            return 3.0
        
        # Build topic clusters: group query words by semantic proximity
        # (co-occurrence within sliding windows in the query)
        def build_topic_clusters(words, window=5):
            """Group words that appear near each other into topic clusters."""
            clusters = []
            if len(words) <= window:
                return [set(words)] if words else []
            
            for i in range(0, len(words), window // 2):
                chunk = set(words[i:i+window])
                if chunk:
                    # Merge with existing cluster if overlap
                    merged = False
                    for c in clusters:
                        if c & chunk:
                            c.update(chunk)
                            merged = True
                            break
                    if not merged:
                        clusters.append(chunk)
            return clusters if clusters else [set(words)]
        
        topic_clusters = build_topic_clusters(query_content, window=6)
        
        # Measure coverage: what fraction of topic clusters are addressed
        response_word_set = set(response_content)
        
        cluster_scores = []
        for cluster in topic_clusters:
            if not cluster:
                continue
            # Check how many words from this cluster appear in response
            # Also check for synonyms/related words via stem matching
            hits = 0
            for word in cluster:
                if word in response_word_set:
                    hits += 1
                else:
                    # Check stem overlap (first 4+ chars match)
                    stem = word[:min(5, len(word))]
                    for rw in response_word_set:
                        if rw.startswith(stem) or stem.startswith(rw[:min(5, len(rw))]):
                            hits += 0.6
                            break
            coverage = hits / len(cluster) if cluster else 0
            cluster_scores.append(min(coverage, 1.0))
        
        topic_coverage = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0
        
        # --- 2. Question Type Detection & Response Appropriateness ---
        query_lower = query.lower()
        
        # Detect query intent type
        is_how_to = bool(re.search(r'\bhow\b.*\b(to|can|could|would|should|do)\b', query_lower))
        is_explain = bool(re.search(r'\b(explain|understand|concept|what is|what are|describe)\b', query_lower))
        is_emotional = bool(re.search(r'\b(feel|feeling|emotion|stress|sad|happy|frustrat|heartbrok|devastat|lonely|loneliness|comfort|support|despair|exhausti|down)\b', query_lower))
        is_advice = bool(re.search(r'\b(advice|help|assist|suggest|recommend|guide|tip|cope|manage|handle)\b', query_lower))
        is_technical = bool(re.search(r'\b(model|system|design|implement|algorithm|ai|compute|process|method|approach)\b', query_lower))
        
        response_lower = response.lower()
        
        appropriateness_score = 0.5  # baseline
        
        if is_emotional:
            # Check for empathetic language
            empathy_markers = [
                r'\b(sorry|understand|hear|feel|okay|natural|valid|normal)\b',
                r'\b(completely|absolutely|totally|genuinely|truly)\b.*\b(understand|okay|fine|natural)\b',
                r'\b(it\'s okay|it\'s fine|that\'s okay|perfectly)\b',
            ]
            empathy_count = sum(1 for p in empathy_markers if re.search(p, response_lower))
            
            # Penalize dismissive language
            dismissive = [
                r'\bjust\b.*\b(get over|move on|stop|forget)\b',
                r'\byou (need|should) to\b.*\b(get|stop|move)\b',
                r'\bget yourself together\b',
            ]
            dismissive_count = sum(1 for p in dismissive if re.search(p, response_lower))
            
            appropriateness_score = min(1.0, 0.3 + empathy_count * 0.2 - dismissive_count * 0.25)
        
        elif is_how_to:
            # Check for actionable steps/instructions
            step_indicators = len(re.findall(r'(?:first|second|third|then|next|finally|step|start|begin|\d+\.)', response_lower))
            action_verbs = len(re.findall(r'\b(add|remove|create|open|click|type|enter|select|choose|press|heat|cook|mix|stir|pour|cut|place|set|turn|use|apply|install|run|build|write)\b', response_lower))
            appropriateness_score = min(1.0, 0.3 + step_indicators * 0.08 + action_verbs * 0.04)
        
        elif is_explain:
            # Check for explanatory structures
            explain_markers = len(re.findall(r'\b(means|because|therefore|essentially|basically|imagine|think of|in other words|for example|such as|like a|similar to)\b', response_lower))
            appropriateness_score = min(1.0, 0.3 + explain_markers * 0.1)
        
        elif is_advice:
            advice_markers = len(re.findall(r'\b(try|consider|might|could|suggest|recommend|tip|approach|strategy|technique|method|way to|help)\b', response_lower))
            appropriateness_score = min(1.0, 0.3 + advice_markers * 0.06)
        
        elif is_technical:
            tech_markers = len(re.findall(r'\b(detect|recognize|store|maintain|handle|process|design|implement|system|model|algorithm|function|data|input|output|stack|queue|track)\b', response_lower))
            appropriateness_score = min(1.0, 0.3 + tech_markers * 0.06)
        
        # --- 3. Weighted Keyword Proximity Analysis ---
        # How close together do query keywords appear in the response?
        # Responses that weave query concepts together are more relevant
        
        response_tokens = tokenize(response)
        query_unique = list(set(query_content))
        
        # Find positions of query keywords in response
        keyword_positions = {}
        for i, token in enumerate(response_tokens):
            for qw in query_unique:
                stem = qw[:min(5, len(qw))]
                if token == qw or (len(token) >= 4 and token.startswith(stem)):
                    if qw not in keyword_positions:
                        keyword_positions[qw] = []
                    keyword_positions[qw].append(i)
        
        # Calculate average minimum distance between different query keywords in response
        proximity_score = 0.0
        if len(keyword_positions) >= 2:
            keys = list(keyword_positions.keys())
            distances = []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    min_dist = float('inf')
                    for p1 in keyword_positions[keys[i]]:
                        for p2 in keyword_positions[keys[j]]:
                            min_dist = min(min_dist, abs(p1 - p2))
                    if min_dist < float('inf'):
                        distances.append(min_dist)
            
            if distances:
                avg_dist = sum(distances) / len(distances)
                # Closer = better, normalize with decay
                proximity_score = math.exp(-avg_dist / 30.0)
        elif len(keyword_positions) >= 1:
            proximity_score = 0.3
        
        # --- 4. Response Engagement & Directness ---
        # Does the response directly engage with the query rather than being evasive?
        
        # Check for hedging/evasion
        hedging_patterns = [
            r'\b(might not|may not|probably won\'t|can\'t|cannot|unable)\b.*\b(able|help|provide|do)\b',
            r'\bit (might|may|probably) not\b',
            r'\bnot (able|sure|certain)\b',
        ]
        hedging_count = sum(1 for p in hedging_patterns if re.search(p, response_lower))
        
        # Check for direct engagement
        engagement_signals = 0
        
        # Addressing the user directly
        if re.search(r'\b(you|your|you\'re|you\'ve)\b', response_lower):
            engagement_signals += 1
        
        # Using specific/concrete language rather than vague
        specific_count = len(re.findall(r'\b(specifically|particular|example|instance|such as|for instance|namely)\b', response_lower))
        engagement_signals += min(specific_count, 3) * 0.3
        
        # Structured response (numbered lists, clear sections)
        structure_markers = len(re.findall(r'(?:\d+[\.\):]|^[-•*]|\n[-•*])', response))
        engagement_signals += min(structure_markers, 5) * 0.2
        
        directness_score = min(1.0, (engagement_signals * 0.15 + 0.4) - hedging_count * 0.15)
        directness_score = max(0.0, directness_score)
        
        # --- 5. Semantic Field Density ---
        # How much of the response is on-topic vs off-topic?
        # Compute the ratio of response words that relate to query topics
        
        query_stems = set()
        for w in query_content:
            query_stems.add(w[:min(5, len(w))])
            query_stems.add(w)
        
        # Expand with simple related terms based on query context
        on_topic_count = 0
        for rw in response_content:
            rstem = rw[:min(5, len(rw))]
            if rw in query_stems or rstem in query_stems:
                on_topic_count += 1
        
        semantic_density = on_topic_count / len(response_content) if response_content else 0
        # Don't over-penalize: some new relevant words are expected
        semantic_density = min(1.0, semantic_density * 3.0)  # scale up since exact match is strict
        
        # --- 6. Response Length Appropriateness ---
        query_complexity = len(query_content)
        response_length = len(response_content)
        
        # Very short responses to complex queries are usually worse
        length_ratio = response_length / max(query_complexity, 1)
        if length_ratio < 1.0:
            length_score = 0.3
        elif length_ratio < 2.0:
            length_score = 0.6
        elif length_ratio < 8.0:
            length_score = 0.9
        else:
            length_score = max(0.5, 1.0 - (length_ratio - 8) * 0.02)  # slight penalty for extreme verbosity
        
        # --- 7. Phrase-level overlap ---
        # Check if meaningful bigrams from query appear in response
        query_phrases = get_phrases(query, 2)
        response_phrases = get_phrases(response, 2)
        
        if query_phrases:
            phrase_overlap = len(query_phrases & response_phrases) / len(query_phrases)
        else:
            phrase_overlap = 0.0
        
        # --- 8. Tone Matching ---
        # If query is casual, response should be casual; if formal, response should be formal
        query_casual_markers = len(re.findall(r'\b(hey|cool|awesome|gonna|wanna|kinda|sorta|nah|yeah|dude|bro|chill|killer|whip|grab|alright|let\'s)\b', query_lower))
        response_casual_markers = len(re.findall(r'\b(hey|cool|awesome|gonna|wanna|kinda|sorta|nah|yeah|dude|bro|chill|killer|whip|grab|alright|let\'s)\b', response_lower))
        
        query_formal = len(re.findall(r'\b(therefore|furthermore|consequently|moreover|nevertheless|regarding|concerning|pursuant)\b', query_lower))
        response_formal = len(re.findall(r'\b(therefore|furthermore|consequently|moreover|nevertheless|regarding|concerning|pursuant)\b', response_lower))
        
        tone_match = 0.5  # neutral baseline
        if query_casual_markers > 1:
            # Query is casual
            if response_casual_markers > 0:
                tone_match = 0.9
            else:
                tone_match = 0.4
        elif query_formal > 0:
            if response_formal > 0 or response_casual_markers == 0:
                tone_match = 0.8
            else:
                tone_match = 0.4
        else:
            tone_match = 0.6  # neutral query, any tone is okay
        
        # --- Combine all signals with weights ---
        # topic_coverage: 0-1, weight 0.25
        # appropriateness_score: 0-1, weight 0.20
        # proximity_score: 0-1, weight 0.10
        # directness_score: 0-1, weight 0.10
        # semantic_density: 0-1, weight 0.15
        # length_score: 0-1, weight 0.05
        # phrase_overlap: 0-1, weight 0.10
        # tone_match: 0-1, weight 0.05
        
        combined = (
            topic_coverage * 0.25 +
            appropriateness_score * 0.20 +
            proximity_score * 0.10 +
            directness_score * 0.10 +
            semantic_density * 0.15 +
            length_score * 0.05 +
            phrase_overlap * 0.10 +
            tone_match * 0.05
        )
        
        # Scale to 1-5 range
        score = 1.0 + combined * 4.0
        
        # Clamp
        score = max(1.0, min(5.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 3.0