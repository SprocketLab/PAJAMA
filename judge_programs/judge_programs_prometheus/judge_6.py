def judging_function(query, response):
    """
    Evaluates response relevance using a query-intent matching approach based on:
    - Semantic field coverage (how many query topic clusters are addressed)
    - Question-type alignment (detecting what kind of answer is expected)
    - Discourse coherence signals (how well the response flows as an answer)
    - Empathy/tone matching when emotional queries are detected
    - Specificity and depth of addressing the query's core intent
    
    This variant focuses on intent decomposition and coverage analysis,
    distinct from overlap/similarity/n-gram/paragraph/concreteness approaches.
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
        
        # ---- Helper functions ----
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
        def get_stopwords():
            return set([
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
                'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
                'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'up',
                'about', 'also', 'much', 'many', 'well', 'get', 'got', 'like',
                'make', 'made', 'go', 'going', 'come', 'take', 'know', 'see',
                'think', 'say', 'said', 'one', 'two', 'first', 'new', 'now',
                'way', 'even', 'back', 'any', 'give', 'day', 'us', 'am',
                'been', 'being', 'down', 'don', 't', 's', 're', 've', 'll',
                'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
                'shouldn', 'wasn', 'weren', 'won', 'wouldn'
            ])
        
        stopwords = get_stopwords()
        
        def content_words(text):
            tokens = tokenize(text)
            return [w for w in tokens if w not in stopwords and len(w) > 2]
        
        # ---- 1. Intent Decomposition ----
        # Extract key intent phrases from the query by identifying noun phrases and verb phrases
        query_tokens = tokenize(query)
        query_content = content_words(query)
        resp_tokens = tokenize(response)
        resp_content = content_words(response)
        
        if not query_content or not resp_content:
            return 2.0
        
        # Build semantic clusters from query: groups of co-occurring content words
        # Use a sliding window approach to find topic clusters
        def extract_topic_clusters(content_words_list, window=4):
            clusters = []
            if len(content_words_list) < 2:
                return [set(content_words_list)]
            for i in range(0, len(content_words_list), max(1, window // 2)):
                chunk = set(content_words_list[i:i + window])
                if chunk:
                    clusters.append(chunk)
            return clusters if clusters else [set(content_words_list)]
        
        query_clusters = extract_topic_clusters(query_content)
        resp_content_set = set(resp_content)
        
        # Measure what fraction of each cluster is covered
        cluster_scores = []
        for cluster in query_clusters:
            if not cluster:
                continue
            covered = len(cluster & resp_content_set)
            cluster_scores.append(covered / len(cluster))
        
        # Average cluster coverage - emphasizes that response touches all topics
        avg_cluster_coverage = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
        # Minimum cluster coverage - penalizes missing any topic area
        min_cluster_coverage = min(cluster_scores) if cluster_scores else 0.0
        
        intent_coverage_score = 0.6 * avg_cluster_coverage + 0.4 * min_cluster_coverage
        
        # ---- 2. Query Type Detection & Response Alignment ----
        query_lower = query.lower()
        
        # Detect query type
        is_emotional = bool(re.search(
            r'\b(feel|feeling|emotion|stress|frustrat|sad|happy|angry|upset|devastat|heartbroken|'
            r'loneli|despair|exhaust|comfort|support|listen|empathy|compassion|down|struggle|'
            r'regret|afraid|fear|anxious|worried|concern|pain|grief|griev|mourn)\b', query_lower
        ))
        
        is_howto = bool(re.search(
            r'\b(how\s+(to|can|do|would|should|could|might)|guide|recipe|steps?|instructions?|'
            r'explain|tutorial|process|method|approach|procedure|way\s+to)\b', query_lower
        ))
        
        is_conceptual = bool(re.search(
            r'\b(concept|understand|explain|what\s+is|define|meaning|theory|principle|'
            r'difference|between|compare|contrast)\b', query_lower
        ))
        
        is_problem_solving = bool(re.search(
            r'\b(problem|issue|solve|fix|handle|manage|deal\s+with|address|resolve|'
            r'challenge|difficult|trouble|error|bug|interrupt)\b', query_lower
        ))
        
        is_ambiguous = bool(re.search(
            r'\b(ambiguous|unclear|vague|context|clarif|interpret)\b', query_lower
        ))
        
        resp_lower = response.lower()
        
        # Check response alignment with query type
        type_alignment_score = 0.5  # default neutral
        
        if is_emotional:
            # Emotional queries should have empathetic language
            empathy_markers = len(re.findall(
                r'\b(understand|sorry|hear|feel|okay|natural|normal|valid|'
                r'completely|absolutely|genuinely|tough|difficult|hard|'
                r'take\s+time|it\'s\s+okay|perfectly|compassion|care|'
                r'support|here\s+for|listen|acknowledge|griev|heal)\b', resp_lower
            ))
            # Penalize dismissive language
            dismissive_markers = len(re.findall(
                r'\b(just\s+get\s+over|move\s+on|stop|shouldn\'t\s+feel|'
                r'get\s+yourself\s+together|not\s+a\s+big\s+deal|whatever|'
                r'bummer|suck\s+it\s+up)\b', resp_lower
            ))
            empathy_density = empathy_markers / max(1, len(resp_tokens) / 20)
            type_alignment_score = min(1.0, empathy_density * 0.4) - dismissive_markers * 0.15
            type_alignment_score = max(0.0, min(1.0, type_alignment_score + 0.3))
        
        if is_howto:
            # How-to queries should have structured, actionable content
            action_markers = len(re.findall(
                r'\b(first|second|third|next|then|step|start|begin|after|'
                r'finally|now|once|before|make\s+sure|remember|don\'t\s+forget|'
                r'grab|take|put|add|remove|heat|cook|try|ensure|follow)\b', resp_lower
            ))
            # Check for numbered or bulleted lists
            has_structure = bool(re.search(r'(\d+[\.\):]|\*|-\s|•|first|second|third)', resp_lower))
            action_density = action_markers / max(1, len(resp_tokens) / 15)
            type_alignment_score = min(1.0, action_density * 0.3 + (0.2 if has_structure else 0.0) + 0.3)
        
        if is_conceptual:
            # Conceptual queries should have explanatory language
            explain_markers = len(re.findall(
                r'\b(means?|imagine|think\s+of|like\s+a|similar|analogy|'
                r'example|instance|basically|essentially|simply\s+put|'
                r'in\s+other\s+words|represents?|refers?\s+to|concept|'
                r'principle|works?\s+by|because|due\s+to|therefore|thus)\b', resp_lower
            ))
            explain_density = explain_markers / max(1, len(resp_tokens) / 20)
            type_alignment_score = min(1.0, explain_density * 0.35 + 0.3)
        
        if is_problem_solving:
            # Problem-solving should have solution-oriented language
            solution_markers = len(re.findall(
                r'\b(solution|solve|fix|resolve|handle|manage|approach|'
                r'strategy|recommend|suggest|should|could\s+try|'
                r'implement|design|detect|recognize|maintain|ensure|'
                r'effective|efficient|track|monitor|store|resume)\b', resp_lower
            ))
            solution_density = solution_markers / max(1, len(resp_tokens) / 15)
            type_alignment_score = min(1.0, solution_density * 0.3 + 0.3)
        
        if is_ambiguous:
            # Ambiguous queries: good response should seek clarification
            clarification_markers = len(re.findall(
                r'\b(clarif|more\s+information|details?|specific|which|'
                r'could\s+you|can\s+you\s+(tell|provide|give|share)|'
                r'what\s+do\s+you\s+mean|refer|without\s+(further|more)|'
                r'context|destination|place)\b', resp_lower
            ))
            # Penalize responses that just guess without asking
            if clarification_markers == 0:
                type_alignment_score = 0.15
            else:
                type_alignment_score = min(1.0, clarification_markers * 0.2 + 0.3)
        
        # ---- 3. Discourse Coherence as Answer ----
        # Check if response directly addresses the query (starts with relevant content)
        
        # First sentence relevance
        first_sentence = re.split(r'[.!?\n]', response)[0] if response else ""
        first_sent_content = set(content_words(first_sentence))
        query_content_set = set(query_content)
        
        first_sent_overlap = len(first_sent_content & query_content_set) / max(1, min(len(first_sent_content), len(query_content_set)))
        
        # Check for direct address patterns (indicating the response is actually answering)
        direct_address = bool(re.search(
            r'^(i\s+(can|understand|see|hear|\'m|am)|to\s+|yes|no|sure|'
            r'absolutely|certainly|of\s+course|great|hey|hello|hi|'
            r'imagine|let|here|it\'s|that\'s|the\s+ai|when|if\s+a|'
            r'to\s+mirror|i\'m\s+(sorry|genuinely)|it\s+sounds)',
            resp_lower
        ))
        
        coherence_score = first_sent_overlap * 0.5 + (0.3 if direct_address else 0.0)
        coherence_score = min(1.0, coherence_score + 0.2)
        
        # ---- 4. Specificity & Depth ----
        # Measure how specific and detailed the response is about query topics
        
        # Unique content word ratio (vocabulary richness applied to query topics)
        resp_content_counter = Counter(resp_content)
        query_related_words = [w for w in resp_content if w in query_content_set or 
                               any(w.startswith(qw[:4]) or qw.startswith(w[:4]) 
                                   for qw in query_content_set if len(qw) >= 4 and len(w) >= 4)]
        
        # Topic word density - how much of the response is about the query topic
        topic_density = len(query_related_words) / max(1, len(resp_content))
        
        # Response elaboration - does it go beyond just echoing query words?
        unique_resp_content = set(resp_content) - query_content_set
        elaboration_ratio = len(unique_resp_content) / max(1, len(set(resp_content)))
        
        # Balance: we want both topic relevance AND elaboration
        # Pure echo = bad, pure tangent = bad, mix = good
        specificity_score = 0.0
        if topic_density > 0.01:  # At least some topic words
            # Sweet spot: moderate topic density with good elaboration
            relevance_component = min(1.0, topic_density * 3.0)  # caps at ~33% topic words
            depth_component = min(1.0, elaboration_ratio)
            specificity_score = 0.5 * relevance_component + 0.5 * depth_component
        else:
            specificity_score = 0.1 * min(1.0, elaboration_ratio)
        
        # ---- 5. Negative Signals (off-topic, contradictory, dismissive) ----
        negative_score = 0.0
        
        # Check for hedging/inability markers that suggest low quality
        inability_markers = len(re.findall(
            r'\b(might\s+not|may\s+not|can\'t|cannot|unable|won\'t\s+be\s+able|'
            r'probably\s+won\'t|not\s+have\s+the\s+ability)\b', resp_lower
        ))
        if not is_ambiguous:  # For ambiguous queries, inability is expected
            negative_score += min(0.3, inability_markers * 0.08)
        
        # Check for responses that are too generic (lots of filler, little substance)
        filler_ratio = sum(1 for w in resp_tokens if w in stopwords) / max(1, len(resp_tokens))
        if filler_ratio > 0.7:
            negative_score += 0.15
        
        # ---- 6. Tone Matching ----
        # Detect if query is casual vs formal and check response matches
        query_casual_markers = len(re.findall(
            r'\b(hey|gonna|wanna|gotta|kinda|sorta|cool|awesome|dude|'
            r'chill|vibe|nifty|killer|whip|laid.?back|slang|informal|casual)\b', query_lower
        ))
        resp_casual_markers = len(re.findall(
            r'\b(hey|gonna|wanna|gotta|kinda|sorta|cool|awesome|dude|'
            r'chill|vibe|nifty|killer|whip|alright|let\'s|grab|sweet)\b', resp_lower
        ))
        
        tone_match_score = 0.5  # neutral default
        if query_casual_markers >= 2:
            # Query is casual, response should be too
            if resp_casual_markers >= 2:
                tone_match_score = 0.9
            elif resp_casual_markers >= 1:
                tone_match_score = 0.6
            else:
                tone_match_score = 0.3
        
        # ---- 7. Response Length Appropriateness ----
        resp_len = len(response)
        # Very short responses are usually lower quality for substantive queries
        length_score = 0.5
        if resp_len < 50:
            length_score = 0.2
        elif resp_len < 100:
            length_score = 0.35
        elif resp_len < 200:
            length_score = 0.5
        elif resp_len < 500:
            length_score = 0.7
        else:
            length_score = 0.6  # very long might be rambling
        
        # ---- 8. Stem-based Coverage (fuzzy matching for morphological variants) ----
        def crude_stem(word):
            """Very crude stemmer - just truncate to capture morphological variants."""
            if len(word) <= 4:
                return word
            # Remove common suffixes
            for suffix in ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful', 
                          'less', 'ous', 'ive', 'ing', 'ated', 'ting', 'ted', 'ed',
                          'ly', 'er', 'est', 'ize', 'ise', 'al', 'ial', 'ual']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word[:max(4, len(word) - 2)]
        
        query_stems = set(crude_stem(w) for w in query_content)
        resp_stems = set(crude_stem(w) for w in resp_content)
        
        stem_coverage = len(query_stems & resp_stems) / max(1, len(query_stems))
        
        # ---- Combine all signals ----
        # Weighted combination
        weights = {
            'intent_coverage': 0.20,
            'type_alignment': 0.20,
            'coherence': 0.10,
            'specificity': 0.15,
            'stem_coverage': 0.15,
            'tone_match': 0.05,
            'length': 0.05,
            'negative': 0.10,  # subtracted
        }
        
        raw_score = (
            weights['intent_coverage'] * intent_coverage_score +
            weights['type_alignment'] * type_alignment_score +
            weights['coherence'] * coherence_score +
            weights['specificity'] * specificity_score +
            weights['stem_coverage'] * stem_coverage +
            weights['tone_match'] * tone_match_score +
            weights['length'] * length_score -
            weights['negative'] * negative_score
        )
        
        # Scale to 1-5 range
        # raw_score is roughly 0.0 to 0.9
        final_score = 1.0 + raw_score * 4.5
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 3.0