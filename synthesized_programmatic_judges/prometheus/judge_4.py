def judging_function(query, response):
    """
    Evaluates response relevance using a semantic field coverage approach:
    - Extracts key concepts/entities from the query
    - Builds semantic fields (related word clusters) around query terms
    - Measures how well the response covers these semantic fields
    - Analyzes discourse markers and response structure for direct addressing
    - Uses query intent classification to weight different aspects
    """
    try:
        import re
        import math
        from collections import Counter, defaultdict
        
        if not query or not response:
            return 1.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 1.0
        if len(query) < 5:
            return 3.0
        
        # --- Preprocessing ---
        def tokenize(text):
            return re.findall(r'[a-z]+(?:\'[a-z]+)?', text.lower())
        
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
            'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
            'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'up', 'about', 'also', 'much',
            'many', 'any', 'well', 'still', 'get', 'got', 'make', 'made',
            'like', 'even', 'back', 'way', 'thing', 'things', 'let', 'say',
            'said', 'one', 'two', 'know', 'take', 'come', 'go', 'see', 'look',
            'think', 'going', 'something', 'anything', 'everything', 'nothing',
            'someone', 'anyone', 'everyone', 'really', 'quite', 'rather',
            'person', 'people', 'model', 'response', 'ai'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = [w for w in query_tokens if w not in STOP_WORDS and len(w) > 2]
        response_content = [w for w in response_tokens if w not in STOP_WORDS and len(w) > 2]
        
        if not query_content or not response_content:
            return 2.0
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        response_content_counter = Counter(response_content)
        query_content_counter = Counter(query_content)
        
        # --- 1. Query Intent Classification ---
        query_lower = query.lower()
        
        intent_scores = {
            'how_to': 0, 'explain': 0, 'emotional': 0,
            'opinion': 0, 'factual': 0, 'manage': 0
        }
        
        if any(p in query_lower for p in ['how to', 'how would', 'how can', 'guide', 'steps', 'recipe']):
            intent_scores['how_to'] = 1
        if any(p in query_lower for p in ['explain', 'understand', 'concept', 'what is', 'describe']):
            intent_scores['explain'] = 1
        if any(p in query_lower for p in ['feeling', 'emotion', 'stress', 'sad', 'frustrat', 'comfort',
                                           'heartbroken', 'devastat', 'loneliness', 'despair', 'down',
                                           'breakup', 'passed away', 'grief']):
            intent_scores['emotional'] = 1
        if any(p in query_lower for p in ['manage', 'handle', 'cope', 'deal with', 'adapt']):
            intent_scores['manage'] = 1
        if any(p in query_lower for p in ['opinion', 'think about', 'view on']):
            intent_scores['opinion'] = 1
        if any(p in query_lower for p in ['tell me', 'what', 'when', 'where', 'fact']):
            intent_scores['factual'] = 1
        
        is_emotional = intent_scores['emotional'] == 1
        
        # --- 2. Stem-based matching (poor man's stemming) ---
        def get_stem(word):
            """Simple suffix stripping for approximate matching."""
            w = word.lower()
            for suffix in ['ation', 'ment', 'ness', 'ting', 'ing', 'tion', 'sion',
                           'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ally',
                           'ity', 'ies', 'ely', 'ize', 'ise', 'ated', 'ling',
                           'ers', 'est', 'ent', 'ant', 'ed', 'er', 'ly', 'al', 'es', 's']:
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    return w[:-len(suffix)]
            return w
        
        query_stems = defaultdict(set)
        for w in query_content:
            query_stems[get_stem(w)].add(w)
        
        response_stems = defaultdict(set)
        for w in response_content:
            response_stems[get_stem(w)].add(w)
        
        # Count how many query stems are covered in response
        stem_matches = 0
        for stem in query_stems:
            if stem in response_stems:
                stem_matches += 1
            else:
                # Check partial prefix match (at least 4 chars)
                for rstem in response_stems:
                    min_len = min(len(stem), len(rstem))
                    if min_len >= 4 and stem[:min_len] == rstem[:min_len]:
                        stem_matches += 0.7
                        break
        
        stem_coverage = stem_matches / max(len(query_stems), 1)
        
        # --- 3. Key concept extraction and weighting ---
        # Words that appear less frequently in general are more important
        # Use position in query as a signal too
        query_word_importance = {}
        for i, w in enumerate(query_content):
            # Words appearing later in query often carry more specific meaning
            position_weight = 0.8 + 0.4 * (i / max(len(query_content) - 1, 1))
            # Longer words tend to be more specific
            length_weight = min(len(w) / 6.0, 1.5)
            # Less frequent in query = more likely a key concept (not repeated filler)
            freq_weight = 1.0 / math.sqrt(query_content_counter[w])
            query_word_importance[w] = position_weight * length_weight * freq_weight
        
        # Weighted coverage
        total_importance = sum(query_word_importance.values())
        covered_importance = 0
        for w, imp in query_word_importance.items():
            if w in response_content_set:
                covered_importance += imp
            else:
                # Check stem match
                w_stem = get_stem(w)
                if w_stem in response_stems:
                    covered_importance += imp * 0.75
        
        weighted_coverage = covered_importance / max(total_importance, 0.001)
        
        # --- 4. Semantic field proximity ---
        # Build character-level trigram profiles for fuzzy matching
        def char_trigrams(word):
            if len(word) < 3:
                return set()
            return {word[i:i+3] for i in range(len(word) - 2)}
        
        def word_similarity(w1, w2):
            t1 = char_trigrams(w1)
            t2 = char_trigrams(w2)
            if not t1 or not t2:
                return 0
            return len(t1 & t2) / len(t1 | t2)
        
        # For each query content word, find best matching response word
        fuzzy_match_scores = []
        for qw in query_content_set:
            best_sim = 0
            if qw in response_content_set:
                best_sim = 1.0
            else:
                for rw in response_content_set:
                    sim = word_similarity(qw, rw)
                    if sim > best_sim:
                        best_sim = sim
            fuzzy_match_scores.append(best_sim)
        
        avg_fuzzy_match = sum(fuzzy_match_scores) / max(len(fuzzy_match_scores), 1)
        
        # --- 5. Response directly addresses query signals ---
        # Check for discourse markers indicating direct addressing
        direct_address_patterns = [
            r'\bi (?:can see|understand|hear|sense|notice)',
            r'\bit(?:\'s| is) (?:completely |totally |absolutely |perfectly )?(?:understandable|okay|normal|natural|fine)',
            r'\bhere (?:are|is) (?:some|a|my|the)',
            r'\blet(?:\'s| us|me)',
            r'\bfirst(?:ly|,| thing)',
            r'\bto (?:answer|address|help|assist|respond)',
            r'\byou(?:\'re| are| can| should| might| could| need| may| will)',
            r'\bi(?:\'m| am) (?:sorry|genuinely|truly|here)',
            r'\bremember (?:that|to)',
            r'\btry (?:to|and)',
            r'\bconsider\b',
            r'\bdon\'t (?:worry|hesitate|forget|be afraid)',
            r'\bstep \d',
            r'\b\d+\.',
        ]
        
        response_lower = response.lower()
        direct_address_count = 0
        for pattern in direct_address_patterns:
            if re.search(pattern, response_lower):
                direct_address_count += 1
        
        direct_address_score = min(direct_address_count / 4.0, 1.0)
        
        # --- 6. Emotional resonance (for emotional queries) ---
        emotional_resonance = 0
        if is_emotional:
            empathy_markers = [
                'sorry', 'understand', 'feeling', 'feel', 'hear', 'tough',
                'difficult', 'hard', 'pain', 'natural', 'okay', 'valid',
                'normal', 'grieve', 'grieving', 'comfort', 'support',
                'care', 'listen', 'compassion', 'empathy', 'strength',
                'courage', 'hope', 'healing', 'process', 'time',
                'breathe', 'moment', 'understandable', 'genuinely',
                'truly', 'sincerely', 'acknowledge'
            ]
            empathy_count = sum(1 for m in empathy_markers if m in response_lower)
            emotional_resonance = min(empathy_count / 5.0, 1.0)
            
            # Penalize dismissive language
            dismissive = ['just', 'simply', 'get over', 'move on', 'not a big deal',
                         'no big deal', 'toughen up', 'deal with it']
            dismissive_count = sum(1 for d in dismissive if d in response_lower)
            emotional_resonance -= min(dismissive_count * 0.15, 0.4)
            emotional_resonance = max(emotional_resonance, 0)
        
        # --- 7. Topic coherence: measure if response stays on topic ---
        # Ratio of response words that relate to query domain
        response_relevant_count = 0
        for rw in response_content:
            if rw in query_content_set:
                response_relevant_count += 1
            else:
                rw_stem = get_stem(rw)
                if rw_stem in query_stems:
                    response_relevant_count += 0.7
                else:
                    # Check fuzzy
                    for qw in query_content_set:
                        if word_similarity(rw, qw) > 0.5:
                            response_relevant_count += 0.4
                            break
        
        topic_density = response_relevant_count / max(len(response_content), 1)
        # Normalize - typical good responses have 10-30% topic density
        topic_density_score = min(topic_density / 0.20, 1.0)
        
        # --- 8. Response length appropriateness ---
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Responses should generally be longer than queries
        length_ratio = resp_len / max(query_len, 1)
        if length_ratio < 0.5:
            length_score = 0.3
        elif length_ratio < 1.0:
            length_score = 0.6
        elif length_ratio < 3.0:
            length_score = 1.0
        else:
            length_score = max(0.7, 1.0 - (length_ratio - 3.0) * 0.05)
        
        # --- 9. Structural quality indicators ---
        # Check for organized response (numbered lists, paragraphs, etc.)
        has_structure = 0
        if re.search(r'\d+[\.\)]\s', response):
            has_structure += 0.3
        if response.count('\n\n') >= 1:
            has_structure += 0.2
        if re.search(r'(?:first|second|third|finally|additionally|moreover|furthermore)', response_lower):
            has_structure += 0.2
        if ':' in response:
            has_structure += 0.1
        structure_score = min(has_structure, 0.8)
        
        # --- 10. Specificity vs vagueness ---
        vague_phrases = ['maybe', 'perhaps', 'kind of', 'sort of', 'i guess',
                        'not sure', 'i think maybe', 'probably', 'might be']
        vague_count = sum(1 for v in vague_phrases if v in response_lower)
        
        specific_indicators = ['because', 'for example', 'for instance', 'such as',
                              'specifically', 'in particular', 'this means',
                              'imagine', 'consider', 'think of']
        specific_count = sum(1 for s in specific_indicators if s in response_lower)
        
        specificity_score = min((specific_count + 1) / (vague_count + 2), 1.0)
        
        # --- 11. Query-response sentence alignment ---
        # Split into sentences and check if response sentences relate to query concepts
        def split_sentences(text):
            return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        resp_sentences = split_sentences(response)
        if resp_sentences:
            aligned_sentences = 0
            for sent in resp_sentences:
                sent_tokens = set(tokenize(sent)) - STOP_WORDS
                overlap = sent_tokens & query_content_set
                # Also check stem overlap
                sent_stems = {get_stem(w) for w in sent_tokens if len(w) > 2}
                stem_overlap = sent_stems & set(query_stems.keys())
                if len(overlap) > 0 or len(stem_overlap) > 0:
                    aligned_sentences += 1
            
            sentence_alignment = aligned_sentences / len(resp_sentences)
        else:
            sentence_alignment = 0.3
        
        # --- Combine scores ---
        if is_emotional:
            final_score = (
                stem_coverage * 1.5 +
                weighted_coverage * 2.0 +
                avg_fuzzy_match * 1.0 +
                direct_address_score * 2.0 +
                emotional_resonance * 2.5 +
                topic_density_score * 1.0 +
                length_score * 0.5 +
                structure_score * 0.5 +
                specificity_score * 1.0 +
                sentence_alignment * 1.0
            )
            max_possible = 1.5 + 2.0 + 1.0 + 2.0 + 2.5 + 1.0 + 0.5 + 0.5 + 1.0 + 1.0
        else:
            final_score = (
                stem_coverage * 2.0 +
                weighted_coverage * 2.5 +
                avg_fuzzy_match * 1.5 +
                direct_address_score * 1.5 +
                topic_density_score * 1.5 +
                length_score * 0.5 +
                structure_score * 0.8 +
                specificity_score * 1.2 +
                sentence_alignment * 1.5
            )
            max_possible = 2.0 + 2.5 + 1.5 + 1.5 + 1.5 + 0.5 + 0.8 + 1.2 + 1.5
        
        # Normalize to 1-5 scale
        normalized = final_score / max(max_possible, 0.001)
        result = 1.0 + normalized * 4.0
        
        # Clamp
        result = max(1.0, min(5.0, result))
        
        return round(result, 2)
        
    except Exception:
        return 3.0