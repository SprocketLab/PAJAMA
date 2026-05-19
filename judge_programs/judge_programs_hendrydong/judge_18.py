def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and topic-coverage approach. Analyzes how many aspects of the query are addressed,
    depth of treatment per aspect, and structural completeness signals.
    
    This variant focuses on:
    1. Query decomposition into semantic "aspects" (question words, topics, entities)
    2. Measuring coverage of each aspect in the response
    3. Depth scoring via explanation patterns and elaboration signals
    4. Penalizing truncation, superficiality, and missing aspects
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.5
        
        # ---- 1. QUERY ASPECT EXTRACTION ----
        # Extract question words/phrases that indicate sub-questions
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract meaningful content words from query (skip very common words)
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'she',
            'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
            'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
            'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
            'won', 'wouldn', 'also', 'would', 'could', 'might', 'shall', 'may',
            'like', 'get', 'got', 'much', 'many', 'really', 'know', 'think', 'see',
            'seem', 'seems', 'make', 'made', 'go', 'going', 'come', 'take', 'thing',
            'things', 'way', 'even', 'well', 'back', 'still', 'new', 'want', 'first',
            'give', 'us', 'use', 'say', 'said', 'one', 'two', 'been', 'im', 'ive',
        }
        
        def extract_content_words(text):
            words = re.findall(r"[a-z][a-z']+", text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        query_content_words = extract_content_words(query)
        
        # Extract bigrams from query for topic phrases
        query_words_raw = re.findall(r"[a-z][a-z']+", query_lower)
        query_bigrams = []
        for i in range(len(query_words_raw) - 1):
            w1, w2 = query_words_raw[i], query_words_raw[i+1]
            if w1 not in stop_words or w2 not in stop_words:
                query_bigrams.append(f"{w1} {w2}")
        
        # Count how many distinct question types are in the query
        question_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b',
            r'\bwhich\b', r'\bwho\b', r'\bis there\b', r'\bcan\b.*\?',
            r'\bdo(?:es)?\b.*\?', r'\bshould\b', r'\bwould\b.*\?',
            r'\bis it\b', r'\bare there\b', r'\bhas\b.*\?'
        ]
        num_question_types = sum(1 for p in question_patterns if re.search(p, query_lower))
        num_question_types = max(num_question_types, 1)
        
        # Count explicit question marks in query
        num_questions = max(query.count('?'), 1)
        
        # ---- 2. ASPECT COVERAGE SCORING ----
        # What fraction of query content words appear in the response?
        if query_content_words:
            unique_query_words = list(set(query_content_words))
            words_covered = sum(1 for w in unique_query_words if w in response_lower)
            word_coverage = words_covered / len(unique_query_words)
        else:
            word_coverage = 0.5
        
        # Bigram coverage
        if query_bigrams:
            unique_bigrams = list(set(query_bigrams))
            bigrams_covered = sum(1 for bg in unique_bigrams if bg in response_lower)
            bigram_coverage = bigrams_covered / len(unique_bigrams)
        else:
            bigram_coverage = 0.5
        
        # ---- 3. DEPTH AND ELABORATION SIGNALS ----
        response_words = response.split()
        response_word_count = len(response_words)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = max(len(sentences), 1)
        
        # Average sentence length (longer = more detailed, up to a point)
        avg_sentence_len = response_word_count / sentence_count
        
        # Explanation/elaboration markers
        explanation_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bthis means\b', r'\bthis is because\b', r'\bthe reason\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\badditionally\b', r'\bin addition\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\balthough\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bin contrast\b',
            r'\bnamely\b', r'\bthat is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bnote that\b', r'\bimportantly\b', r'\bkeep in mind\b',
        ]
        explanation_count = sum(
            len(re.findall(p, response_lower)) for p in explanation_markers
        )
        
        # Causal/reasoning chain depth
        causal_words = ['because', 'since', 'therefore', 'thus', 'hence', 
                        'consequently', 'as a result', 'so that', 'in order to',
                        'leads to', 'results in', 'causes', 'due to']
        causal_count = sum(response_lower.count(w) for w in causal_words)
        
        # Examples and evidence
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.', r'\blike\s+\w+\s+and\s+\w+', r'\bincluding\b',
            r'\bconsider\b', r'\bimagine\b', r'\bsuppose\b',
        ]
        example_count = sum(
            len(re.findall(p, response_lower)) for p in example_patterns
        )
        
        # ---- 4. STRUCTURAL COMPLETENESS ----
        # Does the response have multiple distinct points/sections?
        # Check for enumeration patterns
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,3}\s', response))
        
        # Count distinct paragraphs (blocks separated by double newline or significant breaks)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        paragraph_count = len(paragraphs)
        
        # ---- 5. TRUNCATION DETECTION ----
        truncation_penalty = 0.0
        # Check if response ends mid-sentence
        response_stripped = response.rstrip()
        if response_stripped:
            last_char = response_stripped[-1]
            if last_char not in '.!?"\')]}:;':
                truncation_penalty = 0.15
            # Check for obvious cut-off patterns
            if response_stripped.endswith(('...', '…')):
                truncation_penalty = 0.05  # mild, could be intentional
            # Ends with a comma or conjunction
            if re.search(r',\s*$', response_stripped) or re.search(r'\b(?:and|or|but|the|a|an|to|in|of)\s*$', response_stripped):
                truncation_penalty = 0.2
        
        # ---- 6. SUPERFICIALITY DETECTION ----
        superficiality_penalty = 0.0
        
        # Very short responses for complex queries
        query_complexity = num_questions + num_question_types + len(query_content_words) / 5
        if response_word_count < 30 and query_complexity > 3:
            superficiality_penalty += 0.2
        if response_word_count < 50 and query_complexity > 5:
            superficiality_penalty += 0.15
        
        # Hedging without substance
        hedge_patterns = [
            r'\bi think\b', r'\bmaybe\b', r'\bprobably\b', r'\bperhaps\b',
            r'\bi guess\b', r'\bnot sure\b', r'\bi believe\b',
        ]
        hedge_count = sum(len(re.findall(p, response_lower)) for p in hedge_patterns)
        # High hedge-to-content ratio
        if response_word_count > 0 and hedge_count / max(sentence_count, 1) > 0.5:
            superficiality_penalty += 0.1
        
        # Deflection patterns (pointing elsewhere without answering)
        deflection_patterns = [
            r'\byou (?:might|should|could) (?:want to |)(?:look|search|google|check)\b',
            r'\bi(?:\'m| am) not (?:sure|certain|qualified)\b',
            r'\bthat\'s a (?:good|great|interesting) question\b',
            r'\byou might be interested in\b',
            r'\bwhile you wait\b',
        ]
        deflection_count = sum(
            len(re.findall(p, response_lower)) for p in deflection_patterns
        )
        if deflection_count > 0 and response_word_count < 80:
            superficiality_penalty += 0.15 * deflection_count
        
        # ---- 7. MULTI-PERSPECTIVE / NUANCE ----
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\balternatively\b',
            r'\bin contrast\b', r'\bconversely\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bwhile\b.*\b(?:also|but)\b',
            r'\bboth\b', r'\bnot only\b.*\bbut also\b',
        ]
        nuance_count = sum(
            len(re.findall(p, response_lower)) for p in contrast_markers
        )
        
        # ---- 8. DOMAIN-SPECIFIC CONTENT DENSITY ----
        # Count unique non-stop content words in response
        response_content_words = extract_content_words(response)
        unique_response_words = set(response_content_words)
        vocabulary_richness = len(unique_response_words) / max(response_word_count, 1)
        
        # Technical/specific term density (words > 6 chars that aren't super common)
        technical_words = [w for w in unique_response_words if len(w) > 6]
        technical_density = len(technical_words) / max(response_word_count, 1)
        
        # ---- SCORING ----
        # Base: response length (diminishing returns via log)
        length_score = min(math.log(max(response_word_count, 1) + 1) / math.log(500), 1.0)
        # Weight: 0-1, ~0.5 at 22 words, ~0.8 at 100 words, ~1.0 at 500 words
        
        # Aspect coverage score (0-1)
        coverage_score = 0.6 * word_coverage + 0.4 * bigram_coverage
        
        # Depth score based on explanation markers (0-1)
        explanation_density = explanation_count / max(sentence_count, 1)
        depth_from_explanations = min(explanation_density / 0.5, 1.0)  # saturates at 0.5 per sentence
        
        # Causal reasoning depth (0-1)
        causal_depth = min(causal_count / 3.0, 1.0)
        
        # Example richness (0-1)
        example_richness = min(example_count / 2.0, 1.0)
        
        # Structural organization bonus (0-1)
        structure_score = 0.0
        if has_numbering or has_bullets:
            structure_score += 0.4
        if paragraph_count >= 2:
            structure_score += 0.3
        if paragraph_count >= 4:
            structure_score += 0.15
        if has_headers:
            structure_score += 0.15
        structure_score = min(structure_score, 1.0)
        
        # Multi-question coverage bonus
        # If query has multiple questions, reward longer/more structured responses
        multi_q_bonus = 0.0
        if num_questions >= 2:
            # Expect at least ~40 words per question
            expected_min_words = num_questions * 40
            if response_word_count >= expected_min_words:
                multi_q_bonus = 0.1
            elif response_word_count >= expected_min_words * 0.5:
                multi_q_bonus = 0.05
        
        # Nuance bonus (0-0.15)
        nuance_bonus = min(nuance_count * 0.05, 0.15)
        
        # Sentence count adequacy relative to query complexity
        expected_sentences = max(query_complexity * 1.5, 3)
        sentence_adequacy = min(sentence_count / expected_sentences, 1.0)
        
        # Vocabulary richness bonus (reward diverse vocabulary, 0-0.1)
        vocab_bonus = min(vocabulary_richness * 0.3, 0.1)
        
        # Technical depth bonus (0-0.1)
        tech_bonus = min(technical_density * 1.5, 0.1)
        
        # Combine into final score
        # Weights chosen to emphasize coverage and depth
        raw_score = (
            2.0 * coverage_score +          # How well query topics are addressed
            2.5 * length_score +             # Sufficient length (log scale)
            1.5 * depth_from_explanations +  # Explanation depth
            1.0 * causal_depth +             # Reasoning chains
            0.8 * example_richness +         # Examples/evidence
            0.7 * structure_score +          # Organization
            1.0 * sentence_adequacy +        # Enough sentences for complexity
            multi_q_bonus * 10 +             # Multi-question bonus (scaled)
            nuance_bonus * 10 +              # Nuance bonus (scaled)
            vocab_bonus * 10 +               # Vocabulary richness
            tech_bonus * 10                  # Technical depth
        )
        
        # Normalize to 0-10 range
        # Max theoretical raw: 2.0 + 2.5 + 1.5 + 1.0 + 0.8 + 0.7 + 1.0 + 1.0 + 1.5 + 1.0 + 1.0 = ~14
        max_raw = 14.0
        normalized = (raw_score / max_raw) * 10.0
        
        # Apply penalties
        total_penalty = truncation_penalty + superficiality_penalty
        total_penalty = min(total_penalty, 0.6)  # Cap penalty at 60%
        
        final_score = normalized * (1.0 - total_penalty)
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: basic length-based score
        try:
            resp_len = len(str(response).split())
            return min(max(resp_len / 50.0, 0.5), 5.0)
        except Exception:
            return 2.5