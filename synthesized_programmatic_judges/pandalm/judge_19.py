def judging_function(query, response):
    """
    Evaluate completeness and coverage of an LLM response using
    query decomposition and information density analysis.
    
    Strategy: Decompose the query into semantic components (question words,
    key concepts, action verbs, sub-tasks) and measure how many are addressed.
    Also analyze information density through unique concept ratio, explanation
    depth, and specificity markers.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        query_lower = query.lower().strip()
        response_lower = response.lower().strip()
        
        response_words = re.findall(r'[a-z]+(?:\'[a-z]+)?', response_lower)
        query_words = re.findall(r'[a-z]+(?:\'[a-z]+)?', query_lower)
        
        if len(response_words) < 2:
            return 0.5
        
        # ============================================================
        # 1. QUERY DECOMPOSITION: Extract what the query is asking for
        # ============================================================
        
        # Extract action verbs / task directives from query
        action_verbs = [
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'generate', 'create', 'write', 'rewrite', 'analyze', 'discuss',
            'define', 'identify', 'suggest', 'recommend', 'evaluate', 'show',
            'demonstrate', 'illustrate', 'outline', 'summarize', 'classify',
            'distinguish', 'elaborate', 'justify', 'argue', 'propose', 'design',
            'develop', 'formulate', 'construct', 'compose', 'produce', 'come up',
            'give', 'name', 'state', 'mention', 'crop', 'reduce', 'add'
        ]
        
        query_actions = []
        for verb in action_verbs:
            if verb in query_lower:
                query_actions.append(verb)
        
        # Count number of distinct tasks/actions requested
        num_tasks = max(len(query_actions), 1)
        
        # Extract key content words from query (nouns, adjectives - approximated by
        # removing stopwords and short words)
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
            'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'these',
            'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'up', 'about', 'also', 'well', 'back', 'even', 'still',
            'way', 'take', 'make', 'like', 'long', 'great', 'little', 'right',
            'think', 'say', 'try', 'ask', 'work', 'call', 'give', 'get', 'go',
            'come', 'know', 'see', 'look', 'find', 'use', 'tell', 'put', 'mean',
            'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play', 'run',
            'move', 'live', 'believe', 'bring', 'happen', 'must', 'following',
            'given', 'input', 'below', 'wrote', 'when', 'did', 'does'
        }
        
        query_content_words = [w for w in query_words if w not in stopwords and len(w) > 2]
        query_content_set = set(query_content_words)
        
        # ============================================================
        # 2. COVERAGE ANALYSIS: How many query concepts appear in response
        # ============================================================
        
        response_word_set = set(response_words)
        
        # Direct concept coverage
        if query_content_set:
            covered = sum(1 for w in query_content_set if w in response_word_set)
            concept_coverage = covered / len(query_content_set)
        else:
            concept_coverage = 0.5
        
        # Semantic coverage via related word stems (crude stemming)
        def crude_stem(word):
            if len(word) <= 3:
                return word
            for suffix in ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful',
                          'less', 'ous', 'ive', 'ing', 'ated', 'ize', 'ise',
                          'ally', 'ity', 'ence', 'ance', 'ers', 'est', 'ies',
                          'ed', 'ly', 'er', 'es', 'al', 'en', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        query_stems = set(crude_stem(w) for w in query_content_words)
        response_stems = set(crude_stem(w) for w in response_words)
        
        if query_stems:
            stem_covered = sum(1 for s in query_stems if s in response_stems)
            stem_coverage = stem_covered / len(query_stems)
        else:
            stem_coverage = 0.5
        
        # ============================================================
        # 3. INFORMATION DENSITY: Unique concepts per unit length
        # ============================================================
        
        # Unique meaningful words ratio
        response_content_words = [w for w in response_words if w not in stopwords and len(w) > 2]
        
        if len(response_content_words) > 0:
            unique_content = set(response_content_words)
            uniqueness_ratio = len(unique_content) / len(response_content_words)
        else:
            uniqueness_ratio = 0.0
        
        # Penalize heavy repetition (same word appearing many times)
        if response_content_words:
            word_counts = Counter(response_content_words)
            max_freq = max(word_counts.values())
            total_content = len(response_content_words)
            repetition_penalty = max(0, (max_freq / total_content) - 0.15) * 3.0
        else:
            repetition_penalty = 0.0
        
        # ============================================================
        # 4. EXPLANATION DEPTH: Markers of thorough explanation
        # ============================================================
        
        # Causal/explanatory connectors
        depth_markers = [
            'because', 'therefore', 'thus', 'hence', 'since', 'as a result',
            'this means', 'in other words', 'for example', 'for instance',
            'such as', 'specifically', 'in particular', 'moreover', 'furthermore',
            'additionally', 'in addition', 'not only', 'but also', 'on the other hand',
            'however', 'although', 'while', 'whereas', 'in contrast', 'similarly',
            'likewise', 'consequently', 'accordingly', 'due to', 'leads to',
            'results in', 'contributes to', 'enables', 'allows', 'ensures',
            'involves', 'includes', 'consists of', 'refers to', 'means that',
            'suggests that', 'indicates that', 'implies that'
        ]
        
        depth_count = sum(1 for marker in depth_markers if marker in response_lower)
        depth_score = min(1.0, depth_count / 4.0)  # Cap at 4 markers for full score
        
        # ============================================================
        # 5. SPECIFICITY: Concrete details vs vague generalities
        # ============================================================
        
        # Count specific/concrete indicators
        specificity_indicators = 0
        
        # Numbers and quantities
        numbers = re.findall(r'\b\d+\b', response)
        specificity_indicators += min(len(numbers), 3)
        
        # Proper nouns (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]+', response)
        for sent in sentences:
            words_in_sent = sent.strip().split()
            for w in words_in_sent[1:]:  # skip first word
                if w and w[0].isupper() and w.lower() not in stopwords:
                    specificity_indicators += 0.5
        
        # Quoted or emphasized text
        quotes = re.findall(r'"[^"]+"|\'[^\']+\'', response)
        specificity_indicators += len(quotes) * 0.5
        
        specificity_score = min(1.0, specificity_indicators / 5.0)
        
        # ============================================================
        # 6. STRUCTURAL COMPLETENESS: Multiple aspects addressed
        # ============================================================
        
        # Count distinct sentences
        sentence_list = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 10]
        num_sentences = len(sentence_list)
        
        # Sentence diversity: how different are the sentences from each other?
        if num_sentences >= 2:
            sentence_word_sets = [set(re.findall(r'[a-z]+', s.lower())) - stopwords for s in sentence_list]
            pairwise_diversity = []
            for i in range(len(sentence_word_sets)):
                for j in range(i+1, len(sentence_word_sets)):
                    union = sentence_word_sets[i] | sentence_word_sets[j]
                    if union:
                        intersection = sentence_word_sets[i] & sentence_word_sets[j]
                        diversity = 1.0 - (len(intersection) / len(union))
                        pairwise_diversity.append(diversity)
            avg_diversity = sum(pairwise_diversity) / len(pairwise_diversity) if pairwise_diversity else 0
        else:
            avg_diversity = 0.0
        
        # ============================================================
        # 7. RESPONSE LENGTH ADEQUACY (relative to query complexity)
        # ============================================================
        
        # Estimate query complexity
        question_words = sum(1 for w in ['what', 'why', 'how', 'when', 'where', 'who', 'which'] if w in query_lower)
        has_multiple_parts = query_lower.count(' and ') + query_lower.count(',')
        query_complexity = 1 + question_words + has_multiple_parts + (num_tasks - 1) * 0.5
        
        # Expected minimum meaningful response length
        expected_min_words = max(15, query_complexity * 20)
        length_adequacy = min(1.0, len(response_words) / expected_min_words)
        
        # Diminishing returns for very long responses
        if len(response_words) > 500:
            length_adequacy *= (1.0 - 0.1 * min(1.0, (len(response_words) - 500) / 1000))
        
        # ============================================================
        # 8. MULTI-TASK COMPLETION CHECK
        # ============================================================
        
        # For queries with multiple explicit tasks (e.g., "crop, reduce, and add border")
        task_completion = 1.0
        if num_tasks > 1:
            tasks_addressed = 0
            for action in query_actions:
                # Check if the action or its result is mentioned in response
                action_stem = crude_stem(action)
                if action in response_lower or action_stem in response_stems:
                    tasks_addressed += 1
                else:
                    # Check for synonyms/related terms loosely
                    tasks_addressed += 0.3  # partial credit
            task_completion = min(1.0, tasks_addressed / num_tasks)
        
        # ============================================================
        # 9. DETECT DEGENERATE RESPONSES
        # ============================================================
        
        degenerate_penalty = 0.0
        
        # Check for excessive repetition of phrases
        if len(response) > 50:
            # Check 3-gram repetition
            trigrams = [' '.join(response_words[i:i+3]) for i in range(len(response_words)-2)]
            if trigrams:
                trigram_counts = Counter(trigrams)
                most_common_tri = trigram_counts.most_common(1)[0][1]
                if most_common_tri > max(3, len(trigrams) * 0.15):
                    degenerate_penalty += 3.0
        
        # Check if response is just echoing the query
        if len(response_words) > 0 and len(query_words) > 0:
            overlap = len(set(response_words) & set(query_words))
            if overlap / max(len(set(response_words)), 1) > 0.8 and len(response_words) < len(query_words) * 1.5:
                degenerate_penalty += 2.0
        
        # Check for noinput / empty-like responses
        if response.strip().lower() in ['<noinput>', 'n/a', 'none', 'no input', '']:
            return 0.5
        
        # ============================================================
        # FINAL SCORING: Weighted combination
        # ============================================================
        
        # Weights emphasizing completeness and coverage
        score = (
            concept_coverage * 15.0 +       # Query concepts addressed
            stem_coverage * 10.0 +           # Broader concept coverage
            depth_score * 15.0 +             # Explanation depth
            specificity_score * 10.0 +       # Concrete details
            uniqueness_ratio * 8.0 +         # Information density
            avg_diversity * 8.0 +            # Sentence diversity (covers multiple aspects)
            length_adequacy * 15.0 +         # Sufficient length for complexity
            task_completion * 12.0 +         # Multi-task completion
            min(1.0, num_sentences / 4.0) * 7.0  # Multiple points made
        )
        
        # Apply penalties
        score -= repetition_penalty * 5.0
        score -= degenerate_penalty
        
        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: simple length-based score
        try:
            return min(50.0, len(response.split()) * 0.5)
        except Exception:
            return 5.0