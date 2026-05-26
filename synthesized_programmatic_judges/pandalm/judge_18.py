def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using question decomposition
    and information density analysis. 
    
    Strategy: Decompose the query into semantic components (question words, key topics,
    requested actions) and check how many are addressed in the response. Also measure
    information density through unique concept coverage and specificity signals.
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
        
        response_words = response_lower.split()
        query_words = query_lower.split()
        
        if len(response_words) == 0:
            return 0.0
        
        # ---- 1. Query Decomposition: Extract sub-tasks and key topics ----
        
        # Extract action verbs from query (imperatives / task requests)
        action_verbs = [
            'compare', 'contrast', 'describe', 'explain', 'provide', 'list',
            'generate', 'create', 'write', 'rewrite', 'analyze', 'discuss',
            'define', 'identify', 'summarize', 'evaluate', 'suggest', 'recommend',
            'show', 'demonstrate', 'illustrate', 'outline', 'elaborate', 'clarify',
            'crop', 'reduce', 'add', 'convert', 'translate', 'come up with',
            'give', 'name', 'state', 'mention', 'classify', 'categorize',
            'differentiate', 'distinguish', 'assess', 'critique', 'review',
            'propose', 'design', 'develop', 'construct', 'formulate', 'compose'
        ]
        
        requested_actions = []
        for verb in action_verbs:
            if verb in query_lower:
                requested_actions.append(verb)
        
        # Check for conjunctions suggesting multiple sub-tasks
        multi_task_markers = [' and ', ', and ', ' also ', ' as well as ', ' plus ', ' then ']
        num_subtasks = 1
        for marker in multi_task_markers:
            num_subtasks += query_lower.count(marker)
        
        # Extract question words indicating what needs to be covered
        question_patterns = {
            'what': r'\bwhat\b',
            'why': r'\bwhy\b',
            'how': r'\bhow\b',
            'when': r'\bwhen\b',
            'where': r'\bwhere\b',
            'who': r'\bwho\b',
            'which': r'\bwhich\b',
        }
        question_types = []
        for qtype, pattern in question_patterns.items():
            if re.search(pattern, query_lower):
                question_types.append(qtype)
        
        # ---- 2. Key Topic Extraction from Query ----
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'that', 'this', 'these', 'those',
            'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'its', 'his', 'her', 'their', 'our',
            'give', 'given', 'following', 'input', 'wrote', 'them'
        }
        
        # Extract content words from query
        query_content_words = []
        for w in re.findall(r'\b[a-z]+\b', query_lower):
            if w not in stopwords and len(w) > 2:
                query_content_words.append(w)
        
        # Also extract bigrams from query for multi-word concepts
        query_bigrams = []
        q_tokens = re.findall(r'\b[a-z]+\b', query_lower)
        for i in range(len(q_tokens) - 1):
            if q_tokens[i] not in stopwords or q_tokens[i+1] not in stopwords:
                query_bigrams.append(q_tokens[i] + ' ' + q_tokens[i+1])
        
        # ---- 3. Coverage Score: How many query topics appear in response ----
        
        unique_query_topics = list(set(query_content_words))
        if len(unique_query_topics) > 0:
            topics_covered = sum(1 for t in unique_query_topics if t in response_lower)
            topic_coverage_ratio = topics_covered / len(unique_query_topics)
        else:
            topic_coverage_ratio = 0.5
        
        # Bigram coverage
        if len(query_bigrams) > 0:
            bigrams_covered = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_coverage_ratio = bigrams_covered / len(query_bigrams)
        else:
            bigram_coverage_ratio = 0.5
        
        # ---- 4. Information Density Metrics ----
        
        # Unique words ratio (penalize repetition)
        response_word_list = re.findall(r'\b[a-z]+\b', response_lower)
        if len(response_word_list) > 0:
            unique_ratio = len(set(response_word_list)) / len(response_word_list)
        else:
            unique_ratio = 0.0
        
        # Severe repetition detection
        word_counts = Counter(response_word_list)
        if len(word_counts) > 0:
            most_common_freq = word_counts.most_common(1)[0][1]
            # Content word repetition (excluding stopwords)
            content_word_counts = {w: c for w, c in word_counts.items() if w not in stopwords and len(w) > 2}
            if content_word_counts:
                max_content_repeat = max(content_word_counts.values())
                repetition_penalty = min(1.0, 3.0 / max(max_content_repeat, 1))
            else:
                repetition_penalty = 0.5
        else:
            repetition_penalty = 0.0
        
        # ---- 5. Specificity and Detail Signals ----
        
        # Count specific detail markers
        detail_markers = [
            r'\bfor example\b', r'\bsuch as\b', r'\bincluding\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bfor instance\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bnamely\b', r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\balso\b', r'\badditionally\b', r'\bas well\b'
        ]
        detail_count = sum(1 for pat in detail_markers if re.search(pat, response_lower))
        detail_score = min(1.0, detail_count / 3.0)
        
        # Count numbers and quantifiers (specificity signal)
        numbers = re.findall(r'\b\d+\b', response_lower)
        quantifiers = re.findall(r'\b(?:many|several|few|multiple|various|numerous|different|diverse)\b', response_lower)
        specificity_count = len(numbers) + len(quantifiers)
        specificity_score = min(1.0, specificity_count / 4.0)
        
        # ---- 6. Structural Completeness ----
        
        # Sentence count as proxy for multi-faceted coverage
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Score based on sentence count relative to query complexity
        expected_sentences = max(2, num_subtasks + len(question_types))
        sentence_coverage = min(1.5, num_sentences / max(expected_sentences, 1))
        
        # Check for list items (bullet points, numbered items)
        list_items = re.findall(r'(?:^|\n)\s*(?:[-•*]|\d+[.)]) ', response)
        has_list_structure = len(list_items) > 0
        
        # ---- 7. Response Length Adequacy ----
        
        # Not just raw length, but length relative to query complexity
        response_len = len(response_words)
        query_complexity = len(requested_actions) + num_subtasks + len(question_types)
        
        # Adaptive length scoring
        if query_complexity <= 1:
            ideal_length = 30
        elif query_complexity <= 3:
            ideal_length = 60
        else:
            ideal_length = 100
        
        if response_len < 5:
            length_score = 0.1
        elif response_len < ideal_length * 0.3:
            length_score = 0.3
        elif response_len < ideal_length * 0.6:
            length_score = 0.6
        elif response_len <= ideal_length * 2.5:
            length_score = 1.0
        else:
            # Diminishing returns for very long responses
            length_score = 1.0 - min(0.3, (response_len - ideal_length * 2.5) / (ideal_length * 10))
        
        # ---- 8. Truncation / Incompleteness Detection ----
        
        truncation_penalty = 1.0
        # Check if response appears cut off
        if response.strip()[-1:] not in '.!?")\']' and len(response) > 50:
            truncation_penalty = 0.75
        # Check for incomplete sentences at end
        last_sentence = sentences[-1] if sentences else ""
        if len(last_sentence.split()) < 3 and len(sentences) > 1:
            truncation_penalty = min(truncation_penalty, 0.85)
        
        # ---- 9. Action Verb Fulfillment ----
        
        # For "compare and contrast" queries, check for comparison language
        comparison_words = ['while', 'whereas', 'unlike', 'similar', 'different', 'both',
                           'however', 'contrast', 'comparison', 'on the other hand',
                           'in contrast', 'differ', 'same', 'like']
        
        action_fulfillment = 1.0
        if 'compare' in requested_actions or 'contrast' in requested_actions:
            comp_count = sum(1 for cw in comparison_words if cw in response_lower)
            action_fulfillment = min(1.0, comp_count / 3.0)
        
        if 'explain' in requested_actions or 'describe' in requested_actions:
            # Check for explanatory depth
            explain_markers = ['because', 'means', 'refers to', 'suggests', 'implies',
                             'in other words', 'this is', 'therefore', 'thus', 'process']
            explain_count = sum(1 for em in explain_markers if em in response_lower)
            action_fulfillment = min(1.0, max(action_fulfillment, explain_count / 2.0))
        
        if 'list' in requested_actions or 'provide' in requested_actions:
            # Check for enumeration
            if has_list_structure or response_lower.count(',') >= 2:
                action_fulfillment = max(action_fulfillment, 0.8)
        
        # ---- 10. Emptiness / Nonsense Detection ----
        
        nonsense_penalty = 1.0
        # Check for "<noinput>" or similar non-answers
        non_answer_patterns = ['<noinput>', 'n/a', 'no input', 'i cannot', 'i can\'t',
                               'as an ai', 'i don\'t have']
        for nap in non_answer_patterns:
            if nap in response_lower:
                nonsense_penalty = 0.3
                break
        
        # Check if response is just echoing the query
        if len(response_words) > 3 and len(query_words) > 3:
            overlap = len(set(response_word_list) & set(re.findall(r'\b[a-z]+\b', query_lower)))
            total = len(set(response_word_list))
            if total > 0 and overlap / total > 0.85:
                nonsense_penalty = min(nonsense_penalty, 0.5)
        
        # ---- Composite Score ----
        
        # Weighted combination
        score = (
            topic_coverage_ratio * 15.0 +      # 0-15: query topics addressed
            bigram_coverage_ratio * 5.0 +       # 0-5: multi-word concept coverage
            unique_ratio * 10.0 +               # 0-10: vocabulary diversity
            repetition_penalty * 8.0 +          # 0-8: penalize heavy repetition
            detail_score * 10.0 +               # 0-10: specific details
            specificity_score * 5.0 +           # 0-5: numbers/quantifiers
            sentence_coverage * 10.0 +          # 0-15: structural coverage
            length_score * 15.0 +               # 0-15: adequate length
            action_fulfillment * 10.0 +         # 0-10: task fulfillment
            (1.0 if has_list_structure else 0.0) * 2.0  # 0-2: list bonus
        )
        
        # Apply penalties
        score *= truncation_penalty
        score *= nonsense_penalty
        
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            return min(50.0, max(1.0, len(response.split()) * 0.5))
        except Exception:
            return 5.0