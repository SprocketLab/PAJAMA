def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a 
    sentence-level semantic coverage approach with query decomposition.
    
    Algorithm: Decomposes the query into semantic components (question words,
    key entities, action verbs, constraints) and measures how many of these
    components are addressed in the response. Also evaluates structural 
    completeness via sentence diversity, information density, and 
    coherence signals.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        if not query or not query.strip():
            return 5.0
        
        query_clean = query.strip()
        response_clean = response.strip()
        
        # Tokenize into words (lowercase)
        def tokenize(text):
            return re.findall(r'[a-z]+(?:\'[a-z]+)?', text.lower())
        
        query_words = tokenize(query_clean)
        response_words = tokenize(response_clean)
        
        if not response_words:
            return 0.0
        
        # === 1. Query Decomposition Analysis ===
        # Extract different semantic components from the query
        
        # Question words and their implicit demands
        question_patterns = {
            'who': ['person', 'name', 'individual', 'author', 'scholar', 'he', 'she'],
            'what': ['is', 'was', 'are', 'means', 'called', 'named', 'definition'],
            'where': ['location', 'place', 'city', 'country', 'site', 'in', 'at'],
            'when': ['time', 'date', 'year', 'century', 'period', 'during', 'after', 'before'],
            'why': ['because', 'reason', 'cause', 'due', 'since', 'result'],
            'how': ['method', 'way', 'process', 'step', 'by', 'through', 'using'],
            'which': ['option', 'choice', 'type', 'kind', 'category'],
        }
        
        # Detect sub-questions in query
        query_sentences = re.split(r'[.?!;]+', query_clean)
        query_sentences = [s.strip() for s in query_sentences if s.strip()]
        num_sub_questions = sum(1 for s in query_sentences if '?' in s or 
                                any(s.lower().strip().startswith(w) for w in 
                                    ['who', 'what', 'where', 'when', 'why', 'how', 'which',
                                     'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does',
                                     'identify', 'list', 'explain', 'describe', 'create', 'write',
                                     'rewrite', 'find', 'tell', 'name']))
        num_sub_questions = max(num_sub_questions, 1)
        
        # Detect imperative/task demands
        task_verbs = ['identify', 'list', 'explain', 'describe', 'create', 'write', 'rewrite',
                      'find', 'tell', 'name', 'compare', 'summarize', 'analyze', 'provide',
                      'give', 'show', 'calculate', 'determine', 'classify', 'categorize',
                      'translate', 'convert', 'generate', 'make', 'remove', 'regenerate']
        
        query_tasks = [w for w in query_words if w in task_verbs]
        num_tasks = max(len(set(query_tasks)), 1)
        
        # Extract content words from query (nouns, adjectives, key terms)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                     'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'all', 'each', 'every', 'both', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
                     'and', 'or', 'if', 'while', 'although', 'though', 'that', 'this',
                     'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'you',
                     'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
                     'them', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
                     'why', 'how', 'about', 'also', 'please', 'make'}
        
        query_content_words = [w for w in query_words if w not in stopwords and len(w) > 2]
        query_content_set = set(query_content_words)
        
        # === 2. Response Sentence Analysis ===
        response_sentences = re.split(r'[.!?]+', response_clean)
        response_sentences = [s.strip() for s in response_sentences if len(s.strip()) > 5]
        num_response_sentences = len(response_sentences)
        
        # === 3. Query Content Coverage Score ===
        # How many query content words appear in or are addressed by the response
        response_word_set = set(response_words)
        
        if query_content_set:
            direct_coverage = len(query_content_set & response_word_set) / len(query_content_set)
        else:
            direct_coverage = 0.5
        
        # === 4. Semantic Expansion Coverage ===
        # Check if the response contains words semantically related to query demands
        # by looking at response-unique content words (information added)
        response_content_words = [w for w in response_words if w not in stopwords and len(w) > 2]
        response_unique_content = set(response_content_words) - query_content_set
        
        # Information added ratio - how much new content does the response bring
        info_added = len(response_unique_content)
        
        # === 5. Response Depth Metrics ===
        # Sentence count relative to query complexity
        query_complexity = len(query_content_set) + num_sub_questions + num_tasks
        
        # Information density: unique content words per sentence
        if num_response_sentences > 0:
            info_density = len(set(response_content_words)) / num_response_sentences
        else:
            info_density = len(set(response_content_words))
        
        # === 6. Structural Completeness Signals ===
        # Check for explanation patterns
        explanation_markers = ['because', 'therefore', 'however', 'although', 'furthermore',
                               'moreover', 'additionally', 'specifically', 'for example',
                               'such as', 'in addition', 'as a result', 'this means',
                               'in other words', 'on the other hand', 'first', 'second',
                               'third', 'finally', 'also', 'including', 'notably']
        
        response_lower = response_clean.lower()
        explanation_count = sum(1 for marker in explanation_markers if marker in response_lower)
        
        # Check for enumeration/listing patterns
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-•*]\s', response_clean))
        has_colons = response_clean.count(':') >= 1
        
        structural_bonus = (0.3 if has_numbering else 0) + (0.2 if has_bullets else 0) + (0.1 if has_colons else 0)
        
        # === 7. Repetition Penalty ===
        # Detect repeated content (a sign of low quality / padding)
        if len(response_sentences) >= 2:
            sentence_texts = [re.sub(r'\s+', ' ', s.lower().strip()) for s in response_sentences]
            unique_sentences = set(sentence_texts)
            repetition_ratio = len(unique_sentences) / len(sentence_texts)
        else:
            repetition_ratio = 1.0
        
        # Word-level repetition: ratio of unique words to total
        if response_words:
            word_diversity = len(set(response_words)) / len(response_words)
        else:
            word_diversity = 0
        
        # === 8. Off-topic / Noise Detection ===
        # Check for HTML tags, code blocks, excessive special characters
        html_tags = len(re.findall(r'<[^>]+>', response_clean))
        code_blocks = len(re.findall(r'```', response_clean))
        
        # Check if response seems to be repeating the query or going off-track
        # by detecting patterns like "Input:", "Output:", "Question:", repeated
        meta_patterns = len(re.findall(r'(?:Input|Output|Question|Answer)\s*:', response_clean))
        
        noise_penalty = 0
        if html_tags > 3:
            noise_penalty += 0.15
        if meta_patterns > 3:
            noise_penalty += 0.1
        
        # === 9. Length Appropriateness ===
        response_len = len(response_words)
        query_len = len(query_words)
        
        # Very short responses are usually incomplete
        if response_len <= 3:
            length_score = 0.1
        elif response_len <= 10:
            length_score = 0.3
        elif response_len <= 25:
            length_score = 0.5
        elif response_len <= 50:
            length_score = 0.7
        elif response_len <= 150:
            length_score = 0.9
        elif response_len <= 300:
            length_score = 1.0
        else:
            # Very long might include noise, but could also be thorough
            length_score = 0.95
        
        # Adjust length expectation based on query complexity
        if query_complexity > 8 and response_len < 30:
            length_score *= 0.6
        
        # === 10. Topical Coherence ===
        # Measure if response stays on topic using a simple approach:
        # what fraction of response sentences contain at least one query content word
        if num_response_sentences > 0 and query_content_set:
            on_topic_sentences = 0
            for sent in response_sentences:
                sent_words = set(tokenize(sent))
                if sent_words & query_content_set:
                    on_topic_sentences += 1
            topical_coherence = on_topic_sentences / num_response_sentences
        else:
            topical_coherence = 0.5
        
        # === 11. Completeness for specific query types ===
        type_bonus = 0
        
        # "Rewrite" tasks: check if multiple versions are provided
        if any(w in query_words for w in ['rewrite', 'regenerate', 'rephrase']):
            # Look for number mentions in query
            number_match = re.search(r'(\d+)\s*(?:different|ways|versions|times)', query_clean.lower())
            if number_match:
                expected_count = int(number_match.group(1))
                # Count distinct outputs in response
                output_count = max(
                    len(re.findall(r'(?:^|\n)', response_clean)) - 1,
                    len(re.findall(r'Output:', response_clean)),
                    num_response_sentences
                )
                if output_count >= expected_count:
                    type_bonus = 0.3
                elif output_count >= expected_count - 1:
                    type_bonus = 0.15
        
        # "Identify/classify" tasks: check if all items are addressed
        if any(w in query_words for w in ['identify', 'classify', 'categorize']):
            # Count comma-separated items in query
            items_in_query = len(re.findall(r',', query_clean)) + 1
            # Check if response mentions those items
            items_covered = sum(1 for item in re.split(r'[,\n]', query_clean) 
                              if any(w in response_lower for w in tokenize(item) if w not in stopwords and len(w) > 2))
            if items_in_query > 0:
                item_coverage = items_covered / items_in_query
                type_bonus = 0.2 * item_coverage
        
        # === SCORING FORMULA ===
        # Weighted combination of all signals
        
        # Core coverage (0-1)
        coverage_score = (
            0.25 * direct_coverage +
            0.15 * min(info_added / max(query_complexity * 3, 1), 1.0) +
            0.20 * length_score +
            0.10 * min(explanation_count / 3, 1.0) +
            0.10 * topical_coherence +
            0.05 * min(info_density / 8, 1.0) +
            0.05 * word_diversity +
            0.05 * repetition_ratio +
            0.05 * min(num_response_sentences / max(num_sub_questions * 2, 2), 1.0)
        )
        
        # Apply bonuses
        coverage_score += structural_bonus * 0.1
        coverage_score += type_bonus
        
        # Apply penalties
        coverage_score *= (1 - noise_penalty)
        
        # Penalty for very low repetition ratio (lots of repeated content)
        if repetition_ratio < 0.5:
            coverage_score *= 0.7
        
        # Penalty for single-word or near-empty responses
        if response_len <= 2:
            coverage_score *= 0.15
        elif response_len <= 5:
            coverage_score *= 0.35
        
        # Harsh penalty for responses that are clearly non-answers
        non_answer_patterns = ['^no$', '^yes$', '^\\.$', '^n/a$', '^none$']
        if any(re.match(p, response_clean.strip().lower()) for p in non_answer_patterns):
            coverage_score *= 0.1
        
        # Scale to 0-10
        final_score = coverage_score * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: basic length-based score
        try:
            words = len(response.split())
            if words <= 2:
                return 1.0
            elif words <= 10:
                return 3.0
            elif words <= 50:
                return 5.0
            else:
                return 6.0
        except Exception:
            return 3.0