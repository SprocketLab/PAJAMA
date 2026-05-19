def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and semantic coverage approach. Decomposes the query into constituent information needs
    (sub-questions, entities, keywords) and checks how many are addressed in the response.
    Also evaluates structural depth indicators like explanations, examples, and reasoning chains.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        query_clean = query.strip().lower()
        response_clean = response.strip().lower()
        response_raw = response.strip()
        
        # === 1. QUERY DECOMPOSITION ===
        # Extract interrogative words to understand what types of answers are needed
        interrogatives = {
            'who': 'person/entity',
            'what': 'definition/thing',
            'where': 'location',
            'when': 'time',
            'why': 'reason',
            'how': 'method/process',
            'which': 'selection',
            'how many': 'quantity',
            'how much': 'quantity',
            'can you': 'action',
            'could you': 'action',
            'is it': 'yes_no',
            'are there': 'existence',
            'do you': 'yes_no',
            'does': 'yes_no',
        }
        
        query_types = []
        for interrog, qtype in interrogatives.items():
            if interrog in query_clean:
                query_types.append(qtype)
        
        # Count sub-questions (sentences ending with ?)
        query_sentences = re.split(r'[.!?]+', query.strip())
        query_questions = [s.strip() for s in re.findall(r'[^.!?]*\?', query.strip())]
        num_sub_questions = max(len(query_questions), 1)
        
        # Extract imperative verbs (tasks) from query
        imperative_patterns = [
            r'\b(identify|list|describe|explain|create|write|rewrite|generate|'
            r'compare|contrast|analyze|summarize|provide|give|tell|show|find|'
            r'name|define|calculate|determine|classify|categorize|translate|'
            r'convert|make|remove|add|include|exclude)\b'
        ]
        tasks_found = []
        for pat in imperative_patterns:
            tasks_found.extend(re.findall(pat, query_clean))
        num_tasks = max(len(set(tasks_found)), 1)
        
        # Extract key content words from query (nouns, adjectives - approximated by
        # removing stopwords and short words)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'it', 'its', 'i', 'me', 'my',
            'you', 'your', 'he', 'she', 'they', 'them', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'whom', 'also', 'please', 'make',
            'any', 'know', 'want', 'like', 'get', 'got', 'us', 'we', 'our',
        }
        
        query_words = re.findall(r'\b[a-z]+\b', query_clean)
        query_content_words = [w for w in query_words if w not in stopwords and len(w) > 2]
        
        # Extract multi-word phrases (bigrams) from query for better matching
        query_bigrams = []
        for i in range(len(query_words) - 1):
            if query_words[i] not in stopwords or query_words[i+1] not in stopwords:
                query_bigrams.append(query_words[i] + ' ' + query_words[i+1])
        
        # === 2. COVERAGE ANALYSIS ===
        
        # 2a. Keyword coverage: what fraction of query content words appear in response
        if query_content_words:
            covered_words = sum(1 for w in query_content_words if w in response_clean)
            keyword_coverage = covered_words / len(query_content_words)
        else:
            keyword_coverage = 0.5
        
        # 2b. Bigram coverage
        if query_bigrams:
            covered_bigrams = sum(1 for bg in query_bigrams if bg in response_clean)
            bigram_coverage = covered_bigrams / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        # === 3. RESPONSE DEPTH ANALYSIS ===
        
        # 3a. Sentence count and variety in response
        response_sentences = [s.strip() for s in re.split(r'[.!?\n]+', response_raw) if s.strip() and len(s.strip()) > 5]
        num_response_sentences = len(response_sentences)
        
        # 3b. Unique content words in response (vocabulary richness)
        response_words = re.findall(r'\b[a-z]+\b', response_clean)
        response_content_words = [w for w in response_words if w not in stopwords and len(w) > 2]
        unique_response_content = set(response_content_words)
        vocab_richness = len(unique_response_content)
        
        # 3c. Information density: ratio of content words to total words
        total_response_words = len(response_words)
        if total_response_words > 0:
            info_density = len(response_content_words) / total_response_words
        else:
            info_density = 0.0
        
        # 3d. Explanation indicators (causal/elaborative connectors)
        explanation_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bhowever\b', r'\balthough\b', r'\bon the other hand\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bnamely\b', r'\bthat is\b', r'\bincluding\b',
            r'\bas a result\b', r'\bconsequently\b',
        ]
        explanation_count = 0
        for marker in explanation_markers:
            explanation_count += len(re.findall(marker, response_clean))
        
        # 3e. Detect if response has structural elements (numbered items, sections)
        structural_patterns = [
            r'\d+[\.\)]\s',  # numbered lists
            r'^[-*•]\s',     # bullet points (line start)
            r'\b(step|option|method|approach|way|point|reason|example)\s*\d',
        ]
        structural_score = 0
        for pat in structural_patterns:
            if re.search(pat, response_clean, re.MULTILINE):
                structural_score += 1
        
        # === 4. RELEVANCE vs NOISE ===
        
        # 4a. Check for repetition (sign of low quality)
        if num_response_sentences > 2:
            seen = set()
            duplicates = 0
            for sent in response_sentences:
                normalized = re.sub(r'\s+', ' ', sent.lower().strip())
                if normalized in seen:
                    duplicates += 1
                seen.add(normalized)
            repetition_penalty = duplicates / num_response_sentences
        else:
            repetition_penalty = 0.0
        
        # 4b. Check for off-topic drift (response contains lots of content unrelated to query)
        # Use a simple approach: ratio of query-relevant sentences
        if num_response_sentences > 0 and query_content_words:
            relevant_sentences = 0
            for sent in response_sentences:
                sent_lower = sent.lower()
                matches = sum(1 for w in query_content_words if w in sent_lower)
                if matches >= 1:
                    relevant_sentences += 1
            relevance_ratio = relevant_sentences / num_response_sentences
        else:
            relevance_ratio = 0.5
        
        # 4c. Check for truncation (response cut off mid-sentence)
        truncation_penalty = 0.0
        if response_raw and response_raw[-1] not in '.!?"\')]}':
            # Might be truncated
            last_chars = response_raw[-20:] if len(response_raw) > 20 else response_raw
            if not re.search(r'[.!?]\s*$', last_chars):
                truncation_penalty = 0.15
        
        # 4d. Check for garbage/irrelevant content patterns
        garbage_patterns = [
            r'<[a-z]+>.*?</[a-z]+>',  # HTML tags (unless query asks for HTML)
            r'import\s+\w+',           # code imports (unless query asks for code)
            r'def\s+\w+\(',            # function definitions
        ]
        
        query_asks_code = any(w in query_clean for w in ['code', 'python', 'function', 'program', 'html', 'tag', 'script'])
        garbage_score = 0
        if not query_asks_code:
            for pat in garbage_patterns:
                if re.search(pat, response_clean):
                    garbage_score += 1
        
        # === 5. PROPORTIONALITY ===
        # Response length should be proportional to query complexity
        
        query_complexity = num_sub_questions + len(set(tasks_found)) + len(query_content_words) / 5
        
        # Ideal response length (in words) based on query complexity
        ideal_min_words = max(10, int(query_complexity * 15))
        ideal_max_words = max(100, int(query_complexity * 200))
        
        if total_response_words < ideal_min_words:
            length_score = total_response_words / ideal_min_words
        elif total_response_words > ideal_max_words:
            # Slight penalty for being too long (might indicate rambling)
            length_score = max(0.7, 1.0 - (total_response_words - ideal_max_words) / (ideal_max_words * 3))
        else:
            length_score = 1.0
        
        # === 6. TASK COMPLETION CHECK ===
        # For specific task types, check if the response actually does what's asked
        
        task_completion = 1.0
        
        # If query asks for multiple items (e.g., "three ways", "list of")
        number_match = re.search(r'\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b', query_clean)
        if number_match:
            num_word = number_match.group(1)
            num_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                       'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
            if num_word in num_map:
                expected_count = num_map[num_word]
            else:
                try:
                    expected_count = int(num_word)
                except ValueError:
                    expected_count = None
            
            if expected_count:
                # Count distinct items in response (by numbered items or line breaks)
                items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s', response_raw)
                if not items:
                    # Try counting by sentences or Output: markers
                    items = re.findall(r'(?:Output:|^)\s*\S', response_raw, re.MULTILINE)
                actual_count = max(len(items), num_response_sentences)
                if expected_count > 0:
                    task_completion = min(1.0, actual_count / expected_count)
        
        # If query asks for identification/classification, check response has a clear answer
        if any(t in tasks_found for t in ['identify', 'classify', 'categorize', 'name']):
            # Response should contain definitive statements
            if total_response_words < 3:
                task_completion *= 0.3
        
        # If query asks yes/no, check response actually answers
        if query_types and all(qt == 'yes_no' for qt in query_types):
            has_answer = bool(re.search(r'\b(yes|no|it is|it isn\'t|it\'s not|absolutely|definitely|certainly|of course|sure)\b', response_clean))
            if not has_answer and total_response_words < 5:
                task_completion *= 0.3
        
        # === 7. RESPONSE SUBSTANCE CHECK ===
        # Penalize responses that are essentially empty or just echo the question
        
        substance_score = 1.0
        
        # Very short responses
        if total_response_words <= 1:
            substance_score = 0.05
        elif total_response_words <= 3:
            substance_score = 0.15
        elif total_response_words <= 5:
            substance_score = 0.3
        elif total_response_words <= 10:
            substance_score = 0.5
        
        # Check if response just echoes the query
        if query_content_words and response_content_words:
            query_set = set(query_content_words)
            response_set = set(response_content_words)
            new_info = response_set - query_set
            if len(response_set) > 0:
                novelty = len(new_info) / len(response_set)
            else:
                novelty = 0.0
        else:
            novelty = 0.5
        
        # === 8. COMPOSITE SCORING ===
        
        # Normalize sub-scores to 0-1 range
        
        # Coverage score (0-1)
        coverage = 0.6 * keyword_coverage + 0.4 * bigram_coverage
        
        # Depth score (0-1)
        sentence_depth = min(1.0, num_response_sentences / max(3, num_sub_questions * 3))
        vocab_depth = min(1.0, vocab_richness / max(15, len(query_content_words) * 5))
        explanation_depth = min(1.0, explanation_count / max(2, num_sub_questions * 2))
        structural_depth = min(1.0, structural_score / 2)
        
        depth = 0.35 * sentence_depth + 0.30 * vocab_depth + 0.20 * explanation_depth + 0.15 * structural_depth
        
        # Quality modifiers
        quality_modifier = 1.0
        quality_modifier -= repetition_penalty * 0.4
        quality_modifier -= truncation_penalty
        quality_modifier -= garbage_score * 0.15
        quality_modifier -= (1.0 - relevance_ratio) * 0.2
        quality_modifier = max(0.1, quality_modifier)
        
        # Combine all scores
        raw_score = (
            0.25 * coverage +
            0.25 * depth +
            0.15 * length_score +
            0.15 * task_completion +
            0.10 * substance_score +
            0.05 * info_density +
            0.05 * novelty
        )
        
        # Apply quality modifier
        raw_score *= quality_modifier
        
        # Scale to 0-10
        final_score = raw_score * 10.0
        
        # Apply substance floor/ceiling
        if substance_score < 0.2:
            final_score = min(final_score, 2.0)
        
        # Ensure bounds
        final_score = max(0.0, min(10.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 1)
    
    except Exception as e:
        # Fallback: simple length-based heuristic
        try:
            if not response or not response.strip():
                return 0.0
            words = len(response.strip().split())
            if words <= 1:
                return 0.5
            elif words <= 5:
                return 2.0
            elif words <= 20:
                return 4.0
            elif words <= 50:
                return 5.5
            else:
                return 6.0
        except:
            return 3.0