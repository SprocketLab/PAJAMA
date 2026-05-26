def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response to a query.
    Returns a score where HIGHER = BETTER quality.
    
    This variant focuses on structural completeness analysis:
    - Query decomposition (identifying sub-questions/aspects)
    - Response depth and breadth metrics
    - Coverage of key query terms
    - Penalization of repetition, truncation, and superficiality
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        # Handle edge cases
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0  # Can't evaluate coverage without query
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        if len(response_stripped) == 0:
            return 0.0
        if len(response_stripped) < 5:
            return 0.5
        
        # ===== 1. RESPONSE LENGTH SCORE (0-15) =====
        # Longer responses tend to be more complete, with diminishing returns
        resp_len = len(response_stripped)
        word_count = len(response_stripped.split())
        
        if word_count <= 3:
            length_score = 0.5
        elif word_count <= 10:
            length_score = 2.0
        elif word_count <= 20:
            length_score = 5.0
        elif word_count <= 40:
            length_score = 8.0
        elif word_count <= 80:
            length_score = 11.0
        elif word_count <= 150:
            length_score = 13.0
        else:
            length_score = min(15.0, 13.0 + math.log(word_count / 150.0))
        
        # ===== 2. QUERY TERM COVERAGE (0-20) =====
        # How well does the response address the key terms from the query?
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'what', 'which', 'who', 'whom', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'give', 'given', 'following', 'input', 'describe', 'explain',
            'provide', 'write', 'create', 'generate', 'make', 'come', 'rewrite'
        }
        
        def tokenize(text):
            text_lower = text.lower()
            text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
            return [w for w in text_clean.split() if len(w) > 1]
        
        query_tokens = tokenize(query_stripped)
        response_tokens = tokenize(response_stripped)
        
        # Extract meaningful query terms (non-stop-words)
        query_content_words = [w for w in query_tokens if w not in stop_words]
        response_words_set = set(response_tokens)
        
        if len(query_content_words) > 0:
            covered = sum(1 for w in set(query_content_words) if w in response_words_set)
            coverage_ratio = covered / len(set(query_content_words))
        else:
            coverage_ratio = 0.5  # neutral if no content words
        
        coverage_score = coverage_ratio * 20.0
        
        # ===== 3. STRUCTURAL COMPLETENESS (0-15) =====
        # Check for structured elements that indicate thoroughness
        struct_score = 0.0
        
        # Sentences count
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences >= 2:
            struct_score += 3.0
        if num_sentences >= 3:
            struct_score += 2.0
        if num_sentences >= 5:
            struct_score += 2.0
        
        # Check for listing/enumeration
        has_list = bool(re.search(r'(\d+[.)]\s|\n[-•*]\s|\n\d+[.)]\s)', response_stripped))
        if has_list:
            struct_score += 2.0
        
        # Check for multiple clauses/commas (indicates elaboration)
        comma_count = response_stripped.count(',')
        if comma_count >= 2:
            struct_score += 1.5
        if comma_count >= 5:
            struct_score += 1.0
        
        # Check for transitional/elaboration words
        elaboration_words = [
            'also', 'additionally', 'furthermore', 'moreover', 'however',
            'for example', 'for instance', 'such as', 'including',
            'in addition', 'on the other hand', 'in contrast', 'while',
            'whereas', 'specifically', 'in particular', 'because',
            'therefore', 'consequently', 'as a result'
        ]
        resp_lower = response_stripped.lower()
        elab_count = sum(1 for ew in elaboration_words if ew in resp_lower)
        struct_score += min(3.5, elab_count * 1.0)
        
        struct_score = min(15.0, struct_score)
        
        # ===== 4. DEPTH AND SPECIFICITY (0-15) =====
        depth_score = 0.0
        
        # Unique word ratio (vocabulary richness)
        if len(response_tokens) > 0:
            unique_ratio = len(set(response_tokens)) / len(response_tokens)
        else:
            unique_ratio = 0
        
        depth_score += unique_ratio * 6.0
        
        # Longer words tend to be more specific/technical
        if len(response_tokens) > 0:
            avg_word_len = sum(len(w) for w in response_tokens) / len(response_tokens)
            depth_score += min(4.0, (avg_word_len - 3.0) * 1.5) if avg_word_len > 3.0 else 0
        
        # Check for specific details: numbers, proper nouns, technical terms
        has_numbers = bool(re.search(r'\d+', response_stripped))
        if has_numbers:
            depth_score += 1.0
        
        # Multi-word phrases that aren't just repetition
        if word_count > 10:
            # Count distinct bigrams
            bigrams = [(response_tokens[i], response_tokens[i+1]) for i in range(len(response_tokens)-1)]
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 0
            depth_score += bigram_diversity * 4.0
        
        depth_score = min(15.0, depth_score)
        
        # ===== 5. REPETITION PENALTY (0 to -15) =====
        repetition_penalty = 0.0
        
        # Word-level repetition
        if len(response_tokens) > 5:
            word_counts = Counter(response_tokens)
            # Remove stop words from repetition check
            content_tokens = [w for w in response_tokens if w not in stop_words]
            if content_tokens:
                content_counts = Counter(content_tokens)
                most_common_freq = content_counts.most_common(1)[0][1] if content_counts else 0
                total_content = len(content_tokens)
                if total_content > 0:
                    dominance = most_common_freq / total_content
                    if dominance > 0.3:
                        repetition_penalty -= (dominance - 0.3) * 20.0
        
        # Phrase-level repetition (repeated substrings)
        if len(response_stripped) > 30:
            # Check for repeated phrases of 4+ words
            words_list = response_stripped.lower().split()
            phrase_len = 4
            if len(words_list) >= phrase_len * 2:
                phrases = [' '.join(words_list[i:i+phrase_len]) for i in range(len(words_list) - phrase_len + 1)]
                phrase_counts = Counter(phrases)
                repeated_phrases = sum(1 for c in phrase_counts.values() if c > 1)
                if repeated_phrases > 0:
                    repetition_penalty -= min(8.0, repeated_phrases * 2.0)
        
        # Check for the exact same word repeated many times consecutively
        consecutive_repeat = re.findall(r'\b(\w+)(?:\s+\1){2,}', resp_lower)
        if consecutive_repeat:
            repetition_penalty -= len(consecutive_repeat) * 3.0
        
        repetition_penalty = max(-15.0, repetition_penalty)
        
        # ===== 6. TRUNCATION PENALTY (0 to -10) =====
        truncation_penalty = 0.0
        
        # Check if response seems truncated (ends mid-word or mid-sentence)
        if response_stripped and response_stripped[-1] not in '.!?"\')':
            # Might be truncated
            last_char = response_stripped[-1]
            if last_char.isalpha():
                truncation_penalty -= 5.0
            elif last_char == ',':
                truncation_penalty -= 3.0
        
        # Check for very abrupt ending patterns
        if re.search(r'\w+$', response_stripped) and not re.search(r'[.!?][\s"\']*$', response_stripped):
            # Doesn't end with punctuation
            if word_count > 15:  # Only penalize longer responses that seem cut off
                truncation_penalty -= 2.0
        
        # ===== 7. QUERY COMPLEXITY ALIGNMENT (0-10) =====
        # More complex queries should have longer, more detailed responses
        alignment_score = 0.0
        
        # Estimate query complexity
        query_word_count = len(query_stripped.split())
        
        # Count sub-tasks in query (indicated by 'and', commas in instructions, multiple verbs)
        action_verbs = ['describe', 'explain', 'compare', 'contrast', 'list', 'provide',
                       'create', 'write', 'generate', 'analyze', 'discuss', 'evaluate',
                       'summarize', 'identify', 'define', 'illustrate', 'crop', 'reduce',
                       'add', 'rewrite', 'come up']
        query_lower = query_stripped.lower()
        num_tasks = sum(1 for v in action_verbs if v in query_lower)
        num_tasks = max(1, num_tasks)
        
        # Check if query has "and" connecting tasks
        and_count = query_lower.count(' and ')
        num_tasks = max(num_tasks, and_count + 1)
        
        # Expected minimum word count based on complexity
        expected_min_words = num_tasks * 15
        
        if word_count >= expected_min_words:
            alignment_score = 8.0
        elif word_count >= expected_min_words * 0.5:
            alignment_score = 5.0
        else:
            alignment_score = max(0, 5.0 * (word_count / expected_min_words))
        
        # Bonus for multi-part queries that address multiple aspects
        if num_tasks >= 2:
            # Check if response seems to address multiple aspects
            paragraph_count = len([p for p in response_stripped.split('\n') if len(p.strip()) > 10])
            if paragraph_count >= 2 or num_sentences >= num_tasks + 1:
                alignment_score += 2.0
        
        alignment_score = min(10.0, alignment_score)
        
        # ===== 8. NOINPUT / EMPTY CONTENT PENALTY =====
        empty_penalty = 0.0
        if response_stripped.lower() in ['<noinput>', 'noinput', 'n/a', 'none', '']:
            return 0.5
        
        # ===== 9. COMPARATIVE/CONTRASTIVE COVERAGE BONUS (0-5) =====
        compare_bonus = 0.0
        if any(w in query_lower for w in ['compare', 'contrast', 'difference', 'similarities', 'vs']):
            # Query asks for comparison - check if response covers both sides
            comparison_words = ['while', 'whereas', 'however', 'on the other hand',
                              'in contrast', 'unlike', 'similarly', 'both', 'differ',
                              'different', 'same', 'similar']
            comp_count = sum(1 for cw in comparison_words if cw in resp_lower)
            compare_bonus = min(5.0, comp_count * 1.25)
        
        # ===== 10. EXAMPLE/EVIDENCE BONUS (0-5) =====
        example_bonus = 0.0
        example_indicators = ['for example', 'for instance', 'such as', 'e.g.',
                            'like ', 'including', 'specifically']
        ex_count = sum(1 for ei in example_indicators if ei in resp_lower)
        example_bonus = min(5.0, ex_count * 1.5)
        
        # ===== FINAL SCORE COMPUTATION =====
        total = (
            length_score +          # 0-15
            coverage_score +        # 0-20
            struct_score +          # 0-15
            depth_score +           # 0-15
            repetition_penalty +    # -15 to 0
            truncation_penalty +    # -10 to 0
            alignment_score +       # 0-10
            compare_bonus +         # 0-5
            example_bonus +         # 0-5
            empty_penalty           # 0
        )
        
        # Normalize to 0-100 range (max theoretical ~90)
        # Clamp to reasonable range
        total = max(0.0, min(100.0, total))
        
        return round(total, 2)
        
    except Exception as e:
        # Fallback: simple length-based score
        try:
            if response and len(response.strip()) > 0:
                return min(50.0, len(response.strip()) / 10.0)
            return 0.0
        except:
            return 0.0