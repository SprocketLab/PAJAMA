def judging_function(query, response):
    """
    Evaluate response completeness and coverage.
    Uses a multi-signal approach focusing on:
    1. Response substance relative to query complexity
    2. Coverage of query terms and concepts
    3. Structural completeness indicators
    4. Penalization of garbage/irrelevant content
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
            return 3.0  # Can't assess relevance without query
        
        query = query.strip()
        response = response.strip()
        
        if len(response) == 0:
            return 0.0
        if len(response) <= 3:
            return 0.5
        
        # =====================
        # 1. RESPONSE LENGTH & SUBSTANCE SCORE
        # =====================
        resp_len = len(response)
        resp_words = response.split()
        num_resp_words = len(resp_words)
        query_words = query.split()
        num_query_words = len(query_words)
        
        # Meaningful length score (logarithmic scaling, diminishing returns)
        # Short responses are penalized heavily for completeness
        if num_resp_words <= 1:
            length_score = 0.5
        elif num_resp_words <= 5:
            length_score = 1.5
        elif num_resp_words <= 15:
            length_score = 2.0 + (num_resp_words - 5) * 0.15
        elif num_resp_words <= 50:
            length_score = 3.5 + (num_resp_words - 15) * 0.06
        elif num_resp_words <= 150:
            length_score = 5.6 + (num_resp_words - 50) * 0.02
        else:
            length_score = 7.6 + min(1.4, math.log(num_resp_words / 150) * 0.8)
        
        # Cap length score
        length_score = min(9.0, length_score)
        
        # =====================
        # 2. QUERY CONCEPT COVERAGE
        # =====================
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'it', 'its', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'also', 'make', 'please', 'identify',
            'create', 'write', 'tell', 'give', 'find', 'know', 'want', 'need',
            'like', 'get', 'got', 'many', 'much'
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-zA-Z]+', text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        query_content = extract_content_words(query)
        resp_lower = response.lower()
        
        if query_content:
            covered = sum(1 for w in query_content if w in resp_lower)
            coverage_ratio = covered / len(query_content)
        else:
            coverage_ratio = 0.5  # neutral if no content words
        
        coverage_score = coverage_ratio * 10.0
        
        # =====================
        # 3. QUESTION DETECTION & ANSWER COMPLETENESS
        # =====================
        # Detect question types in query
        question_indicators = re.findall(
            r'\b(what|where|when|who|why|how|which|can|could|is it|are there|do you|does|list|name|identify|describe|explain|compare|rewrite|create|generate)\b',
            query.lower()
        )
        num_questions = max(1, len(set(question_indicators)))
        
        # Check if response contains multiple sub-parts (lists, sentences, paragraphs)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Check for list items or structured content
        list_items = re.findall(r'(?:^|\n)\s*[-•*\d]+[.)]\s', response)
        has_structure = len(list_items) > 0
        
        # Multi-part query detection
        sub_questions = re.split(r'[.?!;]\s+|,\s+(?:and|also|plus)\s+', query)
        sub_questions = [sq for sq in sub_questions if len(sq.strip()) > 5]
        num_sub_parts = max(1, len(sub_questions))
        
        # Depth score based on sentence count relative to query complexity
        if num_sentences == 0:
            depth_score = 1.0
        elif num_sentences == 1:
            depth_score = 3.0
        else:
            depth_ratio = num_sentences / num_sub_parts
            depth_score = min(9.0, 3.0 + depth_ratio * 1.5)
        
        if has_structure:
            depth_score = min(10.0, depth_score + 1.0)
        
        # =====================
        # 4. GARBAGE / IRRELEVANCE PENALTIES
        # =====================
        penalty = 0.0
        
        # Repetition detection
        if num_resp_words > 10:
            word_counter = Counter(extract_content_words(response))
            if word_counter:
                most_common_freq = word_counter.most_common(1)[0][1]
                unique_content = len(word_counter)
                if unique_content > 0:
                    rep_ratio = most_common_freq / max(1, num_resp_words)
                    if rep_ratio > 0.15:
                        penalty += 1.5
        
        # Detect repeated phrases (copy-paste patterns)
        resp_chunks = [response[i:i+50] for i in range(0, len(response)-50, 25)]
        if len(resp_chunks) > 2:
            chunk_counter = Counter(resp_chunks)
            repeated_chunks = sum(1 for c, n in chunk_counter.items() if n > 1)
            if repeated_chunks > 1:
                penalty += min(3.0, repeated_chunks * 0.8)
        
        # Detect HTML/code when not asked for it
        query_asks_code = bool(re.search(r'\b(html|code|program|script|tag|function)\b', query.lower()))
        has_code = bool(re.search(r'<[a-zA-Z]+|import\s+\w+|def\s+\w+|class\s+\w+', response))
        if has_code and not query_asks_code:
            penalty += 2.5
        
        # Detect if response is mostly questions back (not answering)
        resp_questions = len(re.findall(r'\?', response))
        if resp_questions > 3 and num_sentences > 0:
            q_ratio = resp_questions / max(1, num_sentences)
            if q_ratio > 0.5:
                penalty += 2.0
        
        # Detect "Input:/Output:" patterns suggesting template garbage
        template_patterns = len(re.findall(r'(?:Input:|Output:|Question:|Answer:)', response))
        if template_patterns > 3:
            penalty += min(3.0, template_patterns * 0.5)
        
        # Very short response for complex query
        if num_query_words > 15 and num_resp_words < 5:
            penalty += 2.0
        elif num_query_words > 10 and num_resp_words < 3:
            penalty += 2.5
        
        # Single word or trivially short
        if num_resp_words <= 2:
            penalty += 3.0
        elif num_resp_words <= 4:
            penalty += 1.5
        
        # Response starts with just a period or is nonsensical
        if response.strip() in ['.', '..', '...', '-', '--']:
            return 0.5
        
        # Detect trailing garbage (response continues with unrelated content)
        # Check if second half diverges from query topic
        if num_resp_words > 30:
            first_half_words = resp_words[:num_resp_words//2]
            second_half_words = resp_words[num_resp_words//2:]
            first_half_text = ' '.join(first_half_words).lower()
            second_half_text = ' '.join(second_half_words).lower()
            
            if query_content:
                first_coverage = sum(1 for w in query_content if w in first_half_text) / len(query_content)
                second_coverage = sum(1 for w in query_content if w in second_half_text) / len(query_content)
                
                # If second half has much less relevance, it might be trailing garbage
                if first_coverage > 0.3 and second_coverage < 0.1:
                    penalty += 1.0
        
        # =====================
        # 5. RELEVANCE BONUS
        # =====================
        # Check for presence of key expected answer elements
        relevance_bonus = 0.0
        
        # If query asks for specific things (e.g., "three different ways", "list")
        number_match = re.search(r'\b(three|3|two|2|four|4|five|5)\b', query.lower())
        if number_match:
            requested_num_map = {
                'two': 2, '2': 2, 'three': 3, '3': 3,
                'four': 4, '4': 4, 'five': 5, '5': 5
            }
            requested_num = requested_num_map.get(number_match.group(1), 3)
            # Count distinct items in response (lines, numbered items, etc.)
            lines = [l.strip() for l in response.split('\n') if len(l.strip()) > 5]
            if len(lines) >= requested_num:
                relevance_bonus += 1.5
            elif len(lines) < requested_num and len(lines) > 0:
                relevance_bonus -= 0.5
        
        # Direct answer indicators
        if re.search(r'\b(because|therefore|thus|the reason|this is due to|as a result)\b', resp_lower):
            relevance_bonus += 0.5
        
        # Explanation depth
        if re.search(r'\b(however|although|moreover|furthermore|additionally|in addition)\b', resp_lower):
            relevance_bonus += 0.5
        
        # =====================
        # 6. COMBINE SCORES
        # =====================
        # Weighted combination
        raw_score = (
            length_score * 0.25 +
            coverage_score * 0.30 +
            depth_score * 0.25 +
            relevance_bonus
        )
        
        # Apply penalty
        final_score = raw_score - penalty
        
        # Normalize to 0-10 range
        final_score = max(0.5, min(10.0, final_score))
        
        # Additional calibration: very short responses rarely deserve high scores
        # for completeness
        if num_resp_words <= 5 and final_score > 4.0:
            final_score = 4.0
        if num_resp_words <= 10 and final_score > 6.0:
            final_score = 6.0
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except:
            return 2.0