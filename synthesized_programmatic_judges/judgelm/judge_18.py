def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a keyword/topic coverage
    approach combined with structural analysis of how thoroughly the response addresses
    the query components.
    
    This variant focuses on:
    1. Query decomposition into sub-questions/aspects
    2. Keyword coverage from query to response
    3. Information density and substantive content ratio
    4. Detection of non-answers, deflections, and off-topic content
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if not response or len(response) < 2:
            return 0.0
        
        # ---- Helper functions ----
        def tokenize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return [w for w in text.split() if len(w) > 1]
        
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'it', 'its', 'this', 'that',
            'these', 'those', 'he', 'she', 'they', 'them', 'his', 'her', 'their',
            'what', 'which', 'who', 'whom', 'me', 'my', 'we', 'our', 'you', 'your',
            'i', 'also', 'am', 'im', 'us', 'get', 'got', 'like'
        }
        
        def content_words(text):
            tokens = tokenize(text)
            return [w for w in tokens if w not in STOP_WORDS]
        
        # ---- 1. Query Aspect Extraction ----
        # Extract key content words from query as "aspects" to be covered
        query_content = content_words(query)
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        response_content = content_words(response)
        response_lower = response.lower()
        
        # ---- 2. Query keyword coverage ----
        # What fraction of query's content words appear in the response?
        if query_content:
            query_content_set = set(query_content)
            covered = sum(1 for w in query_content_set if w in response_lower)
            keyword_coverage = covered / len(query_content_set)
        else:
            keyword_coverage = 0.5  # neutral if no content words
        
        # ---- 3. Sub-question detection and coverage ----
        # Count question marks, conjunctions, list items in query
        question_marks = query.count('?')
        # Look for "and", "also", "additionally", enumeration patterns
        conjunction_words = ['and', 'also', 'additionally', 'moreover', 'furthermore', 'plus']
        sub_aspects = max(1, question_marks + sum(1 for w in query_tokens if w in conjunction_words))
        
        # Detect if query asks for multiple items (e.g., "three different ways", "list")
        multi_match = re.search(r'(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+\w+\s*(ways|things|items|examples|reasons|steps|points|methods|differences|similarities)', query.lower())
        requested_count = 0
        if multi_match:
            num_word = multi_match.group(1)
            num_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
            if num_word.isdigit():
                requested_count = int(num_word)
            else:
                requested_count = num_map.get(num_word, 0)
        
        # Count distinct items in response (sentences, bullet points, numbered items)
        response_sentences = [s.strip() for s in re.split(r'[.!?\n]', response) if s.strip() and len(s.strip()) > 5]
        bullet_items = re.findall(r'(?:^|\n)\s*(?:[-•*]|\d+[.):])\s*.+', response)
        
        multi_coverage_score = 1.0
        if requested_count > 0:
            # How many distinct items did the response provide?
            provided = max(len(bullet_items), len(response_sentences))
            multi_coverage_score = min(1.0, provided / requested_count)
        
        # ---- 4. Information density ----
        # Ratio of unique content words to total words (penalize repetition)
        if response_tokens:
            unique_content = set(response_content)
            total_tokens = len(response_tokens)
            info_density = len(unique_content) / max(1, total_tokens)
            # Normalize: typical good density is 0.2-0.5
            info_density_score = min(1.0, info_density / 0.35)
        else:
            info_density_score = 0.0
        
        # ---- 5. Response length adequacy ----
        resp_len = len(response)
        query_len = len(query)
        
        # Very short responses are usually incomplete
        if resp_len < 5:
            return 0.5
        
        # Length score: logarithmic scaling
        # Short responses get penalized, but we don't reward infinite length
        len_score = min(1.0, math.log(1 + resp_len) / math.log(1 + 500))
        
        # ---- 6. Non-answer / deflection detection ----
        non_answer_patterns = [
            r'^(no|yes|ok|okay|sure|maybe|idk|dunno)[\s.!?]*$',
            r'^\.+$',
            r'^\s*$',
        ]
        deflection_score = 1.0
        for pat in non_answer_patterns:
            if re.match(pat, response.strip(), re.IGNORECASE):
                deflection_score = 0.1
                break
        
        # Check if response is just the query repeated
        if response.strip().lower() == query.strip().lower():
            deflection_score = 0.1
        
        # ---- 7. Off-topic / garbage detection ----
        # Check for excessive code when not asked, random HTML, repetition
        off_topic_penalty = 1.0
        
        # Repetition detection: check if large chunks repeat
        if len(response) > 100:
            half = len(response) // 2
            first_half = response[:half]
            second_half = response[half:]
            # Simple repetition check
            if first_half == second_half:
                off_topic_penalty *= 0.4
            
            # Check for repeated phrases (n-gram repetition)
            if response_tokens:
                trigrams = [' '.join(response_tokens[i:i+3]) for i in range(len(response_tokens)-2)]
                if trigrams:
                    trigram_counts = Counter(trigrams)
                    most_common_count = trigram_counts.most_common(1)[0][1]
                    total_trigrams = len(trigrams)
                    if total_trigrams > 10 and most_common_count / total_trigrams > 0.15:
                        off_topic_penalty *= 0.5
        
        # Detect if response drifts into unrelated content (e.g., random questions/answers)
        # Count "Question:" or "Input:" or "Output:" patterns that suggest template bleeding
        template_bleed = len(re.findall(r'(?:Question|Input|Output)\s*:', response))
        if template_bleed > 2:
            off_topic_penalty *= 0.5
        
        # Check for excessive HTML/code when query doesn't ask for it
        query_asks_code = any(w in query.lower() for w in ['code', 'html', 'program', 'script', 'function', 'tag', 'css', 'javascript'])
        if not query_asks_code:
            code_chars = len(re.findall(r'[{}<>/\\=;]', response))
            if resp_len > 0 and code_chars / resp_len > 0.15:
                off_topic_penalty *= 0.6
        
        # ---- 8. Substantive content check ----
        # Does the response contain actual informational content?
        substantive_score = 1.0
        if len(response_content) < 3:
            substantive_score = 0.3
        elif len(response_content) < 6:
            substantive_score = 0.6
        
        # ---- 9. Sentence completeness ----
        # Check if response ends mid-sentence (truncation)
        truncation_penalty = 1.0
        if response and response[-1] not in '.!?"\')]}':
            # Might be truncated
            last_sentence = response_sentences[-1] if response_sentences else response
            if len(last_sentence) > 30:
                # Likely truncated mid-sentence
                truncation_penalty = 0.85
        
        # ---- 10. Coverage breadth via distinct topic words ----
        # More unique content words = broader coverage
        unique_resp_content = set(response_content)
        breadth_score = min(1.0, len(unique_resp_content) / 25.0)
        
        # ---- Combine scores with weights ----
        # Weights tuned for completeness/coverage focus
        score = (
            keyword_coverage * 1.5 +       # Query keyword coverage
            multi_coverage_score * 1.5 +    # Multi-item request coverage
            info_density_score * 1.0 +      # Information density
            len_score * 1.5 +               # Length adequacy
            deflection_score * 2.0 +        # Non-answer penalty
            off_topic_penalty * 1.5 +       # Off-topic/garbage penalty
            substantive_score * 1.5 +       # Substantive content
            truncation_penalty * 0.5 +      # Truncation penalty
            breadth_score * 1.0             # Coverage breadth
        )
        
        # Max possible raw score: 1.5 + 1.5 + 1.0 + 1.5 + 2.0 + 1.5 + 1.5 + 0.5 + 1.0 = 12.0
        # Normalize to 0-10 scale
        max_raw = 12.0
        normalized = (score / max_raw) * 10.0
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0