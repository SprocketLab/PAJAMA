def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a novel approach:
    - Question decomposition: identifies sub-questions/aspects in the query
    - Response segmentation into distinct "information units"
    - Coverage mapping: how many query aspects are addressed
    - Depth analysis: recursive detail detection (examples, explanations, caveats)
    - Gap detection: identifies missing expected content patterns
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # 1. QUERY DECOMPOSITION - identify aspects/sub-questions
        # ============================================================
        
        # Extract question words and their contexts to identify sub-questions
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Count explicit questions in query
        question_marks = query.count('?')
        num_sub_questions = max(1, question_marks)
        
        # Identify query "aspects" via noun phrases and key terms
        # Remove common stop words to find content words
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
            'they', 'them', 'this', 'that', 'these', 'those', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'shall', 'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'not', 'so',
            'if', 'then', 'than', 'too', 'very', 'just', 'about', 'above',
            'after', 'again', 'all', 'also', 'am', 'any', 'because', 'before',
            'between', 'both', 'by', 'come', 'each', 'few', 'for', 'from',
            'get', 'got', 'her', 'here', 'him', 'his', 'how', 'in', 'into',
            'its', 'let', 'like', 'make', 'many', 'more', 'most', 'much',
            'must', 'no', 'of', 'off', 'on', 'one', 'only', 'other', 'own',
            'per', 'put', 'said', 'same', 'see', 'some', 'still', 'such',
            'take', 'tell', 'their', 'to', 'up', 'us', 'use', 'want', 'way',
            'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
            'with', 'yet', 'need', 'think', 'know', 'help', 'bit', 'im',
            'dont', "don't", "i'm", "i've", 'ive', 'wanna', 'gonna'
        }
        
        query_words = re.findall(r'[a-z]+', query_lower)
        query_content_words = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        # Build bigrams from query for aspect detection
        query_bigrams = []
        for i in range(len(query_words) - 1):
            if query_words[i] not in stop_words or query_words[i+1] not in stop_words:
                query_bigrams.append(query_words[i] + ' ' + query_words[i+1])
        
        # ============================================================
        # 2. INFORMATION UNIT EXTRACTION from response
        # ============================================================
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Identify distinct "information units" - sentences that introduce new info
        # Track unique content per sentence
        seen_content = set()
        information_units = 0
        for sent in sentences:
            sent_words = set(re.findall(r'[a-z]+', sent.lower())) - stop_words
            new_words = sent_words - seen_content
            if len(new_words) >= 2:  # at least 2 new content words = new info unit
                information_units += 1
            seen_content.update(sent_words)
        
        # ============================================================
        # 3. QUERY ASPECT COVERAGE
        # ============================================================
        
        # Check how many query content words appear in response
        if query_content_words:
            covered_words = sum(1 for w in query_content_words if w in response_lower)
            word_coverage_ratio = covered_words / len(query_content_words)
        else:
            word_coverage_ratio = 0.5
        
        # Check bigram coverage
        if query_bigrams:
            covered_bigrams = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_coverage = covered_bigrams / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        # ============================================================
        # 4. DEPTH ANALYSIS - look for explanatory patterns
        # ============================================================
        
        # Causal/explanatory connectors (showing depth of explanation)
        explanation_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bas a result\b', r'\bdue to\b', r'\bthis means\b', r'\bin order to\b',
            r'\bthe reason\b', r'\bwhich means\b', r'\bconsequently\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\billustrat', r'\bconsider\b', r'\bimagine\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\balthough\b',
            r'\bwhile\b', r'\bconversely\b', r'\bin contrast\b',
            r'\badditionally\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\balso\b', r'\banother\b', r'\bin addition\b',
        ]
        
        explanation_count = 0
        for pat in explanation_patterns:
            explanation_count += len(re.findall(pat, response_lower))
        
        # Normalize explanation density
        response_word_count = len(response_lower.split())
        if response_word_count > 0:
            explanation_density = explanation_count / (response_word_count / 100.0)
        else:
            explanation_density = 0
        
        # ============================================================
        # 5. STRUCTURAL COMPLETENESS INDICATORS
        # ============================================================
        
        # Numbered/ordered items (indicates systematic coverage)
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*\s|-\s|•)', response)
        num_list_items = len(numbered_items)
        
        # Bold/emphasized items (markdown headers or bold text)
        bold_items = re.findall(r'\*\*[^*]+\*\*', response)
        num_bold = len(bold_items)
        
        # Section markers (###, etc.)
        section_headers = re.findall(r'#{1,4}\s+', response)
        num_sections = len(section_headers)
        
        # ============================================================
        # 6. EDGE CASE AND CAVEAT COVERAGE
        # ============================================================
        
        caveat_patterns = [
            r'\bhowever\b', r'\bbut\b', r'\bnote that\b', r'\bkeep in mind\b',
            r'\bbe aware\b', r'\bcaution\b', r'\bwarning\b', r'\bimportant\b',
            r'\bexception\b', r'\bunless\b', r'\bdepending on\b', r'\bvaries\b',
            r'\bif you\b', r'\bin case\b', r'\balternative\b', r'\boption\b',
            r'\bconsider\b', r'\bmake sure\b', r'\bdon\'t forget\b',
            r'\btip[s]?\b', r'\bpro tip\b', r'\btroubleshoot\b',
        ]
        
        caveat_count = 0
        for pat in caveat_patterns:
            caveat_count += len(re.findall(pat, response_lower))
        
        # ============================================================
        # 7. RESPONSE COMPLETENESS SIGNALS
        # ============================================================
        
        # Check if response appears truncated (ends mid-sentence)
        truncation_penalty = 0.0
        stripped_response = response.rstrip()
        if stripped_response:
            last_char = stripped_response[-1]
            # Truncated if ends without proper punctuation
            if last_char not in '.!?"\')]}':
                truncation_penalty = -3.0
            # Also check if the last sentence seems incomplete
            last_sentence = sentences[-1] if sentences else ""
            last_words = last_sentence.split()
            if len(last_words) > 0 and last_words[-1].lower() in (
                'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'for',
                'with', 'that', 'this', 'is', 'are', 'was', 'were', 'be', 'can',
                'will', 'would', 'should', 'could', 'may', 'might'
            ):
                truncation_penalty = -4.0
        
        # Check for conclusion/summary (indicates completeness)
        has_conclusion = bool(re.search(
            r'\b(in summary|in conclusion|to summarize|overall|to sum up|finally|'
            r'in short|the key|remember|good luck|happy|enjoy|hope this helps)\b',
            response_lower
        ))
        
        # Check for introduction/framing
        has_intro = bool(re.search(
            r'^.{0,200}(here are|here\'s|let me|let\'s|great question|'
            r'certainly|absolutely|of course|sure|there are several|'
            r'this is a|that\'s a)',
            response_lower
        ))
        
        # ============================================================
        # 8. SEMANTIC DENSITY - unique concepts per unit length
        # ============================================================
        
        response_words = re.findall(r'[a-z]+', response_lower)
        if response_words:
            # Count unique "content" words
            content_words = [w for w in response_words if w not in stop_words and len(w) > 2]
            unique_content = len(set(content_words))
            total_content = len(content_words)
            
            if total_content > 0:
                # Vocabulary richness (type-token ratio, adjusted for length)
                ttr = unique_content / math.sqrt(total_content)
            else:
                ttr = 0
        else:
            ttr = 0
            unique_content = 0
        
        # ============================================================
        # 9. MULTI-PERSPECTIVE CHECK
        # ============================================================
        
        # Does the response address multiple angles/perspectives?
        perspective_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bone\b.*\btwo\b', r'\bon one hand\b', r'\bon the other\b',
            r'\balternatively\b', r'\banother\b', r'\bin addition\b',
            r'\bstep \d\b', r'\bphase \d\b', r'\bpart \d\b',
            r'\bmethod\b', r'\bapproach\b', r'\boption\b',
        ]
        
        perspective_count = 0
        for pat in perspective_markers:
            if re.search(pat, response_lower):
                perspective_count += 1
        
        # ============================================================
        # 10. SPECIFICITY SCORE - concrete details vs vague statements
        # ============================================================
        
        # Look for specific/concrete indicators
        specific_patterns = [
            r'\d+',  # numbers
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # proper nouns (multi-word)
            r'"[^"]+"',  # quoted text
            r'\b(?:e\.g\.|i\.e\.|etc\.)\b',  # abbreviations indicating examples
            r'\b(?:approximately|about|around|roughly)\s+\d',  # quantified estimates
        ]
        
        specificity_count = 0
        for pat in specific_patterns:
            specificity_count += len(re.findall(pat, response))
        
        # ============================================================
        # SCORING FORMULA
        # ============================================================
        
        # Length score (log scale, diminishing returns)
        length_score = min(2.5, math.log(max(1, response_word_count) / 30.0 + 1) * 1.5)
        
        # Information units score
        info_unit_score = min(2.0, information_units * 0.15)
        
        # Query coverage score
        coverage_score = (word_coverage_ratio * 1.5 + bigram_coverage * 0.5)
        
        # Explanation depth score
        depth_score = min(2.0, explanation_density * 0.25)
        
        # Structure score
        structure_score = min(1.5, (
            min(0.5, num_list_items * 0.06) +
            min(0.5, num_bold * 0.07) +
            min(0.5, num_sections * 0.15)
        ))
        
        # Caveat/edge case score
        caveat_score = min(1.0, caveat_count * 0.12)
        
        # Completeness signals
        completeness_score = 0.0
        if has_conclusion:
            completeness_score += 0.5
        if has_intro:
            completeness_score += 0.3
        
        # Vocabulary richness
        richness_score = min(1.0, ttr * 0.12)
        
        # Multi-perspective score
        perspective_score = min(1.0, perspective_count * 0.15)
        
        # Specificity score
        spec_score = min(0.8, specificity_count * 0.06)
        
        # Sentence count contribution (more sentences = more thorough, with diminishing returns)
        sentence_score = min(1.0, math.log(max(1, num_sentences) + 1) * 0.3)
        
        # Combine all scores
        total = (
            length_score +          # 0-2.5
            info_unit_score +       # 0-2.0
            coverage_score +        # 0-2.0
            depth_score +           # 0-2.0
            structure_score +       # 0-1.5
            caveat_score +          # 0-1.0
            completeness_score +    # 0-0.8
            richness_score +        # 0-1.0
            perspective_score +     # 0-1.0
            spec_score +            # 0-0.8
            sentence_score +        # 0-1.0
            truncation_penalty      # -4.0 to 0
        )
        
        # Clamp to 0-15 range
        total = max(0.0, min(15.0, total))
        
        # Scale to 0-10
        final_score = total * (10.0 / 15.0)
        
        return round(final_score, 3)
    
    except Exception:
        return 0.0