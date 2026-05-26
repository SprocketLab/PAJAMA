def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and information density approach. This variant focuses on:
    1. Query decomposition - identifying sub-questions/aspects the query asks about
    2. Response coverage mapping - checking how many query aspects are addressed
    3. Information density and depth signals
    4. Structural completeness (no truncation, proper conclusions)
    5. Example/evidence density
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
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # 1. QUERY DECOMPOSITION - Extract aspects/sub-questions
        # ============================================================
        
        # Extract question words and their associated phrases
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Identify distinct aspects the query is asking about
        # Look for conjunctions, commas, question marks that indicate multiple sub-questions
        query_aspects = []
        
        # Split query by conjunctions and punctuation to find sub-aspects
        aspect_splits = re.split(r'\band\b|\bor\b|\balso\b|,|;|\?|\.|!', query_lower)
        aspect_splits = [a.strip() for a in aspect_splits if len(a.strip()) > 3]
        
        # Extract key noun phrases from each aspect (simplified)
        query_content_words = set()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                      'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                      'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'if',
                      'while', 'about', 'what', 'which', 'who', 'whom', 'this', 'that',
                      'these', 'those', 'am', 'it', 'its', 'my', 'your', 'i', 'me', 'you',
                      'he', 'she', 'they', 'we', 'them', 'his', 'her', 'our', 'their',
                      'any', 'up', 'get', 'got', 'im', 'ive', 'dont', 'cant', 'wont'}
        
        query_words = re.findall(r'[a-z]+', query_lower)
        query_content_words = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        # Build bigrams from query for more precise matching
        query_bigrams = []
        for i in range(len(query_words) - 1):
            if query_words[i] not in stop_words or query_words[i+1] not in stop_words:
                query_bigrams.append(query_words[i] + ' ' + query_words[i+1])
        
        # ============================================================
        # 2. COVERAGE MAPPING - How many query aspects are addressed
        # ============================================================
        
        if query_content_words:
            # Check what fraction of query content words appear in response
            words_covered = sum(1 for w in query_content_words if w in response_lower)
            word_coverage_ratio = words_covered / len(query_content_words)
            score += word_coverage_ratio * 8  # up to 8 points
        
        if query_bigrams:
            bigrams_covered = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_coverage = bigrams_covered / len(query_bigrams)
            score += bigram_coverage * 5  # up to 5 points
        
        # ============================================================
        # 3. INFORMATION DENSITY AND DEPTH
        # ============================================================
        
        response_words = re.findall(r'[a-z]+', response_lower)
        response_word_count = len(response_words)
        
        # Unique word ratio (vocabulary richness) - indicates depth
        if response_word_count > 0:
            unique_words = set(response_words)
            vocab_richness = len(unique_words) / response_word_count
            # Sweet spot: not too repetitive, not too random
            # Typical good range: 0.4-0.7
            richness_score = min(vocab_richness / 0.55, 1.0) * 4
            score += richness_score
        
        # Count distinct factual/informational tokens
        # Numbers, percentages, measurements indicate specificity
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        num_count = min(len(numbers), 15)
        score += num_count * 0.4  # up to 6 points
        
        # Count specific/technical terms (words with more syllables tend to be more specific)
        def approx_syllables(word):
            word = word.lower()
            count = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e') and len(word) > 2:
                count -= 1
            return max(count, 1)
        
        technical_words = [w for w in response_words if approx_syllables(w) >= 3 and w not in stop_words]
        technical_density = len(technical_words) / max(response_word_count, 1)
        score += min(technical_density * 30, 6)  # up to 6 points
        
        # ============================================================
        # 4. STRUCTURAL COMPLETENESS SIGNALS
        # ============================================================
        
        # Check for proper sentence endings (not truncated)
        sentences = re.split(r'[.!?]\s', response)
        last_chars = response[-3:] if len(response) >= 3 else response
        
        # Truncation detection - penalize if response appears cut off
        truncation_penalty = 0
        if not re.search(r'[.!?:)\]}\*]$', response.rstrip()):
            truncation_penalty = 8  # significant penalty for truncation
        
        # Check if last sentence is very short (another truncation signal)
        if sentences and len(sentences[-1].strip()) < 15 and not re.search(r'[.!?]$', response.rstrip()):
            truncation_penalty += 3
        
        score -= truncation_penalty
        
        # Check for conclusion/summary signals
        conclusion_patterns = [
            r'\bin (?:summary|conclusion|short)\b',
            r'\boverall\b',
            r'\bto (?:sum up|summarize|conclude|wrap up)\b',
            r'\bin essence\b',
            r'\bultimately\b',
            r'\bhope (?:this|that|these)\b',
            r'\bgood luck\b',
            r'\bhappy \w+ing\b',
            r'\bfeel free\b',
            r'\blet me know\b',
        ]
        has_conclusion = any(re.search(p, response_lower) for p in conclusion_patterns)
        if has_conclusion:
            score += 3
        
        # ============================================================
        # 5. MULTI-PERSPECTIVE / REASONING DEPTH
        # ============================================================
        
        # Causal reasoning indicators
        causal_words = ['because', 'therefore', 'consequently', 'thus', 'hence',
                        'as a result', 'due to', 'since', 'leads to', 'causes',
                        'results in', 'reason', 'explains why', 'this means',
                        'in turn', 'so that', 'which means', 'the effect']
        causal_count = sum(1 for cw in causal_words if cw in response_lower)
        score += min(causal_count * 1.2, 6)  # up to 6 points
        
        # Contrast/nuance indicators (shows multiple perspectives)
        contrast_words = ['however', 'although', 'on the other hand', 'conversely',
                          'in contrast', 'nevertheless', 'while', 'whereas',
                          'despite', 'alternatively', 'instead', 'rather than',
                          'pros and cons', 'advantage', 'disadvantage', 'trade-off',
                          'downside', 'upside', 'limitation']
        contrast_count = sum(1 for cw in contrast_words if cw in response_lower)
        score += min(contrast_count * 1.5, 5)  # up to 5 points
        
        # ============================================================
        # 6. ENUMERATION AND EXAMPLE DENSITY
        # ============================================================
        
        # Count enumerated items (numbered lists, lettered lists)
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[.)\]]|\*|-|•|[a-z][.)\]])\s', response)
        enum_count = len(numbered_items)
        
        # Reward enumeration but with diminishing returns
        if enum_count > 0:
            score += min(math.log2(enum_count + 1) * 3, 8)  # up to ~8 points
        
        # Count examples/illustrations
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\blike\b(?:\s+\w+){1,3}',
            r'\bconsider\b', r'\bimagine\b', r'\bsuppose\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
        ]
        example_count = sum(len(re.findall(p, response_lower)) for p in example_patterns)
        score += min(example_count * 1.0, 5)  # up to 5 points
        
        # ============================================================
        # 7. RESPONSE LENGTH CALIBRATION
        # ============================================================
        
        # Longer responses tend to be more complete, but with diminishing returns
        # Use a logarithmic scale
        if response_word_count > 0:
            length_score = math.log(response_word_count + 1, 2)
            # Normalize: ~50 words = ~5.7, ~200 words = ~7.7, ~500 words = ~9.0
            length_contribution = min(length_score * 1.0, 9)
            score += length_contribution
        
        # ============================================================
        # 8. QUERY-TYPE SPECIFIC COMPLETENESS
        # ============================================================
        
        # Detect query type and check for type-specific completeness signals
        is_how_to = bool(re.search(r'\bhow (?:can|do|to|should|would|could)\b', query_lower))
        is_why = bool(re.search(r'\bwhy\b', query_lower))
        is_what = bool(re.search(r'\bwhat\b', query_lower))
        is_comparison = bool(re.search(r'\bvs\.?\b|\bversus\b|\bcompare\b|\bdifference\b|\bcontrast\b', query_lower))
        is_list_request = bool(re.search(r'\blist\b|\bideas\b|\bsuggestions\b|\btips\b|\bways\b|\bexamples\b|\boptions\b', query_lower))
        is_opinion = bool(re.search(r'\bdo you think\b|\bshould\b|\bopinion\b|\bbelieve\b', query_lower))
        
        # How-to: check for step-by-step structure
        if is_how_to:
            step_patterns = re.findall(r'step\s*\d|first|second|third|next|then|finally|lastly', response_lower)
            if len(step_patterns) >= 3:
                score += 4
            elif len(step_patterns) >= 1:
                score += 2
        
        # Why: check for explanatory depth
        if is_why:
            if causal_count >= 2:
                score += 3
        
        # Comparison: check for both sides
        if is_comparison:
            if contrast_count >= 2:
                score += 3
        
        # List request: reward more items
        if is_list_request:
            if enum_count >= 5:
                score += 4
            elif enum_count >= 3:
                score += 2
        
        # Opinion: check for reasoning/justification
        if is_opinion:
            if causal_count >= 1 and response_word_count > 50:
                score += 3
        
        # ============================================================
        # 9. FORMATTING QUALITY (as completeness signal)
        # ============================================================
        
        # Bold/emphasized terms indicate structured, thorough responses
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        score += min(bold_count * 0.3, 3)
        
        # Headers indicate organized, comprehensive coverage
        header_count = len(re.findall(r'(?:^|\n)#{1,4}\s', response))
        score += min(header_count * 1.0, 4)
        
        # ============================================================
        # 10. SEMANTIC COVERAGE via SENTENCE-TOPIC DIVERSITY
        # ============================================================
        
        # Split response into sentences and check topic diversity
        resp_sentences = re.split(r'[.!?]\s+', response)
        resp_sentences = [s.strip() for s in resp_sentences if len(s.strip()) > 10]
        
        if len(resp_sentences) >= 2:
            # Get content words per sentence
            sentence_topics = []
            for sent in resp_sentences:
                sent_words = set(re.findall(r'[a-z]+', sent.lower()))
                sent_content = sent_words - stop_words
                sentence_topics.append(sent_content)
            
            # Calculate average Jaccard distance between consecutive sentences
            # Higher distance = more diverse topics covered
            distances = []
            for i in range(len(sentence_topics) - 1):
                s1, s2 = sentence_topics[i], sentence_topics[i+1]
                if s1 or s2:
                    union = s1 | s2
                    intersection = s1 & s2
                    if union:
                        distances.append(1 - len(intersection) / len(union))
            
            if distances:
                avg_diversity = sum(distances) / len(distances)
                score += avg_diversity * 6  # up to 6 points
        
        # Sentence count as coverage signal
        sent_count = len(resp_sentences)
        if sent_count >= 3:
            score += min(math.log2(sent_count) * 1.5, 5)
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        
        # Clamp to [0, 100] range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        # Never crash - return neutral score
        return 25.0