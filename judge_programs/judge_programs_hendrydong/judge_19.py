def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response.
    
    This variant focuses on:
    1. Query decomposition - identifying question components and checking coverage
    2. Information density - ratio of unique informational tokens to total tokens
    3. Specificity markers - concrete details, examples, evidence
    4. Structural completeness - introduction, body, conclusion patterns
    5. Topic keyword coverage from query
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        if not query or not query.strip():
            return 2.0
        
        query = query.strip()
        response = response.strip()
        
        # ============================================================
        # 1. QUERY DECOMPOSITION & COVERAGE ANALYSIS
        # ============================================================
        # Extract question fragments and key topics from the query
        
        # Find explicit questions (sentences ending with ?)
        query_sentences = re.split(r'[.!?\n]+', query)
        question_marks = query.count('?')
        num_questions = max(question_marks, 1)
        
        # Extract meaningful content words from query (stopword-filtered)
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
            've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
            'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
            'won', 'wouldn', 'could', 'would', 'might', 'must', 'shall', 'may', 'like',
            'also', 'much', 'many', 'even', 'still', 'already', 'since', 'into', 'really',
            'seem', 'seems', 'seemed', 'think', 'know', 'get', 'got', 'going', 'go',
            'make', 'made', 'say', 'said', 'take', 'took', 'come', 'came', 'see', 'saw',
            'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'try', 'call', 'need',
            'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'show', 'hear',
            'play', 'run', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'provide',
            'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn',
            'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
            'read', 'spend', 'grow', 'open', 'walk', 'win', 'teach', 'offer', 'remember',
            'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build',
            'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'im', 'ive', 'dont', 'doesnt',
            'thats', 'its', 'been', 'would', 'any', 'has', 'had', 'have', 'was', 'were',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-zA-Z]{3,}', text.lower())
            return [w for w in words if w not in stopwords]
        
        query_content_words = extract_content_words(query)
        response_content_words = extract_content_words(response)
        
        # Query topic coverage: what fraction of query's content words appear in response
        query_word_set = set(query_content_words)
        response_word_set = set(response_content_words)
        
        if query_word_set:
            topic_coverage = len(query_word_set & response_word_set) / len(query_word_set)
        else:
            topic_coverage = 0.5
        
        # ============================================================
        # 2. INFORMATION DENSITY & RICHNESS
        # ============================================================
        
        response_words = re.findall(r'\S+', response)
        response_word_count = len(response_words)
        
        if response_word_count == 0:
            return 0.0
        
        # Unique content word ratio (type-token ratio on content words)
        if response_content_words:
            content_counter = Counter(response_content_words)
            unique_content = len(content_counter)
            total_content = len(response_content_words)
            # Adjusted TTR (log-based to handle length variation)
            if total_content > 1:
                info_density = unique_content / math.log2(total_content + 1)
            else:
                info_density = unique_content
        else:
            info_density = 0
            unique_content = 0
            total_content = 0
        
        # Normalize info density (typical range 5-30)
        info_density_score = min(info_density / 20.0, 1.0)
        
        # ============================================================
        # 3. SPECIFICITY MARKERS
        # ============================================================
        
        # Count concrete specificity indicators
        specificity_count = 0
        
        # Numbers and quantities
        numbers = re.findall(r'\b\d+[\d,.]*\b', response)
        specificity_count += min(len(numbers), 8)
        
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response)
        specificity_count += min(len(proper_nouns), 8)
        
        # Quoted terms or technical terms
        quoted = re.findall(r'["\*\'`][\w\s]+["\*\'`]', response)
        specificity_count += min(len(quoted), 5)
        
        # Parenthetical clarifications
        parens = re.findall(r'\([^)]+\)', response)
        specificity_count += min(len(parens), 5)
        
        # Example/evidence markers
        example_markers = re.findall(
            r'\b(?:for example|for instance|e\.g\.|such as|specifically|in particular|'
            r'notably|consider|imagine|suppose|case in point|evidence|research|study|'
            r'according to|data shows|historically)\b',
            response.lower()
        )
        specificity_count += len(example_markers) * 2
        
        # Causal/explanatory connectors (shows depth of reasoning)
        causal_markers = re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as a result|this means|'
            r'this leads to|the reason|due to|caused by|resulting in|implies|suggests)\b',
            response.lower()
        )
        specificity_count += len(causal_markers)
        
        # Contrast/nuance markers (shows consideration of multiple angles)
        contrast_markers = re.findall(
            r'\b(?:however|although|on the other hand|conversely|while|whereas|'
            r'nevertheless|nonetheless|despite|in contrast|alternatively|'
            r'that said|having said that|but also|yet|though)\b',
            response.lower()
        )
        specificity_count += len(contrast_markers) * 1.5
        
        # Normalize specificity (typical range 0-30)
        specificity_score = min(specificity_count / 20.0, 1.0)
        
        # ============================================================
        # 4. STRUCTURAL COMPLETENESS
        # ============================================================
        
        # Check for multi-faceted response structure
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Response length adequacy relative to query complexity
        # More questions / longer query = need longer response
        query_word_count = len(re.findall(r'\S+', query))
        expected_min_words = max(30, query_word_count * 0.5 + num_questions * 20)
        length_adequacy = min(response_word_count / expected_min_words, 2.0) / 2.0
        
        # Paragraph structure
        paragraphs = [p.strip() for p in response.split('\n') if p.strip() and len(p.strip()) > 20]
        num_paragraphs = max(len(paragraphs), 1)
        
        # Structural variety score
        has_code_blocks = bool(re.search(r'```', response))
        has_lists = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response)) or bool(re.search(r'(?:^|\n)\s*\d+[.)]\s', response))
        has_emphasis = bool(re.search(r'[*_]{1,2}\w+', response))
        
        structural_variety = (
            (0.3 if num_paragraphs >= 2 else 0.1) +
            (0.2 if has_lists else 0.0) +
            (0.1 if has_code_blocks else 0.0) +
            (0.1 if has_emphasis else 0.0) +
            (0.3 if num_sentences >= 3 else 0.15 if num_sentences >= 2 else 0.05)
        )
        
        # ============================================================
        # 5. MULTI-ASPECT COVERAGE DETECTION
        # ============================================================
        
        # Detect if response addresses multiple distinct aspects
        # Use sentence-level topic shifts as a proxy
        
        if len(sentences) >= 2:
            sentence_word_sets = []
            for s in sentences:
                s_words = set(extract_content_words(s))
                sentence_word_sets.append(s_words)
            
            # Measure average pairwise Jaccard distance between sentences
            # Higher distance = more diverse topics covered
            if len(sentence_word_sets) >= 2:
                distances = []
                for i in range(min(len(sentence_word_sets), 10)):
                    for j in range(i + 1, min(len(sentence_word_sets), 10)):
                        s1, s2 = sentence_word_sets[i], sentence_word_sets[j]
                        union = s1 | s2
                        if union:
                            jaccard = 1.0 - len(s1 & s2) / len(union)
                            distances.append(jaccard)
                
                if distances:
                    avg_diversity = sum(distances) / len(distances)
                else:
                    avg_diversity = 0.0
            else:
                avg_diversity = 0.0
        else:
            avg_diversity = 0.0
        
        # Diversity score: moderate diversity is best (too high might mean incoherent)
        # Sweet spot around 0.6-0.8
        diversity_score = min(avg_diversity / 0.7, 1.0)
        
        # ============================================================
        # 6. QUESTION-ADDRESSING HEURISTIC
        # ============================================================
        
        # Extract sub-questions from query
        query_questions = re.findall(r'[^.!?\n]*\?', query)
        
        # For each question, check if response seems to address it
        # by checking for keyword overlap
        questions_addressed = 0
        if query_questions:
            for q in query_questions:
                q_words = set(extract_content_words(q))
                if q_words:
                    overlap = len(q_words & response_word_set) / len(q_words)
                    if overlap >= 0.25:
                        questions_addressed += 1
            question_coverage = questions_addressed / len(query_questions)
        else:
            # No explicit questions - use topic coverage as proxy
            question_coverage = topic_coverage
        
        # ============================================================
        # 7. DEPTH INDICATORS
        # ============================================================
        
        # Average sentence length (longer sentences often = more complex ideas)
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            avg_sentence_len = response_word_count
        
        # Normalize: sweet spot 12-25 words per sentence
        if avg_sentence_len < 5:
            sentence_depth = 0.2
        elif avg_sentence_len < 12:
            sentence_depth = 0.4 + 0.3 * (avg_sentence_len - 5) / 7
        elif avg_sentence_len <= 25:
            sentence_depth = 0.7 + 0.3 * (avg_sentence_len - 12) / 13
        else:
            sentence_depth = max(0.7, 1.0 - (avg_sentence_len - 25) * 0.02)
        
        # Multi-sentence depth: having enough sentences to develop ideas
        sentence_count_score = min(num_sentences / 5.0, 1.0)
        
        # Conditional/qualifying language (shows nuanced thinking)
        qualifying_patterns = re.findall(
            r'\b(?:depends|it depends|in some cases|sometimes|often|typically|generally|'
            r'usually|tend to|can be|may be|might be|could be|varies|context|'
            r'nuance|complex|complicated|trade-off|tradeoff|on one hand|'
            r'important to note|worth noting|keep in mind|caveat)\b',
            response.lower()
        )
        nuance_score = min(len(qualifying_patterns) / 4.0, 1.0)
        
        # ============================================================
        # 8. PENALIZE LOW-EFFORT / DEFLECTING RESPONSES
        # ============================================================
        
        deflection_penalty = 0.0
        
        # Check for pure deflection / non-answers
        deflection_patterns = [
            r'\bplease read\b',
            r'\bcheck the (wiki|faq|sidebar)\b',
            r'\bgoogle it\b',
            r'\bjust (google|search)\b',
            r'\bwelcome to\b.*\bplease read\b',
            r'\byou might be interested in\b',
            r'\bwhile you wait\b',
        ]
        for pat in deflection_patterns:
            if re.search(pat, response.lower()):
                deflection_penalty += 0.15
        
        # Very short responses for complex queries
        if response_word_count < 20 and query_word_count > 50:
            deflection_penalty += 0.2
        
        # Single sentence response to multi-question query
        if num_sentences <= 1 and num_questions >= 2:
            deflection_penalty += 0.15
        
        deflection_penalty = min(deflection_penalty, 0.5)
        
        # ============================================================
        # FINAL SCORING
        # ============================================================
        
        # Weighted combination
        raw_score = (
            topic_coverage * 1.5 +        # How well query topics are covered
            question_coverage * 2.0 +      # How many sub-questions addressed
            info_density_score * 1.0 +     # Richness of information
            specificity_score * 2.0 +      # Concrete details and examples
            structural_variety * 1.0 +     # Good formatting/structure
            diversity_score * 1.0 +        # Multiple aspects covered
            length_adequacy * 1.5 +        # Sufficient length for query
            sentence_depth * 0.8 +         # Sentence-level depth
            sentence_count_score * 1.0 +   # Enough sentences
            nuance_score * 1.2             # Nuanced, qualified thinking
        )
        
        # Max possible raw ~13.0
        # Apply deflection penalty
        raw_score *= (1.0 - deflection_penalty)
        
        # Scale to 0-10
        final_score = (raw_score / 13.0) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        # Fallback: simple length-based score
        try:
            words = len(response.split()) if response else 0
            return min(words / 20.0, 5.0)
        except Exception:
            return 2.0