def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and information density approach. This variant focuses on:
    1. Query decomposition - identifying sub-questions/aspects the query asks about
    2. Information density and specificity scoring
    3. Structural completeness (intro, body, conclusion patterns)
    4. Example/evidence provision detection
    5. Depth estimation via unique concept density
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
        # 1. QUERY DECOMPOSITION - Identify what aspects need covering
        # ============================================================
        
        # Extract question words and their associated topics
        query_lower = query.lower()
        
        # Count distinct question aspects
        question_markers = re.findall(r'\b(what|how|why|when|where|who|which|can|should|do|does|is|are|could|would)\b', query_lower)
        num_question_aspects = max(1, len(set(question_markers)))
        
        # Extract key noun phrases / content words from query
        stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them',
                      'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                      'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                      'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither',
                      'that', 'this', 'these', 'those', 'what', 'how', 'why', 'when', 'where',
                      'who', 'which', 'if', 'then', 'than', 'very', 'just', 'about', 'up',
                      'out', 'any', 'some', 'all', 'each', 'every', 'no', 'more', 'most',
                      'other', 'also', 'too', 'only', 'own', 'same', 'such', 'here', 'there',
                      'am', 'im', 'need', 'want', 'get', 'got', 'bit'}
        
        query_words = re.findall(r'[a-z]+', query_lower)
        query_content_words = [w for w in query_words if w not in stop_words and len(w) > 2]
        query_content_set = set(query_content_words)
        
        response_lower = response.lower()
        response_words = re.findall(r'[a-z]+', response_lower)
        response_content_words = [w for w in response_words if w not in stop_words and len(w) > 2]
        response_content_set = set(response_content_words)
        
        # Query topic coverage ratio
        if query_content_set:
            topic_coverage = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            topic_coverage = 0.5
        
        score += topic_coverage * 12  # 0-12 points
        
        # ============================================================
        # 2. INFORMATION DENSITY & SPECIFICITY
        # ============================================================
        
        # Count specific/concrete information markers
        # Numbers and quantities
        numbers = re.findall(r'\b\d+[\d,.]*\b', response)
        number_score = min(len(numbers) * 0.5, 5)
        
        # Proper nouns (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]\s+', response)
        proper_noun_count = 0
        for sent in sentences:
            words_in_sent = sent.split()
            for i, w in enumerate(words_in_sent):
                if i > 0 and w and w[0].isupper() and len(w) > 1 and not w.isupper():
                    proper_noun_count += 1
        proper_noun_score = min(proper_noun_count * 0.3, 4)
        
        # Technical/specific terms (longer words tend to be more specific)
        specific_words = [w for w in response_content_words if len(w) >= 7]
        specificity_ratio = len(specific_words) / max(len(response_content_words), 1)
        specificity_score = specificity_ratio * 8  # 0-8
        
        # Unique concept density (unique content words / total content words)
        if response_content_words:
            unique_ratio = len(response_content_set) / len(response_content_words)
            # Sweet spot: not too repetitive, not too scattered
            concept_density_score = min(unique_ratio * 8, 6)
        else:
            concept_density_score = 0
        
        score += number_score + proper_noun_score + specificity_score + concept_density_score
        
        # ============================================================
        # 3. STRUCTURAL COMPLETENESS PATTERN
        # ============================================================
        
        # Check for introduction (first sentence/paragraph sets context)
        first_sentence = sentences[0] if sentences else ""
        first_words = first_sentence.lower().split()[:10]
        
        # Introduction indicators
        intro_patterns = ['certainly', 'great', 'sure', 'absolutely', 'yes', 'no', 'here',
                         'there are', 'this is', 'the', 'when', 'to', 'organizing', 'brewing',
                         'writing', 'traveling', 'getting']
        has_intro = any(p in ' '.join(first_words) for p in intro_patterns)
        
        # Check for conclusion/summary at end
        last_200 = response[-200:].lower() if len(response) > 200 else response_lower
        conclusion_markers = ['overall', 'in summary', 'in conclusion', 'finally', 'remember',
                            'good luck', 'enjoy', 'hope this', 'happy', 'have fun',
                            'key takeaway', 'most importantly', 'in short', 'to sum up',
                            'these are', 'this should', 'feel free']
        has_conclusion = any(m in last_200 for m in conclusion_markers)
        
        # Body structure - multiple distinct points/sections
        # Detect enumeration patterns (numbered lists, lettered lists)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        lettered_items = re.findall(r'(?:^|\n)\s*[a-z][\.\)]\s', response)
        dash_items = re.findall(r'(?:^|\n)\s*[-•*]\s', response)
        
        total_list_items = len(numbered_items) + len(lettered_items) + len(dash_items)
        
        structural_score = 0
        if has_intro:
            structural_score += 2
        if has_conclusion:
            structural_score += 2
        
        # Reward organized multi-point responses
        if total_list_items >= 3:
            structural_score += min(total_list_items * 0.5, 4)
        
        # Headers/sections (markdown or plain)
        headers = re.findall(r'(?:^|\n)\s*#{1,4}\s+.+', response)
        bold_headers = re.findall(r'\*\*[^*]{3,50}\*\*', response)
        section_count = len(headers) + len(bold_headers) // 2  # bold headers might be inline
        
        if section_count >= 2:
            structural_score += min(section_count * 0.8, 4)
        
        score += min(structural_score, 10)  # cap at 10
        
        # ============================================================
        # 4. EXAMPLE/EVIDENCE PROVISION
        # ============================================================
        
        example_markers = ['for example', 'for instance', 'such as', 'e.g.', 'like ',
                          'including', 'consider', 'imagine', 'suppose', 'say,',
                          'specifically', 'in particular', 'one example', 'another example',
                          'case in point', 'to illustrate']
        example_count = sum(1 for m in example_markers if m in response_lower)
        example_score = min(example_count * 1.5, 6)
        
        # Causal/explanatory depth markers
        explanation_markers = ['because', 'since', 'therefore', 'thus', 'consequently',
                              'as a result', 'due to', 'this means', 'which means',
                              'the reason', 'this is why', 'this leads to', 'in order to',
                              'so that', 'which allows', 'which enables', 'resulting in',
                              'caused by', 'leads to', 'contributes to']
        explanation_count = sum(1 for m in explanation_markers if m in response_lower)
        explanation_score = min(explanation_count * 1.0, 6)
        
        score += example_score + explanation_score
        
        # ============================================================
        # 5. DEPTH ESTIMATION - Sentence-level analysis
        # ============================================================
        
        # Average sentence length (longer sentences often carry more information)
        clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(clean_sentences)
        
        if num_sentences > 0:
            avg_sent_len = sum(len(s.split()) for s in clean_sentences) / num_sentences
            # Optimal range: 12-25 words per sentence
            if 12 <= avg_sent_len <= 25:
                sent_len_score = 3
            elif 8 <= avg_sent_len < 12 or 25 < avg_sent_len <= 35:
                sent_len_score = 2
            else:
                sent_len_score = 1
        else:
            sent_len_score = 0
        
        # Paragraph count and diversity
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response) if len(p.strip()) > 20]
        num_paragraphs = len(paragraphs)
        
        # Multiple paragraphs suggest more thorough coverage
        paragraph_score = min(num_paragraphs * 0.8, 4)
        
        # Sentence count contribution (more sentences = more coverage, with diminishing returns)
        sentence_count_score = min(math.log(max(num_sentences, 1) + 1) * 2, 6)
        
        score += sent_len_score + paragraph_score + sentence_count_score
        
        # ============================================================
        # 6. RESPONSE LENGTH CALIBRATED TO QUERY COMPLEXITY
        # ============================================================
        
        query_complexity = len(query_content_words) + num_question_aspects
        response_length = len(response_words)
        
        # Expected minimum response length based on query complexity
        expected_min_length = max(30, query_complexity * 8)
        
        length_ratio = response_length / max(expected_min_length, 1)
        
        if length_ratio >= 2.0:
            length_score = 8
        elif length_ratio >= 1.5:
            length_score = 7
        elif length_ratio >= 1.0:
            length_score = 5
        elif length_ratio >= 0.5:
            length_score = 3
        else:
            length_score = 1
        
        score += length_score
        
        # ============================================================
        # 7. COMPLETENESS SIGNALS - Checking for truncation/incompleteness
        # ============================================================
        
        # Penalize truncated responses
        truncation_penalty = 0
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        
        # Response ends mid-sentence (no terminal punctuation)
        if last_char not in '.!?:"\')]}':
            truncation_penalty += 5
        
        # Response ends with incomplete thought indicators
        last_50 = response[-50:].lower() if len(response) > 50 else response_lower
        incomplete_endings = [' a ', ' an ', ' the ', ' and ', ' or ', ' but ', ' to ',
                            ' in ', ' on ', ' for ', ' with ', ' that ', ' which ',
                            ' this ', ' these ', ' of ']
        for ending in incomplete_endings:
            if last_50.endswith(ending.rstrip()):
                truncation_penalty += 3
                break
        
        score -= truncation_penalty
        
        # ============================================================
        # 8. MULTI-PERSPECTIVE / NUANCE DETECTION
        # ============================================================
        
        # Contrasting viewpoints or considerations
        contrast_markers = ['however', 'on the other hand', 'alternatively', 'in contrast',
                           'conversely', 'although', 'while', 'whereas', 'nevertheless',
                           'nonetheless', 'despite', 'even though', 'that said',
                           'keep in mind', 'note that', 'be aware', 'important to note',
                           'worth noting', 'on the flip side']
        nuance_count = sum(1 for m in contrast_markers if m in response_lower)
        nuance_score = min(nuance_count * 1.2, 5)
        
        score += nuance_score
        
        # ============================================================
        # 9. ACTIONABILITY / PRACTICAL DETAIL (for how-to queries)
        # ============================================================
        
        is_howto = any(w in query_lower for w in ['how', 'can i', 'help me', 'need to', 'want to',
                                                    'figure out', 'learn', 'prepare', 'make', 'get'])
        
        if is_howto:
            # Check for actionable language
            action_verbs = ['start', 'begin', 'first', 'next', 'then', 'after', 'finally',
                          'make sure', 'ensure', 'check', 'add', 'remove', 'place', 'put',
                          'mix', 'combine', 'choose', 'select', 'decide', 'plan',
                          'step', 'tip', 'note', 'gather', 'prepare', 'set up']
            action_count = sum(1 for v in action_verbs if v in response_lower)
            action_score = min(action_count * 0.6, 5)
            score += action_score
        
        # ============================================================
        # 10. SEMANTIC FIELD COVERAGE
        # ============================================================
        
        # Build semantic clusters from response and check diversity
        # Use word length distribution as proxy for vocabulary diversity
        if response_content_words:
            word_lengths = [len(w) for w in response_content_set]
            if word_lengths:
                length_variance = sum((l - sum(word_lengths)/len(word_lengths))**2 
                                    for l in word_lengths) / len(word_lengths)
                vocab_diversity_score = min(math.sqrt(length_variance) * 1.5, 4)
            else:
                vocab_diversity_score = 0
            
            # Unique vocabulary size relative to response
            vocab_richness = len(response_content_set)
            vocab_size_score = min(math.log(max(vocab_richness, 1) + 1) * 1.2, 5)
        else:
            vocab_diversity_score = 0
            vocab_size_score = 0
        
        score += vocab_diversity_score + vocab_size_score
        
        # Normalize to 0-100 range
        # Theoretical max is roughly: 12 + 5+4+8+6 + 10 + 6+6 + 3+4+6 + 8 + 0 + 5 + 5 + 4+5 = ~97
        # But typical good responses score 40-70 raw
        score = max(0, score)
        
        # Scale to 0-100
        final_score = min(score * 1.1, 100)
        
        return round(final_score, 2)
    
    except Exception as e:
        # Fallback: return a basic length-based score
        try:
            return min(len(str(response).split()) / 5, 50)
        except:
            return 0.0