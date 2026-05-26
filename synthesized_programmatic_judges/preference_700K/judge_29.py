def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a substantially different approach:
    - Named entity density (capitalized multi-word phrases, proper nouns)
    - Citation/reference pattern detection
    - Specificity signals (dates, numbers, measurements, named works)
    - Hallucination red-flags (absolute claims, unsourced precise stats, sensationalism)
    - Epistemic calibration (appropriate uncertainty vs overconfidence)
    - Discourse structure (causal reasoning, evidence-based argumentation)
    - Information density ratio (content words vs filler)
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0

        resp = response.strip()
        if len(resp) < 10:
            return 0.5

        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0

        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        sent_count = max(len(sentences), 1)

        score = 50.0  # Start at midpoint

        # ============================================================
        # 1. NAMED ENTITY DENSITY — detect capitalized phrases as proxy
        #    for specific references (people, places, works, institutions)
        # ============================================================
        # Find sequences of capitalized words (2+ words) not at sentence start
        cap_phrases = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', resp)
        # Also find mid-sentence capitalized words
        mid_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', resp)
        
        entity_density = (len(cap_phrases) * 2 + len(mid_caps)) / max(word_count, 1)
        score += min(entity_density * 120, 8.0)

        # ============================================================
        # 2. SPECIFICITY SIGNALS — dates, numbers, measurements, titles
        # ============================================================
        # Dates (various formats)
        date_patterns = re.findall(r'\b\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)\s+(?:century|Century)', resp)
        # Numbers with context (not just bare numbers)
        contextual_numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|years|months|days|hours|kg|lb|miles|km|degrees))', resp)
        # Quoted or italicized titles
        titles = re.findall(r'\*[^*]+\*|"[^"]{3,}"', resp)
        # Specific references like "u/username", "Section X", "Chapter X"
        references = re.findall(r'u/\w+|(?:Section|Chapter|Article|Part)\s+\d+', resp)
        
        specificity_count = len(date_patterns) + len(contextual_numbers) * 1.5 + len(titles) * 2 + len(references) * 1.5
        specificity_score = min(specificity_count / max(sent_count, 1) * 3.0, 8.0)
        score += specificity_score

        # ============================================================
        # 3. CITATION / ATTRIBUTION PATTERNS
        # ============================================================
        attribution_patterns = [
            r'according to', r'research (?:shows|suggests|indicates|found)',
            r'studies (?:show|suggest|indicate|found)', r'as (?:noted|described|mentioned|argued) by',
            r'(?:wrote|stated|argued|claimed|noted) (?:that|in)',
            r'in (?:his|her|their) (?:book|paper|article|work|essay)',
            r'published in', r'(?:source|cited|reference|bibliography)',
            r'(?:for example|for instance|e\.g\.|i\.e\.)',
            r'an (?:earlier|previous) (?:answer|response|post)',
        ]
        attribution_count = 0
        resp_lower = resp.lower()
        for pat in attribution_patterns:
            attribution_count += len(re.findall(pat, resp_lower))
        
        score += min(attribution_count * 2.5, 7.0)

        # ============================================================
        # 4. EPISTEMIC CALIBRATION — appropriate hedging vs overconfidence
        # ============================================================
        # Good hedging (shows intellectual honesty)
        calibrated_phrases = [
            r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin many cases\b', r'\bit depends\b', r'\bnot necessarily\b',
            r'\bone (?:could|might|can) argue\b', r'\bthere\'s a chance\b',
            r'\bif\b.*\bthen\b', r'\bceteris paribus\b',
            r'\bto my knowledge\b', r'\bas far as I know\b',
            r'\bthe trade-off\b', r'\bon the other hand\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b.*\b(?:also|but)\b',
        ]
        calibration_count = 0
        for pat in calibrated_phrases:
            calibration_count += len(re.findall(pat, resp_lower))
        
        calibration_density = calibration_count / max(sent_count, 1)
        # Sweet spot: some hedging is good, too much is wishy-washy
        if calibration_density > 0:
            cal_score = min(calibration_density * 4.0, 6.0)
            if calibration_density > 1.5:
                cal_score *= 0.7  # Penalize excessive hedging
            score += cal_score

        # ============================================================
        # 5. HALLUCINATION RED-FLAGS — penalize suspicious patterns
        # ============================================================
        red_flag_score = 0.0
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+\s*%', resp)
        red_flag_score += len(precise_stats) * 2.0
        
        # Absolute claims without evidence
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\beveryone knows\b',
            r'\bit is (?:a )?fact that\b', r'\bundeniably\b', r'\bunquestionably\b',
            r'\bwithout (?:a )?doubt\b', r'\bobviously\b',
            r'\bclearly\b(?!.*\bbut\b|\bhowever\b)',
        ]
        for pat in absolute_patterns:
            matches = re.findall(pat, resp_lower)
            red_flag_score += len(matches) * 0.8
        
        # Sensationalism / conspiracy language
        sensational_patterns = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind-blowing\b',
            r'\bthey don\'t want you to know\b', r'\bhidden truth\b',
            r'\bwake up\b', r'\bsheeple\b', r'\bconspiracy\b',
            r'\bcover[- ]?up\b', r'\bsecretly\b',
            r'\b(?:exposed|debunked|destroyed)\b',
        ]
        for pat in sensational_patterns:
            red_flag_score += len(re.findall(pat, resp_lower)) * 3.0
        
        score -= min(red_flag_score, 12.0)

        # ============================================================
        # 6. DISCOURSE STRUCTURE — causal/logical reasoning indicators
        # ============================================================
        reasoning_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bthis (?:means|implies|suggests)\b',
            r'\bthe reason\b', r'\bdue to\b', r'\bin order to\b',
            r'\bif you\b.*\bthen\b', r'\bthe trade-off\b',
            r'\bon one hand\b', r'\bin contrast\b', r'\bsimilarly\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bspecifically\b', r'\bin particular\b',
        ]
        reasoning_count = 0
        for pat in reasoning_markers:
            reasoning_count += len(re.findall(pat, resp_lower))
        
        reasoning_density = reasoning_count / max(sent_count, 1)
        score += min(reasoning_density * 3.5, 6.0)

        # ============================================================
        # 7. INFORMATION DENSITY — content words vs function words
        # ============================================================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'that', 'this',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
            'if', 'then', 'else', 'also', 'up', 'out', 'about',
        }
        
        words_lower = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words]
        content_words = [w for w in words_lower if w and w not in function_words and len(w) > 2]
        info_density = len(content_words) / max(word_count, 1)
        
        # Moderate info density is best (0.4-0.6 range)
        if info_density > 0.35:
            score += min((info_density - 0.35) * 15, 4.0)

        # ============================================================
        # 8. RESPONSE SUBSTANTIVENESS — length and depth signals
        # ============================================================
        # Longer, more developed responses tend to be more informative
        # but with diminishing returns
        length_score = 0.0
        if word_count >= 20:
            length_score = min(math.log(word_count / 20 + 1) * 3.0, 6.0)
        elif word_count < 15:
            length_score = -2.0
        score += length_score

        # Multiple sentences show developed thought
        if sent_count >= 3:
            score += min((sent_count - 2) * 0.5, 3.0)

        # ============================================================
        # 9. QUERY RELEVANCE — topical alignment via n-gram analysis
        # ============================================================
        query_lower = query.lower()
        query_words_raw = re.findall(r'[a-z]+', query_lower)
        query_content = [w for w in query_words_raw if w not in function_words and len(w) > 2]
        
        resp_words_set = set(words_lower)
        
        if query_content:
            # Bigram overlap from query to response
            query_bigrams = set()
            for i in range(len(query_content) - 1):
                query_bigrams.add((query_content[i], query_content[i+1]))
            
            resp_content = [w for w in words_lower if w not in function_words and len(w) > 2]
            resp_bigrams = set()
            for i in range(len(resp_content) - 1):
                resp_bigrams.add((resp_content[i], resp_content[i+1]))
            
            if query_bigrams:
                bigram_overlap = len(query_bigrams & resp_bigrams) / len(query_bigrams)
                score += bigram_overlap * 5.0
            
            # Unigram topical coverage
            query_content_set = set(query_content)
            if query_content_set:
                unigram_coverage = len(query_content_set & resp_words_set) / len(query_content_set)
                score += unigram_coverage * 4.0

        # ============================================================
        # 10. ELABORATION PATTERNS — examples, explanations, contrasts
        # ============================================================
        elaboration_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\blike\b.*\band\b', r'\be\.g\.\b',
            r'\bin other words\b', r'\bthat is\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bconsider\b',
            r'\bwhereas\b', r'\bwhile\b', r'\bunlike\b',
        ]
        elab_count = 0
        for pat in elaboration_patterns:
            elab_count += len(re.findall(pat, resp_lower))
        score += min(elab_count * 1.5, 5.0)

        # ============================================================
        # 11. PERSONAL EXPERIENCE / FIRST-HAND KNOWLEDGE signals
        # ============================================================
        experience_patterns = [
            r'\bin my experience\b', r'\bi\'ve (?:seen|found|noticed|worked)\b',
            r'\bwhen i\b', r'\bi (?:used to|have|had)\b',
            r'\bfrom what i\'ve\b', r'\bpersonally\b',
        ]
        exp_count = 0
        for pat in experience_patterns:
            exp_count += len(re.findall(pat, resp_lower))
        # Some personal experience is good (adds credibility), but not too much
        if exp_count > 0:
            score += min(exp_count * 1.5, 4.0)

        # ============================================================
        # 12. STRUCTURAL SOPHISTICATION — parentheticals, qualifications
        # ============================================================
        parentheticals = len(re.findall(r'\([^)]{5,}\)', resp))
        em_dashes = resp.count('--') + resp.count('—')
        semicolons = resp.count(';')
        
        structural_complexity = parentheticals + em_dashes * 0.5 + semicolons * 0.5
        score += min(structural_complexity * 1.0, 3.0)

        # ============================================================
        # 13. PENALIZE BOT/META RESPONSES — automated disclaimers, etc.
        # ============================================================
        bot_patterns = [
            r'\bwelcome to /r/\b', r'\bplease read our rules\b',
            r'\byour (?:comment|post) (?:will be|has been) removed\b',
            r'\bi am a bot\b', r'\bthis is an automated\b',
            r'\bdo not fear\b.*\bassist\b',
        ]
        for pat in bot_patterns:
            if re.search(pat, resp_lower):
                score -= 8.0

        # ============================================================
        # 14. CODE BLOCK HANDLING — if query involves code, reward structured code
        # ============================================================
        has_code_query = bool(re.search(r'CREATE TABLE|SELECT|SQL|code|function|def |class |import ', query, re.IGNORECASE))
        code_blocks = re.findall(r'```[\s\S]*?```', resp)
        if has_code_query and code_blocks:
            # Reward well-structured code responses
            total_code_len = sum(len(b) for b in code_blocks)
            if total_code_len > 50:
                score += 3.0
            # Check for explanatory text alongside code
            non_code = re.sub(r'```[\s\S]*?```', '', resp).strip()
            if len(non_code) > 30:
                score += 2.0

        # Clamp final score
        score = max(0.0, min(100.0, score))

        return round(score, 2)

    except Exception:
        return 25.0