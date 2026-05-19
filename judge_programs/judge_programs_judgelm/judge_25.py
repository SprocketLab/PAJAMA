def judging_function(query, response):
    """
    Evaluate factual accuracy indicators in an LLM response.
    
    Focuses on:
    - Presence of verifiable facts (names, dates, numbers, citations)
    - Appropriate hedging for uncertain claims
    - Absence of hallucination red-flags
    - Penalizing sensationalism and conspiracy-style language
    - Response completeness and coherence
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        # If response is essentially empty or trivially short
        if len(response_stripped) < 2:
            return 0.0
        
        score = 5.0  # Start at midpoint
        
        # ===== 1. RESPONSE LENGTH AND SUBSTANCE =====
        resp_words = response_stripped.split()
        num_words = len(resp_words)
        
        if num_words <= 1:
            return 0.5
        elif num_words <= 3:
            score -= 2.5
        elif num_words <= 8:
            score -= 1.0
        elif num_words >= 20:
            score += 0.5
        elif num_words >= 50:
            score += 0.8
        
        # ===== 2. FACTUAL INDICATORS: specific names, dates, numbers =====
        # Dates (years, full dates)
        year_pattern = r'\b(1[0-9]{3}|20[0-2][0-9])\b'
        years_found = re.findall(year_pattern, response_stripped)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates_found = re.findall(date_pattern, response_stripped)
        
        # Numbers with context (not just random digits)
        number_pattern = r'\b\d+[,.]?\d*\s*(percent|%|million|billion|thousand|km|miles|meters|feet|dollars|euros|pounds|people|years|months|days|hours)\b'
        numbers_with_context = re.findall(number_pattern, response_stripped, re.IGNORECASE)
        
        # Proper nouns (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]\s+', response_stripped)
        proper_noun_count = 0
        for sent in sentences:
            words_in_sent = sent.split()
            for i, w in enumerate(words_in_sent):
                if i > 0 and len(w) > 1 and w[0].isupper() and not w.isupper():
                    proper_noun_count += 1
        
        # Citation-like patterns
        citation_patterns = [
            r'according to',
            r'research\s+(shows?|suggests?|indicates?|found)',
            r'studies?\s+(show|suggest|indicate|found)',
            r'published\s+in',
            r'reported\s+by',
            r'source[s]?:',
            r'reference[s]?:',
            r'\[\d+\]',
            r'\([\w\s]+,\s*\d{4}\)',
        ]
        citation_count = 0
        for pat in citation_patterns:
            citation_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        factual_indicator_score = 0.0
        factual_indicator_score += min(len(years_found) * 0.3, 1.0)
        factual_indicator_score += min(len(dates_found) * 0.3, 0.6)
        factual_indicator_score += min(len(numbers_with_context) * 0.3, 0.9)
        factual_indicator_score += min(proper_noun_count * 0.15, 1.0)
        factual_indicator_score += min(citation_count * 0.4, 1.2)
        
        score += factual_indicator_score
        
        # ===== 3. APPROPRIATE HEDGING =====
        hedging_phrases = [
            r'\bit is (difficult|hard|challenging) to\b',
            r'\bgenerally\b',
            r'\btypically\b',
            r'\busually\b',
            r'\boften\b',
            r'\bmay\b',
            r'\bmight\b',
            r'\bcould\b',
            r'\bperhaps\b',
            r'\bapproximately\b',
            r'\babout\b',
            r'\baround\b',
            r'\bestimated\b',
            r'\blikely\b',
            r'\bpossibly\b',
            r'\bit (seems|appears)\b',
            r'\bhowever\b',
            r'\balthough\b',
            r'\bdepending on\b',
            r'\bcan vary\b',
            r'\bnot without\b',
            r'\bsubjective\b',
            r'\binterpretation\b',
        ]
        hedge_count = 0
        for pat in hedging_phrases:
            hedge_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        # Moderate hedging is good; too much might indicate evasion
        if hedge_count > 0:
            hedge_bonus = min(hedge_count * 0.2, 1.0)
            score += hedge_bonus
        
        # ===== 4. HALLUCINATION RED FLAGS =====
        hallucination_flags = 0
        
        # Overly precise unsourced statistics
        precise_stat_pattern = r'\b\d{2,}\.\d{2,}\s*(%|percent)\b'
        precise_stats = re.findall(precise_stat_pattern, response_stripped)
        hallucination_flags += len(precise_stats) * 0.5
        
        # Absolute claims without evidence
        absolute_patterns = [
            r'\b(always|never|every single|without exception|100%|guaranteed|proven fact)\b',
            r'\b(undeniable|irrefutable|unquestionable|beyond doubt)\b',
            r'\b(everyone knows|it is a fact that|the truth is)\b',
        ]
        for pat in absolute_patterns:
            matches = re.findall(pat, response_stripped, re.IGNORECASE)
            hallucination_flags += len(matches) * 0.3
        
        score -= min(hallucination_flags, 2.0)
        
        # ===== 5. SENSATIONALISM AND CONSPIRACY LANGUAGE =====
        sensational_terms = [
            r'\b(shocking|bombshell|explosive|mind-blowing|unbelievable)\b',
            r'\b(they don\'t want you to know|wake up|sheeple|mainstream media lies)\b',
            r'\b(cover[- ]?up|conspiracy|deep state|big pharma|new world order)\b',
            r'\b(exposed|revealed|secret|hidden truth|suppressed)\b',
            r'\b(!!+|EXPOSED|SHOCKING|BREAKING)\b',
        ]
        sensational_count = 0
        for pat in sensational_terms:
            sensational_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        score -= min(sensational_count * 0.5, 2.0)
        
        # ===== 6. COHERENCE AND STRUCTURE =====
        # Check for repetition (a sign of low quality / broken generation)
        if num_words >= 6:
            # Check for repeated phrases (3-grams)
            trigrams = [' '.join(resp_words[i:i+3]).lower() for i in range(len(resp_words)-2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                most_common_count = trigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / max(len(trigrams), 1)
                if repetition_ratio > 0.3:
                    score -= 2.0
                elif repetition_ratio > 0.15:
                    score -= 1.0
        
        # Check for broken/truncated text
        truncation_indicators = [
            response_stripped.endswith('...'),
            response_stripped.endswith(' the'),
            response_stripped.endswith(' a'),
            response_stripped.endswith(' an'),
            response_stripped.endswith(' is'),
            response_stripped.endswith(' of'),
            response_stripped.endswith(' in'),
            response_stripped.endswith(' to'),
            response_stripped.endswith(' and'),
            response_stripped.endswith(' or'),
            response_stripped.endswith(' for'),
            response_stripped.endswith(' with'),
        ]
        # Mild penalty for truncation - it happens but not ideal
        if any(truncation_indicators):
            score -= 0.3
        
        # ===== 7. RELEVANCE TO QUERY =====
        query_words_set = set(re.findall(r'\b\w+\b', query_stripped.lower()))
        response_words_set = set(re.findall(r'\b\w+\b', response_stripped.lower()))
        
        # Remove very common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'out', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                      'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
                      'and', 'or', 'if', 'while', 'about', 'up', 'down', 'it', 'its',
                      'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
                      'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
                      'their', 'what', 'which', 'who', 'whom'}
        
        query_content = query_words_set - stop_words
        response_content = response_words_set - stop_words
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
            if overlap < 0.1:
                score -= 1.5
            elif overlap >= 0.3:
                score += 0.5
        
        # ===== 8. GARBAGE / IRRELEVANT CONTENT DETECTION =====
        # HTML/code in non-code responses
        query_lower = query_stripped.lower()
        is_code_query = any(term in query_lower for term in ['html', 'code', 'program', 'script', 'function', 'tag'])
        
        if not is_code_query:
            html_tags = re.findall(r'<[a-zA-Z/][^>]*>', response_stripped)
            code_indicators = re.findall(r'(import |def |class |function |var |let |const )', response_stripped)
            
            if len(html_tags) > 3:
                score -= 1.5
            if len(code_indicators) > 2:
                score -= 1.5
        
        # Check for "Input:/Output:" patterns that suggest template confusion
        template_patterns = re.findall(r'(Input:|Output:|Question:|Answer:)', response_stripped)
        if len(template_patterns) > 3:
            score -= 1.5
        
        # ===== 9. SENTENCE STRUCTURE QUALITY =====
        # Well-formed sentences suggest higher quality
        sentence_count = len([s for s in sentences if len(s.strip()) > 5])
        if sentence_count >= 2:
            score += 0.3
        if sentence_count >= 4:
            score += 0.3
        
        # Check for sentences that start with capital letters (proper formatting)
        well_formed = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if sentences and len(sentences) > 0:
            formation_ratio = well_formed / max(len(sentences), 1)
            if formation_ratio > 0.7:
                score += 0.3
        
        # ===== 10. UNIQUE VOCABULARY RICHNESS =====
        if num_words >= 10:
            unique_ratio = len(set(w.lower() for w in resp_words)) / num_words
            if unique_ratio > 0.7:
                score += 0.3
            elif unique_ratio < 0.3:
                score -= 0.5
        
        # ===== 11. RESPONSE DIRECTLY ADDRESSES QUERY TYPE =====
        # Questions expecting information should get informative responses
        is_question = '?' in query_stripped or any(
            query_lower.startswith(w) for w in ['what', 'who', 'where', 'when', 'why', 'how', 'is ', 'can ', 'do ', 'does ']
        )
        
        if is_question and num_words < 5:
            # Very short answers to questions are usually bad
            score -= 1.0
        
        # Single word "no" or "yes" responses to complex questions
        if response_stripped.lower().strip('.!') in ['no', 'yes', 'maybe', 'ok', 'okay']:
            if len(query_words_set) > 5:
                score -= 2.0
        
        # ===== 12. EXPLANATORY CONNECTORS =====
        # Good factual responses often use explanatory language
        explanatory_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bin other words\b', r'\bspecifically\b', r'\bnamely\b',
            r'\bincluding\b', r'\bknown as\b', r'\breferred to as\b',
            r'\bis (also )?(called|known|named|referred)\b',
        ]
        explanatory_count = 0
        for pat in explanatory_patterns:
            explanatory_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        score += min(explanatory_count * 0.2, 0.8)
        
        # ===== FINAL CLAMPING =====
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a middle-ish score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except:
            return 0.0