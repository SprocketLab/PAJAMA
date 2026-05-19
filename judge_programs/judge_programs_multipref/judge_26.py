def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Strategy: Analyze linguistic patterns associated with factual reliability vs. hallucination.
    Uses a combination of:
    1. Specificity signals (dates, numbers, proper nouns, citations)
    2. Appropriate hedging vs. overconfidence detection
    3. Hallucination red-flags (fabricated precision, absolute claims)
    4. Structural credibility markers (organized reasoning, caveats)
    5. Sensationalism / conspiracy language detection
    6. Response completeness and engagement quality
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        resp = response.strip()
        query_lower = query.lower()
        resp_lower = resp.lower()
        
        words = resp.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. SPECIFICITY SIGNALS — verifiable facts indicators
        # ============================================================
        
        # Dates (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', resp)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', resp)
        specific_dates = len(year_pattern) + len(date_pattern)
        score += min(specific_dates * 1.5, 8.0)
        
        # Specific numbers (measurements, statistics with units)
        numbers_with_units = re.findall(
            r'\b\d+\.?\d*\s*(?:kg|lb|m|km|miles?|feet|ft|meters?|degrees?|°|%|percent|hours?|minutes?|seconds?|days?|years?|months?|weeks?|mph|km/h|m/s|lbs?|oz|grams?|mg|ml|liters?|gallons?|inches?|cm|mm)\b',
            resp_lower
        )
        score += min(len(numbers_with_units) * 1.0, 6.0)
        
        # Named entities heuristic — capitalized multi-word phrases (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', resp)
        # Filter out sentence starters by checking if preceded by period
        score += min(len(proper_nouns) * 0.5, 5.0)
        
        # ============================================================
        # 2. APPROPRIATE HEDGING — epistemic humility
        # ============================================================
        
        hedging_phrases = [
            'it is possible', 'it\'s possible', 'may be', 'might be',
            'generally', 'typically', 'usually', 'often', 'tends to',
            'in most cases', 'it depends', 'depending on', 'can vary',
            'not always', 'in some cases', 'approximately', 'roughly',
            'about', 'around', 'estimated', 'likely', 'unlikely',
            'it seems', 'it appears', 'suggests that', 'according to',
            'research suggests', 'studies suggest', 'evidence suggests',
            'as far as i know', 'to my knowledge', 'i believe',
            'however', 'on the other hand', 'that said', 'keep in mind',
            'it\'s worth noting', 'it is worth noting', 'note that',
            'please note', 'be aware', 'consider', 'arguably'
        ]
        
        hedge_count = 0
        for phrase in hedging_phrases:
            hedge_count += resp_lower.count(phrase)
        
        # Moderate hedging is good; too much is wishy-washy
        hedge_ratio = hedge_count / max(word_count / 100, 1)
        if hedge_ratio <= 5:
            score += hedge_ratio * 2.0
        else:
            score += 10.0 - (hedge_ratio - 5) * 1.0
        
        # ============================================================
        # 3. HALLUCINATION RED FLAGS — overconfidence, fabrication
        # ============================================================
        
        # Absolute/overconfident language
        absolute_phrases = [
            'always', 'never', 'absolutely', 'definitely', 'certainly',
            'without a doubt', 'undoubtedly', 'unquestionably',
            'there is no doubt', 'it is certain', 'guaranteed',
            'proven fact', 'everyone knows', 'obviously',
            'clearly the best', 'the only way', 'no one can',
            'impossible to', 'without exception', 'in every case',
            '100%', 'completely proven'
        ]
        
        absolute_count = 0
        for phrase in absolute_phrases:
            absolute_count += resp_lower.count(phrase)
        
        score -= min(absolute_count * 2.5, 12.0)
        
        # Overly precise unsourced statistics (e.g., "exactly 73.2% of people")
        suspicious_stats = re.findall(r'\b\d{1,3}\.\d+\s*%', resp)
        # Check if these are in a math/science context (less suspicious)
        math_context = any(w in query_lower for w in ['calculate', 'compute', 'solve', 'equation', 'formula', 'speed', 'mass', 'energy', 'physics'])
        if not math_context:
            score -= min(len(suspicious_stats) * 2.0, 6.0)
        
        # ============================================================
        # 4. SENSATIONALISM & CONSPIRACY DETECTION
        # ============================================================
        
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'mind-blowing',
            'they don\'t want you to know', 'the truth about',
            'wake up', 'sheeple', 'mainstream media', 'big pharma',
            'cover-up', 'coverup', 'conspiracy', 'deep state',
            'new world order', 'illuminati', 'secret agenda',
            'brainwash', 'propaganda', 'hoax', 'scam',
            'exposed', 'you won\'t believe', 'insane', 'crazy',
            'unbelievable', 'jaw-dropping', 'terrifying truth',
            'what they\'re hiding', 'banned', 'censored',
            'the real truth', 'exposed the truth'
        ]
        
        sensational_count = 0
        for phrase in sensational_words:
            sensational_count += resp_lower.count(phrase)
        
        score -= min(sensational_count * 4.0, 20.0)
        
        # ============================================================
        # 5. CITATION / SOURCE INDICATORS
        # ============================================================
        
        citation_patterns = [
            r'according to\b', r'research (?:shows|indicates|suggests|finds)',
            r'studies? (?:show|indicate|suggest|find|have shown)',
            r'published in\b', r'journal of\b', r'university of\b',
            r'\bsource:', r'\breference:', r'\bcited\b',
            r'(?:a|the) (?:recent|new|latest) study', r'data (?:shows?|suggests?|indicates?)',
            r'experts? (?:say|suggest|recommend|believe|note)',
            r'scientists? (?:say|suggest|found|discovered)',
            r'researchers? (?:say|suggest|found|discovered)'
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, resp_lower))
        
        score += min(citation_count * 2.0, 8.0)
        
        # ============================================================
        # 6. STRUCTURAL QUALITY — organized, clear reasoning
        # ============================================================
        
        # Numbered steps or ordered lists
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', resp)
        has_structure = len(numbered_items) >= 2
        
        # Markdown headers
        headers = re.findall(r'#{1,4}\s+\S', resp)
        has_headers = len(headers) >= 1
        
        # Bold markers for emphasis/organization
        bold_markers = re.findall(r'\*\*[^*]+\*\*', resp)
        has_bold = len(bold_markers) >= 2
        
        structure_score = 0
        if has_structure:
            structure_score += 3.0
        if has_headers:
            structure_score += 2.0
        if has_bold:
            structure_score += 2.0
        
        score += min(structure_score, 6.0)
        
        # ============================================================
        # 7. SENTENCE COMPLEXITY & COHERENCE PROXIES
        # ============================================================
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length — moderate is good
        avg_sent_len = word_count / num_sentences
        if 10 <= avg_sent_len <= 25:
            score += 3.0
        elif 8 <= avg_sent_len <= 30:
            score += 1.5
        else:
            score -= 1.0
        
        # Vocabulary diversity (type-token ratio for first 200 words)
        sample_words = [w.lower().strip('.,!?:;()[]{}"\'-') for w in words[:200]]
        sample_words = [w for w in sample_words if w]
        if len(sample_words) > 10:
            ttr = len(set(sample_words)) / len(sample_words)
            if ttr > 0.65:
                score += 3.0
            elif ttr > 0.50:
                score += 1.5
            else:
                score -= 1.0
        
        # ============================================================
        # 8. EXPLANATORY DEPTH — causal/logical connectors
        # ============================================================
        
        causal_connectors = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'this means',
            'this is because', 'the reason', 'which leads to',
            'which means', 'in other words', 'for example',
            'for instance', 'such as', 'specifically',
            'in particular', 'namely', 'that is'
        ]
        
        causal_count = 0
        for connector in causal_connectors:
            causal_count += resp_lower.count(connector)
        
        score += min(causal_count * 1.0, 6.0)
        
        # ============================================================
        # 9. RESPONSE ENGAGEMENT & COMPLETENESS
        # ============================================================
        
        # Penalize very short responses (likely incomplete)
        if word_count < 20:
            score -= 10.0
        elif word_count < 50:
            score -= 5.0
        elif word_count >= 80:
            score += 2.0
        
        # Check if response seems cut off (ends mid-sentence)
        last_chars = resp[-10:] if len(resp) >= 10 else resp
        if not re.search(r'[.!?:)\]}\n]', last_chars):
            # Might be truncated — slight penalty but don't over-penalize
            score -= 2.0
        
        # Greeting/engagement with the user
        engagement_phrases = [
            'great question', 'good question', 'that\'s a great',
            'let me', 'let\'s', 'i\'d be happy', 'here\'s',
            'here are', 'certainly', 'absolutely',  # used as engagement, not overconfidence
        ]
        
        # Only count engagement at the start
        first_50_chars = resp_lower[:80]
        engagement_count = sum(1 for p in engagement_phrases if p in first_50_chars)
        score += min(engagement_count * 1.5, 3.0)
        
        # ============================================================
        # 10. QUERY RELEVANCE — does response address the query?
        # ============================================================
        
        # Extract meaningful query words (remove stop words)
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
            'they', 'them', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'not', 'if', 'that', 'this', 'what', 'how',
            'which', 'who', 'when', 'where', 'why', 'all', 'each', 'any',
            'some', 'no', 'so', 'than', 'too', 'very', 'just', 'about',
            'up', 'out', 'into', 'over', 'after', 'before', 'between',
            'am', 'im', 'its', 'there', 'here', 'need', 'want', 'get',
            'think', 'know', 'like', 'make', 'go', 'see', 'come', 'take'
        }
        
        query_words = set(
            w.lower().strip('.,!?:;()[]{}"\'-') 
            for w in query.split() 
            if len(w) > 2
        ) - stop_words
        
        resp_word_set = set(
            w.lower().strip('.,!?:;()[]{}"\'-') 
            for w in words
        )
        
        if query_words:
            overlap = len(query_words & resp_word_set) / len(query_words)
            score += overlap * 5.0
        
        # ============================================================
        # 11. BALANCED PERSPECTIVE INDICATORS
        # ============================================================
        
        balance_phrases = [
            'on the other hand', 'however', 'alternatively',
            'pros and cons', 'advantages and disadvantages',
            'while', 'although', 'whereas', 'conversely',
            'both', 'different perspectives', 'some argue',
            'others believe', 'it\'s important to consider',
            'there are several', 'multiple factors'
        ]
        
        balance_count = 0
        for phrase in balance_phrases:
            balance_count += min(resp_lower.count(phrase), 2)
        
        score += min(balance_count * 1.5, 6.0)
        
        # ============================================================
        # 12. CONFIDENCE CALIBRATION — appropriate certainty level
        # ============================================================
        
        # "I'm not sure" or "I don't know" — can be good if honest
        uncertainty_admissions = [
            'i\'m not sure', 'i am not sure', 'i don\'t know',
            'i do not know', 'i\'m not aware', 'i am not aware',
            'i cannot confirm', 'i can\'t confirm'
        ]
        
        uncertainty_count = sum(1 for p in uncertainty_admissions if p in resp_lower)
        # Mild bonus for honesty, but not too much
        score += min(uncertainty_count * 1.0, 2.0)
        
        # ============================================================
        # 13. EXCLAMATION MARK DENSITY — enthusiasm vs sensationalism
        # ============================================================
        
        exclamation_count = resp.count('!')
        excl_ratio = exclamation_count / max(num_sentences, 1)
        
        # A little enthusiasm is fine, too much is sensational
        if excl_ratio > 0.5:
            score -= min((excl_ratio - 0.5) * 4.0, 6.0)
        
        # ============================================================
        # 14. FIRST-PERSON OPINION vs FACTUAL FRAMING
        # ============================================================
        
        # For factual queries, less first-person is better
        # For opinion queries, first-person is fine
        opinion_query = any(w in query_lower for w in [
            'think', 'opinion', 'believe', 'feel', 'should',
            'recommend', 'suggest', 'best', 'favorite', 'prefer'
        ])
        
        first_person = len(re.findall(r'\bi\b|\bmy\b|\bme\b', resp_lower))
        fp_ratio = first_person / max(word_count, 1)
        
        if not opinion_query and fp_ratio > 0.03:
            score -= min((fp_ratio - 0.03) * 50, 5.0)
        
        # ============================================================
        # 15. TRANSITION WORD QUALITY — logical flow
        # ============================================================
        
        transition_words = [
            'first', 'second', 'third', 'finally', 'additionally',
            'moreover', 'furthermore', 'in addition', 'next',
            'then', 'subsequently', 'lastly', 'in conclusion',
            'to summarize', 'overall', 'in summary'
        ]
        
        transition_count = 0
        for tw in transition_words:
            transition_count += resp_lower.count(tw)
        
        score += min(transition_count * 0.8, 4.0)
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        
        # Clamp score to [0, 100]
        score = max(0.0, min(100.0, score))
        
        # Scale to [0, 10] for cleaner output
        final_score = score / 10.0
        
        return round(final_score, 3)
        
    except Exception:
        return 5.0