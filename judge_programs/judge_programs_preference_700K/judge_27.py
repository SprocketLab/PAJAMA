def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Strategy: Named entity density, citation/reference patterns, epistemic calibration,
    specificity scoring, red-flag detection, and structural authority signals.
    
    This variant focuses on:
    1. Named entity proxies (capitalized multi-word phrases, proper nouns)
    2. Citation and reference patterns (quotes, attributions, source mentions)
    3. Epistemic calibration (appropriate certainty vs hedging balance)
    4. Specificity tokens (dates, numbers, technical terms)
    5. Red flags (conspiracy language, sensationalism, unsourced absolutes)
    6. Discourse coherence (logical connectives, explanation depth)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        resp = response.strip()
        query_str = query.strip()
        
        if len(resp) < 5:
            return 0.5
        
        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.5
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint of 0-100
        
        # ============================================================
        # 1. NAMED ENTITY PROXIES - Capitalized phrases as entity proxy
        # ============================================================
        # Find capitalized words that aren't sentence starters
        cap_pattern = re.findall(r'(?<=[a-z] )[A-Z][a-z]+', resp)
        # Multi-word capitalized sequences (proper nouns / titles)
        multi_cap = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', resp)
        
        entity_density = (len(cap_pattern) + len(multi_cap) * 2) / max(word_count, 1)
        # Reward moderate entity density (0.02 - 0.15 is good)
        if entity_density > 0.01:
            score += min(entity_density * 80, 8.0)
        
        # ============================================================
        # 2. CITATION AND REFERENCE PATTERNS
        # ============================================================
        citation_signals = 0
        
        # Book/work references with italics or quotes
        citation_signals += len(re.findall(r'\*[A-Z][^*]+\*', resp)) * 3
        citation_signals += len(re.findall(r'"[^"]{5,}"', resp)) * 1.5
        
        # Attribution phrases
        attribution_phrases = [
            r'according to', r'as (?:noted|described|mentioned|stated|argued) (?:by|in)',
            r'(?:research|studies|evidence) (?:shows?|suggests?|indicates?)',
            r'published (?:in|by)', r'(?:wrote|writes|written by)',
            r'(?:coined|proposed|developed|introduced) by',
            r'in (?:his|her|their) (?:book|paper|work|article|essay)',
            r'(?:professor|dr\.|researcher|author|historian|scientist)\s+[A-Z]',
            r'university of', r'journal of',
            r'/u/\w+',  # Reddit user references
            r'u/\w+',
        ]
        for pat in attribution_phrases:
            matches = re.findall(pat, resp, re.IGNORECASE)
            citation_signals += len(matches) * 2
        
        # Source-type words
        source_words = ['source', 'reference', 'citation', 'study', 'paper', 
                       'article', 'book', 'chapter', 'volume', 'journal',
                       'publication', 'report', 'survey', 'analysis',
                       'tradition', 'scripture', 'text']
        resp_lower = resp.lower()
        for sw in source_words:
            if sw in resp_lower:
                citation_signals += 1.5
        
        score += min(citation_signals * 1.5, 12.0)
        
        # ============================================================
        # 3. SPECIFICITY TOKENS
        # ============================================================
        # Dates (years, specific dates)
        years = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', resp)
        specific_dates = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)', resp, re.IGNORECASE)
        
        # Numbers with context (not just random digits)
        contextual_numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|degrees|miles|km|kg|lbs|dollars|euros|pounds|years|months|hours|minutes|feet|meters|inches|gallons|liters|watts|volts|amps|ohms|psi|mph|kph))\b', resp, re.IGNORECASE)
        
        # Any numbers at all
        all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', resp)
        
        specificity_score = (
            len(years) * 2.0 +
            len(specific_dates) * 3.0 +
            len(contextual_numbers) * 2.5 +
            min(len(all_numbers), 8) * 0.5
        )
        
        score += min(specificity_score, 10.0)
        
        # ============================================================
        # 4. EPISTEMIC CALIBRATION - Balance of confidence and hedging
        # ============================================================
        # Appropriate hedging phrases
        hedging_phrases = [
            r'\bmight\b', r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bgenerally\b', r'\btypically\b', r'\btends? to\b',
            r'\busually\b', r'\boften\b', r'\blikely\b', r'\bprobably\b',
            r'\bit seems\b', r'\bit appears\b', r'\bsome (?:argue|suggest|believe)\b',
            r'\bin (?:some|many|most) cases\b', r'\bnot necessarily\b',
            r'\bdepends on\b', r'\bvariety of\b', r'\brange of\b',
            r'\bto (?:some|a certain) (?:extent|degree)\b',
            r'\bas far as (?:I|we) know\b', r'\bif I (?:recall|remember)\b',
            r'\bI\'?m not (?:sure|certain)\b', r'\bI think\b', r'\bI believe\b',
            r'\bone (?:could|might|may) argue\b',
        ]
        hedge_count = 0
        for pat in hedging_phrases:
            hedge_count += len(re.findall(pat, resp_lower))
        
        hedge_ratio = hedge_count / max(sent_count, 1)
        
        # Confidence phrases
        confidence_phrases = [
            r'\bclearly\b', r'\bobviously\b', r'\bdefinitely\b',
            r'\bcertainly\b', r'\bundoubtedly\b', r'\bwithout (?:a )?doubt\b',
            r'\bof course\b', r'\bnaturally\b', r'\bthe fact (?:is|that)\b',
            r'\bit is (?:well )?known\b', r'\beveryone knows\b',
        ]
        confidence_count = 0
        for pat in confidence_phrases:
            confidence_count += len(re.findall(pat, resp_lower))
        
        confidence_ratio = confidence_count / max(sent_count, 1)
        
        # Good calibration: some hedging (0.1-0.5 per sentence) and moderate confidence
        if 0.05 <= hedge_ratio <= 0.6:
            score += 5.0
        elif hedge_ratio > 0.6:
            score += 2.0  # Over-hedging is slightly less good but not terrible
        
        # Penalize excessive absolute confidence
        if confidence_ratio > 0.3:
            score -= 4.0
        elif confidence_ratio > 0.15:
            score -= 1.5
        
        # ============================================================
        # 5. RED FLAG DETECTION
        # ============================================================
        red_flag_penalty = 0.0
        
        # Conspiracy / sensationalism language
        conspiracy_words = [
            r'\bthey don\'?t want you to know\b', r'\bwake up\b',
            r'\bsheeple\b', r'\bcover[- ]?up\b', r'\bconspiracy\b',
            r'\bmainstream media\b', r'\bmsm\b', r'\bdeep state\b',
            r'\bsecret(?:ly)?\b', r'\bhidden (?:truth|agenda)\b',
            r'\bthe truth is\b.*\bthey\b', r'\bBig (?:Pharma|Tech|Oil)\b',
        ]
        for pat in conspiracy_words:
            if re.search(pat, resp_lower):
                red_flag_penalty += 3.0
        
        # Sensationalist language
        sensational_words = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind[- ]?blowing\b',
            r'\byou won\'?t believe\b', r'\binsane(?:ly)?\b',
            r'\bcrazy\b', r'\bterr?ifying\b', r'\bhorr?ific\b',
            r'\babsolutely (?:devastating|incredible|amazing)\b',
        ]
        for pat in sensational_words:
            if re.search(pat, resp_lower):
                red_flag_penalty += 1.5
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+\s*%', resp)
        if precise_stats and citation_signals < 2:
            red_flag_penalty += len(precise_stats) * 2.0
        
        # Absolute universal claims
        absolute_claims = [
            r'\balways\b', r'\bnever\b', r'\beveryone\b', r'\bnobody\b',
            r'\bno one\b', r'\ball (?:people|humans|scientists|experts)\b',
            r'\bthe only\b', r'\bimpossible\b',
        ]
        abs_count = 0
        for pat in absolute_claims:
            abs_count += len(re.findall(pat, resp_lower))
        
        # Mild penalty for absolutes, worse if no hedging to balance
        if abs_count > 0 and hedge_count == 0:
            red_flag_penalty += abs_count * 1.0
        elif abs_count > 3:
            red_flag_penalty += (abs_count - 3) * 0.5
        
        score -= min(red_flag_penalty, 15.0)
        
        # ============================================================
        # 6. DISCOURSE COHERENCE & EXPLANATION DEPTH
        # ============================================================
        # Logical connectives indicate structured reasoning
        connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\bthis (?:means|implies|suggests)\b',
            r'\bthe (?:trade-off|tradeoff|difference|distinction)\b',
        ]
        connective_count = 0
        for pat in connectives:
            connective_count += len(re.findall(pat, resp_lower))
        
        connective_density = connective_count / max(sent_count, 1)
        # Reward moderate connective density
        if connective_density > 0.1:
            score += min(connective_density * 10, 7.0)
        
        # ============================================================
        # 7. RESPONSE LENGTH AND SUBSTANCE
        # ============================================================
        # Longer, more substantive responses tend to be better (up to a point)
        # But reward is logarithmic to avoid gaming
        length_score = math.log(max(word_count, 1) + 1) * 1.5
        score += min(length_score, 8.0)
        
        # Average sentence length - moderate is best (10-25 words)
        avg_sent_len = word_count / max(sent_count, 1)
        if 10 <= avg_sent_len <= 25:
            score += 3.0
        elif 7 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
            score += 1.5
        
        # ============================================================
        # 8. DOMAIN EXPERTISE SIGNALS
        # ============================================================
        # Technical/domain vocabulary (longer words as proxy)
        long_words = [w for w in words if len(w) > 8]
        long_word_ratio = len(long_words) / max(word_count, 1)
        if 0.05 <= long_word_ratio <= 0.3:
            score += long_word_ratio * 20
        
        # Parenthetical explanations (sign of pedagogical care)
        parentheticals = re.findall(r'\([^)]{5,}\)', resp)
        score += min(len(parentheticals) * 1.5, 5.0)
        
        # ============================================================
        # 9. ENGAGEMENT WITH QUERY
        # ============================================================
        # Check if response addresses query terms
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_str.lower()))
        resp_words_set = set(re.findall(r'\b[a-z]{4,}\b', resp_lower))
        
        if query_words:
            overlap = len(query_words & resp_words_set) / len(query_words)
            score += overlap * 5.0
        
        # ============================================================
        # 10. FIRST-PERSON EXPERIENCE vs AUTHORITATIVE TONE
        # ============================================================
        # Personal experience can be valuable but check for authority signals too
        personal_exp = len(re.findall(r'\bI (?:have|had|was|am|worked|found|think|believe|noticed|experienced)\b', resp))
        
        # If personal experience is combined with specifics, it's good
        if personal_exp > 0 and (specificity_score > 2 or entity_density > 0.02):
            score += min(personal_exp * 1.0, 4.0)
        
        # ============================================================
        # 11. STRUCTURAL SIGNALS
        # ============================================================
        # Code blocks (relevant for technical queries)
        has_code = '```' in resp or resp.count('    ') > 2
        query_is_technical = any(kw in query_str.lower() for kw in 
                                ['code', 'sql', 'table', 'create', 'select', 'function', 
                                 'program', 'script', 'algorithm', 'api', 'database'])
        if has_code and query_is_technical:
            score += 4.0
        
        # Enumeration / structured explanation
        has_enumeration = bool(re.search(r'(?:^|\n)\s*(?:\d+[\.\):]|\*|-)\s', resp))
        if has_enumeration:
            score += 2.0
        
        # ============================================================
        # 12. PENALIZE EMPTY/META RESPONSES
        # ============================================================
        # Responses that are just meta-commentary or moderation notices
        meta_patterns = [
            r'please read our rules', r'your (?:comment|post) (?:was|has been) removed',
            r'welcome to /r/', r'this is an automated',
            r'I cannot help', r'I\'m sorry,? (?:but )?I (?:can\'?t|cannot)',
        ]
        for pat in meta_patterns:
            if re.search(pat, resp_lower):
                score -= 10.0
                break
        
        # Very short non-substantive responses
        if word_count < 15 and hedge_count == 0 and connective_count == 0:
            score -= 5.0
        
        # Clamp to range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0