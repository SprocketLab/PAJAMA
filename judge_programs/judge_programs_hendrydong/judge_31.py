def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a different approach:
    - Named entity density (capitalized multi-word phrases, dates, numbers)
    - Citation/reference patterns (quotes, attributions, source mentions)
    - Hallucination red-flags (absolute claims, sensationalism, conspiracy language)
    - Epistemic calibration (appropriate certainty vs uncertainty markers)
    - Structural credibility (balanced reasoning, counterpoints, qualifications)
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
        if len(resp) < 5:
            return 0.0
        
        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. NAMED ENTITY / SPECIFICITY DENSITY (unique approach)
        # ============================================================
        # Detect capitalized phrases (potential proper nouns / named entities)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', resp)
        cap_phrase_density = len(cap_phrases) / max(word_count, 1) * 100
        score += min(cap_phrase_density * 3.0, 6.0)
        
        # Detect dates in various formats
        date_patterns = [
            r'\b\d{4}\b',  # years like 1250, 2023
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
            r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:century|Century)\b',
        ]
        date_count = 0
        for pat in date_patterns:
            date_count += len(re.findall(pat, resp))
        score += min(date_count * 1.5, 5.0)
        
        # Detect specific numbers (not just dates)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', resp)
        number_density = len(numbers) / max(word_count, 1) * 100
        score += min(number_density * 1.0, 4.0)
        
        # ============================================================
        # 2. CITATION / ATTRIBUTION PATTERNS
        # ============================================================
        citation_markers = [
            r'according to', r'as (?:noted|stated|described|mentioned) (?:by|in)',
            r'(?:research|studies|evidence) (?:shows?|suggests?|indicates?)',
            r'\bu/\w+',  # Reddit usernames
            r'(?:Dr\.|Prof\.|Professor)\s+[A-Z]',
            r'\*[^*]+\*',  # italicized titles
            r'"[^"]{5,}"',  # quoted text
            r'(?:book|paper|article|study) (?:by|titled|called)',
            r'(?:Chapter|Section|Part)\s+\d',
        ]
        citation_count = 0
        resp_lower = resp.lower()
        for pat in citation_markers:
            citation_count += len(re.findall(pat, resp, re.IGNORECASE))
        score += min(citation_count * 2.5, 8.0)
        
        # Detect inline references like (Author, Year) or [1]
        inline_refs = re.findall(r'\([A-Z][a-z]+(?:\s+(?:et al\.?|&|and)\s+[A-Z][a-z]+)?,?\s*\d{4}\)', resp)
        inline_refs += re.findall(r'\[\d+\]', resp)
        score += min(len(inline_refs) * 2.0, 5.0)
        
        # ============================================================
        # 3. EPISTEMIC CALIBRATION (nuanced certainty/uncertainty)
        # ============================================================
        # Appropriate hedging (good - shows calibration)
        hedging_phrases = [
            'might', 'could', 'possibly', 'perhaps', 'likely', 'unlikely',
            'it seems', 'it appears', 'tends to', 'generally', 'typically',
            'in most cases', 'often', 'sometimes', 'arguably', 'presumably',
            'to my knowledge', 'as far as i know', 'i believe', 'i think',
            'it depends', 'not necessarily', 'in some cases', 'may',
            'probably', 'roughly', 'approximately', 'around', 'about',
            'one possibility', 'it\'s worth noting', 'keep in mind',
            'that said', 'however', 'on the other hand', 'although',
            'while', 'whereas', 'but', 'nonetheless', 'nevertheless',
        ]
        hedge_count = 0
        for phrase in hedging_phrases:
            hedge_count += resp_lower.count(phrase)
        
        # Moderate hedging is good; too much or too little is bad
        hedge_ratio = hedge_count / max(sent_count, 1)
        if 0.2 <= hedge_ratio <= 2.0:
            score += 5.0  # Sweet spot
        elif hedge_ratio > 0:
            score += 2.0  # Some hedging
        # No hedging: no bonus
        
        # ============================================================
        # 4. HALLUCINATION RED FLAGS (penalize)
        # ============================================================
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+%', resp)
        score -= len(precise_stats) * 2.0
        
        # Absolute/sensational language
        absolute_terms = [
            'always', 'never', 'absolutely', 'definitely', 'certainly',
            'without a doubt', 'guaranteed', 'proven fact', 'undeniable',
            'everyone knows', 'it is known', 'obvious', 'clearly',
            'no one', 'nothing', 'everything', 'all experts agree',
        ]
        absolute_count = 0
        for term in absolute_terms:
            absolute_count += resp_lower.count(term)
        score -= min(absolute_count * 1.5, 6.0)
        
        # Conspiracy / sensationalism language
        conspiracy_terms = [
            'they don\'t want you to know', 'mainstream media', 'cover up',
            'coverup', 'big pharma', 'wake up', 'sheeple', 'hoax',
            'conspiracy', 'suppressed', 'hidden truth', 'secret agenda',
            'deep state', 'false flag', 'mind control', 'brainwash',
            'shocking', 'unbelievable', 'jaw-dropping', 'bombshell',
        ]
        conspiracy_count = 0
        for term in conspiracy_terms:
            conspiracy_count += resp_lower.count(term)
        score -= conspiracy_count * 4.0
        
        # ============================================================
        # 5. REASONING STRUCTURE / DEPTH INDICATORS
        # ============================================================
        # Causal reasoning markers
        causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'given that', 'this means',
            'which leads to', 'the reason', 'this is because', 'so that',
            'in order to', 'for this reason', 'it follows',
        ]
        causal_count = 0
        for marker in causal_markers:
            causal_count += resp_lower.count(marker)
        score += min(causal_count * 1.2, 6.0)
        
        # Contrastive/balanced reasoning
        contrast_markers = [
            'however', 'on the other hand', 'conversely', 'in contrast',
            'alternatively', 'that said', 'nonetheless', 'despite',
            'although', 'while this', 'but', 'yet', 'still',
        ]
        contrast_count = 0
        for marker in contrast_markers:
            contrast_count += resp_lower.count(marker)
        score += min(contrast_count * 1.5, 5.0)
        
        # Example/evidence markers
        example_markers = [
            'for example', 'for instance', 'such as', 'e.g.', 'i.e.',
            'specifically', 'in particular', 'namely', 'to illustrate',
            'consider', 'take the case', 'one example',
        ]
        example_count = 0
        for marker in example_markers:
            example_count += resp_lower.count(marker)
        score += min(example_count * 2.0, 6.0)
        
        # ============================================================
        # 6. INFORMATION DENSITY (content vs filler ratio)
        # ============================================================
        filler_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'our', 'their', 'and', 'or', 'but',
            'not', 'no', 'so', 'if', 'as', 'just', 'very', 'really',
            'also', 'too', 'up', 'out', 'about', 'there', 'here',
        }
        words_lower = [w.lower().strip('.,;:!?()[]"\'') for w in words]
        content_words = [w for w in words_lower if w and w not in filler_words and len(w) > 2]
        content_ratio = len(content_words) / max(word_count, 1)
        
        # Good content density is around 0.4-0.7
        if 0.35 <= content_ratio <= 0.75:
            score += 3.0
        elif content_ratio > 0.75:
            score += 1.0  # Might be too dense / jargon-heavy
        
        # Unique content word ratio (vocabulary richness for content words)
        if content_words:
            unique_content = len(set(content_words)) / len(content_words)
            score += min(unique_content * 4.0, 4.0)
        
        # ============================================================
        # 7. RESPONSE SUBSTANTIVENESS
        # ============================================================
        # Longer, more detailed responses tend to be more informative
        # But with diminishing returns
        length_score = math.log(max(word_count, 1) + 1) * 1.5
        score += min(length_score, 8.0)
        
        # Average sentence length (moderate is best for clarity)
        avg_sent_len = word_count / max(sent_count, 1)
        if 10 <= avg_sent_len <= 25:
            score += 3.0
        elif 8 <= avg_sent_len <= 30:
            score += 1.5
        
        # ============================================================
        # 8. DOMAIN-SPECIFIC KNOWLEDGE SIGNALS
        # ============================================================
        # Technical/domain terms (longer words often indicate specificity)
        long_words = [w for w in words_lower if len(w) >= 8]
        long_word_ratio = len(long_words) / max(word_count, 1)
        score += min(long_word_ratio * 15, 5.0)
        
        # Parenthetical clarifications (sign of careful explanation)
        parentheticals = re.findall(r'\([^)]{3,}\)', resp)
        score += min(len(parentheticals) * 1.0, 3.0)
        
        # ============================================================
        # 9. QUERY RELEVANCE (semantic connection check)
        # ============================================================
        query_words = set(query.lower().split())
        query_content = {w.strip('.,;:!?()[]"\'') for w in query_words 
                        if w.strip('.,;:!?()[]"\'') not in filler_words and len(w.strip('.,;:!?()[]"\'')) > 2}
        resp_content_set = set(content_words)
        
        if query_content:
            # Check how many query content words appear in response
            overlap = len(query_content & resp_content_set)
            relevance = overlap / max(len(query_content), 1)
            score += min(relevance * 8.0, 6.0)
        
        # ============================================================
        # 10. PENALIZE LOW-EFFORT INDICATORS
        # ============================================================
        # Very short responses
        if word_count < 15:
            score -= 8.0
        elif word_count < 30:
            score -= 4.0
        
        # Responses that are mostly meta-commentary (not answering)
        meta_phrases = [
            'please read our rules', 'your comment', 'will be removed',
            'welcome to', 'this is a bot', 'i am a bot',
            'while you wait for an answer', 'you might be interested',
        ]
        meta_count = 0
        for phrase in meta_phrases:
            if phrase in resp_lower:
                meta_count += 1
        score -= meta_count * 5.0
        
        # Responses that defer rather than answer
        defer_phrases = [
            'i can\'t help', 'i don\'t know', 'i\'m not sure about that',
            'you should ask', 'try googling', 'i have no idea',
        ]
        defer_count = 0
        for phrase in defer_phrases:
            if phrase in resp_lower:
                defer_count += 1
        score -= defer_count * 3.0
        
        # ============================================================
        # 11. PERSONAL EXPERIENCE / ANECDOTAL EVIDENCE SIGNALS
        # ============================================================
        # First-person experience can add credibility in certain contexts
        experience_markers = [
            'in my experience', 'i have', 'i\'ve', 'when i',
            'i worked', 'i found', 'i noticed', 'i learned',
            'years of', 'in practice', 'from what i\'ve seen',
        ]
        exp_count = 0
        for marker in experience_markers:
            if marker in resp_lower:
                exp_count += 1
        score += min(exp_count * 1.0, 3.0)
        
        # ============================================================
        # 12. MULTI-PERSPECTIVE / THOROUGHNESS
        # ============================================================
        # Enumeration patterns (covering multiple points)
        enum_patterns = re.findall(r'(?:first|second|third|fourth|1\.|2\.|3\.|4\.|\d\))', resp_lower)
        if len(enum_patterns) >= 2:
            score += 3.0
        
        # Multiple paragraph structure
        paragraphs = [p.strip() for p in resp.split('\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 2.0
        if len(paragraphs) >= 3:
            score += 1.0
        
        # Clamp score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0