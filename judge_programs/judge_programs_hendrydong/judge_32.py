def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a substantially different approach:
    - Named entity density (capitalized multi-word phrases, dates, numbers)
    - Citation/reference pattern detection
    - Hallucination red-flag detection (absolute claims, unsourced precise stats)
    - Epistemic calibration (appropriate confidence levels)
    - Sensationalism/conspiracy language detection
    - Information density via unique noun-like tokens
    - Structural credibility signals (parentheticals, qualifiers, examples)
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
        query_text = query.strip()
        
        if len(resp) < 10:
            return 1.0
        
        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if s.strip()]
        sent_count = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint of 0-100
        
        # === 1. NAMED ENTITY / PROPER NOUN DENSITY ===
        # Detect capitalized words that aren't sentence starters
        proper_nouns = 0
        for i, w in enumerate(words):
            if i == 0:
                continue
            # Check if previous word ended a sentence
            if i > 0 and words[i-1][-1:] in '.!?':
                continue
            if w[0:1].isupper() and len(w) > 1 and w.isalpha():
                proper_nouns += 1
        
        proper_noun_rate = proper_nouns / max(word_count, 1)
        # Moderate proper noun density is good (suggests specific references)
        if proper_noun_rate > 0.02 and proper_noun_rate < 0.25:
            score += min(proper_noun_rate * 80, 6)
        
        # === 2. SPECIFIC NAMES, DATES, NUMBERS ===
        # Dates (years, specific dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', resp)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', resp)
        specific_dates = len(year_pattern) + len(date_pattern)
        score += min(specific_dates * 1.5, 6)
        
        # Numbers (but not overly precise unsourced ones)
        numbers = re.findall(r'\b\d+\.?\d*\b', resp)
        num_count = len(numbers)
        # Moderate use of numbers is good
        if 1 <= num_count <= 15:
            score += min(num_count * 0.8, 5)
        
        # Detect suspiciously precise statistics without attribution
        precise_stats = re.findall(r'\b\d{2,}\.\d{2,}%?\b', resp)
        if len(precise_stats) > 2:
            score -= len(precise_stats) * 2
        
        # === 3. CITATION / REFERENCE PATTERNS ===
        # Academic-style references
        citation_patterns = [
            r'(?:according to|as (?:noted|described|mentioned|stated|argued|shown) (?:by|in))',
            r'(?:u/\w+)',  # Reddit user references
            r'(?:\*[A-Z][^*]+\*)',  # Italicized titles
            r'(?:"[A-Z][^"]{3,}")',  # Quoted titles
            r'(?:St\.\s+\w+)',  # Saint references
            r'(?:Dr\.\s+\w+)',  # Doctor references
            r'(?:Prof(?:essor)?\.\s+\w+)',
            r'(?:Acts of the \w+)',
            r'(?:Book of \w+)',
            r'\b(?:chapter|section|verse|page)\s+\d+',
            r'(?:et al\.)',
            r'(?:cf\.|ibid|op\.?\s*cit)',
        ]
        citation_count = 0
        for pat in citation_patterns:
            citation_count += len(re.findall(pat, resp, re.IGNORECASE))
        score += min(citation_count * 3, 10)
        
        # Named works / titles (words in italics or quotes)
        title_refs = re.findall(r'\*[^*]+\*|"[^"]{4,}"', resp)
        score += min(len(title_refs) * 2.5, 7)
        
        # === 4. HALLUCINATION RED FLAGS ===
        absolute_claims = [
            r'\b(?:always|never|every single|without exception|100%|guaranteed)\b',
            r'\b(?:everyone knows|it is a fact that|undeniably|unquestionably)\b',
            r'\b(?:the truth is|the reality is|obviously|clearly)\b',
        ]
        absolute_count = 0
        for pat in absolute_claims:
            absolute_count += len(re.findall(pat, resp, re.IGNORECASE))
        # Mild penalty for absolute language
        score -= min(absolute_count * 2, 8)
        
        # === 5. SENSATIONALISM / CONSPIRACY DETECTION ===
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'mind-blowing', 'insane',
            'they don\'t want you to know', 'wake up', 'sheeple', 'cover-up',
            'mainstream media', 'big pharma', 'deep state', 'false flag',
            'hoax', 'scam', 'brainwash', 'propaganda', 'agenda',
            'conspiracy', 'illuminati', 'cabal', 'puppet masters',
            'exposed', 'bombshell', 'terrifying', 'horrifying',
        ]
        resp_lower = resp.lower()
        sensational_count = sum(1 for w in sensational_words if w in resp_lower)
        score -= sensational_count * 4
        
        # === 6. EPISTEMIC CALIBRATION ===
        # Appropriate hedging (shows intellectual honesty)
        calibration_phrases = [
            r'\b(?:it (?:seems|appears) (?:that|to))',
            r'\b(?:might|may|could|possibly|perhaps|likely|unlikely)\b',
            r'\b(?:in my (?:experience|understanding|view))\b',
            r'\b(?:tend(?:s)? to)\b',
            r'\b(?:generally|typically|usually|often|sometimes)\b',
            r'\b(?:one (?:possible|potential) (?:explanation|reason|interpretation))\b',
            r'\b(?:there\'s a chance|there is a chance)\b',
            r'\b(?:if (?:I\'m|I am) not mistaken)\b',
            r'\b(?:as far as I (?:know|understand|can tell))\b',
            r'\b(?:I (?:think|believe|suspect|imagine|suppose))\b',
            r'\b(?:not necessarily)\b',
            r'\b(?:it depends)\b',
            r'\b(?:ceteris paribus)\b',
            r'\b(?:more or less)\b',
        ]
        hedge_count = 0
        for pat in calibration_phrases:
            hedge_count += len(re.findall(pat, resp, re.IGNORECASE))
        
        # Good calibration: some hedging but not excessive
        if 1 <= hedge_count <= 8:
            score += hedge_count * 2
        elif hedge_count > 8:
            score += 10 - (hedge_count - 8)  # Diminishing returns / slight penalty for over-hedging
        
        # === 7. INFORMATION DENSITY (unique content words) ===
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'it', 'its', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'like', 'get', 'got', 'much',
        }
        
        content_words = [w.lower().strip('.,!?;:()[]{}"\'-') for w in words 
                         if w.lower().strip('.,!?;:()[]{}"\'-') not in stop_words 
                         and len(w.strip('.,!?;:()[]{}"\'-')) > 2]
        
        unique_content = set(content_words)
        content_density = len(unique_content) / max(word_count, 1)
        
        # Higher content density suggests more informative response
        score += min(content_density * 20, 8)
        
        # === 8. STRUCTURAL CREDIBILITY SIGNALS ===
        # Parenthetical asides (often used for clarification/precision)
        parentheticals = re.findall(r'\([^)]{3,}\)', resp)
        score += min(len(parentheticals) * 1.5, 5)
        
        # Use of examples ("for example", "e.g.", "such as", "for instance")
        example_phrases = re.findall(
            r'\b(?:for example|for instance|e\.g\.|such as|specifically|in particular)\b',
            resp, re.IGNORECASE
        )
        score += min(len(example_phrases) * 2, 6)
        
        # Causal/logical connectors
        logical_connectors = re.findall(
            r'\b(?:because|therefore|however|although|moreover|furthermore|consequently|thus|hence|nonetheless|nevertheless|whereas|meanwhile|in contrast|on the other hand)\b',
            resp, re.IGNORECASE
        )
        score += min(len(logical_connectors) * 1.2, 6)
        
        # === 9. RESPONSE DEPTH / ENGAGEMENT ===
        # Average sentence length (moderate is best: 10-25 words)
        avg_sent_len = word_count / sent_count
        if 10 <= avg_sent_len <= 25:
            score += 4
        elif 7 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
            score += 2
        # Very short or very long sentences are less credible
        
        # Response length bonus (longer responses tend to be more thorough, up to a point)
        length_score = min(math.log(word_count + 1, 2) * 1.5, 12)
        score += length_score
        
        # === 10. QUERY-RESPONSE TOPICAL ALIGNMENT ===
        # Check that response addresses the query topic using content word overlap
        query_words = set(
            w.lower().strip('.,!?;:()[]{}"\'-') for w in query_text.split()
            if w.lower().strip('.,!?;:()[]{}"\'-') not in stop_words
            and len(w.strip('.,!?;:()[]{}"\'-')) > 2
        )
        
        if query_words:
            overlap = len(unique_content & query_words)
            overlap_ratio = overlap / max(len(query_words), 1)
            # Some overlap is good (shows relevance) but we don't over-weight this
            score += min(overlap_ratio * 8, 5)
        
        # === 11. FIRST-PERSON EXPERIENCE SIGNALS ===
        # Personal experience can add credibility in certain contexts
        experience_patterns = re.findall(
            r'\b(?:in my experience|I(?:\'ve| have) (?:seen|found|noticed|worked|been)|I (?:am|was) a)\b',
            resp, re.IGNORECASE
        )
        score += min(len(experience_patterns) * 1.5, 4)
        
        # === 12. SPECIFICITY MARKERS ===
        # Domain-specific or technical terms (longer, less common words)
        long_words = [w for w in content_words if len(w) >= 8]
        long_word_rate = len(long_words) / max(word_count, 1)
        score += min(long_word_rate * 30, 6)
        
        # === 13. DISCLAIMER / META-COMMENTARY PENALTY ===
        # Responses that are mostly disclaimers or meta-commentary rather than substance
        meta_patterns = [
            r'welcome to /r/',
            r'please read our rules',
            r'your (?:comment|post) (?:will be|has been) removed',
            r'this (?:is|was) (?:automatically|auto) generated',
            r'I am a bot',
        ]
        for pat in meta_patterns:
            if re.search(pat, resp, re.IGNORECASE):
                score -= 15
        
        # === 14. MULTI-PERSPECTIVE / NUANCE ===
        contrast_markers = re.findall(
            r'\b(?:on the other hand|alternatively|conversely|in contrast|while .+ also|both .+ and)\b',
            resp, re.IGNORECASE
        )
        score += min(len(contrast_markers) * 2.5, 5)
        
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0