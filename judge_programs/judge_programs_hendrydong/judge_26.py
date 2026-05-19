def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Strategy: Analyze response through multiple factual reliability lenses:
    1. Specificity signals (named entities, dates, numbers, proper nouns)
    2. Source/reference indicators (citations, quotes, attributions)
    3. Hallucination red flags (overly absolute claims, unsourced statistics)
    4. Appropriate epistemic calibration (hedging where needed, confidence where warranted)
    5. Discourse sophistication (reasoning chains, nuanced qualifications)
    6. Sensationalism/conspiracy detection
    
    This variant focuses on TOKEN-LEVEL classification and ratio-based scoring
    rather than pattern counting or structural analysis.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        resp = response.strip()
        if len(resp) < 5:
            return 0.5
        
        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.5
        
        # Lowercase version for matching
        resp_lower = resp.lower()
        words_lower = [w.lower() for w in words]
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. NAMED ENTITY DENSITY (proper nouns, specific references)
        # ============================================================
        # Detect capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]+\s*', resp)
        non_start_caps = 0
        total_non_start = 0
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                if i == 0:
                    continue
                total_non_start += 1
                # Check if word starts with uppercase and isn't all caps (acronym handled separately)
                if w and w[0].isupper() and not w.isupper() and len(w) > 1:
                    non_start_caps += 1
        
        if total_non_start > 0:
            proper_noun_ratio = non_start_caps / total_non_start
            # Reward moderate density of proper nouns (0.02-0.15 is good)
            if 0.02 <= proper_noun_ratio <= 0.20:
                score += proper_noun_ratio * 40  # up to +8
            elif proper_noun_ratio > 0.20:
                score += 5  # still some credit but diminishing
        
        # ============================================================
        # 2. NUMERIC SPECIFICITY
        # ============================================================
        # Find numbers in response (dates, quantities, measurements)
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', resp)
        num_count = len(numbers)
        
        # Dates (years specifically)
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', resp)
        year_count = len(years)
        
        # Reward specific numbers proportionally
        num_density = num_count / max(word_count, 1)
        if 0.005 < num_density < 0.08:
            score += num_density * 120  # up to ~9.6
        elif num_density >= 0.08:
            # Too many numbers without context might be suspicious
            score += 4
        
        # Extra credit for year references (historical specificity)
        score += min(year_count * 1.5, 6)
        
        # ============================================================
        # 3. ATTRIBUTION AND SOURCE SIGNALS
        # ============================================================
        attribution_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bfound that\b', r'\bevidence\b', r'\bdata\b',
            r'\bu/\w+', r'\b(?:dr|prof|professor)\b\.?\s',
            r'\bpublished\b', r'\bjournal\b', r'\breport(?:ed|s)?\b',
            r'\bsurvey\b', r'\banalysis\b', r'\bexpert\b',
            r'\bhistorian\b', r'\bscientist\b', r'\bscholar\b',
        ]
        
        attribution_count = 0
        for pat in attribution_patterns:
            attribution_count += len(re.findall(pat, resp_lower))
        
        score += min(attribution_count * 1.8, 8)
        
        # Book/work titles (italic markers or quoted titles)
        title_refs = re.findall(r'\*[A-Z][^*]+\*', resp)  # *Title*
        title_refs += re.findall(r'"[A-Z][^"]{3,}"', resp)  # "Title"
        score += min(len(title_refs) * 2.5, 6)
        
        # ============================================================
        # 4. EPISTEMIC CALIBRATION SCORING
        # ============================================================
        # Appropriate hedging words (shows intellectual honesty)
        calibration_words = [
            'typically', 'generally', 'tends to', 'often', 'usually',
            'might', 'could', 'possibly', 'likely', 'unlikely',
            'it seems', 'it appears', 'arguably', 'perhaps',
            'in some cases', 'depending on', 'varies', 'nuanced',
            'complex', 'debatable', 'controversial', 'uncertain',
            'roughly', 'approximately', 'about', 'around',
            'in my experience', 'from what i', 'as far as i know',
            'i believe', 'i think', 'may be', 'can be',
        ]
        
        calibration_count = 0
        for phrase in calibration_words:
            calibration_count += resp_lower.count(phrase)
        
        # Moderate calibration is good; too much is wishy-washy
        cal_ratio = calibration_count / max(word_count / 50, 1)
        if 0.5 <= cal_ratio <= 4:
            score += cal_ratio * 2.5
        elif cal_ratio > 4:
            score += 5  # cap benefit
        else:
            # Some credit for any calibration
            score += calibration_count * 0.8
        
        # ============================================================
        # 5. HALLUCINATION RED FLAGS (penalize)
        # ============================================================
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{1,2}\.\d{1,2}%\b', resp)  # e.g., "73.42%"
        score -= len(precise_stats) * 2
        
        # Absolute/universal claims
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\beveryone knows\b',
            r'\bit is certain\b', r'\bundeniably\b', r'\bwithout a doubt\b',
            r'\bobviously\b', r'\bclearly\b(?!\s+(?:defined|stated|written))',
            r'\bno one\b(?!\s+(?:should|would|could))',
            r'\ball\s+(?:people|humans|scientists|experts)\b',
            r'\bthe truth is\b', r'\bthe fact is\b',
        ]
        
        absolute_count = 0
        for pat in absolute_patterns:
            absolute_count += len(re.findall(pat, resp_lower))
        
        # Mild penalty for absolutes
        score -= min(absolute_count * 1.2, 6)
        
        # ============================================================
        # 6. SENSATIONALISM / CONSPIRACY DETECTION (penalize)
        # ============================================================
        sensational_terms = [
            'shocking', 'bombshell', 'wake up', 'sheeple', 'they don\'t want you to know',
            'mainstream media', 'cover-up', 'coverup', 'big pharma', 'deep state',
            'conspiracy', 'suppressed', 'censored', 'hidden truth', 'secret agenda',
            'mind-blowing', 'unbelievable', 'you won\'t believe', 'exposed',
            'whistleblower', 'false flag', 'puppet master', 'new world order',
            'illuminati', 'globalist', 'controlled opposition',
        ]
        
        sensational_count = 0
        for term in sensational_terms:
            sensational_count += resp_lower.count(term)
        
        score -= sensational_count * 4
        
        # Excessive exclamation marks
        exclamation_count = resp.count('!')
        if exclamation_count > 2:
            score -= (exclamation_count - 2) * 0.8
        
        # ALL CAPS words (shouting = sensationalism)
        caps_words = [w for w in words if w.isupper() and len(w) > 2 and not re.match(r'^[A-Z]{2,4}$', w)]
        score -= min(len(caps_words) * 1.5, 5)
        
        # ============================================================
        # 7. REASONING CHAIN INDICATORS
        # ============================================================
        reasoning_markers = [
            'because', 'therefore', 'however', 'although', 'while',
            'on the other hand', 'in contrast', 'for example', 'for instance',
            'specifically', 'in particular', 'moreover', 'furthermore',
            'nevertheless', 'nonetheless', 'consequently', 'thus',
            'this means', 'this suggests', 'this implies', 'in other words',
            'the reason', 'one reason', 'another', 'first', 'second',
            'essentially', 'fundamentally', 'the trade-off', 'trade off',
        ]
        
        reasoning_count = 0
        for marker in reasoning_markers:
            reasoning_count += resp_lower.count(marker)
        
        reasoning_density = reasoning_count / max(word_count / 30, 1)
        score += min(reasoning_density * 3, 8)
        
        # ============================================================
        # 8. RESPONSE SUBSTANTIVENESS
        # ============================================================
        # Unique word ratio (vocabulary richness)
        unique_words = set(words_lower)
        vocab_richness = len(unique_words) / max(word_count, 1)
        
        # Reward moderate richness (0.5-0.85 is typical for good content)
        if vocab_richness > 0.4:
            score += (vocab_richness - 0.4) * 8
        
        # Response length consideration (not too short, not just padding)
        # Moderate length bonus with diminishing returns
        if word_count >= 15:
            length_bonus = math.log(word_count / 15 + 1) * 3
            score += min(length_bonus, 8)
        elif word_count < 10:
            score -= 3
        
        # ============================================================
        # 9. DOMAIN-SPECIFIC TERMINOLOGY DENSITY
        # ============================================================
        # Words with 3+ syllables (proxy for technical vocabulary)
        # Simple syllable estimation: count vowel groups
        def est_syllables(word):
            w = word.lower().strip(".,!?;:'\"")
            count = len(re.findall(r'[aeiouy]+', w))
            return max(count, 1)
        
        complex_words = sum(1 for w in words if est_syllables(w) >= 3 and len(w) > 5)
        complex_ratio = complex_words / max(word_count, 1)
        
        if 0.08 < complex_ratio < 0.35:
            score += complex_ratio * 20  # up to ~7
        
        # ============================================================
        # 10. QUERY RELEVANCE (topical alignment)
        # ============================================================
        query_words = set(query.lower().split())
        # Remove very common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                     'than', 'too', 'very', 'just', 'because', 'if', 'when', 'what',
                     'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                     'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
                     'how', 'why', 'where', 'am', 'about', 'up', 'out', 'like'}
        
        query_content_words = query_words - stopwords
        resp_content_words = set(words_lower) - stopwords
        
        if query_content_words:
            overlap = query_content_words & resp_content_words
            relevance = len(overlap) / len(query_content_words)
            score += relevance * 6  # up to +6
        
        # ============================================================
        # 11. CONVERSATIONAL ENGAGEMENT vs DEFLECTION
        # ============================================================
        # Penalize responses that deflect rather than answer
        deflection_patterns = [
            r'^welcome to\b', r'\bplease read our rules\b',
            r'\bthis question has been asked\b', r'\bi cannot\b',
            r'\bi\'m not able to\b', r'\bi don\'t have enough\b',
            r'^do not fear\b', r'\bwhile you wait\b',
        ]
        
        for pat in deflection_patterns:
            if re.search(pat, resp_lower):
                score -= 5
        
        # Reward direct engagement with the question
        if resp_lower.startswith(('yes', 'no,', 'essentially', 'the', 'in ', 'as ', 'if ', 'being', 'a lot', 'for')):
            score += 2
        
        # ============================================================
        # 12. CONDITIONAL/CONTEXTUAL REASONING
        # ============================================================
        conditional_patterns = [
            r'\bif\s+you\b', r'\bdepending on\b', r'\bin the case\b',
            r'\bwhen\s+\w+ing\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bprovided\b', r'\bunless\b', r'\bexcept\b',
            r'\bceteris paribus\b', r'\ball else (?:being )?equal\b',
        ]
        
        conditional_count = 0
        for pat in conditional_patterns:
            conditional_count += len(re.findall(pat, resp_lower))
        
        score += min(conditional_count * 1.5, 5)
        
        # ============================================================
        # 13. MULTI-PERSPECTIVE / NUANCE INDICATORS
        # ============================================================
        perspective_markers = [
            'on the other hand', 'alternatively', 'some argue',
            'others believe', 'one perspective', 'another view',
            'both', 'while some', 'it depends', 'there are',
            'multiple', 'various', 'different', 'range of',
            'spectrum', 'tradition', 'perspective',
        ]
        
        perspective_count = sum(1 for m in perspective_markers if m in resp_lower)
        score += min(perspective_count * 1.5, 5)
        
        # ============================================================
        # 14. SENTENCE COMPLEXITY AND VARIETY
        # ============================================================
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) >= 2:
            import statistics
            try:
                sent_stdev = statistics.stdev(sentence_lengths)
                avg_sent_len = statistics.mean(sentence_lengths)
                # Reward sentence length variety (indicates sophisticated writing)
                if sent_stdev > 3:
                    score += min(sent_stdev * 0.5, 3)
                # Reward moderate average sentence length
                if 8 <= avg_sent_len <= 25:
                    score += 2
            except:
                pass
        
        # Clamp final score
        score = max(0, min(100, score))
        
        # Rescale to 0-10 range for cleaner output
        final_score = score / 10.0
        
        return round(final_score, 2)
    
    except Exception as e:
        # Fallback: return neutral score
        try:
            return max(1.0, min(5.0, len(str(response)) / 100.0))
        except:
            return 3.0