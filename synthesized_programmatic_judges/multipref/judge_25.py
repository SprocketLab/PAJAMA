def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    Higher scores = better factual reliability indicators.
    
    Strategy: Analyze specificity of claims, appropriate hedging, citation-like patterns,
    structured presentation, absence of hallucination red flags, and sensationalism.
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
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. SPECIFIC FACTUAL MARKERS (names, dates, numbers, units)
        # ============================================================
        
        # Specific numbers (not just "1" or "2" but meaningful quantities)
        number_pattern = r'\b\d{2,}\b'
        numbers_found = re.findall(number_pattern, response)
        number_score = min(len(numbers_found) * 0.8, 6.0)
        score += number_score
        
        # Dates (years, specific dates)
        year_pattern = r'\b(1[0-9]{3}|20[0-2][0-9])\b'
        years_found = re.findall(year_pattern, response)
        score += min(len(years_found) * 1.5, 5.0)
        
        # Specific date formats
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates_found = re.findall(date_pattern, response)
        score += min(len(dates_found) * 1.0, 3.0)
        
        # Units of measurement (indicators of precision)
        unit_patterns = [
            r'\b\d+\s*(kg|lb|lbs|g|mg|oz|km|mi|miles|m|cm|mm|ft|feet|inches|in)\b',
            r'\b\d+\s*(mph|km/h|m/s|kph)\b',
            r'\b\d+\s*(degrees?|°|celsius|fahrenheit|kelvin)\b',
            r'\b\d+\s*(hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b',
            r'\b\d+\s*(%|percent)\b',
            r'\b\d+\s*(calories|kcal|watts?|volts?|amps?)\b',
        ]
        units_count = 0
        for pat in unit_patterns:
            units_count += len(re.findall(pat, response_lower))
        score += min(units_count * 1.2, 5.0)
        
        # ============================================================
        # 2. STRUCTURED PRESENTATION (lists, headers, steps)
        # ============================================================
        
        # Numbered lists
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        has_numbered_list = len(numbered_items) >= 2
        if has_numbered_list:
            score += min(len(numbered_items) * 0.5, 4.0)
        
        # Bullet points
        bullet_items = re.findall(r'(?:^|\n)\s*[-•*]\s', response)
        if len(bullet_items) >= 2:
            score += min(len(bullet_items) * 0.3, 3.0)
        
        # Markdown headers
        headers = re.findall(r'#{1,4}\s+\S', response)
        if headers:
            score += min(len(headers) * 1.0, 4.0)
        
        # Bold formatting (indicates organized thought)
        bold_items = re.findall(r'\*\*[^*]+\*\*', response)
        if bold_items:
            score += min(len(bold_items) * 0.4, 3.0)
        
        # ============================================================
        # 3. APPROPRIATE HEDGING LANGUAGE
        # ============================================================
        
        hedging_phrases = [
            'may', 'might', 'could', 'possibly', 'potentially',
            'it is possible', 'it\'s possible', 'generally', 'typically',
            'often', 'usually', 'in most cases', 'tends to', 'likely',
            'approximately', 'roughly', 'about', 'around',
            'depending on', 'it depends', 'varies',
            'some experts', 'research suggests', 'studies suggest',
            'it appears', 'it seems', 'in general',
            'can be', 'may be', 'might be',
        ]
        
        hedge_count = 0
        for phrase in hedging_phrases:
            hedge_count += response_lower.count(phrase)
        
        # Moderate hedging is good; too much is wishy-washy
        hedge_ratio = hedge_count / max(num_words, 1) * 100
        if 0.5 <= hedge_ratio <= 5.0:
            score += 4.0
        elif 0.1 <= hedge_ratio < 0.5:
            score += 2.0
        elif hedge_ratio > 5.0:
            score += 1.0  # Over-hedging
        
        # ============================================================
        # 4. CITATION / REFERENCE INDICATORS
        # ============================================================
        
        citation_phrases = [
            'according to', 'research shows', 'studies show',
            'research indicates', 'evidence suggests',
            'data shows', 'data suggests', 'published in',
            'reported by', 'as noted by', 'as stated by',
            'based on', 'findings suggest', 'literature suggests',
            'source:', 'reference:', 'cited in',
        ]
        
        citation_count = 0
        for phrase in citation_phrases:
            citation_count += response_lower.count(phrase)
        score += min(citation_count * 2.0, 5.0)
        
        # Parenthetical references or bracketed citations
        paren_refs = re.findall(r'\([A-Z][a-z]+(?:\s+(?:et al\.?|and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}\)', response)
        bracket_refs = re.findall(r'\[\d+\]', response)
        score += min((len(paren_refs) + len(bracket_refs)) * 1.5, 4.0)
        
        # URLs
        urls = re.findall(r'https?://\S+', response)
        score += min(len(urls) * 1.0, 2.0)
        
        # ============================================================
        # 5. HALLUCINATION RED FLAGS (penalize)
        # ============================================================
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d+\.\d{2,}\s*%', response)
        score -= len(precise_stats) * 1.5
        
        # Absolute claims without evidence
        absolute_phrases = [
            'it is a fact that', 'it is proven that', 'everyone knows',
            'it has been proven', 'undeniably', 'without a doubt',
            'there is no question', 'absolutely certain',
            'it is certain that', 'definitely true',
            'always works', 'never fails', 'guaranteed to',
            'the truth is', 'the fact is',
            '100% effective', '100% safe', '100% accurate',
        ]
        
        absolute_count = 0
        for phrase in absolute_phrases:
            absolute_count += response_lower.count(phrase)
        score -= absolute_count * 2.5
        
        # Fabricated-sounding specific claims (very specific numbers without context)
        # e.g., "exactly 73.847% of people..."
        fabricated_stats = re.findall(r'exactly\s+\d+\.?\d*\s*%', response_lower)
        score -= len(fabricated_stats) * 3.0
        
        # ============================================================
        # 6. SENSATIONALISM / CONSPIRACY RED FLAGS (penalize)
        # ============================================================
        
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'mind-blowing',
            'unbelievable', 'jaw-dropping', 'insane', 'crazy',
            'they don\'t want you to know', 'the government is hiding',
            'big pharma', 'cover-up', 'coverup', 'wake up sheeple',
            'mainstream media won\'t tell you', 'deep state',
            'conspiracy', 'illuminati', 'new world order',
            'false flag', 'hoax', 'brainwash', 'propaganda',
            'exposed!', 'revealed!', 'secret agenda',
            'you won\'t believe', 'doctors hate', 'one weird trick',
        ]
        
        sensational_count = 0
        for phrase in sensational_words:
            sensational_count += response_lower.count(phrase)
        score -= sensational_count * 3.0
        
        # Excessive exclamation marks (sensationalism indicator)
        exclamation_count = response.count('!')
        exclamation_ratio = exclamation_count / max(num_words, 1) * 100
        if exclamation_ratio > 2.0:
            score -= min(exclamation_ratio * 1.5, 6.0)
        
        # ALL CAPS words (shouting/sensationalism)
        caps_words = [w for w in response.split() if w.isupper() and len(w) > 2 and not re.match(r'^[A-Z]{1,3}$', w)]
        if len(caps_words) > 3:
            score -= min(len(caps_words) * 0.5, 4.0)
        
        # ============================================================
        # 7. RESPONSE SUBSTANCE AND DEPTH
        # ============================================================
        
        # Adequate length (not too short, not excessively padded)
        if num_words < 20:
            score -= 5.0
        elif num_words < 50:
            score -= 2.0
        elif 50 <= num_words <= 600:
            score += 3.0
        elif num_words > 600:
            score += 1.5  # Long but might be padded
        
        # Sentence count and variety
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences >= 3:
            # Average sentence length variety (good writing indicator)
            sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
            if len(sent_lengths) >= 2:
                mean_len = sum(sent_lengths) / len(sent_lengths)
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Some variety in sentence length is good
                if 2.0 <= std_dev <= 15.0:
                    score += 2.0
        
        # ============================================================
        # 8. VOCABULARY SOPHISTICATION (domain-appropriate language)
        # ============================================================
        
        # Technical/precise vocabulary indicators
        precise_vocab = [
            'specifically', 'particularly', 'furthermore', 'moreover',
            'however', 'nevertheless', 'consequently', 'therefore',
            'additionally', 'alternatively', 'subsequently',
            'respectively', 'approximately', 'significantly',
            'essentially', 'fundamentally', 'primarily',
            'in particular', 'for instance', 'for example',
            'in contrast', 'on the other hand', 'in addition',
            'as a result', 'due to', 'in order to',
        ]
        
        vocab_count = 0
        for word in precise_vocab:
            vocab_count += response_lower.count(word)
        score += min(vocab_count * 0.6, 4.0)
        
        # ============================================================
        # 9. QUERY RELEVANCE
        # ============================================================
        
        # Check if response addresses key terms from query
        query_words = set(query_lower.split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'no', 'if', 'so', 'as', 'it',
                      'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they',
                      'me', 'my', 'your', 'what', 'how', 'why', 'when', 'where',
                      'which', 'who', 'whom', 'am', 'im', "i'm", 'about', 'up',
                      'there', 'here', 'some', 'any', 'all', 'each', 'every'}
        
        query_content_words = query_words - stop_words
        if query_content_words:
            response_word_set = set(words)
            overlap = len(query_content_words & response_word_set)
            relevance_ratio = overlap / len(query_content_words)
            score += relevance_ratio * 5.0
        
        # ============================================================
        # 10. COHERENCE INDICATORS
        # ============================================================
        
        # Transition words and logical connectors
        transitions = [
            'first', 'second', 'third', 'finally', 'next', 'then',
            'also', 'in summary', 'to summarize', 'in conclusion',
            'overall', 'step', 'importantly', 'note that',
            'keep in mind', 'remember that', 'consider',
        ]
        
        transition_count = 0
        for t in transitions:
            transition_count += response_lower.count(t)
        score += min(transition_count * 0.5, 3.0)
        
        # ============================================================
        # 11. PROPER NOUN USAGE (specificity indicator)
        # ============================================================
        
        # Words starting with capital letter (not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response)
        score += min(len(set(proper_nouns)) * 0.5, 3.0)
        
        # ============================================================
        # 12. COMPLETENESS INDICATOR
        # ============================================================
        
        # Check if response appears truncated (ends mid-sentence)
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?:"\')]}':
            score -= 3.0  # Likely truncated
        
        # Check if it ends with a complete thought
        if stripped and stripped[-1] in '.!?':
            score += 1.0
        
        # ============================================================
        # 13. BALANCED PERSPECTIVE INDICATORS
        # ============================================================
        
        balance_phrases = [
            'on the other hand', 'however', 'although', 'while',
            'conversely', 'in contrast', 'alternatively',
            'pros and cons', 'advantages and disadvantages',
            'both sides', 'different perspectives',
        ]
        
        balance_count = 0
        for phrase in balance_phrases:
            balance_count += response_lower.count(phrase)
        score += min(balance_count * 1.0, 3.0)
        
        # ============================================================
        # 14. OPENING QUALITY
        # ============================================================
        
        # Penalize overly casual/filler openings
        filler_openers = [
            'great question', 'that\'s a great question',
            'oh wow', 'well well well', 'haha',
        ]
        
        first_50_chars = response_lower[:50]
        for filler in filler_openers:
            if first_50_chars.startswith(filler):
                score -= 1.0
                break
        
        # Reward direct, informative openings
        informative_openers = [
            'the ', 'in ', 'according to', 'based on',
            'there are', 'this is', 'when ',
        ]
        for opener in informative_openers:
            if first_50_chars.startswith(opener):
                score += 1.0
                break
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        
        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0  # Safe fallback mid-range score