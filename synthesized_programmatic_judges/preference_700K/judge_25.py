def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Focuses on:
    - Presence of specific factual markers (names, dates, numbers, citations)
    - Appropriate hedging language for uncertain claims
    - Absence of hallucination red-flags (unsourced absolute claims, sensationalism)
    - Structural indicators of well-researched responses
    - Depth and substantiveness of the response
    
    Returns a score where HIGHER = BETTER quality.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        score = 50.0  # Start at midpoint of 0-100 scale
        
        # ========== 1. RESPONSE LENGTH & SUBSTANCE ==========
        resp_len = len(response.strip())
        word_count = len(response.split())
        
        # Very short responses are usually low quality
        if word_count < 10:
            score -= 15
        elif word_count < 25:
            score -= 8
        elif word_count < 50:
            score -= 2
        elif word_count > 80:
            score += 5
        elif word_count > 150:
            score += 8
        elif word_count > 300:
            score += 10
        
        # ========== 2. SPECIFIC FACTUAL MARKERS ==========
        
        # Dates (years, specific dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', response)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', response)
        specific_dates = len(year_pattern) + len(date_pattern)
        score += min(specific_dates * 2.5, 12)
        
        # Numbers and statistics (specific quantities suggest factual grounding)
        numbers = re.findall(r'\b\d+\.?\d*\s*(%|percent|million|billion|thousand|kg|lbs?|miles?|km|hours?|minutes?|degrees?|feet|meters?)\b', response_lower)
        score += min(len(numbers) * 2.0, 8)
        
        # Proper nouns (capitalized words not at sentence start) - names, places, etc.
        sentences = re.split(r'[.!?]\s+', response)
        proper_noun_count = 0
        for sent in sentences:
            words = sent.split()
            for i, w in enumerate(words):
                if i > 0 and len(w) > 1 and w[0].isupper() and not w.isupper():
                    proper_noun_count += 1
        score += min(proper_noun_count * 1.0, 10)
        
        # ========== 3. CITATION & REFERENCE INDICATORS ==========
        
        # Book/work references (italics in markdown, quotes around titles)
        italic_refs = re.findall(r'\*[A-Z][^*]+\*', response)
        quoted_refs = re.findall(r'"[A-Z][^"]{3,}"', response)
        citation_markers = re.findall(r'\([^)]*\d{4}[^)]*\)', response)
        url_refs = re.findall(r'https?://\S+', response)
        
        ref_count = len(italic_refs) + len(quoted_refs) + len(citation_markers) + len(url_refs)
        score += min(ref_count * 3.0, 12)
        
        # Mentions of specific people/authors by name pattern (e.g., "St. Peter", "u/username")
        name_patterns = re.findall(r'\b(?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|St\.|Sir|Lord)\s+[A-Z]\w+', response)
        username_patterns = re.findall(r'u/\w+', response)
        score += min((len(name_patterns) + len(username_patterns)) * 2.0, 8)
        
        # ========== 4. HEDGING & EPISTEMIC RESPONSIBILITY ==========
        
        # Appropriate hedging phrases (shows intellectual honesty)
        hedging_phrases = [
            'might be', 'could be', 'it seems', 'it appears', 'possibly',
            'perhaps', 'likely', 'unlikely', 'it\'s possible', 'may have',
            'tends to', 'generally', 'typically', 'often', 'usually',
            'in some cases', 'it depends', 'not always', 'arguably',
            'as far as i know', 'to my knowledge', 'i believe',
            'there\'s a chance', 'one possibility', 'some argue',
            'it\'s worth noting', 'keep in mind', 'however',
            'on the other hand', 'that said', 'though', 'although',
            'while', 'but', 'essentially', 'roughly', 'approximately',
            'traditionally', 'historically'
        ]
        
        hedge_count = 0
        for phrase in hedging_phrases:
            hedge_count += response_lower.count(phrase)
        
        # Moderate hedging is good; too much might indicate vagueness
        if hedge_count >= 1:
            score += min(hedge_count * 1.5, 8)
        
        # ========== 5. HALLUCINATION RED FLAGS ==========
        
        # Overly precise unsourced statistics
        very_precise_stats = re.findall(r'\b\d{2,}\.\d{2,}\s*%', response)
        score -= len(very_precise_stats) * 3
        
        # Absolute/sensational language (red flags)
        absolute_phrases = [
            'absolutely', 'definitely', 'without a doubt', 'undeniably',
            'there is no question', 'it is certain', 'everyone knows',
            'always', 'never', 'completely', 'totally', 'entirely',
            'no one', 'everyone', 'all experts agree'
        ]
        
        absolute_count = 0
        for phrase in absolute_phrases:
            # Count but be careful - some uses are fine in context
            occurrences = response_lower.count(phrase)
            absolute_count += occurrences
        
        # Mild penalty for absolutes
        score -= min(absolute_count * 1.5, 8)
        
        # Sensationalist/conspiracy language
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'they don\'t want you to know',
            'secret', 'conspiracy', 'cover-up', 'mainstream media', 'wake up',
            'sheeple', 'big pharma', 'elites', 'suppressed', 'censored',
            'bombshell', 'explosive', 'jaw-dropping'
        ]
        
        sensational_count = 0
        for word in sensational_words:
            sensational_count += response_lower.count(word)
        score -= sensational_count * 5
        
        # ========== 6. STRUCTURAL QUALITY INDICATORS ==========
        
        # Multiple sentences suggest more thorough response
        sentence_count = len(re.split(r'[.!?]+', response.strip()))
        if sentence_count >= 3:
            score += 3
        if sentence_count >= 5:
            score += 3
        
        # Paragraph structure
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 3
        
        # Lists/bullet points suggest organized thinking
        list_items = re.findall(r'(?:^|\n)\s*[-*•]\s+\S', response)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[.)]\s+\S', response)
        if len(list_items) + len(numbered_items) >= 2:
            score += 4
        
        # Code blocks (relevant for technical queries)
        code_blocks = re.findall(r'```', response)
        if len(code_blocks) >= 2:
            score += 3
        
        # ========== 7. EXPLANATORY DEPTH ==========
        
        # Causal/explanatory connectors suggest reasoning
        explanatory_words = [
            'because', 'therefore', 'thus', 'hence', 'since',
            'as a result', 'consequently', 'this means', 'in other words',
            'for example', 'for instance', 'such as', 'specifically',
            'in particular', 'namely', 'e.g.', 'i.e.',
            'the reason', 'this is due to', 'which means'
        ]
        
        explain_count = 0
        for phrase in explanatory_words:
            explain_count += response_lower.count(phrase)
        score += min(explain_count * 2.0, 10)
        
        # ========== 8. QUERY RELEVANCE ==========
        
        # Extract meaningful words from query (not stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'and', 'but', 'or', 'not', 'no', 'nor',
                     'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
                     'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
                     'than', 'too', 'very', 'just', 'about', 'up', 'out', 'if',
                     'then', 'that', 'this', 'what', 'which', 'who', 'whom',
                     'how', 'when', 'where', 'why', 'i', 'me', 'my', 'we', 'our',
                     'you', 'your', 'he', 'she', 'it', 'they', 'them', 'their',
                     'am', 'im', 'ive', 'its'}
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower)) - stopwords
        response_words = set(re.findall(r'\b[a-z]{3,}\b', response_lower)) - stopwords
        
        if query_words:
            overlap = len(query_words & response_words) / len(query_words)
            score += overlap * 8
        
        # ========== 9. VOCABULARY RICHNESS ==========
        
        # Unique word ratio (type-token ratio) — higher suggests more informative
        all_words = re.findall(r'\b[a-z]+\b', response_lower)
        if len(all_words) > 10:
            ttr = len(set(all_words)) / len(all_words)
            # TTR typically between 0.3 and 0.9
            if ttr > 0.5:
                score += (ttr - 0.5) * 10
        
        # ========== 10. CONVERSATIONAL vs AUTHORITATIVE TONE ==========
        
        # First person singular overuse can indicate anecdotal rather than factual
        first_person = len(re.findall(r'\bi\b', response_lower))
        first_person_ratio = first_person / max(word_count, 1)
        
        # Some first person is fine (especially for experience-based queries)
        # but excessive use without other factual markers is penalized
        if first_person_ratio > 0.08 and ref_count == 0 and specific_dates == 0:
            score -= 3
        
        # ========== 11. FILLER / LOW-CONTENT PHRASES ==========
        
        filler_phrases = [
            'i hope this helps', 'hope that helps', 'let me know if',
            'feel free to', 'happy to help', 'glad to help',
            'great question', 'good question', 'interesting question',
            'sure, i can help', 'of course', 'no problem'
        ]
        
        filler_count = 0
        for phrase in filler_phrases:
            filler_count += response_lower.count(phrase)
        score -= filler_count * 2
        
        # ========== 12. BOT/TEMPLATE DETECTION ==========
        
        # Automated/template responses tend to be lower quality
        bot_indicators = [
            'welcome to', 'please read our rules', 'your comment',
            'will be removed', 'this is an automated', 'i am a bot',
            'moderator', 'flair'
        ]
        
        bot_count = 0
        for phrase in bot_indicators:
            bot_count += response_lower.count(phrase)
        score -= bot_count * 8
        
        # ========== 13. DOMAIN-SPECIFIC TERMINOLOGY ==========
        
        # Longer, more technical words suggest domain expertise
        long_words = [w for w in all_words if len(w) >= 8]
        long_word_ratio = len(long_words) / max(len(all_words), 1)
        score += min(long_word_ratio * 20, 6)
        
        # ========== 14. COMPARATIVE/NUANCED THINKING ==========
        
        nuance_markers = [
            'on the other hand', 'alternatively', 'in contrast',
            'whereas', 'compared to', 'distinction between',
            'trade-off', 'tradeoff', 'nuance', 'complexity',
            'it depends on', 'context', 'perspective'
        ]
        
        nuance_count = 0
        for phrase in nuance_markers:
            nuance_count += response_lower.count(phrase)
        score += min(nuance_count * 3.0, 9)
        
        # ========== FINAL NORMALIZATION ==========
        
        # Clamp to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        # Never crash — return neutral score
        return 50.0