def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Focuses on:
    - Presence of verifiable facts (specific names, dates, numbers, citations)
    - Appropriate hedging for uncertain claims
    - Absence of hallucination red-flags (overly precise unsourced stats, absolute claims)
    - Absence of sensationalism and conspiracy-style language
    - Structured, well-organized presentation (indicates reliability)
    - Appropriate specificity without fabrication signals
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 1.0
        
        if not isinstance(query, str) or not isinstance(response, str):
            return 1.0
        
        query = query.strip()
        response = response.strip()
        
        if len(response) < 10:
            return 1.0
        
        score = 50.0  # Start at midpoint of 0-100
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        # ============================================================
        # 1. HEDGING & EPISTEMIC HUMILITY (positive indicator)
        # ============================================================
        hedging_phrases = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends? to\b', r'\bit\'?s possible\b', r'\blikely\b',
            r'\bin many cases\b', r'\bin some cases\b', r'\bsome\b',
            r'\bapproximately\b', r'\babout\b', r'\baround\b',
            r'\bit depends\b', r'\bdepending on\b', r'\bnot always\b',
            r'\bcan vary\b', r'\bvaries\b', r'\bsometimes\b',
            r'\bit\'?s important to note\b', r'\bkeep in mind\b',
            r'\bworth noting\b', r'\bplease note\b',
            r'\bconsider\b', r'\byou may want\b', r'\byou might\b',
            r'\bcan be\b', r'\bcould be\b',
        ]
        hedge_count = 0
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_ratio = hedge_count / max(word_count, 1)
        # Reward moderate hedging, penalize none or excessive
        if hedge_ratio > 0.001 and hedge_ratio < 0.08:
            score += min(hedge_count * 1.5, 8)
        elif hedge_ratio >= 0.08:
            score += 3  # Excessive hedging is less ideal but not terrible
        
        # ============================================================
        # 2. ABSOLUTE / OVERCONFIDENT CLAIMS (negative indicator)
        # ============================================================
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\babsolutely\b', r'\bwithout a doubt\b', r'\bundeniably\b',
            r'\bguaranteed\b', r'\bwithout question\b', r'\bno question\b',
            r'\beveryone knows\b', r'\bobviously\b', r'\bclearly\b',
            r'\bthe fact is\b', r'\bthe truth is\b', r'\bno one can deny\b',
            r'\bimpossible\b', r'\bperfect\b', r'\bflawless\b',
            r'\b100%\b', r'\b100 percent\b',
        ]
        absolute_count = 0
        for pattern in absolute_patterns:
            absolute_count += len(re.findall(pattern, response_lower))
        
        # Penalize absolute claims
        score -= min(absolute_count * 2.0, 10)
        
        # ============================================================
        # 3. SENSATIONALISM & CONSPIRACY LANGUAGE (negative indicator)
        # ============================================================
        sensational_words = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind-?blowing\b',
            r'\binsane\b', r'\bcrazy\b', r'\bwild\b', r'\bincredible\b',
            r'\bamazing\b', r'\bterrible\b', r'\bhorrible\b', r'\bawful\b',
            r'\bdevastating\b', r'\bcatastrophic\b', r'\bepidemic\b',
            r'\bconspiracy\b', r'\bcover-?up\b', r'\bthey don\'t want you to know\b',
            r'\bwake up\b', r'\bsheeple\b', r'\bmainstream media\b',
            r'\bhidden truth\b', r'\bsecret\w*\b', r'\bexposed\b',
            r'\bbombshell\b', r'\bbreaking\b', r'\bexclusive\b',
            r'\byou won\'t believe\b', r'\binsider\b',
        ]
        sensational_count = 0
        for pattern in sensational_words:
            sensational_count += len(re.findall(pattern, response_lower))
        
        score -= min(sensational_count * 3.0, 15)
        
        # ============================================================
        # 4. SPECIFICITY & VERIFIABLE FACTS (positive indicator)
        # ============================================================
        # Numbers and dates
        number_matches = re.findall(r'\b\d+[\.,]?\d*\b', response)
        number_count = len(number_matches)
        
        # Specific dates
        date_patterns = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', response, re.IGNORECASE)
        date_count = len(date_patterns)
        
        # Moderate specificity is good (shows verifiable content)
        specificity_score = min(number_count * 0.3 + date_count * 1.0, 6)
        score += specificity_score
        
        # ============================================================
        # 5. OVERLY PRECISE UNSOURCED STATISTICS (negative indicator)
        # ============================================================
        # Very specific percentages or statistics without attribution
        precise_stats = re.findall(r'\b\d{2,}\.\d+\s*%', response)
        # Phrases like "studies show" without citation
        vague_citations = re.findall(r'\bstudies show\b|\bresearch shows\b|\bscientists say\b|\bexperts say\b|\baccording to research\b', response_lower)
        
        # Penalize precise stats without proper attribution
        unsourced_stat_penalty = len(precise_stats) * 1.5 + len(vague_citations) * 0.5
        score -= min(unsourced_stat_penalty, 6)
        
        # ============================================================
        # 6. PROPER CITATIONS & ATTRIBUTION (positive indicator)
        # ============================================================
        citation_patterns = [
            r'according to [A-Z]',  # "According to [Name/Org]"
            r'\bcited\b', r'\breference\b', r'\bsource\b',
            r'\b(?:University|Institute|Organization|Association)\b',
            r'(?:https?://|www\.)',  # URLs
            r'\(\d{4}\)',  # Year citations like (2023)
            r'\bet al\.',  # Academic citation
        ]
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, response))
        
        score += min(citation_count * 1.5, 5)
        
        # ============================================================
        # 7. STRUCTURAL QUALITY (positive indicator for reliability)
        # ============================================================
        # Numbered lists or bullet points
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^\s*[-•*]\s', response, re.MULTILINE))
        has_headers = bool(re.search(r'^\s*#{1,3}\s|^\s*[A-Z][^.!?]*:\s*$', response, re.MULTILINE))
        
        structure_score = 0
        if has_numbered_list:
            structure_score += 3
        if has_bullet_list:
            structure_score += 2
        if has_headers:
            structure_score += 2
        score += min(structure_score, 5)
        
        # ============================================================
        # 8. RESPONSE LENGTH & COMPLETENESS
        # ============================================================
        # Very short responses lack substance
        if word_count < 20:
            score -= 10
        elif word_count < 50:
            score -= 5
        elif word_count >= 80:
            score += 3
        
        if word_count >= 120:
            score += 2
        
        # ============================================================
        # 9. VOCABULARY SOPHISTICATION (indicator of quality)
        # ============================================================
        sophisticated_terms = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bnevertheless\b',
            r'\bconsequently\b', r'\btherefore\b', r'\bhowever\b',
            r'\bin addition\b', r'\bspecifically\b', r'\bfor instance\b',
            r'\bfor example\b', r'\bin particular\b', r'\bnotably\b',
            r'\bsignificantly\b', r'\bfundamentally\b', r'\bessentially\b',
            r'\bimportantly\b', r'\bcrucially\b',
        ]
        sophistication_count = 0
        for pattern in sophisticated_terms:
            sophistication_count += len(re.findall(pattern, response_lower))
        
        score += min(sophistication_count * 1.0, 5)
        
        # ============================================================
        # 10. DISMISSIVE / LOW-EFFORT LANGUAGE (negative indicator)
        # ============================================================
        dismissive_patterns = [
            r'\bjust\s+(?:do|get|try|go|buy|make)\b',
            r'\bit\'?s (?:just|only|simply)\b',
            r'\bwhatever\b', r'\bno big deal\b',
            r'\bget over it\b', r'\bmove on\b',
            r'\bstop\s+(?:worrying|complaining|overthinking)\b',
            r'\byou should be\b', r'\byou need to\b',
        ]
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        score -= min(dismissive_count * 2.5, 10)
        
        # ============================================================
        # 11. EMPATHY & ACKNOWLEDGMENT (positive for support queries)
        # ============================================================
        empathy_indicators = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'?s understandable\b',
            r'\bit\'?s (?:completely |perfectly |totally )?(?:understandable|natural|normal|okay|ok|fine|valid)\b',
            r'\bi\'?m sorry\b', r'\bsorry to hear\b',
            r'\bi hear you\b', r'\bi can imagine\b',
            r'\byour feelings\b', r'\byour concerns?\b',
            r'\bcompletely understandable\b', r'\bperfectly (?:fine|normal|okay|natural)\b',
            r'\bwe (?:value|appreciate|understand)\b',
        ]
        
        # Detect if query is emotional/support-seeking
        emotional_query_words = ['feeling', 'frustrated', 'stress', 'sad', 'lonely', 
                                  'heartbroken', 'devastated', 'struggling', 'difficult',
                                  'upset', 'angry', 'anxious', 'worried', 'fear',
                                  'breakup', 'passed away', 'died', 'lost', 'regret']
        is_emotional_query = any(w in query_lower for w in emotional_query_words)
        
        empathy_count = 0
        for pattern in empathy_indicators:
            empathy_count += len(re.findall(pattern, response_lower))
        
        if is_emotional_query:
            score += min(empathy_count * 3.0, 10)
            if empathy_count == 0:
                score -= 8
        else:
            score += min(empathy_count * 1.0, 4)
        
        # ============================================================
        # 12. QUERY RELEVANCE (basic keyword overlap)
        # ============================================================
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
        
        # Remove very common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
                     'their', 'them', 'will', 'would', 'could', 'should', 'about',
                     'which', 'when', 'what', 'where', 'there', 'here', 'more',
                     'some', 'than', 'other', 'into', 'also', 'just', 'very',
                     'your', 'does', 'make', 'like', 'each', 'only', 'most'}
        query_content_words = query_words - stopwords
        
        if query_content_words:
            overlap = len(query_content_words & response_words) / len(query_content_words)
            score += overlap * 8
        
        # ============================================================
        # 13. FABRICATION RED FLAGS
        # ============================================================
        # Making up directions without context
        direction_words = re.findall(r'\bturn (?:left|right)\b|\btake the (?:first|second|third)\b|\bcontinue straight\b', response_lower)
        if len(direction_words) > 2 and 'direction' not in query_lower and 'navigate' not in query_lower:
            score -= 10  # Likely fabricating directions
        
        # Inventing specific but unverifiable details
        # (very specific numbers in casual advice context)
        
        # ============================================================
        # 14. CONDITIONAL / NUANCED REASONING (positive)
        # ============================================================
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bon the other hand\b',
            r'\bwhile\b.*\b(?:also|however)\b', r'\balthough\b',
            r'\bdepends on\b', r'\bin contrast\b', r'\bbalance\b',
        ]
        nuance_count = 0
        for pattern in conditional_patterns:
            nuance_count += len(re.findall(pattern, response_lower))
        
        score += min(nuance_count * 1.5, 5)
        
        # ============================================================
        # 15. SENTENCE VARIETY & AVERAGE LENGTH
        # ============================================================
        if sentence_count > 1:
            avg_sentence_length = word_count / sentence_count
            if 10 <= avg_sentence_length <= 25:
                score += 3  # Good sentence length
            elif avg_sentence_length > 35:
                score -= 2  # Run-on sentences
            elif avg_sentence_length < 6:
                score -= 2  # Too choppy
        
        # ============================================================
        # 16. FIRST-PERSON ENGAGEMENT (appropriate for support)
        # ============================================================
        first_person = len(re.findall(r'\b(?:I\'m|I am|I can|I understand|we|our|let\'s|let us)\b', response, re.IGNORECASE))
        second_person = len(re.findall(r'\byou\b|\byour\b|\byou\'re\b', response_lower))
        
        # Good balance of engagement
        if first_person > 0 and second_person > 0:
            score += 3
        
        # ============================================================
        # 17. ACTIONABLE ADVICE (positive)
        # ============================================================
        action_patterns = [
            r'\btry\b', r'\bconsider\b', r'\bstart by\b', r'\bbegin with\b',
            r'\bstep \d\b', r'\bfirst\b.*\bthen\b', r'\bhere are\b',
            r'\bone way\b', r'\banother (?:way|option|approach)\b',
            r'\byou can\b', r'\byou could\b', r'\bfor example\b',
        ]
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, response_lower))
        
        score += min(action_count * 1.0, 5)
        
        # ============================================================
        # 18. TONE APPROPRIATENESS
        # ============================================================
        # Casual language markers
        casual_markers = re.findall(r'\bhey\b|\bkinda\b|\bsorta\b|\bwanna\b|\bgonna\b|\byeah\b|\bnah\b|\bstuff\b|\bcool\b|\bawesome\b|\bkiller\b|\bdude\b|\bbuddy\b|\bpal\b', response_lower)
        
        # Check if query is casual
        query_casual = any(w in query_lower for w in ['casual', 'informal', 'slang', 'laid-back', 'chill'])
        
        if query_casual:
            # Casual tone is appropriate
            if len(casual_markers) > 0:
                score += 3
        else:
            # Penalize overly casual in formal context
            if len(casual_markers) > 3:
                score -= 2
        
        # ============================================================
        # FINAL NORMALIZATION: Map to 1-5 scale
        # ============================================================
        # Clamp score to reasonable range
        score = max(10, min(90, score))
        
        # Map 10-90 to 1-5
        final_score = 1 + (score - 10) * 4.0 / 80.0
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        return 3.0  # Default middle score on error