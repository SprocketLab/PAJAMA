def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level structure analysis (well-formed sentences as reliability indicator)
    2. Named entity density (proper nouns, dates, numbers as factual grounding)
    3. Epistemic calibration (appropriate certainty/uncertainty markers)
    4. Anti-hallucination signals (detecting overly precise unsourced claims, sensationalism)
    5. Information density ratio (content words vs filler)
    6. Response coherence with query
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses are almost always low quality
        if len(response_stripped) < 5:
            return 0.5
        
        score = 5.0  # Start at midpoint
        
        # === 1. Sentence-level structure analysis ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length (well-formed responses have moderate sentence lengths)
        words = response_stripped.split()
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        avg_words_per_sentence = num_words / num_sentences
        
        # Reward moderate sentence length (10-25 words), penalize extremes
        if 10 <= avg_words_per_sentence <= 25:
            score += 0.5
        elif avg_words_per_sentence < 5:
            score -= 0.8
        elif avg_words_per_sentence > 50:
            score -= 0.5
        
        # === 2. Named entity / factual grounding density ===
        # Detect capitalized words that aren't sentence starters (potential proper nouns)
        proper_noun_pattern = re.compile(r'(?<!\. )(?<!\.\s)(?<!^)\b[A-Z][a-z]{2,}\b')
        proper_nouns = proper_noun_pattern.findall(response_stripped)
        
        # Detect dates
        date_patterns = [
            r'\b\d{4}\b',  # years
            r'\b\d{1,2}(?:st|nd|rd|th)\b',  # ordinal dates
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
        ]
        date_count = 0
        for pat in date_patterns:
            date_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        # Detect numbers (as factual indicators)
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|,\d{3})*\b', response_stripped)
        number_count = len(numbers)
        
        # Factual grounding score
        entity_density = (len(proper_nouns) + date_count + number_count * 0.5) / num_words
        # Moderate density is good, excessive might indicate hallucination
        if 0.02 <= entity_density <= 0.15:
            score += 1.0
        elif 0.005 <= entity_density < 0.02:
            score += 0.3
        elif entity_density > 0.25:
            score -= 0.3  # Suspiciously dense
        
        # === 3. Epistemic calibration ===
        # Appropriate hedging phrases (shows intellectual honesty)
        hedging_phrases = [
            r'\bit is (?:difficult|hard|challenging) to\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bapproximately\b', r'\babout\b', r'\baround\b',
            r'\boften\b', r'\btends? to\b',
            r'\bin many cases\b', r'\bin some cases\b',
            r'\bdepending on\b', r'\bvaries?\b',
            r'\blikely\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bcan be\b', r'\bsome\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bnot without\b', r'\bsubjective\b',
        ]
        hedge_count = 0
        response_lower = response_stripped.lower()
        for pat in hedging_phrases:
            hedge_count += len(re.findall(pat, response_lower))
        
        hedge_ratio = hedge_count / num_words
        if 0.01 <= hedge_ratio <= 0.08:
            score += 0.8  # Appropriate hedging
        elif hedge_ratio > 0.12:
            score -= 0.3  # Over-hedging
        
        # Overconfidence markers (red flags)
        overconfidence = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b',
            r'\babsolutely\b', r'\bwithout a doubt\b',
            r'\beveryone knows\b', r'\bobviously\b',
            r'\bundeniably\b', r'\bguaranteed\b',
            r'\b100%\b', r'\bno question\b',
        ]
        overconf_count = 0
        for pat in overconfidence:
            overconf_count += len(re.findall(pat, response_lower))
        
        if overconf_count > 2:
            score -= 0.8
        elif overconf_count > 0:
            score -= 0.3
        
        # === 4. Anti-hallucination and sensationalism detection ===
        sensational_words = [
            r'\bshocking\b', r'\bunbelievable\b', r'\binsane\b',
            r'\bmind-?blowing\b', r'\bconspiracy\b', r'\bcover-?up\b',
            r'\bthey don\'t want you to know\b', r'\bsecret(?:ly)?\b',
            r'\bexposed\b', r'\brevealed\b', r'\bhidden truth\b',
            r'\bwake up\b', r'\bsheeple\b', r'\bmanipulat\b',
            r'\bbreaking\b', r'\bexclusive\b',
            r'\bdestroy(?:ed|ing)?\b', r'\bcatastroph\b',
        ]
        sensational_count = 0
        for pat in sensational_words:
            sensational_count += len(re.findall(pat, response_lower))
        
        if sensational_count > 2:
            score -= 1.5
        elif sensational_count > 0:
            score -= 0.5
        
        # === 5. Information density and content quality ===
        # Ratio of unique words to total words (lexical diversity)
        word_list_lower = [w.lower() for w in words]
        unique_words = set(word_list_lower)
        lexical_diversity = len(unique_words) / num_words if num_words > 0 else 0
        
        if lexical_diversity > 0.65:
            score += 0.5
        elif lexical_diversity < 0.3:
            score -= 1.0  # Very repetitive
        elif lexical_diversity < 0.45:
            score -= 0.4
        
        # Detect excessive repetition (hallucination red flag)
        # Check for repeated phrases (3-grams)
        if num_words >= 6:
            trigrams = [' '.join(word_list_lower[i:i+3]) for i in range(len(word_list_lower)-2)]
            trigram_counts = Counter(trigrams)
            max_trigram_repeat = max(trigram_counts.values()) if trigram_counts else 1
            if max_trigram_repeat > 3:
                score -= 1.5
            elif max_trigram_repeat > 2:
                score -= 0.5
        
        # === 6. Structural quality indicators ===
        # Check for explanation/elaboration connectors
        explanation_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\bspecifically\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\baccording to\b', r'\bbased on\b',
            r'\bthis (?:means|implies|suggests|indicates)\b',
            r'\bin other words\b', r'\bthat is\b',
        ]
        explanation_count = 0
        for pat in explanation_markers:
            explanation_count += len(re.findall(pat, response_lower))
        
        if explanation_count >= 2:
            score += 0.7
        elif explanation_count >= 1:
            score += 0.3
        
        # Citation-like patterns
        citation_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bsource\b', r'\breport(?:ed|s)?\b',
            r'\bknown as\b', r'\bcalled\b', r'\breferred to as\b',
            r'\b(?:is|was|are|were) (?:a|an|the)\b',  # definitional structures
        ]
        citation_count = 0
        for pat in citation_patterns:
            citation_count += len(re.findall(pat, response_lower))
        
        if citation_count >= 2:
            score += 0.5
        
        # === 7. Response relevance to query ===
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        response_words_set = set(re.findall(r'\b[a-z]{3,}\b', response_lower))
        
        # Remove very common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'been', 'some', 'them', 'than', 'its', 'over',
                      'also', 'that', 'this', 'with', 'from', 'they', 'will',
                      'what', 'when', 'make', 'like', 'how', 'each', 'which',
                      'their', 'said', 'many', 'more', 'other', 'about', 'into',
                      'could', 'would', 'there', 'these', 'where', 'does'}
        
        query_content = query_words - stop_words
        response_content = response_words_set - stop_words
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
            if overlap >= 0.3:
                score += 0.6
            elif overlap < 0.1:
                score -= 0.5
        
        # === 8. Garbage / formatting detection ===
        # HTML tags in non-HTML queries
        html_tags = re.findall(r'<[^>]+>', response_stripped)
        if 'html' not in query.lower() and 'tag' not in query.lower():
            if len(html_tags) > 3:
                score -= 1.5
        
        # Code in non-code queries
        code_indicators = ['import ', 'def ', 'class ', 'return ', 'print(', '#!/']
        if 'code' not in query.lower() and 'program' not in query.lower() and 'function' not in query.lower():
            code_count = sum(1 for ci in code_indicators if ci in response_stripped)
            if code_count >= 2:
                score -= 1.5
        
        # Excessive special characters
        special_chars = sum(1 for c in response_stripped if c in '#*=|{}[]<>')
        special_ratio = special_chars / len(response_stripped)
        if special_ratio > 0.1:
            score -= 1.0
        
        # === 9. Length appropriateness ===
        # Moderate length is generally better
        if 20 <= num_words <= 200:
            score += 0.5
        elif num_words < 10:
            score -= 1.0
        elif num_words > 500:
            score -= 0.3
        
        # Penalize responses that are just the query repeated
        if response_stripped.strip().lower() == query.strip().lower():
            score -= 3.0
        
        # === 10. Completeness indicator ===
        # Check if response ends mid-sentence (truncation)
        last_char = response_stripped[-1] if response_stripped else ''
        if last_char in '.!?)':
            score += 0.3  # Properly terminated
        elif last_char in ',;:':
            score -= 0.3  # Truncated
        
        # Multiple complete sentences is a good sign
        if num_sentences >= 2:
            score += 0.4
        if num_sentences >= 4:
            score += 0.3
        
        # === 11. Input/Output pattern detection (bad sign for factual responses) ===
        io_patterns = re.findall(r'\b(?:Input|Output|Question|Answer)\s*:', response_stripped)
        if len(io_patterns) > 3:
            score -= 1.0
        
        # Clamp score to [0, 10]
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 3.0