def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant focuses on:
    - Specificity of claims (names, dates, numbers, units)
    - Citation-like patterns and source references
    - Appropriate hedging vs. overconfident absolute claims
    - Hallucination red flags (overly precise unsourced stats, sensationalism)
    - Structured reasoning and logical connectives
    - Ratio-based analysis of factual vs. filler content
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        resp = response.strip()
        query_clean = query.strip()
        
        if len(resp) < 10:
            return 0.5
        
        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. SPECIFIC FACTUAL MARKERS ===
        # Dates (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', resp)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', resp)
        # Specific numbers with units
        number_unit_pattern = re.findall(
            r'\b\d+[\.,]?\d*\s*(?:kg|lb|lbs|m|km|miles|feet|ft|cm|mm|inches|'
            r'degrees|°|mph|km/h|m/s|Hz|kHz|MHz|GHz|watts|W|volts|V|amps|A|'
            r'liters|gallons|oz|mg|grams|g|tons|percent|%|minutes|hours|seconds|'
            r'days|weeks|months|years|dollars|\$|euros|€|pounds|£)\b',
            resp, re.IGNORECASE
        )
        # Proper nouns (capitalized words not at sentence start) - approximate
        proper_noun_pattern = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', resp)
        
        specificity_score = 0.0
        specificity_score += min(len(year_pattern) * 1.5, 6.0)
        specificity_score += min(len(date_pattern) * 2.0, 4.0)
        specificity_score += min(len(number_unit_pattern) * 1.0, 8.0)
        specificity_score += min(len(proper_noun_pattern) * 0.3, 4.0)
        
        score += specificity_score
        
        # === 2. CITATION AND SOURCE INDICATORS ===
        citation_phrases = [
            r'according to', r'research\s+(?:shows|suggests|indicates|has shown)',
            r'studies?\s+(?:show|suggest|indicate|found|have shown)',
            r'data\s+(?:shows|suggests|indicates)',
            r'evidence\s+(?:shows|suggests|indicates)',
            r'reported\s+(?:by|that|in)', r'published\s+(?:in|by)',
            r'sourced?\s+from', r'based\s+on',
            r'as\s+noted\s+(?:by|in)', r'as\s+described\s+(?:by|in)',
            r'documented\s+(?:by|in)', r'established\s+(?:by|in|that)',
        ]
        citation_count = 0
        resp_lower = resp.lower()
        for pat in citation_phrases:
            citation_count += len(re.findall(pat, resp_lower))
        
        score += min(citation_count * 2.0, 8.0)
        
        # === 3. APPROPRIATE HEDGING (positive signal) ===
        hedging_phrases = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\btends?\s+to\b',
            r'\bin\s+some\s+cases\b', r'\bin\s+many\s+cases\b',
            r'\bit\s+is\s+(?:likely|possible|probable)\b',
            r'\bapproximately\b', r'\babout\b', r'\baround\b',
            r'\broughly\b', r'\bestimated\b',
            r'\bit\s+depends\b', r'\bdepending\s+on\b',
            r'\bnot\s+necessarily\b', r'\bnot\s+always\b',
            r'\bcan\s+vary\b', r'\bvaries\b',
        ]
        hedge_count = 0
        for pat in hedging_phrases:
            hedge_count += len(re.findall(pat, resp_lower))
        
        # Hedging is good but too much hedging is wishy-washy
        hedge_ratio = hedge_count / sent_count
        if hedge_ratio <= 0.5:
            score += hedge_count * 1.5
        elif hedge_ratio <= 1.0:
            score += hedge_count * 0.8
        else:
            score += hedge_count * 0.3  # diminishing returns
        score = min(score, score)  # cap handled below
        
        # === 4. HALLUCINATION RED FLAGS (negative signals) ===
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d{2,}\s*%', resp)
        score -= len(precise_stats) * 3.0
        
        # Absolute/sensational language
        absolute_terms = [
            r'\balways\b', r'\bnever\b', r'\bcompletely\b',
            r'\btotally\b', r'\babsolutely\b', r'\bwithout\s+(?:a\s+)?doubt\b',
            r'\bundeniably\b', r'\bunquestionably\b', r'\bindisputably\b',
            r'\bguaranteed\b', r'\bproven\s+fact\b', r'\b100\s*%\b',
            r'\beveryone\s+knows\b', r'\bit\s+is\s+(?:a\s+)?fact\s+that\b',
            r'\bno\s+one\s+can\s+deny\b',
        ]
        absolute_count = 0
        for pat in absolute_terms:
            absolute_count += len(re.findall(pat, resp_lower))
        
        score -= absolute_count * 1.5
        
        # Sensationalism / conspiracy language
        sensational_terms = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind[\s-]?blowing\b',
            r'\bconspiracy\b', r'\bcover[\s-]?up\b', r'\bthey\s+don\'?t\s+want\s+you\s+to\s+know\b',
            r'\bsecret(?:ly)?\b', r'\bhidden\s+truth\b', r'\bwake\s+up\b',
            r'\bsheeple\b', r'\bmanipulat', r'\bbrainwash',
            r'\binsane(?:ly)?\b', r'\bcrazy\b', r'\bdestroy',
        ]
        sensational_count = 0
        for pat in sensational_terms:
            sensational_count += len(re.findall(pat, resp_lower))
        
        score -= sensational_count * 3.0
        
        # === 5. LOGICAL STRUCTURE AND REASONING CONNECTIVES ===
        reasoning_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas\s+a\s+result\b', r'\bdue\s+to\b',
            r'\bfor\s+this\s+reason\b', r'\bsince\b', r'\bgiven\s+that\b',
            r'\bthis\s+means\b', r'\bwhich\s+(?:means|implies|suggests|indicates)\b',
            r'\bin\s+other\s+words\b', r'\bspecifically\b',
            r'\bfor\s+(?:example|instance)\b', r'\bsuch\s+as\b',
            r'\bnamely\b', r'\bin\s+particular\b',
            r'\bhowever\b', r'\bon\s+the\s+other\s+hand\b',
            r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bwhile\b', r'\balthough\b', r'\bdespite\b',
            r'\bin\s+contrast\b', r'\bconversely\b',
        ]
        connective_count = 0
        for pat in reasoning_connectives:
            connective_count += len(re.findall(pat, resp_lower))
        
        connective_density = connective_count / sent_count
        score += min(connective_density * 4.0, 6.0)
        score += min(connective_count * 0.5, 5.0)
        
        # === 6. STRUCTURAL FORMATTING QUALITY ===
        # Numbered/lettered steps suggest organized factual presentation
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', resp)
        lettered_items = re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s', resp)
        bold_markers = re.findall(r'\*\*[^*]+\*\*', resp)
        markdown_headers = re.findall(r'(?:^|\n)\s*#{1,4}\s', resp)
        
        structure_score = 0.0
        structure_score += min(len(numbered_items) * 0.8, 4.0)
        structure_score += min(len(bold_markers) * 0.4, 3.0)
        structure_score += min(len(markdown_headers) * 1.0, 3.0)
        
        score += structure_score
        
        # === 7. CONTENT DENSITY AND INFORMATION RICHNESS ===
        # Unique word ratio (vocabulary richness)
        words_lower = [w.lower() for w in words]
        unique_ratio = len(set(words_lower)) / max(word_count, 1)
        
        # Penalize very repetitive text
        if unique_ratio < 0.3:
            score -= 5.0
        elif unique_ratio > 0.5:
            score += 3.0
        
        # Average word length (longer words often = more technical/specific)
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        if avg_word_len > 5.5:
            score += 2.0
        elif avg_word_len > 5.0:
            score += 1.0
        elif avg_word_len < 4.0:
            score -= 2.0
        
        # === 8. RESPONSE LENGTH APPROPRIATENESS ===
        # Moderate-length responses tend to be more informative
        # But we don't want to over-reward length
        length_score = 0.0
        if word_count >= 50:
            length_score += 2.0
        if word_count >= 100:
            length_score += 2.0
        if word_count >= 150:
            length_score += 1.0
        if word_count >= 200:
            length_score += 0.5
        # Very short responses are often incomplete
        if word_count < 30:
            length_score -= 3.0
        
        score += length_score
        
        # === 9. QUERY RELEVANCE (topic coverage) ===
        # Extract meaningful query words (exclude stopwords)
        stopwords = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
            'they', 'them', 'this', 'that', 'these', 'those', 'is', 'am', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'shall', 'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'not',
            'so', 'yet', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
            'about', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'if', 'what', 'which', 'who', 'whom', 'up', 'down',
            'need', 'want', 'get', 'got', 'im', "i'm", 'bit', 'any',
        }
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_clean.lower())) - stopwords
        resp_words_set = set(words_lower)
        
        if query_words:
            overlap = len(query_words & resp_words_set) / len(query_words)
            score += overlap * 5.0
        
        # === 10. OPENING QUALITY ===
        # Good responses often start with direct acknowledgment or clear framing
        first_100 = resp_lower[:min(200, len(resp_lower))]
        
        # Filler openers (mild penalty)
        filler_openers = [
            r'^(?:great|awesome|excellent|wonderful|fantastic)\s+(?:question|idea|choice)',
            r'^(?:that\'s|thats)\s+a\s+(?:great|good|excellent|wonderful)',
            r'^a\s+classic\s+(?:problem|question)',
        ]
        for pat in filler_openers:
            if re.search(pat, first_100):
                score -= 1.0
                break
        
        # Direct/confident openers (mild bonus)
        direct_openers = [
            r'^(?:there\s+are|the\s+\w+|to\s+\w+|when\s+\w+)',
            r'^(?:here\s+(?:are|is)\s+)',
        ]
        for pat in direct_openers:
            if re.search(pat, first_100):
                score += 0.5
                break
        
        # === 11. CONDITIONAL AND NUANCED LANGUAGE ===
        nuance_patterns = [
            r'\bif\s+.{5,}?\bthen\b', r'\bon\s+one\s+hand\b',
            r'\bit\'?s?\s+(?:important|worth)\s+(?:to\s+)?not(?:e|ing)\b',
            r'\bkeep\s+in\s+mind\b', r'\bconsider(?:ing)?\b',
            r'\bthere\s+are\s+(?:several|multiple|various|many|different)\b',
            r'\bdepends\s+on\b',
        ]
        nuance_count = 0
        for pat in nuance_patterns:
            nuance_count += len(re.findall(pat, resp_lower))
        
        score += min(nuance_count * 1.5, 5.0)
        
        # === 12. MATHEMATICAL/TECHNICAL NOTATION (bonus for technical queries) ===
        has_math = bool(re.search(r'[=+\-*/^]', resp)) and bool(re.search(r'\d', resp))
        has_formula = bool(re.search(r'[a-zA-Z]\s*=\s*[\d(]', resp))
        has_latex = bool(re.search(r'\\[a-z]+\{', resp))
        
        # Check if query seems technical
        technical_query_words = {'calculate', 'find', 'solve', 'equation', 'formula',
                                  'speed', 'distance', 'mass', 'energy', 'force',
                                  'angle', 'coefficient', 'function', 'algorithm'}
        query_is_technical = bool(set(query_clean.lower().split()) & technical_query_words)
        
        if query_is_technical:
            if has_formula:
                score += 3.0
            if has_latex:
                score += 2.0
            if has_math:
                score += 1.0
        
        # === FINAL NORMALIZATION ===
        # Clamp to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 50.0