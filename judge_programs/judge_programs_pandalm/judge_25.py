def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    Higher scores = better quality based on factual reliability signals.
    
    This variant focuses on a lexical/structural approach:
    - Presence of specific factual markers (numbers, dates, proper nouns)
    - Appropriate hedging language
    - Absence of hallucination red flags
    - Absence of sensationalism/conspiracy language
    - Structural completeness and informativeness
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
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        if len(response_stripped) == 0:
            return 0.0
        
        score = 50.0  # Start at midpoint out of 100
        
        # ---- 1. Response length and completeness (0 to +15) ----
        resp_len = len(response_stripped)
        word_count = len(response_stripped.split())
        
        if word_count < 3:
            score -= 20.0
        elif word_count < 10:
            score -= 5.0
        elif word_count < 20:
            score += 2.0
        elif word_count < 80:
            score += 8.0 + min(7.0, (word_count - 20) * 0.12)
        elif word_count < 200:
            score += 12.0
        else:
            score += 10.0  # Very long might be verbose
        
        # ---- 2. Sentence completeness ----
        # Check if response ends mid-sentence (truncation)
        if response_stripped and response_stripped[-1] not in '.!?"\')':
            # Likely truncated
            score -= 8.0
        
        # Count complete sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences >= 2:
            score += min(5.0, num_sentences * 1.0)
        
        # ---- 3. Factual specificity indicators ----
        # Numbers and dates suggest specific factual claims
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response_stripped)
        num_count = len(numbers)
        if 1 <= num_count <= 5:
            score += num_count * 1.5
        elif num_count > 5:
            score += 7.0  # cap benefit
        
        # Dates (years, specific dates)
        dates = re.findall(r'\b(?:19|20)\d{2}\b', response_stripped)
        if dates:
            score += min(4.0, len(dates) * 2.0)
        
        # Proper nouns (capitalized words not at sentence start)
        words = response_stripped.split()
        proper_nouns = 0
        for i, w in enumerate(words):
            if i > 0 and w and w[0].isupper() and not re.match(r'^[.!?]', words[i-1][-1:] if words[i-1] else ''):
                clean_w = w.strip(string.punctuation)
                if len(clean_w) > 1 and clean_w not in {'The', 'This', 'That', 'These', 'Those', 'It', 'I', 'A', 'An'}:
                    proper_nouns += 1
        if proper_nouns > 0:
            score += min(4.0, proper_nouns * 0.8)
        
        # ---- 4. Appropriate hedging language (sign of factual care) ----
        hedging_phrases = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends? to\b', r'\bin general\b', r'\bapproximately\b',
            r'\babout\b', r'\baround\b', r'\bestimated\b',
            r'\blikely\b', r'\bunlikely\b', r'\bperhaps\b',
            r'\bsome\b', r'\bmany\b', r'\bmost\b',
            r'\baccording to\b', r'\bresearch suggests\b',
            r'\bit is (widely )?believed\b', r'\bevidence suggests\b',
        ]
        resp_lower = response_stripped.lower()
        hedge_count = 0
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, resp_lower))
        
        if 1 <= hedge_count <= 6:
            score += hedge_count * 1.2
        elif hedge_count > 6:
            score += 7.0  # cap
        
        # ---- 5. Citation/source indicators ----
        citation_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstudy\b', r'\bstudies\b',
            r'\bsource\b', r'\breport\b', r'\bfound that\b',
            r'\bdata\b', r'\bevidence\b', r'\bexpert\b',
            r'\bscientist\b', r'\bprofessor\b', r'\buniversity\b',
            r'\bjournal\b', r'\bpublished\b',
        ]
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, resp_lower))
        if citation_count > 0:
            score += min(5.0, citation_count * 1.5)
        
        # ---- 6. PENALTY: Hallucination red flags ----
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+%', response_stripped)
        if precise_stats:
            score -= len(precise_stats) * 3.0
        
        # Absolute claims without hedging
        absolute_phrases = [
            r'\balways\b', r'\bnever\b', r'\bimpossible\b',
            r'\bguaranteed\b', r'\babsolutely\b', r'\bundeniably\b',
            r'\bwithout a doubt\b', r'\bno one\b', r'\beveryone knows\b',
            r'\bit is certain\b', r'\bproven fact\b',
            r'\b100%\b', r'\bcompletely\b',
        ]
        absolute_count = 0
        for pattern in absolute_phrases:
            absolute_count += len(re.findall(pattern, resp_lower))
        if absolute_count > 0:
            score -= absolute_count * 2.0
        
        # ---- 7. PENALTY: Sensationalism and conspiracy language ----
        sensational_words = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind-blowing\b',
            r'\binsane\b', r'\bconspiracy\b', r'\bcover-up\b', r'\bcoverup\b',
            r'\bthey don\'t want you to know\b', r'\bhidden truth\b',
            r'\bwake up\b', r'\bsheeple\b', r'\bdeep state\b',
            r'\bsecret(ly)?\b', r'\bmanipulat\w*\b',
            r'\bexposed\b', r'\bbombshell\b', r'\bbreaking\b',
            r'\byou won\'t believe\b', r'\binsider\b',
            r'\bthey are hiding\b', r'\blies\b', r'\bhoax\b',
        ]
        sensational_count = 0
        for pattern in sensational_words:
            sensational_count += len(re.findall(pattern, resp_lower))
        if sensational_count > 0:
            score -= sensational_count * 3.0
        
        # ---- 8. PENALTY: Repetition (sign of low quality / hallucination) ----
        if word_count > 5:
            word_list = [w.lower().strip(string.punctuation) for w in words if w.strip(string.punctuation)]
            word_freq = Counter(word_list)
            if word_list:
                most_common_word, most_common_count = word_freq.most_common(1)[0]
                # Exclude common stop words from repetition penalty
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 
                             'to', 'of', 'in', 'for', 'on', 'with', 'that', 'this', 'it',
                             'be', 'as', 'by', 'at', 'from', 'its', 'their', 'they', 'can'}
                
                # Check for excessive repetition of content words
                content_words = [w for w in word_list if w not in stop_words and len(w) > 2]
                if content_words:
                    content_freq = Counter(content_words)
                    top_content_word, top_content_count = content_freq.most_common(1)[0]
                    repetition_ratio = top_content_count / len(content_words)
                    if repetition_ratio > 0.3 and top_content_count > 3:
                        score -= 15.0
                    elif repetition_ratio > 0.2 and top_content_count > 3:
                        score -= 8.0
                
                # Check for repeated phrases (bigrams/trigrams)
                if len(word_list) > 6:
                    trigrams = [' '.join(word_list[i:i+3]) for i in range(len(word_list)-2)]
                    trigram_freq = Counter(trigrams)
                    if trigram_freq:
                        top_trigram, top_tri_count = trigram_freq.most_common(1)[0]
                        if top_tri_count > 3:
                            score -= min(15.0, top_tri_count * 3.0)
        
        # ---- 9. Structural quality indicators ----
        # Use of transitional/explanatory language
        explanatory_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bthis means\b', r'\bin other words\b', r'\bthat is\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bhowever\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\badditionally\b', r'\balso\b',
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
        ]
        explanatory_count = 0
        for pattern in explanatory_patterns:
            explanatory_count += len(re.findall(pattern, resp_lower))
        if explanatory_count > 0:
            score += min(6.0, explanatory_count * 1.0)
        
        # ---- 10. Query relevance (basic keyword overlap) ----
        query_words = set(w.lower().strip(string.punctuation) for w in query_stripped.split() 
                         if len(w.strip(string.punctuation)) > 3)
        resp_words = set(w.lower().strip(string.punctuation) for w in response_stripped.split()
                        if len(w.strip(string.punctuation)) > 3)
        
        if query_words:
            overlap = len(query_words & resp_words) / len(query_words)
            score += overlap * 5.0
        
        # ---- 11. Vocabulary diversity (type-token ratio for content) ----
        if word_count > 10:
            all_words_lower = [w.lower().strip(string.punctuation) for w in words if w.strip(string.punctuation)]
            if all_words_lower:
                ttr = len(set(all_words_lower)) / len(all_words_lower)
                if ttr > 0.7:
                    score += 3.0
                elif ttr > 0.5:
                    score += 1.5
                elif ttr < 0.3:
                    score -= 5.0
        
        # ---- 12. Check for pure echo/copy of query ----
        if response_stripped.strip().lower() == query_stripped.strip().lower():
            score -= 25.0
        
        # Check if response is just a trivial restatement
        if resp_len < len(query_stripped) * 0.5 and word_count < 8:
            score -= 5.0
        
        # ---- 13. Penalize <noinput> or placeholder responses ----
        if resp_lower.strip() in ['<noinput>', 'noinput', 'n/a', 'none', 'no response']:
            score -= 30.0
        
        # Clamp score to [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0  # Safe fallback mid-low score