def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant focuses on:
    - Citation/reference patterns (specific names, dates, numbers as factual anchors)
    - Hallucination red flags (overly precise unsourced stats, absolute claims)
    - Appropriate hedging vs over-hedging
    - Sensationalism and conspiracy language detection
    - Structured reasoning indicators (step-by-step, logical connectives)
    - Source attribution language
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. SPECIFIC FACTUAL ANCHORS ===
        # Dates (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', response)
        # Specific numbers with units
        number_with_unit = re.findall(r'\b\d+[\.,]?\d*\s*(kg|km|m|miles|feet|meters|degrees|°|%|lbs|pounds|gallons|liters|hours|minutes|seconds|mph|m/s|cm|mm|inches)\b', response_lower)
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response)
        
        factual_anchor_score = 0.0
        factual_anchor_score += min(len(year_pattern) * 1.5, 6)
        factual_anchor_score += min(len(date_pattern) * 2, 4)
        factual_anchor_score += min(len(number_with_unit) * 1.0, 5)
        factual_anchor_score += min(len(proper_nouns) * 0.5, 5)
        
        score += factual_anchor_score
        
        # === 2. HALLUCINATION RED FLAGS ===
        # Overly precise unsourced statistics
        suspiciously_precise = re.findall(r'\b\d{1,3}\.\d{2,}\s*%', response)
        score -= len(suspiciously_precise) * 3
        
        # Absolute/universal claims
        absolute_phrases = [
            'always', 'never', 'everyone knows', 'it is a fact that',
            'without exception', 'in all cases', 'no one has ever',
            'proven beyond doubt', 'undeniable', 'irrefutable',
            'there is no doubt', 'absolutely certain', 'guaranteed',
            'every single', 'without a doubt', '100%'
        ]
        absolute_count = sum(1 for phrase in absolute_phrases if phrase in response_lower)
        score -= absolute_count * 2.5
        
        # === 3. SENSATIONALISM & CONSPIRACY LANGUAGE ===
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'unbelievable',
            'mind-blowing', 'jaw-dropping', 'outrageous', 'insane',
            'they don\'t want you to know', 'cover-up', 'coverup',
            'conspiracy', 'wake up', 'sheeple', 'mainstream media lies',
            'big pharma', 'deep state', 'new world order',
            'exposed', 'revealed the truth', 'suppressed',
            'what they\'re hiding', 'the real truth'
        ]
        sensational_count = sum(1 for w in sensational_words if w in response_lower)
        score -= sensational_count * 5
        
        # Excessive exclamation marks (sensationalism indicator)
        exclamation_count = response.count('!')
        if exclamation_count > 3:
            score -= (exclamation_count - 3) * 1.5
        
        # === 4. APPROPRIATE HEDGING ===
        hedging_phrases = [
            'may', 'might', 'could', 'possibly', 'potentially',
            'it appears', 'it seems', 'suggests that', 'indicates that',
            'approximately', 'roughly', 'around', 'about',
            'in general', 'typically', 'usually', 'often',
            'depending on', 'it depends', 'varies',
            'one possible', 'some evidence', 'research suggests',
            'according to', 'based on', 'studies suggest',
            'it is likely', 'it is possible', 'there is evidence'
        ]
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
        # Moderate hedging is good, too much is bad
        if hedge_count <= 8:
            score += hedge_count * 1.2
        else:
            score += 8 * 1.2 - (hedge_count - 8) * 0.5  # diminishing/negative returns
        
        # === 5. SOURCE ATTRIBUTION LANGUAGE ===
        source_phrases = [
            'according to', 'research shows', 'studies have shown',
            'a study by', 'researchers found', 'data from',
            'published in', 'reported by', 'as noted by',
            'the evidence suggests', 'findings indicate',
            'based on research', 'scientific consensus',
            'peer-reviewed', 'meta-analysis', 'systematic review',
            'as described in', 'documented in', 'referenced in'
        ]
        source_count = sum(1 for phrase in source_phrases if phrase in response_lower)
        score += min(source_count * 2.5, 8)
        
        # === 6. LOGICAL STRUCTURE & REASONING CONNECTIVES ===
        reasoning_connectives = [
            'therefore', 'consequently', 'as a result', 'because',
            'since', 'thus', 'hence', 'this means',
            'in contrast', 'however', 'on the other hand',
            'furthermore', 'moreover', 'additionally',
            'for example', 'for instance', 'such as',
            'specifically', 'in particular', 'namely',
            'first', 'second', 'third', 'finally',
            'in summary', 'to summarize', 'in conclusion'
        ]
        connective_count = sum(1 for c in reasoning_connectives if c in response_lower)
        score += min(connective_count * 0.8, 6)
        
        # === 7. ENUMERATION AND STRUCTURED PRESENTATION ===
        # Numbered lists or bullet points with content
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+\S', response)
        bullet_items = re.findall(r'(?:^|\n)\s*[-*•]\s+\S', response)
        has_structure = len(numbered_items) + len(bullet_items)
        
        if has_structure >= 2:
            score += min(has_structure * 0.7, 4)
        
        # Bold/formatted key terms (indicates organized thought)
        bold_terms = re.findall(r'\*\*[^*]+\*\*', response)
        score += min(len(bold_terms) * 0.4, 3)
        
        # === 8. RESPONSE COMPLETENESS SIGNALS ===
        # Check if response seems truncated
        if response.rstrip()[-1:] not in '.!?"\')' and word_count > 20:
            score -= 3  # Likely truncated
        
        # Average sentence length (too short = superficial, too long = rambling)
        avg_sent_len = word_count / num_sentences
        if 10 <= avg_sent_len <= 25:
            score += 3
        elif avg_sent_len < 6:
            score -= 2
        elif avg_sent_len > 35:
            score -= 2
        
        # === 9. QUERY RELEVANCE via key term matching ===
        # Extract meaningful query terms (>3 chars, not stopwords)
        stopwords = {'what', 'when', 'where', 'which', 'that', 'this', 'with', 
                     'from', 'your', 'have', 'will', 'been', 'were', 'they',
                     'their', 'about', 'would', 'could', 'should', 'there',
                     'some', 'other', 'than', 'then', 'them', 'these', 'those',
                     'does', 'doing', 'being', 'having', 'here', 'just',
                     'also', 'very', 'much', 'many', 'more', 'most', 'into',
                     'over', 'such', 'only', 'need', 'help', 'want', 'like',
                     'know', 'think', 'make', 'good', 'well'}
        query_terms = [w for w in query_lower.split() if len(w) > 3 and w not in stopwords]
        if query_terms:
            matched = sum(1 for t in query_terms if t in response_lower)
            relevance_ratio = matched / len(query_terms)
            score += relevance_ratio * 5
        
        # === 10. CONFIDENCE CALIBRATION ===
        # Penalize responses that are overconfident on uncertain topics
        uncertain_query_markers = ['think', 'opinion', 'believe', 'should', 'would', 'could',
                                    'possible', 'happening', 'going on']
        query_is_uncertain = any(m in query_lower for m in uncertain_query_markers)
        
        if query_is_uncertain:
            # In uncertain contexts, hedging is extra valuable
            score += min(hedge_count * 0.5, 3)
            # And absolute claims are extra bad
            score -= absolute_count * 1.5
        
        # === 11. MATHEMATICAL/TECHNICAL PRECISION ===
        # Formulas, equations, calculations
        math_patterns = re.findall(r'[=×÷±∑∫√π]|\\frac|\\times|\^2|\^3', response)
        equation_patterns = re.findall(r'\b[A-Za-z]+\s*=\s*[\d\(]', response)
        score += min((len(math_patterns) + len(equation_patterns)) * 0.5, 4)
        
        # === 12. OPENING QUALITY ===
        # Responses that start with direct engagement tend to be better
        opening = response_lower[:100]
        
        # Filler openings (slightly penalize)
        filler_openings = ['that\'s a great', 'great question', 'sure!', 'of course!',
                           'absolutely!', 'wow,', 'oh,']
        for filler in filler_openings:
            if opening.startswith(filler):
                score -= 1
                break
        
        # Direct/substantive openings (reward)
        substantive_starts = ['the ', 'in ', 'to ', 'there ', 'a ', 'an ']
        for start in substantive_starts:
            if opening.startswith(start):
                score += 0.5
                break
        
        # === 13. VOCABULARY SOPHISTICATION (without being pretentious) ===
        # Unique word ratio as proxy for information density
        unique_words = len(set(words))
        if word_count > 10:
            vocab_ratio = unique_words / word_count
            # Sweet spot: 0.4-0.7 unique ratio
            if 0.4 <= vocab_ratio <= 0.7:
                score += 3
            elif vocab_ratio > 0.7:
                score += 1.5  # Still okay, just very diverse
            else:
                score += 0  # Very repetitive
        
        # === 14. CONDITIONAL/NUANCED LANGUAGE ===
        conditional_phrases = [
            'if ', 'whether ', 'in some cases', 'under certain',
            'on one hand', 'alternatively', 'that said',
            'it\'s worth noting', 'keep in mind', 'note that',
            'an important consideration', 'one caveat',
            'pros and cons', 'advantages and disadvantages',
            'trade-off', 'tradeoff'
        ]
        conditional_count = sum(1 for p in conditional_phrases if p in response_lower)
        score += min(conditional_count * 1.5, 5)
        
        # === 15. LENGTH BONUS (moderate) ===
        # Longer responses tend to be more informative, but with diminishing returns
        if word_count >= 30:
            length_bonus = math.log(word_count / 30 + 1) * 2
            score += min(length_bonus, 5)
        
        # Clamp score to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0