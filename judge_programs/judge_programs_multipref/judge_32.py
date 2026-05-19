def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a novel approach focused on:
    - Citation/reference pattern detection (names, dates, numbers, sources)
    - Hallucination red-flag detection (unsourced precise stats, absolute claims)
    - Appropriate hedging vs. overconfidence calibration
    - Sensationalism/conspiracy language penalties
    - Structural credibility signals (logical connectors, evidence-based reasoning)
    - Specificity-to-hedging ratio (balanced factual claims)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower().strip()
        query_lower = query.lower().strip()
        words = resp_lower.split()
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. VERIFIABLE FACT INDICATORS ===
        # Detect specific dates (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', response)
        specific_dates_score = min(len(year_pattern) * 1.5 + len(date_pattern) * 2.0, 8.0)
        
        # Detect specific numbers/measurements (not just any digit)
        measurement_patterns = re.findall(
            r'\b\d+\.?\d*\s*(?:km|miles?|meters?|feet|inches|lbs?|kg|°[CF]|percent|%|mph|m/s|cm|mm|liters?|gallons?|oz|mg|grams?|hours?|minutes?|seconds?)\b',
            resp_lower
        )
        measurement_score = min(len(measurement_patterns) * 1.2, 6.0)
        
        # Detect proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[.!?]\s)[a-z].*?([A-Z][a-z]{2,})', response)
        mid_sentence_caps = re.findall(r'(?<=\s)([A-Z][a-z]{2,})(?=\s)', response)
        # Filter out common sentence starters
        common_starters = {'the', 'this', 'that', 'these', 'those', 'here', 'there', 'it', 'they', 'we', 'you', 'he', 'she', 'if', 'when', 'while', 'however', 'also', 'but', 'and', 'or', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'some', 'many', 'most', 'each', 'every'}
        proper_count = sum(1 for w in mid_sentence_caps if w.lower() not in common_starters)
        proper_noun_score = min(proper_count * 0.6, 5.0)
        
        score += specific_dates_score + measurement_score + proper_noun_score
        
        # === 2. CITATION/SOURCE SIGNALS ===
        citation_phrases = [
            'according to', 'research shows', 'studies show', 'studies suggest',
            'research indicates', 'data shows', 'evidence suggests', 'reports indicate',
            'as reported by', 'published in', 'based on', 'source:', 'reference:',
            'cited in', 'per the', 'as noted by', 'findings suggest',
            'documented', 'peer-reviewed', 'meta-analysis', 'systematic review'
        ]
        citation_count = sum(1 for phrase in citation_phrases if phrase in resp_lower)
        citation_score = min(citation_count * 2.5, 8.0)
        score += citation_score
        
        # === 3. HALLUCINATION RED FLAGS ===
        hallucination_penalty = 0.0
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d{2,}\s*%', response)  # e.g., "73.47%"
        hallucination_penalty += len(precise_stats) * 3.0
        
        # Absolute/universal claims without hedging
        absolute_phrases = [
            'always', 'never', 'every single', 'without exception',
            'guaranteed to', 'proven to', 'definitely will', 'certainly will',
            'impossible to', 'everyone knows', 'it is a fact that',
            'undeniable', 'unquestionable', 'beyond doubt',
            'no one has ever', 'all experts agree', '100% of',
            'the truth is', 'the fact is'
        ]
        absolute_count = sum(1 for phrase in absolute_phrases if phrase in resp_lower)
        hallucination_penalty += absolute_count * 2.0
        
        # Vague authority appeals
        vague_authority = [
            'many experts say', 'scientists say', 'they say that',
            'it is well known', 'everybody knows', 'common knowledge',
            'most people agree', 'it has been said'
        ]
        vague_count = sum(1 for phrase in vague_authority if phrase in resp_lower)
        hallucination_penalty += vague_count * 1.5
        
        score -= min(hallucination_penalty, 15.0)
        
        # === 4. SENSATIONALISM & CONSPIRACY PENALTIES ===
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'insane', 'crazy',
            'you won\'t believe', 'bombshell', 'explosive', 'devastating',
            'terrifying', 'horrifying', 'nightmare', 'catastrophic',
            'wake up', 'sheeple', 'mainstream media', 'cover-up', 'coverup',
            'they don\'t want you to know', 'hidden truth', 'big pharma',
            'deep state', 'conspiracy', 'hoax', 'fake news', 'propaganda',
            'brainwash', 'suppressed', 'censored truth', 'globalist'
        ]
        sensational_count = sum(1 for w in sensational_words if w in resp_lower)
        sensational_penalty = min(sensational_count * 4.0, 20.0)
        score -= sensational_penalty
        
        # === 5. APPROPRIATE HEDGING CALIBRATION ===
        # Good hedging: shows epistemic humility
        hedging_phrases = [
            'may', 'might', 'could', 'possibly', 'perhaps', 'likely',
            'it appears', 'it seems', 'suggests that', 'tends to',
            'in some cases', 'generally', 'typically', 'usually',
            'approximately', 'roughly', 'about', 'around',
            'it depends', 'depending on', 'in most cases',
            'not necessarily', 'it\'s worth noting', 'keep in mind',
            'however', 'on the other hand', 'that said',
            'to some extent', 'arguably', 'reportedly'
        ]
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in resp_lower)
        hedge_ratio = hedge_count / num_sentences
        
        # Optimal hedging: not too little, not too much
        if hedge_ratio < 0.1:
            hedge_score = -2.0  # Too assertive
        elif hedge_ratio < 0.3:
            hedge_score = 4.0   # Good balance
        elif hedge_ratio < 0.6:
            hedge_score = 5.0   # Nicely calibrated
        elif hedge_ratio < 1.0:
            hedge_score = 3.0   # Slightly over-hedged
        else:
            hedge_score = 0.0   # Too wishy-washy
        
        score += hedge_score
        
        # === 6. LOGICAL CONNECTORS & REASONING STRUCTURE ===
        reasoning_connectors = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'given that',
            'this means', 'which leads to', 'in order to',
            'for this reason', 'it follows that', 'specifically',
            'for example', 'for instance', 'such as', 'namely',
            'in particular', 'to illustrate', 'in other words',
            'first', 'second', 'third', 'finally', 'additionally',
            'moreover', 'furthermore', 'in addition'
        ]
        connector_count = sum(1 for c in reasoning_connectors if c in resp_lower)
        connector_ratio = connector_count / num_sentences
        reasoning_score = min(connector_ratio * 6.0, 7.0)
        score += reasoning_score
        
        # === 7. CONDITIONAL/NUANCED LANGUAGE ===
        nuance_patterns = [
            'on one hand', 'on the other hand', 'while', 'although',
            'despite', 'nevertheless', 'nonetheless', 'conversely',
            'in contrast', 'alternatively', 'rather than',
            'it\'s important to note', 'worth considering',
            'there are several', 'multiple factors', 'various',
            'both', 'pros and cons', 'advantages and disadvantages',
            'trade-off', 'tradeoff', 'nuance', 'context'
        ]
        nuance_count = sum(1 for p in nuance_patterns if p in resp_lower)
        nuance_score = min(nuance_count * 1.5, 6.0)
        score += nuance_score
        
        # === 8. RESPONSE COMPLETENESS & COHERENCE ===
        # Average sentence length (too short = superficial, too long = rambling)
        avg_sent_len = num_words / num_sentences
        if 10 <= avg_sent_len <= 25:
            coherence_bonus = 3.0
        elif 8 <= avg_sent_len <= 30:
            coherence_bonus = 1.5
        else:
            coherence_bonus = -1.0
        score += coherence_bonus
        
        # Sentence-to-sentence coherence: check for word overlap between consecutive sentences
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                words_a = set(sentences[i].lower().split())
                words_b = set(sentences[i + 1].lower().split())
                # Remove very common words
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                           'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with',
                           'as', 'at', 'by', 'this', 'from', 'or', 'but', 'not', 'have',
                           'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
                           'could', 'should', 'may', 'might', 'shall'}
                words_a -= stopwords
                words_b -= stopwords
                if words_a and words_b:
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    coherence_scores.append(overlap)
            if coherence_scores:
                avg_coherence = sum(coherence_scores) / len(coherence_scores)
                # Moderate coherence is good (0.1-0.4), too high means repetitive
                if 0.05 <= avg_coherence <= 0.35:
                    score += 3.0
                elif avg_coherence > 0.5:
                    score -= 2.0  # Repetitive
        
        # === 9. QUERY RELEVANCE (semantic alignment) ===
        query_words = set(query_lower.split()) - {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with', 'how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'i', 'my', 'me', 'you', 'your'}
        if query_words:
            response_word_set = set(resp_lower.split())
            relevance = len(query_words & response_word_set) / len(query_words)
            relevance_score = relevance * 5.0
            score += relevance_score
        
        # === 10. STRUCTURAL FORMATTING SIGNALS ===
        # Organized responses with clear structure tend to be more reliable
        has_numbered_steps = bool(re.search(r'\b[1-9]\.\s', response))
        has_bold_markers = '**' in response or '__' in response
        has_sections = bool(re.search(r'###?\s', response))
        
        structure_score = 0.0
        if has_numbered_steps:
            structure_score += 1.5
        if has_bold_markers:
            structure_score += 1.0
        if has_sections:
            structure_score += 1.5
        score += min(structure_score, 3.5)
        
        # === 11. INFORMATION DENSITY ===
        # Ratio of content words to total words
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with', 'as',
                         'at', 'by', 'this', 'from', 'or', 'but', 'not', 'have', 'has', 'had',
                         'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
                         'may', 'might', 'shall', 'if', 'then', 'than', 'so', 'very',
                         'just', 'also', 'too', 'here', 'there', 'its', 'their', 'our',
                         'your', 'my', 'his', 'her', 'we', 'they', 'you', 'he', 'she', 'i'}
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        content_ratio = len(content_words) / num_words if num_words > 0 else 0
        
        # Good information density: 0.4-0.7
        if 0.4 <= content_ratio <= 0.7:
            score += 2.0
        elif content_ratio > 0.7:
            score += 1.0  # Dense but potentially hard to read
        
        # === 12. OPENING QUALITY ===
        # Good factual responses typically start with direct, relevant content
        first_sentence = sentences[0].lower() if sentences else ""
        
        # Penalize filler openings
        filler_openings = [
            'great question', 'good question', 'that\'s a great',
            'that\'s a good', 'oh,', 'well,', 'hmm',
            'interesting question', 'nice question'
        ]
        for filler in filler_openings:
            if first_sentence.startswith(filler):
                score -= 1.5
                break
        
        # Reward direct, informative openings
        direct_openers = ['certainly', 'here', 'the', 'to', 'there are', 'no,', 'yes,']
        for opener in direct_openers:
            if first_sentence.startswith(opener):
                score += 0.5
                break
        
        # === 13. RESPONSE LENGTH CALIBRATION ===
        # Not too short (superficial) and not too long (rambling)
        if num_words < 20:
            score -= 5.0
        elif num_words < 50:
            score -= 2.0
        elif 50 <= num_words <= 300:
            score += 2.0
        elif num_words > 500:
            score -= 1.0  # Slight penalty for very long responses
        
        # === 14. DISCLAIMER/LIMITATION AWARENESS ===
        limitation_phrases = [
            'i\'m not', 'i am not', 'i don\'t have', 'i cannot',
            'consult a', 'seek professional', 'professional advice',
            'not a substitute', 'please note', 'important to note',
            'disclaimer', 'limitation', 'beyond my', 'as of my',
            'my knowledge', 'i should note', 'it\'s worth mentioning'
        ]
        limitation_count = sum(1 for p in limitation_phrases if p in resp_lower)
        # A little self-awareness is good, too much is deflecting
        if 1 <= limitation_count <= 3:
            score += 2.0
        elif limitation_count > 3:
            score += 0.5
        
        # Clamp score to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0