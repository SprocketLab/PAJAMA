def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication in LLM responses.
    
    Focuses on:
    - Appropriate hedging language for uncertain claims
    - Distinguishing facts from speculation
    - Avoiding overconfident absolute statements on ambiguous topics
    - Quality of epistemic markers
    - Overall response quality as a baseline
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        # If response is essentially empty or just whitespace
        if len(response_lower) < 3:
            return 0.0
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. HEDGING AND CALIBRATION LANGUAGE (reward appropriate uncertainty)
        # ============================================================
        
        # Positive hedging words/phrases that show calibrated uncertainty
        hedging_phrases = [
            'likely', 'unlikely', 'probably', 'possibly', 'perhaps',
            'may', 'might', 'could be', 'it seems', 'it appears',
            'suggests', 'research suggests', 'studies suggest',
            'evidence suggests', 'generally', 'typically', 'often',
            'in most cases', 'tends to', 'it is believed',
            'according to', 'some argue', 'one perspective',
            'it depends', 'arguably', 'approximately', 'roughly',
            'estimated', 'potential', 'potentially',
            'in some cases', 'not always', 'can vary',
            'it is possible', 'there is evidence', 'commonly',
            'frequently', 'sometimes', 'on the other hand',
            'however', 'although', 'while', 'whereas',
            'to some extent', 'in certain contexts', 'depending on',
        ]
        
        hedging_count = 0
        for phrase in hedging_phrases:
            count = response_lower.count(phrase)
            hedging_count += count
        
        # ============================================================
        # 2. OVERCONFIDENCE MARKERS (penalize false certainty)
        # ============================================================
        
        overconfident_phrases = [
            'always', 'never', 'certainly', 'definitely', 'absolutely',
            'without a doubt', 'undoubtedly', 'unquestionably',
            'there is no question', 'it is clear that', 'obviously',
            'of course', 'everyone knows', 'it is a fact',
            'the truth is', 'without exception', 'guaranteed',
            'proven beyond', 'indisputable', 'irrefutable',
            'no one can deny', 'it is obvious', 'clearly',
        ]
        
        overconfident_count = 0
        for phrase in overconfident_phrases:
            count = response_lower.count(phrase)
            overconfident_count += count
        
        # ============================================================
        # 3. DETECT QUERY TYPE - is this factual or opinion/speculative?
        # ============================================================
        
        # Queries that are more likely to need hedging
        speculative_indicators = [
            'what do you think', 'opinion', 'hypothetical', 'predict',
            'future', 'might', 'could', 'would', 'should',
            'compare', 'contrast', 'argue', 'debate', 'controversial',
            'best', 'worst', 'better', 'why do', 'what if',
            'describe', 'explain', 'meaning', 'meant', 'interpret',
        ]
        
        # Queries that are more factual/instructional
        factual_indicators = [
            'rewrite', 'generate', 'create', 'write', 'list',
            'provide', 'give me', 'make', 'come up with',
            'crop', 'edit', 'convert', 'translate', 'summarize',
        ]
        
        is_speculative_query = any(ind in query_lower for ind in speculative_indicators)
        is_factual_query = any(ind in query_lower for ind in factual_indicators)
        
        # ============================================================
        # 4. SCORING: Hedging and calibration
        # ============================================================
        
        # Reward hedging, especially for speculative queries
        if is_speculative_query:
            # For speculative/opinion queries, hedging is more important
            hedging_bonus = min(hedging_count * 2.5, 12.0)
            overconfidence_penalty = overconfident_count * 3.0
        elif is_factual_query:
            # For factual/creative tasks, hedging matters less
            hedging_bonus = min(hedging_count * 0.8, 5.0)
            overconfidence_penalty = overconfident_count * 1.0
        else:
            # Default
            hedging_bonus = min(hedging_count * 1.5, 8.0)
            overconfidence_penalty = overconfident_count * 2.0
        
        score += hedging_bonus
        score -= overconfidence_penalty
        
        # ============================================================
        # 5. RESPONSE COMPLETENESS AND SUBSTANCE
        # ============================================================
        
        words = response.split()
        word_count = len(words)
        
        # Very short responses are usually lower quality
        if word_count < 5:
            score -= 15.0
        elif word_count < 10:
            score -= 8.0
        elif word_count < 20:
            score -= 2.0
        elif word_count >= 20 and word_count <= 200:
            # Good range - reward proportionally
            score += min(word_count * 0.15, 8.0)
        elif word_count > 200:
            # Still okay but diminishing returns
            score += 8.0
        
        # ============================================================
        # 6. STRUCTURAL QUALITY INDICATORS
        # ============================================================
        
        # Sentence count (more sentences = more developed response)
        import re
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        if sentence_count >= 2:
            score += min(sentence_count * 1.0, 6.0)
        
        # ============================================================
        # 7. REPETITION DETECTION (penalize repetitive content)
        # ============================================================
        
        # Check for repeated words
        if word_count > 3:
            unique_words = set(w.lower() for w in words)
            uniqueness_ratio = len(unique_words) / word_count
            
            if uniqueness_ratio < 0.3:
                score -= 20.0  # Extremely repetitive
            elif uniqueness_ratio < 0.5:
                score -= 10.0  # Very repetitive
            elif uniqueness_ratio < 0.6:
                score -= 5.0   # Somewhat repetitive
            elif uniqueness_ratio > 0.75:
                score += 3.0   # Good vocabulary diversity
        
        # Check for repeated phrases (bigrams)
        if word_count > 6:
            words_lower = [w.lower() for w in words]
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            if bigrams:
                from collections import Counter
                bigram_counts = Counter(bigrams)
                most_common_count = bigram_counts.most_common(1)[0][1]
                if most_common_count > 3:
                    score -= min(most_common_count * 2.0, 15.0)
        
        # ============================================================
        # 8. EPISTEMIC STRUCTURE: Does response distinguish claims?
        # ============================================================
        
        # Phrases that show epistemic awareness (distinguishing fact from opinion)
        epistemic_structure_phrases = [
            'on one hand', 'on the other hand', 'in contrast',
            'for example', 'for instance', 'such as',
            'this means', 'in other words', 'that is',
            'specifically', 'in particular', 'notably',
            'while', 'whereas', 'but', 'however', 'although',
            'both', 'differ', 'similar', 'different',
        ]
        
        structure_count = 0
        for phrase in epistemic_structure_phrases:
            if phrase in response_lower:
                structure_count += 1
        
        score += min(structure_count * 1.5, 8.0)
        
        # ============================================================
        # 9. RESPONSE RELEVANCE (basic check)
        # ============================================================
        
        # Check if response words overlap with query words
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        
        # Remove very common words for relevance check
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'or', 'but', 'not', 'no', 'if', 'then', 'than',
                     'that', 'this', 'it', 'its', 'i', 'you', 'he', 'she', 'we',
                     'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
                     'what', 'which', 'who', 'when', 'where', 'how', 'why'}
        
        query_content_words = query_words - stopwords
        response_content_words = response_words - stopwords
        
        if query_content_words:
            overlap = query_content_words & response_content_words
            relevance_ratio = len(overlap) / len(query_content_words)
            score += relevance_ratio * 5.0
        
        # ============================================================
        # 10. EXPLANATION DEPTH
        # ============================================================
        
        # Causal/explanatory language shows deeper engagement
        explanatory_phrases = [
            'because', 'therefore', 'thus', 'hence', 'as a result',
            'this is because', 'the reason', 'due to', 'leads to',
            'results in', 'contributes to', 'enables', 'allows',
            'in order to', 'so that', 'which means',
        ]
        
        explanation_count = 0
        for phrase in explanatory_phrases:
            if phrase in response_lower:
                explanation_count += 1
        
        score += min(explanation_count * 1.5, 6.0)
        
        # ============================================================
        # 11. PENALIZE GIBBERISH / BROKEN RESPONSES
        # ============================================================
        
        # Check if response is cut off (ends mid-word or mid-sentence without punctuation)
        if response_lower and response_lower[-1] not in '.!?"\')':
            # Might be truncated
            score -= 3.0
        
        # Check for excessive special characters or formatting artifacts
        special_char_ratio = sum(1 for c in response if c in '[]{}|\\<>') / max(len(response), 1)
        if special_char_ratio > 0.1:
            score -= 8.0
        
        # Check for "<noinput>" or similar non-responses
        if '<noinput>' in response_lower or response_lower.strip() == 'noinput':
            score -= 25.0
        
        # ============================================================
        # 12. INFORMATION DENSITY
        # ============================================================
        
        # Count unique content-bearing words (rough proxy for information)
        if word_count > 0:
            content_words_in_response = response_words - stopwords
            info_density = len(content_words_in_response) / max(word_count, 1)
            
            if info_density > 0.5:
                score += 3.0
            elif info_density > 0.4:
                score += 1.5
            elif info_density < 0.2:
                score -= 3.0
        
        # ============================================================
        # 13. PRESENTATION OF MULTIPLE PERSPECTIVES
        # ============================================================
        
        perspective_markers = [
            'also', 'additionally', 'furthermore', 'moreover',
            'another', 'in addition', 'as well as', 'not only',
            'both', 'various', 'several', 'multiple',
        ]
        
        perspective_count = sum(1 for p in perspective_markers if p in response_lower)
        score += min(perspective_count * 1.0, 5.0)
        
        # ============================================================
        # FINAL: Clamp score to reasonable range [0, 100]
        # ============================================================
        
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception as e:
        # Never crash - return a neutral score
        return 25.0