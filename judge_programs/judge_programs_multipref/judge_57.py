def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication in LLM responses.
    
    Focuses on:
    - Appropriate use of hedging/uncertainty language
    - Distinguishing facts from speculation
    - Avoiding overconfident claims on ambiguous topics
    - Structured, well-organized communication
    - Appropriate confidence calibration given the query type
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not query:
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        response_words = response_lower.split()
        query_words = query_lower.split()
        num_words = len(response_words)
        
        if num_words < 3:
            return 1.0
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # 1. QUERY CLASSIFICATION: Determine if query needs uncertainty
        # ============================================================
        
        # Opinion/subjective queries - should have nuanced responses
        opinion_markers = [
            'do you think', 'what do you think', 'should', 'is it better',
            'what\'s your opinion', 'would you recommend', 'is it worth',
            'what are your thoughts', 'how do you feel', 'best way',
            'is it true that', 'is it possible', 'can you believe',
            'what would happen if', 'why do people'
        ]
        is_opinion_query = any(m in query_lower for m in opinion_markers)
        
        # Factual/technical queries - confidence is more appropriate
        factual_markers = [
            'how to', 'how do', 'how can i', 'what is', 'what are',
            'explain', 'describe', 'define', 'calculate', 'find the',
            'steps to', 'recipe', 'instructions', 'directions'
        ]
        is_factual_query = any(m in query_lower for m in factual_markers)
        
        # Ambiguous or uncertain topic queries
        uncertain_topic_markers = [
            'what\'s happening', 'recent', 'latest', 'news', 'current',
            'future', 'predict', 'will', 'forecast', 'upcoming'
        ]
        is_uncertain_topic = any(m in query_lower for m in uncertain_topic_markers)
        
        # ============================================================
        # 2. HEDGING AND UNCERTAINTY LANGUAGE ANALYSIS
        # ============================================================
        
        # Appropriate hedging phrases
        hedging_phrases = [
            'likely', 'unlikely', 'possibly', 'perhaps', 'may', 'might',
            'could be', 'it seems', 'it appears', 'suggests', 'indicates',
            'research suggests', 'studies suggest', 'evidence suggests',
            'generally', 'typically', 'often', 'usually', 'tends to',
            'in most cases', 'it depends', 'depending on', 'varies',
            'approximately', 'roughly', 'about', 'around',
            'one possibility', 'another option', 'alternatively',
            'in my opinion', 'from my perspective', 'i believe',
            'it\'s worth noting', 'keep in mind', 'however',
            'on the other hand', 'that said', 'although',
            'not necessarily', 'it\'s important to note',
            'there are several', 'can be', 'some', 'many',
            'while', 'though', 'but', 'nevertheless'
        ]
        
        hedging_count = 0
        for phrase in hedging_phrases:
            hedging_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', response_lower))
        
        hedging_density = hedging_count / max(num_words, 1) * 100
        
        # ============================================================
        # 3. OVERCONFIDENCE DETECTION
        # ============================================================
        
        overconfident_phrases = [
            'always', 'never', 'absolutely', 'definitely', 'certainly',
            'without a doubt', 'undoubtedly', 'unquestionably',
            'there is no', 'it is impossible', 'everyone knows',
            'obviously', 'clearly', 'of course', 'no question',
            'the fact is', 'the truth is', 'without exception',
            'guaranteed', 'proven fact', 'indisputable',
            'you must', 'you should always', 'you should never'
        ]
        
        overconfident_count = 0
        for phrase in overconfident_phrases:
            overconfident_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', response_lower))
        
        overconfident_density = overconfident_count / max(num_words, 1) * 100
        
        # ============================================================
        # 4. STRUCTURAL QUALITY (correlates with thoughtful responses)
        # ============================================================
        
        # Markdown formatting (headers, bold, lists)
        has_headers = bool(re.search(r'#{1,4}\s', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
        
        structure_score = 0
        if has_headers:
            structure_score += 2
        if has_bold:
            structure_score += 1.5
        if has_numbered_list:
            structure_score += 1.5
        if has_bullet_list:
            structure_score += 1
        
        # Paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        if num_paragraphs >= 3:
            structure_score += 2
        elif num_paragraphs >= 2:
            structure_score += 1
        
        # ============================================================
        # 5. NUANCE AND MULTIPLE PERSPECTIVES
        # ============================================================
        
        perspective_markers = [
            'on the other hand', 'however', 'alternatively',
            'another perspective', 'some argue', 'others believe',
            'pros and cons', 'advantages and disadvantages',
            'one option', 'another option', 'depending on',
            'it depends', 'there are several', 'various',
            'multiple', 'different approaches', 'considerations'
        ]
        
        perspective_count = 0
        for phrase in perspective_markers:
            if phrase in response_lower:
                perspective_count += 1
        
        # ============================================================
        # 6. ACKNOWLEDGMENT OF LIMITATIONS
        # ============================================================
        
        limitation_phrases = [
            'i\'m not sure', 'i don\'t know', 'i\'m not aware',
            'i cannot', 'i can\'t', 'beyond my', 'outside my',
            'as of my', 'my knowledge', 'my training',
            'consult a professional', 'seek advice', 'talk to',
            'verify', 'double-check', 'check with',
            'this may vary', 'results may vary', 'individual results',
            'not a substitute', 'disclaimer'
        ]
        
        limitation_count = 0
        for phrase in limitation_phrases:
            if phrase in response_lower:
                limitation_count += 1
        
        # ============================================================
        # 7. SENTENCE VARIETY AND COMPLEXITY
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length
        avg_sentence_len = num_words / num_sentences
        
        # Sentence length variance (good writing has variety)
        if num_sentences > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            sent_variety = min(math.sqrt(variance) / max(mean_len, 1), 1.0)
        else:
            sent_variety = 0
        
        # ============================================================
        # 8. VOCABULARY RICHNESS
        # ============================================================
        
        # Type-token ratio (unique words / total words)
        unique_words = len(set(response_words))
        ttr = unique_words / max(num_words, 1)
        
        # ============================================================
        # 9. RESPONSE COMPLETENESS
        # ============================================================
        
        # Check if response appears truncated
        is_truncated = response.rstrip()[-1] not in '.!?"\')' if response.rstrip() else True
        
        # Reasonable length (not too short, not excessively long)
        length_score = 0
        if 50 <= num_words <= 500:
            length_score = 3
        elif 30 <= num_words < 50:
            length_score = 1.5
        elif num_words > 500:
            length_score = 2
        elif num_words < 30:
            length_score = 0.5
        
        # ============================================================
        # 10. CONTEXTUAL APPROPRIATENESS
        # ============================================================
        
        # Greeting/engagement phrases
        engagement_phrases = [
            'great question', 'that\'s a great', 'certainly',
            'absolutely', 'of course', 'happy to help',
            'let\'s', 'here are', 'here\'s', 'i\'d be happy',
            'glad you asked', 'interesting question'
        ]
        
        engagement_count = sum(1 for p in engagement_phrases if p in response_lower)
        
        # ============================================================
        # SCORING ASSEMBLY
        # ============================================================
        
        # Hedging appropriateness (context-dependent)
        if is_opinion_query or is_uncertain_topic:
            # For opinion/uncertain queries, hedging is very valuable
            hedging_bonus = min(hedging_density * 3.0, 8.0)
            overconfident_penalty = overconfident_density * 5.0
        elif is_factual_query:
            # For factual queries, moderate hedging is fine, overconfidence less penalized
            hedging_bonus = min(hedging_density * 1.5, 4.0)
            overconfident_penalty = overconfident_density * 2.0
        else:
            # Default
            hedging_bonus = min(hedging_density * 2.0, 6.0)
            overconfident_penalty = overconfident_density * 3.5
        
        score += hedging_bonus
        score -= overconfident_penalty
        
        # Structure bonus
        score += min(structure_score, 8.0)
        
        # Perspective/nuance bonus
        score += min(perspective_count * 2.0, 8.0)
        
        # Limitation acknowledgment bonus (especially for uncertain topics)
        if is_uncertain_topic:
            score += min(limitation_count * 3.0, 6.0)
        else:
            score += min(limitation_count * 1.5, 4.0)
        
        # Sentence variety bonus
        score += sent_variety * 4.0
        
        # Vocabulary richness
        # TTR naturally decreases with length, so adjust
        adjusted_ttr = ttr * math.log(max(num_words, 2))
        score += min(adjusted_ttr * 2.0, 5.0)
        
        # Length score
        score += length_score
        
        # Truncation penalty
        if is_truncated:
            score -= 3.0
        
        # Engagement bonus (mild)
        score += min(engagement_count * 0.5, 2.0)
        
        # ============================================================
        # ADDITIONAL QUALITY SIGNALS
        # ============================================================
        
        # Transition words (show logical flow)
        transition_words = [
            'first', 'second', 'third', 'additionally', 'furthermore',
            'moreover', 'in addition', 'finally', 'in conclusion',
            'therefore', 'thus', 'consequently', 'as a result',
            'for example', 'for instance', 'specifically',
            'in particular', 'namely', 'such as'
        ]
        transition_count = sum(1 for t in transition_words if t in response_lower)
        score += min(transition_count * 0.8, 4.0)
        
        # Conditional language (shows calibrated thinking)
        conditional_phrases = [
            'if you', 'depending on', 'in case', 'when possible',
            'where applicable', 'as needed', 'you may want to',
            'consider', 'you might', 'it would be'
        ]
        conditional_count = sum(1 for c in conditional_phrases if c in response_lower)
        score += min(conditional_count * 1.2, 5.0)
        
        # Comparative language (shows balanced thinking)
        comparative_phrases = [
            'compared to', 'in contrast', 'whereas', 'while',
            'on one hand', 'on the other', 'rather than',
            'instead of', 'as opposed to', 'unlike'
        ]
        comparative_count = sum(1 for c in comparative_phrases if c in response_lower)
        score += min(comparative_count * 1.5, 4.0)
        
        # ============================================================
        # PENALTY: Presenting speculation as fact
        # ============================================================
        
        # Strong claims without evidence markers
        strong_claim_patterns = [
            r'\bis\b.*\bthe best\b', r'\bis\b.*\bthe worst\b',
            r'\bis\b.*\bthe only\b', r'\bwill\b.*\bdefinitely\b',
            r'\bwill\b.*\bcertainly\b', r'\bno one\b.*\bcan\b',
            r'\beveryone\b.*\bshould\b', r'\beveryone\b.*\bknows\b'
        ]
        
        strong_claim_count = 0
        for pattern in strong_claim_patterns:
            strong_claim_count += len(re.findall(pattern, response_lower))
        
        score -= strong_claim_count * 2.0
        
        # ============================================================
        # BONUS: Well-structured options/alternatives
        # ============================================================
        
        # Presenting multiple options (epistemic humility)
        option_patterns = [
            r'option \d', r'approach \d', r'method \d',
            r'way \d', r'alternative \d', r'choice \d',
            r'### \d', r'\d+\.\s+\*\*'
        ]
        
        option_count = 0
        for pattern in option_patterns:
            option_count += len(re.findall(pattern, response_lower))
        
        if option_count >= 2:
            score += min(option_count * 1.0, 4.0)
        
        # ============================================================
        # NORMALIZE to 0-100 range
        # ============================================================
        
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            return max(0.0, min(100.0, len(response.split()) * 0.1 + 25.0))
        except:
            return 25.0