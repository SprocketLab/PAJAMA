def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Contextual appropriateness of certainty level (matching query type)
    2. Gradient of confidence language (not just binary hedge/confident)
    3. Evidence attribution patterns
    4. Claim density analysis (ratio of claims to supporting context)
    5. Structural indicators of thoughtful reasoning
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        
        if len(words) < 3:
            return 1.0
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # 1. QUERY TYPE CLASSIFICATION - determines expected calibration
        # ============================================================
        # Factual queries expect more confidence; opinion/speculative queries expect more hedging
        
        factual_indicators = ['how are', 'how is', 'how do', 'how to', 'what is', 'what are',
                              'explain', 'describe', 'define', 'tell me about', 'steps to',
                              'recipe', 'calculate', 'find the', 'solve']
        opinion_indicators = ['do you think', 'should', 'opinion', 'what do you feel',
                              'is it better', 'would you', 'best way', 'recommend']
        speculative_indicators = ['what if', 'could', 'might', 'future', 'predict',
                                  'what will', "what's happening", 'why is', 'why do']
        
        query_type_score = 0  # -1 = factual, 0 = neutral, 1 = speculative/opinion
        for ind in factual_indicators:
            if ind in query_lower:
                query_type_score -= 1
                break
        for ind in opinion_indicators:
            if ind in query_lower:
                query_type_score += 1
                break
        for ind in speculative_indicators:
            if ind in query_lower:
                query_type_score += 0.5
                break
        
        query_type_score = max(-1, min(1, query_type_score))
        
        # ============================================================
        # 2. GRADUATED CONFIDENCE LANGUAGE ANALYSIS
        # ============================================================
        # Instead of binary, we score on a spectrum
        
        # Strong hedging (high uncertainty)
        strong_hedge = ['it is unclear', 'remains debated', 'no consensus', 'highly uncertain',
                        'we cannot be sure', 'it is difficult to say', 'evidence is mixed',
                        'there is no definitive', 'opinions vary', 'it depends on']
        
        # Moderate hedging (appropriate uncertainty)
        moderate_hedge = ['likely', 'probably', 'research suggests', 'evidence suggests',
                          'it appears', 'generally', 'typically', 'in most cases',
                          'tends to', 'may', 'might', 'could', 'often', 'usually',
                          'it seems', 'arguably', 'potentially', 'approximately',
                          'roughly', 'some experts', 'many believe']
        
        # Epistemic markers (showing awareness of knowledge limits)
        epistemic_markers = ['i believe', 'in my view', 'from my understanding',
                             'as far as i know', 'to my knowledge', 'i think',
                             'it\'s worth noting', 'keep in mind', 'note that',
                             'however', 'that said', 'on the other hand',
                             'it\'s important to consider', 'there are several',
                             'one perspective', 'another view']
        
        # Overconfidence markers
        overconfident = ['definitely', 'absolutely', 'certainly', 'without a doubt',
                         'there is no question', 'it is obvious', 'clearly',
                         'undeniably', 'unquestionably', 'always', 'never',
                         'everyone knows', 'the fact is', 'the truth is',
                         'it is proven', 'without exception', 'guaranteed']
        
        # Evidence/source attribution
        evidence_markers = ['according to', 'studies show', 'research indicates',
                            'data suggests', 'experts say', 'historically',
                            'based on', 'for example', 'for instance',
                            'such as', 'e.g.', 'i.e.', 'specifically',
                            'in particular', 'one example']
        
        strong_hedge_count = sum(1 for phrase in strong_hedge if phrase in response_lower)
        moderate_hedge_count = sum(1 for phrase in moderate_hedge if phrase in response_lower)
        epistemic_count = sum(1 for phrase in epistemic_markers if phrase in response_lower)
        overconfident_count = sum(1 for phrase in overconfident if phrase in response_lower)
        evidence_count = sum(1 for phrase in evidence_markers if phrase in response_lower)
        
        # Normalize by number of sentences
        hedge_density = (strong_hedge_count * 2 + moderate_hedge_count) / num_sentences
        epistemic_density = epistemic_count / num_sentences
        overconfident_density = overconfident_count / num_sentences
        evidence_density = evidence_count / num_sentences
        
        # ============================================================
        # 3. CLAIM DENSITY ANALYSIS
        # ============================================================
        # Count declarative statements vs. qualified ones
        
        declarative_patterns = [
            r'\b(?:is|are|was|were)\s+(?:a|an|the)\b',
            r'\b(?:this|that|it)\s+(?:is|was|will)\b',
        ]
        
        qualified_patterns = [
            r'\b(?:may|might|could|can)\s+(?:be|have|cause|lead|result)\b',
            r'\b(?:if|when|unless|although|while)\b',
            r'\b(?:some|many|most|few|several|certain)\b',
        ]
        
        declarative_count = 0
        for pat in declarative_patterns:
            declarative_count += len(re.findall(pat, response_lower))
        
        qualified_count = 0
        for pat in qualified_patterns:
            qualified_count += len(re.findall(pat, response_lower))
        
        # Ratio of qualified to total claims
        total_claims = declarative_count + qualified_count
        if total_claims > 0:
            qualification_ratio = qualified_count / total_claims
        else:
            qualification_ratio = 0.5  # neutral
        
        # ============================================================
        # 4. STRUCTURAL DEPTH INDICATORS
        # ============================================================
        
        # Multi-perspective indicators
        contrast_words = ['however', 'although', 'but', 'nevertheless', 'on the other hand',
                          'conversely', 'in contrast', 'while', 'whereas', 'yet',
                          'alternatively', 'despite']
        contrast_count = sum(1 for w in contrast_words if w in response_lower)
        
        # Conditional reasoning
        conditional_patterns = re.findall(r'\bif\b.*?\bthen\b|\bif\b.*?,', response_lower)
        conditional_count = len(conditional_patterns)
        
        # Numbered/structured points (indicates organized thinking)
        has_structure = bool(re.search(r'(?:\d+\.|#{1,3}\s|\*\*.*?\*\*|•|-\s)', response))
        
        # Depth: average sentence length in words (moderate is good)
        avg_sentence_len = len(words) / num_sentences
        # Optimal range: 12-25 words per sentence
        if 12 <= avg_sentence_len <= 25:
            sentence_quality = 1.0
        elif 8 <= avg_sentence_len < 12 or 25 < avg_sentence_len <= 35:
            sentence_quality = 0.7
        else:
            sentence_quality = 0.4
        
        # ============================================================
        # 5. RESPONSE COMPLETENESS AND ENGAGEMENT
        # ============================================================
        
        # Check if response appears truncated
        is_truncated = response.rstrip()[-1] not in '.!?")\']' if response.rstrip() else True
        
        # Response length (reasonable range rewards)
        word_count = len(words)
        if word_count < 20:
            length_score = 0.3
        elif 20 <= word_count < 50:
            length_score = 0.6
        elif 50 <= word_count <= 300:
            length_score = 1.0
        elif 300 < word_count <= 500:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # Query engagement - does the response address the query?
        query_words = set(query_lower.split()) - {'a', 'an', 'the', 'is', 'are', 'to', 'in',
                                                     'of', 'and', 'or', 'for', 'i', 'my', 'me',
                                                     'do', 'you', 'can', 'how', 'what', 'why',
                                                     'it', 'that', 'this', 'with', 'from', 'on'}
        if query_words:
            overlap = sum(1 for w in query_words if w in response_lower)
            query_relevance = min(overlap / max(len(query_words), 1), 1.0)
        else:
            query_relevance = 0.5
        
        # ============================================================
        # 6. UNIQUE LEXICAL RICHNESS (type-token ratio on content words)
        # ============================================================
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                      'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this', 'it', 'its',
                      'you', 'your', 'i', 'my', 'me', 'we', 'our', 'they', 'their', 'he',
                      'she', 'him', 'her', 'not', 'no', 'so', 'up', 'out', 'about'}
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        if len(content_words) > 5:
            # Use root TTR (type-token ratio adjusted for length)
            types = len(set(content_words))
            tokens = len(content_words)
            ttr = types / math.sqrt(tokens)  # Guiraud's index
            lexical_richness = min(ttr / 8.0, 1.0)  # normalize
        else:
            lexical_richness = 0.3
        
        # ============================================================
        # 7. SCORING COMPOSITION
        # ============================================================
        
        # Calibration score: reward appropriate uncertainty for query type
        # For factual queries: moderate confidence is OK, some hedging is fine
        # For opinion/speculative: more hedging expected
        
        calibration_score = 0.0
        
        # Base epistemic quality
        calibration_score += min(hedge_density * 3, 1.5)  # up to 1.5
        calibration_score += min(epistemic_density * 4, 1.5)  # up to 1.5
        calibration_score += min(evidence_density * 3, 1.5)  # up to 1.5
        calibration_score -= overconfident_density * 2.0  # penalty
        
        # Adjust for query type appropriateness
        if query_type_score > 0:  # opinion/speculative query
            # Reward more hedging
            calibration_score += min(hedge_density * 2, 1.0)
            calibration_score -= overconfident_density * 1.5  # extra penalty
        elif query_type_score < 0:  # factual query
            # Slight reward for confidence when appropriate, but still penalize overconfidence
            calibration_score += 0.3  # small bonus for being willing to state facts
            calibration_score -= overconfident_density * 1.0
        
        # Qualification ratio contribution
        calibration_score += qualification_ratio * 1.5
        
        # Contrast/nuance bonus
        calibration_score += min(contrast_count * 0.3, 1.0)
        calibration_score += min(conditional_count * 0.2, 0.5)
        
        # Structural and presentation quality
        structure_score = 0.0
        structure_score += sentence_quality * 1.5
        structure_score += (1.0 if has_structure else 0.0) * 1.0
        structure_score += length_score * 1.5
        structure_score += query_relevance * 1.5
        structure_score += lexical_richness * 1.5
        
        # Truncation penalty
        if is_truncated:
            structure_score *= 0.85
        
        # Combine scores
        # Calibration: 0-~7 range, Structure: 0-~7 range
        # Weight calibration at 40%, structure at 60% (since many examples are factual)
        raw_score = calibration_score * 0.4 + structure_score * 0.6
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
    
    except Exception:
        return 5.0