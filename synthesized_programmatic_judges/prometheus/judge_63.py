def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    a novel approach based on:
    1. Claim density analysis (ratio of assertive claims to total content)
    2. Evidential reasoning patterns (because, due to, as a result, etc.)
    3. Perspective-taking markers (acknowledging multiple viewpoints)
    4. Conditional/modal verb analysis (would, could, might vs is, must, will)
    5. Source attribution patterns
    6. Emotional attunement and empathy markers
    7. Structural sophistication (not just bullets, but reasoning chains)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0

        import re
        import math
        from collections import Counter

        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        word_count = len(words)

        if word_count < 5:
            return 1.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)

        # ============================================================
        # 1. MODAL VERB SPECTRUM ANALYSIS
        # Analyze the spectrum from strong certainty to uncertainty
        # ============================================================
        
        # Epistemic modals indicating appropriate uncertainty
        soft_modals = ['could', 'might', 'may', 'would', 'can']
        soft_modal_count = sum(1 for w in words if w in soft_modals)
        
        # Strong/absolute modals indicating high certainty
        strong_modals = ['must', 'shall', 'will', 'always', 'never', 'definitely',
                         'certainly', 'absolutely', 'undoubtedly', 'obviously',
                         'clearly', 'undeniably', 'without doubt', 'guaranteed']
        strong_modal_count = sum(1 for w in words if w in strong_modals)
        
        # Ratio: prefer soft modals over strong ones
        modal_ratio = (soft_modal_count + 0.5) / (strong_modal_count + soft_modal_count + 1.0)
        modal_score = modal_ratio * 10  # 0-10

        # ============================================================
        # 2. EVIDENTIAL REASONING PATTERNS
        # Look for causal/explanatory connectives that show reasoning
        # ============================================================
        
        evidential_patterns = [
            r'\bbecause\b', r'\bdue to\b', r'\bas a result\b', r'\btherefore\b',
            r'\bconsequently\b', r'\bthis means\b', r'\bwhich leads to\b',
            r'\bfor this reason\b', r'\bgiven that\b', r'\bsince\b',
            r'\bthis is why\b', r'\bit follows\b', r'\bhence\b',
            r'\bexplains why\b', r'\bthe reason\b', r'\bin order to\b'
        ]
        evidential_count = sum(len(re.findall(p, response_lower)) for p in evidential_patterns)
        evidential_density = evidential_count / num_sentences
        evidential_score = min(evidential_density * 5, 10)

        # ============================================================
        # 3. PERSPECTIVE-TAKING AND ACKNOWLEDGMENT
        # Recognizing the user's situation, feelings, viewpoint
        # ============================================================
        
        perspective_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bit\'s understandable\b',
            r'\bthat\'s understandable\b', r'\bit makes sense\b',
            r'\byour (feelings?|concerns?|situation|experience|frustration)\b',
            r'\bit\'s (completely |perfectly |totally )?(ok|okay|fine|normal|natural|valid)\b',
            r'\bi hear\b', r'\bi (can )?imagine\b', r'\bfrom your perspective\b',
            r'\bin your (shoes|position|situation)\b', r'\bcompletely understandable\b',
            r'\bperfectly (fine|normal|ok|okay|natural|understandable)\b',
            r'\bit\'s (a |quite )?(common|normal|natural)\b',
            r'\byou\'re (right|feeling)\b', r'\byou (might|may) (be )?feel\b'
        ]
        perspective_count = sum(len(re.findall(p, response_lower)) for p in perspective_patterns)
        perspective_score = min(perspective_count * 2.5, 10)

        # ============================================================
        # 4. CLAIM DENSITY AND ASSERTIVENESS ANALYSIS
        # Count declarative assertions vs qualified statements
        # ============================================================
        
        # Declarative assertion patterns (potentially overconfident)
        assertion_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b'
        ]
        assertion_count = sum(len(re.findall(p, response_lower)) for p in assertion_patterns)
        
        # Qualification patterns
        qualification_patterns = [
            r'\b(perhaps|possibly|potentially|arguably|seemingly)\b',
            r'\b(tends? to|often|sometimes|usually|generally|typically)\b',
            r'\b(in (many|some|most) cases)\b',
            r'\b(it (seems?|appears?|looks? like))\b',
            r'\b(one (way|approach|option|possibility))\b',
            r'\b(you (might|could|may) (want|consider|try))\b',
            r'\b(consider|keep in mind|bear in mind|note that)\b'
        ]
        qualification_count = sum(len(re.findall(p, response_lower)) for p in qualification_patterns)
        
        # Balance ratio: qualifications relative to raw assertions
        if assertion_count > 0:
            qual_ratio = qualification_count / (assertion_count * 0.3 + 1)
        else:
            qual_ratio = qualification_count * 0.5
        qualification_score = min(qual_ratio * 5, 10)

        # ============================================================
        # 5. STRUCTURAL REASONING CHAINS
        # Look for multi-step reasoning, not just lists
        # ============================================================
        
        # Sequential reasoning markers
        reasoning_chain_patterns = [
            r'\bfirst(ly)?\b.*\bthen\b', r'\bif\b.*\bthen\b',
            r'\bon one hand\b', r'\bon the other hand\b',
            r'\bwhile\b.*\b(also|however|but)\b',
            r'\bnot only\b.*\bbut also\b',
            r'\balthough\b', r'\bhowever\b', r'\bnevertheless\b',
            r'\bthat (said|being said)\b', r'\bat the same time\b',
            r'\bbalance\b', r'\bnuance\b'
        ]
        chain_count = sum(1 for p in reasoning_chain_patterns if re.search(p, response_lower))
        chain_score = min(chain_count * 2, 10)

        # ============================================================
        # 6. EMPATHY AND EMOTIONAL INTELLIGENCE MARKERS
        # ============================================================
        
        empathy_patterns = [
            r'\bi\'m (sorry|glad|happy|pleased)\b',
            r'\bsorry to hear\b', r'\bthat (must|sounds) (be |like )?(hard|tough|difficult|frustrating)\b',
            r'\bgive yourself\b', r'\bbe (kind|gentle|patient) (to|with) yourself\b',
            r'\btake (your |a )?time\b', r'\bit\'s okay to\b',
            r'\bfeel(ing)? (this|that|the) way\b', r'\byour (pain|grief|loss)\b',
            r'\bhang in there\b', r'\byou\'re not alone\b',
            r'\bwe (value|appreciate|understand)\b',
            r'\bsincerely\b', r'\bgenuinely\b'
        ]
        empathy_count = sum(len(re.findall(p, response_lower)) for p in empathy_patterns)
        
        # Check if query seems emotional/needs empathy
        emotional_query_indicators = [
            'feeling', 'feel', 'stress', 'frustrat', 'sad', 'angry', 'upset',
            'heartbroken', 'devastat', 'loneli', 'despair', 'exhaust', 'struggle',
            'difficult', 'tough', 'regret', 'passed away', 'breakup', 'broke up'
        ]
        query_is_emotional = any(ind in query_lower for ind in emotional_query_indicators)
        
        if query_is_emotional:
            empathy_score = min(empathy_count * 2.5, 10)
        else:
            empathy_score = min(empathy_count * 1.5, 5)  # Less weight if not emotional context

        # ============================================================
        # 7. NEGATIVE INDICATOR: DISMISSIVENESS / FALSE SIMPLICITY
        # ============================================================
        
        dismissive_patterns = [
            r'\bjust (do|get|try|buy|find|move|keep)\b',
            r'\byou should be able to\b',
            r'\bit\'s (just|only|simply) a\b',
            r'\bget over it\b', r'\bmove on\b',
            r'\bdon\'t (worry|let it|think about)\b',
            r'\bstop (being|feeling|thinking)\b',
            r'\byou\'re (just|probably) not\b',
            r'\bmaybe you\'re (just|not)\b',
            r'\bremember.*(just|only)\b'
        ]
        dismissive_count = sum(len(re.findall(p, response_lower)) for p in dismissive_patterns)
        dismissive_penalty = min(dismissive_count * 1.5, 8)

        # ============================================================
        # 8. OVERCONFIDENCE PENALTY
        # Penalize absolute statements on ambiguous/complex topics
        # ============================================================
        
        overconfident_phrases = [
            r'\bthe (only|best|right|correct) (way|answer|solution|approach)\b',
            r'\byou (need|have|must) to\b',
            r'\bthere\'s no (other|better)\b',
            r'\bwithout (a )?doubt\b',
            r'\b100 ?%\b', r'\bguaranteed?\b',
            r'\beveryone knows\b', r'\bit\'s (a )?fact\b',
            r'\bobviously\b', r'\bclearly\b'
        ]
        overconfident_count = sum(len(re.findall(p, response_lower)) for p in overconfident_phrases)
        overconfidence_penalty = min(overconfident_count * 1.5, 8)

        # ============================================================
        # 9. RESPONSE COMPLETENESS AND DEPTH
        # ============================================================
        
        # Reward adequate length (not too short, not excessively long)
        if word_count < 20:
            length_score = 2
        elif word_count < 50:
            length_score = 5
        elif word_count < 150:
            length_score = 8
        elif word_count < 300:
            length_score = 10
        else:
            length_score = 9

        # ============================================================
        # 10. QUERY-RESPONSE ALIGNMENT
        # Check if the response actually addresses what was asked
        # ============================================================
        
        # Extract meaningful words from query (not stopwords)
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                     'than', 'too', 'very', 'just', 'because', 'if', 'when',
                     'where', 'how', 'what', 'which', 'who', 'whom', 'this',
                     'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
                     'our', 'ours', 'you', 'your', 'he', 'him', 'his', 'she',
                     'her', 'it', 'its', 'they', 'them', 'their', 'about', 'up',
                     'out', 'then', 'there', 'here'}
        
        query_words = set(re.findall(r'\b[a-z]+\b', query_lower)) - stopwords
        response_words = set(re.findall(r'\b[a-z]+\b', response_lower)) - stopwords
        
        if query_words:
            overlap = len(query_words & response_words) / len(query_words)
        else:
            overlap = 0.5
        alignment_score = min(overlap * 12, 10)

        # ============================================================
        # 11. TONE APPROPRIATENESS
        # Does the response match the expected tone?
        # ============================================================
        
        # Check for casual markers when formality might be needed
        casual_markers = [r'\bhey\b', r'\bdude\b', r'\bkinda\b', r'\bgonna\b',
                         r'\bwanna\b', r'\byeah\b', r'\bnope\b', r'\bcool\b',
                         r'\bawesome\b', r'\bstuff\b', r'\bthing(s|y)?\b']
        casual_count = sum(len(re.findall(p, response_lower)) for p in casual_markers)
        
        # Check if query expects casual tone
        query_casual = any(re.search(p, query_lower) for p in 
                          [r'\bcasual\b', r'\blaid.?back\b', r'\bslang\b', r'\binformal\b'])
        
        if query_casual:
            # Casual is appropriate - reward it
            tone_score = min(casual_count * 1.5, 8)
        else:
            # Penalize excessive casualness in non-casual contexts
            tone_score = max(7 - casual_count * 0.8, 2)

        # ============================================================
        # 12. ACTIONABILITY - Does it provide concrete, useful guidance?
        # ============================================================
        
        actionable_patterns = [
            r'\btry\b', r'\bconsider\b', r'\bstart (by|with)\b',
            r'\bhere (are|is)\b', r'\bstep\b', r'\bfirst\b',
            r'\bnext\b', r'\bfinally\b', r'\bfor (example|instance)\b',
            r'\bsuch as\b', r'\blike\b', r'\bspecifically\b',
            r'\bfor instance\b', r'\bremember to\b', r'\bmake sure\b',
            r'\bdon\'t forget\b', r'\bkeep in mind\b'
        ]
        actionable_count = sum(len(re.findall(p, response_lower)) for p in actionable_patterns)
        actionable_score = min(actionable_count * 1.2, 10)

        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        # Weighted combination
        raw_score = (
            modal_score * 0.08 +
            evidential_score * 0.08 +
            perspective_score * 0.12 +
            qualification_score * 0.07 +
            chain_score * 0.06 +
            empathy_score * 0.13 +
            length_score * 0.10 +
            alignment_score * 0.12 +
            tone_score * 0.06 +
            actionable_score * 0.08 +
            - dismissive_penalty * 0.10 +
            - overconfidence_penalty * 0.08
        )
        
        # Normalize to 1-5 scale
        # raw_score theoretical range: roughly -1.5 to 10
        # Map to 1-5
        final_score = 1.0 + (raw_score / 10.0) * 4.0
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        # Round to 1 decimal
        final_score = round(final_score, 1)
        
        return final_score

    except Exception:
        return 3.0