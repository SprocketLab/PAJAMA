def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication in LLM responses.
    
    This variant focuses on:
    1. Hedging/uncertainty markers and their appropriate usage
    2. Overconfidence detection (absolute/definitive language)
    3. Acknowledgment of limitations or ambiguity
    4. Empathetic and nuanced framing
    5. Structural sophistication (reasoning depth)
    6. Contextual responsiveness to the query's emotional/informational needs
    
    Uses a token-ratio and pattern-density approach (different from sentence-length/bullet/overlap).
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 1.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # ============================================================
        # 1. HEDGING & CALIBRATION LANGUAGE (positive signal)
        # ============================================================
        hedging_phrases = [
            r'\bit\'?s\s+(completely\s+)?understandable\b',
            r'\bit\'?s\s+(perfectly\s+)?(okay|ok|fine|natural|normal)\b',
            r'\bresearch\s+suggests\b', r'\bstudies\s+suggest\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\boften\b', r'\btend\s+to\b', r'\bmay\b', r'\bmight\b',
            r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b', r'\blikely\b',
            r'\bprobably\b', r'\bin\s+many\s+cases\b', r'\bin\s+some\s+cases\b',
            r'\bit\s+depends\b', r'\bit\s+can\s+vary\b',
            r'\bnot\s+always\b', r'\bnot\s+necessarily\b',
            r'\bcan\s+be\b', r'\bmay\s+be\b', r'\bmight\s+be\b',
            r'\bsome\s+people\b', r'\bsome\s+experts\b',
            r'\bappears?\s+to\b', r'\bseems?\s+to\b', r'\bseems?\s+like\b',
            r'\bto\s+some\s+extent\b', r'\bin\s+general\b',
            r'\bone\s+approach\b', r'\bone\s+way\b',
            r'\bfor\s+instance\b', r'\bfor\s+example\b',
        ]
        
        hedge_count = 0
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_density = hedge_count / max(num_words, 1) * 100
        hedge_score = min(hedge_density * 3.0, 10.0)  # cap at 10
        
        # ============================================================
        # 2. OVERCONFIDENCE / FALSE CERTAINTY DETECTION (negative signal)
        # ============================================================
        overconfident_patterns = [
            r'\byou\s+need\s+to\b', r'\byou\s+must\b', r'\byou\s+should\s+just\b',
            r'\bjust\s+do\b', r'\bjust\s+get\b', r'\bjust\s+make\b',
            r'\bobviously\b', r'\bclearly\b', r'\bdefinitely\b',
            r'\balways\b', r'\bnever\b', r'\babsolutely\b',
            r'\bwithout\s+a\s+doubt\b', r'\bundeniably\b',
            r'\beveryone\s+knows\b', r'\bno\s+question\b',
            r'\bthe\s+only\s+way\b', r'\bthe\s+best\s+way\b',
            r'\byou\'?re\s+just\b', r'\bget\s+over\s+it\b',
            r'\bstop\s+being\b', r'\bdon\'?t\s+be\b',
            r'\bsimply\s+put\b', r'\bit\'?s\s+simple\b',
            r'\bit\'?s\s+easy\b', r'\bjust\s+remember\b',
            r'\bmaybe\s+you\'?re\s+just\s+not\b',
        ]
        
        overconfident_count = 0
        for pattern in overconfident_patterns:
            overconfident_count += len(re.findall(pattern, response_lower))
        
        overconfident_density = overconfident_count / max(num_words, 1) * 100
        overconfident_penalty = min(overconfident_density * 4.0, 12.0)
        
        # ============================================================
        # 3. EMPATHETIC / ACKNOWLEDGING LANGUAGE (positive for emotional queries)
        # ============================================================
        empathy_patterns = [
            r'\bi\s+understand\b', r'\bi\s+can\s+(see|hear|imagine|sense)\b',
            r'\bthat\'?s\s+(totally\s+)?understandable\b',
            r'\bi\'?m\s+(genuinely\s+)?sorry\b', r'\bi\s+hear\s+you\b',
            r'\byour\s+feelings?\b', r'\byour\s+experience\b',
            r'\bcompletely\s+understandable\b', r'\bperfectly\s+(fine|okay|ok|normal|natural)\b',
            r'\bgive\s+yourself\b', r'\bbe\s+kind\s+to\s+yourself\b',
            r'\btake\s+(your\s+)?time\b', r'\btake\s+a\s+moment\b',
            r'\bvalid\b', r'\bnatural\s+(to\s+feel|process|response)\b',
            r'\bwe\s+(highly\s+)?value\b', r'\bsincerely\s+apologize\b',
            r'\bwe\'?re\s+here\b', r'\bi\'?m\s+here\b',
        ]
        
        # Detect if query is emotional
        emotional_query_words = [
            'stress', 'frustrated', 'sad', 'lonely', 'heartbroken', 'devastated',
            'anxious', 'worried', 'overwhelmed', 'upset', 'angry', 'fear',
            'regret', 'struggling', 'difficult', 'tough', 'hard time',
            'breakup', 'passed away', 'died', 'loss', 'grief', 'feeling down',
            'exhaustion', 'despair', 'loneliness', 'emotionally', 'comfort',
            'end of my tether', 'interrupt'
        ]
        
        is_emotional = any(w in query_lower for w in emotional_query_words)
        
        empathy_count = 0
        for pattern in empathy_patterns:
            empathy_count += len(re.findall(pattern, response_lower))
        
        empathy_score = 0.0
        if is_emotional:
            empathy_score = min(empathy_count * 1.5, 8.0)
        else:
            empathy_score = min(empathy_count * 0.5, 3.0)
        
        # ============================================================
        # 4. DISMISSIVE LANGUAGE DETECTION (negative, especially for emotional)
        # ============================================================
        dismissive_patterns = [
            r'\bjust\s+a\b', r'\bit\'?s\s+just\b', r'\bget\s+over\b',
            r'\bmove\s+on\b', r'\bstop\s+worrying\b', r'\bdon\'?t\s+let\s+it\b',
            r'\byou\'?ll\s+be\s+fine\b', r'\bno\s+big\s+deal\b',
            r'\bthat\'?s\s+a\s+bummer\b', r'\bwhatever\b',
            r'\bnot\s+using\s+it\s+correctly\b',
            r'\bread\s+the\s+manual\b',
            r'\byou\s+should\s+be\s+able\s+to\s+handle\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        dismissive_penalty = dismissive_count * (2.5 if is_emotional else 1.0)
        
        # ============================================================
        # 5. STRUCTURAL SOPHISTICATION (reasoning depth proxy)
        # ============================================================
        # Count distinct clauses via punctuation and connectors
        connectors = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin\s+addition\b', r'\bon\s+the\s+other\s+hand\b',
            r'\bthat\s+said\b', r'\bwhile\b', r'\balthough\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bconsequently\b',
            r'\btherefore\b', r'\bthus\b', r'\bas\s+a\s+result\b',
            r'\bremember\s+that\b', r'\bkeep\s+in\s+mind\b',
            r'\bit\'?s\s+(important|crucial|essential|worth)\b',
        ]
        
        connector_count = 0
        for pattern in connectors:
            connector_count += len(re.findall(pattern, response_lower))
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Avg sentence length variety (std dev of word counts per sentence)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
            variety_score = min(std_sl / 5.0, 1.5)
        else:
            variety_score = 0.0
        
        structure_score = min(connector_count * 0.6 + variety_score, 5.0)
        
        # ============================================================
        # 6. CONTEXTUAL RESPONSIVENESS
        # ============================================================
        # Does the response acknowledge ambiguity or ask for clarification when needed?
        ambiguity_words_in_query = [
            'ambiguous', 'unclear', 'no context', 'no previous context',
            'how to get there', 'interpret'
        ]
        query_is_ambiguous = any(w in query_lower for w in ambiguity_words_in_query)
        
        clarification_patterns = [
            r'\bcan\s+you\s+(provide|give|share|tell)\b',
            r'\bmore\s+(details?|information|context|specifics)\b',
            r'\bwithout\s+(further|more)\s+(details?|information|context)\b',
            r'\bwhat\s+(do\s+you\s+mean|are\s+you\s+referring)\b',
            r'\bcould\s+you\s+clarify\b', r'\bplease\s+specify\b',
            r'\bwhich\b.*\bare\s+you\s+referring\b',
        ]
        
        clarification_count = 0
        for pattern in clarification_patterns:
            clarification_count += len(re.findall(pattern, response_lower))
        
        clarification_score = 0.0
        if query_is_ambiguous:
            if clarification_count > 0:
                clarification_score = 4.0
            else:
                clarification_score = -3.0  # penalty for not acknowledging ambiguity
        
        # ============================================================
        # 7. VOCABULARY RICHNESS (type-token ratio as proxy for thoughtfulness)
        # ============================================================
        if num_words > 10:
            unique_words = len(set(words))
            ttr = unique_words / num_words
            # Normalize: typical TTR for good text is 0.5-0.7
            vocab_score = min(max((ttr - 0.3) * 5.0, 0.0), 3.0)
        else:
            vocab_score = 0.5
        
        # ============================================================
        # 8. RESPONSE LENGTH ADEQUACY
        # ============================================================
        # Too short responses are usually dismissive; too long may ramble
        if num_words < 20:
            length_score = -2.0
        elif num_words < 40:
            length_score = 0.0
        elif num_words < 60:
            length_score = 1.0
        elif num_words < 150:
            length_score = 2.0
        else:
            length_score = 1.5  # slight reduction for very long
        
        # ============================================================
        # 9. TONE MATCHING (casual query -> casual response, etc.)
        # ============================================================
        # Detect formality of query
        informal_markers_q = len(re.findall(r'\b(hey|yo|dude|gonna|wanna|kinda|sorta|lol|haha|chill|cool|awesome|killer|whip)\b', query_lower))
        informal_markers_r = len(re.findall(r'\b(hey|yo|dude|gonna|wanna|kinda|sorta|lol|haha|chill|cool|awesome|killer|whip|nifty|wild)\b', response_lower))
        
        formal_markers_r = len(re.findall(r'\b(furthermore|moreover|consequently|therefore|thus|hereby|henceforth|approximately)\b', response_lower))
        
        tone_match_score = 0.0
        if informal_markers_q > 0:
            # Query is informal, reward informal response
            if informal_markers_r > 0:
                tone_match_score = 1.5
            elif formal_markers_r > 2:
                tone_match_score = -1.0
        
        # ============================================================
        # 10. ACTIONABLE ADVICE WITH APPROPRIATE FRAMING
        # ============================================================
        actionable_patterns = [
            r'\bhere\s+are\s+some\b', r'\byou\s+(can|could|might)\s+try\b',
            r'\bone\s+approach\b', r'\bconsider\b', r'\bit\s+may\s+help\b',
            r'\ba\s+good\s+strategy\b', r'\bsome\s+ways\b',
            r'\bsteps?\b', r'\btips?\b', r'\bsuggestions?\b',
            r'\brecommend\b', r'\blet\'?s\b',
        ]
        
        actionable_count = 0
        for pattern in actionable_patterns:
            actionable_count += len(re.findall(pattern, response_lower))
        
        actionable_score = min(actionable_count * 0.7, 3.0)
        
        # ============================================================
        # 11. NEGATIVE: Fabrication / unsupported specificity without hedging
        # ============================================================
        # Detect very specific claims (numbers, directions) without hedging
        specific_claims = len(re.findall(r'\b(take\s+a\s+(left|right)\s+turn|turn\s+(left|right)|first\s+exit|second\s+exit)\b', response_lower))
        if specific_claims > 0 and hedge_count == 0 and query_is_ambiguous:
            fabrication_penalty = specific_claims * 3.0
        else:
            fabrication_penalty = 0.0
        
        # ============================================================
        # 12. NEGATIVE: Probabilistic language about capabilities ("might not", "probably won't")
        #     when describing what a system SHOULD do
        # ============================================================
        weak_capability_patterns = [
            r'\b(might\s+not|may\s+not|probably\s+won\'?t|it\s+probably)\b',
            r'\bit\s+might\s+not\s+be\s+able\b',
            r'\bit\s+may\s+not\s+be\s+able\b',
        ]
        
        # Check if query asks about system design/capabilities
        is_capability_query = any(w in query_lower for w in ['ai model', 'system', 'design', 'manage', 'keep track'])
        
        weak_cap_count = 0
        for pattern in weak_capability_patterns:
            weak_cap_count += len(re.findall(pattern, response_lower))
        
        weak_capability_penalty = 0.0
        if is_capability_query and weak_cap_count > 0:
            weak_capability_penalty = weak_cap_count * 2.0
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        total = (
            hedge_score * 0.8           # Appropriate hedging
            + empathy_score * 1.0       # Empathetic language
            + structure_score * 0.7     # Structural sophistication
            + clarification_score * 0.8 # Handling ambiguity
            + vocab_score * 0.6         # Vocabulary richness
            + length_score * 0.5        # Length adequacy
            + tone_match_score * 0.5    # Tone matching
            + actionable_score * 0.6    # Actionable framing
            - overconfident_penalty * 0.7  # Overconfidence
            - dismissive_penalty * 0.9     # Dismissiveness
            - fabrication_penalty * 0.8    # Fabrication
            - weak_capability_penalty * 0.6  # Weak capability language
        )
        
        # Normalize to 1-5 range
        # Raw total typically ranges from about -8 to +15
        # Map to 1-5
        normalized = 1.0 + (total + 5.0) * (4.0 / 20.0)
        
        # Clamp
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception as e:
        return 3.0