def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Contextual appropriateness of confidence level (matching query type)
    2. Presence of epistemic markers and their distribution throughout the response
    3. Absence of overconfident/absolutist language patterns
    4. Quality of reasoning structure (conditional/causal reasoning)
    5. Acknowledgment of limitations and alternative perspectives
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        words = resp_lower.split()
        
        if len(words) < 3:
            return 0.5
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Query-type detection and response calibration match
        # ============================================================
        # Detect if the query is about emotions/support, factual, ambiguous, or advice-seeking
        
        emotional_query_words = ['feeling', 'feel', 'sad', 'frustrated', 'stress', 'heartbroken',
                                  'devastated', 'lonely', 'loneliness', 'despair', 'exhausted',
                                  'down', 'upset', 'angry', 'anxious', 'worried', 'fear',
                                  'breakup', 'passed away', 'died', 'grief', 'comfort']
        
        ambiguous_query_words = ['ambiguous', 'unclear', 'no context', 'no previous context',
                                  'interpret', 'without further', 'vague']
        
        advice_query_words = ['how to', 'guide', 'advice', 'help me', 'suggest', 'recommend',
                              'explain', 'understand', 'struggling', 'need to', 'seeking']
        
        emotional_query_score = sum(1 for w in emotional_query_words if w in query_lower)
        ambiguous_query_score = sum(1 for w in ambiguous_query_words if w in query_lower)
        advice_query_score = sum(1 for w in advice_query_words if w in query_lower)
        
        is_emotional = emotional_query_score >= 2
        is_ambiguous = ambiguous_query_score >= 1
        is_advice = advice_query_score >= 2
        
        # ============================================================
        # FEATURE 2: Empathy and emotional acknowledgment (for emotional queries)
        # ============================================================
        empathy_phrases = [
            "i understand", "i can see", "i can hear", "it's understandable",
            "that's understandable", "completely understandable", "it's okay",
            "it's absolutely okay", "it's perfectly", "i'm sorry", "i'm genuinely sorry",
            "must be", "must feel", "natural to feel", "valid", "normal to",
            "give yourself", "it's fine to", "perfectly fine", "take your time",
            "i hear you", "that sounds", "sounds difficult", "sounds tough",
            "acknowledge", "recognize"
        ]
        empathy_count = sum(1 for phrase in empathy_phrases if phrase in resp_lower)
        empathy_score = min(empathy_count * 1.5, 8.0) if is_emotional else min(empathy_count * 0.5, 3.0)
        
        # ============================================================
        # FEATURE 3: Epistemic hedging markers (distributed through response)
        # ============================================================
        hedging_markers = [
            'perhaps', 'maybe', 'possibly', 'likely', 'unlikely', 'probably',
            'might', 'could be', 'may be', 'it seems', 'it appears',
            'suggests', 'research suggests', 'studies suggest', 'evidence suggests',
            'tends to', 'generally', 'typically', 'often', 'sometimes',
            'in some cases', 'depending on', 'it depends', 'not always',
            'can vary', 'varies', 'approximately', 'roughly', 'around',
            'to some extent', 'in many cases', 'for the most part'
        ]
        
        hedge_count = sum(1 for marker in hedging_markers if marker in resp_lower)
        # Check distribution: hedging in different thirds of the response
        thirds = [resp_lower[:len(resp_lower)//3], 
                  resp_lower[len(resp_lower)//3:2*len(resp_lower)//3],
                  resp_lower[2*len(resp_lower)//3:]]
        hedge_distribution = sum(1 for third in thirds 
                                  if any(m in third for m in hedging_markers))
        
        hedge_score = min(hedge_count * 0.8 + hedge_distribution * 0.5, 6.0)
        
        # ============================================================
        # FEATURE 4: Overconfidence detection (PENALTY)
        # ============================================================
        overconfident_patterns = [
            r'\byou must\b', r'\byou need to\b', r'\byou should just\b',
            r'\bjust do\b', r'\bobviously\b', r'\bclearly\b',
            r'\bwithout a doubt\b', r'\bundeniably\b', r'\babsolutely\b',
            r'\bdefinitely\b', r'\bcertainly\b', r'\balways\b',
            r'\bnever\b', r'\beveryone knows\b', r'\bit\'s simple\b',
            r'\bjust\b.*\bjust\b', r'\bget yourself together\b',
            r'\bget over it\b', r'\bstop\b.*\bworrying\b',
            r'\byou\'re just not\b', r'\bwon\'t be able\b',
            r'\bcan\'t\b.*\bhandle\b', r'\bmight not\b.*\bable\b',
            r'\bprobably won\'t\b'
        ]
        
        overconfident_count = 0
        for pattern in overconfident_patterns:
            matches = re.findall(pattern, resp_lower)
            overconfident_count += len(matches)
        
        # Context-sensitive: "just" used dismissively is worse
        dismissive_just = len(re.findall(r'\bjust\s+(do|get|move|stop|try|buy|make|keep)\b', resp_lower))
        overconfident_count += dismissive_just * 0.5
        
        overconfidence_penalty = min(overconfident_count * 1.2, 8.0)
        
        # ============================================================
        # FEATURE 5: Structural quality - reasoning connectives and conditional logic
        # ============================================================
        reasoning_connectives = [
            'because', 'therefore', 'however', 'although', 'while',
            'on the other hand', 'alternatively', 'in contrast',
            'for instance', 'for example', 'such as', 'consider',
            'if', 'when', 'in case', 'provided that', 'assuming',
            'this means', 'this suggests', 'which means', 'as a result',
            'keep in mind', 'remember that', 'it\'s worth noting',
            'importantly', 'notably'
        ]
        
        connective_count = sum(1 for c in reasoning_connectives if c in resp_lower)
        connective_score = min(connective_count * 0.6, 5.0)
        
        # ============================================================
        # FEATURE 6: Acknowledgment of limitations / seeking clarification
        # ============================================================
        limitation_phrases = [
            'without further', 'more information', 'could you', 'can you',
            'please provide', 'tell me more', 'what specific', 'which',
            'i\'m not sure', 'hard to say', 'difficult to determine',
            'it\'s not clear', 'there are different', 'there are various',
            'one approach', 'another way', 'alternatively',
            'this is just one', 'there may be other'
        ]
        
        limitation_count = sum(1 for phrase in limitation_phrases if phrase in resp_lower)
        # Higher reward for ambiguous queries
        if is_ambiguous:
            limitation_score = min(limitation_count * 2.0, 8.0)
        else:
            limitation_score = min(limitation_count * 0.8, 4.0)
        
        # ============================================================
        # FEATURE 7: Response engagement and personalization
        # ============================================================
        engagement_markers = [
            'you', 'your', 'yourself', 'you\'re', 'you\'ve',
            'let\'s', 'we can', 'together', 'imagine',
            'think of', 'picture', 'consider'
        ]
        
        engagement_count = sum(resp_lower.count(m) for m in engagement_markers)
        engagement_ratio = engagement_count / max(len(words), 1)
        # Moderate engagement is good, too much or too little is bad
        if engagement_ratio < 0.01:
            engagement_score = 1.0
        elif engagement_ratio < 0.05:
            engagement_score = 3.0
        elif engagement_ratio < 0.12:
            engagement_score = 5.0
        else:
            engagement_score = 3.5  # Slightly too much
        
        # ============================================================
        # FEATURE 8: Tone appropriateness
        # ============================================================
        # For emotional queries, check warmth vs coldness
        cold_phrases = [
            'it\'s noted', 'we suggest', 'you should be able',
            'it\'s unfortunate', 'that\'s a bummer', 'just a',
            'part of life', 'get some rest', 'you\'ll feel better',
            'not a big deal', 'no big deal', 'move on',
            'get over', 'suck it up'
        ]
        
        warm_phrases = [
            'take your time', 'it\'s okay to', 'perfectly normal',
            'here for you', 'don\'t hesitate', 'reach out',
            'you\'re not alone', 'we care', 'valued', 'we value',
            'dedicated to', 'committed to', 'sincerely',
            'genuinely', 'deeply', 'truly'
        ]
        
        cold_count = sum(1 for p in cold_phrases if p in resp_lower)
        warm_count = sum(1 for p in warm_phrases if p in resp_lower)
        
        if is_emotional:
            tone_score = warm_count * 1.5 - cold_count * 2.0
        else:
            tone_score = warm_count * 0.5 - cold_count * 0.5
        
        tone_score = max(min(tone_score, 6.0), -4.0)
        
        # ============================================================
        # FEATURE 9: Response length and completeness
        # ============================================================
        word_count = len(words)
        if word_count < 20:
            length_score = 1.0
        elif word_count < 40:
            length_score = 2.5
        elif word_count < 80:
            length_score = 4.0
        elif word_count < 150:
            length_score = 5.0
        elif word_count < 250:
            length_score = 4.5
        else:
            length_score = 3.5  # Overly verbose
        
        # ============================================================
        # FEATURE 10: Structured advice (numbered lists, step-by-step)
        # ============================================================
        has_numbered_list = bool(re.search(r'\b\d+[\.\)]\s', response))
        has_structure_words = sum(1 for w in ['first', 'second', 'third', 'finally', 'next', 'then', 'lastly']
                                  if w in resp_lower)
        
        structure_score = 0.0
        if has_numbered_list:
            structure_score += 2.0
        structure_score += min(has_structure_words * 0.5, 2.0)
        structure_score = min(structure_score, 4.0)
        
        # ============================================================
        # FEATURE 11: Avoidance of fabrication / making stuff up
        # ============================================================
        # For ambiguous queries, penalize responses that fabricate specific details
        fabrication_penalty = 0.0
        if is_ambiguous:
            # Check for overly specific directions/details without context
            specific_patterns = [
                r'take a (left|right)', r'turn (left|right)', r'continue straight',
                r'you will (see|find|reach)', r'the answer is', r'it is located'
            ]
            fab_count = sum(1 for p in specific_patterns if re.search(p, resp_lower))
            fabrication_penalty = fab_count * 2.5
        
        # ============================================================
        # FEATURE 12: Sentence-level quality variance
        # ============================================================
        # Good responses have consistent quality; check for very short dismissive sentences
        if num_sentences > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            very_short_sents = sum(1 for sl in sent_lengths if sl < 4)
            short_sent_ratio = very_short_sents / num_sentences
            
            if short_sent_ratio > 0.4:
                sentence_quality = -1.5  # Too many terse/dismissive sentences
            elif avg_sent_len > 8 and avg_sent_len < 30:
                sentence_quality = 2.0  # Good sentence length
            else:
                sentence_quality = 0.5
        else:
            sentence_quality = 0.0
        
        # ============================================================
        # AGGREGATE SCORING
        # ============================================================
        
        raw_score = (
            empathy_score * 1.0 +          # Up to 8
            hedge_score * 0.5 +             # Up to 3
            connective_score * 0.8 +        # Up to 4
            limitation_score * 0.7 +        # Up to 5.6
            engagement_score * 0.8 +        # Up to 4
            tone_score * 0.9 +              # -3.6 to 5.4
            length_score * 0.7 +            # Up to 3.5
            structure_score * 0.6 +         # Up to 2.4
            sentence_quality * 0.6 +        # -0.9 to 1.2
            - overconfidence_penalty * 1.0 + # Up to -8
            - fabrication_penalty * 1.0      # Up to -variable
        )
        
        # Normalize to 1-5 range
        # Expected raw range: roughly -10 to 35
        # Map to 1-5
        normalized = 1.0 + (raw_score + 5.0) / 10.0
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception:
        return 2.5