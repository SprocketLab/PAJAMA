def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant uses a novel approach based on:
    1. Claim density analysis (ratio of assertive claims to total content)
    2. Evidential reasoning markers (references to evidence, reasons, examples)
    3. Perspective-taking and acknowledgment patterns
    4. Conditional/nuanced language structures
    5. Dogmatic vs. tentative framing ratio
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        from collections import Counter
        
        resp_lower = response.lower()
        words = re.findall(r'\b[a-z]+\b', resp_lower)
        word_count = len(words)
        
        if word_count < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Conditional and nuanced language structures
        # Look for if-then constructions, conditional mood, nuanced framing
        # ============================================================
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\byou\b', r'\bif\b.*\bmight\b',
            r'\bdepending on\b', r'\bin case\b', r'\bassuming\b',
            r'\bprovided that\b', r'\bwhether\b', r'\bgiven that\b',
            r'\bunder certain\b', r'\bin some cases\b', r'\bin certain\b',
            r'\bwhen\b.*\bconsider\b', r'\bit depends\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bcontext\b', r'\bcircumstances\b',
        ]
        conditional_count = 0
        for pat in conditional_patterns:
            conditional_count += len(re.findall(pat, resp_lower))
        conditional_score = min(conditional_count / num_sentences * 3.0, 5.0)
        
        # ============================================================
        # FEATURE 2: Evidential reasoning markers
        # References to evidence, examples, reasons - shows grounded claims
        # ============================================================
        evidential_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bresearch\b', r'\bstudies\b', r'\bevidence\b',
            r'\baccording to\b', r'\bdata\b', r'\bfindings\b',
            r'\bbecause\b', r'\bdue to\b', r'\bas a result\b',
            r'\bthis means\b', r'\bthis is because\b', r'\bthe reason\b',
            r'\bin practice\b', r'\bin theory\b', r'\btypically\b',
            r'\bgenerally\b', r'\busually\b', r'\boften\b',
            r'\bcommonly\b', r'\bfrequently\b',
        ]
        evidential_count = 0
        for pat in evidential_markers:
            evidential_count += len(re.findall(pat, resp_lower))
        evidential_score = min(evidential_count / num_sentences * 2.5, 5.0)
        
        # ============================================================
        # FEATURE 3: Dogmatic assertion detection (PENALTY)
        # Absolute/universal claims without qualification
        # ============================================================
        dogmatic_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b',
            r'\babsolutely\b', r'\bwithout a doubt\b', r'\bcertainly\b',
            r'\bobviously\b', r'\bclearly\b', r'\bundoubtedly\b',
            r'\bno question\b', r'\bwithout question\b',
            r'\beveryone knows\b', r'\bit is a fact\b',
            r'\bthe truth is\b', r'\bthe fact is\b',
            r'\byou must\b', r'\byou need to\b', r'\byou should\b',
            r'\bjust\b.*\bdo\b', r'\bsimply\b',
        ]
        dogmatic_count = 0
        for pat in dogmatic_patterns:
            dogmatic_count += len(re.findall(pat, resp_lower))
        
        # Normalize dogmatic count by sentence count
        dogmatic_ratio = dogmatic_count / num_sentences
        dogmatic_penalty = min(dogmatic_ratio * 2.5, 4.0)
        
        # ============================================================
        # FEATURE 4: Acknowledgment and empathy patterns
        # Shows awareness of complexity, other perspectives, emotions
        # ============================================================
        acknowledgment_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bthat\'s understandable\b', r'\bunderstandable\b',
            r'\bit\'s okay\b', r'\bit\'s natural\b', r'\bit\'s normal\b',
            r'\bcompletely\b.*\bunderstandable\b', r'\bperfectly\b.*\bfine\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\bthat said\b',
            r'\bwhile\b.*\balso\b', r'\bboth\b.*\band\b',
            r'\bsome people\b', r'\bsome may\b', r'\bothers might\b',
            r'\bit\'s worth noting\b', r'\bkeep in mind\b',
            r'\bimportant to\b', r'\bworth considering\b',
            r'\bi\'m sorry\b', r'\bsorry to hear\b',
        ]
        ack_count = 0
        for pat in acknowledgment_patterns:
            ack_count += len(re.findall(pat, resp_lower))
        ack_score = min(ack_count / num_sentences * 3.0, 5.0)
        
        # ============================================================
        # FEATURE 5: Epistemic verb usage
        # Verbs that signal reasoning vs. bare assertion
        # ============================================================
        epistemic_verbs = [
            r'\bseem[s]?\b', r'\bappear[s]?\b', r'\bsuggest[s]?\b',
            r'\bindicate[s]?\b', r'\bimpl(?:y|ies)\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bwould\b',
            r'\bcan\b', r'\btend[s]?\b', r'\bbelieve\b',
            r'\bconsider\b', r'\bthink\b', r'\bfeel\b',
            r'\bpossib(?:le|ly|ility)\b', r'\blikely\b', r'\bunlikely\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bprobab(?:le|ly)\b',
        ]
        epistemic_count = 0
        for pat in epistemic_verbs:
            epistemic_count += len(re.findall(pat, resp_lower))
        epistemic_ratio = epistemic_count / word_count if word_count > 0 else 0
        epistemic_score = min(epistemic_ratio * 80, 5.0)
        
        # ============================================================
        # FEATURE 6: Sentence-level claim analysis
        # Ratio of sentences that contain qualifiers vs bare assertions
        # ============================================================
        qualifier_set = {
            'may', 'might', 'could', 'would', 'can', 'likely', 'possibly',
            'perhaps', 'maybe', 'probably', 'sometimes', 'often', 'usually',
            'generally', 'typically', 'tends', 'seems', 'appears', 'suggests',
            'consider', 'potentially', 'arguably'
        }
        qualified_sentences = 0
        for sent in sentences:
            sent_words = set(re.findall(r'\b[a-z]+\b', sent.lower()))
            if sent_words & qualifier_set:
                qualified_sentences += 1
        
        qualification_ratio = qualified_sentences / num_sentences
        qualification_score = qualification_ratio * 4.0
        
        # ============================================================
        # FEATURE 7: Dismissive language detection (PENALTY)
        # Patterns that dismiss concerns or oversimplify
        # ============================================================
        dismissive_patterns = [
            r'\bjust\b', r'\bsimply\b', r'\beasy\b', r'\bno big deal\b',
            r'\bget over\b', r'\bstop\b.*\bworrying\b',
            r'\bdon\'t worry\b', r'\bit\'s nothing\b',
            r'\bwhatever\b', r'\banyway\b',
            r'\byou\'re wrong\b', r'\bthat\'s wrong\b',
            r'\bprobably won\'t\b', r'\bcan\'t\b.*\bdo\b',
            r'\bmight not\b.*\bable\b',
        ]
        dismissive_count = 0
        for pat in dismissive_patterns:
            dismissive_count += len(re.findall(pat, resp_lower))
        dismissive_penalty = min(dismissive_count / num_sentences * 1.5, 3.0)
        
        # ============================================================
        # FEATURE 8: Structured reasoning indicators
        # Numbered steps, cause-effect chains, organized thought
        # ============================================================
        structure_patterns = [
            r'\d+[\.\)]\s', r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b',
            r'\bthird(?:ly)?\b', r'\bnext\b', r'\bfinally\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b',
            r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
            r'\bas a result\b', r'\bhence\b',
            r'\bon one hand\b', r'\bon the other\b',
        ]
        structure_count = 0
        for pat in structure_patterns:
            structure_count += len(re.findall(pat, resp_lower))
        structure_score = min(structure_count / num_sentences * 2.0, 3.0)
        
        # ============================================================
        # FEATURE 9: Response engagement with query
        # Does the response address the query's concerns?
        # ============================================================
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        resp_words = set(re.findall(r'\b[a-z]{4,}\b', resp_lower))
        
        if query_words:
            relevance = len(query_words & resp_words) / len(query_words)
        else:
            relevance = 0.5
        relevance_score = relevance * 3.0
        
        # ============================================================
        # FEATURE 10: Depth and elaboration
        # Sufficient content to be helpful, but not rambling
        # ============================================================
        # Optimal range: moderate length with substance
        if word_count < 20:
            depth_score = 0.5
        elif word_count < 50:
            depth_score = 1.5
        elif word_count < 100:
            depth_score = 2.5
        elif word_count < 250:
            depth_score = 3.0
        else:
            depth_score = 2.5  # Slight reduction for very long
        
        # Unique word ratio as proxy for informational density
        if word_count > 0:
            unique_ratio = len(set(words)) / word_count
            # Moderate diversity is good (0.4-0.7 range)
            if 0.4 <= unique_ratio <= 0.75:
                depth_score += 1.0
            elif unique_ratio > 0.75:
                depth_score += 0.5
        
        # ============================================================
        # FEATURE 11: Tone appropriateness
        # Detect if query is emotional and response matches
        # ============================================================
        emotional_query_markers = [
            'feel', 'feeling', 'sad', 'frustrated', 'angry', 'upset',
            'worried', 'anxious', 'stress', 'struggling', 'difficult',
            'hard', 'lonely', 'heartbroken', 'devastated', 'exhausted',
            'overwhelmed', 'depressed', 'scared', 'afraid', 'fear'
        ]
        query_is_emotional = sum(1 for m in emotional_query_markers if m in query_lower) >= 2
        
        emotional_response_markers = [
            r'\bunderstand\b', r'\bhear you\b', r'\bvalid\b',
            r'\bnatural\b', r'\bnormal\b', r'\bokay\b',
            r'\bsorry\b', r'\bempathize\b', r'\bcompassion\b',
            r'\bfeel\b', r'\bfeelings?\b', r'\bemotion\b',
            r'\bgriev\b', r'\bprocess\b', r'\bheal\b',
        ]
        emotional_response_count = 0
        for pat in emotional_response_markers:
            emotional_response_count += len(re.findall(pat, resp_lower))
        
        tone_score = 0.0
        if query_is_emotional:
            if emotional_response_count >= 3:
                tone_score = 3.0
            elif emotional_response_count >= 1:
                tone_score = 1.5
            else:
                tone_score = -1.0  # Penalty for ignoring emotional context
        else:
            tone_score = 1.0  # Neutral baseline for non-emotional queries
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        raw_score = (
            conditional_score * 1.0 +      # max ~5
            evidential_score * 1.0 +        # max ~5
            epistemic_score * 1.2 +          # max ~6
            qualification_score * 1.0 +      # max ~4
            ack_score * 1.0 +                # max ~5
            structure_score * 0.8 +          # max ~2.4
            relevance_score * 1.0 +          # max ~3
            depth_score * 0.7 +              # max ~2.8
            tone_score * 1.0 +               # max ~3
            - dogmatic_penalty * 1.2 +       # penalty
            - dismissive_penalty * 1.0       # penalty
        )
        
        # Normalize to 1-5 scale
        # Theoretical max is around 36, typical good response ~15-25
        # Typical bad response ~3-10
        min_expected = -2.0
        max_expected = 30.0
        
        normalized = (raw_score - min_expected) / (max_expected - min_expected)
        normalized = max(0.0, min(1.0, normalized))
        
        final_score = 1.0 + normalized * 4.0  # Scale to 1-5
        
        # Round to 1 decimal
        final_score = round(final_score, 1)
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return final_score
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5