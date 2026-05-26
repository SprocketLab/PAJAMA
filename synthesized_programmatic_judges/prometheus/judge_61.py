def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    a discourse-structure and pragmatic-analysis approach.
    
    This variant focuses on:
    1. Pragmatic speech act classification (assertions vs suggestions vs acknowledgments)
    2. Epistemic stance markers at clause level
    3. Source attribution and evidence framing
    4. Conditional/hypothetical reasoning structures
    5. Perspective-taking and empathy markers
    6. Response appropriateness relative to query ambiguity
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
        resp_sentences = re.split(r'[.!?]+', response)
        resp_sentences = [s.strip() for s in resp_sentences if s.strip()]
        num_sentences = max(len(resp_sentences), 1)
        
        words = re.findall(r'\b\w+\b', resp_lower)
        num_words = max(len(words), 1)
        
        # ============================================================
        # 1. SPEECH ACT CLASSIFICATION
        # Classify sentences into speech acts and score balance
        # ============================================================
        
        # Suggestion/advisory markers
        suggestion_patterns = [
            r'\byou (?:could|might|may|can)\b', r'\bconsider\b', r'\btry to\b',
            r'\bit (?:might|may|could) (?:help|be)\b', r'\bone (?:approach|option|way)\b',
            r'\byou (?:may want|might want)\b', r'\bit\'?s? worth\b',
            r'\ba good (?:idea|approach|strategy)\b', r'\bperhaps you\b',
        ]
        
        # Acknowledgment/empathy markers
        acknowledgment_patterns = [
            r'\bi (?:understand|see|hear|recognize|appreciate)\b',
            r'\bthat\'?s? (?:completely |totally |absolutely )?(?:understandable|normal|natural|okay|valid)\b',
            r'\bit\'?s? (?:completely |totally |absolutely )?(?:understandable|normal|natural|okay|fine|valid)\b',
            r'\bi\'?m? (?:sorry|glad)\b', r'\bI can (?:see|hear|imagine|understand)\b',
            r'\byour (?:feelings?|concerns?|frustration|experience)\b',
        ]
        
        # Bare assertion markers (stating things as absolute fact)
        bare_assertion_patterns = [
            r'\byou (?:need|must|should|have to|will|are going to)\b',
            r'\bjust (?:do|make|get|go|try)\b',
            r'\b(?:always|never|definitely|certainly|obviously|clearly|undoubtedly)\b',
            r'\bthe (?:only|best|right|correct) (?:way|answer|solution|approach)\b',
            r'\bwithout (?:a )?doubt\b', r'\bthere\'?s? no (?:question|doubt)\b',
        ]
        
        suggestion_count = sum(len(re.findall(p, resp_lower)) for p in suggestion_patterns)
        acknowledgment_count = sum(len(re.findall(p, resp_lower)) for p in acknowledgment_patterns)
        bare_assertion_count = sum(len(re.findall(p, resp_lower)) for p in bare_assertion_patterns)
        
        # Score: suggestions and acknowledgments are good; bare assertions penalized
        speech_act_score = 0.0
        suggestion_ratio = suggestion_count / num_sentences
        acknowledgment_ratio = acknowledgment_count / num_sentences
        bare_assertion_ratio = bare_assertion_count / num_sentences
        
        speech_act_score += min(suggestion_ratio * 3.0, 2.5)
        speech_act_score += min(acknowledgment_ratio * 3.0, 2.5)
        speech_act_score -= min(bare_assertion_ratio * 2.5, 3.0)
        
        # ============================================================
        # 2. EPISTEMIC STANCE AT CLAUSE LEVEL
        # Look for epistemic markers embedded within clauses
        # ============================================================
        
        # Evidential markers (sourcing knowledge)
        evidential_patterns = [
            r'\bresearch (?:suggests?|shows?|indicates?|has shown)\b',
            r'\bstudies (?:suggest|show|indicate|have shown)\b',
            r'\baccording to\b', r'\bevidence (?:suggests?|shows?|indicates?)\b',
            r'\bit has been (?:shown|found|observed|noted)\b',
            r'\bexperts (?:suggest|recommend|believe|say)\b',
            r'\bgenerally (?:speaking|accepted|considered)\b',
            r'\bin (?:many|most|some) cases\b',
        ]
        
        # Conditional/hypothetical structures
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\b(?:might|could|may|would)\b',
            r'\bdepending on\b', r'\bin case\b', r'\bwhen\b.*\b(?:might|could|may)\b',
            r'\bassuming\b', r'\bprovided that\b', r'\bunless\b',
            r'\bit depends\b', r'\bthis (?:depends|varies)\b',
        ]
        
        # Approximation and qualification
        qualification_patterns = [
            r'\bapproximately\b', r'\babout\b', r'\broughly\b',
            r'\btypically\b', r'\busually\b', r'\bgenerally\b',
            r'\boften\b', r'\bsometimes\b', r'\btend(?:s)? to\b',
            r'\bin general\b', r'\bfor the most part\b',
            r'\bto some (?:extent|degree)\b', r'\brelatively\b',
            r'\bsomewhat\b', r'\bpartially\b', r'\blargely\b',
        ]
        
        evidential_count = sum(len(re.findall(p, resp_lower)) for p in evidential_patterns)
        conditional_count = sum(len(re.findall(p, resp_lower)) for p in conditional_patterns)
        qualification_count = sum(len(re.findall(p, resp_lower)) for p in qualification_patterns)
        
        epistemic_score = 0.0
        epistemic_score += min(evidential_count * 0.8, 2.0)
        epistemic_score += min(conditional_count * 0.5, 1.5)
        epistemic_score += min(qualification_count / num_sentences * 2.0, 1.5)
        
        # ============================================================
        # 3. PERSPECTIVE-TAKING AND ENGAGEMENT DEPTH
        # ============================================================
        
        # Second-person engagement (addressing the user directly and thoughtfully)
        second_person_refs = len(re.findall(r'\byou(?:r|\'re|\'ve|\'ll)?\b', resp_lower))
        second_person_ratio = second_person_refs / num_words
        
        # First-person inclusive (we, our, let's) - collaborative framing
        inclusive_refs = len(re.findall(r'\b(?:we|our|let\'?s)\b', resp_lower))
        inclusive_ratio = inclusive_refs / num_words
        
        # Emotional validation phrases
        validation_phrases = [
            r'\bit\'?s? (?:okay|ok|alright|fine|natural|normal|valid) to (?:feel|be)\b',
            r'\bgive yourself (?:permission|time|space)\b',
            r'\b(?:completely|totally|absolutely|perfectly) (?:understandable|normal|natural|fine|okay|valid)\b',
            r'\byour feelings? (?:are|is) (?:valid|important|understandable)\b',
            r'\bdon\'?t be (?:afraid|shy|hesitant)\b',
        ]
        validation_count = sum(len(re.findall(p, resp_lower)) for p in validation_phrases)
        
        engagement_score = 0.0
        engagement_score += min(second_person_ratio * 15, 1.5)
        engagement_score += min(inclusive_ratio * 30, 1.0)
        engagement_score += min(validation_count * 0.5, 1.5)
        
        # ============================================================
        # 4. STRUCTURAL SOPHISTICATION
        # (Different from bullet detection - looking at discourse coherence)
        # ============================================================
        
        # Sentence variety (mix of short and long sentences indicates thoughtful composition)
        sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in resp_sentences]
        if len(sent_lengths) > 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variance is good (not all same length, not wildly different)
            cv = std_dev / max(mean_len, 1)
            if 0.2 <= cv <= 0.8:
                structure_variety = 1.0
            elif cv < 0.2:
                structure_variety = cv / 0.2 * 0.5
            else:
                structure_variety = max(0, 1.0 - (cv - 0.8) * 0.5)
        else:
            structure_variety = 0.3
        
        # Logical connectors (cause-effect, contrast, elaboration)
        causal_connectors = len(re.findall(
            r'\b(?:because|since|therefore|thus|hence|consequently|as a result|due to|so that|in order to)\b',
            resp_lower
        ))
        contrast_connectors = len(re.findall(
            r'\b(?:however|but|although|though|nevertheless|on the other hand|despite|yet|while|whereas)\b',
            resp_lower
        ))
        elaboration_connectors = len(re.findall(
            r'\b(?:for (?:instance|example)|such as|specifically|in particular|namely|that is|in other words)\b',
            resp_lower
        ))
        
        connector_total = causal_connectors + contrast_connectors + elaboration_connectors
        connector_ratio = connector_total / num_sentences
        
        structure_score = 0.0
        structure_score += structure_variety * 1.0
        structure_score += min(connector_ratio * 1.5, 1.5)
        # Bonus for having diverse connector types
        connector_types_present = sum(1 for c in [causal_connectors, contrast_connectors, elaboration_connectors] if c > 0)
        structure_score += connector_types_present * 0.3
        
        # ============================================================
        # 5. QUERY-RESPONSE ALIGNMENT (Pragmatic Appropriateness)
        # ============================================================
        
        # Detect query characteristics
        query_has_emotion = bool(re.search(
            r'\b(?:feeling|feel|emotion|stress|frustrat|sad|happy|angry|upset|devastat|heartbrok|loneli|despair|exhaust|struggling|difficult)\b',
            query_lower
        ))
        query_is_ambiguous = bool(re.search(
            r'\b(?:ambiguous|unclear|no (?:previous )?context|without (?:further )?details?)\b',
            query_lower
        ))
        query_seeks_help = bool(re.search(
            r'\b(?:seeking|need|want|looking for|how (?:to|can|do|would)|assist|advice|help|guide|explain)\b',
            query_lower
        ))
        query_is_technical = bool(re.search(
            r'\b(?:technical|code|programming|algorithm|system|model|design|implement|software|compute|data)\b',
            query_lower
        ))
        
        alignment_score = 0.0
        
        # If query is emotional, response should have empathy
        if query_has_emotion:
            empathy_markers = len(re.findall(
                r'\b(?:understand|sorry|hear|feel|natural|okay|valid|normal|support|care|comfort)\b',
                resp_lower
            ))
            if empathy_markers >= 2:
                alignment_score += 1.5
            elif empathy_markers >= 1:
                alignment_score += 0.7
            else:
                alignment_score -= 1.0  # Penalty for ignoring emotional context
        
        # If query seeks help, response should provide actionable content
        if query_seeks_help:
            actionable_markers = len(re.findall(
                r'\b(?:step|first|then|next|start|begin|try|consider|approach|method|way|tip|strateg)\b',
                resp_lower
            ))
            if actionable_markers >= 3:
                alignment_score += 1.0
            elif actionable_markers >= 1:
                alignment_score += 0.5
        
        # Response length appropriateness
        if num_words < 20:
            alignment_score -= 1.5  # Too short
        elif num_words < 50:
            alignment_score -= 0.5
        elif num_words > 300:
            alignment_score += 0.2  # Detailed responses slightly rewarded
        
        # ============================================================
        # 6. DISMISSIVENESS AND NEGATIVITY DETECTION
        # ============================================================
        
        dismissive_patterns = [
            r'\bjust (?:do|get|go|buy|read|try)\b',
            r'\bit\'?s? (?:just|only) a\b',
            r'\bget over it\b', r'\bmove on\b',
            r'\bstop (?:being|feeling|worrying)\b',
            r'\byou\'?re (?:just|probably) not\b',
            r'\bmaybe you\'?re (?:just|not)\b',
            r'\bthat\'?s? (?:just|how) (?:life|it) (?:is|works)\b',
        ]
        dismissive_count = sum(len(re.findall(p, resp_lower)) for p in dismissive_patterns)
        
        # Imperative/commanding tone (without softening)
        imperative_patterns = re.findall(
            r'(?:^|\. )(?:Do|Get|Make|Stop|Go|Read|Try|Keep|Remember|Don\'t)\b',
            response
        )
        imperative_count = len(imperative_patterns)
        
        negativity_score = 0.0
        negativity_score -= min(dismissive_count * 0.7, 2.5)
        # Imperatives are okay if balanced with empathy
        if acknowledgment_count == 0 and imperative_count > 2:
            negativity_score -= min(imperative_count * 0.3, 1.5)
        
        # ============================================================
        # 7. OVERCONFIDENCE PENALTY (specific to epistemic calibration)
        # ============================================================
        
        # Absolute/universal claims
        absolute_terms = len(re.findall(
            r'\b(?:always|never|every|all|none|no one|everyone|everything|nothing|impossible|guaranteed|certain(?:ly)?|undoubtedly|without (?:a )?doubt|absolutely|100%)\b',
            resp_lower
        ))
        
        # Negation of uncertainty where uncertainty would be appropriate
        false_certainty = len(re.findall(
            r'\b(?:there is no|there\'s no|it is not|it\'s not|cannot be|won\'t|will not) (?:question|doubt|way|chance|possibility)\b',
            resp_lower
        ))
        
        # "Probably not" or hedged negatives show good calibration
        hedged_negatives = len(re.findall(
            r'\b(?:probably not|might not|may not|unlikely|less likely|not necessarily|not always)\b',
            resp_lower
        ))
        
        overconfidence_penalty = 0.0
        overconfidence_penalty -= min(absolute_terms / num_sentences * 2.0, 2.0)
        overconfidence_penalty -= min(false_certainty * 0.8, 1.5)
        overconfidence_penalty += min(hedged_negatives * 0.4, 1.0)
        
        # ============================================================
        # 8. MODAL VERB DIVERSITY (epistemic modality spectrum)
        # ============================================================
        
        modal_categories = {
            'possibility': len(re.findall(r'\b(?:might|may|could|can)\b', resp_lower)),
            'probability': len(re.findall(r'\b(?:likely|probably|perhaps|possibly)\b', resp_lower)),
            'necessity': len(re.findall(r'\b(?:must|need to|have to|should|ought)\b', resp_lower)),
            'volition': len(re.findall(r'\b(?:would|will|want to|wish)\b', resp_lower)),
        }
        
        modal_types_used = sum(1 for v in modal_categories.values() if v > 0)
        total_modals = sum(modal_categories.values())
        
        # Diversity of modal usage indicates nuanced epistemic stance
        modal_score = 0.0
        if modal_types_used >= 3:
            modal_score += 1.5
        elif modal_types_used >= 2:
            modal_score += 0.8
        elif modal_types_used >= 1:
            modal_score += 0.3
        
        # Possibility modals are especially good for calibration
        possibility_ratio = modal_categories['possibility'] / max(total_modals, 1)
        if total_modals > 0 and possibility_ratio >= 0.2:
            modal_score += 0.5
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        # Weights reflect importance for epistemic calibration
        total_score = (
            speech_act_score * 1.2 +      # max ~6
            epistemic_score * 1.3 +         # max ~6.5
            engagement_score * 1.0 +        # max ~4
            structure_score * 0.8 +         # max ~2.4
            alignment_score * 1.0 +         # max ~2.5
            negativity_score * 1.0 +        # max 0, min -4
            overconfidence_penalty * 1.0 +  # max ~1, min -3.5
            modal_score * 0.8              # max ~1.6
        )
        
        # Normalize to 0-10 scale
        # Theoretical range roughly: -7 to +23
        # Map to 0-10
        normalized = (total_score + 7) / 30 * 10
        
        # Clamp
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0