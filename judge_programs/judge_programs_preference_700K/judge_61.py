def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    a sentence-level analysis approach. Analyzes each sentence for its
    epistemic stance (claim type, hedging, evidence attribution, etc.)
    and computes a holistic score based on the distribution of epistemic
    stances across the response.
    
    This variant focuses on:
    1. Sentence-level epistemic classification
    2. Transition patterns between epistemic stances
    3. Source/evidence attribution density
    4. Proportional analysis of claim types
    5. Query ambiguity detection to calibrate expectations
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
        resp_len = len(response)
        
        # ============================================================
        # STEP 1: Split response into sentences for granular analysis
        # ============================================================
        # Custom sentence splitter that handles common abbreviations
        def split_sentences(text):
            # Replace common abbreviations to avoid false splits
            temp = text
            abbrevs = ['dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'e.g.', 'i.e.', 
                       'etc.', 'vs.', 'st.', 'jr.', 'sr.', 'u.s.', 'u.k.']
            for a in abbrevs:
                temp = temp.replace(a, a.replace('.', '<DOT>'))
            
            # Split on sentence-ending punctuation
            parts = re.split(r'(?<=[.!?])\s+', temp)
            # Restore dots
            parts = [p.replace('<DOT>', '.') for p in parts]
            # Filter out very short fragments
            return [p.strip() for p in parts if len(p.strip()) > 5]
        
        sentences = split_sentences(response)
        if not sentences:
            return 1.0
        
        num_sentences = len(sentences)
        
        # ============================================================
        # STEP 2: Classify each sentence's epistemic stance
        # ============================================================
        
        # Categories: 'hedged', 'attributed', 'definitive', 'speculative', 
        #             'meta_epistemic', 'conditional', 'personal', 'neutral'
        
        hedging_phrases = [
            'it seems', 'it appears', 'likely', 'unlikely', 'probably',
            'perhaps', 'possibly', 'might be', 'could be', 'may be',
            'tends to', 'generally', 'typically', 'often', 'sometimes',
            'in many cases', 'it is thought', 'arguably', 'plausibly',
            'roughly', 'approximately', 'around', 'about', 'estimated',
            'not entirely clear', 'not well understood', 'debatable',
            'to some extent', 'in some ways', 'more or less',
            'as far as we know', 'to my knowledge', 'as I understand',
        ]
        
        attribution_phrases = [
            'research suggests', 'studies show', 'according to',
            'evidence indicates', 'data suggests', 'scholars argue',
            'experts believe', 'the literature', 'findings suggest',
            'research indicates', 'has been shown', 'was demonstrated',
            'peer-reviewed', 'published in', 'a study by', 'meta-analysis',
            'systematic review', 'researchers found', 'scientists',
            'historians note', 'as noted by', 'as described by',
            'one theory', 'one view', 'some argue', 'others contend',
            'the consensus', 'mainstream view', 'widely accepted',
        ]
        
        definitive_markers = [
            'always', 'never', 'certainly', 'definitely', 'absolutely',
            'without doubt', 'undoubtedly', 'unquestionably', 'clearly',
            'obviously', 'of course', 'everyone knows', 'it is certain',
            'there is no question', 'beyond dispute', 'indisputably',
            'proven fact', 'undeniable', 'irrefutable', 'guaranteed',
            'without exception', 'in every case', 'no one disputes',
        ]
        
        speculative_markers = [
            'i think', 'i believe', 'in my opinion', 'i would guess',
            'i suspect', 'i imagine', 'i reckon', 'my guess is',
            'speculation', 'speculative', 'hypothetically',
            'one could argue', 'it\'s conceivable', 'theoretically',
            'if we assume', 'assuming that', 'it\'s possible that',
        ]
        
        meta_epistemic_markers = [
            'we don\'t know', 'it\'s unclear', 'remains uncertain',
            'open question', 'not enough evidence', 'hard to say',
            'difficult to determine', 'no consensus', 'debated',
            'controversial', 'disputed', 'uncertain', 'unknown',
            'we can\'t be sure', 'there\'s disagreement', 'ambiguous',
            'complex issue', 'nuanced', 'depends on', 'it varies',
            'more research is needed', 'further study',
        ]
        
        conditional_markers = [
            'if ', 'depending on', 'in the case of', 'when ',
            'assuming', 'provided that', 'unless', 'given that',
            'in some contexts', 'under certain', 'it depends',
            'in certain circumstances', 'contingent on',
        ]
        
        personal_experience_markers = [
            'in my experience', 'i\'ve found', 'i\'ve seen',
            'personally', 'from what i\'ve', 'anecdotally',
            'in my case', 'for me', 'i have', 'i\'ve been',
            'i worked', 'i did', 'i was', 'my own',
        ]
        
        def classify_sentence(sent):
            s = sent.lower()
            scores = {
                'hedged': 0,
                'attributed': 0,
                'definitive': 0,
                'speculative': 0,
                'meta_epistemic': 0,
                'conditional': 0,
                'personal': 0,
            }
            
            for phrase in hedging_phrases:
                if phrase in s:
                    scores['hedged'] += 1
            
            for phrase in attribution_phrases:
                if phrase in s:
                    scores['attributed'] += 1
            
            for phrase in definitive_markers:
                if phrase in s:
                    scores['definitive'] += 1
            
            for phrase in speculative_markers:
                if phrase in s:
                    scores['speculative'] += 1
            
            for phrase in meta_epistemic_markers:
                if phrase in s:
                    scores['meta_epistemic'] += 1
            
            for phrase in conditional_markers:
                if phrase in s:
                    scores['conditional'] += 1
            
            for phrase in personal_experience_markers:
                if phrase in s:
                    scores['personal'] += 1
            
            # Return the dominant category, or 'neutral' if none
            max_score = max(scores.values())
            if max_score == 0:
                return 'neutral'
            
            # Return category with highest score
            return max(scores, key=scores.get)
        
        classifications = [classify_sentence(s) for s in sentences]
        class_counts = Counter(classifications)
        
        # ============================================================
        # STEP 3: Detect query ambiguity/complexity to calibrate expectations
        # ============================================================
        
        ambiguity_signals = [
            'why', 'how', 'what causes', 'what went wrong', 'is there',
            'ethical', 'moral', 'opinion', 'debate', 'controversial',
            'best way', 'should i', 'what do you think', 'argument for',
            'argument against', 'pros and cons', 'impact', 'effect',
            'philosophy', 'theory', 'hypothesis', 'speculate',
            'imagine', 'what if', 'predict', 'future',
        ]
        
        factual_signals = [
            'what is', 'define', 'when did', 'where is', 'who was',
            'how many', 'how much', 'list', 'name the', 'sql',
            'code', 'create table', 'write a', 'calculate',
        ]
        
        ambiguity_count = sum(1 for sig in ambiguity_signals if sig in query_lower)
        factual_count = sum(1 for sig in factual_signals if sig in query_lower)
        
        # Higher = more ambiguous topic where hedging is more appropriate
        query_ambiguity = min(1.0, ambiguity_count * 0.15) - min(0.5, factual_count * 0.1)
        query_ambiguity = max(0.0, min(1.0, query_ambiguity + 0.3))  # baseline ambiguity
        
        # ============================================================
        # STEP 4: Analyze epistemic stance distribution
        # ============================================================
        
        total = max(num_sentences, 1)
        
        hedged_ratio = class_counts.get('hedged', 0) / total
        attributed_ratio = class_counts.get('attributed', 0) / total
        definitive_ratio = class_counts.get('definitive', 0) / total
        speculative_ratio = class_counts.get('speculative', 0) / total
        meta_ratio = class_counts.get('meta_epistemic', 0) / total
        conditional_ratio = class_counts.get('conditional', 0) / total
        personal_ratio = class_counts.get('personal', 0) / total
        neutral_ratio = class_counts.get('neutral', 0) / total
        
        # ============================================================
        # STEP 5: Analyze epistemic transitions (sentence-to-sentence)
        # ============================================================
        
        # Good responses often transition between claim types
        # e.g., fact -> hedge -> attribution -> conditional
        transitions = 0
        if len(classifications) > 1:
            for i in range(len(classifications) - 1):
                if classifications[i] != classifications[i+1]:
                    transitions += 1
        
        transition_rate = transitions / max(len(classifications) - 1, 1)
        
        # ============================================================
        # STEP 6: Compute component scores
        # ============================================================
        
        # Component 1: Epistemic diversity (variety of stance types used)
        unique_stances = len(set(classifications))
        # Shannon entropy of stance distribution
        stance_entropy = 0.0
        for cls in set(classifications):
            p = class_counts[cls] / total
            if p > 0:
                stance_entropy -= p * math.log2(p)
        
        max_entropy = math.log2(8)  # 8 possible categories
        diversity_score = stance_entropy / max_entropy if max_entropy > 0 else 0
        
        # Component 2: Appropriate hedging (scaled by query ambiguity)
        # More ambiguous queries should have more hedging
        calibrated_hedging = hedged_ratio + meta_ratio + conditional_ratio
        hedging_target = 0.1 + query_ambiguity * 0.35  # target hedging ratio
        hedging_deviation = abs(calibrated_hedging - hedging_target)
        hedging_score = max(0, 1.0 - hedging_deviation * 2.0)
        
        # Component 3: Attribution quality
        attribution_score = min(1.0, attributed_ratio * 4.0)
        
        # Component 4: Overconfidence penalty
        # Definitive language is penalized more for ambiguous topics
        overconfidence_penalty = definitive_ratio * (1.0 + query_ambiguity)
        confidence_score = max(0, 1.0 - overconfidence_penalty * 2.5)
        
        # Component 5: Speculative transparency
        # Marking speculation as such is good
        spec_score = min(1.0, speculative_ratio * 3.0) if speculative_ratio > 0 else 0.3
        
        # Component 6: Transition richness
        transition_score = min(1.0, transition_rate * 1.5)
        
        # Component 7: Personal experience framing (moderately good - shows epistemic humility)
        personal_score = min(1.0, personal_ratio * 3.0)
        
        # ============================================================
        # STEP 7: Response substance and engagement quality
        # ============================================================
        
        # Word count as a proxy for substantiveness
        words = response.split()
        word_count = len(words)
        
        # Substantiveness score (diminishing returns)
        substance_score = min(1.0, math.log(max(word_count, 1) + 1) / math.log(200))
        
        # Check for multi-perspective presentation
        perspective_markers = [
            'on the other hand', 'however', 'alternatively', 'conversely',
            'another view', 'some would say', 'others argue', 'in contrast',
            'while', 'whereas', 'but', 'although', 'despite', 'yet',
            'that said', 'at the same time', 'on one hand',
            'from one perspective', 'from another angle',
        ]
        
        perspective_count = sum(1 for m in perspective_markers if m in response_lower)
        perspective_score = min(1.0, perspective_count * 0.2)
        
        # ============================================================
        # STEP 8: Check for explicit uncertainty acknowledgment patterns
        # ============================================================
        
        # Patterns that show good epistemic practice at the discourse level
        good_patterns = [
            r'this is (?:a |an )?(?:complex|nuanced|debated|complicated)',
            r'there (?:are|is) (?:multiple|several|various|different) (?:views|perspectives|opinions|interpretations)',
            r'(?:not|isn\'t) (?:entirely |completely )?(?:clear|certain|settled)',
            r'(?:more|further) research',
            r'the (?:short|quick|simple) answer is.*(?:but|however|though)',
            r'it (?:really )?depends',
            r'(?:broadly|generally|roughly) speaking',
            r'to (?:oversimplify|simplify)',
            r'(?:caveat|disclaimer|note|important to note)',
            r'(?:as far as|to the best of) (?:I|we) know',
        ]
        
        good_pattern_count = 0
        for pattern in good_patterns:
            if re.search(pattern, response_lower):
                good_pattern_count += 1
        
        good_pattern_score = min(1.0, good_pattern_count * 0.25)
        
        # ============================================================
        # STEP 9: Detect bad epistemic practices
        # ============================================================
        
        bad_patterns = [
            r'everyone knows',
            r'it\'s (?:a )?(?:proven )?fact that',
            r'there is no (?:doubt|question)',
            r'(?:any|every) (?:reasonable|rational|intelligent) person',
            r'the (?:truth|reality|fact) is',
            r'(?:simply|just) (?:put|stated)',
            r'(?:obviously|clearly|undeniably),? (?:the|this|it)',
            r'you(?:\'re| are) wrong if',
            r'only (?:an? )?(?:fool|idiot)',
        ]
        
        bad_pattern_count = 0
        for pattern in bad_patterns:
            if re.search(pattern, response_lower):
                bad_pattern_count += 1
        
        bad_pattern_penalty = min(1.0, bad_pattern_count * 0.3)
        
        # ============================================================
        # STEP 10: Compute final weighted score
        # ============================================================
        
        # Weights emphasize different aspects of epistemic calibration
        final_score = (
            diversity_score * 1.5 +        # Epistemic stance variety
            hedging_score * 2.0 +           # Calibrated hedging
            attribution_score * 1.5 +       # Source attribution
            confidence_score * 2.0 +        # Avoiding overconfidence
            spec_score * 0.8 +              # Speculative transparency
            transition_score * 1.0 +        # Stance transitions
            personal_score * 0.5 +          # Personal framing
            substance_score * 2.0 +         # Response substance
            perspective_score * 1.5 +       # Multiple perspectives
            good_pattern_score * 1.5 -      # Good epistemic patterns
            bad_pattern_penalty * 2.0       # Bad epistemic patterns
        )
        
        # Normalize to 0-10 range
        max_possible = (1.5 + 2.0 + 1.5 + 2.0 + 0.8 + 1.0 + 0.5 + 2.0 + 1.5 + 1.5)
        # = 14.3
        
        normalized = (final_score / max_possible) * 10.0
        
        # Clamp to [0, 10]
        normalized = max(0.0, min(10.0, normalized))
        
        return round(normalized, 3)
        
    except Exception:
        return 3.0