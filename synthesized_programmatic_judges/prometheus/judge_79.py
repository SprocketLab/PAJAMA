def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    
    This variant analyzes responses at the sentence level, scoring each sentence for
    information density, then aggregates. It also uses a novel approach of measuring
    "assertion strength" - how confidently and specifically claims are made vs hedged.
    
    Different from other variants by focusing on:
    1. Sentence-level information density scoring
    2. Assertion strength analysis (confident specific claims vs weak hedges)
    3. Discourse structure quality (progression, coherence markers)
    4. Specificity gradient (measuring how specific vs generic each phrase is)
    5. Engagement with query terms through paraphrase/elaboration detection
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        word_count = len(words)
        if word_count < 5:
            return 0.5
        
        # ============================================================
        # 1. SENTENCE-LEVEL INFORMATION DENSITY
        # Score each sentence for how much concrete information it carries
        # ============================================================
        
        def sentence_info_density(sent):
            """Score a single sentence for information density 0-1."""
            s_words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
            if len(s_words) < 2:
                return 0.0
            
            score = 0.0
            
            # Numbers and quantities in sentence
            nums = re.findall(r'\b\d+[\d.,]*\b', sent)
            score += min(len(nums) * 0.15, 0.4)
            
            # Named entities (capitalized words not at sentence start)
            # Check words after the first word that are capitalized
            cap_words = re.findall(r'(?<!^)(?<!\. )(?<!\? )(?<!\! )\b[A-Z][a-z]{2,}\b', sent)
            score += min(len(cap_words) * 0.1, 0.3)
            
            # Technical/specific vocabulary (longer words tend to be more specific)
            long_specific = [w for w in s_words if len(w) >= 8]
            specificity_ratio = len(long_specific) / max(len(s_words), 1)
            score += specificity_ratio * 0.3
            
            # Action verbs and concrete language
            action_indicators = {'create', 'build', 'implement', 'design', 'develop', 'use',
                                'apply', 'measure', 'calculate', 'analyze', 'detect', 'store',
                                'maintain', 'ensure', 'provide', 'demonstrate', 'explore',
                                'recognize', 'handle', 'resume', 'break', 'tackle', 'start',
                                'follow', 'turn', 'add', 'remove', 'heat', 'cook', 'brown',
                                'grab', 'pour', 'mix', 'stir', 'place', 'set', 'connect',
                                'contact', 'reach', 'identify', 'specify', 'describe'}
            action_count = sum(1 for w in s_words if w in action_indicators)
            score += min(action_count * 0.08, 0.25)
            
            # Penalize very short uninformative sentences
            if len(s_words) < 5:
                score *= 0.5
            
            return min(score, 1.0)
        
        sentence_scores = [sentence_info_density(s) for s in sentences]
        avg_sentence_density = sum(sentence_scores) / max(len(sentence_scores), 1)
        
        # Also compute variance - consistent density is better than sporadic
        if len(sentence_scores) > 1:
            mean_sd = avg_sentence_density
            variance = sum((s - mean_sd) ** 2 for s in sentence_scores) / len(sentence_scores)
            consistency_bonus = max(0, 0.1 - variance) * 2  # up to 0.2
        else:
            consistency_bonus = 0.0
        
        # ============================================================
        # 2. ASSERTION STRENGTH ANALYSIS
        # Measure confident, specific assertions vs weak hedges
        # ============================================================
        
        # Strong assertion patterns (confident, specific claims)
        strong_patterns = [
            r'\bthis (?:means|ensures|allows|enables|creates|provides|results)\b',
            r'\bby (?:using|applying|implementing|creating|breaking|detecting)\b',
            r'\b(?:specifically|precisely|exactly|directly|explicitly)\b',
            r'\bfor (?:example|instance)\b',
            r'\bsuch as\b',
            r'\b(?:first|second|third|finally|next|then)\b',
            r'\bhere(?:\'s| is| are)\b',
            r'\b(?:because|since|therefore|thus|consequently|as a result)\b',
            r'\b(?:ensure|guarantee|require|must|need to|should)\b',
            r'\b(?:key|critical|essential|important|crucial|fundamental)\b',
            r'\bremember (?:to|that)\b',
            r'\b(?:imagine|consider|think of|picture)\b',  # concrete analogies
        ]
        
        strong_count = 0
        response_lower = response_clean.lower()
        for pat in strong_patterns:
            strong_count += len(re.findall(pat, response_lower))
        
        # Weak/hedge patterns (vague, non-committal)
        weak_patterns = [
            r'\b(?:maybe|perhaps|possibly|probably|might|could be)\b',
            r'\bit depends\b',
            r'\bthere are (?:many|various|several|different|some) (?:ways|factors|things|reasons|aspects)\b',
            r'\bmany people\b',
            r'\bin general\b',
            r'\bit\'?s? (?:kind of|sort of|like)\b',
            r'\bjust (?:try|do|keep|get|be)\b',
            r'\byou (?:could|might|may) (?:try|want|consider)\b',
            r'\bor something\b',
            r'\band stuff\b',
            r'\bwhatever\b',
            r'\bi guess\b',
            r'\bwho knows\b',
            r'\bnot sure\b',
            r'\b(?:somehow|somewhere|sometime|someone|something)\b',
            r'\bprobably won\'?t\b',
            r'\bmight not\b',
        ]
        
        weak_count = 0
        for pat in weak_patterns:
            weak_count += len(re.findall(pat, response_lower))
        
        # Assertion strength ratio
        total_assertions = strong_count + weak_count
        if total_assertions > 0:
            assertion_strength = (strong_count - weak_count * 1.5) / (total_assertions + 5)
        else:
            assertion_strength = 0.0
        
        assertion_score = max(0, min(1.0, 0.5 + assertion_strength))
        
        # ============================================================
        # 3. DISCOURSE STRUCTURE QUALITY
        # How well-organized and progressive is the response?
        # ============================================================
        
        structure_score = 0.0
        
        # Enumerated/numbered steps or points
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean)
        if len(numbered_items) >= 2:
            structure_score += 0.3
        
        # Transition/progression words
        transitions = re.findall(
            r'\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|additionally|moreover|furthermore|'
            r'however|in addition|on the other hand|next|then|after that|once|meanwhile|'
            r'in contrast|similarly|as a result|to begin|to start|lastly|also)\b',
            response_lower
        )
        structure_score += min(len(transitions) * 0.05, 0.3)
        
        # Paragraph breaks (indicates organized thought)
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            structure_score += 0.15
        
        # Colon usage (often introduces specific details/lists)
        colon_count = response_clean.count(':')
        structure_score += min(colon_count * 0.05, 0.15)
        
        structure_score = min(structure_score, 1.0)
        
        # ============================================================
        # 4. SPECIFICITY GRADIENT
        # Measure the ratio of specific vs generic words/phrases
        # ============================================================
        
        # Generic filler words that add no information
        generic_fillers = {
            'things', 'stuff', 'something', 'anything', 'everything',
            'various', 'many', 'some', 'several', 'different', 'certain',
            'good', 'bad', 'nice', 'great', 'fine', 'okay', 'ok',
            'really', 'very', 'quite', 'pretty', 'rather', 'somewhat',
            'basically', 'essentially', 'generally', 'usually', 'normally',
            'kind', 'sort', 'type', 'like', 'just', 'actually',
            'obviously', 'clearly', 'definitely', 'absolutely',
        }
        
        # Count content words (not stopwords, not fillers)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'these', 'those', 'am', 'about', 'up', 'down',
        }
        
        content_words = [w for w in words if w not in stopwords]
        filler_count = sum(1 for w in content_words if w in generic_fillers)
        
        if len(content_words) > 0:
            filler_ratio = filler_count / len(content_words)
            specificity_score = max(0, 1.0 - filler_ratio * 3)
        else:
            specificity_score = 0.3
        
        # Bonus for domain-specific or technical vocabulary
        # Words with certain suffixes tend to be more technical
        technical_suffixes = ('tion', 'sion', 'ment', 'ness', 'ity', 'ical', 'ious',
                             'ance', 'ence', 'ive', 'ful', 'able', 'ible', 'ology',
                             'ism', 'ist', 'ize', 'ise', 'ify')
        technical_words = [w for w in content_words if len(w) > 5 and w.endswith(technical_suffixes)]
        tech_ratio = len(technical_words) / max(len(content_words), 1)
        specificity_score += min(tech_ratio * 1.5, 0.3)
        
        specificity_score = min(specificity_score, 1.0)
        
        # ============================================================
        # 5. QUERY ENGAGEMENT DEPTH
        # Does the response deeply engage with query terms or just superficially?
        # ============================================================
        
        query_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        query_words -= stopwords
        
        if query_words:
            # Direct mentions
            response_word_set = set(words)
            direct_mentions = len(query_words & response_word_set)
            mention_ratio = direct_mentions / max(len(query_words), 1)
            
            # Check if query concepts are elaborated on (appear in longer phrases/explanations)
            elaboration_score = 0.0
            for qw in query_words:
                # Find if the query word appears in a context with additional specific info
                pattern = r'\b' + re.escape(qw) + r'\b'
                matches = list(re.finditer(pattern, response_lower))
                for m in matches:
                    # Get surrounding context (50 chars each side)
                    start = max(0, m.start() - 50)
                    end = min(len(response_lower), m.end() + 50)
                    context = response_lower[start:end]
                    # Check if context has numbers, specific terms
                    if re.search(r'\d', context):
                        elaboration_score += 0.1
                    context_words = re.findall(r'\b[a-zA-Z]{5,}\b', context)
                    if len(context_words) >= 4:
                        elaboration_score += 0.05
            
            engagement_score = min(1.0, mention_ratio * 0.5 + min(elaboration_score, 0.5))
        else:
            engagement_score = 0.5
        
        # ============================================================
        # 6. EMPATHY AND ACKNOWLEDGMENT QUALITY (for emotional queries)
        # ============================================================
        
        emotional_query_indicators = re.findall(
            r'\b(?:feel|feeling|emotion|stress|frustrat|sad|lonely|heartbroken|'
            r'devastat|exhaust|anxious|worried|concern|comfort|support|struggling|'
            r'difficult|tough|hard time|upset|angry|afraid|fear|regret|pain)\b',
            query.lower()
        )
        
        empathy_score = 0.5  # neutral default
        if len(emotional_query_indicators) >= 2:
            # This is an emotional query - check for quality empathetic response
            empathy_patterns = [
                r'\bi (?:can |)(?:hear|see|understand|sense|feel)\b',
                r'\bit\'?s? (?:completely |totally |absolutely |perfectly )?(?:okay|ok|understandable|natural|normal|valid|fine)\b',
                r'\byour (?:feelings?|emotions?|experience|pain|frustration|concern)\b',
                r'\b(?:acknowledge|validate|recognize|appreciate)\b',
                r'\b(?:genuinely|sincerely|truly|deeply)\b',
                r'\blet (?:yourself|me)\b',
                r'\btake (?:a moment|some time|a breath)\b',
            ]
            
            empathy_hits = 0
            for pat in empathy_patterns:
                if re.search(pat, response_lower):
                    empathy_hits += 1
            
            # Dismissive patterns
            dismissive_patterns = [
                r'\bjust (?:get over|move on|deal with|forget|stop)\b',
                r'\bit\'?s? (?:not (?:a |that )?big deal|nothing|no big)\b',
                r'\byou (?:should(?:n\'t)?|need to) (?:just|stop|get)\b',
            ]
            
            dismissive_hits = 0
            for pat in dismissive_patterns:
                if re.search(pat, response_lower):
                    dismissive_hits += 1
            
            empathy_score = min(1.0, 0.3 + empathy_hits * 0.12 - dismissive_hits * 0.2)
            empathy_score = max(0, empathy_score)
        
        # ============================================================
        # 7. RESPONSE COMPLETENESS AND DEPTH
        # ============================================================
        
        # Measure response length relative to query complexity
        query_complexity = len(re.findall(r'\b[a-zA-Z]+\b', query))
        
        # Ideal response should be substantial but not padded
        length_score = 0.0
        if word_count < 20:
            length_score = 0.2
        elif word_count < 50:
            length_score = 0.4
        elif word_count < 100:
            length_score = 0.7
        elif word_count < 200:
            length_score = 0.9
        elif word_count < 400:
            length_score = 1.0
        else:
            length_score = 0.9  # very long might be padded
        
        # Check for concrete examples or analogies
        analogy_patterns = [
            r'\b(?:imagine|picture|think of|consider|for example|for instance|such as|like when|'
            r'similar to|analogous to|comparable to|just like|as if|suppose)\b',
        ]
        analogy_count = 0
        for pat in analogy_patterns:
            analogy_count += len(re.findall(pat, response_lower))
        
        example_bonus = min(analogy_count * 0.08, 0.25)
        
        # ============================================================
        # 8. NEGATIVE SIGNAL DETECTION
        # Detect responses that are actively unhelpful or contradictory
        # ============================================================
        
        negative_score = 0.0
        
        # Contradicting or dismissing the user's concern
        dismissal_phrases = [
            r'\bit\'?s? just\b',
            r'\byou\'?re? (?:just|probably) (?:not|over)\b',
            r'\bdon\'?t (?:worry|think) (?:about|too much)\b',
            r'\bnot (?:a |that )?big deal\b',
        ]
        for pat in dismissal_phrases:
            if re.search(pat, response_lower):
                negative_score += 0.05
        
        # Responses that express inability/uncertainty about their own capability
        inability_patterns = [
            r'\bmight not (?:be able|have)\b',
            r'\bprobably (?:won\'?t|can\'?t|couldn\'?t)\b',
            r'\bmay not (?:be able|have)\b',
            r'\bit (?:might|may|could) not\b',
        ]
        for pat in inability_patterns:
            if re.search(pat, response_lower):
                negative_score += 0.08
        
        negative_score = min(negative_score, 0.4)
        
        # ============================================================
        # FINAL SCORE AGGREGATION
        # ============================================================
        
        # Weight the components
        final_score = (
            avg_sentence_density * 2.0 +      # Sentence-level info density (0-2.0)
            consistency_bonus * 1.0 +           # Consistency bonus (0-0.2)
            assertion_strength * 1.5 +          # Assertion strength (adjusted, ~0-1.5)
            assertion_score * 0.8 +             # Assertion score (0-0.8)
            structure_score * 1.2 +             # Discourse structure (0-1.2)
            specificity_score * 1.5 +           # Specificity gradient (0-1.5)
            engagement_score * 0.8 +            # Query engagement (0-0.8)
            empathy_score * 0.7 +               # Empathy quality (0-0.7)
            length_score * 0.8 +                # Completeness (0-0.8)
            example_bonus * 1.0 -               # Example/analogy bonus (0-0.25)
            negative_score * 2.0                # Negative signals penalty
        )
        
        # Normalize to 1-5 scale
        # Theoretical max around ~10-11, typical range 2-8
        normalized = 1.0 + (final_score / 9.0) * 4.0
        
        # Clamp to 1-5
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception as e:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 50:
                return 2.5
            return 1.5
        except:
            return 2.0