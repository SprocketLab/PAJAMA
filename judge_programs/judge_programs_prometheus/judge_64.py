def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using a
    discourse-level analysis approach focused on:
    1. Claim-to-evidence ratio (assertions vs. supporting reasoning)
    2. Modal verb spectrum analysis (strength of modal verbs used)
    3. Attribution and sourcing patterns
    4. Conditional/hypothetical framing detection
    5. Perspective-taking and acknowledgment of alternatives
    6. Query-appropriate confidence calibration
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-z]+\b', response_lower)
        num_words = max(len(words), 1)
        word_set = set(words)
        
        # ============================================================
        # 1. MODAL VERB SPECTRUM ANALYSIS
        # Categorize modal verbs by epistemic strength
        # ============================================================
        
        # Weak/tentative modals (good for uncertainty)
        weak_modals = ['might', 'could', 'may', 'would', 'can']
        # Strong/assertive modals (can indicate overconfidence)
        strong_modals = ['must', 'shall', 'will', 'need to', 'have to', 'should']
        
        weak_modal_count = sum(1 for w in words if w in weak_modals)
        strong_modal_count = 0
        for sm in strong_modals:
            strong_modal_count += len(re.findall(r'\b' + sm + r'\b', response_lower))
        
        # Ratio of weak to total modals — higher means more calibrated
        total_modals = weak_modal_count + strong_modal_count
        modal_calibration = 0.0
        if total_modals > 0:
            modal_calibration = weak_modal_count / total_modals
        else:
            modal_calibration = 0.3  # neutral if no modals used
        
        # ============================================================
        # 2. CLAIM-TO-REASONING RATIO
        # Detect bare assertions vs. supported claims
        # ============================================================
        
        # Reasoning/explanation indicators
        reasoning_markers = [
            'because', 'since', 'due to', 'as a result', 'therefore',
            'this means', 'which means', 'the reason', 'this is because',
            'given that', 'considering', 'in light of', 'for this reason',
            'this suggests', 'this indicates', 'evidence', 'based on',
            'according to', 'studies show', 'research indicates'
        ]
        
        reasoning_count = 0
        for marker in reasoning_markers:
            reasoning_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', response_lower))
        
        # Bare assertion patterns (declarative without qualification)
        bare_assertion_patterns = [
            r'\bis\b(?! (?:likely|possible|probable|uncertain|unclear))',
            r'\bare\b(?! (?:likely|possible|probable|uncertain|unclear))',
            r'^[A-Z][^.?!]*\bis\b[^.?!]*[.!]',  # Simple declarative
        ]
        
        # Count sentences that are pure declarations without hedging
        bare_assertions = 0
        qualified_claims = 0
        for sent in sentences:
            sent_lower = sent.lower().strip()
            has_qualifier = any(q in sent_lower for q in [
                'likely', 'possibly', 'perhaps', 'probably', 'might',
                'could', 'may', 'seems', 'appears', 'suggest',
                'generally', 'typically', 'often', 'usually',
                'in many cases', 'tend to', 'can be'
            ])
            has_reasoning = any(r in sent_lower for r in reasoning_markers)
            
            if has_qualifier or has_reasoning:
                qualified_claims += 1
            else:
                # Check if it's making a factual claim
                if re.search(r'\b(is|are|was|were|will|always|never|every|all|none)\b', sent_lower):
                    bare_assertions += 1
        
        claim_ratio = qualified_claims / num_sentences if num_sentences > 0 else 0
        
        # ============================================================
        # 3. CONDITIONAL/HYPOTHETICAL FRAMING
        # ============================================================
        
        conditional_patterns = [
            r'\bif\b', r'\bwhen\b.*\bmight\b', r'\bassuming\b',
            r'\bin case\b', r'\bprovided that\b', r'\bunless\b',
            r'\bdepending on\b', r'\bit depends\b', r'\bcontingent\b',
            r'\bhypothetically\b', r'\bin theory\b', r'\bin practice\b',
            r'\bunder certain\b', r'\bin some cases\b', r'\bin certain\b',
            r'\bwhere possible\b', r'\bwhen applicable\b'
        ]
        
        conditional_count = 0
        for pat in conditional_patterns:
            conditional_count += len(re.findall(pat, response_lower))
        
        conditional_score = min(conditional_count / num_sentences, 1.0) if num_sentences > 0 else 0
        
        # ============================================================
        # 4. PERSPECTIVE-TAKING & ALTERNATIVE ACKNOWLEDGMENT
        # ============================================================
        
        perspective_markers = [
            'on the other hand', 'alternatively', 'another perspective',
            'some people', 'others may', 'different views', 'it varies',
            'not everyone', 'there are different', 'various approaches',
            'one approach', 'another way', 'from one perspective',
            'while some', 'although', 'however', 'that said',
            'on one hand', 'conversely', 'at the same time',
            'different people', 'your situation', 'your case',
            'individual', 'personal', 'specific circumstances'
        ]
        
        perspective_count = sum(1 for m in perspective_markers if m in response_lower)
        perspective_score = min(perspective_count / max(num_sentences * 0.3, 1), 1.0)
        
        # ============================================================
        # 5. OVERCONFIDENCE DETECTION
        # ============================================================
        
        overconfidence_markers = [
            'obviously', 'clearly', 'undoubtedly', 'without a doubt',
            'absolutely', 'definitely', 'certainly', 'unquestionably',
            'there is no question', 'everyone knows', 'it is clear that',
            'the fact is', 'the truth is', 'always', 'never',
            'impossible', 'guaranteed', 'no way', 'for sure',
            'without question', 'beyond doubt', 'indisputable',
            'you need to', 'you must', 'you should just',
            'just do', 'simply', 'easy', 'no brainer'
        ]
        
        overconfidence_count = 0
        for marker in overconfidence_markers:
            overconfidence_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', response_lower))
        
        overconfidence_penalty = min(overconfidence_count * 0.15, 1.0)
        
        # ============================================================
        # 6. EMPATHY AND ACKNOWLEDGMENT SIGNALS
        # (relevant because many queries involve emotional/ambiguous situations)
        # ============================================================
        
        empathy_markers = [
            "i understand", "i can see", "it's understandable",
            "it's natural", "it's okay", "it's completely",
            "i hear you", "that makes sense", "i'm sorry",
            "i appreciate", "thank you for", "it sounds like",
            "you're feeling", "your feelings", "perfectly normal",
            "perfectly fine", "absolutely okay", "completely understandable",
            "i can hear", "i can imagine", "that must be"
        ]
        
        empathy_count = sum(1 for m in empathy_markers if m in response_lower)
        
        # ============================================================
        # 7. QUERY COMPLEXITY / AMBIGUITY DETECTION
        # ============================================================
        
        # Detect if the query involves ambiguity, emotion, or uncertainty
        ambiguity_indicators = [
            'how', 'why', 'what if', 'should', 'could', 'feeling',
            'struggling', 'difficult', 'complex', 'ambiguous',
            'uncertain', 'opinion', 'advice', 'help', 'concern',
            'worry', 'stress', 'fear', 'confus'
        ]
        
        query_ambiguity = sum(1 for ind in ambiguity_indicators if ind in query_lower)
        is_ambiguous_query = query_ambiguity >= 2
        
        # For ambiguous queries, we expect more hedging and empathy
        # For factual queries, we expect more reasoning and sourcing
        
        # ============================================================
        # 8. STRUCTURAL SOPHISTICATION
        # (numbered lists, structured advice, clear organization)
        # ============================================================
        
        has_enumeration = bool(re.search(r'(?:^|\n)\s*\d+[.)]\s', response))
        has_paragraphs = response.count('\n\n') >= 1
        
        # Sentence variety (not all same length — indicates thoughtfulness)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            length_variety = min(math.sqrt(variance) / max(mean_len, 1), 1.0)
        else:
            length_variety = 0.3
        
        # ============================================================
        # 9. DISMISSIVENESS DETECTION
        # ============================================================
        
        dismissive_patterns = [
            'just do', 'just get', 'get over it', 'move on',
            'not a big deal', 'no big deal', "it's just",
            'stop worrying', 'stop being', "don't worry about it",
            'you should be able to', 'it\'s not that hard',
            'you probably', 'maybe you\'re just', 'you\'re just not'
        ]
        
        dismissive_count = sum(1 for d in dismissive_patterns if d in response_lower)
        dismissive_penalty = min(dismissive_count * 0.2, 1.0)
        
        # ============================================================
        # 10. RESPONSE COMPLETENESS & DEPTH
        # ============================================================
        
        # Very short responses are usually lower quality
        length_score = 0.0
        if num_words < 20:
            length_score = 0.1
        elif num_words < 50:
            length_score = 0.4
        elif num_words < 100:
            length_score = 0.7
        elif num_words < 300:
            length_score = 1.0
        else:
            length_score = 0.9  # Very long might be rambling
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        # Weight components differently based on query type
        if is_ambiguous_query:
            # For ambiguous/emotional queries, empathy and perspective matter more
            score = (
                modal_calibration * 1.0 +          # 0-1
                claim_ratio * 1.5 +                  # 0-1.5
                conditional_score * 0.8 +            # 0-0.8
                perspective_score * 1.5 +            # 0-1.5
                min(empathy_count * 0.4, 1.5) +      # 0-1.5
                min(reasoning_count * 0.15, 0.8) +   # 0-0.8
                length_variety * 0.5 +               # 0-0.5
                length_score * 1.0 +                 # 0-1.0
                (0.3 if has_paragraphs else 0) +     # 0-0.3
                (0.3 if has_enumeration else 0)      # 0-0.3
            )
            # Apply penalties
            score -= overconfidence_penalty * 1.5
            score -= dismissive_penalty * 2.0
            score -= (bare_assertions / num_sentences) * 0.5
        else:
            # For more factual queries, reasoning and structure matter more
            score = (
                modal_calibration * 0.8 +            # 0-0.8
                claim_ratio * 1.2 +                  # 0-1.2
                conditional_score * 1.0 +            # 0-1.0
                perspective_score * 1.0 +            # 0-1.0
                min(empathy_count * 0.3, 1.0) +      # 0-1.0
                min(reasoning_count * 0.2, 1.2) +    # 0-1.2
                length_variety * 0.5 +               # 0-0.5
                length_score * 1.2 +                 # 0-1.2
                (0.4 if has_paragraphs else 0) +     # 0-0.4
                (0.4 if has_enumeration else 0)      # 0-0.4
            )
            # Apply penalties
            score -= overconfidence_penalty * 1.5
            score -= dismissive_penalty * 1.5
            score -= (bare_assertions / num_sentences) * 0.4
        
        # ============================================================
        # 11. CONTEXTUAL APPROPRIATENESS BONUS
        # If the response asks clarifying questions for ambiguous queries
        # ============================================================
        
        has_clarifying_question = bool(re.search(r'\?', response))
        # Check if query itself is ambiguous (e.g., "how to get there")
        query_is_vague = len(query.split()) < 30 and any(w in query_lower for w in ['ambiguous', 'no context', 'no previous'])
        
        if query_is_vague and has_clarifying_question:
            score += 0.8  # Bonus for seeking clarification
        
        # ============================================================
        # 12. NEGATION AWARENESS
        # "can't", "won't", "might not" — shows awareness of limitations
        # ============================================================
        
        limitation_markers = [
            "can't", "cannot", "won't", "might not", "may not",
            "not always", "not necessarily", "doesn't always",
            "isn't always", "not guaranteed", "no guarantee",
            "limitations", "drawback", "downside", "challenge"
        ]
        
        limitation_count = sum(1 for m in limitation_markers if m in response_lower)
        score += min(limitation_count * 0.15, 0.6)
        
        # Normalize to 1-5 scale
        # Theoretical max is roughly 9-10, min is around -2
        # Map to 1-5
        normalized = 1.0 + (score / 8.0) * 4.0
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception as e:
        return 2.5  # Safe fallback