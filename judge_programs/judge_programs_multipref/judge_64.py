def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant uses a unique approach based on:
    1. Claim density analysis (ratio of assertive claims to total content)
    2. Source attribution and evidence referencing patterns
    3. Conditional/nuanced language detection (beyond simple hedging)
    4. Epistemic verb analysis (know/believe/think/assume spectrum)
    5. Scope-limiting expressions detection
    6. Contrast and concession pattern analysis (however, although, on the other hand)
    7. Question rhetorical awareness (does response match query's epistemic demands)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        import re
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. Epistemic Verb Spectrum Analysis ===
        # Categorize verbs by epistemic strength
        strong_epistemic = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bwill\b',
            r'\balways\b', r'\bnever\b', r'\bcertainly\b', r'\bdefinitely\b',
            r'\babsolutely\b', r'\bundoubtedly\b', r'\bwithout doubt\b',
            r'\bno question\b', r'\bclearly\b', r'\bobviously\b',
            r'\bof course\b', r'\bguaranteed\b', r'\bproven\b',
            r'\bunquestionably\b', r'\bindisputably\b'
        ]
        
        moderate_epistemic = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bcommonly\b', r'\bin most cases\b', r'\btends to\b',
            r'\bwidely\b', r'\bfrequently\b', r'\bnormally\b',
            r'\bas a rule\b', r'\bby and large\b'
        ]
        
        weak_epistemic = [
            r'\bmight\b', r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bmaybe\b', r'\bseem\b', r'\bseems\b', r'\bappear\b',
            r'\bappears\b', r'\bsuggest\b', r'\bsuggests\b',
            r'\blikely\b', r'\bunlikely\b', r'\bprobably\b',
            r'\bmay\b', r'\bpotentially\b', r'\bapproximately\b',
            r'\broughly\b', r'\bestimated\b', r'\btentatively\b'
        ]
        
        strong_count = sum(len(re.findall(p, resp_lower)) for p in strong_epistemic)
        moderate_count = sum(len(re.findall(p, resp_lower)) for p in moderate_epistemic)
        weak_count = sum(len(re.findall(p, resp_lower)) for p in weak_epistemic)
        
        total_epistemic = strong_count + moderate_count + weak_count
        if total_epistemic > 0:
            # Ratio of nuanced (moderate+weak) to total epistemic markers
            nuance_ratio = (moderate_count + weak_count) / total_epistemic
            # Reward balanced epistemic expression
            score += nuance_ratio * 8
        
        # === 2. Source Attribution and Evidence Patterns ===
        evidence_patterns = [
            r'\bresearch\b', r'\bstud(?:y|ies)\b', r'\baccording to\b',
            r'\bevidence\b', r'\bdata\b', r'\bfindings\b',
            r'\bexperts?\b', r'\bscientists?\b', r'\banalysis\b',
            r'\breported\b', r'\bsource\b', r'\bjournal\b',
            r'\bpublished\b', r'\bstatistics?\b', r'\bsurvey\b',
            r'\bexperiment\b', r'\bobserv(?:ed|ation)\b'
        ]
        evidence_count = sum(len(re.findall(p, resp_lower)) for p in evidence_patterns)
        evidence_score = min(evidence_count * 1.5, 8)
        score += evidence_score
        
        # === 3. Scope-Limiting Expressions ===
        scope_limiters = [
            r'\bin some cases\b', r'\bin certain\b', r'\bdepending on\b',
            r'\bunder certain\b', r'\bin particular\b', r'\bspecifically\b',
            r'\bin general\b', r'\bfor the most part\b', r'\bto some extent\b',
            r'\bpartially\b', r'\bto a degree\b', r'\bin part\b',
            r'\bone of\b', r'\bamong\b', r'\bvaries\b', r'\bvary\b',
            r'\bdepends\b', r'\bcontext\b', r'\bsituation\b',
            r'\bcircumstances?\b', r'\bnot always\b', r'\bnot necessarily\b',
            r'\bit depends\b', r'\bcan vary\b', r'\brange\b'
        ]
        scope_count = sum(len(re.findall(p, resp_lower)) for p in scope_limiters)
        scope_score = min(scope_count * 2.0, 10)
        score += scope_score
        
        # === 4. Contrast and Concession Patterns ===
        contrast_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\bin contrast\b',
            r'\bthat said\b', r'\bhaving said that\b', r'\bat the same time\b',
            r'\bbut\b', r'\byet\b', r'\bstill\b', r'\bnonetheless\b',
            r'\bon the contrary\b', r'\balternatively\b'
        ]
        contrast_count = sum(len(re.findall(p, resp_lower)) for p in contrast_patterns)
        # Normalize by sentence count
        contrast_density = contrast_count / num_sentences
        contrast_score = min(contrast_density * 12, 8)
        score += contrast_score
        
        # === 5. Overconfidence Penalty (unique: per-sentence absolutism detection) ===
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bevery\b', r'\bnone\b',
            r'\ball\b', r'\bno one\b', r'\beveryone\b', r'\bnothing\b',
            r'\bcompletely\b', r'\btotally\b', r'\bentirely\b',
            r'\bperfect(?:ly)?\b', r'\bimpossible\b', r'\bcertain(?:ly)?\b',
            r'\babsolutely\b', r'\bundoubtedly\b', r'\bwithout a doubt\b',
            r'\bguarantee\b', r'\bno way\b', r'\b100%\b',
            r'\bthe best\b', r'\bthe worst\b', r'\bthe only\b'
        ]
        
        absolute_sentence_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            sent_abs = sum(1 for p in absolute_patterns if re.search(p, sent_lower))
            if sent_abs >= 2:  # Sentence has multiple absolute claims
                absolute_sentence_count += 1
        
        absolute_ratio = absolute_sentence_count / num_sentences
        overconfidence_penalty = absolute_ratio * 15
        score -= overconfidence_penalty
        
        # === 6. Claim Density Analysis ===
        # Count declarative statements (sentences ending with period that make claims)
        declarative_markers = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bhas\b', r'\bhave\b', r'\bhad\b', r'\bwill\b',
            r'\bshould\b', r'\bmust\b', r'\bcan\b'
        ]
        
        claim_sentences = 0
        qualified_claims = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_claim = any(re.search(p, sent_lower) for p in declarative_markers)
            if has_claim:
                claim_sentences += 1
                # Check if the claim is qualified
                has_qualifier = any(re.search(p, sent_lower) for p in 
                                   weak_epistemic + moderate_epistemic + scope_limiters)
                if has_qualifier:
                    qualified_claims += 1
        
        if claim_sentences > 0:
            qualification_ratio = qualified_claims / claim_sentences
            # Reward having some qualified claims (but not necessarily all)
            # Optimal: 20-50% of claims are qualified
            if 0.15 <= qualification_ratio <= 0.6:
                score += 6
            elif qualification_ratio > 0.6:
                score += 3  # Over-hedging slightly penalized
            elif qualification_ratio > 0:
                score += 2
        
        # === 7. Query Epistemic Demand Matching ===
        # Detect if query asks about uncertain/debatable topics
        uncertain_query_markers = [
            r'\bdo you think\b', r'\bwhat do you\b', r'\bshould\b',
            r'\bopinion\b', r'\bbest\b', r'\bworst\b', r'\brecommend\b',
            r'\badvice\b', r'\bwhat happens\b', r'\bwhy\b',
            r'\bhow can\b', r'\bwhat if\b', r'\bis it true\b',
            r'\bis it possible\b', r'\bcan you explain\b'
        ]
        
        factual_query_markers = [
            r'\bhow (?:is|are|do|does|many|much)\b', r'\bwhat is\b',
            r'\bwhat are\b', r'\bdefine\b', r'\bcalculate\b',
            r'\bfind\b', r'\bwhen\b', r'\bwhere\b',
            r'\blist\b', r'\bname\b', r'\bdescribe\b'
        ]
        
        query_uncertainty = sum(1 for p in uncertain_query_markers if re.search(p, query_lower))
        query_factual = sum(1 for p in factual_query_markers if re.search(p, query_lower))
        
        if query_uncertainty > query_factual:
            # For uncertain topics, reward hedging more
            hedging_bonus = (weak_count + moderate_count) * 1.0
            score += min(hedging_bonus, 6)
            # Penalize overconfidence more on uncertain topics
            score -= overconfidence_penalty * 0.5  # Additional penalty
        
        # === 8. Structural Sophistication (unique: paragraph transitions and logical flow) ===
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Multi-paragraph responses tend to be more thoughtful
        if num_paragraphs >= 2:
            score += min(num_paragraphs * 1.0, 4)
        
        # Check for transition words between ideas (sign of nuanced thinking)
        transition_words = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b',
            r'\bthird(?:ly)?\b', r'\bfinally\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\boverall\b', r'\bin summary\b',
            r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
            r'\bas a result\b', r'\bfor (?:example|instance)\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b'
        ]
        transition_count = sum(len(re.findall(p, resp_lower)) for p in transition_words)
        transition_density = transition_count / num_sentences
        score += min(transition_density * 6, 5)
        
        # === 9. Assertive first-person certainty detection ===
        first_person_certain = [
            r'\bi (?:am sure|am certain|know for a fact|guarantee)\b',
            r'\bi believe\b', r'\bi think\b', r'\bin my opinion\b',
            r'\bi feel\b', r'\bfrom my perspective\b',
            r'\bi would say\b', r'\bi\'d say\b'
        ]
        
        # "I believe" and "I think" are actually good epistemic markers
        opinion_markers = sum(len(re.findall(p, resp_lower)) for p in first_person_certain[1:])
        certainty_markers = sum(len(re.findall(p, resp_lower)) for p in first_person_certain[:1])
        
        score += min(opinion_markers * 1.5, 4)
        score -= certainty_markers * 2
        
        # === 10. Numerical precision / vagueness balance ===
        # Detect if response uses precise numbers vs ranges/approximations
        precise_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        approximate_nums = re.findall(r'(?:about|around|approximately|roughly|nearly|almost)\s+\d+', resp_lower)
        
        if len(precise_numbers) > 3:
            if len(approximate_nums) > 0:
                # Using approximations with numbers = good calibration
                score += 3
            else:
                # Many precise numbers without hedging = slight overconfidence
                score -= 1
        
        # === 11. Response completeness and engagement ===
        # Longer, more complete responses tend to score better in examples
        length_score = 0
        if word_count >= 50:
            length_score += 2
        if word_count >= 100:
            length_score += 2
        if word_count >= 150:
            length_score += 1
        score += length_score
        
        # Detect formatting richness (headers, bold, lists)
        has_headers = bool(re.search(r'#{1,3}\s', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        
        formatting_bonus = sum([has_headers * 1.5, has_bold * 1.0, has_numbered_list * 1.0])
        score += min(formatting_bonus, 3)
        
        # === 12. Vocabulary sophistication (unique: academic/analytical word usage) ===
        analytical_words = [
            r'\banalysis\b', r'\bimplication\b', r'\bperspective\b',
            r'\bconsider(?:ation)?\b', r'\bfactor\b', r'\baspect\b',
            r'\bcomponent\b', r'\bframework\b', r'\bapproach\b',
            r'\bmethod\b', r'\bstrategy\b', r'\bprocess\b',
            r'\bcriteria\b', r'\bprinciple\b', r'\bconcept\b',
            r'\bsignificant\b', r'\bsubstantial\b', r'\bnotable\b',
            r'\brelevant\b', r'\beffective\b', r'\befficient\b',
            r'\bappropriate\b', r'\bimportant\b', r'\bessential\b',
            r'\bprimary\b', r'\bfundamental\b'
        ]
        analytical_count = sum(len(re.findall(p, resp_lower)) for p in analytical_words)
        analytical_density = analytical_count / word_count if word_count > 0 else 0
        score += min(analytical_density * 80, 5)
        
        # Clamp score to 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0