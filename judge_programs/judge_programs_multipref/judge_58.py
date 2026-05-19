def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Detecting hedging/uncertainty markers and their appropriate usage
    2. Detecting overconfident/absolute language
    3. Analyzing claim density and whether claims are qualified
    4. Checking for source attribution and evidence references
    5. Distinguishing factual vs opinion/speculative content
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. HEDGING AND UNCERTAINTY MARKERS ===
        # Words/phrases that appropriately communicate uncertainty
        hedging_phrases = [
            r'\bmight\b', r'\bcould\b', r'\bmay\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\blikely\b', r'\bunlikely\b', r'\bprobably\b', r'\bpotentially\b',
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\boften\b', r'\bsometimes\b', r'\bin some cases\b', r'\bin many cases\b',
            r'\bresearch suggests\b', r'\bstudies suggest\b', r'\bevidence suggests\b',
            r'\bit appears\b', r'\bit seems\b', r'\bseems to\b', r'\bappears to\b',
            r'\bto some extent\b', r'\bto a degree\b', r'\bapproximately\b',
            r'\broughly\b', r'\babout\b', r'\baround\b',
            r'\bone possible\b', r'\banother possible\b', r'\bone approach\b',
            r'\bdepending on\b', r'\bit depends\b', r'\bvaries\b',
            r'\bnot necessarily\b', r'\bnot always\b', r'\bnot everyone\b',
            r'\bin general\b', r'\bas a rule\b', r'\bfor the most part\b',
            r'\bcan be\b', r'\bmay be\b', r'\bcould be\b',
            r'\bi think\b', r'\bi believe\b', r'\bin my view\b', r'\bin my opinion\b',
            r'\bsuggests that\b', r'\bindicates that\b', r'\bimplies that\b',
        ]
        
        hedge_count = 0
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_density = hedge_count / num_sentences
        # Reward moderate hedging (not too little, not too much)
        if hedge_density < 0.1:
            score -= 3  # Too few hedges
        elif hedge_density < 0.5:
            score += 5  # Good amount
        elif hedge_density < 1.0:
            score += 3  # Acceptable
        else:
            score -= 2  # Over-hedging can indicate lack of confidence
        
        # === 2. OVERCONFIDENCE MARKERS ===
        overconfident_phrases = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\babsolutely\b', r'\bundoubtedly\b', r'\bwithout a doubt\b',
            r'\bwithout question\b', r'\bobviously\b', r'\bclearly\b',
            r'\beveryone knows\b', r'\bit is clear that\b', r'\bit is obvious\b',
            r'\bthere is no doubt\b', r'\bno question\b', r'\bguaranteed\b',
            r'\bimpossible\b', r'\bincontrovertible\b', r'\bunquestionable\b',
            r'\bthe fact is\b', r'\bthe truth is\b', r'\bin fact\b',
            r'\bof course\b', r'\bneedless to say\b',
            r'\beveryone\b', r'\bnobody\b', r'\bno one\b',
            r'\ball\s+\w+\s+are\b', r'\bevery\s+\w+\s+is\b',
        ]
        
        overconfident_count = 0
        for pattern in overconfident_phrases:
            overconfident_count += len(re.findall(pattern, response_lower))
        
        overconfident_density = overconfident_count / num_sentences
        # Penalize overconfidence
        score -= min(overconfident_density * 8, 15)
        
        # === 3. CLAIM QUALIFICATION ANALYSIS ===
        # Look at declarative sentences and check if they're qualified
        declarative_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bwill\b',
            r'\bcauses\b', r'\bleads to\b', r'\bresults in\b', r'\bproduces\b',
        ]
        
        qualified_claims = 0
        unqualified_claims = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            has_claim = any(re.search(p, sent_lower) for p in declarative_patterns)
            if has_claim:
                has_qualifier = any(re.search(p, sent_lower) for p in hedging_phrases)
                if has_qualifier:
                    qualified_claims += 1
                else:
                    unqualified_claims += 1
        
        total_claims = qualified_claims + unqualified_claims
        if total_claims > 0:
            qualification_ratio = qualified_claims / total_claims
            # Reward moderate qualification (not everything needs qualifying)
            if 0.15 <= qualification_ratio <= 0.6:
                score += 5
            elif qualification_ratio > 0.6:
                score += 2  # Slightly over-qualified
            else:
                score -= 3  # Under-qualified
        
        # === 4. SOURCE/EVIDENCE ATTRIBUTION ===
        evidence_patterns = [
            r'\bresearch\b', r'\bstudy\b', r'\bstudies\b', r'\baccording to\b',
            r'\bexperts\b', r'\bscientists\b', r'\bdata\b', r'\bfindings\b',
            r'\bevidence\b', r'\bsurvey\b', r'\breport\b', r'\banalysis\b',
            r'\bsource\b', r'\breference\b', r'\bliterature\b',
            r'\bpublished\b', r'\bjournal\b', r'\bexperiment\b',
            r'\bstatistics\b', r'\bstatistical\b',
        ]
        
        evidence_count = sum(len(re.findall(p, response_lower)) for p in evidence_patterns)
        evidence_score = min(evidence_count * 1.5, 8)
        score += evidence_score
        
        # === 5. QUERY SENSITIVITY ===
        # Determine if the query is about something factual, subjective, or uncertain
        subjective_query_markers = [
            r'\bdo you think\b', r'\bwhat do you\b', r'\bshould\b', r'\bopinion\b',
            r'\bbest\b', r'\bworst\b', r'\bfavorite\b', r'\brecommend\b',
            r'\bwhat if\b', r'\bcould\b', r'\bwould\b',
        ]
        
        speculative_query_markers = [
            r'\bwhy\b', r'\bhow\b', r'\bexplain\b', r'\bwhat causes\b',
            r'\bwhat happens\b', r'\bpredict\b', r'\bfuture\b',
        ]
        
        factual_query_markers = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen\b', r'\bwhere\b',
            r'\bhow many\b', r'\bhow much\b', r'\bdefine\b', r'\blist\b',
        ]
        
        is_subjective = any(re.search(p, query_lower) for p in subjective_query_markers)
        is_speculative = any(re.search(p, query_lower) for p in speculative_query_markers)
        is_factual = any(re.search(p, query_lower) for p in factual_query_markers)
        
        # For subjective queries, reward acknowledgment of subjectivity
        if is_subjective:
            subj_acknowledgment = [
                r'\bsubjective\b', r'\bdepends\b', r'\bperspective\b',
                r'\bpoint of view\b', r'\bvaried\b', r'\bdifferent opinions\b',
                r'\bsome people\b', r'\bothers\b', r'\bpersonal\b',
                r'\bpreference\b', r'\bvaries\b',
            ]
            ack_count = sum(1 for p in subj_acknowledgment if re.search(p, response_lower))
            score += min(ack_count * 2, 6)
        
        # === 6. STRUCTURAL QUALITY INDICATORS ===
        # Check for organized presentation (numbered lists, headers, etc.)
        has_structure = bool(re.search(r'(\d+\.\s|\*\s|#{1,3}\s|•)', response))
        if has_structure:
            score += 3
        
        # Check for balanced presentation (pros/cons, multiple viewpoints)
        balance_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\balternatively\b',
            r'\bconversely\b', r'\bthat said\b', r'\bnevertheless\b',
            r'\bbut\b', r'\balthough\b', r'\bwhile\b', r'\bdespite\b',
            r'\bin contrast\b', r'\bon the contrary\b',
        ]
        balance_count = sum(1 for p in balance_markers if re.search(p, response_lower))
        score += min(balance_count * 1.5, 6)
        
        # === 7. CONDITIONAL LANGUAGE ===
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bprovided that\b', r'\bin the case\b', r'\bwhen\b.*\byou\b',
            r'\bdepending on\b', r'\bconditional\b',
        ]
        conditional_count = sum(len(re.findall(p, response_lower)) for p in conditional_patterns)
        score += min(conditional_count * 1.5, 5)
        
        # === 8. SPECIFICITY AND DETAIL ===
        # Specific details suggest knowledge rather than vague hand-waving
        # Check for numbers, proper nouns, specific terms
        number_count = len(re.findall(r'\b\d+\.?\d*\b', response))
        score += min(number_count * 0.3, 4)
        
        # Check for parenthetical explanations (e.g., ...)
        paren_count = len(re.findall(r'\(.*?\)', response))
        score += min(paren_count * 0.5, 3)
        
        # === 9. RESPONSE COMPLETENESS ===
        # Penalize truncated responses
        if response.rstrip()[-1:] not in '.!?")\n' and num_words > 20:
            # Likely truncated
            score -= 3
        
        # Check if response ends mid-sentence
        last_char = response.strip()[-1] if response.strip() else ''
        if last_char in ',;:' or (last_char.isalpha() and not response.strip().endswith('etc')):
            score -= 4  # Truncated
        
        # === 10. APPROPRIATE RESPONSE LENGTH ===
        # Moderate length is generally better for well-calibrated responses
        if num_words < 20:
            score -= 5
        elif num_words < 50:
            score -= 2
        elif num_words > 100:
            score += 2
        elif num_words > 200:
            score += 3
        
        # === 11. EPISTEMIC VOCABULARY RICHNESS ===
        # Count unique epistemic terms used
        epistemic_vocab = set()
        all_epistemic = hedging_phrases + evidence_patterns + balance_markers
        for pattern in all_epistemic:
            matches = re.findall(pattern, response_lower)
            for m in matches:
                epistemic_vocab.add(m.strip())
        
        vocab_richness = len(epistemic_vocab)
        score += min(vocab_richness * 0.8, 6)
        
        # === 12. ACKNOWLEDGMENT OF LIMITATIONS ===
        limitation_phrases = [
            r'\bi\'m not sure\b', r'\bi don\'t know\b', r'\bi\'m not aware\b',
            r'\bunclear\b', r'\bunknown\b', r'\blimited information\b',
            r'\bmore research\b', r'\bfurther investigation\b',
            r'\bnot enough\b', r'\bdifficult to say\b', r'\bhard to say\b',
            r'\bcannot be certain\b', r'\bnot certain\b',
            r'\bbeyond my\b', r'\boutside my\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bimportant to note\b',
            r'\bworth noting\b', r'\bbear in mind\b',
        ]
        
        limitation_count = sum(1 for p in limitation_phrases if re.search(p, response_lower))
        score += min(limitation_count * 2, 6)
        
        # === 13. ENGAGEMENT QUALITY ===
        # Responses that engage with the query topic specifically
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
        response_words_set = set(re.findall(r'\b\w{4,}\b', response_lower))
        
        if query_words:
            overlap = len(query_words & response_words_set) / len(query_words)
            score += overlap * 4  # Reward relevance
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        return round(score, 2)
        
    except Exception as e:
        return 25.0