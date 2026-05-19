def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication in LLM responses.
    
    This variant focuses on:
    1. Sentence-level confidence analysis (classifying each sentence as hedged, factual, or overconfident)
    2. Topic ambiguity detection from the query
    3. Source/evidence attribution patterns
    4. Conditional reasoning markers
    5. Perspective-taking and acknowledgment of alternatives
    
    Returns a score 0-100 where higher = better epistemic calibration.
    """
    try:
        if not response or not isinstance(response, str):
            return 5.0
        if not query or not isinstance(query, str):
            return 25.0
        
        resp_lower = response.lower()
        query_lower = query.lower()
        resp_len = len(response)
        
        import re
        
        # === 1. Sentence-level analysis ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = max(len(sentences), 1)
        
        # Hedging/uncertainty phrases (appropriate epistemic humility)
        hedge_phrases = [
            r'\bperhaps\b', r'\bpossibly\b', r'\bprobably\b', r'\bmight\b',
            r'\bcould be\b', r'\bseems?\b', r'\bappears?\b', r'\btends? to\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bin many cases\b', r'\bin some cases\b', r'\bit depends\b',
            r'\bnot necessarily\b', r'\bnot always\b', r'\bcan vary\b',
            r'\bto some extent\b', r'\bto a degree\b', r'\bargua?bly\b',
            r'\bplausib\w+\b', r'\breasonabl[ey]\b', r'\blikely\b',
            r'\bunlikely\b', r'\buncertain\b', r'\bdebat\w+\b',
            r'\bcontrovers\w+\b', r'\bnuanc\w+\b', r'\bcomplex\b',
            r'\bcomplicated\b', r'\bit\'s worth noting\b', r'\bthat said\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b.*\b(also|but|however)\b',
            r'\bon the other hand\b', r'\bconversely\b',
        ]
        
        # Evidence/source attribution patterns
        evidence_phrases = [
            r'\bresearch\s+(suggests?|shows?|indicates?|has\s+found)\b',
            r'\bstudies\s+(suggest|show|indicate|have\s+found)\b',
            r'\baccording to\b', r'\bevidence\s+(suggests?|indicates?|shows?)\b',
            r'\bdata\s+(suggests?|indicates?|shows?)\b',
            r'\bexperts?\s+(say|believe|suggest|argue|note)\b',
            r'\bscholars?\s+(argue|suggest|note|have)\b',
            r'\bliterature\s+(suggests?|shows?)\b',
            r'\bin\s+my\s+experience\b', r'\bfrom\s+what\s+i\b',
            r'\bas\s+far\s+as\s+i\s+know\b', r'\bto\s+my\s+knowledge\b',
            r'\bi\s+believe\b', r'\bi\s+think\b', r'\bin\s+my\s+opinion\b',
            r'\bhistorically\b', r'\btraditionally\b',
        ]
        
        # Overconfidence markers
        overconfidence_phrases = [
            r'\bdefinitely\b', r'\babsolutely\b', r'\bwithout\s+a?\s*doubt\b',
            r'\bundeniab\w+\b', r'\bunquestion\w+\b', r'\bobviously\b',
            r'\bclearly\b(?!\s+(not|uncertain|debat))', r'\bof\s+course\b',
            r'\beveryone\s+knows\b', r'\bit\s+is\s+a\s+fact\b',
            r'\bthe\s+truth\s+is\b', r'\bno\s+question\b',
            r'\balways\b(?!\s+(been|have|had|was|were|will|a\s+good))',
            r'\bnever\b(?!\s+(been|have|had|was|were|will|mind|the))',
            r'\bimpossible\b', r'\bcertain(ly)?\b(?!\s*(not|un|degree|extent))',
            r'\bguaranteed?\b', r'\bproven\s+fact\b',
            r'\bno\s+doubt\b', r'\bindisputab\w+\b',
        ]
        
        # Conditional reasoning markers
        conditional_phrases = [
            r'\bif\b.*\bthen\b', r'\bdepending\s+on\b', r'\bit\s+depends\b',
            r'\bin\s+the\s+case\s+(of|that|where)\b', r'\bassuming\b',
            r'\bgiven\s+that\b', r'\bprovided\s+that\b',
            r'\bunder\s+(certain|some|these|those)\s+conditions?\b',
            r'\bcontext\s+(matters?|dependent|specific)\b',
            r'\bcase[\s-]by[\s-]case\b', r'\bsituation\w*\b',
            r'\bcircumstances?\b',
        ]
        
        # Perspective/alternative acknowledgment
        perspective_phrases = [
            r'\bone\s+(view|perspective|argument|approach)\b',
            r'\banother\s+(view|perspective|argument|approach|way)\b',
            r'\bsome\s+(people|argue|believe|think|say|would)\b',
            r'\bothers\s+(argue|believe|think|say|would|might|may)\b',
            r'\bthere\s+are\s+(different|various|multiple|several)\b',
            r'\bdifferent\s+(perspectives?|views?|opinions?|approaches?)\b',
            r'\bon\s+one\s+hand\b', r'\bon\s+the\s+other\b',
            r'\balternative\w*\b', r'\bcounter[\s-]?argument\b',
            r'\bfrom\s+(a|the|this|another)\s+\w+\s+perspective\b',
            r'\bfor\s+example\b', r'\bfor\s+instance\b',
            r'\bsuch\s+as\b', r'\bincluding\b',
        ]
        
        # Count matches per category
        def count_pattern_matches(text, patterns):
            count = 0
            for pat in patterns:
                count += len(re.findall(pat, text, re.IGNORECASE))
            return count
        
        hedge_count = count_pattern_matches(resp_lower, hedge_phrases)
        evidence_count = count_pattern_matches(resp_lower, evidence_phrases)
        overconfidence_count = count_pattern_matches(resp_lower, overconfidence_phrases)
        conditional_count = count_pattern_matches(resp_lower, conditional_phrases)
        perspective_count = count_pattern_matches(resp_lower, perspective_phrases)
        
        # === 2. Query ambiguity/subjectivity detection ===
        ambiguity_indicators = [
            r'\bethic\w*\b', r'\bmoral\w*\b', r'\bopinion\b', r'\bthink\b',
            r'\bfeel\b', r'\bbelieve\b', r'\bshould\b', r'\bwould\b',
            r'\bbest\b', r'\bworst\b', r'\bwhy\s+(do|does|did|is|are|was|were)\b',
            r'\bhow\s+(do|does|did|should|would|can|could)\b',
            r'\bwhat\s+(do\s+you|should|would|is\s+the\s+best)\b',
            r'\bcontrovers\w+\b', r'\bdebat\w+\b', r'\bargument\b',
            r'\bphilosoph\w+\b', r'\bpolitics?\b', r'\bpolitical\b',
            r'\bsubjectiv\w+\b', r'\bperspective\b', r'\bexperience\b',
            r'\bimpact\b', r'\baffect\b', r'\bworth\b',
            r'\bwent\s+wrong\b', r'\bwhat\s+happened\b',
        ]
        
        query_ambiguity = count_pattern_matches(query_lower, ambiguity_indicators)
        is_ambiguous = query_ambiguity >= 2
        is_somewhat_ambiguous = query_ambiguity >= 1
        
        # Factual/technical query indicators
        factual_indicators = [
            r'\bhow\s+to\b', r'\bwhat\s+is\s+the\b', r'\bdefine\b',
            r'\bcalculat\w+\b', r'\bformula\b', r'\bsql\b', r'\bcode\b',
            r'\bprogram\w*\b', r'\bsyntax\b', r'\berror\b',
            r'\bstep\w*\b', r'\bmethod\b', r'\brecipe\b',
            r'CREATE\s+TABLE', r'SELECT\b', r'\bfunction\b',
        ]
        query_factual = count_pattern_matches(query_lower, factual_indicators)
        # Also check original case for SQL
        query_factual += len(re.findall(r'CREATE\s+TABLE|SELECT\b', query))
        is_factual = query_factual >= 2
        
        # === 3. Sentence-level classification ===
        hedged_sentences = 0
        overconfident_sentences = 0
        neutral_sentences = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            has_hedge = any(re.search(p, sent_lower) for p in hedge_phrases[:20])
            has_overconf = any(re.search(p, sent_lower) for p in overconfidence_phrases)
            
            if has_hedge and not has_overconf:
                hedged_sentences += 1
            elif has_overconf and not has_hedge:
                overconfident_sentences += 1
            else:
                neutral_sentences += 1
        
        hedged_ratio = hedged_sentences / num_sentences
        overconfident_ratio = overconfident_sentences / num_sentences
        
        # === 4. Scoring ===
        score = 50.0  # Base score
        
        # --- Response length and substance ---
        word_count = len(response.split())
        if word_count < 15:
            score -= 15
        elif word_count < 30:
            score -= 8
        elif word_count > 50:
            score += min((word_count - 50) * 0.05, 8)
        
        # --- Hedging score (normalized by response length) ---
        hedge_density = hedge_count / max(word_count, 1) * 100
        hedge_score = min(hedge_density * 3.0, 12)
        score += hedge_score
        
        # --- Evidence attribution ---
        evidence_density = evidence_count / max(word_count, 1) * 100
        evidence_score = min(evidence_density * 5.0, 10)
        score += evidence_score
        
        # --- Conditional reasoning ---
        conditional_density = conditional_count / max(word_count, 1) * 100
        conditional_score = min(conditional_density * 4.0, 8)
        score += conditional_score
        
        # --- Perspective acknowledgment ---
        perspective_density = perspective_count / max(word_count, 1) * 100
        perspective_score = min(perspective_density * 4.0, 10)
        score += perspective_score
        
        # --- Overconfidence penalty ---
        overconfidence_density = overconfidence_count / max(word_count, 1) * 100
        if is_ambiguous:
            # Penalize overconfidence more on ambiguous topics
            overconfidence_penalty = min(overconfidence_density * 6.0, 20)
        else:
            overconfidence_penalty = min(overconfidence_density * 3.0, 12)
        score -= overconfidence_penalty
        
        # --- Sentence-level calibration ---
        if is_ambiguous or is_somewhat_ambiguous:
            # For ambiguous queries, reward having some hedged sentences
            if hedged_ratio > 0.1:
                score += 5
            if hedged_ratio > 0.25:
                score += 3
            # Penalize high overconfidence ratio on ambiguous topics
            if overconfident_ratio > 0.3:
                score -= 8
        
        # === 5. Structural quality signals ===
        
        # Multiple paragraphs or structured response
        paragraphs = [p.strip() for p in response.split('\n') if len(p.strip()) > 20]
        if len(paragraphs) > 1:
            score += 3
        if len(paragraphs) > 3:
            score += 2
        
        # Acknowledges complexity or limitations
        complexity_markers = [
            r'\bthis is (a )?(complex|complicated|nuanced)\b',
            r'\bthere\'s no (simple|easy|straightforward)\b',
            r'\bit\'s (not|more) (that )?straightforward\b',
            r'\bi\'m not (sure|certain|an expert)\b',
            r'\bi don\'t (know|have) (enough|all)\b',
            r'\btake this with\b', r'\bgrain of salt\b',
            r'\byour mileage may vary\b', r'\bymmv\b',
            r'\bdisclaimer\b', r'\bcaveat\b',
            r'\bkeep in mind\b', r'\bnote that\b',
            r'\bimportant(ly)?\s+to\s+(note|consider|remember)\b',
        ]
        complexity_count = count_pattern_matches(resp_lower, complexity_markers)
        score += min(complexity_count * 3, 8)
        
        # === 6. Engagement and depth signals ===
        
        # References to specific examples, names, works
        specific_refs = len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', response))  # Proper names
        book_refs = len(re.findall(r'\*[^*]+\*', response))  # Italicized titles
        quote_refs = len(re.findall(r'"[^"]{10,}"', response))  # Quotes
        
        specificity_score = min((specific_refs * 0.5 + book_refs * 2 + quote_refs * 1.5), 8)
        score += specificity_score
        
        # === 7. Contextual calibration ===
        
        # For factual/technical queries, don't penalize confident factual statements as much
        if is_factual:
            # Restore some overconfidence penalty since factual answers can be confident
            score += overconfidence_penalty * 0.4
            # But still reward evidence and structure
        
        # For creative/roleplay queries, different calibration
        creative_indicators = [r'\bimagine\b', r'\bstory\b', r'\bdialogue\b', 
                              r'\bcreative\b', r'\bwrite\b', r'\bfiction\b',
                              r'\broleplay\b', r'\bprotagonist\b']
        is_creative = any(re.search(p, query_lower) for p in creative_indicators)
        
        if is_creative:
            # For creative tasks, epistemic calibration matters less
            # Reward engagement and immersion instead
            engagement_markers = [r'\*[^*]+\*', r'\n', r':', r'\.\.\.']
            engagement = sum(len(re.findall(p, response)) for p in engagement_markers)
            score += min(engagement * 1.5, 10)
            # Reduce epistemic penalties
            score += overconfidence_penalty * 0.5
        
        # === 8. Personal experience framing ===
        personal_markers = [
            r'\bin my experience\b', r'\bpersonally\b', r'\bfor me\b',
            r'\bi\'ve (found|seen|noticed|experienced)\b',
            r'\bfrom my\b', r'\bwhen i\b', r'\bi (was|am|have|had|did)\b',
        ]
        personal_count = count_pattern_matches(resp_lower, personal_markers)
        # Personal framing is good epistemic practice - it scopes claims
        if personal_count > 0:
            score += min(personal_count * 2, 6)
        
        # === 9. Balanced "both sides" or multi-faceted analysis ===
        balance_markers = [
            r'\bboth\b', r'\bon\s+(one|the\s+other)\s+hand\b',
            r'\bpros?\s+and\s+cons?\b', r'\btrade[\s-]?off\b',
            r'\badvantage\w*\b.*\bdisadvantage\w*\b',
            r'\bwhile\b.*\balso\b', r'\bbut\s+also\b',
            r'\bessentially,?\s+both\b',
        ]
        balance_count = count_pattern_matches(resp_lower, balance_markers)
        score += min(balance_count * 2.5, 7)
        
        # === 10. Explanation depth / elaboration ===
        # Responses that explain reasoning show better calibration
        explanation_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bthis\s+(means|implies|suggests)\b', r'\bin\s+other\s+words\b',
            r'\bthe\s+reason\b', r'\bdue\s+to\b', r'\bas\s+a\s+result\b',
            r'\bconsequen\w+\b',
        ]
        explanation_count = count_pattern_matches(resp_lower, explanation_markers)
        explanation_density = explanation_count / max(word_count, 1) * 100
        score += min(explanation_density * 3, 7)
        
        # === Final normalization ===
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0