def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/explanatory connective density (because, since, therefore, thus, hence, etc.)
    2. Sequential/procedural markers (first, then, next, finally, step N)
    3. Hedging and qualification language (showing nuanced reasoning)
    4. Question-answer internal structure (rhetorical questions followed by answers)
    5. Conditional reasoning patterns (if...then, when...would)
    6. Depth of elaboration per claim (ratio of supporting sentences to claim sentences)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.0
        
        import re
        from collections import Counter
        
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Tokenize into words
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # ============================================================
        # 1. CAUSAL/EXPLANATORY CONNECTIVES (why behind claims)
        # ============================================================
        causal_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bthat\'s why\b',
            r'\bwhich means\b', r'\bwhich leads to\b', r'\bso that\b',
            r'\bin order to\b', r'\bthis ensures\b', r'\bthis helps\b',
            r'\bthis allows\b', r'\bthis makes\b', r'\benabling\b',
            r'\bleading to\b', r'\bresulting in\b', r'\bcaused by\b',
        ]
        causal_count = 0
        resp_lower = response_clean.lower()
        for pat in causal_patterns:
            causal_count += len(re.findall(pat, resp_lower))
        
        # Normalize by number of sentences
        causal_density = causal_count / num_sentences
        # Score: up to 15 points
        score += min(causal_density * 10, 15)
        
        # ============================================================
        # 2. SEQUENTIAL/PROCEDURAL MARKERS (step-by-step)
        # ============================================================
        sequential_patterns = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bafter that\b', r'\bfinally\b',
            r'\bto begin\b', r'\bto start\b', r'\bfollowing that\b',
            r'\bsubsequently\b', r'\bonce\b', r'\bafter\b',
            r'\bstep\s*\d', r'\bphase\s*\d', r'\b\d+[\.\)]\s',
            r'\blast(?:ly)?\b', r'\bin the end\b', r'\bto conclude\b',
            r'\bmoving on\b', r'\bnow\b', r'\bat this point\b',
        ]
        seq_count = 0
        for pat in sequential_patterns:
            seq_count += len(re.findall(pat, resp_lower))
        
        seq_density = seq_count / num_sentences
        score += min(seq_density * 8, 12)
        
        # ============================================================
        # 3. ELABORATION MARKERS (explaining further)
        # ============================================================
        elaboration_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bconsider\b', r'\bimagine\b',
            r'\blet\'s say\b', r'\bsuppose\b', r'\bthink of\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bput simply\b', r'\bput differently\b', r'\bmeaning\b',
            r'\blike\b.*\bswitch\b', r'\bjust like\b', r'\bsimilar to\b',
            r'\banalog(?:y|ous)\b', r'\bmetaphor\b', r'\bcompare\b',
        ]
        elab_count = 0
        for pat in elaboration_patterns:
            elab_count += len(re.findall(pat, resp_lower))
        
        elab_density = elab_count / num_sentences
        score += min(elab_density * 10, 12)
        
        # ============================================================
        # 4. CONDITIONAL/HYPOTHETICAL REASONING
        # ============================================================
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\bwould\b', r'\bif\b.*\bcould\b',
            r'\bif\b.*\bshould\b', r'\bif\b.*\bmight\b',
            r'\bwhen\b.*\bthen\b', r'\bwhenever\b',
            r'\bassuming\b', r'\bgiven that\b', r'\bprovided that\b',
            r'\bin case\b', r'\bsuppose\b', r'\bwhat if\b',
            r'\bunless\b', r'\beven if\b', r'\bwhether\b',
        ]
        cond_count = 0
        for pat in conditional_patterns:
            # Check per sentence to avoid over-counting across sentence boundaries
            for sent in sentences:
                sent_lower = sent.lower()
                if re.search(pat, sent_lower):
                    cond_count += 1
        
        cond_density = cond_count / num_sentences
        score += min(cond_density * 8, 10)
        
        # ============================================================
        # 5. HEDGING & QUALIFICATION (nuanced reasoning)
        # ============================================================
        hedge_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bthat said\b', r'\bwhile\b',
            r'\bdespite\b', r'\byet\b', r'\bbut\b', r'\bstill\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\bprobably\b', r'\blikely\b',
            r'\bit depends\b', r'\bit\'s important to note\b',
            r'\bkeep in mind\b', r'\bremember\b', r'\bnote that\b',
            r'\bnot necessarily\b', r'\bnot always\b',
        ]
        hedge_count = 0
        for pat in hedge_patterns:
            hedge_count += len(re.findall(pat, resp_lower))
        
        hedge_density = hedge_count / num_sentences
        # Moderate hedging is good, excessive is not
        hedge_score = min(hedge_density * 5, 8)
        score += hedge_score
        
        # ============================================================
        # 6. STRUCTURAL DEPTH: Sentence complexity and variety
        # ============================================================
        # Measure clause density via commas and subordinating conjunctions per sentence
        clause_indicators = 0
        for sent in sentences:
            clause_indicators += sent.count(',')
            clause_indicators += sent.count(';')
            # Subordinating conjunctions within sentences
            sub_conj = re.findall(r'\b(which|that|who|whom|where|when|while|although|because|since|if|unless|whereas)\b', sent.lower())
            clause_indicators += len(sub_conj)
        
        clause_density = clause_indicators / num_sentences
        # Multi-clause sentences show reasoning chains
        score += min(clause_density * 2, 10)
        
        # ============================================================
        # 7. ACKNOWLEDGMENT & EMPATHY BEFORE REASONING
        # ============================================================
        # Check if response starts with acknowledgment (shows structured approach)
        first_sentence = sentences[0].lower() if sentences else ""
        ack_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bthat\'s\b.*\bunderstandable\b', r'\bit\'s\b.*\bunderstandable\b',
            r'\bi\'m sorry\b', r'\bsorry to hear\b',
            r'\bi can imagine\b', r'\bthat must\b', r'\bcompletely\b',
            r'\babsolutely\b', r'\bof course\b', r'\bgreat question\b',
        ]
        has_ack = any(re.search(pat, first_sentence) for pat in ack_patterns)
        if has_ack:
            score += 4
        
        # ============================================================
        # 8. EXPLICIT STRUCTURE SIGNALS (not just bullets, but labeled sections)
        # ============================================================
        # Numbered items with content
        numbered_items = re.findall(r'\b\d+[\.\)]\s+\w', response_clean)
        if len(numbered_items) >= 2:
            score += min(len(numbered_items) * 1.5, 8)
        
        # Colon-based explanations (label: explanation pattern)
        colon_explanations = re.findall(r'[A-Z][a-zA-Z\s]+:\s+[A-Z]', response_clean)
        score += min(len(colon_explanations) * 2, 6)
        
        # ============================================================
        # 9. RESPONSE SUBSTANTIVENESS
        # ============================================================
        # Very short responses can't show reasoning
        if num_words < 20:
            score *= 0.3
        elif num_words < 40:
            score *= 0.6
        elif num_words < 60:
            score *= 0.8
        
        # Bonus for moderate-to-long responses that maintain density
        if num_words > 80 and (causal_count + seq_count + elab_count) > 3:
            score += 3
        
        # ============================================================
        # 10. NEGATIVE SIGNALS: Opaque/dismissive patterns
        # ============================================================
        dismissive_patterns = [
            r'\bjust\b.*\bdo\b', r'\bjust\b.*\bget\b',
            r'\byou need to\b', r'\byou should\b.*\bjust\b',
            r'\bget yourself together\b', r'\bget over it\b',
            r'\bstop\b.*\bcomplaining\b', r'\bman up\b',
        ]
        dismissive_count = 0
        for pat in dismissive_patterns:
            dismissive_count += len(re.findall(pat, resp_lower))
        score -= dismissive_count * 2
        
        # Penalize very terse, directive-only responses
        imperative_starts = 0
        for sent in sentences:
            sent_words = sent.strip().split()
            if sent_words and sent_words[0].lower() in ['do', 'get', 'go', 'try', 'make', 'stop', 'take', 'buy']:
                imperative_starts += 1
        imperative_ratio = imperative_starts / num_sentences
        if imperative_ratio > 0.5:
            score -= 3
        
        # Penalize responses that assert without support
        # (short sentences with no connectives)
        unsupported_claims = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_any_reasoning = any(re.search(pat, sent_lower) for pat in 
                                   causal_patterns + elaboration_patterns + conditional_patterns[:5])
            if len(sent.split()) < 8 and not has_any_reasoning:
                unsupported_claims += 1
        unsupported_ratio = unsupported_claims / num_sentences
        if unsupported_ratio > 0.6:
            score -= 3
        
        # ============================================================
        # 11. COHERENCE: Topic words from query appearing in response
        # ============================================================
        query_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                      'their', 'them', 'will', 'would', 'could', 'should', 'about',
                      'which', 'when', 'where', 'what', 'there', 'here', 'some',
                      'more', 'very', 'just', 'also', 'into', 'over', 'such',
                      'than', 'other', 'each', 'only', 'does', 'being', 'made',
                      'after', 'before', 'most', 'much', 'many', 'well', 'back'}
        query_content = query_words - stop_words
        if query_content:
            response_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', resp_lower))
            overlap = len(query_content & response_words) / len(query_content)
            score += overlap * 5  # up to 5 points for relevance
        
        # ============================================================
        # FINAL SCALING: Map to 1-5 range
        # ============================================================
        # Raw score typically ranges from ~0 to ~70+
        # Normalize to 1-5
        
        # Clamp and scale
        score = max(score, 0)
        
        # Sigmoid-like mapping to 1-5
        import math
        # Use a logistic function centered around score=25
        normalized = 1 + 4 / (1 + math.exp(-0.08 * (score - 22)))
        
        # Round to 1 decimal
        final_score = round(max(1.0, min(5.0, normalized)), 2)
        
        return final_score
        
    except Exception as e:
        return 2.5  # neutral fallback