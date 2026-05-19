def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a STRUCTURAL approach: analyzing sentence-level epistemic patterns,
    claim density, source attribution, conditional reasoning, and the ratio of 
    qualified vs unqualified assertions. Unlike word-overlap or simple marker counting,
    this analyzes the grammatical and rhetorical structure of claims.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 3.0
        
        resp_clean = response.strip()
        query_clean = query.strip()
        
        # Tokenize into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', resp_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        words = resp_clean.lower().split()
        word_count = len(words)
        
        if word_count < 2:
            return 0.5
        
        # ============================================================
        # FEATURE 1: Claim Structure Analysis
        # Analyze each sentence for whether it makes qualified or 
        # unqualified assertions. Look at sentence beginnings and 
        # verb patterns.
        # ============================================================
        
        # Patterns indicating qualified/hedged claims
        qualification_patterns = [
            r'\b(it\s+(?:is|seems?|appears?)\s+(?:likely|possible|plausible|probable|unlikely|uncertain))',
            r'\b((?:may|might|could)\s+(?:be|have|indicate|suggest|represent))',
            r'\b((?:some|many|most|certain|several)\s+(?:researchers|scholars|experts|scientists|studies|sources))',
            r'\b(according\s+to)',
            r'\b(research\s+(?:suggests?|indicates?|shows?|has\s+shown))',
            r'\b(evidence\s+(?:suggests?|indicates?|points?\s+to))',
            r'\b((?:one|another)\s+(?:possibility|interpretation|explanation|theory|view))',
            r'\b((?:tends?\s+to|generally|typically|usually|often|sometimes))',
            r'\b(in\s+(?:some|many|most)\s+cases)',
            r'\b(it\s+(?:depends|varies))',
            r'\b(not\s+(?:entirely|completely|necessarily|always)\s+clear)',
            r'\b((?:while|although|though|whereas)\s+)',
            r'\b(on\s+(?:the\s+)?one\s+hand)',
            r'\b(there\s+(?:is|are)\s+(?:some|limited|growing|mixed)\s+evidence)',
            r'\b(as\s+far\s+as\s+(?:we|I)\s+know)',
            r'\b(to\s+(?:my|the\s+best\s+of\s+(?:my|our))\s+knowledge)',
        ]
        
        # Patterns indicating absolute/unqualified assertions
        absolute_patterns = [
            r'\b((?:it\s+)?is\s+(?:definitely|certainly|absolutely|undoubtedly|unquestionably|obviously|clearly)\b)',
            r'\b(there\s+is\s+no\s+(?:doubt|question))',
            r'\b((?:always|never|every|none|all)\s+(?:are|is|do|does|will|have|has))',
            r'\b(without\s+(?:a\s+)?doubt)',
            r'\b((?:the\s+)?(?:fact|truth)\s+(?:is|remains))',
            r'\b(everyone\s+(?:knows?|agrees?))',
            r'\b(it\s+is\s+(?:a\s+)?(?:well-known|established)\s+fact)',
        ]
        
        resp_lower = resp_clean.lower()
        
        qualification_count = 0
        for pat in qualification_patterns:
            qualification_count += len(re.findall(pat, resp_lower))
        
        absolute_count = 0
        for pat in absolute_patterns:
            absolute_count += len(re.findall(pat, resp_lower))
        
        # ============================================================
        # FEATURE 2: Conditional and Nuanced Reasoning
        # Look for if-then structures, caveats, exceptions, and
        # multi-perspective presentation
        # ============================================================
        
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided\s+that\b',
            r'\bassuming\b', r'\bin\s+the\s+(?:event|case)\b',
            r'\bdepending\s+on\b', r'\bcontingent\b',
        ]
        
        nuance_markers = [
            r'\bhowever\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bon\s+the\s+other\s+hand\b', r'\bthat\s+said\b',
            r'\bat\s+the\s+same\s+time\b', r'\bconversely\b',
            r'\bin\s+contrast\b', r'\balternatively\b',
            r'\bbut\b', r'\byet\b', r'\bstill\b',
        ]
        
        perspective_markers = [
            r'\bsome\s+(?:people|argue|believe|think|say)\b',
            r'\bothers\s+(?:argue|believe|think|say|contend|maintain)\b',
            r'\bcritics\b', r'\bproponents\b', r'\bsupporters\b',
            r'\bdebate\b', r'\bcontrovers(?:y|ial)\b',
            r'\bdisagree(?:ment)?\b', r'\bperspective\b',
            r'\bviewpoint\b', r'\bpoint\s+of\s+view\b',
        ]
        
        conditional_count = sum(len(re.findall(pat, resp_lower)) for pat in conditional_markers)
        nuance_count = sum(len(re.findall(pat, resp_lower)) for pat in nuance_markers)
        perspective_count = sum(len(re.findall(pat, resp_lower)) for pat in perspective_markers)
        
        # ============================================================
        # FEATURE 3: Source Attribution and Evidence References
        # ============================================================
        
        source_patterns = [
            r'\baccording\s+to\b', r'\bcited\b', r'\bsource\b',
            r'\breference\b', r'\bstudy\b', r'\bstudies\b',
            r'\bjournal\b', r'\bpublished\b', r'\breport(?:ed|s)?\b',
            r'\bsurvey\b', r'\bdata\b', r'\bstatistic(?:s|al)?\b',
            r'\bfinding(?:s)?\b', r'\banalysis\b', r'\bexpert(?:s)?\b',
        ]
        
        source_count = sum(len(re.findall(pat, resp_lower)) for pat in source_patterns)
        
        # ============================================================
        # FEATURE 4: Epistemic Verb Analysis
        # Count verbs that indicate different levels of certainty
        # ============================================================
        
        # High certainty verbs (neutral - these are fine for established facts)
        factual_verbs = re.findall(r'\b(?:is|are|was|were|has|have|does|do)\b', resp_lower)
        
        # Epistemic verbs (showing calibrated uncertainty)
        epistemic_verbs = re.findall(
            r'\b(?:seems?|appears?|suggests?|indicates?|implies?|'
            r'believes?|thinks?|considers?|estimates?|assumes?|'
            r'supposes?|suspects?|speculates?|hypothesizes?|'
            r'proposes?|argues?|contends?|maintains?|claims?)\b', 
            resp_lower
        )
        
        # Modal verbs showing uncertainty
        uncertainty_modals = re.findall(
            r'\b(?:may|might|could|would|should|can)\b', resp_lower
        )
        
        # Certainty modals
        certainty_modals = re.findall(
            r'\b(?:must|shall|will|definitely|certainly)\b', resp_lower
        )
        
        # ============================================================
        # FEATURE 5: Response Completeness and Coherence
        # ============================================================
        
        # Check for truncation
        is_truncated = resp_clean[-1] not in '.!?"\')' and word_count > 20
        
        # Check for repetition (sign of low quality)
        if word_count > 10:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
            repetition_ratio = repeated_bigrams / max(len(bigram_counts), 1)
        else:
            repetition_ratio = 0
        
        # Check for garbage/irrelevant content
        has_html = bool(re.search(r'<[a-z]+[^>]*>', resp_lower))
        has_code_artifacts = bool(re.search(r'(?:import\s+\w+|def\s+\w+|class\s+\w+)', resp_lower))
        has_prompt_leakage = bool(re.search(r'(?:input:|output:|question:|answer:)', resp_lower))
        
        # ============================================================
        # FEATURE 6: Query-Response Alignment
        # Assess if the response actually addresses the query
        # ============================================================
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_clean.lower()))
        resp_words = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
        
        # Remove very common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'some', 'them', 'than', 'its', 'over',
                     'such', 'that', 'this', 'with', 'will', 'each', 'from',
                     'they', 'what', 'which', 'their', 'said', 'about', 'would',
                     'make', 'like', 'just', 'into', 'also', 'could', 'more',
                     'these', 'other', 'there', 'where', 'when', 'how', 'who'}
        
        query_content = query_words - stopwords
        resp_content = resp_words - stopwords
        
        if query_content:
            relevance = len(query_content & resp_content) / len(query_content)
        else:
            relevance = 0.5
        
        # ============================================================
        # FEATURE 7: Sentence Complexity and Information Density
        # Well-calibrated responses tend to have moderate complexity
        # ============================================================
        
        avg_sentence_len = word_count / max(len(sentences), 1)
        
        # Unique word ratio (vocabulary richness)
        unique_ratio = len(set(words)) / max(word_count, 1)
        
        # ============================================================
        # SCORING: Combine features into a final score
        # ============================================================
        
        score = 5.0  # Base score
        
        # --- Epistemic calibration score ---
        # Reward qualification and hedging proportional to response length
        norm_factor = math.log(word_count + 1) / math.log(100)  # normalize by length
        norm_factor = max(norm_factor, 0.5)
        
        qualification_score = min(qualification_count / norm_factor, 3.0)
        score += qualification_score * 0.4
        
        # Reward epistemic verbs
        epistemic_score = min(len(epistemic_verbs) / norm_factor, 2.0)
        score += epistemic_score * 0.3
        
        # Reward uncertainty modals (moderately)
        modal_score = min(len(uncertainty_modals) / norm_factor, 2.0)
        score += modal_score * 0.2
        
        # Penalize excessive absolute claims
        if absolute_count > 0:
            absolute_penalty = min(absolute_count * 0.3, 1.5)
            score -= absolute_penalty
        
        # Penalize excessive certainty modals relative to uncertainty modals
        if len(certainty_modals) > len(uncertainty_modals) + len(epistemic_verbs):
            score -= 0.5
        
        # --- Nuance and reasoning score ---
        nuance_total = conditional_count + nuance_count + perspective_count
        nuance_score = min(nuance_total / norm_factor, 2.5)
        score += nuance_score * 0.3
        
        # --- Source attribution bonus ---
        source_score = min(source_count / norm_factor, 2.0)
        score += source_score * 0.2
        
        # --- Relevance ---
        score += relevance * 1.5
        
        # --- Response length and completeness ---
        # Very short responses are usually bad
        if word_count < 5:
            score -= 3.0
        elif word_count < 10:
            score -= 1.5
        elif word_count < 20:
            score -= 0.5
        
        # Moderate length bonus (diminishing returns)
        length_bonus = min(math.log(word_count + 1) / math.log(200), 1.0) * 1.0
        score += length_bonus
        
        # Truncation penalty
        if is_truncated:
            score -= 0.3
        
        # --- Quality penalties ---
        # Repetition
        if repetition_ratio > 0.1:
            score -= min(repetition_ratio * 5, 2.0)
        
        # HTML/code artifacts in non-code queries
        query_asks_code = bool(re.search(r'\b(?:code|html|program|script|function|tag)\b', query_clean.lower()))
        if not query_asks_code:
            if has_html:
                score -= 1.0
            if has_code_artifacts:
                score -= 1.0
        
        # Prompt leakage
        if has_prompt_leakage:
            # Count how many times
            leakage_count = len(re.findall(r'(?:Input:|Output:|Question:|Answer:)', resp_clean))
            if leakage_count > 2:
                score -= min(leakage_count * 0.3, 2.0)
        
        # --- Vocabulary richness bonus ---
        if word_count > 15:
            if unique_ratio > 0.7:
                score += 0.3
            elif unique_ratio < 0.3:
                score -= 0.5
        
        # --- Sentence structure bonus ---
        # Good responses have reasonable sentence lengths
        if 10 <= avg_sentence_len <= 25:
            score += 0.3
        elif avg_sentence_len > 40:
            score -= 0.3
        
        # --- Special case: response is just a single word or very terse ---
        if word_count <= 3 and len(sentences) <= 1:
            # Check if query expects a short answer
            query_expects_short = bool(re.search(
                r'\b(?:identify|name|list|which|what\s+is\s+the\s+(?:name|biggest|largest|smallest))\b',
                query_clean.lower()
            ))
            if query_expects_short:
                score = max(score, 4.0)  # Don't penalize too much
            else:
                score = min(score, 2.0)
        
        # --- Detect completely off-topic or nonsensical responses ---
        if word_count > 20 and relevance < 0.1:
            score -= 1.5
        
        # Clamp to range
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 3.0