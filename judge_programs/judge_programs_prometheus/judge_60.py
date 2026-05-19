def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    a discourse-structure and pragmatic-markers approach.
    
    This variant focuses on:
    1. Pragmatic discourse markers (acknowledgment, empathy, transitions)
    2. Epistemic stance markers categorized by strength
    3. Response-query alignment via question detection and addressing
    4. Structural coherence signals (logical connectors, enumeration)
    5. Assertiveness calibration (detecting overconfident absolute claims)
    """
    try:
        if not query or not response:
            return 1.0
        
        if len(response.strip()) < 20:
            return 1.0
        
        import re
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        words = re.findall(r'[a-z]+(?:\'[a-z]+)?', resp_lower)
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        
        score = 5.0  # Start at midpoint
        
        # ============================================================
        # 1. EPISTEMIC STANCE MARKERS (graded by strength)
        # ============================================================
        
        # Appropriate uncertainty/hedging markers (good calibration)
        soft_hedges = [
            r'\bmight\b', r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b',
            r'\boften\b', r'\busually\b', r'\bin many cases\b',
        ]
        
        epistemic_markers = [
            r'\bresearch suggests\b', r'\bstudies indicate\b', r'\bevidence suggests\b',
            r'\bit appears\b', r'\bit seems\b', r'\blikely\b', r'\bunlikely\b',
            r'\bprobably\b', r'\bto some extent\b', r'\bin some cases\b',
            r'\bdepending on\b', r'\bcan vary\b', r'\bnot always\b',
            r'\bsome experts\b', r'\baccording to\b',
        ]
        
        nuanced_qualifiers = [
            r'\bhowever\b', r'\balthough\b', r'\bon the other hand\b',
            r'\bthat said\b', r'\bnevertheless\b', r'\bwhile\b.*\b(also|still|yet)\b',
            r'\bit\'s worth noting\b', r'\bkeep in mind\b', r'\bimportant to note\b',
            r'\bnot necessarily\b', r'\bnot always\b', r'\bit depends\b',
        ]
        
        soft_hedge_count = sum(1 for p in soft_hedges if re.search(p, resp_lower))
        epistemic_count = sum(1 for p in epistemic_markers if re.search(p, resp_lower))
        nuance_count = sum(1 for p in nuanced_qualifiers if re.search(p, resp_lower))
        
        # Reward calibrated language
        calibration_score = min(soft_hedge_count * 0.15, 0.6) + \
                           min(epistemic_count * 0.25, 0.75) + \
                           min(nuance_count * 0.2, 0.6)
        score += calibration_score
        
        # ============================================================
        # 2. OVERCONFIDENCE DETECTION (penalize absolute claims)
        # ============================================================
        
        absolute_markers = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\babsolutely\b',
            r'\bwithout a doubt\b', r'\bundeniably\b', r'\bcertainly\b',
            r'\bobviously\b', r'\bclearly\b', r'\beveryone knows\b',
            r'\bit is a fact\b', r'\bno question\b', r'\bguaranteed\b',
            r'\bwithout question\b', r'\bindisputably\b',
        ]
        
        absolute_count = sum(len(re.findall(p, resp_lower)) for p in absolute_markers)
        absolute_density = absolute_count / num_sentences
        
        # Penalize high density of absolutes
        if absolute_density > 0.5:
            score -= min(absolute_density * 1.0, 1.5)
        elif absolute_density > 0.2:
            score -= 0.3
        
        # ============================================================
        # 3. DISMISSIVE vs EMPATHETIC PRAGMATIC MARKERS
        # ============================================================
        
        # Check if query involves emotional/subjective content
        emotional_query_signals = [
            r'\bfeel\b', r'\bstruggl', r'\bfrustrat', r'\bsad\b', r'\bstress',
            r'\banxi', r'\bworr', r'\bupset\b', r'\bconfus', r'\bhelp\b',
            r'\badvice\b', r'\bcomfort\b', r'\bseeking\b', r'\bcope\b',
            r'\bdevast', r'\bheartbr', r'\bloneli', r'\bdespair',
            r'\bexhaust', r'\btired\b', r'\bdown\b', r'\bpain\b',
        ]
        is_emotional = sum(1 for p in emotional_query_signals if re.search(p, query_lower)) >= 2
        
        # Empathy markers
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bthat\'s understandable\b', r'\bcompletely understandable\b',
            r'\bit\'s okay\b', r'\bit\'s perfectly\b', r'\bit\'s natural\b',
            r'\bi\'m sorry\b', r'\bsorry to hear\b',
            r'\byour feelings\b', r'\byour experience\b',
            r'\bvalid\b', r'\bnormal to feel\b', r'\bnatural to\b',
            r'\bgive yourself\b', r'\bbe kind to yourself\b',
        ]
        
        empathy_count = sum(1 for p in empathy_patterns if re.search(p, resp_lower))
        
        # Dismissive markers
        dismissive_patterns = [
            r'\bjust\s+(get|do|move|stop|try|be)\b',
            r'\byou should be able to\b', r'\byou need to get\b',
            r'\bget over it\b', r'\bmove on\b', r'\bstop\s+(being|feeling)\b',
            r'\bit\'s not that\b', r'\bdon\'t let it\b',
            r'\byou\'re just\b', r'\bmaybe you\'re\s+not\b',
            r'\bjust a\b',
        ]
        
        dismissive_count = sum(1 for p in dismissive_patterns if re.search(p, resp_lower))
        
        if is_emotional:
            score += min(empathy_count * 0.35, 1.2)
            score -= dismissive_count * 0.4
        
        # ============================================================
        # 4. STRUCTURAL COHERENCE & ELABORATION SIGNALS
        # ============================================================
        
        # Logical connectors showing structured thinking
        logical_connectors = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bfinally\b', r'\badditionally\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\bin addition\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bthis means\b', r'\bin other words\b',
            r'\btherefore\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bbecause\b', r'\bsince\b', r'\bdue to\b',
        ]
        
        connector_count = sum(1 for p in logical_connectors if re.search(p, resp_lower))
        connector_density = connector_count / num_sentences
        
        # Reward moderate connector usage (shows structured reasoning)
        if 0.15 <= connector_density <= 0.8:
            score += min(connector_count * 0.1, 0.6)
        
        # Numbered or structured lists
        has_numbered_list = bool(re.search(r'\b\d+[\.\)]\s+\w', response))
        if has_numbered_list:
            score += 0.3
        
        # ============================================================
        # 5. RESPONSE ADEQUACY & ENGAGEMENT
        # ============================================================
        
        # Check if response addresses the query topic
        query_content_words = set(re.findall(r'[a-z]{4,}', query_lower))
        stop_words = {'this', 'that', 'with', 'from', 'they', 'them', 'their',
                      'have', 'been', 'were', 'will', 'would', 'could', 'should',
                      'about', 'which', 'there', 'where', 'when', 'what', 'some',
                      'more', 'very', 'also', 'just', 'than', 'then', 'into',
                      'each', 'make', 'like', 'over', 'such', 'take', 'only',
                      'come', 'made', 'after', 'being', 'does', 'need', 'person',
                      'individual', 'must', 'during'}
        query_content_words -= stop_words
        
        if query_content_words:
            resp_word_set = set(words)
            topic_overlap = len(query_content_words & resp_word_set) / max(len(query_content_words), 1)
            score += min(topic_overlap * 1.5, 0.8)
        
        # ============================================================
        # 6. CONDITIONAL & CONTEXTUAL REASONING
        # ============================================================
        
        # Conditional structures show calibrated thinking
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\b(you|they|it|we)\b',
            r'\bwhen\b.*\b(you|they|it|we)\b',
            r'\bin case\b', r'\bassuming\b', r'\bprovided that\b',
            r'\bdepends on\b', r'\bvaries\b', r'\bcontext\b',
        ]
        
        conditional_count = sum(1 for p in conditional_patterns if re.search(p, resp_lower))
        score += min(conditional_count * 0.15, 0.5)
        
        # ============================================================
        # 7. QUESTION AWARENESS (acknowledging ambiguity)
        # ============================================================
        
        # Does the query contain ambiguity that should be acknowledged?
        query_has_ambiguity = bool(re.search(r'\bambiguous\b|\bunclear\b|\bno.*context\b|\bno.*previous\b', query_lower))
        
        # Does response ask clarifying questions or acknowledge limitations?
        asks_clarification = bool(re.search(r'\?', response))
        acknowledges_limits = bool(re.search(
            r'\bwithout (further|more|additional)\b|\bi (don\'t|cannot|can\'t) (know|tell|determine)\b|'
            r'\bmore (information|details|context)\b|\bclarif',
            resp_lower
        ))
        
        if query_has_ambiguity:
            if acknowledges_limits:
                score += 0.8
            if asks_clarification:
                score += 0.5
        
        # ============================================================
        # 8. IMPERATIVE vs SUGGESTIVE TONE
        # ============================================================
        
        # Count imperative commands (potentially dismissive/overconfident)
        imperative_starts = 0
        suggestive_forms = 0
        
        for sent in sentences:
            sent_stripped = sent.strip().lower()
            # Imperatives: sentences starting with a verb
            if re.match(r'^(do|get|stop|go|make|take|try|be|keep|remember|don\'t|just)\b', sent_stripped):
                imperative_starts += 1
            # Suggestive forms
            if re.search(r'\b(you (might|could|may) want to|consider|it may help|one approach|you can try)\b', sent_stripped):
                suggestive_forms += 1
        
        imperative_ratio = imperative_starts / num_sentences
        
        # Penalize heavily imperative responses (lacks nuance)
        if imperative_ratio > 0.5:
            score -= 0.5
        
        # Reward suggestive framing
        score += min(suggestive_forms * 0.2, 0.5)
        
        # ============================================================
        # 9. RESPONSE DEPTH (not just length, but information density)
        # ============================================================
        
        # Unique content words (excluding very common words)
        content_words = [w for w in words if len(w) > 3 and w not in stop_words]
        unique_content = set(content_words)
        
        # Information density: unique content words per sentence
        info_density = len(unique_content) / num_sentences if num_sentences > 0 else 0
        
        if info_density > 8:
            score += 0.4
        elif info_density > 5:
            score += 0.2
        elif info_density < 3:
            score -= 0.3
        
        # Minimum response substance
        if num_words < 30:
            score -= 1.0
        elif num_words < 50:
            score -= 0.3
        
        # ============================================================
        # 10. SPECULATIVE-AS-FACT DETECTION
        # ============================================================
        
        # Patterns where speculative content is presented as definitive
        speculative_as_fact = [
            r'\b(will|is going to) (definitely|certainly|always|never)\b',
            r'\beveryone (knows|agrees|understands)\b',
            r'\bthe (only|best|right|correct) (way|answer|solution|approach)\b',
            r'\byou (must|have to|need to)\b(?!.*\b(understand|know|remember)\b)',
            r'\bthere is no (other|alternative|different)\b',
        ]
        
        spec_as_fact = sum(1 for p in speculative_as_fact if re.search(p, resp_lower))
        score -= spec_as_fact * 0.3
        
        # ============================================================
        # 11. METACOGNITIVE AWARENESS
        # ============================================================
        
        metacognitive = [
            r'\bit\'s (important|worth|helpful) to (note|consider|remember|understand)\b',
            r'\bkeep in mind\b', r'\bone thing to consider\b',
            r'\bthis (can|may|might) (vary|differ|depend)\b',
            r'\bthere are (many|several|various|different) (ways|approaches|perspectives)\b',
            r'\bfrom (one|another|a different) perspective\b',
        ]
        
        meta_count = sum(1 for p in metacognitive if re.search(p, resp_lower))
        score += min(meta_count * 0.25, 0.5)
        
        # ============================================================
        # FINAL SCORE NORMALIZATION
        # ============================================================
        
        # Clamp to 1-5 range
        score = max(1.0, min(5.0, score))
        
        # Round to one decimal
        score = round(score, 1)
        
        return score
        
    except Exception:
        return 3.0