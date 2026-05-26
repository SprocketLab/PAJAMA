def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using a
    sentence-level analysis approach. Analyzes each sentence individually for
    claim strength, hedging, evidence attribution, and epistemic markers,
    then computes a composite score based on the distribution of sentence types.
    
    This variant uses a fundamentally different approach: sentence-level classification
    into epistemic categories, plus analysis of claim-to-evidence ratios and
    discourse-level patterns of uncertainty acknowledgment.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 1.0
        
        # Split into sentences using multiple delimiters
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 2.0
        
        # ---- SENTENCE-LEVEL CLASSIFICATION ----
        # Classify each sentence into epistemic categories
        
        # Category 1: Evidential sentences (cite sources, studies, data)
        evidential_patterns = [
            r'\bresearch\b', r'\bstud(?:y|ies)\b', r'\bdata\b', r'\bevidence\b',
            r'\baccording to\b', r'\bfindings?\b', r'\bliterature\b', r'\bsurvey\b',
            r'\bexpert[s]?\b', r'\bscientist[s]?\b', r'\bscholar[s]?\b',
            r'\bpublished\b', r'\breported\b', r'\bdocumented\b', r'\banalysis\b',
            r'\bstatistic(?:s|al)\b', r'\bmeta-analysis\b', r'\bpeer[- ]review',
            r'\bjournal\b', r'\bcited\b', r'\breference\b'
        ]
        
        # Category 2: Hedged/qualified sentences
        hedging_patterns = [
            r'\bmight\b', r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bprobably\b', r'\blikely\b', r'\bunlikely\b', r'\btend[s]? to\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bsometimes\b', r'\bin some cases\b', r'\bit depends\b',
            r'\bnot necessarily\b', r'\bnot always\b', r'\bto some (?:extent|degree)\b',
            r'\bappear[s]? to\b', r'\bseem[s]? to\b', r'\bsuggest[s]?\b',
            r'\bimpl(?:y|ies)\b', r'\bindicat(?:e[s]?|ing)\b', r'\bmay\b',
            r'\broughly\b', r'\bapproximately\b', r'\bon average\b',
            r'\bit\'s (?:possible|plausible)\b', r'\bone could argue\b',
            r'\barguably\b', r'\bin general\b', r'\bfor the most part\b'
        ]
        
        # Category 3: Absolute/overconfident sentences
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\babsolutely\b', r'\bundoubtedly\b', r'\bwithout (?:a )?doubt\b',
            r'\bobviously\b', r'\bclearly\b', r'\bof course\b',
            r'\beveryone knows\b', r'\bit is (?:a )?fact\b', r'\bundeniab(?:le|ly)\b',
            r'\bno question\b', r'\bguarantee[ds]?\b', r'\bimpossible\b',
            r'\bproven\b', r'\bthe truth is\b', r'\bwithout exception\b',
            r'\bno doubt\b', r'\bincontrovertib(?:le|ly)\b'
        ]
        
        # Category 4: Perspective-acknowledging sentences
        perspective_patterns = [
            r'\bsome (?:people|argue|believe|think|say|scholars|experts)\b',
            r'\bon (?:the )?one hand\b', r'\bon the other hand\b',
            r'\balternative(?:ly)?\b', r'\bhowever\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bconversely\b', r'\bin contrast\b',
            r'\bdebat(?:e[ds]?|able)\b', r'\bcontrovers(?:y|ial)\b',
            r'\bdisagree(?:ment)?\b', r'\bperspective\b', r'\bviewpoint\b',
            r'\bopinion\b', r'\binterpretation\b', r'\bdepending on\b',
            r'\bfrom (?:a|one|another) (?:perspective|viewpoint|angle)\b',
            r'\bwhile (?:some|others)\b', r'\bcriticism\b', r'\bcritique\b'
        ]
        
        # Category 5: Self-awareness / epistemic humility markers
        humility_patterns = [
            r'\bI\'m not (?:sure|certain)\b', r'\bI don\'t know\b',
            r'\bI believe\b', r'\bI think\b', r'\bin my (?:experience|opinion|view)\b',
            r'\bas far as I (?:know|understand)\b', r'\bcorrect me if\b',
            r'\bI could be wrong\b', r'\bthis is (?:just )?my\b',
            r'\bfrom what I (?:understand|recall|know)\b',
            r'\bif I (?:remember|recall) correctly\b',
            r'\btake this with\b', r'\bgrain of salt\b',
            r'\bnot an expert\b', r'\bmy understanding\b',
            r'\bas I understand it\b', r'\bto my knowledge\b'
        ]
        
        # Category 6: Conditional/nuanced sentences
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bin the case (?:of|that|where)\b', r'\bwhen\b.*\b(?:but|however)\b',
            r'\bit depends on\b', r'\bcontext\b', r'\bnuance[ds]?\b',
            r'\bcomplex(?:ity)?\b', r'\bmultifaceted\b', r'\bvaries\b',
            r'\brange[s]? from\b', r'\bspectrum\b', r'\bdepends\b',
            r'\bcase[- ]by[- ]case\b', r'\bsituation\b'
        ]
        
        def count_pattern_matches(text, patterns):
            text_lower = text.lower()
            count = 0
            for p in patterns:
                count += len(re.findall(p, text_lower))
            return count
        
        # Classify each sentence
        n_sentences = len(sentences)
        evidential_count = 0
        hedged_count = 0
        absolute_count = 0
        perspective_count = 0
        humility_count = 0
        conditional_count = 0
        bare_assertion_count = 0  # sentences with none of the above
        
        sentence_scores = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            ev = count_pattern_matches(sent, evidential_patterns)
            hd = count_pattern_matches(sent, hedging_patterns)
            ab = count_pattern_matches(sent, absolute_patterns)
            pr = count_pattern_matches(sent, perspective_patterns)
            hu = count_pattern_matches(sent, humility_patterns)
            co = count_pattern_matches(sent, conditional_patterns)
            
            # Sentence-level score: positive for good epistemic markers, negative for overconfidence
            sent_score = (ev * 2.0 + hd * 1.5 + pr * 1.8 + hu * 1.5 + co * 1.2 - ab * 2.5)
            sentence_scores.append(sent_score)
            
            if ev > 0: evidential_count += 1
            if hd > 0: hedged_count += 1
            if ab > 0: absolute_count += 1
            if pr > 0: perspective_count += 1
            if hu > 0: humility_count += 1
            if co > 0: conditional_count += 1
            if ev == 0 and hd == 0 and ab == 0 and pr == 0 and hu == 0 and co == 0:
                bare_assertion_count += 1
        
        # ---- COMPUTE COMPONENT SCORES ----
        
        # 1. Epistemic richness: proportion of sentences with any epistemic marker
        marked_ratio = 1.0 - (bare_assertion_count / max(n_sentences, 1))
        # Moderate marking is ideal (not everything needs hedging)
        # Optimal around 0.3-0.6
        if marked_ratio < 0.1:
            richness_score = 0.1
        elif marked_ratio <= 0.6:
            richness_score = marked_ratio * 1.5
        else:
            richness_score = 0.9 - (marked_ratio - 0.6) * 0.3  # slight penalty for over-hedging
        
        # 2. Overconfidence penalty
        absolute_ratio = absolute_count / max(n_sentences, 1)
        overconfidence_penalty = absolute_ratio * 3.0  # strong penalty
        
        # 3. Hedging quality score
        hedge_ratio = hedged_count / max(n_sentences, 1)
        # Sweet spot: some hedging but not excessive
        if hedge_ratio == 0:
            hedge_score = 0.0
        elif hedge_ratio <= 0.4:
            hedge_score = hedge_ratio * 2.0
        else:
            hedge_score = 0.8 - (hedge_ratio - 0.4) * 0.5
        
        # 4. Evidence attribution score
        evidence_ratio = evidential_count / max(n_sentences, 1)
        evidence_score = min(evidence_ratio * 3.0, 1.0)
        
        # 5. Perspective diversity score
        perspective_ratio = perspective_count / max(n_sentences, 1)
        perspective_score = min(perspective_ratio * 3.5, 1.0)
        
        # 6. Epistemic humility score
        humility_ratio = humility_count / max(n_sentences, 1)
        humility_score = min(humility_ratio * 4.0, 1.0)
        
        # 7. Conditional reasoning score
        conditional_ratio = conditional_count / max(n_sentences, 1)
        conditional_score = min(conditional_ratio * 3.0, 1.0)
        
        # ---- QUERY ANALYSIS ----
        # Determine if the query is about a factual, subjective, or ambiguous topic
        query_lower = query.lower()
        
        # Subjective/ambiguous queries demand MORE epistemic calibration
        subjective_indicators = [
            r'\bethic(?:s|al)\b', r'\bmoral\b', r'\bshould\b', r'\bwrong\b',
            r'\bright\b', r'\bbest\b', r'\bworst\b', r'\bopinion\b',
            r'\bthink\b', r'\bbelieve\b', r'\bfeel\b', r'\bwhat (?:do|does|would)\b',
            r'\bhow (?:do|does|would|should)\b', r'\bwhy\b', r'\bdebat\b',
            r'\bargum\b', r'\bcontrovers\b', r'\bphilosoph\b', r'\bpolitics\b',
            r'\bpolitical\b', r'\bimpact\b', r'\beffect\b', r'\bworth\b'
        ]
        
        subjectivity_level = sum(1 for p in subjective_indicators 
                                  if re.search(p, query_lower))
        is_subjective = subjectivity_level >= 2
        
        # ---- RESPONSE QUALITY INDICATORS ----
        
        # Length component (longer responses tend to be more detailed)
        word_count = len(response_text.split())
        if word_count < 20:
            length_factor = 0.4
        elif word_count < 50:
            length_factor = 0.6
        elif word_count < 100:
            length_factor = 0.8
        elif word_count < 300:
            length_factor = 1.0
        else:
            length_factor = 1.0 + min((word_count - 300) / 1000, 0.2)
        
        # Structural sophistication: variety in sentence types used
        categories_present = sum(1 for x in [evidential_count, hedged_count, 
                                              perspective_count, humility_count,
                                              conditional_count] if x > 0)
        diversity_bonus = categories_present * 0.15
        
        # ---- DISTRIBUTION ANALYSIS OF SENTENCE SCORES ----
        # Look at the variance and mean of per-sentence epistemic scores
        if sentence_scores:
            mean_sent_score = sum(sentence_scores) / len(sentence_scores)
            variance = sum((s - mean_sent_score) ** 2 for s in sentence_scores) / len(sentence_scores)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            
            # Moderate variance is good (shows differentiation between fact/speculation)
            if std_dev > 0:
                distribution_score = min(std_dev * 0.3, 0.5)
            else:
                distribution_score = 0.0
            
            # Mean score contribution
            mean_contribution = max(min(mean_sent_score * 0.2, 1.0), -0.5)
        else:
            distribution_score = 0.0
            mean_contribution = 0.0
        
        # ---- CLAIM DENSITY ANALYSIS ----
        # Count declarative statements vs qualified ones
        declarative_markers = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bwill\b',
            r'\bhas\b', r'\bhave\b', r'\bhad\b'
        ]
        resp_lower = response_text.lower()
        total_declarative = sum(len(re.findall(p, resp_lower)) for p in declarative_markers)
        
        # High declarative density without hedging = potentially overconfident
        declarative_density = total_declarative / max(word_count, 1)
        total_hedging = hedged_count + humility_count + conditional_count + perspective_count
        
        if total_hedging == 0 and declarative_density > 0.08:
            claim_density_penalty = min((declarative_density - 0.08) * 5.0, 0.5)
        else:
            claim_density_penalty = 0.0
        
        # ---- COMPOSITE SCORE ----
        
        base_score = 3.0  # Start at middle
        
        # Positive contributions
        base_score += richness_score * 1.2
        base_score += hedge_score * 1.0
        base_score += evidence_score * 1.0
        base_score += perspective_score * 0.8
        base_score += humility_score * 0.7
        base_score += conditional_score * 0.6
        base_score += diversity_bonus
        base_score += distribution_score
        base_score += mean_contribution
        
        # Penalties
        base_score -= overconfidence_penalty
        base_score -= claim_density_penalty
        
        # Adjustments for subjective queries
        if is_subjective:
            # For subjective queries, epistemic markers matter more
            if total_hedging == 0 and absolute_count == 0:
                base_score -= 0.5  # No epistemic awareness on a subjective topic
            if perspective_count > 0:
                base_score += 0.3  # Bonus for acknowledging perspectives
        
        # Length factor as a multiplier (subtle)
        base_score *= (0.7 + 0.3 * min(length_factor, 1.2))
        
        # Ensure score is in [0, 10]
        final_score = max(0.0, min(10.0, base_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 3.0