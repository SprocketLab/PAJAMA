def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    sentence-level analysis of claim types, evidential reasoning patterns,
    and discourse structure markers.
    
    This variant focuses on:
    1. Sentence-level classification (factual vs speculative vs evidential)
    2. Source attribution and reasoning patterns
    3. Discourse connectives indicating nuanced thinking
    4. Proportional balance between confident and tentative claims
    5. Response engagement depth with query complexity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        resp_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        if len(resp_lower) < 10:
            return 1.0
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ---- Feature 1: Evidential reasoning markers (sentence-level) ----
        # Words/phrases that indicate the response is grounding claims in evidence
        evidential_markers = [
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bevidence\b', r'\bdata\b', r'\bfindings?\b',
            r'\bhistorically\b', r'\bempirically\b', r'\bstatistic(?:s|ally)\b',
            r'\bdocumented\b', r'\bobserved\b', r'\bmeasured\b',
            r'\bexperiment(?:s|al|ally)?\b', r'\banalysis\b', r'\bsurvey(?:s|ed)?\b',
            r'\bliterature\b', r'\bpeer[- ]review\b', r'\bpublished\b',
            r'\breported\b', r'\bdemonstrated\b',
        ]
        
        evidential_sentence_count = 0
        for sent in sentences:
            sent_l = sent.lower()
            for pat in evidential_markers:
                if re.search(pat, sent_l):
                    evidential_sentence_count += 1
                    break
        
        evidential_ratio = evidential_sentence_count / num_sentences
        evidential_score = min(evidential_ratio * 8, 4.0)  # cap at 4
        
        # ---- Feature 2: Epistemic stance markers per sentence ----
        # Classify sentences by epistemic stance
        certainty_markers = [
            r'\bdefinitely\b', r'\babsolutely\b', r'\bundoubtedly\b',
            r'\bwithout (?:a )?doubt\b', r'\bclearly\b', r'\bobviously\b',
            r'\bcertainly\b', r'\bunquestionably\b', r'\bindisputably\b',
            r'\bguaranteed\b', r'\bimpossible\b', r'\bnever\b.*\bwrong\b',
            r'\balways\b(?!.*\bnot\b)', r'\beveryone knows\b',
            r'\bno question\b', r'\bwithout exception\b',
            r'\bthe fact is\b', r'\bthe truth is\b',
        ]
        
        tentative_markers = [
            r'\bperhaps\b', r'\bpossibly\b', r'\bmight\b', r'\bcould\b',
            r'\bmay\b(?!\s+\d)', r'\btend(?:s)? to\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bin (?:many|some|most) cases\b', r'\bit depends\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bit\'s (?:possible|plausible)\b', r'\bto some (?:extent|degree)\b',
            r'\barguably\b', r'\bseems?\b', r'\bappears?\b',
            r'\blikely\b', r'\bunlikely\b', r'\bprobably\b',
            r'\bin general\b', r'\bfor the most part\b',
            r'\bi think\b', r'\bi believe\b', r'\bi\'d say\b',
            r'\bin my (?:experience|opinion|view)\b',
            r'\bif (?:i recall|i remember|memory serves)\b',
        ]
        
        certain_count = 0
        tentative_count = 0
        for sent in sentences:
            sent_l = sent.lower()
            is_certain = any(re.search(p, sent_l) for p in certainty_markers)
            is_tentative = any(re.search(p, sent_l) for p in tentative_markers)
            if is_certain:
                certain_count += 1
            if is_tentative:
                tentative_count += 1
        
        certain_ratio = certain_count / num_sentences
        tentative_ratio = tentative_count / num_sentences
        
        # We want some tentative language but not too much; penalize overconfidence
        # Ideal: moderate tentative ratio, low certainty ratio
        tentative_score = min(tentative_ratio * 6, 3.0)
        overconfidence_penalty = min(certain_ratio * 5, 3.0)
        
        # ---- Feature 3: Discourse connectives indicating nuanced reasoning ----
        nuance_connectives = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\balthough\b', r'\bwhile\b(?=.*,)', r'\bdespite\b',
            r'\bthat said\b', r'\bhaving said that\b', r'\bconversely\b',
            r'\bin contrast\b', r'\bnonetheless\b', r'\bstill\b,',
            r'\bbut\b(?=.*\b(?:also|still|yet)\b)', r'\byet\b',
            r'\bon balance\b', r'\boverall\b',
            r'\bthe tradeoff\b', r'\btrade-off\b',
            r'\bmore complex\b', r'\bnuanc(?:e|ed)\b',
            r'\bit\'s worth noting\b', r'\bworth mentioning\b',
            r'\bimportantly\b', r'\bnotably\b',
        ]
        
        nuance_count = sum(1 for pat in nuance_connectives if re.search(pat, resp_lower))
        nuance_score = min(nuance_count * 0.8, 3.5)
        
        # ---- Feature 4: Conditional and contextual framing ----
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bdepending on\b', r'\bin (?:this|that) case\b',
            r'\bunder (?:certain|some|these) (?:conditions|circumstances)\b',
            r'\bcontext\b', r'\bsituation\b', r'\bscenario\b',
            r'\bcase[- ]by[- ]case\b', r'\bvaries\b', r'\bvariation\b',
            r'\bdepends\b', r'\bcontingent\b', r'\brelative to\b',
            r'\bcompared to\b', r'\bwhereas\b',
            r'\bfor (?:example|instance)\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b',
        ]
        
        conditional_count = sum(1 for pat in conditional_patterns if re.search(pat, resp_lower))
        conditional_score = min(conditional_count * 0.6, 3.0)
        
        # ---- Feature 5: Source/authority attribution ----
        source_patterns = [
            r'\bsource\b', r'\bcite[sd]?\b', r'\bcitation\b',
            r'\breference\b', r'\bjournal\b', r'\barticle\b',
            r'\bbook\b', r'\bauthor\b', r'\bwriter\b',
            r'\bprofessor\b', r'\bexpert\b', r'\bspecialist\b',
            r'\buniversity\b', r'\binstitut(?:e|ion)\b',
            r'\b(?:19|20)\d{2}\b',  # year references
            r'\bu/\w+\b',  # reddit user references
            r'/r/\w+',  # subreddit references
            r'\*[A-Z].*?\*',  # italicized titles
        ]
        
        source_count = sum(1 for pat in source_patterns if re.search(pat, resp_lower))
        source_score = min(source_count * 0.5, 2.5)
        
        # ---- Feature 6: Response depth and engagement ----
        # Longer, more substantive responses tend to be better calibrated
        word_count = len(resp_lower.split())
        
        # Log-scaled length bonus (diminishing returns)
        length_score = min(math.log(max(word_count, 1) + 1) / math.log(500) * 3, 3.0)
        
        # ---- Feature 7: Explanation structure ----
        # Detecting explanatory patterns
        explanation_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b',
            r'\bthus\b', r'\bhence\b', r'\bas a result\b',
            r'\bthis means\b', r'\bthis suggests\b',
            r'\bthe reason\b', r'\bdue to\b',
            r'\bin other words\b', r'\bput (?:simply|differently)\b',
            r'\bto clarify\b', r'\bto be (?:clear|specific|precise)\b',
            r'\bspecifically\b', r'\bnamely\b',
        ]
        
        explanation_count = sum(1 for pat in explanation_patterns if re.search(pat, resp_lower))
        explanation_score = min(explanation_count * 0.6, 2.5)
        
        # ---- Feature 8: Acknowledging limitations or alternatives ----
        limitation_patterns = [
            r'\bi\'m not (?:sure|certain)\b', r'\bi don\'t know\b',
            r'\bthere(?:\'s| is) (?:no|limited) (?:consensus|agreement)\b',
            r'\bdebat(?:e|ed|able)\b', r'\bcontrovers(?:y|ial)\b',
            r'\balternative(?:ly|s)?\b', r'\banother (?:view|perspective|way)\b',
            r'\bother(?:s)? (?:argue|suggest|believe|think)\b',
            r'\bnot everyone\b', r'\bsome (?:people|argue|say|believe)\b',
            r'\byour mileage may vary\b', r'\bymmv\b',
            r'\bfwiw\b', r'\bfor what it\'s worth\b',
            r'\btake (?:this|it) with\b', r'\bgrain of salt\b',
            r'\bdisclaimer\b', r'\bcaveat\b',
            r'\bboth\b.*\band\b', r'\bon one hand\b',
        ]
        
        limitation_count = sum(1 for pat in limitation_patterns if re.search(pat, resp_lower))
        limitation_score = min(limitation_count * 1.0, 3.0)
        
        # ---- Feature 9: Query complexity assessment ----
        # More complex queries deserve more nuanced responses
        query_complexity_signals = [
            r'\?', r'\bhow\b', r'\bwhy\b', r'\bwhat\b',
            r'\bethic(?:s|al)\b', r'\bphilosoph\b', r'\bopinion\b',
            r'\badvice\b', r'\bexperience\b', r'\bimpact\b',
            r'\bdebate\b', r'\bargument\b', r'\bcontrovers\b',
        ]
        query_complexity = sum(1 for pat in query_complexity_signals if re.search(pat, query_lower))
        is_complex_query = query_complexity >= 3
        
        # For complex queries, reward nuance more; penalize overconfidence more
        complexity_multiplier = 1.2 if is_complex_query else 1.0
        
        # ---- Feature 10: Personal experience framing ----
        # Framing opinions as personal experience is good calibration
        personal_exp_patterns = [
            r'\bin my experience\b', r'\bpersonally\b', r'\bfrom my\b',
            r'\bi\'ve (?:found|seen|noticed)\b', r'\bwhen i\b',
            r'\bfor me\b', r'\bi would\b', r'\bi\'d recommend\b',
            r'\banecdot(?:e|al|ally)\b',
        ]
        personal_count = sum(1 for pat in personal_exp_patterns if re.search(pat, resp_lower))
        personal_score = min(personal_count * 0.7, 2.0)
        
        # ---- Feature 11: Specificity and detail ----
        # Specific details suggest knowledge rather than vague claims
        specificity_patterns = [
            r'\b\d+(?:\.\d+)?%\b',  # percentages
            r'\b\d+(?:,\d{3})+\b',  # large numbers
            r'\b(?:first|second|third|fourth|1\)|2\)|3\))\b',  # enumeration
            r'["""].*?["""]',  # quoted text
            r'\b(?:chapter|section|part|volume)\b',
            r'\*\*.*?\*\*',  # bold text (markdown)
            r'`.*?`',  # code
            r'\bhttps?://\b',  # URLs
        ]
        specificity_count = sum(1 for pat in specificity_patterns if re.search(pat, resp_lower))
        specificity_score = min(specificity_count * 0.5, 2.0)
        
        # ---- Composite Score ----
        raw_score = (
            evidential_score * 1.0 +
            tentative_score * complexity_multiplier +
            nuance_score * 1.2 +
            conditional_score * 1.0 +
            source_score * 0.8 +
            length_score * 1.5 +
            explanation_score * 1.0 +
            limitation_score * complexity_multiplier +
            personal_score * 0.8 +
            specificity_score * 0.6 -
            overconfidence_penalty * complexity_multiplier
        )
        
        # ---- Bonus: Multi-perspective or balanced response ----
        if re.search(r'\bboth\b', resp_lower) and re.search(r'\band\b', resp_lower):
            raw_score += 0.5
        
        # Bonus for acknowledging the question's complexity
        if re.search(r'\b(?:great|good|interesting|important|complex) question\b', resp_lower):
            raw_score += 0.3
        
        # ---- Penalty: Very short dismissive responses ----
        if word_count < 25:
            raw_score *= 0.5
        elif word_count < 50:
            raw_score *= 0.75
        
        # ---- Penalty: All-caps or aggressive tone ----
        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', response))
        if caps_words > 3:
            raw_score -= min(caps_words * 0.3, 2.0)
        
        # ---- Penalty: Empty platitudes without substance ----
        platitude_patterns = [
            r'\bthat\'s a (?:great|good) question\b',
            r'\bi\'m glad you asked\b',
            r'\blet me help\b',
        ]
        platitude_count = sum(1 for pat in platitude_patterns if re.search(pat, resp_lower))
        if platitude_count > 0 and word_count < 80:
            raw_score -= 1.0
        
        # Normalize to 0-10 range
        # Expected raw range roughly -3 to 25
        final_score = max(0.0, min(10.0, raw_score * 0.45 + 2.5))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0