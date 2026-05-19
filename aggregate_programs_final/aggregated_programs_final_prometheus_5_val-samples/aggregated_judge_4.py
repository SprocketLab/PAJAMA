def judging_function(query, response):
    try:
        import re, math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not isinstance(query, str):
            query = ""

        resp = response.strip()
        if len(resp) < 8:
            return 2.5
        rl = resp.lower()
        ql = query.lower()

        words = re.findall(r"[a-zA-Z']+", rl)
        n_words = len(words)
        if n_words < 3:
            return 2.5

        # ---------- Estimate syllables (approximate) ----------
        def syllables(w):
            w = w.lower()
            if len(w) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev = False
            for c in w:
                is_v = c in vowels
                if is_v and not prev:
                    count += 1
                prev = is_v
            if w.endswith('e') and count > 1:
                count -= 1
            return max(1, count)

        # ---------- Response register metrics ----------
        avg_word_len = sum(len(w) for w in words) / n_words
        total_syl = sum(syllables(w) for w in words)
        avg_syl = total_syl / n_words
        long_word_ratio = sum(1 for w in words if len(w) >= 8) / n_words
        complex_word_ratio = sum(1 for w in words if syllables(w) >= 3) / n_words

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)
        avg_sl = n_words / n_sents

        # Flesch-like readability (higher = easier)
        flesch = 206.835 - 1.015 * avg_sl - 84.6 * avg_syl

        # Jargon suffix density
        jargon_suffix_re = re.compile(
            r'\b\w*(tion|sion|ology|ography|metry|trophic|otic|ynth|chemic|lytic|'
            r'onomic|cyte|plasm|genesis|morphic|ological|otically|ically|istic)\b',
            re.IGNORECASE)
        jargon_count = len(jargon_suffix_re.findall(resp))
        jargon_density = jargon_count / max(n_words, 1)

        # Specific scientific lexicon
        sci_lex = re.findall(
            r'\b(biochemical|autotrophic|chlorophyll|chloroplast|photolysis|redox|'
            r'photons?|chromodynamic|hadron|quark|gluon|cuneiform|ziggurat|inertia|'
            r'momentum|VO2|anaerobic|aerobic|asymptotic|stochastic|paradigm|'
            r'empirical|epistemological|hermeneutic|phenomenological|ontological|'
            r'reminiscent|decennium|elucidation|cordially|esteemed|aforementioned|'
            r'glucose|chloroplasts|protocol|algorithm|inelastic|electromagnetic)\b',
            rl)
        sci_count = len(sci_lex)

        # Simple/concrete framing markers
        simple_markers_re = re.compile(
            r'\b(imagine|think of|picture|like a|just like|as if|it\'?s like|'
            r'pretend|consider this|for example|let\'?s say|kind of like|'
            r'similar to)\b', re.IGNORECASE)
        simple_markers = len(simple_markers_re.findall(resp))

        # Concrete physical objects (toy, ball, pizza, kitchen, etc.)
        concrete_re = re.compile(
            r'\b(toy|ball|kitchen|cooking|pizza|cake|sandwich|car|book|tree|'
            r'garden|coin|flip|pumpkin|apple|football|movie|friend|family|'
            r'house|street|home|simple|easy|stove|sun|sunshine|water|sugar)\b',
            re.IGNORECASE)
        concrete_count = len(concrete_re.findall(resp))

        # ---------- Query register signal ----------
        wants_simple_score = 0
        simple_signals = [
            (r'\bsimple\b', 2), (r'\bsimpler\b', 2), (r'\beasy(- |to)\b', 2),
            (r'\blayman', 3), (r'\bbeginner', 2), (r'\bnovice', 2),
            (r'\bplain (english|language)\b', 3), (r'\bwithout (jargon|technical)', 3),
            (r'\bnon[- ]?technical\b', 3), (r'\beasy to (understand|grasp)\b', 3),
            (r'\bnew to\b', 2), (r'\bfirst time\b', 2), (r'\blanguage barrier\b', 3),
            (r'\bforeigner\b', 3), (r'\bintermediate level\b', 2),
            (r'\bbasic understanding\b', 2), (r'\b\d+th grade\b', 3),
            (r'\bhigh school\b', 1), (r'\bkid', 2), (r'\byoung mind', 3),
            (r'\bstruggle.{0,20}(complex|terminol|jargon|hard)', 3),
            (r'\bcasual\b.*\b(tone|style|language|laid)\b', 2),
            (r'\bslang\b', 2), (r'\bsuccinct\b', 1), (r'\bconcise\b', 1),
            (r'\bclear and simple\b', 3), (r'\bsimple terms\b', 3),
            (r'\bsimple words\b', 3), (r'\bsimple way\b', 3),
        ]
        for pat, w in simple_signals:
            if re.search(pat, ql):
                wants_simple_score += w

        wants_expert_score = 0
        expert_signals = [
            (r'\bacademic\b', 2), (r'\bexpert audience\b', 3), (r'\bscholarly\b', 2),
            (r'\bresearcher\b', 2), (r'\bsymposium\b', 3),
            (r'\bcomprehensive review\b', 2), (r'\bspecialized\b', 2),
            (r'\bphd\b', 3), (r'\btechnical (terminology|jargon)\b', 3),
            (r'\bprofessional terminology\b', 3),
            (r'\b(bioinformatics|nuclear physics|genomic)\b', 2),
        ]
        for pat, w in expert_signals:
            if re.search(pat, ql):
                wants_expert_score += w

        # ---------- Compute match score ----------
        # Response "simplicity" composite (higher = simpler)
        resp_simplicity = 0.0
        # Flesch normalized: 90+ = very easy, 30- = very hard
        resp_simplicity += max(0, min(1, (flesch - 30) / 60))
        # short words bonus
        resp_simplicity += max(0, min(1, 1.0 - (avg_word_len - 4) / 4))
        # low jargon bonus
        resp_simplicity += max(0, min(1, 1.0 - jargon_density * 5))
        # analogies
        resp_simplicity += min(1.0, simple_markers * 0.25)
        # concrete words
        resp_simplicity += min(0.8, concrete_count * 0.1)
        # short sentences
        if avg_sl < 18:
            resp_simplicity += 0.5
        # penalize sci lex
        resp_simplicity -= min(1.5, sci_count * 0.3)

        resp_simplicity = resp_simplicity / 4.0  # normalize roughly

        # Response "expertise" composite
        resp_expertise = 0.0
        resp_expertise += min(1.0, jargon_density * 8)
        resp_expertise += min(1.0, sci_count * 0.3)
        resp_expertise += min(1.0, long_word_ratio * 3)
        resp_expertise += min(1.0, complex_word_ratio * 4)
        if avg_sl > 18:
            resp_expertise += 0.3
        resp_expertise = resp_expertise / 4.0

        # ---------- Score by match ----------
        score = 3.0  # baseline neutral

        if wants_simple_score >= 2:
            # Match: high simplicity wins
            score += (resp_simplicity * 4.0) - (resp_expertise * 3.5)
            # Bonus for explicit beginner-friendly construction
            if simple_markers >= 2 and sci_count <= 1:
                score += 0.5
            # Penalty for clear jargon dump
            if sci_count >= 3 or jargon_density > 0.08:
                score -= 0.8
        elif wants_expert_score >= 2:
            # Match: high expertise wins
            score += (resp_expertise * 3.0) - (resp_simplicity * 0.5)
            # Penalty for over-simplification with kid analogies
            if simple_markers >= 3 and sci_count == 0 and n_words < 100:
                score -= 0.6
        else:
            # Neutral query: prefer balanced
            # Reward moderate complexity (Flesch 40-70)
            if 40 <= flesch <= 75:
                score += 0.4
            elif flesch < 25 or flesch > 90:
                score -= 0.2
            # Reward content density slightly
            score += min(0.4, jargon_density * 2.5)

        # ---------- Universal quality signals (keep coverage) ----------
        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for',
                'on','with','at','by','from','it','this','that','and','or','but',
                'i','you','we','they','have','has','will','would','can','your','my'}
        q_content = {w for w in re.findall(r"[a-z']+", ql) if w not in STOP and len(w) > 3}
        r_content_set = set(words) - STOP
        if q_content:
            overlap = len(q_content & r_content_set) / len(q_content)
        else:
            overlap = 0.3
        score += overlap * 0.7

        if n_words < 15:
            score -= 0.5
        elif n_words > 500:
            score -= 0.2

        # Non-answer penalty
        if re.search(r"\b(don'?t have|i don'?t know|can'?t help|you'?ll figure it out)\b", rl):
            score -= 0.6
        # Generic platitudes
        plat = len(re.findall(
            r"\b(keep working|good luck|you got this|believe in yourself|"
            r"don'?t worry|stay positive)\b", rl))
        if plat >= 2:
            score -= 0.3

        # Dismissive
        dism = len(re.findall(
            r"\b(just (get over|move on|deal)|you'?re just not|maybe you'?re just|"
            r"suck it up|cut out for|lower your expectations)\b", rl))
        score -= dism * 0.6

        # ---------- Clamp ----------
        score = max(1.0, min(5.0, score))
        return round(score, 3)

    except Exception:
        return 3.0
