def judging_function(query, response):
    try:
        import re, math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not isinstance(query, str):
            query = ""

        resp = response.strip()
        ql = (query or "").lower()
        rl = resp.lower()
        if len(resp) < 8:
            return 1.0

        words = re.findall(r"[a-zA-Z']+", rl)
        n_words = len(words)
        if n_words < 3:
            return 1.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)
        avg_sl = n_words / n_sents

        STOP = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                'do','does','did','will','would','could','should','may','might','can','to',
                'of','in','for','on','with','at','by','from','as','into','through','and','but',
                'or','if','that','this','these','those','it','its','i','me','my','we','our',
                'you','your','he','she','they','them','their','him','her','what','which','who',
                'about','up','out','over','down','also','very','just','too','so','than','then',
                'here','there','when','where','why','how','all','any','some','no','not','more',
                'most','other','such','only','own','same','too','very','s','t','am','being'}

        content = [w for w in words if w not in STOP and len(w) > 2]
        n_content = len(content)

        # ---------- 1. Audience detection: does query ask for simple language? ----------
        simple_request_patterns = [
            r'\bsimple(r| terms| words| language| way| explanation)?\b',
            r'\blayman', r'\bnon[- ]?technical', r'\bwithout (jargon|technical|complex)',
            r'\bbeginner', r'\bnovice', r'\bfirst time\b', r'\bnew to\b',
            r'\beasy to (understand|grasp|follow)\b', r'\beasy[- ]friendly\b',
            r'\bplain (english|language|terms)\b', r'\bavoiding (scientific )?jargon\b',
            r'\bhigh school\b', r'\b\d+th grade\b', r'\bintermediate level\b',
            r'\bbasic understanding\b', r'\blanguage barrier\b', r'\bforeigner\b',
            r'\byoung (minds|students|learners)\b', r'\bkid(s|dies)?\b',
            r'\bcasual\b.*\b(language|tone|laid[- ]back|slang)\b',
            r'\bsuccinct\b', r'\bconcise\b', r'\bclear and simple\b',
            r'\bjust starting\b', r'\bstruggle.*\bcomplex\b', r'\bgrade\b',
        ]
        wants_simple = any(re.search(p, ql) for p in simple_request_patterns)

        # ---------- 2. Audience detection: does query ask for expert/academic register? ----------
        expert_request_patterns = [
            r'\b(academic|expert|professional|technical|scholarly) (audience|terminology|language|jargon)\b',
            r'\bphd\b', r'\bresearcher\b', r'\bscholarly\b', r'\bsymposium\b',
            r'\bcomprehensive review\b', r'\bdetailed (information|analysis)\b',
            r'\bspecializ\w+ in\b', r'\bbioinformatics\b', r'\bcomprehensive study\b',
            r'\bsocio-?political\b', r'\bexpert in\b',
        ]
        wants_expert = any(re.search(p, ql) for p in expert_request_patterns)

        # ---------- 3. Jargon / technical word counting ----------
        jargon_suffix_pat = re.compile(
            r'\b\w*(tion|sion|ology|ography|metry|tropic|trophic|otic|ynth|chemic|lytic|onomic|'
            r'cyte|plasm|chromat|emiology|ostasis|cardial|chondria|genesis|morphic)\w*\b',
            re.IGNORECASE)
        long_words = [w for w in content if len(w) >= 9]
        tech_suffixes = jargon_suffix_pat.findall(resp)
        jargon_density = (len(long_words) + len(tech_suffixes) * 0.5) / max(n_content, 1)

        # Latinate / scientific lexicon
        sci_terms = re.findall(
            r'\b(biochemical|autotrophic|chlorophyll|chloroplast|photolysis|redox|'
            r'quantum|qubit|chromodynamic|hadron|gluon|quark|cuneiform|ziggurat|'
            r'inertia|momentum|VO2|anaerobic|aerobic|lactic|protocol|algorithm|'
            r'inelastic|asymptotic|stochastic|paradigm|empirical|epistemological|'
            r'eschatolog|hermeneutic|phenomenological|ontological|electromagnetic|'
            r'electromechanical|reminiscent|decennium|elucidation|residential|'
            r'glucose|chloroplasts|photons)\b', rl)
        n_sci = len(sci_terms)

        # ---------- 4. Simplicity markers (good when wants_simple) ----------
        simple_markers = [
            r'\b(think of|imagine|like a|as if|picture this)\b',
            r'\b(for example|for instance|such as|like when)\b',
            r'\bit\'?s like\b', r'\bjust like\b',
            r'\b(simple|easy|basic|plain)\b',
        ]
        simplicity = sum(1 for p in simple_markers if re.search(p, rl))

        # ---------- 5. Query-relevance via content overlap (unigram + bigram) ----------
        q_words = set(re.findall(r"[a-zA-Z']+", ql)) - STOP
        q_content = {w for w in q_words if len(w) > 2}
        r_content_set = set(content)
        if q_content:
            uni_overlap = len(q_content & r_content_set) / max(len(q_content), 1)
        else:
            uni_overlap = 0.4

        def bigrams(toks):
            return set(zip(toks[:-1], toks[1:]))
        qtoks = [w for w in re.findall(r"[a-zA-Z']+", ql) if w not in STOP]
        rtoks = [w for w in words if w not in STOP]
        qbg, rbg = bigrams(qtoks), bigrams(rtoks)
        bg_overlap = len(qbg & rbg) / max(len(qbg), 1) if qbg else 0.0

        relevance = min(1.0, uni_overlap * 0.7 + bg_overlap * 1.2)

        # ---------- 6. Length adequacy ----------
        if n_words < 15:
            length_score = 0.25
        elif n_words < 35:
            length_score = 0.45 + (n_words - 15) * 0.015
        elif n_words <= 250:
            length_score = 0.9
        elif n_words <= 400:
            length_score = 0.85
        else:
            length_score = 0.7

        # ---------- 7. Structure ----------
        has_numbered = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', resp))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-•*]\s', resp))
        has_para = '\n\n' in resp
        has_colon_h = bool(re.search(r'\n[A-Z][^\n]{0,40}:\s', resp))
        structure = 0.4 + 0.25*has_numbered + 0.15*has_bullets + 0.1*has_para + 0.1*has_colon_h
        structure = min(1.0, structure)

        # ---------- 8. Sentence-length quality ----------
        if 9 <= avg_sl <= 24:
            sl_score = 1.0
        elif avg_sl < 9:
            sl_score = max(0.35, avg_sl / 9)
        else:
            sl_score = max(0.3, 1.0 - (avg_sl - 24) * 0.04)

        # ---------- 9. Filler / vagueness ----------
        filler_re = re.compile(
            r'\b(basically|literally|honestly|just|kinda|sort of|kind of|stuff|things|'
            r'and stuff|or something|whatever|i guess|i think|you know|i mean|'
            r'pretty much|more or less)\b', re.IGNORECASE)
        filler_count = len(filler_re.findall(resp))
        filler_pen = min(0.4, filler_count * 0.04)

        # ---------- 10. Non-answer / unhelpful patterns ----------
        nonanswer_pat = [
            r"\bdon'?t have (the |any |specific )?(information|recommendations?|details)\b",
            r"\bi don'?t know\b",
            r"\bcan'?t (provide|help|give|tell)\b",
            r"\byou'?ll figure (it|this) out\b",
            r"\bgood luck\b\s*$",
            r"\bkeep (trying|working)\b",
            r"\byou'?ll get there\b",
            r"\bjust remember\b",
        ]
        nonanswer = sum(1 for p in nonanswer_pat if re.search(p, rl))

        # ---------- 11. Compute composite ----------
        score = 0.0
        score += relevance * 2.2
        score += length_score * 1.0
        score += structure * 0.9
        score += sl_score * 0.8
        score -= filler_pen * 1.5

        # information density
        content_ratio = n_content / max(n_words, 1)
        score += min(1.0, content_ratio * 2.2) * 0.7

        # AUDIENCE GATE - the key denoising mechanism
        if wants_simple:
            # Penalize jargon heavily; reward simplicity
            score -= min(2.5, jargon_density * 6.0 + n_sci * 0.35)
            score += min(1.5, simplicity * 0.35)
            # short sentences are GOOD here
            if avg_sl < 16:
                score += 0.4
            # explicit analogy reward
            if re.search(r'\b(imagine|think of|like a|picture|just like)\b', rl):
                score += 0.4
        elif wants_expert:
            # Reward technical vocabulary, penalize over-simplification
            score += min(1.2, jargon_density * 4.0 + n_sci * 0.2)
            # but penalize childish analogies in expert contexts
            if re.search(r"\b(like a kitchen|like a game|like a friend|imagine)\b", rl) and n_words < 80:
                score -= 0.4
        else:
            # neutral - mild reward for specificity
            score += min(0.7, jargon_density * 2.0 + n_sci * 0.1)

        # Non-answer penalty
        score -= nonanswer * 0.5

        # very short response in long-query context
        if len(ql) > 200 and n_words < 25:
            score -= 0.6

        # Empathy bonus when query is emotional
        emo_q = bool(re.search(
            r'\b(stress|anxious|overwhelm|frustrat|sad|lonely|heartbroken|devastated|'
            r'cry|crying|sleepless|struggl|exhaust|upset|discouraged|disappointment|'
            r'feeling|emotional)\b', ql))
        if emo_q:
            empathy = len(re.findall(
                r"\b(i understand|i hear|i can see|i'?m sorry|it'?s okay|"
                r"completely understandable|natural to|not alone|your feelings)\b", rl))
            score += min(0.8, empathy * 0.3)
            # dismissive penalty
            dism = len(re.findall(
                r"\b(get over it|move on|just (deal|try|keep)|"
                r"you'?re just not|maybe you'?re just|stop (being|feeling)|"
                r"suck it up|cut out for)\b", rl))
            score -= dism * 0.7

        # Normalize to 1-5
        # Typical raw range: roughly -2 to 7
        final = 3.0 + score * 0.4
        final = max(1.0, min(5.0, final))
        return round(final, 3)

    except Exception:
        return 3.0
