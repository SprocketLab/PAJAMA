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
            return 1.0
        rl = resp.lower()
        ql = query.lower()

        words = re.findall(r"[a-zA-Z']+", rl)
        n_words = len(words)
        if n_words < 3:
            return 1.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)

        STOP = {'the','a','an','is','are','was','were','be','been','being','have','has',
                'had','do','does','did','will','would','could','should','may','might',
                'can','to','of','in','for','on','with','at','by','from','as','into',
                'and','but','or','if','that','this','it','its','i','me','my','we','our',
                'you','your','he','she','they','them','their','him','her','what','which',
                'who','about','also','very','just','too','so','than','then','here','there',
                'when','where','why','how','all','any','some','no','not','more','most',
                'other','such','only','own','same','very'}

        content = [w for w in words if w not in STOP and len(w) > 2]

        # ---------- 1. Sentence-level evidence scoring ----------
        num_re = re.compile(r'\b\d+[\d,\.]*\b')
        measure_re = re.compile(
            r'\b\d+\s*(%|percent|pounds?|lbs?|kg|oz|ounces?|cups?|tbsp|tsp|ml|'
            r'liters?|hours?|minutes?|seconds?|miles?|km|feet|inches?|cm|mm|'
            r'degrees?|years?|days?|weeks?|months?)\b', re.IGNORECASE)
        proper_re = re.compile(r'(?<![\.!\?]\s)(?<!^)\b[A-Z][a-z]{2,}\b')
        example_re = re.compile(r'\b(for example|for instance|such as|specifically|e\.g\.)\b', re.IGNORECASE)
        causal_re = re.compile(
            r'\b(because|therefore|thus|hence|since|due to|as a result|leads to|results in)\b',
            re.IGNORECASE)

        sent_scores = []
        for s in sentences:
            ws = s.split()
            if len(ws) < 2:
                continue
            sc = 0.0
            sc += min(2.5, len(num_re.findall(s)) * 0.6)
            sc += min(2.0, len(measure_re.findall(s)) * 1.5)
            sc += min(2.0, len(proper_re.findall(s)) * 0.6)
            sc += min(1.0, len(example_re.findall(s)) * 0.8)
            sc += min(1.0, len(causal_re.findall(s)) * 0.5)
            # length-normalized density
            sc = sc * 0.6 + (sc / max(len(ws), 5)) * 4
            sent_scores.append(sc)

        avg_evidence = sum(sent_scores) / len(sent_scores) if sent_scores else 0.5
        peak_evidence = max(sent_scores) if sent_scores else 0.5

        # ---------- 2. Audience inversion ----------
        wants_simple = bool(re.search(
            r'\b(simple|simpler|layman|beginner|novice|easy to (understand|grasp)|'
            r'plain (english|language)|without (jargon|technical)|non[- ]?technical|'
            r'high school|7th grade|8th grade|young (mind|student)|new to|first time|'
            r'language barrier|foreigner|intermediate level|basic understanding|'
            r'succinct|concise|easy|kid)', ql))
        wants_expert = bool(re.search(
            r'\b(academic|expert|professional|scholarly|researcher|symposium|'
            r'comprehensive|detailed|specialized|phd|technical (terminology|jargon))', ql))

        # ---------- 3. Constraint extraction & compliance ----------
        constraint_violations = 0
        constraint_satisfactions = 0

        # Allergy / dietary
        allergy_match = re.search(r'\b(allergy|allergic|intolerance|gluten[- ]?free|'
                                  r'wheat|dairy|nut|vegan|vegetarian|kosher|halal)\b', ql)
        if allergy_match:
            allergen = allergy_match.group(1).lower()
            if allergen in rl or 'free' in rl or 'alternative' in rl or 'substitute' in rl:
                constraint_satisfactions += 1
            else:
                # Check if response contains the forbidden ingredient blatantly
                if allergen in ('wheat',) and re.search(r'\b(all[- ]purpose flour|wheat flour|regular flour)\b', rl):
                    constraint_violations += 1

        # Time/specific details requested
        time_req = re.search(r'\b(time|hour|when|schedule)\b', ql)
        if time_req:
            if re.search(r'\b\d{1,2}\s*(am|pm|:\d{2}|o\'?clock)\b', rl) or \
               re.search(r'\b(at \d|by \d|on \w+day)\b', rl):
                constraint_satisfactions += 1

        # Location/place requested
        loc_req = re.search(r'\b(where|location|venue|address|place)\b', ql)
        if loc_req:
            if re.search(r'\b(at|on|in)\b.{1,30}\b(street|avenue|road|house|hall|building|center|venue)\b', rl) or \
               re.search(r'\b\d+\s*[A-Z][a-z]+\b', resp):
                constraint_satisfactions += 1

        # Topic specificity (e.g., "chocolate cake") - check key noun present
        spec_nouns = re.findall(r"\b(chocolate|vanilla|wheat|gluten|formal|casual|"
                                r"superposition|qubit|relativity|gravity|"
                                r"neural network|random forest|photosynthesis)\b", ql)
        for n in spec_nouns:
            if n.lower() in rl:
                constraint_satisfactions += 1

        # ---------- 4. Topic relevance ----------
        q_content = {w for w in re.findall(r"[a-z']+", ql) if w not in STOP and len(w) > 3}
        r_content = set(content)
        if q_content:
            uni_ov = len(q_content & r_content) / len(q_content)
        else:
            uni_ov = 0.4

        # ---------- 5. Length / completeness ----------
        if n_words < 20:
            length_factor = 0.4
        elif n_words < 50:
            length_factor = 0.65
        elif n_words <= 350:
            length_factor = 1.0
        elif n_words <= 500:
            length_factor = 0.9
        else:
            length_factor = 0.75

        # ---------- 6. Structure ----------
        numbered = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', resp))
        bullets = len(re.findall(r'(?:^|\n)\s*[-•*]\s', resp))
        paras = resp.count('\n\n')
        structure_bonus = min(1.2, numbered * 0.15 + bullets * 0.1 + paras * 0.1)

        # ---------- 7. Non-answer / off-topic penalty ----------
        nonanswer = 0
        if re.search(r"\bdon'?t have (the |any |specific )?(info|information|recommendations?|details|data)\b", rl):
            nonanswer += 1
        if re.search(r"\bi don'?t know\b|\bcan'?t (help|tell|provide)\b", rl):
            nonanswer += 1
        if re.search(r"\byou'?ll figure it out\b|\bjust (keep|try) working\b", rl):
            nonanswer += 1

        # Off-topic rambling
        off_topic = (uni_ov < 0.1 and n_words > 60)

        # ---------- 8. Compose ----------
        score = 0.0

        # Evidence (audience-conditional sign)
        if wants_simple:
            # In simple mode, raw evidence is mostly orthogonal; we penalize jargon explicitly
            score += avg_evidence * 0.3  # modest reward
            # explicit jargon penalty
            jargon = len(re.findall(
                r'\b\w{10,}\b', resp))
            sci_terms = len(re.findall(
                r'\b(biochemical|autotrophic|chlorophyll|chloroplast|photolysis|'
                r'redox|quantum|qubit|cuneiform|inertia|momentum|VO2|anaerobic|'
                r'asymptotic|elucidation|reminiscent|decennium)\b', rl))
            score -= min(1.5, jargon * 0.02 + sci_terms * 0.4)
            # reward analogies / simple framings
            analogies = len(re.findall(
                r'\b(imagine|think of|like a|picture|just like|as if|it\'?s like)\b', rl))
            score += min(0.8, analogies * 0.25)
        elif wants_expert:
            score += avg_evidence * 1.0
            score += peak_evidence * 0.3
            # penalize childish framings
            if re.search(r"\b(like a kitchen|like a game|just like a friend)\b", rl):
                score -= 0.5
        else:
            score += avg_evidence * 0.7
            score += peak_evidence * 0.2

        # Universal
        score += uni_ov * 1.4
        score += length_factor * 0.9
        score += structure_bonus * 0.6
        score += constraint_satisfactions * 0.5
        score -= constraint_violations * 1.2
        score -= nonanswer * 0.7
        if off_topic:
            score -= 1.2

        # Hedging at start (slight neg)
        first_sent = sentences[0].lower() if sentences else ""
        if re.match(r'^(well|so|hmm|uh|um|okay so)\b', first_sent):
            score -= 0.2

        # Inability statements
        inability = len(re.findall(
            r"\b(might not|may not|probably won'?t|can'?t|cannot|won'?t be able)\b", rl))
        score -= min(0.6, inability * 0.15)

        # Map to 1-5
        # raw ranges roughly -3 to +6
        final = 3.0 + score * 0.42
        final = max(1.0, min(5.0, final))
        return round(final, 3)

    except Exception:
        return 3.0
