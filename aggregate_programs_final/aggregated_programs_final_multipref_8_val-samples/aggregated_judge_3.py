def judging_function(query, response):
    """
    Logical coherence and reasoning transparency scorer.
    Combines discourse-marker analysis with step-by-step reasoning detection.
    Gates against penalizing concise responses on explicit-brief queries.
    """
    try:
        import re
        import math

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0

        r = response.strip()
        if len(r) < 10:
            return 2.0
        rl = r.lower()
        ql = query.lower()

        score = 50.0
        words = r.split()
        wc = len(words)
        if wc == 0:
            return 2.0

        # Detect short-response requests
        short_req = bool(re.search(
            r'\b(very short|concise|brief|short response|one sentence|raw message|in (?:a |one )?sentence)\b',
            ql))

        # === 1. Discourse marker categories ===
        causal = ['because','therefore','thus','hence','consequently','as a result',
                  'due to','since','this means','this implies','it follows','leading to',
                  'which means','so that']
        contrast = ['however','although','nevertheless','on the other hand',
                    'in contrast','while','despite','yet','whereas','even though']
        additive = ['furthermore','moreover','in addition','additionally','also',
                    'besides','not only']
        sequential = ['first','second','third','next','then','finally','to begin',
                      'in conclusion','to summarize','overall']
        example = ['for example','for instance','such as','including','namely',
                   'specifically','in particular','to illustrate']

        c_count = sum(rl.count(m) for m in causal)
        ct_count = sum(rl.count(m) for m in contrast)
        a_count = sum(rl.count(m) for m in additive)
        s_count = sum(rl.count(m) for m in sequential)
        e_count = sum(rl.count(m) for m in example)

        total = c_count + ct_count + a_count + s_count + e_count
        types_used = sum([c_count > 0, ct_count > 0, a_count > 0, s_count > 0, e_count > 0])

        # Marker density per 100 words
        density = total / max(wc, 1) * 100

        if short_req:
            # For short requests, don't penalize lack of markers
            score += min(types_used, 3) * 1.5
        else:
            if density < 0.5:
                score -= 3
            elif density < 2.0:
                score += 2
            elif density < 6.0:
                score += 7
            elif density < 10:
                score += 5
            else:
                score += 2

            score += types_used * 1.5

        # === 2. Step-by-step structural markers ===
        numbered = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|step\s+\d+)', rl))
        bullets = len(re.findall(r'(?:^|\n)\s*[-*•]\s+\S', r))
        headers = len(re.findall(r'(?:^|\n)#{1,4}\s+', r))

        if not short_req:
            score += min(numbered, 8) * 1.0
            score += min(bullets, 8) * 0.5
            score += min(headers, 4) * 1.0

        # === 3. Intermediate conclusion markers ===
        intermediate = ['this tells us','we can see','this shows','this indicates',
                       'note that','notice that','importantly','the key point',
                       'in other words','to put it','what this means','having established']
        int_count = sum(rl.count(m) for m in intermediate)
        score += min(int_count, 6) * 1.5

        # === 4. Sentence-level coherence (topic continuity) ===
        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        n_sent = len(sentences)

        if n_sent >= 3:
            overlaps = []
            stopw = {'the','a','and','is','are','to','of','for','in','on','this','that',
                     'with','from','it','its','you','your','we','they','their'}
            for i in range(n_sent - 1):
                a_words = set(re.findall(r'\b\w{3,}\b', sentences[i].lower())) - stopw
                b_words = set(re.findall(r'\b\w{3,}\b', sentences[i+1].lower())) - stopw
                if a_words and b_words:
                    overlaps.append(len(a_words & b_words) / min(len(a_words), len(b_words)))
            if overlaps:
                avg = sum(overlaps) / len(overlaps)
                if avg < 0.05:
                    score -= 4  # disjoint
                elif avg < 0.5:
                    score += 4
                else:
                    score += 1  # too repetitive

        # === 5. Reasoning words bonus ===
        reasoning = ['because','therefore','so that','in order to','the reason',
                     'this is because','reason why']
        reason_count = sum(rl.count(m) for m in reasoning)
        score += min(reason_count, 6) * 1.3

        # === 6. Truncation penalty ===
        last_char = r.rstrip()[-1] if r.rstrip() else ''
        if last_char in ',:;-':
            score -= 6
        elif last_char not in '.!?"\')]>*}':
            score -= 5
        else:
            score += 2

        # Mid-sentence truncation
        if re.search(r'\b(and|or|the|a|to|in|of|for|with|is|are|was|were|that|this)\s*$',
                     r.rstrip().lower()):
            score -= 6

        # === 7. Length-appropriate to query ===
        if short_req:
            if wc <= 30:
                score += 6
            elif wc <= 60:
                score -= 1
            else:
                score -= 6
        else:
            if wc >= 80:
                score += 3
            elif wc >= 40:
                score += 1
            elif wc < 15:
                score -= 3

        # === 8. Nuance/conditional reasoning ===
        nuance = [r'\bif\b.*\bthen\b', r'\bdepending on\b', r'\bgenerally\b',
                  r'\btypically\b', r'\bin (?:some|certain) cases\b',
                  r'\bnot necessarily\b', r'\bthat said\b']
        nuance_count = sum(1 for p in nuance if re.search(p, rl))
        score += min(nuance_count, 4) * 1.0

        # === 9. Strong openings ===
        first_100 = rl[:100]
        if re.search(r'^(yes|no)[,.]', first_100):
            score += 2
        if re.search(r'^(certainly|absolutely|here|to (?:answer|understand))', first_100):
            score += 1

        # === 10. Weak/filler openings ===
        if re.search(r'^(well,?\s|um,?\s|so,?\s+basically|it depends)', first_100):
            score -= 2

        # === 11. Direct vs evasive answer detection ===
        # For "why" / "is X..." questions, evasive answers hurt
        if re.search(r'^\s*(why|is\s|are\s|can\s|does\s|do\s|should)', ql):
            evasive = ['not necessarily','depends on','it varies','hard to say',
                       'no clear answer']
            ev_count = sum(rl.count(p) for p in evasive)
            if ev_count >= 2 and n_sent <= 3:
                score -= 4  # evasion when direct answer wanted

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 25.0
