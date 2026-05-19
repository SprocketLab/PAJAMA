def judging_function(query, response):
    try:
        import re
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        response = response.strip()
        if not response:
            return 0.0
        query = (query or "").strip()

        ql = query.lower()
        rl = response.lower()
        rw = response.split()
        n = len(rw)
        if n == 0:
            return 5.0

        score = 50.0

        # === 1. Word-count constraint detection ===
        # "in 50 words", "100 word", "fewer than 7 words"
        m = re.search(r'\b(\d+)\s+word', ql)
        if m:
            target = int(m.group(1))
            diff = abs(n - target)
            ratio = diff / max(target, 1)
            if ratio <= 0.15:
                score += 12
            elif ratio <= 0.35:
                score += 5
            elif ratio > 1.5:
                score -= 12
            elif ratio > 0.7:
                score -= 6
        m2 = re.search(r'\bfewer than (\d+) word', ql)
        if m2:
            limit = int(m2.group(1))
            # Check each sentence
            sents = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            violations = sum(1 for s in sents if len(s.split()) >= limit)
            score -= violations * 6

        # === 2. Sentence count constraint ===
        m3 = re.search(r'\b(two|three|four|five)\s+sentences?\b', ql)
        if m3:
            wmap = {'two':2,'three':3,'four':4,'five':5}
            target = wmap[m3.group(1)]
            sents = [s for s in re.split(r'[.!?]+', response) if s.strip()]
            if len(sents) == target:
                score += 8
            elif abs(len(sents) - target) == 1:
                score += 2
            else:
                score -= 5

        # === 3. List count constraint ===
        m4 = re.search(r'\btop\s+(\w+)\b', ql)
        if m4:
            wm = {'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
            tn = wm.get(m4.group(1).lower(), None)
            if tn:
                items = max(
                    len(re.findall(r'(?:^|\n)\s*\d+[.)]\s', response)),
                    response.count(',') + 1 if ',' in response else 0
                )
                if abs(items - tn) <= 1:
                    score += 6

        # === 4. Selection answer (pick a sentence by number) ===
        if re.search(r'\bselect\b.*\bsentence\b', ql) or re.search(r'\(\d+\).*\(\d+\)', ql):
            if re.search(r'^\s*\d+\s*$', response) or re.search(r'^\s*[12345]\.?\s*$', response):
                score += 8
            elif n <= 30:
                score += 3

        # === 5. Type/form constraints ===
        # "in 1-2 sentences": penalize long responses
        if re.search(r'\b1-?2\s+sentence|\bone\s+(?:or two )?sentences?\b', ql):
            sents = [s for s in re.split(r'[.!?]+', response) if s.strip()]
            if len(sents) <= 2:
                score += 8
            elif len(sents) <= 3:
                score += 2
            else:
                score -= 6

        # === 6. "Summarize" with explicit length should be concise ===
        if 'summar' in ql:
            input_m = re.search(r'Input:\s*(.+?)$', query, re.DOTALL)
            if input_m:
                inp = input_m.group(1).strip()
                inp_lower = inp.lower()
                # Verbatim copy penalty
                if inp_lower in rl and len(inp) > 30:
                    if len(inp) / max(len(response), 1) > 0.85:
                        score -= 18
                # Good summary is shorter than input
                inp_words = len(inp.split())
                if inp_words > 20 and n >= inp_words * 0.95:
                    score -= 8

        # === 7. "Replace word X" pattern ===
        m5 = re.search(r'replace\s+the\s+word\s+["\']?(\w+)["\']?', ql)
        if m5:
            target = m5.group(1).lower()
            if target in rl:
                score -= 4  # didn't replace
            else:
                score += 3

        # === 8. Specific format keywords ===
        # "bulleted list"
        if 'bullet' in ql or 'bulleted' in ql:
            if re.search(r'(?:^|\n)\s*[-•*]\s', response) or re.search(r'(?:^|\n)\s*\d+[.)]\s', response):
                score += 5
            else:
                score -= 3

        # === 9. Off-topic short circuit ===
        STOP = {'the','a','an','is','are','was','were','be','to','of','and','in','for',
                'on','with','that','it','as','at','by','from','this','their','its',
                'or','but','not','so','if','than','too','very','just','have','has','had',
                'what','which','who','when','where','why','how'}
        qcontent = [w for w in re.findall(r'[a-z]+', ql)
                    if w not in STOP and len(w) > 3
                    and w not in {'write','create','generate','give','provide','make',
                                  'come','rewrite','describe','explain','outline',
                                  'design','suggest','example','following','input'}]
        rcontent = set(re.findall(r'[a-z]+', rl)) - STOP
        if len(qcontent) >= 2 and rcontent:
            cov = sum(1 for w in qcontent if w in rcontent) / len(qcontent)
            if cov < 0.15 and n >= 8:
                score -= 15
            else:
                score += cov * 8

        # === 10. Repetition penalty (always relevant) ===
        rwl = [w.lower() for w in rw]
        if n >= 5:
            tg = [tuple(rwl[i:i+3]) for i in range(n-2)]
            tc = Counter(tg)
            tr = sum(v-1 for v in tc.values() if v > 1)
            score -= min(tr * 1.5, 12)
        # Block (line-block) repetition
        lines = [l.strip().lower() for l in response.split('\n') if l.strip()]
        if len(lines) >= 6:
            blocks = [tuple(lines[i:i+3]) for i in range(len(lines)-2)]
            bc = Counter(blocks)
            br = sum(v-1 for v in bc.values() if v > 1)
            score -= min(br * 8, 20)

        # === 11. Truncation ===
        if response[-1] not in '.!?")]}>' and n > 30:
            score -= 4

        # === 12. Length sanity ===
        if n < 2:
            score -= 12

        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 30.0
