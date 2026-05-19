def judging_function(query, response):
    try:
        import re
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        q = query.strip() if isinstance(query, str) else ""
        resp = response.strip()
        if not resp:
            return 0.0

        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        rl = resp.lower()
        ql = q.lower()

        base = 5.5

        # ===== Length component =====
        if n_words < 2:
            base = 1.0
        elif n_words < 6:
            base = 3.5
        elif n_words < 15:
            base = 5.0
        elif n_words <= 250:
            base = 6.5
        else:
            base = 6.0

        # ===== KEY DENOISING: Template repetition =====
        # "Instruction:" or "Input:" or "Output:" repeated
        instruction_hits = len(re.findall(r'\b[Ii]nstruction\s*:', resp))
        input_hits = len(re.findall(r'\b[Ii]nput\s*:', resp))
        output_hits = len(re.findall(r'\b[Oo]utput\s*:', resp))
        question_hits = len(re.findall(r'\b[Qq]uestion\s*:', resp))
        answer_hits = len(re.findall(r'\b[Aa]nswer\s*:', resp))

        total_tmpl = instruction_hits + input_hits + output_hits + question_hits + answer_hits

        # First "Output:" or "Answer:" prefix can be legit (one occurrence)
        # But 3+ is a strong signal of fabricated continuation
        if total_tmpl >= 4:
            base -= min(5.0, total_tmpl * 0.8)
        elif total_tmpl == 3:
            base -= 2.0
        elif total_tmpl == 2:
            base -= 0.8

        # ===== "1. ... 2. ... 3. ..." numbered list of fabricated Q&A =====
        # If numbered items each end with "?", it's likely a fake quiz
        num_items = re.findall(r'(?:^|\n)\s*\d+[.)]\s+([^\n]+)', resp)
        if len(num_items) >= 3:
            q_endings = sum(1 for it in num_items if it.strip().endswith('?'))
            if q_endings >= len(num_items) * 0.6:
                base -= min(3.0, q_endings * 0.6)

        # ===== Multiple choice options A) B) C) D) =====
        mc_opts = len(re.findall(r'(?:^|\n)\s*[A-FH-Z]\)\s+\S', resp))
        if mc_opts >= 3 and 'choice' not in ql and 'option' not in ql:
            base -= min(2.5, mc_opts * 0.5)

        # ===== Excessive paragraph fragments (Case 12, 25) =====
        # E.g., random product/news text appearing after the answer
        # Heuristic: detect topic drift between halves
        STOP = {'a','an','the','is','are','was','were','to','of','in','for','on','with','at',
                'by','from','as','and','or','but','not','it','this','that','i','you','he','she','they','we'}
        if n_words >= 60:
            mid = len(resp) // 2
            q_content = set(w.lower() for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 3)
            if q_content:
                first_half_content = set(re.findall(r'[a-z]+', resp[:mid].lower())) & q_content
                second_half_content = set(re.findall(r'[a-z]+', resp[mid:].lower())) & q_content
                if len(first_half_content) >= 2 and len(second_half_content) == 0:
                    base -= 2.0

        # ===== Detect fabricated continuation patterns =====
        # "9. Which is..." "8. Which is..." sequences (Case 6)
        # "Question:10\n... Question:" type spam
        repeated_starts = Counter()
        sents = [s.strip() for s in re.split(r'[.!?\n]+', resp) if len(s.strip()) > 5]
        for s in sents:
            opener = ' '.join(s.split()[:3]).lower()
            repeated_starts[opener] += 1
        max_repeat = max(repeated_starts.values()) if repeated_starts else 1
        if max_repeat >= 4:
            base -= min(2.5, max_repeat * 0.4)

        # ===== Reward concise on-topic answers =====
        q_content = set(w.lower() for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 2)
        r_content = set(w.lower() for w in re.findall(r'[a-z]+', rl) if w not in STOP and len(w) > 2)

        # Recognize "brief output" queries
        brief_query = any(s in ql for s in [
            'output the','output directly','identify the','identify a',
            'name a','name the','which is','which of','classify',
            'compute','find the','tell me which',
            'biggest','largest','longest','shortest'
        ])

        if brief_query:
            if n_words <= 10 and total_tmpl == 0 and max_repeat < 3:
                base += 2.0  # Big bonus for clean direct answer
            elif n_words <= 30 and total_tmpl <= 1:
                base += 0.8
            elif n_words > 100 and total_tmpl >= 1:
                base -= 1.0  # Verbose response with template artifacts

        # Coverage
        if q_content:
            cov = len(q_content & r_content) / len(q_content)
            base += cov * 1.5

        # ===== Repetition =====
        wl = [w.lower() for w in words]
        if n_words >= 6:
            tri = [tuple(wl[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            rep_ratio = rep_tri / max(len(tri), 1)
            if rep_ratio > 0.15:
                base -= min(3.0, rep_ratio * 8)

        # Unique word ratio for long responses
        if n_words > 20:
            uniq_ratio = len(set(wl)) / n_words
            if uniq_ratio < 0.3:
                base -= 2.0
            elif uniq_ratio < 0.4:
                base -= 1.0

        # ===== Garbled foreign text =====
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            base -= min(3.0, nonlatin / 15.0)

        # ===== Random parenthetical garble =====
        weird_parens = len(re.findall(r'\([^)]{1,30}\)\s*\([^)]{1,30}\)', resp))
        if weird_parens >= 2:
            base -= 2.0

        # ===== Numeric noise (rows of 0s/1s) =====
        if len(re.findall(r'(?:[-\d]+\s+){8,}', resp)) > 0:
            if not any(k in ql for k in ['number','sequence','count','matrix','array','math']):
                base -= 2.0

        return round(max(0.0, min(10.0, base)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 2.0
        except:
            return 3.0
