def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0

        text = response.strip()
        if len(text) == 0:
            return 0.0
        if len(text) < 3:
            return 1.0

        q = (query or "").strip()
        q_lower = q.lower()

        # ===== Tokenize =====
        words = re.findall(r"[a-zA-Z']+", text)
        words_lower = [w.lower() for w in words]
        n_words = len(words)
        if n_words == 0:
            return 0.5

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        n_sents = max(len(sentences), 1)

        # ===== 1. Capitalization quality =====
        cap_score = 0.5
        if sentences:
            cap_correct = sum(1 for s in sentences if s and s[0].isupper())
            cap_score = cap_correct / len(sentences)

        # ===== 2. Punctuation =====
        punct_count = sum(1 for c in text if c in '.,;:!?')
        punct_density = punct_count / max(n_words, 1)
        if 0.04 <= punct_density <= 0.3:
            punct_score = 1.0
        elif punct_density < 0.04:
            punct_score = max(0.3, punct_density / 0.04)
        else:
            punct_score = max(0.4, 1.0 - (punct_density - 0.3))

        # Ends with proper punctuation
        ends_well = 1.0 if text.rstrip()[-1] in '.!?"\')]:' else 0.6

        # ===== 3. Sentence length variety =====
        if n_sents >= 2:
            sl = [len(re.findall(r"\w+", s)) for s in sentences]
            sl = [x for x in sl if x > 0]
            if sl and len(sl) >= 2:
                m = sum(sl)/len(sl)
                if m > 0:
                    var = sum((x-m)**2 for x in sl)/len(sl)
                    std = math.sqrt(var)
                    cv = std/m
                    if 0.15 <= cv <= 0.75:
                        variety = 1.0
                    elif cv < 0.15:
                        variety = 0.6
                    else:
                        variety = 0.6
                else:
                    variety = 0.5
            else:
                variety = 0.5
        else:
            variety = 0.6

        # ===== 4. Vocabulary richness =====
        wf = Counter(words_lower)
        hapax = sum(1 for w,c in wf.items() if c == 1)
        if wf:
            hapax_ratio = hapax / len(wf)
            ttr = len(wf) / n_words
        else:
            hapax_ratio = 0; ttr = 0
        vocab_score = 0.5 * min(ttr/0.6, 1.0) + 0.5 * min(hapax_ratio/0.6, 1.0)

        # ===== 5. Word length quality =====
        wl = [len(w) for w in words]
        avg_wl = sum(wl)/max(len(wl),1)
        if 3.8 <= avg_wl <= 6.5:
            wl_score = 1.0
        elif 2.5 <= avg_wl < 3.8:
            wl_score = 0.7
        else:
            wl_score = 0.5

        # ===== 6. Repetition penalty =====
        rep_pen = 0.0
        if n_words >= 6:
            tg = [tuple(words_lower[i:i+3]) for i in range(len(words_lower)-2)]
            tc = Counter(tg)
            rep_tg = sum(c-1 for c in tc.values() if c > 1)
            if tg:
                rep_pen += min(rep_tg/len(tg) * 1.5, 0.5)

        # Repeated lines / sentences
        sl_lower = [s.lower().strip() for s in sentences if len(s.strip()) > 8]
        if len(sl_lower) >= 2:
            sc = Counter(sl_lower)
            dup_s = sum(c-1 for c in sc.values() if c > 1)
            rep_pen += min(dup_s * 0.15, 0.4)

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 2:
            lc = Counter(lines)
            dup_l = sum(c-1 for c in lc.values() if c > 1)
            rep_pen += min(dup_l * 0.08, 0.3)

        # Repeated "Zealously" type patterns - same word starting many lines
        if len(lines) >= 4:
            starts = [l.split()[0].lower() if l.split() else '' for l in lines]
            if starts:
                most_start = Counter(starts).most_common(1)[0]
                if most_start[1] >= 4 and most_start[1] / len(lines) > 0.4:
                    rep_pen += 0.3

        # ===== 7. Template artifact / leakage =====
        template_hits = len(re.findall(
            r'\b(?:Instruction|Input|Output|Question|Answer|Example|Note)\s*:',
            text, re.IGNORECASE))
        template_pen = 0.0
        if template_hits >= 2:
            template_pen = min(0.55, template_hits * 0.12)

        # ===== 8. Code/HTML noise when not asked =====
        asks_code = any(kw in q_lower for kw in
            ['code','program','script','python','html','css','javascript','function','debug','c++','java'])
        html_tags = re.findall(r'<[a-zA-Z/][^>]*>', text)
        code_hits = len(re.findall(r'(?:#include|import\s+\w+|def\s+\w+\(|class\s+\w+|public\s+class)', text))
        noise_pen = 0.0
        if not asks_code:
            if len(html_tags) > 2:
                noise_pen += min(0.4, len(html_tags) * 0.08)
            if code_hits >= 2:
                noise_pen += min(0.4, code_hits * 0.1)

        # ===== 9. Non-ASCII bleed =====
        non_ascii = sum(1 for c in text if ord(c) > 127)
        nonascii_ratio = non_ascii / max(len(text), 1)
        q_nonascii = sum(1 for c in q if ord(c) > 127) / max(len(q), 1) if q else 0
        garble_pen = 0.0
        if nonascii_ratio > 0.05 and q_nonascii < 0.05:
            garble_pen = min(0.55, nonascii_ratio * 2.5)

        # Alpha ratio
        alpha = sum(1 for c in text if c.isalpha())
        ar = alpha / max(len(text), 1)
        if ar < 0.4 and not asks_code:
            garble_pen = max(garble_pen, 0.3)

        # ===== 10. Length-appropriate scoring =====
        is_short_q = bool(re.search(r'\b(name|identify|which|classify|list|biggest|next number)\b', q_lower))
        if n_words < 3:
            length_score = 0.85 if is_short_q else 0.3
        elif n_words < 8:
            length_score = 0.9 if is_short_q else 0.6
        elif n_words <= 200:
            length_score = 1.0
        elif n_words <= 400:
            length_score = 0.85
        else:
            length_score = 0.7

        # ===== Combine =====
        raw = (
            0.10 * cap_score +
            0.08 * punct_score +
            0.06 * ends_well +
            0.10 * variety +
            0.12 * vocab_score +
            0.06 * wl_score +
            0.18 * length_score +
            0.30  # base for being valid
        )

        total_pen = rep_pen + template_pen + noise_pen + garble_pen
        total_pen = min(total_pen, 0.85)

        final = raw * 10.0 * (1.0 - total_pen)

        # Hard floors
        if ar < 0.2:
            final = min(final, 1.0)
        if n_words <= 1 and not is_short_q:
            final = min(final, 1.5)

        return round(max(0.0, min(10.0, final)), 2)

    except Exception:
        return 3.0
