def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        response = response.strip()
        if len(response) == 0:
            return 0.0
        query = (query or "").strip()

        STOP = {
            'the','a','an','is','are','was','were','be','been','being','have','has','had',
            'do','does','did','will','would','could','should','may','might','can','shall',
            'to','of','in','for','on','with','at','by','from','as','into','through',
            'and','but','or','nor','not','so','yet','if','than','too','very','just',
            'this','that','these','those','it','its','they','them','their','we','our',
            'you','your','he','him','his','she','her','i','me','my','also'
        }

        def tokenize(t):
            return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", t.lower())

        rw = tokenize(response)
        qw = tokenize(query)
        n = len(rw)
        if n == 0:
            return 5.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        ns = max(len(sentences), 1)

        # ------ Query type classification ------
        ql = query.lower()
        is_simple_factual = bool(re.search(
            r'^(what is|what are|name|identify|find|select|extract|complete|check|order|list|give\s+(?:me|three)|how many|when|where|who)\b', ql)) \
            or len(qw) <= 6 \
            or bool(re.search(r'\b(yes/no|true/false|subject|verb|adjective)\b', ql))
        is_creative = bool(re.search(r'\b(write|generate|create|compose|design|come up with|invent|construct|outline|develop|build|make a|provide a|suggest)\b', ql))
        is_explain  = bool(re.search(r'\b(explain|describe|discuss|summarize|elaborate|why|how does|how do)\b', ql))
        is_code     = bool(re.search(r'\b(function|algorithm|regex|code|python|javascript|sort|return)\b', ql))

        # ------ Length appropriateness ------
        if is_simple_factual:
            if n <= 3:    length_score = 8.0
            elif n <= 10: length_score = 9.0
            elif n <= 25: length_score = 7.0
            elif n <= 60: length_score = 4.5
            else:         length_score = 2.5
        elif is_creative or is_explain:
            if n < 5:     length_score = 1.5
            elif n < 20:  length_score = 4.0
            elif n < 60:  length_score = 8.0
            elif n < 200: length_score = 9.0
            else:         length_score = 7.0
        else:
            if n < 3:     length_score = 1.0
            elif n < 10:  length_score = 5.0
            elif n < 80:  length_score = 8.5
            elif n < 200: length_score = 8.0
            else:         length_score = 6.5

        # ------ Repetition penalty (phrase + stanza level) ------
        rep_penalty = 0.0
        if n >= 4:
            for ng in (3, 4, 5):
                if n < ng: continue
                grams = [tuple(rw[i:i+ng]) for i in range(n-ng+1)]
                c = Counter(grams)
                rep = sum(v-1 for v in c.values() if v > 1)
                rep_penalty += (rep / max(len(grams),1)) * (ng * 1.5)
        # Stanza-level (line-level) repetition: critical for poems/essays
        lines = [l.strip().lower() for l in response.split('\n') if l.strip()]
        if len(lines) >= 3:
            lc = Counter(lines)
            line_dup = sum(v-1 for v in lc.values() if v > 1)
            rep_penalty += line_dup * 2.5
        # Multi-line stanza repetition: detect repeated 3-line blocks
        if len(lines) >= 6:
            blocks = [tuple(lines[i:i+3]) for i in range(len(lines)-2)]
            bc = Counter(blocks)
            block_rep = sum(v-1 for v in bc.values() if v > 1)
            rep_penalty += block_rep * 4.0
        rep_penalty = min(rep_penalty, 12.0)

        # ------ Query coverage ------
        qc = set(w for w in qw if w not in STOP and len(w) > 2)
        rc_set = set(w for w in rw if w not in STOP and len(w) > 2)
        if qc:
            coverage = len(qc & rc_set) / len(qc)
        else:
            coverage = 0.5
        coverage_score = coverage * 7.0

        # ------ Information density ------
        content_words = [w for w in rw if w not in STOP and len(w) > 2]
        if content_words:
            ttr = len(set(content_words)) / len(content_words)
        else:
            ttr = 0.5
        if ttr < 0.25:    density_score = 1.0
        elif ttr < 0.45:  density_score = 4.0
        elif ttr <= 0.85: density_score = 8.0
        else:             density_score = 6.5

        # ------ Structural completeness ------
        struct = 0.0
        if response[-1] in '.!?"\')]}':
            struct += 1.5
        if ns >= 2:
            struct += 1.5
        if re.search(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', response):
            struct += 1.0

        # ------ Tautology / echo penalty ------
        tautology_pen = 0.0
        if qc and rc_set:
            novelty = len(rc_set - qc) / max(len(rc_set), 1)
            if novelty < 0.15 and n < 25:
                tautology_pen += 3.5
            elif novelty < 0.3:
                tautology_pen += 1.5
        # Detect verbatim copy of input
        m_input = re.search(r'Input:\s*(.+?)(?:\n\n|\Z)', query, re.DOTALL)
        if m_input:
            inp = m_input.group(1).strip().lower()
            if inp and inp in response.lower() and len(inp) > 20:
                ratio = len(inp) / max(len(response), 1)
                if ratio > 0.85:
                    tautology_pen += 5.0

        # ------ Self-referential meta echo ------
        if re.search(r'\b(the most frequently asked question|three examples? of words? that describe|the chemical element in the given)\b', response.lower()):
            tautology_pen += 2.5

        # ------ Final composition ------
        raw = (length_score * 1.4
               + coverage_score * 1.2
               + density_score * 1.1
               + struct * 1.0
               - rep_penalty * 1.3
               - tautology_pen * 1.2)

        # Normalize to roughly 0-100
        score = (raw + 5) * 4.0
        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 25.0
