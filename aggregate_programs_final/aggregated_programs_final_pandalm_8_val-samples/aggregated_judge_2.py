def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        response = response.strip()
        if not response:
            return 0.0

        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", response.lower())
        n = len(words)
        if n == 0:
            return 3.0

        score = 60.0  # healthy baseline; we subtract for repetition issues

        # 1. Word-level repetition (content words)
        STOP = {'the','a','an','is','are','was','were','be','to','of','and','in','for','on',
                'with','that','it','as','at','by','from','this','their','they','its','also',
                'or','but','not','no','so','if','than','very','just','have','has','had',
                'do','does','did','will','would','could','should','may','might','can'}
        content = [w for w in words if w not in STOP and len(w) > 2]
        if content:
            cc = Counter(content)
            max_c = max(cc.values())
            dom_ratio = max_c / len(content)
            if dom_ratio > 0.30 and max_c >= 4:
                score -= 25
            elif dom_ratio > 0.20 and max_c >= 4:
                score -= 12
            elif dom_ratio > 0.15 and max_c >= 5:
                score -= 6
            unique_ratio = len(set(content)) / len(content)
            if unique_ratio < 0.35:
                score -= 15
            elif unique_ratio < 0.5:
                score -= 6

        # 2. N-gram repetition
        def ngram_rep(toks, ng):
            if len(toks) < ng: return 0.0, 0
            grams = [tuple(toks[i:i+ng]) for i in range(len(toks)-ng+1)]
            c = Counter(grams)
            rep = sum(v-1 for v in c.values() if v > 1)
            return rep / max(len(grams),1), max(c.values()) if c else 0
        b_rep, _ = ngram_rep(words, 2)
        t_rep, t_max = ngram_rep(words, 3)
        q_rep, q_max = ngram_rep(words, 4)
        score -= min(b_rep * 25, 12)
        score -= min(t_rep * 50, 20)
        score -= min(q_rep * 80, 25)
        if q_max >= 3: score -= 5

        # 3. Line/stanza-level repetition (critical for poems/structured text)
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if len(lines) >= 3:
            ll = [l.lower() for l in lines]
            lc = Counter(ll)
            dup = sum(v-1 for v in lc.values() if v > 1)
            score -= min(dup * 6, 25)
        # 4-line block repetition (typical stanza)
        if len(lines) >= 8:
            blocks4 = [tuple(l.lower() for l in lines[i:i+4]) for i in range(len(lines)-3)]
            bc = Counter(blocks4)
            blkrep = sum(v-1 for v in bc.values() if v > 1)
            score -= min(blkrep * 12, 30)

        # 4. Sentence-level duplication
        sents = [s.strip().lower() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 8]
        if sents:
            sc = Counter(sents)
            sd = sum(v-1 for v in sc.values() if v > 1)
            score -= min(sd * 8, 20)

        # 5. Tautological / echo-of-question penalty
        # Responses that paraphrase the question rather than answer it
        rl = response.lower()
        ql = (query or "").lower()
        tautology_phrases = [
            r'\bis a (?:type|kind|form) of\b.*' + r'\b' + (re.split(r'\bwhat is\b', ql)[-1].strip().split('?')[0].split()[0] if 'what is' in ql else 'XYZNOMATCH') + r'\b',
            r'the most frequently asked question about',
            r'three examples? of words? that describe',
            r'the chemical element in the given',
        ]
        for ph in tautology_phrases:
            try:
                if re.search(ph, rl):
                    score -= 8
            except Exception:
                pass

        # Check if response just echoes the query content
        qwords = set(re.findall(r'[a-z]+', ql)) - STOP
        rwords = set(words) - STOP
        if qwords and rwords:
            overlap = len(qwords & rwords) / max(len(rwords), 1)
            if overlap > 0.75 and n < 20:
                score -= 10

        # 6. Verbatim copy of "Input:" segment
        m = re.search(r'Input:\s*(.+?)(?:\n\n|\Z)', query or "", re.DOTALL)
        if m:
            inp = m.group(1).strip().lower()
            if inp and len(inp) > 30 and inp in rl:
                if len(inp) / max(len(response), 1) > 0.85:
                    score -= 18

        # 7. Length sanity floor
        if n < 2:
            score -= 25
        elif n < 4:
            score -= 8

        # 8. Award for having actual fresh content
        if content:
            unique_content = len(set(content))
            score += min(unique_content * 0.4, 12)

        # 9. Bonus for well-formed completion
        if response[-1] in '.!?"\')]':
            score += 2

        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 30.0
