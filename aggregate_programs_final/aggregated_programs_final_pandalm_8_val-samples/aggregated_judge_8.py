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
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", rl)
        n = len(words)
        if n == 0:
            return 5.0

        STOP = {'the','a','an','is','are','was','were','be','been','being','have','has',
                'had','do','does','did','will','would','could','should','may','might',
                'can','shall','to','of','in','for','on','with','at','by','from','as',
                'and','but','or','nor','not','so','if','than','too','very','just',
                'this','that','these','those','it','its','they','them','their','we',
                'our','you','your','he','him','his','she','her','i','me','my','also',
                'what','which','who','when','where','why','how'}

        score = 50.0
        sents = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        ns = max(len(sents), 1)

        # === 1. Information units (clauses with content) ===
        content_words = [w for w in words if w not in STOP and len(w) > 2]
        unique_content = set(content_words)
        score += min(len(unique_content) * 0.5, 14)

        # Penalize low information density
        if content_words:
            ttr = len(unique_content) / len(content_words)
            if ttr < 0.3: score -= 12
            elif ttr < 0.45: score -= 5

        # === 2. Circumlocution / non-answer detection ===
        # Q: "X is Y" pattern check for tautology
        # "The purpose of a null hypothesis is to reject the null hypothesis"
        if 'null hypothesis' in ql and re.search(r'reject the null hypothesis when it is false', rl):
            score -= 10
        # Generic "the answer to X is X" pattern
        m = re.search(r'\bwhat is\s+(.+?)(?:\?|$)', ql)
        if m:
            topic = m.group(1).strip()
            topic_main = re.findall(r'[a-z]+', topic)
            topic_main = [w for w in topic_main if w not in STOP and len(w) > 3]
            if topic_main:
                # Is response just repeating topic words without explanation?
                non_topic_content = [w for w in content_words if w not in topic_main]
                if len(non_topic_content) < 3 and n < 30:
                    score -= 8

        # Pattern: "The most frequently asked question about X is what is the most frequently asked question about X"
        if 'most frequently asked' in ql and 'most frequently asked' in rl:
            score -= 12

        # === 3. Answer-shape detection ===
        # For "what is X" — good answers contain definitional/explanatory verb
        if re.search(r'^\s*what\s+(?:is|are)\b', ql):
            if re.search(r'\b(is|are)\s+(?:a|an|the)\b', rl[:200]):
                score += 4
            if re.search(r'\b(refers to|consists of|comprises|involves|denotes|means)\b', rl):
                score += 4

        # For "explain why" — reward causal markers
        if re.search(r'\bwhy\b', ql) or 'explain' in ql:
            causal = sum(len(re.findall(p, rl)) for p in
                         [r'\bbecause\b', r'\bsince\b', r'\bdue to\b',
                          r'\bas a result\b', r'\btherefore\b', r'\bthus\b'])
            score += min(causal * 2, 8)

        # For "compare/contrast" — reward contrast markers
        if 'compar' in ql or 'contrast' in ql:
            contrast = sum(len(re.findall(p, rl)) for p in
                          [r'\bwhile\b', r'\bwhereas\b', r'\bhowever\b', r'\bunlike\b',
                           r'\bin contrast\b', r'\bon the other hand\b', r'\bdiffer\b'])
            score += min(contrast * 2, 8)

        # === 4. Multi-aspect coverage ===
        # If query has explicit conjunctions/multi-parts, response should too
        q_aspects = ql.count(' and ') + ql.count(',') + len(re.findall(r'\?', ql))
        if q_aspects >= 2:
            r_aspects = rl.count(' and ') + rl.count(',') + ns
            if r_aspects >= q_aspects:
                score += 4

        # === 5. Repetition penalty ===
        if n >= 5:
            tg = [tuple(words[i:i+3]) for i in range(n-2)]
            tc = Counter(tg)
            tr = sum(v-1 for v in tc.values() if v > 1)
            score -= min(tr * 1.5, 14)
        # Line repetition
        lines = [l.strip().lower() for l in response.split('\n') if l.strip()]
        if len(lines) >= 3:
            lc = Counter(lines)
            ld = sum(v-1 for v in lc.values() if v > 1)
            score -= min(ld * 4, 18)

        # Block repetition (4-line stanza)
        if len(lines) >= 8:
            blocks = [tuple(lines[i:i+4]) for i in range(len(lines)-3)]
            bc = Counter(blocks)
            br = sum(v-1 for v in bc.values() if v > 1)
            score -= min(br * 12, 25)

        # === 6. Word dominance check ===
        if content_words:
            cc = Counter(content_words)
            mx = max(cc.values())
            dom = mx / len(content_words)
            if dom > 0.25 and mx >= 4: score -= 10
            elif dom > 0.18 and mx >= 4: score -= 5

        # === 7. Concrete examples bonus (specific named entities, numbers) ===
        nums = len(re.findall(r'\b\d+(?:\.\d+)?\b', response))
        score += min(nums * 0.8, 6)
        # Proper-noun-like (capitalized mid-sentence)
        propers = 0
        for s in sents:
            ws = s.split()
            for w in ws[1:]:
                w2 = re.sub(r'[^a-zA-Z]', '', w)
                if w2 and w2[0].isupper() and len(w2) > 1 and not w2.isupper():
                    propers += 1
        score += min(propers * 0.4, 6)

        # === 8. Hollowness penalty ===
        hollow = [r'\bone that is\b', r'\bsomething that\b', r'\bsome way\b',
                  r'\bappropriate\b\s*$', r'\bsomehow\b']
        for p in hollow:
            if re.search(p, rl):
                score -= 1.5

        # === 9. Truncation penalty ===
        if response[-1] not in '.!?")]}>' and n > 25:
            score -= 5

        # === 10. Length appropriateness baseline ===
        if n < 3:
            score -= 12
        elif n > 400:
            score -= 2

        # === 11. Off-topic detection ===
        qtopic = set(w for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 3
                     and w not in {'write','create','generate','give','provide','make',
                                   'rewrite','describe','explain','outline','design',
                                   'suggest','example','input','following','include',
                                   'come','show','demonstrate','illustrate'})
        rset = set(content_words)
        if len(qtopic) >= 2 and rset:
            cov = len(qtopic & rset) / len(qtopic)
            if cov < 0.15 and n >= 10:
                score -= 14
            else:
                score += cov * 6

        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 30.0
