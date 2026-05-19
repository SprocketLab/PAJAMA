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
        query = (query or "").strip()

        rw = response.split()
        n = len(rw)
        if n == 0:
            return 5.0

        rl = response.lower()
        ql = query.lower()

        score = 50.0

        # === 1. Specificity markers ===
        nums = re.findall(r'\b\d+(?:[.,]\d+)?%?\b', response)
        score += min(len(nums) * 1.3, 10)

        # Capitalized non-sentence-start words (proper nouns)
        sents = re.split(r'[.!?]+', response)
        propers = 0
        for s in sents:
            ws = s.strip().split()
            for w in ws[1:]:
                w2 = re.sub(r'[^a-zA-Z]', '', w)
                if w2 and w2[0].isupper() and len(w2) > 1 and not w2.isupper():
                    propers += 1
        score += min(propers * 0.6, 8)

        # Specificity language
        spec_markers = [r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
                        r'\bincluding\b', r'\bspecifically\b', r'\bin particular\b',
                        r'\bnamely\b', r'\bi\.e\.\b', r'\be\.g\.\b']
        spec_count = sum(len(re.findall(p, rl)) for p in spec_markers)
        score += min(spec_count * 2, 8)

        # === 2. Hallucination guards ===

        # Overly precise unsourced figures (e.g., "110.00 pounds per mile")
        odd_decimal = re.findall(r'\b\d{2,}\.\d{1,3}\s*(?:pounds?|dollars?|euros?|cents?|percent|%)', response)
        if odd_decimal:
            # Sanity check: is the figure plausibly large?
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:pounds?|dollars?|euros?)', response):
                try:
                    v = float(m.group(1))
                    if v > 50 and 'mile' in rl:  # mileage rate > $50/mi is absurd
                        score -= 12
                except:
                    pass

        # Sensational/conspiracy language
        sensation = [r'\bshocking\b', r'\bunbelievable\b', r'\bcover-up\b',
                     r'\bthey don\'t want you to know\b', r'\bsheeple\b']
        for p in sensation:
            if re.search(p, rl):
                score -= 6

        # === 3. Vagueness penalty ===
        vague = [r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
                 r'\bvarious factors\b', r'\bvarious ways\b',
                 r'\ba lot of\b', r'\bsort of\b', r'\bkind of\b',
                 r'\band so on\b', r'\betc\.?\b', r'\bthings\b', r'\bstuff\b']
        vague_count = sum(len(re.findall(p, rl)) for p in vague)
        score -= min(vague_count * 1.5, 8)

        # === 4. Hedging (mild bonus when appropriate) ===
        hedge = [r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
                 r'\bmay\b', r'\bmight\b', r'\bcan be\b', r'\blikely\b',
                 r'\baccording to\b', r'\bresearch suggests\b']
        hedge_count = sum(len(re.findall(p, rl)) for p in hedge)
        score += min(hedge_count * 0.7, 4)

        # === 5. Overconfidence penalty ===
        absol = [r'\balways\b', r'\bnever\b', r'\babsolutely\b', r'\bdefinitely\b',
                 r'\bguaranteed\b', r'\b100%\b', r'\bproven fact\b']
        absol_count = sum(len(re.findall(p, rl)) for p in absol)
        score -= min(absol_count * 1.2, 5)

        # === 6. Information density ===
        words_lower = re.findall(r'[a-z]+', rl)
        if words_lower:
            STOP = {'the','a','an','is','are','was','were','be','to','of','and','in','for',
                    'on','with','that','it','as','at','by','from','this','their','its'}
            content = [w for w in words_lower if w not in STOP and len(w) > 2]
            if content:
                ttr = len(set(content)) / len(content)
                score += ttr * 8

        # === 7. Repetition penalty (lighter here, focus on density) ===
        if n >= 6:
            tg = [tuple(words_lower[i:i+3]) for i in range(len(words_lower)-2)] if len(words_lower) >= 3 else []
            if tg:
                tc = Counter(tg)
                rep = sum(v-1 for v in tc.values() if v > 1)
                score -= min(rep * 1.2, 12)

        # === 8. Length appropriateness for evidence-seeking queries ===
        if re.search(r'\b(explain|describe|discuss|effects of|impact of|benefits of)\b', ql):
            if n < 15: score -= 6
            elif n < 50: score += 3
            elif n < 200: score += 6

        # === 9. Truncation ===
        if response[-1] not in '.!?")]}>' and n > 30:
            score -= 3

        # === 10. Examples enumeration for "list" queries ===
        if re.search(r'\b(list|examples|five|three|ten)\b', ql):
            list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', response))
            commas = response.count(',')
            if list_items >= 2 or commas >= 2:
                score += 5

        # === 11. Quick echo check ===
        qw = set(re.findall(r'[a-z]+', ql))
        rwset = set(words_lower)
        if qw and rwset:
            novelty = len(rwset - qw) / max(len(rwset), 1)
            if novelty < 0.2 and n < 25:
                score -= 5

        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 30.0
