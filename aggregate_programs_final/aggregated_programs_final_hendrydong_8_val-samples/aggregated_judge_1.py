def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""

        resp = response.strip()
        if len(resp) < 3:
            return 0.0

        STOP = {
            'the','a','an','is','are','was','were','be','been','being','have','has','had',
            'do','does','did','will','would','could','should','may','might','shall','can',
            'to','of','in','for','on','with','at','by','from','as','into','through','during',
            'before','after','and','but','or','nor','not','so','yet','if','then','than','too',
            'very','just','that','this','these','those','it','its','i','me','my','we','our',
            'you','your','he','him','his','she','her','they','them','their','what','which',
            'who','whom','when','where','why','how','about','up','out','also','like','some',
            'any','all','no','one','two','am','re','ve','ll','m','s','d','t','don','im','ive'
        }

        def toks(t):
            t = t.lower()
            t = re.sub(r"[^a-z0-9\s']", ' ', t)
            return t.split()

        def content(ws):
            return [w for w in ws if w not in STOP and len(w) > 2]

        q_toks = toks(query)
        r_toks = toks(resp)
        q_c = content(q_toks)
        r_c = content(r_toks)
        q_set = set(q_c)
        r_set = set(r_c)

        score = 50.0  # midpoint of 0..100

        # 1. Query coverage (broad relevance)
        if q_set:
            cov = len(q_set & r_set) / len(q_set)
            score += cov * 18.0
        else:
            cov = 0.5

        # 2. Bigram overlap
        q_bg = set(zip(q_c, q_c[1:])) if len(q_c) > 1 else set()
        r_bg = set(zip(r_c, r_c[1:])) if len(r_c) > 1 else set()
        if q_bg:
            bcov = len(q_bg & r_bg) / len(q_bg)
            score += bcov * 8.0

        # 3. Length appropriateness (NOT pure "longer = better")
        wc = len(r_toks)
        if wc < 3:
            score -= 25
        elif wc < 8:
            score -= 5
        elif wc < 20:
            score += 2
        elif wc < 60:
            score += 6
        elif wc < 200:
            score += 8
        elif wc < 400:
            score += 5
        else:
            score += 2

        # 4. Question-type matching
        ql = query.lower()
        rl = resp.lower()
        if 'how' in ql[:60]:
            if re.search(r'\b(step|first|then|next|because|by |through|use|method)\b', rl):
                score += 3
        if 'why' in ql[:60]:
            if re.search(r'\b(because|reason|due to|since|caused)\b', rl):
                score += 3
        if 'what' in ql[:60]:
            if re.search(r'\b(is|are|refers|means|defined)\b', rl):
                score += 1

        # 5. Specificity signals
        nums = len(re.findall(r'\b\d+\b', resp))
        score += min(nums * 0.6, 4)
        propn = len(re.findall(r'(?<=[a-z,;:]\s)[A-Z][a-z]+', resp))
        score += min(propn * 0.4, 4)

        # 6. Explanation markers
        explain = len(re.findall(
            r'\b(for example|for instance|such as|because|specifically|in particular|'
            r'i\.e\.|e\.g\.|in other words|that is)\b', rl))
        score += min(explain * 1.5, 6)

        # === GATES against misleading "long but bad" responses ===

        # Gate: response that REFUSES to answer / asks clarifying question (often worse than direct)
        clarify_q = (
            re.search(r'^\s*(can you|could you|do you|did you|please|when you say)', rl) and
            resp.count('?') >= 1 and wc < 60
        )
        if clarify_q:
            score -= 8

        # Gate: meta/safety refusal patterns
        refusal_patterns = [
            r"i must point out", r"not factually coherent",
            r"i cannot provide", r"i'?m not able to provide",
            r"the question (?:itself )?may not be meaningful",
            r"butter is not a healthy", r"unhealthy ingredient",
            r"i hope this helps", r"happy to help",
        ]
        refusals = sum(1 for p in refusal_patterns if re.search(p, rl))
        # Only penalize when refusal seems to dominate (no real answer content)
        if refusals >= 1 and wc < 150:
            score -= refusals * 4
        elif refusals >= 2:
            score -= 4

        # Gate: response is mostly off-topic - detect with very low content overlap
        if q_set and len(q_set) >= 3:
            overlap_words = len(q_set & r_set)
            if overlap_words == 0 and wc > 20:
                score -= 12
            elif overlap_words / len(q_set) < 0.15 and wc > 40:
                score -= 6

        # Gate: emoji/childish flavor when query is serious translation/technical
        emoji_count = len(re.findall(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', resp))
        if emoji_count >= 2 and ('translate' in ql or 'latex' in ql or 'sql' in ql):
            score -= 10

        # Gate: bot/template welcome
        if re.search(r'welcome to /r/|please read our rules|i am a bot', rl):
            score -= 10

        # Gate: pure boilerplate "great question" with little content
        if re.match(r'^(sure|great question|that\'s a great|i\'?d be happy)', rl) and wc < 25:
            score -= 5

        # Gate: response is a translation task but didn't translate
        if 'translate' in ql[:80] or 'romanian' in ql.lower() or 'spanish:' in ql.lower():
            # Detect non-ASCII letters as evidence of translation attempt
            non_ascii = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii < 5 and wc > 20:
                score -= 15  # English-only when translation requested

        # Gate: code-block presence when code requested
        if re.search(r'\b(code|sql|function|python|java|script)\b', ql):
            if '```' in resp or re.search(r'\b(def |class |SELECT|FROM)\b', resp):
                score += 3

        # 7. Type-token ratio (info density)
        if r_c:
            ttr = len(set(r_c)) / len(r_c)
            score += (ttr - 0.4) * 8

        # 8. Penalize incoherent/garbled text (random latex/special chars density)
        garbage = len(re.findall(r'[^\x00-\x7F]', resp))
        garbage_ratio = garbage / max(len(resp), 1)
        if garbage_ratio > 0.15 and 'translate' not in ql.lower():
            score -= 15

        # 9. Conciseness reward for short queries
        if len(q_toks) < 12 and 8 <= wc <= 40:
            score += 3

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 50.0
