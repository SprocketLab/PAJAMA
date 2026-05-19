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

        STOP = {'the','a','an','is','are','was','were','be','to','of','and','in','for',
                'on','with','that','it','as','at','by','from','this','these','those','they',
                'its','their','or','but','not','so','if','than','too','very','just','have',
                'has','had','do','does','did','will','would','could','should','can','may',
                'i','me','my','we','our','you','your','he','him','his','she','her','also'}

        rw = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", response.lower())
        qw = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", query.lower())
        n = len(rw)
        if n == 0:
            return 5.0

        ql = query.lower()
        rl = response.lower()

        # === Detect query "shape" ===
        # Simple/factual queries that prefer SHORT answers
        short_pref = False
        if re.search(r'^\s*(what is|who is|when did|where is|name|identify|find|select|extract|complete|check if|is\b|are\b|does\b|do\b)', ql):
            short_pref = True
        if re.search(r'\b(yes/no|true/false|one word|in two|fewer than|less than)\b', ql):
            short_pref = True
        if len(qw) < 8 and not re.search(r'\b(explain|describe|discuss|why|how does)\b', ql):
            short_pref = True

        # Length-loving queries (essays, stories, descriptions)
        long_pref = bool(re.search(r'\b(explain|describe|discuss|essay|story|paragraph|elaborate|in detail|comprehensive)\b', ql)) \
                    and not re.search(r'\b(in two|fewer than|brief|concise|short)\b', ql)

        # Pattern queries (poems, recipes, code, lists)
        is_poem = 'poem' in ql or 'verse' in ql
        is_list = bool(re.search(r'\b(list|bulleted?|top \d|examples?|three|five|ten|enumerate)\b', ql))
        is_code = bool(re.search(r'\b(function|algorithm|regex|code|program)\b', ql))

        # === Base scoring ===
        score = 50.0

        # === Length calibration ===
        if short_pref:
            if n <= 5:    score += 14
            elif n <= 15: score += 8
            elif n <= 35: score += 2
            elif n <= 80: score -= 4
            else:         score -= 10
        elif long_pref:
            if n < 10:    score -= 8
            elif n < 30:  score += 2
            elif n < 80:  score += 8
            elif n < 250: score += 10
            else:         score += 4
        elif is_poem:
            if n < 15:    score -= 2
            elif n < 100: score += 7
            else:         score += 3
        elif is_code:
            if n < 5:     score -= 4
            elif n < 80:  score += 8
            else:         score += 2
        else:
            if n < 5:     score -= 5
            elif n < 15:  score += 2
            elif n < 60:  score += 7
            elif n < 200: score += 6
            else:         score += 1

        # === Repetition penalty (multi-level) ===
        rep_pen = 0
        # Word-level
        content = [w for w in rw if w not in STOP and len(w) > 2]
        if content:
            cc = Counter(content)
            mx = max(cc.values())
            if mx / len(content) > 0.30 and mx >= 4: rep_pen += 12
            elif mx / len(content) > 0.20 and mx >= 4: rep_pen += 6
        # Trigram
        if n >= 4:
            tg = [tuple(rw[i:i+3]) for i in range(n-2)]
            tc = Counter(tg)
            tr = sum(v-1 for v in tc.values() if v > 1)
            rep_pen += min(tr * 1.5, 12)
        # Line-level
        lines = [l.strip().lower() for l in response.split('\n') if l.strip()]
        if len(lines) >= 3:
            lc = Counter(lines)
            ld = sum(v-1 for v in lc.values() if v > 1)
            rep_pen += min(ld * 3, 15)
        score -= rep_pen

        # === Filler/bloat penalty ===
        bloat = [
            r'\bit is important to note\b', r'\bit should be noted\b',
            r'\bit goes without saying\b', r'\bneedless to say\b',
            r'\bin general\b', r'\bgenerally speaking\b',
            r'\bbasically\b', r'\bessentially\b',
            r'\bthings like\b', r'\bstuff\b',
        ]
        bloat_count = sum(len(re.findall(p, rl)) for p in bloat)
        if short_pref and bloat_count > 0:
            score -= bloat_count * 3
        else:
            score -= min(bloat_count * 1.2, 5)

        # === Echo penalty ===
        q_topic = set(w for w in qw if w not in STOP and len(w) > 2)
        r_set = set(content) if content else set()
        if q_topic and r_set:
            cov = len(q_topic & r_set) / len(q_topic)
            score += cov * 8
            # If response is mostly query echo with no novel content
            novel = len(r_set - q_topic) / max(len(r_set), 1)
            if novel < 0.2 and n < 20:
                score -= 6

        # === Specificity bonus (for explain/list queries) ===
        if long_pref or is_list:
            nums = len(re.findall(r'\b\d+\b', response))
            score += min(nums * 0.6, 4)
            proper = len(re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response))
            score += min(proper * 0.4, 4)

        # === Tautology penalty ===
        if re.search(r'\bthe purpose of\b.*\bis to reject the .*\bwhen it is false', rl):
            score -= 8
        # "X is X" definition style
        if re.search(r'a (\w+) is a \1\b', rl):
            score -= 6

        # === Truncation ===
        if response[-1] not in '.!?")]}>' and n > 30:
            score -= 4

        return max(0.0, min(100.0, round(score, 2)))
    except Exception:
        return 30.0
