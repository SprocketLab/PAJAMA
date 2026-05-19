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
        if len(resp) < 5:
            return 1.0

        words = resp.split()
        wc = len(words)
        sents = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 3]
        sc = max(len(sents), 1)
        rl = resp.lower()
        ql = query.lower()

        score = 30.0

        # === Numeric/quantitative specificity ===
        years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', resp)
        pcts = re.findall(r'\b\d+\.?\d*\s*%', resp)
        money = re.findall(r'[\$€£]\s*\d+', resp)
        measures = re.findall(
            r'\b\d+\.?\d*\s*(?:mg|kg|g|lb|oz|ml|l|km|mi|miles?|ft|cm|mm|m|'
            r'hours?|minutes?|seconds?|days?|weeks?|months?|years?|degrees?|mph|kph|gb|mb|tb)\b',
            resp, re.IGNORECASE)
        plain_nums = re.findall(r'\b\d{2,}\b', resp)

        score += min(len(years) * 1.8, 6)
        score += min(len(pcts) * 1.5, 5)
        score += min(len(money) * 1.5, 4)
        score += min(len(measures) * 1.5, 6)
        score += min(len(plain_nums) * 0.4, 4)

        # === Named entities (capitalized mid-sentence) ===
        propn = 0
        for s in sents:
            ws = s.split()
            for w in ws[1:]:
                cw = re.sub(r'[^a-zA-Z]', '', w)
                if cw and cw[0].isupper() and len(cw) > 1 and not cw.isupper():
                    propn += 1
        score += min(propn * 0.6, 8)

        # Multi-word proper nouns (likely real entities)
        multi_prop = re.findall(r'(?<![.!?]\s)[A-Z][a-z]+\s+[A-Z][a-z]+', resp)
        score += min(len(multi_prop) * 1.5, 6)

        # === References / citations ===
        score += min(len(re.findall(r'\*[^*]{3,60}\*', resp)) * 2.0, 4)
        score += min(len(re.findall(r'"[A-Z][^"]{4,80}"', resp)) * 1.8, 4)
        score += min(len(re.findall(r'https?://\S+', resp)) * 2.0, 4)
        score += min(len(re.findall(r'\bu/\w+|\@\w+', resp)) * 1.5, 3)
        attrib = len(re.findall(
            r'\baccording to|\bresearch (?:shows|suggests|indicates)|'
            r'\bstud(?:y|ies) (?:show|suggest|indicate)', rl))
        score += min(attrib * 2.0, 4)

        # === Examples / concrete elaboration ===
        examples = len(re.findall(
            r'\b(?:for example|for instance|such as|e\.g\.|specifically|in particular|namely)\b',
            rl))
        score += min(examples * 1.8, 7)

        # === Causal/explanatory chains ===
        causal = len(re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as a result|due to|'
            r'this means|which means|leading to|resulting in)\b', rl))
        score += min(causal * 1.0, 6)

        # === Code blocks (especially relevant for technical queries) ===
        cb = len(re.findall(r'```', resp)) // 2
        ic = len(re.findall(r'`[^`]+`', resp))
        score += min(cb * 2.5 + ic * 0.5, 6)

        # === Vague language penalty ===
        vague = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bvarious things\b',
            r'\ba lot of\b', r'\btons of\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bgenerally speaking\b', r'\bfor the most part\b',
        ]
        vc = sum(len(re.findall(p, rl)) for p in vague)
        score -= min(vc * 1.5, 8)

        # === Anti-hallucination: suspiciously precise unsourced stats ===
        precise = re.findall(r'\b\d{1,3}\.\d{2,}%?', resp)
        if len(precise) > 1 and attrib == 0 and len(re.findall(r'\*|"', resp)) == 0:
            score -= min(len(precise) * 1.5, 6)

        # === Absolute / sensational language penalty ===
        absolute = len(re.findall(
            r'\b(?:always|never|undeniably|definitely|absolutely|100%|guaranteed|'
            r'everyone knows|obviously|clearly|without a doubt)\b', rl))
        score -= min(absolute * 0.8, 5)

        sensational = len(re.findall(
            r'\b(?:shocking|mind-blowing|conspiracy|cover-?up|sheeple|wake up|'
            r'they don\'?t want you to know)\b', rl))
        score -= sensational * 4

        # === Hedging calibration (some hedging is GOOD) ===
        hedges = len(re.findall(
            r'\b(?:might|may|could|possibly|perhaps|likely|approximately|roughly|'
            r'tends? to|generally|typically|usually|i think|i believe)\b', rl))
        hedge_ratio = hedges / max(sc, 1)
        if 0.1 <= hedge_ratio <= 1.2:
            score += min(hedges * 0.6, 4)
        elif hedge_ratio > 2.0:
            score -= 2  # over-hedging

        # === Length-aware: short concise answers can still be excellent ===
        if wc < 5:
            score -= 8
        elif wc < 15:
            score += 0
        elif wc < 40:
            score += 3
        elif wc < 120:
            score += 5
        elif wc < 300:
            score += 4
        else:
            score += 2

        # === Diversity bonus: multiple types of evidence ===
        types_present = sum([
            len(years) + len(pcts) + len(money) + len(measures) > 0,
            propn > 2,
            len(multi_prop) > 0,
            examples > 0,
            cb > 0 or ic > 0,
            causal > 1,
            attrib > 0,
        ])
        score += types_present * 1.2

        # === Gates ===
        # If response is essentially confidently WRONG on a math/logic check, downweight
        if re.search(r'\bconfidence:\s*\d+%', rl):
            # don't penalize confidence assertions per se, but if combined with no reasoning, neutral
            pass

        # Bot/meta responses
        if re.search(r'welcome to /r/|please read our rules|i am a bot', rl):
            score -= 12

        # Pure clarifying question
        if resp.endswith('?') and wc < 40 and resp.count('?') >= 1:
            # could still be useful but often suboptimal
            score -= 4

        # Translation task: penalize if no translation produced
        if re.search(r'\btranslat(?:e|ion)\b|romanian:|spanish:|french:|german:', ql):
            non_ascii_alpha = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii_alpha < 4 and wc > 15:
                score -= 18

        # Off-topic gibberish detector
        if len(re.findall(r'[^\x00-\x7F]', resp)) / max(len(resp), 1) > 0.2:
            if 'translate' not in ql and 'latex' not in ql:
                score -= 10

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 30.0
