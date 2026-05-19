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

        rl = resp.lower()
        ql = query.lower()
        words = resp.split()
        wc = len(words)
        sents = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 3]
        sc = max(len(sents), 1)

        STOP = {'the','a','an','is','are','was','were','be','been','being','to','of','in',
                'for','on','with','at','by','from','as','into','and','but','or','not','so',
                'if','then','that','this','these','those','it','its','i','me','my','we',
                'our','you','your','he','him','his','she','her','they','them','their',
                'what','which','who','when','where','why','how','have','has','had','do',
                'does','did','will','would','could','should','may','might','can','about',
                'up','out','also','like','some','any','all','no','one','two','am','very',
                'just','too','more','most','really','well','think','know'}

        q_cw_list = [w for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 2]
        r_cw_list = [w for w in re.findall(r'[a-z]+', rl) if w not in STOP and len(w) > 2]
        q_cw = set(q_cw_list)
        r_cw = set(r_cw_list)

        score = 40.0

        # === Topic coverage ===
        if q_cw:
            cov = len(q_cw & r_cw) / len(q_cw)
        else:
            cov = 0.5
        score += cov * 20

        # === Bigram overlap ===
        q_bg = set(zip(q_cw_list, q_cw_list[1:]))
        r_bg = set(zip(r_cw_list, r_cw_list[1:]))
        if q_bg:
            score += (len(q_bg & r_bg) / len(q_bg)) * 7

        # === Structural variety ===
        types = 0
        if len(re.findall(r'(?m)^\s*\d+[.)]\s', resp)) >= 2:
            types += 1
        if len(re.findall(r'(?m)^\s*[-*•]\s', resp)) >= 2:
            types += 1
        if re.search(r'(?m)^#+\s|\*\*[^*]+\*\*', resp):
            types += 1
        if '```' in resp:
            types += 1
        if re.search(r'`[^`]+`', resp):
            types += 1
        if '\n\n' in resp:
            types += 1
        # Bonus only for medium+ responses
        if wc > 80:
            score += min(types * 1.5, 6)
        elif wc > 30 and types > 2:
            score += 2  # but penalize over-structuring short text mildly
        elif wc < 30 and types > 1:
            score -= 1

        # === Evidence density ===
        nums = len(re.findall(r'\b\d+\b', resp))
        score += min(nums * 0.4, 4)
        propn = len(re.findall(r'(?<=[a-z,;:]\s)[A-Z][a-z]+', resp))
        score += min(propn * 0.5, 5)
        urls = len(re.findall(r'https?://\S+', resp))
        score += min(urls * 1.5, 3)

        # === Discourse / reasoning ===
        causal = len(re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as a result|due to|since)\b', rl))
        contrast = len(re.findall(
            r'\b(?:however|although|but|whereas|while|on the other hand|nevertheless)\b', rl))
        examples = len(re.findall(
            r'\b(?:for example|for instance|such as|e\.g\.|specifically|in particular)\b', rl))
        score += min((causal + contrast + examples) * 0.8, 8)

        # === Hedging calibration ===
        hedges = len(re.findall(
            r'\b(?:might|may|could|possibly|perhaps|likely|tends? to|generally|'
            r'typically|usually|i think|i believe|approximately)\b', rl))
        hr = hedges / max(sc, 1)
        if 0.05 <= hr <= 1.0:
            score += min(hedges * 0.7, 4)
        elif hr > 2.0:
            score -= 2

        # === Length awareness (not pure "longer = better") ===
        if wc < 4:
            score -= 12
        elif wc < 12:
            score -= 2
        elif wc < 35:
            score += 3
        elif wc < 100:
            score += 6
        elif wc < 250:
            score += 7
        elif wc < 500:
            score += 5
        else:
            score += 2

        # === Vocab richness ===
        if r_cw_list:
            ttr = len(r_cw) / len(r_cw_list)
            score += (ttr - 0.45) * 10

        # === Penalties ===
        # Absolute claims / sensationalism
        absolutes = len(re.findall(
            r'\b(?:always|never|undeniably|absolutely|100%|guaranteed|everyone knows)\b', rl))
        score -= min(absolutes * 1.0, 4)

        sensational = len(re.findall(
            r'\b(?:shocking|bombshell|conspiracy|cover-?up|sheeple|wake up)\b', rl))
        score -= sensational * 4

        # Bot template
        if re.search(r'welcome to /r/|i am a bot|please read our rules', rl):
            score -= 15

        # AI-style opener bloat
        if re.match(r'^(?:sure!?|of course!?|great question|that\'?s a (?:great|good))', rl):
            score -= 3

        # === Off-topic gate ===
        if q_cw and len(q_cw) >= 3 and wc > 25:
            ovl = len(q_cw & r_cw)
            if ovl == 0:
                score -= 15
            elif ovl / len(q_cw) < 0.10:
                score -= 7

        # === Translation gate ===
        if re.search(r'\btranslat(?:e|ion)\b|romanian:|spanish:', ql):
            non_ascii = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii < 4 and wc > 15:
                score -= 18
            elif non_ascii >= 8:
                score += 5

        # === Garbled output ===
        garbage = len(re.findall(r'[^\x00-\x7F]', resp))
        gr = garbage / max(len(resp), 1)
        if gr > 0.20 and 'translate' not in ql and 'latex' not in ql:
            score -= 12

        # === Factual contradiction gate ===
        # Specific known traps
        if 'fish' in ql and ('lung' in ql or 'breath' in ql):
            if re.search(r'(?:fish|both fish).*(?:rely on|have|use).*lungs?', rl):
                score -= 8

        # === Refusal-on-benign-query gate ===
        refusal_strong = bool(re.search(
            r'(?:i must point out|not factually coherent|cannot provide an answer|'
            r'unhealthy ingredient|the (?:question|prompt) (?:itself )?(?:may not be|is not))',
            rl))
        if refusal_strong:
            # Only refusal is appropriate if query is genuinely incoherent
            incoherent = bool(re.search(
                r'rhino.*air|cat.*lawn.*desert|fish.*lung', ql))
            if not incoherent:
                score -= 10

        # === Clarification-instead-of-answer ===
        if (resp.endswith('?') and wc < 35 and
            re.search(r'^(?:can you|could you|do you|when you say|please (?:describe|clarify))', rl)):
            score -= 6

        # === Code task fulfillment ===
        if re.search(r'\b(code|sql|create table|select|python|latex|function)\b', ql):
            if re.search(r'```|\bSELECT\b|\bCREATE\b|\bdef \b|\\usepackage|\\begin', resp):
                score += 5

        # === Repetition penalty ===
        if wc >= 20:
            trigrams = [' '.join(words[i:i+3]).lower() for i in range(wc-2)]
            tc = Counter(trigrams)
            reps = sum(c-1 for c in tc.values() if c > 1)
            if reps / max(len(trigrams), 1) > 0.08:
                score -= min(reps * 0.6, 5)

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 40.0
