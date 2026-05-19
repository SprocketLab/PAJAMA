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
            return 1.0

        rl = resp.lower()
        ql = query.lower()

        words_alpha = re.findall(r"[a-zA-Z']+", resp)
        wc = len(words_alpha)
        if wc == 0:
            return 1.0

        sents = [s.strip() for s in re.split(r'[.!?]+', resp) if s.strip()]
        sc = max(len(sents), 1)

        STOP = {
            'the','a','an','is','are','was','were','be','been','being','have','has','had',
            'do','does','did','will','would','could','should','may','might','can','to',
            'of','in','for','on','with','at','by','from','as','into','through','and','but',
            'or','not','so','if','then','than','too','very','just','that','this','these',
            'those','it','its','i','me','my','we','our','you','your','he','him','his',
            'she','her','they','them','their','what','which','who','when','where','why',
            'how','about','up','out','also','like','am','some','any','all','no','one',
            're','ve','ll','m','s','d','t'
        }
        content_w = [w.lower() for w in words_alpha if w.lower() not in STOP and len(w) > 2]
        info_density = len(content_w) / max(wc, 1)

        score = 50.0

        # === Information density ===
        score += (info_density - 0.40) * 30  # +/- around 0.40 baseline

        # === Type-token (vocabulary richness) ===
        if content_w:
            ttr = len(set(content_w)) / len(content_w)
            score += (ttr - 0.5) * 12

        # === Filler / weasel penalties ===
        fillers = [
            r'\bit is worth noting\b', r'\bit should be noted\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bat the end of the day\b', r'\bneedless to say\b',
            r'\bas a matter of fact\b', r'\bbasically\b', r'\bactually\b',
            r'\bliterally\b', r'\bobviously\b', r'\bsimply put\b',
            r'\bi think that\b', r'\bin my opinion\b',
            r'\bthat being said\b', r'\bhaving said that\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bquite\b', r'\bsomewhat\b',
        ]
        fc = sum(len(re.findall(p, rl)) for p in fillers)
        score -= min(fc * 1.5, 10)

        # === Sentence length appropriateness ===
        sent_lens = [len(re.findall(r"[a-zA-Z']+", s)) for s in sents]
        sent_lens = [s for s in sent_lens if s > 0]
        if sent_lens:
            avg_sl = sum(sent_lens) / len(sent_lens)
            if 8 <= avg_sl <= 25:
                score += 4
            elif avg_sl > 40:
                score -= 4
            elif avg_sl < 4:
                score -= 2

        # === Directness: opening pleasantry penalty ===
        first_chunk = rl[:80]
        if re.match(r'^(sure|of course|great question|that\'s a great|hello|hi |hey )', first_chunk):
            score -= 3
        if re.match(r'^(thank you for|i\'?d be happy to)', first_chunk):
            score -= 2
        if re.match(r'^(i must point out|i cannot|i\'?m not able)', first_chunk):
            score -= 5

        # === Query-response relevance ===
        q_cw = [w.lower() for w in re.findall(r'[a-zA-Z]+', query) if w.lower() not in STOP and len(w) > 2]
        if q_cw:
            qc_set = set(q_cw)
            rc_set = set(content_w)
            cov = len(qc_set & rc_set) / len(qc_set)
            score += cov * 12
        else:
            cov = 0.5

        # === Length calibration based on query complexity ===
        q_wc = len(query.split())
        # Detect "simple factual / opinion" queries that warrant short answers
        simple_factual = (
            q_wc < 20 and (
                re.search(r'\bname (?:a|one|some)\b', ql) or
                re.search(r'\bwhat (?:is|are) the\b', ql) or
                re.search(r'\bwhich (?:is|are)\b', ql) or
                re.search(r'\bwho (?:is|wrote|composed|invented)\b', ql) or
                re.search(r'\bsuggest (?:a|some|one)\b', ql) or
                re.search(r'\bfavorite\b', ql) or
                re.search(r'\bbest (?:cut|way|method)\b', ql) or
                resp.startswith('Answer:') or
                'option' in ql.lower()
            )
        )

        if simple_factual:
            # Short direct answers are GOOD here
            if 1 <= wc <= 30:
                score += 8
            elif wc <= 60:
                score += 4
            elif wc > 200:
                score -= 6  # bloat
        else:
            # General queries: moderate length preferred
            if wc < 8:
                score -= 5
            elif wc < 25:
                score += 1
            elif wc < 80:
                score += 5
            elif wc < 250:
                score += 6
            elif wc < 500:
                score += 3
            else:
                score -= 2

        # === Repetition penalty ===
        if len(content_w) >= 6:
            trigrams = [tuple(content_w[i:i+3]) for i in range(len(content_w)-2)]
            tc = Counter(trigrams)
            reps = sum(c-1 for c in tc.values() if c > 1)
            score -= min(reps * 0.8, 6)

        # === Concrete vs vague balance ===
        concrete = len(re.findall(
            r'\b(?:specifically|for example|for instance|such as|namely|'
            r'in particular|e\.g\.|i\.e\.)\b', rl))
        score += min(concrete * 1.5, 5)

        # === Gates ===

        # Pure question-back / clarification responses
        if resp.count('?') >= 1 and wc < 35 and not simple_factual:
            # Asking clarification is usually worse than answering
            score -= 5

        # Bot/template
        if re.search(r'welcome to /r/|i am a bot|please read our rules', rl):
            score -= 15

        # Translation gate
        if re.search(r'\btranslat(?:e|ion)\b|romanian:|spanish:', ql):
            non_ascii = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii < 4 and wc > 15:
                score -= 18

        # Confident-but-wrong meta penalty: very specific made-up technical details
        # (heuristic only - excessive fake-specific claims)
        if re.search(r'\b(?:fake|invented|made-up)\b', rl):
            score -= 3

        # Gibberish/garbled detector
        garbage = len(re.findall(r'[^\x00-\x7F]', resp))
        if garbage / max(len(resp), 1) > 0.18 and 'translate' not in ql:
            score -= 12

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 50.0
