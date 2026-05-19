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
        if len(resp) < 8:
            return 5.0

        rl = resp.lower()
        ql = query.lower()
        words = resp.split()
        wc = len(words)
        sents = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 3]
        sc = max(len(sents), 1)

        score = 50.0

        # === Length substantiveness ===
        if wc < 5:
            score -= 12
        elif wc < 15:
            score -= 3
        elif wc < 40:
            score += 1
        elif wc < 100:
            score += 5
        elif wc < 250:
            score += 7
        elif wc < 500:
            score += 5
        else:
            score += 2

        # === Specific factual markers ===
        years = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', resp)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', resp)
        score += min((len(years) + len(dates)) * 1.5, 7)

        measured_nums = re.findall(
            r'\b\d+\.?\d*\s*(?:%|percent|million|billion|kg|lbs?|miles?|km|'
            r'hours?|minutes?|degrees?|ft|meters?|years?)\b', rl)
        score += min(len(measured_nums) * 1.4, 6)

        # === Proper nouns ===
        propn = 0
        for s in sents:
            ws = s.split()
            for w in ws[1:]:
                cw = re.sub(r'[^a-zA-Z]', '', w)
                if cw and cw[0].isupper() and len(cw) > 1 and not cw.isupper():
                    propn += 1
        score += min(propn * 0.7, 7)

        # === Citation/reference patterns ===
        ital = len(re.findall(r'\*[A-Z][^*]+\*', resp))
        qt = len(re.findall(r'"[A-Z][^"]{3,}"', resp))
        cites = len(re.findall(r'\([^)]*\d{4}[^)]*\)', resp))
        urls = len(re.findall(r'https?://\S+', resp))
        attrib = len(re.findall(
            r'\b(?:according to|as noted by|research (?:shows|suggests)|'
            r'stud(?:y|ies) (?:show|suggest))', rl))
        score += min((ital + qt + cites + urls + attrib) * 2.0, 10)

        # === Hedging calibration ===
        hedges = len(re.findall(
            r'\b(?:might|may|could|possibly|perhaps|likely|unlikely|probably|'
            r'tends? to|generally|typically|often|usually|approximately|roughly|'
            r'it seems|appears to|suggests?|i think|i believe|as far as i know|'
            r'in my experience|arguably|plausibly)\b', rl))
        hr = hedges / max(sc, 1)
        if 0.05 <= hr <= 1.0:
            score += min(hedges * 1.0, 6)
        elif hr > 1.8:
            score -= 2

        # === Anti-hallucination: absolute claims ===
        absolutes = len(re.findall(
            r'\b(?:always|never|every single|undeniably|definitely|absolutely|'
            r'without (?:a )?doubt|100% (?:certain|sure|proven)|guaranteed|'
            r'everyone knows|it is a fact that)\b', rl))
        score -= min(absolutes * 1.5, 6)

        sensational = len(re.findall(
            r'\b(?:shocking|bombshell|mind-blowing|conspiracy|cover-?up|'
            r'wake up|sheeple|big pharma|deep state|exposed|'
            r'they don\'?t want you to know|secret(?:ly)? agenda)\b', rl))
        score -= sensational * 4

        # === Overly precise unsourced stats ===
        precise_pct = re.findall(r'\b\d{1,3}\.\d{2,}%', resp)
        if len(precise_pct) > 0 and attrib == 0 and ital == 0 and qt == 0:
            score -= min(len(precise_pct) * 2.0, 6)

        # === Causal/explanatory connectives ===
        causal = len(re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as a result|due to|'
            r'this means|for example|for instance|such as|specifically|in particular)\b',
            rl))
        score += min(causal * 0.9, 6)

        # === Contrast / nuance markers ===
        contrast = len(re.findall(
            r'\b(?:however|although|on the other hand|nevertheless|yet|'
            r'in contrast|whereas|while|trade-?off)\b', rl))
        score += min(contrast * 1.2, 5)

        # === Personal experience (mildly positive in many contexts) ===
        exp = len(re.findall(
            r'\b(?:in my experience|i\'?ve (?:seen|found|noticed|worked)|'
            r'when i was|personally)\b', rl))
        score += min(exp * 1.0, 3)

        # === Sensational/exclamation overuse ===
        exclam = resp.count('!')
        if exclam > 3:
            score -= min((exclam - 3) * 0.8, 5)

        # === ALL CAPS shouting ===
        caps_words = re.findall(r'\b[A-Z]{4,}\b', resp)
        known_acronyms = {'NASA','HTTP','HTML','SQL','HVAC','HIV','AIDS','API','URL',
                          'SELECT','FROM','JOIN','WHERE','CREATE','TABLE','NULL',
                          'INSERT','LEFT','RIGHT','COMMENT','MIRI','VARCHAR'}
        non_ac = [w for w in caps_words if w not in known_acronyms]
        score -= min(len(non_ac) * 1.0, 6)

        # === Filler boilerplate ===
        boilerplate = len(re.findall(
            r'\b(?:i hope this helps|hope that helps|happy to help|'
            r'great question|good question|interesting question)\b', rl))
        score -= min(boilerplate * 1.5, 5)

        # === Bot template ===
        if re.search(r'welcome to /r/|please read our rules|i am a bot', rl):
            score -= 15

        # === Query topical alignment ===
        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on',
                'and','but','or','not','it','its','this','that','these','those','i','me',
                'my','we','our','you','your','what','which','how','why','when','where',
                'with','from','as','have','has','had','do','does','did','will','would',
                'can','could','should','about'}
        q_cw = set(w for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 2)
        r_cw = set(w for w in re.findall(r'[a-z]+', rl) if w not in STOP and len(w) > 2)
        if q_cw:
            cov = len(q_cw & r_cw) / len(q_cw)
            score += cov * 8

        # === Information density ===
        if r_cw:
            density = len(r_cw) / max(wc, 1)
            score += min((density - 0.3) * 18, 5)

        # === Gates ===
        # Translation tasks
        if re.search(r'\btranslat(?:e|ion)\b|romanian:|spanish:', ql):
            non_ascii = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii < 4 and wc > 15:
                score -= 18

        # Confident wrong-style structure: long pseudo-formal answer to factual Q
        # without any actual entity/date/citation, but using "Step 1, Step 2..."
        if re.search(r'\bstep \d+:', rl) and wc > 80:
            entities_present = (propn > 2 or len(years) > 0 or len(ital) > 0 or attrib > 0)
            if not entities_present:
                score -= 4  # mild penalty: pseudo-reasoning without specifics

        # Made-up technical detail trap: "VIN" + code expansion fantasy etc.
        # Hard to detect reliably; skip.

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 50.0
