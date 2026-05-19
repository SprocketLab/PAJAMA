def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        q = (query or "").strip()
        q_lower = q.lower()

        resp = response.strip()
        if len(resp) == 0:
            return 0.0
        if len(resp) < 3:
            return 0.8

        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)

        score = 5.0  # midpoint start

        # ===== 1. Length adequacy =====
        is_short_q = bool(re.search(r'\b(name|identify|which|classify|list|biggest|next|capital)\b', q_lower))
        if n_words < 3:
            score += 0.5 if is_short_q else -2.0
        elif n_words < 8:
            score += 0.3 if is_short_q else -0.5
        elif 8 <= n_words <= 250:
            score += 0.5
        elif n_words > 400:
            score -= 0.3

        # ===== 2. Named-entity / specificity signals =====
        # Mid-sentence capitalized words (proper nouns)
        mid_caps = re.findall(r'(?<=\s)([A-Z][a-z]{2,})', resp)
        common_starters = {'The','This','That','These','Those','However','Moreover','Also','But',
                          'When','Where','What','How','Why','Once','After','Before','Since',
                          'Although','Yet','Still','Here','There','Some','Many','Most','Each',
                          'Every','Any','All','Such','Other','Sure','Yes','No','Note',
                          'Output','Input','Question','Answer','Instruction','Example'}
        proper = [w for w in mid_caps if w not in common_starters]
        proper_density = len(proper) / max(n_words, 1)
        score += min(proper_density * 12, 1.2)

        # Multi-word entities
        multi_ent = re.findall(r'(?<=\s|^)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', resp)
        score += min(len(multi_ent) * 0.2, 0.8)

        # Numbers
        nums = re.findall(r'\b\d+[\d,.\-/]*\b', resp)
        if 0 < len(nums) <= 6:
            score += min(len(nums) * 0.15, 0.6)
        elif len(nums) > 12:
            score -= 0.3  # excessive unsourced

        # Dates
        years = re.findall(r'\b(1[5-9][0-9]{2}|20[0-2][0-9])\b', resp)
        score += min(len(years) * 0.15, 0.6)

        # ===== 3. Hedging (moderate is good) =====
        hedging = [
            r'\bgenerally\b', r'\btypically\b', r'\bcan vary\b', r'\bdepending on\b',
            r'\bapproximately\b', r'\bestimated\b', r'\bmay\b', r'\busually\b',
            r'\boften\b', r'\bin some cases\b', r'\btends to\b', r'\blikely\b',
        ]
        rl = resp.lower()
        hedge_n = sum(len(re.findall(p, rl)) for p in hedging)
        if 0 < hedge_n <= 5:
            score += hedge_n * 0.15

        # ===== 4. Explanatory connectors =====
        explanatory = [
            r'\bbecause\b', r'\btherefore\b', r'\bas a result\b', r'\bsuch as\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bhowever\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bincluding\b', r'\bthus\b', r'\bsince\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
        ]
        exp_n = sum(len(re.findall(p, rl)) for p in explanatory)
        score += min(exp_n * 0.18, 1.0)

        # ===== 5. Red flags - hallucination indicators =====
        # Absolute claims
        abs_claims = len(re.findall(
            r'\b(?:always|never|every single|100%|guaranteed|undeniable|absolutely no)\b',
            resp, re.IGNORECASE))
        score -= min(abs_claims * 0.25, 1.0)

        # Sensationalism
        sens = len(re.findall(
            r'\b(?:shocking|unbelievable|mind-blowing|cover-up|conspiracy|hoax|sheeple)\b',
            resp, re.IGNORECASE))
        score -= sens * 0.5

        # ===== 6. Repetition =====
        if n_words >= 6:
            lower_words = [w.lower() for w in words]
            tg = [' '.join(lower_words[i:i+3]) for i in range(len(lower_words)-2)]
            tc = Counter(tg)
            rep_t = sum(c-1 for c in tc.values() if c > 1)
            if tg:
                rep_ratio = rep_t / len(tg)
                score -= min(rep_ratio * 8, 2.0)

        # Repeated lines/sentences
        sent_lc = [s.lower().strip() for s in sentences if len(s.strip()) > 8]
        if len(sent_lc) >= 2:
            sc = Counter(sent_lc)
            dup = sum(c-1 for c in sc.values() if c > 1)
            score -= min(dup * 0.5, 2.0)

        # ===== 7. Template leakage (CRITICAL DENOISER) =====
        tmpl_hits = len(re.findall(
            r'\b(?:Instruction|Input|Output|Question|Answer|Example|Note)\s*:',
            resp, re.IGNORECASE))
        if tmpl_hits >= 2:
            score -= min(tmpl_hits * 0.6, 3.0)

        # ===== 8. Question-echoing penalty (CRITICAL DENOISER) =====
        q_in_resp = resp.count('?')
        q_sents = sum(1 for s in sentences if '?' in s)
        if n_sents >= 2 and q_sents / n_sents > 0.5:
            score -= 2.0
        elif n_sents >= 2 and q_sents / n_sents > 0.35:
            score -= 1.0
        elif q_in_resp >= 4 and q_in_resp / max(n_sents, 1) > 0.5:
            score -= 1.5

        # ===== 9. Relevance =====
        stops = {'the','a','an','is','are','was','were','be','been','have','has','had',
                 'do','does','did','will','would','could','should','may','might','can',
                 'to','of','in','for','on','with','at','by','from','as','about','what',
                 'which','who','how','when','where','why','this','that','it','its','i',
                 'you','your','we','our','they','them','and','or','but','not','no','so',
                 'if','also'}
        q_words = set(w.lower().strip('.,!?;:()[]{}"\'-') for w in q.split())
        q_content = {w for w in q_words if w not in stops and len(w) > 2}
        r_words = set(w.lower().strip('.,!?;:()[]{}"\'-') for w in words)
        if q_content:
            overlap = len(q_content & r_words) / len(q_content)
            if overlap >= 0.5:
                score += 0.8
            elif overlap >= 0.3:
                score += 0.4
            elif overlap == 0 and not is_short_q:
                score -= 1.0

        # ===== 10. Code/HTML noise when not asked =====
        asks_code = any(kw in q_lower for kw in
            ['code','program','script','python','html','css','javascript','function','debug','c++'])
        if not asks_code:
            html_n = len(re.findall(r'<[a-zA-Z/][^>]*>', resp))
            if html_n > 3:
                score -= 1.0
            code_n = len(re.findall(r'(?:#include|import\s+\w+|def\s+\w+\(|public\s+class)', resp))
            if code_n >= 2:
                score -= 1.0

        # ===== 11. Non-ASCII bleed =====
        non_ascii = sum(1 for c in resp if ord(c) > 127)
        nonascii_ratio = non_ascii / max(len(resp), 1)
        q_nonascii = sum(1 for c in q if ord(c) > 127) / max(len(q), 1) if q else 0
        if nonascii_ratio > 0.08 and q_nonascii < 0.05:
            score -= min(nonascii_ratio * 5, 2.5)

        # ===== 12. Off-topic drift =====
        if len(sentences) >= 4 and q_content:
            mid = len(sentences) // 2
            second_half = ' '.join(sentences[mid:]).lower()
            second_words = set(re.findall(r'[a-z]+', second_half))
            if not (second_words & q_content) and len(second_half) > 50:
                score -= 0.8

        # ===== 13. Completeness =====
        if resp[-1] not in '.!?"\')]}' and n_words > 15:
            score -= 0.3

        score = max(0.0, min(10.0, score))
        return round(score, 2)

    except Exception:
        return 4.0
