def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query: query = ""

        resp = response.strip()
        q = query.strip()
        q_lower = q.lower()

        if len(resp) == 0:
            return 0.0
        if len(resp) < 2:
            return 1.0

        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)

        # Start at a calibrated baseline - we WILL spread scores
        score = 5.5

        # ===== Detect query type =====
        is_short_q = bool(re.search(
            r'\b(name|identify|which|classify|list|biggest|largest|next number|how many|capital)\b',
            q_lower))
        is_yes_no = bool(re.search(r'^\s*(is |are |do |does |can |could |should |would |will |has |have |did )', q_lower))
        is_explain_q = bool(re.search(r'\b(why|how|explain|describe|tell me about|what is|what are|what does)\b', q_lower))
        asks_code = any(kw in q_lower for kw in
            ['code','program','script','python','html','css','javascript','function','debug','c++','java','excel','rewrite'])

        # ===== Signal 1: Question-echoing detection (MAJOR DENOISER) =====
        # Count question-like sentences
        q_marks = resp.count('?')
        question_sents = sum(1 for s in sentences if s.rstrip().endswith('?') or '?' in s[-2:])

        # Lines that start with interrogatives
        interrog_starts = 0
        for s in sentences:
            sw = s.lstrip().split()
            if sw:
                first = sw[0].lower().rstrip('.,;:!?')
                if first in ('what','who','where','when','why','how','which','can','could','would','should','is','are','do','does'):
                    interrog_starts += 1

        question_echo_score = 0  # 0 = normal, higher = more echo-like
        if n_sents >= 2:
            q_ratio = question_sents / n_sents
            interrog_ratio = interrog_starts / n_sents
            if q_ratio >= 0.5 or interrog_ratio >= 0.5:
                question_echo_score = 3.0
            elif q_ratio >= 0.3 or interrog_ratio >= 0.35:
                question_echo_score = 1.8
            elif q_ratio >= 0.2 or interrog_ratio >= 0.25:
                question_echo_score = 0.8

        # Special: if query is a question and the response has more ? than . it's bad
        if '?' in q and q_marks > resp.count('.') and q_marks >= 3:
            question_echo_score = max(question_echo_score, 2.0)

        score -= question_echo_score

        # ===== Signal 2: Template leakage (MAJOR DENOISER) =====
        # "Instruction:", "Input:", "Output:", "Question:", "Answer:" hits
        template_patterns = re.findall(
            r'\b(?:Instruction|Input|Output|Question|Answer|Example|Note)\s*:',
            resp, re.IGNORECASE)
        tmpl_n = len(template_patterns)

        template_penalty = 0
        if tmpl_n >= 4:
            template_penalty = 3.5
        elif tmpl_n == 3:
            template_penalty = 2.5
        elif tmpl_n == 2:
            template_penalty = 1.5

        score -= template_penalty

        # ===== Signal 3: Repetition / spam patterns =====
        rep_penalty = 0
        if n_words >= 6:
            lw = [w.lower() for w in words]
            tg = [' '.join(lw[i:i+3]) for i in range(len(lw)-2)]
            if tg:
                tc = Counter(tg)
                rep_t = sum(c-1 for c in tc.values() if c > 1)
                rep_penalty += min((rep_t / len(tg)) * 8, 2.5)

        # Repeated sentence start words (like "Zealously" pattern in case 13)
        if len(sentences) >= 4:
            first_words = []
            for s in sentences:
                sw = s.split()
                if sw: first_words.append(sw[0].lower())
            if first_words:
                fc = Counter(first_words)
                most = fc.most_common(1)[0]
                if most[1] >= 4 and most[1] / len(first_words) >= 0.4:
                    rep_penalty += 1.5

        # Repeated lines
        lines = [l.strip().lower() for l in resp.split('\n') if l.strip()]
        if len(lines) >= 2:
            lc = Counter(lines)
            dups = sum(c-1 for c in lc.values() if c > 1)
            if dups > 0:
                rep_penalty += min(dups * 0.4, 1.5)

        score -= min(rep_penalty, 3.5)

        # ===== Signal 4: Off-topic drift =====
        # Particularly common: response addresses query for first few sentences,
        # then drifts into unrelated content
        stops = {'the','a','an','is','are','was','were','be','to','of','in','for','on','with',
                 'at','by','from','as','it','its','this','that','they','them','and','or','but',
                 'not','no','so','if','what','which','who','how','when','where','why','can',
                 'could','would','should','may','might','will','i','you','your','we','our','any','also'}
        q_words = set(re.findall(r'[a-z]+', q_lower)) - stops
        q_content_words = {w for w in q_words if len(w) > 2}

        drift_penalty = 0
        if len(sentences) >= 4 and q_content_words:
            third = max(len(sentences) // 3, 1)
            front = ' '.join(sentences[:third]).lower()
            back = ' '.join(sentences[-third:]).lower()
            front_words = set(re.findall(r'[a-z]+', front))
            back_words = set(re.findall(r'[a-z]+', back))
            front_rel = len(front_words & q_content_words)
            back_rel = len(back_words & q_content_words)
            if front_rel >= 1 and back_rel == 0 and len(back) > 80:
                drift_penalty = 1.5

        score -= drift_penalty

        # ===== Signal 5: Code/HTML when not asked =====
        if not asks_code:
            html_n = len(re.findall(r'<[a-zA-Z/][^>]*>', resp))
            code_n = len(re.findall(
                r'(?:#include|import\s+\w+|def\s+\w+\(|public\s+class|void\s+\w+\(|function\s*\()',
                resp))
            if html_n >= 4:
                score -= 1.5
            elif html_n >= 2:
                score -= 0.6
            if code_n >= 3:
                score -= 2.0
            elif code_n >= 1:
                score -= 0.7

        # ===== Signal 6: Non-ASCII bleed =====
        non_ascii = sum(1 for c in resp if ord(c) > 127)
        nonascii_ratio = non_ascii / max(len(resp), 1)
        q_nonascii_ratio = sum(1 for c in q if ord(c) > 127) / max(len(q), 1) if q else 0
        if nonascii_ratio > 0.1 and q_nonascii_ratio < 0.05:
            score -= min(nonascii_ratio * 6, 3.0)

        # ===== Signal 7: Direct-answer reward (positive denoiser) =====
        # If query is short factoid and response is a short clean answer, REWARD it
        if (is_short_q or is_yes_no) and n_words <= 12 and tmpl_n == 0 and question_echo_score == 0:
            # Check it's alphabetic content (not gibberish)
            alpha = sum(1 for c in resp if c.isalpha())
            ar = alpha / max(len(resp), 1)
            if ar > 0.6:
                score += 1.5

        # ===== Signal 8: Relevance =====
        r_words = set(re.findall(r'[a-z]+', resp.lower()))
        if q_content_words:
            overlap = len(q_content_words & r_words) / len(q_content_words)
            if overlap >= 0.5:
                score += 1.2
            elif overlap >= 0.3:
                score += 0.6
            elif overlap == 0 and n_words > 5 and not is_short_q:
                score -= 1.5

        # ===== Signal 9: Substance =====
        # Detect "Output:" that the model wrote after answering - common in case 26
        if re.search(r'Output\s*:\s*$', resp, re.MULTILINE):
            score -= 0.5
        # Detect explicit "Input:" empty fields
        if re.search(r'^Input\s*:\s*$', resp, re.MULTILINE):
            score -= 0.5

        # ===== Signal 10: First-line direct address =====
        if sentences:
            fs = sentences[0]
            # Bonus if first sentence is a declarative answer addressing query
            if not fs.endswith('?') and len(fs.split()) >= 2 and q_content_words:
                fs_words = set(re.findall(r'[a-z]+', fs.lower()))
                if fs_words & q_content_words:
                    score += 0.6

        # ===== Length sanity =====
        if n_words <= 1 and not (is_short_q or is_yes_no):
            score = min(score, 1.5)
        if n_words > 400 and (tmpl_n > 2 or question_echo_score > 1):
            score -= 0.5

        # Alpha ratio sanity
        alpha_chars = sum(1 for c in resp if c.isalpha())
        ar = alpha_chars / max(len(resp), 1)
        if ar < 0.25 and not asks_code:
            score -= 2.0

        score = max(0.0, min(10.0, score))
        return round(score, 2)

    except Exception:
        return 4.0
