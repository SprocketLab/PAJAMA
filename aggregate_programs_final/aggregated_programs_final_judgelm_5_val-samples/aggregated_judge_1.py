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
        q = query.strip()

        if len(resp) == 0:
            return 0.0
        if len(resp) < 2:
            return 0.5

        STOP = {
            'a','an','the','is','are','was','were','be','been','being','have','has','had',
            'do','does','did','will','would','could','should','may','might','shall','can',
            'to','of','in','for','on','with','at','by','from','as','into','through','during',
            'before','after','above','below','between','out','off','over','under','again',
            'further','then','once','here','there','when','where','why','how','all','both',
            'each','few','more','most','other','some','such','no','nor','not','only','own',
            'same','so','than','too','very','just','because','but','and','or','if','while',
            'about','up','down','that','this','these','those','it','its','i','me','my','we',
            'our','you','your','he','him','his','she','her','they','them','their','what',
            'which','who','whom','also','am','any'
        }

        def tok(t): return re.findall(r"[a-zA-Z']+", t.lower())
        def content(t): return [w for w in tok(t) if w not in STOP and len(w) > 1]

        q_tokens = tok(q)
        r_tokens = tok(resp)
        q_content = set(content(q))
        r_content = content(resp)
        r_content_set = set(r_content)

        n_words = len(r_tokens)
        if n_words == 0:
            return 0.5

        # ===== 1. Relevance: keyword overlap =====
        if q_content:
            kw_recall = len(q_content & r_content_set) / len(q_content)
        else:
            kw_recall = 0.5

        # Bigram overlap
        q_bigrams = set()
        q_cl = content(q)
        for i in range(len(q_cl) - 1):
            q_bigrams.add((q_cl[i], q_cl[i+1]))
        r_bigrams = set()
        for i in range(len(r_content) - 1):
            r_bigrams.add((r_content[i], r_content[i+1]))
        if q_bigrams:
            bg_recall = len(q_bigrams & r_bigrams) / len(q_bigrams)
        else:
            bg_recall = 0.0

        relevance = 0.65 * kw_recall + 0.35 * bg_recall

        # ===== 2. Length adequacy (with concise-answer awareness) =====
        # Detect query types where short answers are appropriate
        q_lower = q.lower()
        is_list_q = bool(re.search(r'\b(list|name|identify|which|classify|tell me if)\b', q_lower))
        is_factoid_q = bool(re.search(r'\b(who|what is the|when|where|how many|next number|capital|biggest|largest)\b', q_lower))
        is_yes_no = bool(re.search(r'^\s*(is |are |do |does |can |could |should |would |will )', q_lower))
        short_ok = is_list_q or is_factoid_q or is_yes_no

        if n_words < 3:
            length_score = 0.7 if short_ok else 0.15
        elif n_words < 8:
            length_score = 0.85 if short_ok else 0.45
        elif n_words < 20:
            length_score = 0.9
        elif n_words <= 150:
            length_score = 1.0
        elif n_words <= 300:
            length_score = 0.85
        else:
            length_score = 0.7

        # ===== 3. Repetition penalty =====
        rep_penalty = 0.0
        if n_words > 5:
            uniq_ratio = len(set(r_tokens)) / n_words
            if uniq_ratio < 0.3:
                rep_penalty += 0.4
            elif uniq_ratio < 0.5:
                rep_penalty += 0.2

        # Trigram repetition
        if len(r_tokens) >= 6:
            trig = [tuple(r_tokens[i:i+3]) for i in range(len(r_tokens)-2)]
            tc = Counter(trig)
            rep_trig = sum(c-1 for c in tc.values() if c > 1)
            if trig:
                trig_rep = rep_trig / len(trig)
                rep_penalty += min(trig_rep * 0.8, 0.4)

        # Repeated lines
        lines = [l.strip().lower() for l in resp.split('\n') if l.strip()]
        if len(lines) > 1:
            lc = Counter(lines)
            dup_lines = sum(c-1 for c in lc.values() if c > 1)
            if dup_lines > 0:
                rep_penalty += min(dup_lines * 0.1, 0.3)

        # ===== 4. Template leakage gate (CRITICAL) =====
        # Detect "Instruction:/Input:/Output:/Question:/Answer:" leakage
        template_hits = len(re.findall(
            r'\b(?:Instruction|Input|Output|Question|Answer|Example|Note)\s*:',
            resp, re.IGNORECASE))
        template_penalty = 0.0
        if template_hits >= 2:
            template_penalty = min(0.5, template_hits * 0.12)

        # ===== 5. Question-echo gate (CRITICAL) =====
        # Responses that mostly ask more questions instead of answering
        q_marks = resp.count('?')
        # Count sentences ending with ?
        sentences = re.split(r'[.!?\n]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        question_sentences = sum(1 for s in sentences if s.endswith('?') or '?' in s)
        echo_penalty = 0.0
        if len(sentences) >= 2:
            q_ratio = question_sentences / len(sentences)
            if q_ratio > 0.5:
                echo_penalty = 0.4
            elif q_ratio > 0.35:
                echo_penalty = 0.2
        # Also: many question marks total relative to length
        if q_marks >= 4 and q_marks / max(len(sentences), 1) > 0.4:
            echo_penalty = max(echo_penalty, 0.35)

        # ===== 6. Off-topic drift detection =====
        # Check coherence: does response stay on topic from start to finish?
        drift_penalty = 0.0
        if len(sentences) >= 4 and q_content:
            mid = len(sentences) // 2
            first_half_words = set()
            second_half_words = set()
            for s in sentences[:mid]:
                first_half_words.update(content(s))
            for s in sentences[mid:]:
                second_half_words.update(content(s))
            first_relevant = len(first_half_words & q_content)
            second_relevant = len(second_half_words & q_content)
            # If first half is on-topic but second half drifts away
            if first_relevant >= 1 and second_relevant == 0 and len(second_half_words) > 5:
                drift_penalty = 0.25

        # ===== 7. Non-ASCII bleed (garbled multi-language) =====
        non_ascii = sum(1 for c in resp if ord(c) > 127)
        nonascii_ratio = non_ascii / max(len(resp), 1)
        # Allow if query has non-ASCII (e.g. Hebrew query)
        q_nonascii = sum(1 for c in q if ord(c) > 127) / max(len(q), 1) if q else 0
        garbled_penalty = 0.0
        if nonascii_ratio > 0.05 and q_nonascii < 0.05:
            garbled_penalty = min(0.5, nonascii_ratio * 2)

        # ===== 8. Code-when-not-asked gate =====
        asks_code = any(kw in q_lower for kw in ['code','program','script','python','html','css','javascript','function','debug','c++','java'])
        code_indicators = len(re.findall(r'(?:#include|import\s+\w+|def\s+\w+\(|class\s+\w+|public\s+class|<\w+>\s*<\w+>)', resp))
        code_penalty = 0.0
        if not asks_code and code_indicators >= 2:
            code_penalty = min(0.4, code_indicators * 0.08)

        # ===== 9. First-sentence directness bonus =====
        directness_bonus = 0.0
        if sentences and q_content:
            fs_content = set(content(sentences[0]))
            if fs_content & q_content:
                directness_bonus = 0.05
            # Also bonus if first sentence is a confident declarative
            fs = sentences[0]
            if not fs.endswith('?') and len(fs.split()) >= 2:
                directness_bonus += 0.03

        # ===== 10. Alpha ratio sanity =====
        alpha = sum(1 for c in resp if c.isalpha())
        alpha_ratio = alpha / max(len(resp), 1)
        if alpha_ratio < 0.4 and not asks_code:
            garbled_penalty = max(garbled_penalty, 0.3)

        # ===== Combine =====
        base = 0.55 * relevance + 0.25 * length_score + 0.20 * (1.0 - min(rep_penalty, 1.0))
        base += directness_bonus

        total_penalty = template_penalty + echo_penalty + drift_penalty + garbled_penalty + code_penalty + rep_penalty * 0.4
        total_penalty = min(total_penalty, 0.85)

        score = base * (1.0 - total_penalty)
        final = score * 10.0

        # Floor for absolute minimum content
        if n_words <= 1 and not short_ok:
            final = min(final, 1.5)

        return round(max(0.0, min(10.0, final)), 2)

    except Exception:
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except Exception:
            return 3.0
