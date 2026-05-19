def judging_function(query, response):
    try:
        import re
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        q = query.strip() if isinstance(query, str) else ""
        resp = response.strip()
        if not resp:
            return 0.0

        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        sentences = [s.strip() for s in re.split(r'[.!?\n]+', resp) if len(s.strip()) > 3]
        n_sent = max(1, len(sentences))
        rl = resp.lower()
        ql = q.lower()

        # ===== Baseline score: substance =====
        base = 5.0
        if n_words < 3:
            base = 1.5
        elif n_words < 10:
            base = 3.0
        elif n_words < 25:
            base = 4.5
        elif n_words <= 250:
            base = 6.5
        else:
            base = 6.0

        # ===== Question-cascade detection =====
        q_marks = resp.count('?')
        sent_with_q = sum(1 for s in sentences if '?' in s[-3:] or s.endswith('?'))

        # Query is asking a question?
        query_is_question = '?' in q or bool(re.match(r'^\s*(what|who|where|when|why|how|which|is|are|do|does|can|will|tell|explain|describe|name|identify|how do|how can)\b', ql))

        # Number of question sentences relative to total
        q_ratio = sent_with_q / max(n_sent, 1)

        # === Specific failure pattern: response is mostly questions ===
        if query_is_question:
            if q_ratio >= 0.6 and q_marks >= 2:
                base -= 3.0  # strong penalty
            elif q_ratio >= 0.4 and q_marks >= 3:
                base -= 2.0
            elif q_marks >= 5:
                base -= 1.5
            elif q_marks >= 3 and n_sent <= 8:
                base -= 1.0

        # ===== First-person prompt echo =====
        # "I want to...", "I am...", "I have..." at the start of response
        first_sent = sentences[0].lower() if sentences else ""
        first_person_start = bool(re.match(
            r'^\s*(i\s+(?:am|want|have|need|would|think|believe|don\'?t|can\'?t|like|love|hate)|let me|tell me|what)\b',
            first_sent
        ))
        if first_person_start and query_is_question:
            base -= 2.0  # responding to a question with "I want..." is bad

        # ===== Pattern: "Also, what is..." or "Hi, can you..." (clarifying question instead of answer) =====
        clarifying_open = bool(re.match(
            r'^\s*(?:also|btw|hi|hello|hey|first|wait|sorry)[,.\s]',
            rl
        ))
        if clarifying_open and query_is_question and n_words < 80:
            base -= 1.5

        # ===== "Can you...?" "Could you...?" patterns =====
        polite_q = len(re.findall(r'\b(can you|could you|will you|would you|do you have)\b', rl))
        if polite_q >= 2 and query_is_question:
            base -= min(1.5, polite_q * 0.5)

        # ===== Reward direct answer indicators =====
        # Declarative opening with content
        if sentences:
            fs = sentences[0]
            declarative = (
                not fs.endswith('?') and
                fs[0].isupper() if fs else False and
                not first_person_start
            )
            if declarative and n_words >= 5:
                base += 0.7

            # Reward responses that begin with topic of query
            STOP = {'the','a','an','is','are','to','of','in','for','on','with','at','by','from'}
            q_topic = [w for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 3]
            fs_lower = fs.lower()
            if q_topic and any(t in fs_lower[:80] for t in q_topic[:5]):
                base += 0.5

        # ===== Detect templates =====
        tmpl = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer|Explanation)\s*:', resp))
        if tmpl >= 3:
            base -= min(3.0, tmpl * 0.6)
        elif tmpl == 2 and n_words < 100:
            base -= 0.7

        # ===== Repetition =====
        wl = [w.lower() for w in words]
        if n_words >= 6:
            tri = [tuple(wl[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            most = tc.most_common(1)[0][1] if tc else 1
            if most >= 5:
                base -= min(3.0, most * 0.4)

        # ===== Coverage of query keywords =====
        STOP2 = {'a','an','the','is','are','was','were','to','of','in','for','on','with','at',
                 'by','from','as','it','this','that','i','you','what','which','how','who','do','does','can','will'}
        q_content = set(w.lower() for w in re.findall(r"[a-z]+", ql) if w not in STOP2 and len(w) > 2)
        r_content = set(w.lower() for w in re.findall(r"[a-z]+", rl) if w not in STOP2 and len(w) > 2)
        if q_content:
            cov = len(q_content & r_content) / len(q_content)
            base += cov * 1.5

        # ===== Echo of query text =====
        if len(q) > 20:
            q_norm = re.sub(r'\s+', ' ', ql).strip()
            if q_norm[:50] in rl:
                # Response contains the query verbatim - might just be echoing
                rest = rl.replace(q_norm[:50], '', 1).strip()
                if len(rest.split()) < n_words * 0.4:
                    base -= 1.5

        # ===== Code/HTML when not asked =====
        asks_code = any(w in ql for w in ['code','python','debug','program','function','script','html','tag'])
        if not asks_code:
            code_hits = len(re.findall(r'(?:import |def |class |#include|public class|<[a-z]+>)', resp))
            if code_hits > 2:
                base -= min(2.5, code_hits * 0.4)

        # ===== Garbled foreign text =====
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            base -= min(3.0, nonlatin / 15.0)

        return round(max(0.0, min(10.0, base)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 2.0
        except:
            return 3.0
