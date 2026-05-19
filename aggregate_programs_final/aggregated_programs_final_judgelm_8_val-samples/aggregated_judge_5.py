def judging_function(query, response):
    try:
        import re
        import math
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
        if n_words == 1:
            return 1.2

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', resp) if len(s.strip()) > 2]
        n_sent = max(1, len(sentences))

        STOP = {'a','an','the','is','are','was','were','be','been','have','has','had','do','does',
                'did','will','would','could','should','can','may','might','to','of','in','for','on',
                'with','at','by','from','as','into','through','and','or','but','not','no','it','its',
                'this','that','these','those','i','me','my','we','you','your','he','him','his','she',
                'her','they','them','their','what','which','who','how','when','where','why'}

        def content(text):
            return set(w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOP and len(w) > 2)

        q_content = content(q)
        r_content = content(resp)

        # ===== Sentence-to-sentence coherence chain =====
        coh_scores = []
        for i in range(1, n_sent):
            a, b = content(sentences[i-1]), content(sentences[i])
            if a and b:
                u = len(a | b)
                if u > 0:
                    coh_scores.append(len(a & b) / u)
        if coh_scores:
            avg_coh = sum(coh_scores) / len(coh_scores)
            if 0.05 <= avg_coh <= 0.4:
                coh_chain = 1.0
            elif avg_coh < 0.05:
                coh_chain = 0.45
            else:
                coh_chain = max(0.3, 1 - (avg_coh - 0.4) * 1.5)
        else:
            coh_chain = 0.5

        # ===== Information progression (new content per sentence) =====
        seen = set()
        new_ratios = []
        for s in sentences:
            sw = content(s)
            if sw:
                new = sw - seen
                new_ratios.append(len(new) / len(sw))
                seen.update(sw)
        info_prog = sum(new_ratios) / max(len(new_ratios), 1) if new_ratios else 0.3

        # ===== Logical connectors =====
        rl = resp.lower()
        connectors = ['because','therefore','thus','hence','consequently','as a result','since',
                      'however','moreover','furthermore','additionally','also','although','while',
                      'for example','for instance','such as','specifically','in particular',
                      'first','second','third','finally','in addition','on the other hand']
        conn_count = sum(rl.count(c) for c in connectors)
        conn_score = min(1.0, conn_count / max(n_sent, 1) * 0.6)

        # ===== Query alignment =====
        if q_content and r_content:
            align = len(q_content & r_content) / len(q_content)
        else:
            align = 0.4

        # ===== Substance =====
        if n_words < 3: subst = 0.1
        elif n_words < 10: subst = 0.4
        elif n_words < 25: subst = 0.65
        elif n_words < 60: subst = 0.85
        elif n_words <= 250: subst = 1.0
        else: subst = 0.85

        # ===== Question-cascade detector (CRITICAL FIX) =====
        q_marks = resp.count('?')
        sent_with_q = sum(1 for s in sentences if '?' in s)
        question_dominance = sent_with_q / max(n_sent, 1)

        # ===== Repetition =====
        if n_words >= 6:
            wl = [w.lower() for w in words]
            tri = [tuple(wl[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            rep_pen = min(0.6, rep_tri / max(len(tri), 1) * 4)
        else:
            rep_pen = 0.0

        # ===== Template artifact =====
        tmpl = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer)\s*:', resp))
        tmpl_pen = 0.0
        if tmpl >= 3: tmpl_pen = 0.4 + min(0.3, (tmpl-3) * 0.08)
        elif tmpl == 2: tmpl_pen = 0.15

        # ===== Combine =====
        score = (
            1.5 * subst +
            1.5 * align +
            1.3 * coh_chain +
            1.0 * info_prog +
            0.8 * conn_score
        )
        max_raw = 1.5 + 1.5 + 1.3 + 1.0 + 0.8
        normalized = score / max_raw

        # Apply penalties
        penalty = rep_pen + tmpl_pen
        # Question cascade is bad when query expects a fact answer
        query_expects_answer = '?' in q or re.search(r'\b(what|who|where|when|why|how|which|is|are|do|does|can|will|tell me|explain|describe|name|identify)\b', q.lower())
        if query_expects_answer and question_dominance > 0.4 and q_marks >= 2:
            penalty += min(0.4, question_dominance * 0.5)

        normalized = max(0, normalized - penalty)
        score = normalized * 10.0

        # First-sentence relevance bonus
        if sentences and q_content:
            first_content = content(sentences[0])
            if first_content and len(first_content & q_content) >= min(2, len(q_content)):
                score += 0.7

        # Garbled foreign chars
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            score -= min(2.5, nonlatin / 18.0)

        # Off-topic code dump
        asks_code = any(w in q.lower() for w in ['code','python','debug','program','function','script','html'])
        if not asks_code:
            code_hits = len(re.findall(r'(?:import |def |class |#include|public class|<[a-z]+>)', resp))
            if code_hits > 2:
                score -= min(2.0, code_hits * 0.3)

        return round(max(0.0, min(10.0, score)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 1.5
        except:
            return 3.0
