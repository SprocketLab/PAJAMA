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

        rl = resp.lower()
        ql = q.lower()

        base = 5.5

        # ===== Length adequacy =====
        if n_words < 2: base = 1.0
        elif n_words < 6: base = 4.0
        elif n_words < 15: base = 5.5
        elif n_words <= 250: base = 6.5
        else: base = 5.8

        # ===== Detect task type =====
        task_rewrite = bool(re.search(r'\b(rewrite|rephrase|reword|convert|translate|edit)\b', ql))
        task_compute = bool(re.search(r'\b(compute|calculate|find|determine the|average of|sum of|product of)\b', ql))
        task_identify = bool(re.search(r'\b(identify|which|name|find|list|classify|categorize|tell me)\b', ql))
        task_explain = bool(re.search(r'\b(explain|describe|what is|what are|how does|why|how do)\b', ql))
        task_generate = bool(re.search(r'\b(generate|create|write a|make a|compose|design)\b', ql))
        task_debug = 'debug' in ql or ('code' in ql and 'fix' in ql)

        # ===== Compliance checks =====

        # Rewrite tasks: response should be a transformed version, not just original
        if task_rewrite:
            # Third person rewrite specifically
            if 'third person' in ql:
                has_third_person = bool(re.search(r'\b(he|she|it|they|him|her|his|hers|them|their)\b', rl))
                has_second_person = bool(re.search(r'\b(you|your|yours)\b', rl[:200]))
                if has_third_person and not has_second_person:
                    base += 1.5
                elif has_second_person and not has_third_person:
                    base -= 1.5
            if 'first person' in ql:
                has_first = bool(re.search(r'\b(i|me|my|mine|we|us|our)\b', rl))
                if has_first:
                    base += 1.0
            if 'vegetarian' in ql or 'vegan' in ql:
                meat_words = ['beef','chicken','pork','bacon','ham','sausage','fish','salmon','tuna','lamb']
                veg_words = ['tofu','seitan','tempeh','vegetable','veggie','plant','mushroom','bean','lentil']
                meat_hits = sum(1 for m in meat_words if m in rl)
                veg_hits = sum(1 for v in veg_words if v in rl)
                if veg_hits > meat_hits:
                    base += 1.5
                elif meat_hits > veg_hits:
                    base -= 1.5

        # Compute tasks: should contain a number
        if task_compute:
            has_number = bool(re.search(r'\d+(?:\.\d+)?', resp))
            if has_number:
                base += 1.0
                # Check if it's near the start (direct answer)
                first_50 = resp[:80]
                if re.search(r'\d', first_50):
                    base += 0.8
            else:
                base -= 2.0
            # If asked to output directly, prefer brief
            if 'directly' in ql or 'output the result' in ql:
                if n_words <= 15:
                    base += 1.2

        # Identify/classify: response should make a definite assignment
        if task_identify:
            if n_words <= 5:
                # Check it contains content
                if any(w.lower() not in {'i','don\'t','know','sorry','none','n/a'} for w in words):
                    base += 0.5
            elif n_words <= 30:
                base += 0.5

        # Debug task: should show code or analysis, not meta-questions
        if task_debug:
            has_code = bool(re.search(r'(?:def |class |return |import |public |private |function|var |let |const |#include|\{|\})', resp))
            question_count = resp.count('?')
            if has_code and question_count < 3:
                base += 1.5
            elif question_count > 5:
                base -= 2.0  # asking lots of meta-questions instead of debugging

        # Generate creative content
        if task_generate:
            # Should NOT be a list of questions
            q_count = resp.count('?')
            if q_count >= 4 and n_words < 150:
                base -= 1.5
            # Should have substantial prose
            if n_words >= 30:
                base += 0.5

        # ===== Coverage of query keywords =====
        STOP = {'a','an','the','is','are','was','were','to','of','in','for','on','with','at',
                'by','from','as','and','or','but','not','it','this','that','i','you','he','she','they','we','what','how','can','do','does','will','my','me','your'}
        q_content = set(w.lower() for w in re.findall(r"[a-z]+", ql) if w not in STOP and len(w) > 2)
        r_content = set(w.lower() for w in re.findall(r"[a-z]+", rl) if w not in STOP and len(w) > 2)
        if q_content:
            cov = len(q_content & r_content) / len(q_content)
            base += cov * 1.5

        # ===== Direct factual question (no task verb) =====
        if not (task_rewrite or task_compute or task_identify or task_explain or task_generate or task_debug):
            if '?' in q and n_words >= 5:
                # Should answer something
                # Penalize if response is mostly questions
                r_q = resp.count('?')
                if r_q >= 3:
                    base -= 1.5

        # ===== Detect non-answers =====
        non_answer_patterns = [
            r"^\s*i don'?t know",
            r"^\s*i'?m sorry",
            r"^\s*you mean",
            r"^\s*also,\s+what",
            r"^\s*what (?:do|does|is) (?:you|the)",
            r"^\s*let me know"
        ]
        if any(re.search(p, rl) for p in non_answer_patterns):
            if n_words < 50:
                base -= 1.0

        # ===== Template artifacts =====
        tmpl = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer)\s*:', resp))
        if tmpl >= 3:
            base -= min(3.5, tmpl * 0.7)
        elif tmpl == 2:
            base -= 0.6

        # ===== Repetition =====
        wl = [w.lower() for w in words]
        if n_words >= 6:
            tri = [tuple(wl[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            if rep_tri > 5:
                base -= min(2.5, rep_tri * 0.15)

        # ===== Garbled foreign text =====
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            base -= min(3.0, nonlatin / 15.0)

        # ===== Code in non-code query =====
        if not (task_debug or 'code' in ql or 'program' in ql or 'function' in ql or 'html' in ql):
            code_hits = len(re.findall(r'(?:import |def |class |#include|public class|<[a-z]+>)', resp))
            if code_hits > 2:
                base -= min(2.5, code_hits * 0.4)

        # ===== Unique vocabulary ratio =====
        if n_words > 20:
            uniq = len(set(wl)) / n_words
            if uniq < 0.3:
                base -= 1.8

        return round(max(0.0, min(10.0, base)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 2.0
        except:
            return 3.0
