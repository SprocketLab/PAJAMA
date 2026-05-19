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
        if len(resp) < 3:
            return 1.0

        STOP = {'a','an','the','is','are','was','were','be','been','being','have','has','had',
                'do','does','did','will','would','could','should','may','might','can','shall',
                'to','of','in','for','on','with','at','by','from','as','into','through','about',
                'and','or','but','not','no','if','then','than','so','too','very','just','also',
                'this','that','these','those','it','its','i','me','my','we','our','you','your',
                'he','him','his','she','her','they','them','their','what','which','who','whom',
                'how','when','where','why','out','up','down','over','under','again','here',
                'there','any','some','all','each','more','most','other','such','only','own','same'}

        def tok(s):
            return re.findall(r"[a-zA-Z']+", s.lower())

        q_tokens = tok(q)
        r_tokens = tok(resp)
        if not r_tokens:
            return 0.5

        q_content = set(w for w in q_tokens if w not in STOP and len(w) > 2)
        r_content = [w for w in r_tokens if w not in STOP and len(w) > 2]
        r_content_set = set(r_content)

        # === Signal 1: keyword coverage ===
        if q_content:
            coverage = len(q_content & r_content_set) / len(q_content)
        else:
            coverage = 0.5

        # === Signal 2: bigram alignment ===
        def bigrams(toks):
            return set((toks[i], toks[i+1]) for i in range(len(toks)-1))
        q_bg = bigrams(q_tokens)
        r_bg = bigrams(r_tokens)
        bigram_score = len(q_bg & r_bg) / max(len(q_bg), 1) if q_bg else 0.3

        # === Signal 3: length adequacy ===
        n_words = len(r_tokens)
        q_words = len(q_tokens)
        if n_words < 2:
            length_score = 0.1
        elif n_words < 6:
            length_score = 0.4
        elif n_words < 12:
            length_score = 0.65
        elif n_words <= 250:
            length_score = 1.0
        else:
            length_score = 0.8

        # === Signal 4: first-sentence relevance (FAILURE FIX) ===
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', resp) if len(s.strip()) > 3]
        first_sent_score = 0.5
        if sentences and q_content:
            first_toks = set(tok(sentences[0]))
            first_content = first_toks - STOP
            if first_content:
                first_sent_score = min(1.0, 0.3 + 0.9 * len(first_content & q_content) / max(len(q_content), 1))

        # === Signal 5: echo/question detector (CRITICAL FIX) ===
        # Responses that mostly ask questions instead of answering are bad
        q_marks = resp.count('?')
        period_marks = resp.count('.')
        q_in_resp = len(re.findall(r'\?', resp))
        statement_count = max(1, period_marks)
        question_ratio = q_in_resp / max(q_in_resp + statement_count, 1)

        echo_penalty = 0.0
        query_is_question = '?' in q or any(w in q.lower().split()[:4] for w in ['what','who','where','when','why','how','can','is','are','do','does','will'])
        if query_is_question and question_ratio > 0.5 and q_in_resp >= 2:
            echo_penalty = 0.4
        elif query_is_question and q_in_resp >= 4 and n_words < 100:
            echo_penalty = 0.3

        # Detect "I want / I am / I have" first-person prompt echo
        first_person_echo = bool(re.match(r'^\s*(i\s+(am|want|have|need|would|think|don\'?t|can\'?t))', resp.lower()))
        if first_person_echo and query_is_question:
            echo_penalty = max(echo_penalty, 0.35)

        # === Signal 6: template artifact penalty ===
        template_patterns = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer)\s*:', resp))
        template_penalty = 0.0
        q_lower = q.lower()
        looks_like_qa_task = any(k in q_lower for k in ['classify','identify','translate','convert'])
        if template_patterns > 2:
            template_penalty = min(0.5, (template_patterns - 2) * 0.12)
        elif template_patterns >= 2 and not looks_like_qa_task:
            template_penalty = 0.15

        # === Signal 7: repetition ===
        unique_ratio = len(set(r_tokens)) / max(len(r_tokens), 1)
        rep_penalty = 0.0
        if unique_ratio < 0.3:
            rep_penalty = 0.45
        elif unique_ratio < 0.5:
            rep_penalty = 0.2

        if len(r_tokens) >= 6:
            trigrams = [tuple(r_tokens[i:i+3]) for i in range(len(r_tokens)-2)]
            tc = Counter(trigrams)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            if rep_tri > 4:
                rep_penalty += min(0.25, rep_tri * 0.03)

        # === Signal 8: garbled text detection (FAILURE FIX) ===
        alpha = sum(1 for c in resp if c.isalpha() and ord(c) < 128)
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        total_alpha = alpha + nonlatin
        nonlatin_ratio = nonlatin / max(total_alpha, 1)
        garbled_penalty = 0.0
        # Heavy non-Latin when query is English: bad
        if nonlatin_ratio > 0.15 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            garbled_penalty = min(0.4, nonlatin_ratio * 0.8)

        # Detect repeated parenthetical garble like "骨髓内 (内在骨髓内)"
        weird_parens = len(re.findall(r'\([^)]{1,20}\)\s*\([^)]{1,20}\)', resp))
        if weird_parens >= 2:
            garbled_penalty += 0.2

        # === Signal 9: code/HTML when not asked ===
        asks_code = any(w in q_lower for w in ['code','python','program','function','script','html','tag','debug'])
        noise_penalty = 0.0
        if not asks_code:
            code_hits = len(re.findall(r'(?:import |def |class |#include|public class|<[a-z]+>|return\s+\w+;)', resp))
            if code_hits > 2:
                noise_penalty = min(0.35, code_hits * 0.06)

        # === Combine ===
        relevance = 0.30*coverage + 0.15*bigram_score + 0.30*first_sent_score + 0.25*min(1.0, len(r_content_set)/15.0)
        base = 0.55 * relevance + 0.30 * length_score + 0.15 * min(1.0, len(sentences)/3.0)

        total_penalty = echo_penalty + template_penalty + rep_penalty + garbled_penalty + noise_penalty
        total_penalty = min(total_penalty, 0.75)

        score = base * (1.0 - total_penalty)

        # Floor for very short responses
        if n_words <= 2 and q_words > 4:
            score = min(score, 0.25)

        final = score * 10.0
        # Bonus for clearly substantive on-topic responses
        if coverage > 0.5 and n_words > 30 and total_penalty < 0.15:
            final = min(10.0, final + 0.5)

        return round(max(0.0, min(10.0, final)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 2.0
        except:
            return 3.0
