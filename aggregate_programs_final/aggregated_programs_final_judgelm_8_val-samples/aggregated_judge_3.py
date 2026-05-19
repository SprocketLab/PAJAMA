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
        if n_words <= 1:
            return 0.5

        # ===== Genuine structural indicators =====
        # Bullets, real numbered lists (not just "1. " followed by garbage)
        bullets = len(re.findall(r'(?:^|\n)\s*[-*•]\s+\S', resp))
        numbered = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s+[A-Z]', resp))
        kv = len(re.findall(r'(?:^|\n)\s*[A-Z][^:\n]{1,40}:\s+\S', resp))
        paragraphs = len([p for p in resp.split('\n\n') if len(p.strip()) > 20])

        good_structure = bullets + numbered + kv

        # ===== Bad pseudo-structure (the main failure pattern) =====
        # Template repetition
        template_repeat = len(re.findall(r'(?:Instruction|Input|Output|Question|Answer)\s*:', resp))

        # Bare question lists (like Case 2, 18, 29)
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', resp) if len(s.strip()) > 5]
        n_sentences = len(sentences)
        question_sentences = sum(1 for s in sentences if s.endswith('?') or '?' in s[-3:])
        question_ratio = question_sentences / max(n_sentences, 1)

        # Repeated section headers (Case 2: "I.", "II.", "III." with bare questions)
        roman_headers = len(re.findall(r'(?:^|\n)\s*[IVX]+\.\s+[A-Z]', resp))

        # ===== Punctuation and capitalization quality =====
        punct_density = sum(1 for c in resp if c in '.,;:!?') / max(n_words, 1)
        if 0.04 <= punct_density <= 0.3:
            punct_score = 1.0
        elif punct_density < 0.04:
            punct_score = max(0.3, punct_density / 0.04)
        else:
            punct_score = max(0.4, 1 - (punct_density - 0.3))

        # First-letter capitalization
        cap_starts = sum(1 for s in sentences[:10] if s and s[0].isupper())
        cap_score = cap_starts / max(min(len(sentences), 10), 1)

        # ===== Repetition =====
        words_lower = [w.lower() for w in words]
        unique_ratio = len(set(words_lower)) / max(n_words, 1)

        rep_penalty = 0.0
        if n_words >= 6:
            tri = [tuple(words_lower[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            rep_penalty = min(3.0, rep_tri / max(len(tri), 1) * 12)

        # Sentence-level duplicates
        sent_norm = [re.sub(r'\s+', ' ', s.lower()) for s in sentences if len(s) > 10]
        if sent_norm:
            sc = Counter(sent_norm)
            dup_sent = sum(c - 1 for c in sc.values() if c > 1)
            rep_penalty += min(2.0, dup_sent * 0.8)

        # ===== Coherence: sentence-to-sentence overlap =====
        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on','with',
                'at','by','from','as','and','or','but','not','it','this','that','these','those',
                'i','you','he','she','they','we','have','has','had','do','does','can','will'}

        coherence = 0.5
        if n_sentences >= 2:
            sent_words_sets = []
            for s in sentences:
                sw = set(re.findall(r"[a-z]+", s.lower())) - STOP
                sent_words_sets.append(sw)
            overlaps = []
            for i in range(len(sent_words_sets)-1):
                a, b = sent_words_sets[i], sent_words_sets[i+1]
                if a and b:
                    union = len(a | b)
                    if union > 0:
                        overlaps.append(len(a & b) / union)
            if overlaps:
                avg = sum(overlaps)/len(overlaps)
                # Sweet spot: 0.05 - 0.3
                if 0.05 <= avg <= 0.4:
                    coherence = 1.0
                elif avg < 0.05:
                    coherence = 0.5
                else:
                    coherence = max(0.4, 1.0 - (avg-0.4))

        # ===== Length =====
        if n_words < 3:
            length_score = 0.15
        elif n_words < 8:
            length_score = 0.4
        elif n_words < 20:
            length_score = 0.7
        elif n_words <= 250:
            length_score = 1.0
        elif n_words <= 500:
            length_score = 0.85
        else:
            length_score = 0.7

        # ===== Combine =====
        base = (
            0.20 * length_score +
            0.15 * punct_score +
            0.15 * cap_score +
            0.20 * coherence +
            0.15 * unique_ratio +
            0.15 * min(1.0, good_structure / 3.0)
        )

        score = base * 10.0

        # ===== Major penalties for pseudo-structure =====
        # Template repetition (Cases 6, 26, 28, 30, 31)
        if template_repeat >= 3:
            score -= min(4.5, template_repeat * 0.9)
        elif template_repeat == 2:
            score -= 1.0

        # Bare question lists (Cases 2, 18, 29)
        if question_ratio > 0.5 and n_sentences >= 3 and not q.strip().endswith('?') == False:
            # Many questions might be appropriate if query asks for questions
            if not any(k in q.lower() for k in ['question','ask','list of questions','generate questions']):
                score -= min(3.0, question_ratio * 3.5)

        # Roman headers with questions (outline-only, no actual content)
        if roman_headers >= 2 and question_ratio > 0.4:
            score -= 2.0

        # Repetition penalty
        score -= rep_penalty

        # Multiple choice options when query doesn't ask for them
        mc = len(re.findall(r'^\s*[A-D]\)\s+\S', resp, re.MULTILINE))
        if mc >= 3 and 'choice' not in q.lower() and 'option' not in q.lower():
            score -= min(2.5, mc * 0.5)

        # Garbled foreign text
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            score -= min(3.0, nonlatin / 15.0)

        # Numeric noise (Case 6 like "0 0 -1 1 0..." patterns)
        num_runs = re.findall(r'(?:\d+\s+){5,}', resp)
        if num_runs and not any(k in q.lower() for k in ['number','sequence','count','math','sum','add','multiply']):
            score -= 1.5

        # Bonus for clean structured response with content
        if good_structure >= 2 and template_repeat <= 1 and question_ratio < 0.3 and rep_penalty < 0.5:
            score += 0.8

        return round(max(0.0, min(10.0, score)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 1.5
        except:
            return 3.0
