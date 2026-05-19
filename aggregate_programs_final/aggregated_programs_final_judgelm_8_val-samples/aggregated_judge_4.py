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
        if len(resp) < 3:
            return 0.8

        text = resp
        words = re.findall(r"[a-zA-Z']+", text)
        if not words:
            return 0.5
        n_words = len(words)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 2]
        n_sent = max(1, len(sentences))

        # ===== Readability =====
        def syl(w):
            w = w.lower()
            if len(w) <= 2: return 1
            v = 'aeiouy'; c = 0; pv = False
            for ch in w:
                iv = ch in v
                if iv and not pv: c += 1
                pv = iv
            if w.endswith('e') and c > 1: c -= 1
            return max(c, 1)
        complex_w = sum(1 for w in words if syl(w) >= 3)
        asl = n_words / n_sent
        fog = 0.4 * (asl + 100 * complex_w / n_words)
        if 6 <= fog <= 14:
            fog_score = 1.0
        elif fog < 6:
            fog_score = max(0.3, fog / 6)
        else:
            fog_score = max(0.3, 1 - (fog - 14) * 0.05)

        # ===== Vocabulary richness =====
        words_lower = [w.lower() for w in words]
        wf = Counter(words_lower)
        hapax = sum(1 for c in wf.values() if c == 1)
        ttr = len(wf) / n_words
        vocab_score = 0.5 * min(ttr / 0.6, 1.0) + 0.5 * min(hapax / max(len(wf), 1) / 0.6, 1.0)

        # ===== Sentence variety =====
        if n_sent >= 2:
            sl = [len(re.findall(r"[a-z']+", s.lower())) for s in sentences]
            avg = sum(sl) / len(sl)
            if avg > 0:
                var = sum((x-avg)**2 for x in sl) / len(sl)
                cv = math.sqrt(var) / avg
                if 0.15 <= cv <= 0.8:
                    var_score = 1.0
                elif cv < 0.15:
                    var_score = 0.6
                else:
                    var_score = max(0.4, 1 - (cv - 0.8) * 0.4)
            else:
                var_score = 0.5
        else:
            var_score = 0.5

        # ===== Factual indicators =====
        # Specific numbers/dates
        years = len(re.findall(r'\b(?:1[789]\d{2}|20[0-2]\d)\b', text))
        dates = len(re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d', text))
        numbers = len(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text))
        # Capitalized multi-word entities (mid-sentence)
        entities = len(re.findall(r'(?<=\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', text))
        common_starts = {'The','This','That','However','Also','But','And','When','Where','What','How','Why','Who','First','Second','Third','Finally','Yes','No','I'}
        # Remove false positives
        capwords = re.findall(r'(?<=\s)([A-Z][a-z]+)', text)
        real_caps = [w for w in capwords if w not in common_starts]
        fact_score = min(1.0, (years*0.2 + dates*0.3 + min(numbers, 5)*0.1 + min(len(real_caps), 8)*0.08))

        # ===== Hedging (appropriate uncertainty) =====
        hedges = ['typically','generally','usually','often','can','may','might','approximately',
                  'about','estimated','tends to','depending on','in some cases','suggests',
                  'indicates','widely considered','known as']
        rl = text.lower()
        hedge_count = sum(rl.count(h) for h in hedges)
        hedge_score = min(1.0, hedge_count * 0.15)

        # ===== Red flags =====
        sensational = ['shocking','unbelievable','conspiracy','cover-up','they don\'t want you','wake up','sheeple','hoax']
        sens_hits = sum(1 for s in sensational if s in rl)
        absolute = ['always','never','everyone knows','undeniable','100%','guaranteed','no doubt']
        abs_hits = sum(1 for a in absolute if a in rl)

        # ===== Length =====
        if n_words < 3:
            length_score = 0.1
        elif n_words < 10:
            length_score = 0.4
        elif n_words <= 250:
            length_score = 1.0
        elif n_words <= 500:
            length_score = 0.85
        else:
            length_score = 0.7

        # ===== Repetition =====
        if n_words >= 6:
            tri = [tuple(words_lower[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            rep_tri = sum(c - 1 for c in tc.values() if c > 1)
            rep_score = max(0, 1 - rep_tri / max(len(tri), 1) * 8)
        else:
            rep_score = 1.0

        # ===== Template/echo penalty (FAILURE FIX) =====
        template_hits = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer)\s*:', text))
        question_count = text.count('?')
        period_count = text.count('.')

        # ===== Relevance =====
        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on','with','at','by','from','as','and','or','but','not','it','this','that','i','you','he','she','they','we'}
        q_content = set(w.lower() for w in re.findall(r"[a-zA-Z']+", q) if w.lower() not in STOP and len(w) > 2)
        r_content = set(w.lower() for w in words if w.lower() not in STOP and len(w) > 2)
        if q_content:
            rel = len(q_content & r_content) / len(q_content)
        else:
            rel = 0.5

        # ===== Combine =====
        readability = (fog_score + vocab_score + var_score) / 3.0
        base = (
            0.20 * readability +
            0.20 * length_score +
            0.15 * fact_score +
            0.10 * hedge_score +
            0.20 * rel +
            0.15 * rep_score
        )

        score = base * 10.0
        score += min(1.5, fact_score * 1.0)

        # Penalties
        score -= sens_hits * 1.0
        score -= abs_hits * 0.3

        # Template repeat
        if template_hits >= 3:
            score -= min(4.0, template_hits * 0.8)
        elif template_hits == 2:
            score -= 0.7

        # Mostly questions (no real factual content)
        if n_sent >= 3 and question_count >= n_sent * 0.5:
            score -= 1.5

        # Code/HTML when not asked
        asks_code = any(w in q.lower() for w in ['code','python','program','function','script','html','debug'])
        if not asks_code:
            code_hits = len(re.findall(r'(?:import |def |class |#include|<[a-z]+>|public class)', text))
            score -= min(2.0, code_hits * 0.4)

        # Garbled foreign chars
        nonlatin = sum(1 for c in text if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            score -= min(2.5, nonlatin / 18.0)

        return round(max(0.0, min(10.0, score)), 2)
    except Exception:
        try:
            return 4.0 if response and len(response.strip()) > 20 else 1.5
        except:
            return 3.0
