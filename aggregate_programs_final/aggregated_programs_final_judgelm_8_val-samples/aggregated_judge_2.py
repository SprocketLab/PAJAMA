def judging_function(query, response):
    try:
        import re
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""

        resp = response.strip()
        q = query.strip()
        if not resp:
            return 0.0

        q_lower = q.lower()
        r_lower = resp.lower()
        n_chars = len(resp)
        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        # === Detect "brief answer" intent in query ===
        brief_signals = [
            'output the', 'output directly', 'identify the', 'identify a',
            'name a', 'name the', 'which is', 'which of', 'classify',
            'compute', 'calculate', 'find the', 'list the', 'tell me which',
            'biggest', 'smallest', 'largest', 'shortest', 'longest',
            'what is the', 'who is the', 'when did', 'what year'
        ]
        is_brief_intent = any(s in q_lower for s in brief_signals)
        # Short queries that look like simple questions
        if not is_brief_intent and len(q.split()) < 15 and re.search(r'^\s*(who|what|which|when|where|name)\b', q_lower):
            is_brief_intent = True

        # === Detect off-topic continuation (the main bug) ===
        # Template repetition is a strong negative signal
        template_hits = len(re.findall(r'\b(?:Instruction|Input|Output|Question|Answer|Comment|Explanation)\s*:', resp))
        template_penalty = 0.0
        if template_hits >= 3:
            template_penalty = min(4.0, (template_hits - 1) * 0.7)
        elif template_hits == 2 and n_words < 80:
            template_penalty = 0.8

        # "Multiple choice option" listing (like A) ... B) ... C) ...)
        mc_options = len(re.findall(r'\b[A-D]\)\s+', resp))
        if mc_options >= 3 and not re.search(r'\b(option|choice|multiple)\b', q_lower):
            template_penalty += min(2.0, mc_options * 0.4)

        # === Detect random off-topic ramble in second half ===
        first_half = resp[:len(resp)//2].lower()
        second_half = resp[len(resp)//2:].lower()
        # Query content words
        STOP = {'a','an','the','is','are','was','were','be','to','of','in','for','on','with',
                'at','by','from','as','and','or','but','not','it','this','that','what','which',
                'how','when','where','why','who','do','does','have','has','i','you'}
        q_content = set(w.strip(".,!?;:()") for w in q_lower.split() 
                       if w.strip(".,!?;:()") not in STOP and len(w.strip(".,!?;:()")) > 2)

        drift_penalty = 0.0
        if q_content and n_words > 80:
            fh_words = set(re.findall(r'[a-z]+', first_half)) & q_content
            sh_words = set(re.findall(r'[a-z]+', second_half)) & q_content
            # Second half is unrelated to query
            if len(fh_words) >= 2 and len(sh_words) == 0:
                drift_penalty = 2.0
            elif len(fh_words) > len(sh_words) + 2:
                drift_penalty = 1.0

        # === Direct answer detection ===
        # Strong signal: short response that mentions a query content word
        is_direct_short = False
        if is_brief_intent and n_words <= 15:
            words_lower = set(re.findall(r"[a-z']+", r_lower))
            if q_content and (words_lower & q_content) or re.search(r'\b\d+', resp):
                is_direct_short = True
            # Or just looks like a clean name/answer
            if re.match(r'^[A-Za-z][\w\s\-.,]{0,80}\.?$', resp.strip()) and n_words >= 1:
                is_direct_short = True

        # === Base scoring ===
        score = 5.0  # neutral base

        # Length appropriateness
        if is_brief_intent:
            if n_words <= 8:
                score += 2.0  # direct, brief answer
            elif n_words <= 25:
                score += 1.0
            elif n_words <= 60:
                score += 0.0
            elif n_words <= 150:
                score -= 0.5
            else:
                score -= 1.5  # likely contains off-topic continuation
        else:
            # For open-ended queries, prefer moderate-to-long
            if n_words < 3:
                score -= 2.5
            elif n_words < 10:
                score -= 1.0
            elif 10 <= n_words <= 250:
                score += 1.5
            else:
                score += 0.5

        # Apply major penalties
        score -= template_penalty
        score -= drift_penalty

        # Bonus for clean direct answers
        if is_direct_short and template_hits == 0:
            score += 1.5

        # Garbled text/foreign characters
        nonlatin = sum(1 for c in resp if c.isalpha() and ord(c) >= 128)
        if nonlatin > 5 and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff\u0590-\u05ff\u0600-\u06ff]', q):
            score -= min(3.0, nonlatin / 20.0)

        # Echo/question response (FAILURE FIX)
        q_in_r = resp.count('?')
        period_r = resp.count('.')
        if q_in_r >= 3 and period_r <= q_in_r:
            score -= 1.5

        # Response starts asking instead of answering
        if re.match(r'^\s*(also,?|what|how|can|do|i\s+(am|want|have|need))', r_lower) and '?' in q:
            score -= 1.0

        # Repetition penalty
        words_lower_list = [w.lower() for w in words]
        if n_words >= 6:
            tri = [tuple(words_lower_list[i:i+3]) for i in range(n_words-2)]
            tc = Counter(tri)
            most = tc.most_common(1)[0][1] if tc else 1
            if most >= 4:
                score -= min(2.5, most * 0.3)

        # Empty/non-answer
        if n_words <= 1 or r_lower.strip('. ') in ('no','n/a','na','none','no answer'):
            if not is_brief_intent or q_content:
                score -= 2.0

        # Contains exact query echo (just repeats the question)
        if len(q) > 20 and q_lower[:40] in r_lower and n_words < len(q.split()) * 1.5:
            score -= 1.5

        # Excessive code when not asked
        asks_code = any(w in q_lower for w in ['code','python','program','function','script','html','debug'])
        code_lines = len(re.findall(r'(?:import |def |class |#include|<[a-z]+>|public class)', resp))
        if not asks_code and code_lines > 2:
            score -= min(2.5, code_lines * 0.3)

        # Truncation
        if resp.rstrip()[-1] not in '.!?")\']}:' and n_words > 30:
            score -= 0.3

        return round(max(0.0, min(10.0, score)), 2)
    except Exception:
        try:
            return 4.5 if response and len(response.strip()) > 15 else 2.0
        except:
            return 3.0
