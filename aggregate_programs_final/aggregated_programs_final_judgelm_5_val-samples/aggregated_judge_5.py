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

        words = resp.split()
        n_words = len(words)
        if n_words == 0:
            return 0.5

        sentences = re.split(r'[.!?\n]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
        n_sents = max(len(sentences), 1)

        # ===== Query type classification =====
        # "Factoid" queries expect short specific answers
        is_factoid = bool(re.search(
            r'\b(name|identify|which|classify|biggest|largest|smallest|next number|'
            r'how many|how much|capital of|who is|who was|what is the|what was the|'
            r'when did|when was|where is|where was|tell me if|find the)\b',
            q_lower))
        # "Yes/no" queries
        is_yn = bool(re.search(
            r'^\s*(is |are |was |were |do |does |did |can |could |should |would |will |has |have )',
            q_lower)) and '?' in q
        # List queries
        is_list = bool(re.search(r'\b(list|enumerate|generate.*list|name (?:some|a few|several))\b', q_lower))
        # Rewrite/transform queries
        is_rewrite = bool(re.search(r'\b(rewrite|rephrase|edit|change|convert|translate|join|combine)\b', q_lower))
        # Explanation queries
        is_explain = bool(re.search(r'\b(why|explain|describe|how does|how do|how can|what are some|tips|advice)\b', q_lower))

        # ===== Quality signals that always matter =====

        # Template leakage check
        tmpl_n = len(re.findall(
            r'\b(?:Instruction|Input|Output|Question|Answer|Example)\s*:',
            resp, re.IGNORECASE))

        # Non-ASCII bleed
        non_ascii = sum(1 for c in resp if ord(c) > 127)
        nonascii_ratio = non_ascii / max(len(resp), 1)
        q_nonascii_ratio = sum(1 for c in q if ord(c) > 127) / max(len(q), 1) if q else 0
        garbled = nonascii_ratio > 0.1 and q_nonascii_ratio < 0.05

        # Question-echo
        q_in_resp_sents = sum(1 for s in sentences if '?' in s)
        question_echo = (n_sents >= 2 and q_in_resp_sents / n_sents > 0.4)

        # Repetition
        rep_high = False
        if n_words >= 6:
            lw = [w.lower() for w in words]
            tg = [' '.join(lw[i:i+3]) for i in range(len(lw)-2)]
            if tg:
                tc = Counter(tg)
                rep_t = sum(c-1 for c in tc.values() if c > 1)
                if rep_t / len(tg) > 0.15:
                    rep_high = True

        # ===== Relevance =====
        stops = {'the','a','an','is','are','was','were','be','to','of','in','for','on','with',
                 'at','by','from','as','it','its','this','that','they','them','and','or','but',
                 'not','no','so','if','what','which','who','how','when','where','why','can',
                 'could','would','should','may','might','will','i','you','your','we','our','also',
                 'about','any','some','do','does','did','have','has','had','make'}
        q_content = set()
        for w in re.findall(r'[a-zA-Z]+', q_lower):
            if w not in stops and len(w) > 2:
                q_content.add(w)

        r_words_lower = set(re.findall(r'[a-z]+', resp.lower()))
        if q_content:
            relevance = len(q_content & r_words_lower) / len(q_content)
        else:
            relevance = 0.5

        # ===== Base scoring with adaptive length expectations =====
        score = 5.0

        # Adapt length expectations to query type
        if is_factoid or is_yn:
            # Short answers OK and rewarded
            if 1 <= n_words <= 8:
                score += 2.0  # strong reward for concise
            elif n_words <= 25:
                score += 1.0
            elif n_words <= 100:
                score += 0.3
            elif n_words > 200:
                score -= 0.5  # likely verbose with extras
        elif is_explain:
            # Want more detail
            if n_words < 5:
                score -= 1.5
            elif n_words < 20:
                score -= 0.3
            elif 20 <= n_words <= 300:
                score += 1.0
            elif n_words > 400:
                score -= 0.3
        elif is_list:
            # Want multiple items
            if n_words < 3:
                score -= 1.0
            elif n_words >= 8 and n_words <= 300:
                score += 1.0
        elif is_rewrite:
            # Want a transformation - similar length to query is OK
            q_words_n = len(q.split())
            if 0.3 <= n_words / max(q_words_n, 1) <= 5:
                score += 1.0
            elif n_words > q_words_n * 8:
                score -= 0.5
        else:
            # Default
            if n_words < 3:
                score -= 1.0
            elif 5 <= n_words <= 250:
                score += 0.8
            elif n_words > 400:
                score -= 0.5

        # ===== Apply quality penalties =====
        if tmpl_n >= 3:
            score -= 2.5
        elif tmpl_n == 2:
            score -= 1.2
        elif tmpl_n == 1:
            score -= 0.4

        if question_echo:
            score -= 1.8

        if garbled:
            score -= min(nonascii_ratio * 6, 3.0)

        if rep_high:
            score -= 1.5

        # ===== Relevance reward / penalty =====
        if q_content:
            if relevance >= 0.5:
                score += 1.2
            elif relevance >= 0.3:
                score += 0.6
            elif relevance == 0:
                # If response has NO query content words, very bad
                # unless it's a list/example response
                if not (is_list and n_words >= 5):
                    score -= 1.8

        # ===== Specific direct-answer reward =====
        # For factoid/yn: clean short response with relevance gets boost
        alpha = sum(1 for c in resp if c.isalpha())
        ar = alpha / max(len(resp), 1)

        if (is_factoid or is_yn) and n_words <= 15 and tmpl_n == 0 and not question_echo and ar > 0.5:
            # This is a clean concise answer
            score += 1.5
            # Extra bonus if it contains query content
            if q_content and relevance > 0:
                score += 0.5

        # ===== "Yes/No + explanation" reward =====
        if is_yn and re.match(r'^\s*(yes|no)\b', resp.lower()) and 5 <= n_words <= 100:
            score += 0.8

        # ===== Code/HTML when not asked =====
        asks_code = any(kw in q_lower for kw in
            ['code','program','script','python','html','css','javascript','function','debug','c++','java'])
        if not asks_code:
            html_n = len(re.findall(r'<[a-zA-Z/][^>]*>', resp))
            code_n = len(re.findall(
                r'(?:#include|import\s+\w+|def\s+\w+\(|public\s+class|void\s+\w+\()',
                resp))
            if html_n >= 4 or code_n >= 3:
                score -= 2.0
            elif html_n >= 2 or code_n >= 1:
                score -= 0.6

        # ===== Off-topic drift =====
        if n_sents >= 4 and q_content:
            half = n_sents // 2
            back_text = ' '.join(sentences[half:]).lower()
            back_words = set(re.findall(r'[a-z]+', back_text))
            if len(back_text) > 80 and not (back_words & q_content):
                score -= 1.0

        # ===== Sanity caps =====
        if ar < 0.25 and not asks_code:
            score -= 1.5

        # Single word answer requires query to be short-answer style
        if n_words == 1 and not (is_factoid or is_yn or is_list):
            score = min(score, 3.5)

        # Empty/gibberish floor
        if n_words <= 2 and relevance == 0:
            score = min(score, 2.5)

        score = max(0.0, min(10.0, score))
        return round(score, 2)

    except Exception:
        return 4.0
