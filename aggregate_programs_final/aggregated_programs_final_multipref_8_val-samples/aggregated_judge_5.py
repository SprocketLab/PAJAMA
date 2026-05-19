def judging_function(query, response):
    """
    Completeness scorer using query decomposition. Checks coverage of query aspects,
    multi-perspective handling, and concrete completeness signals. Heavily gates
    truncation since incomplete answers fail completeness regardless of length.
    """
    try:
        import re
        import math
        from collections import Counter

        if not query or not response:
            return 0.0
        q = str(query).strip()
        r = str(response).strip()
        if len(r) < 10:
            return 2.0

        ql = q.lower()
        rl = r.lower()

        score = 35.0

        stop = {'a','an','the','is','are','was','were','be','been','being','have','has','had',
                'do','does','did','will','would','could','should','may','might','can','to',
                'of','in','for','on','with','at','by','from','as','into','through','and','or',
                'but','if','that','this','these','those','it','its','you','your','we','our',
                'they','their','i','me','my','what','which','who','how','why','when','where',
                'about','also','so','not','no','too','very','just','some','any','all','more',
                'most','am','im'}

        q_words = re.findall(r'[a-z]+', ql)
        q_content = [w for w in q_words if w not in stop and len(w) > 2]

        r_words = re.findall(r'[a-z]+', rl)
        r_content = [w for w in r_words if w not in stop and len(w) > 2]
        r_content_set = set(r_content)

        wc = len(r.split())

        # === 1. Query content coverage ===
        if q_content:
            covered = sum(1 for w in set(q_content) if w in rl)
            cov_ratio = covered / len(set(q_content))
            score += cov_ratio * 15

        # === 2. Bigram coverage (more precise) ===
        q_bigrams = []
        for i in range(len(q_words) - 1):
            if q_words[i] not in stop or q_words[i+1] not in stop:
                q_bigrams.append(q_words[i] + ' ' + q_words[i+1])
        if q_bigrams:
            bg_covered = sum(1 for bg in q_bigrams if bg in rl)
            score += (bg_covered / max(len(q_bigrams), 1)) * 6

        # === 3. Sentence count ===
        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
        n_sent = len(sentences)
        score += min(n_sent, 12) * 0.8

        # === 4. Multi-perspective markers ===
        perspective = ['however','on the other hand','alternatively','in contrast',
                      'although','while','whereas','first','second','third','finally',
                      'one','another','additionally','also']
        p_count = sum(rl.count(m) for m in perspective)
        score += min(p_count, 8) * 0.7

        # === 5. Example/explanation ===
        explain = ['because','since','therefore','for example','for instance','such as',
                   'specifically','this means','due to','consequently']
        e_count = sum(rl.count(m) for m in explain)
        score += min(e_count, 8) * 0.8

        # === 6. Specificity ===
        numbers = re.findall(r'\b\d+\b', r)
        score += min(len(numbers), 8) * 0.5

        # === 7. Structure ===
        list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+\S', r))
        score += min(list_items, 8) * 0.6

        # === 8. STRONG TRUNCATION GATE ===
        is_truncated = False
        stripped = r.rstrip()
        if stripped:
            lc = stripped[-1]
            if lc in ',:;-' or lc not in '.!?"\')]>*}':
                is_truncated = True
            last_30 = stripped[-30:].lower()
            if re.search(r'\b(the|a|an|and|or|but|to|of|in|for|with|is|are|that|this)\s*$',
                         last_30):
                is_truncated = True

        if is_truncated:
            score *= 0.65  # heavy penalty - incomplete fails completeness

        # === 9. Conclusion signal ===
        last_300 = rl[-300:] if len(rl) > 300 else rl
        conclusion = ['in summary','overall','in conclusion','to summarize','finally',
                     'hope this helps','good luck','feel free to','remember','key point']
        if any(c in last_300 for c in conclusion):
            score += 3

        # === 10. Query type completeness checks ===
        is_howto = bool(re.search(r'\bhow (?:can|do|to|should)\b', ql))
        is_what = bool(re.search(r'\bwhat (?:is|are|was|were)\b', ql))
        is_list_req = bool(re.search(r'\b(list|name|give me|provide)\b', ql))
        is_short_req = bool(re.search(r'\b(very short|concise|brief|raw message|one sentence)\b', ql))

        if is_howto and not is_short_req:
            step_pattern = re.findall(r'\b(first|second|third|next|then|finally|step \d)\b', rl)
            if len(step_pattern) >= 2:
                score += 4
            elif list_items >= 3:
                score += 3

        if is_list_req and not is_short_req:
            m = re.search(r'\b(\d+)\b', ql)
            requested = int(m.group(1)) if m else None
            if requested:
                if list_items >= requested:
                    score += 5
                elif list_items >= requested - 1:
                    score += 2
                else:
                    score -= 3
            elif list_items >= 3:
                score += 4

        # === 11. Vocabulary richness ===
        if r_content:
            unique = len(set(r_content)) / len(r_content)
            score += unique * 5

        # === 12. Length calibration ===
        if is_short_req:
            if wc <= 30:
                score += 8
            elif wc > 80:
                score -= 6
        else:
            if wc >= 60:
                score += 4
            elif wc >= 30:
                score += 2
            elif wc < 15:
                score -= 5

        # === 13. Repetition penalty ===
        if r_content and len(r_content) > 20:
            cnt = Counter(r_content)
            most_common_freq = cnt.most_common(1)[0][1] / len(r_content)
            if most_common_freq > 0.08:
                score -= 4

        # === 14. Echo penalty ===
        q_set = set(q_words) - stop
        if r_content_set and q_set and len(r_content_set - q_set) < 4 and wc > 5:
            score -= 8

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 25.0
