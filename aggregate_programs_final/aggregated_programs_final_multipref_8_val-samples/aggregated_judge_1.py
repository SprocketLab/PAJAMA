def judging_function(query, response):
    """
    Broad quality composite: query coverage + depth + structure + specificity.
    Strongly penalizes truncation. Calibrates expectations to query type.
    """
    try:
        import re
        import math
        from collections import Counter

        if not query or not response or not isinstance(response, str):
            return 0.0

        q = str(query).strip()
        r = str(response).strip()
        if len(r) < 5:
            return 0.0

        ql = q.lower()
        rl = r.lower()

        stop = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                'do','does','did','will','would','could','should','may','might','can','shall',
                'to','of','in','for','on','with','at','by','from','as','into','through','and',
                'but','or','if','then','than','that','this','these','those','it','its','i','me',
                'my','we','our','you','your','he','she','they','them','their','what','which',
                'who','how','why','when','where','also','about','very','just','so','not','no',
                'too','am','im','get','got','some','any','all','more','most','other','such'}

        q_words = re.findall(r'[a-z]+', ql)
        q_content = [w for w in q_words if w not in stop and len(w) > 2]
        q_content_set = set(q_content)

        r_words = re.findall(r'[a-z]+', rl)
        r_content = [w for w in r_words if w not in stop and len(w) > 2]
        r_content_set = set(r_content)

        score = 50.0

        # === 1. Query coverage (0-15) ===
        if q_content_set:
            cov = len(q_content_set & r_content_set) / len(q_content_set)
            score += cov * 15 - 5  # center on partial coverage

        # === 2. Information breadth (0-12) ===
        unique = len(r_content_set)
        score += min(unique / 40, 1.0) * 12 - 4

        # === 3. Sentence structure (0-10) ===
        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
        n_sent = len(sentences)
        if n_sent >= 4:
            score += 6
        elif n_sent >= 2:
            score += 3
        elif n_sent == 1:
            score -= 2

        # === 4. Depth markers (0-10) ===
        depth_markers = ['because','since','therefore','thus','as a result','due to',
                         'this means','for example','for instance','such as','specifically',
                         'in particular','however','while','although','first','second','finally',
                         'additionally','furthermore','moreover','important','note that']
        depth = sum(rl.count(m) for m in depth_markers)
        score += min(depth, 8) * 0.8

        # === 5. Specificity (0-10) ===
        numbers = re.findall(r'\b\d+\.?\d*\b', r)
        score += min(len(numbers), 6) * 0.7
        # Proper nouns mid-sentence
        proper = 0
        for s in sentences:
            sw = s.split()
            for i, w in enumerate(sw):
                if i > 0 and len(w) > 2 and w[0].isupper() and not w.isupper():
                    proper += 1
        score += min(proper, 8) * 0.5

        # === 6. Structure / formatting (0-8) ===
        list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', r))
        bold = len(re.findall(r'\*\*[^*]+\*\*', r))
        headers = len(re.findall(r'(?:^|\n)#{1,4}\s+', r))
        struct = min(list_items, 8) * 0.5 + min(bold, 6) * 0.3 + min(headers, 4) * 0.6
        score += min(struct, 8)

        # === 7. TRUNCATION GATE (heavy) ===
        # Detect if response was cut off mid-thought
        is_truncated = False
        stripped = r.rstrip()
        if stripped:
            last_char = stripped[-1]
            if last_char not in '.!?"\')]}:;>*':
                is_truncated = True
            last_50 = stripped[-50:].lower()
            if re.search(r'\b(the|a|an|and|or|but|to|of|in|for|with|is|are|was|were|that|this|these|in an a)\s*$', last_50):
                is_truncated = True
            # Mid-word truncation
            if len(stripped) > 30 and not re.search(r'[.!?]\s*[A-Z"\']?\s*$|^.{0,3}$', stripped[-10:]):
                # check if last token looks like incomplete sentence
                last_token = stripped.split()[-1] if stripped.split() else ''
                if last_token and last_token[-1].isalpha() and last_char.isalpha():
                    is_truncated = True

        if is_truncated:
            score *= 0.70  # significant penalty but not destructive

        # === 8. Query-type calibration ===
        # Detect "very short" / "concise" requests
        short_request = bool(re.search(r'\b(very short|concise|brief|short and|raw message|one sentence|in (?:a |one )?sentence|short response)\b', ql))
        word_count = len(r.split())
        if short_request:
            if word_count <= 35:
                score += 8  # reward brevity
            elif word_count <= 70:
                score -= 2
            else:
                score -= 8

        # Detect explicit list/count requests
        list_req = re.search(r'\b(list|name|give me|provide)\s+(\d+|some|a few)\b', ql)
        if list_req:
            if list_items >= 2 or any(re.search(r'^\d+[.)]', s.strip()) for s in r.split('\n')):
                score += 3

        # === 9. Response too short for substantial query ===
        if word_count < 15 and not short_request:
            score *= 0.6
        elif word_count < 30 and len(q_content) > 4 and not short_request:
            score *= 0.85

        # === 10. Repetition/echo penalty ===
        # Penalize if response just echoes the query
        if word_count > 5:
            q_set = set(q_words) - stop
            r_set = set(r_words) - stop
            if q_set and r_set:
                if r_set.issubset(q_set) or len(r_set - q_set) < 3:
                    score -= 10

        # === 11. Vague filler penalty ===
        vague = ['many people think','it depends','various factors',
                 'there are many ways','generally speaking','some people',
                 'it really depends']
        score -= sum(rl.count(v) for v in vague) * 1.2

        # === 12. Hedging-vs-direct-answer calibration ===
        # For "why" / "what is" queries, excessive hedging hurts
        is_direct = bool(re.search(r'^\s*(why|what is|what are|who is|who was|name|when did|where is|how many)\b', ql))
        if is_direct:
            hedges = sum(1 for h in ['not necessarily','it may be','arguably','perhaps','possibly','i\'m not sure']
                        if h in rl)
            if hedges >= 2:
                score -= 4

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 25.0
