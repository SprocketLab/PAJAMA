def judging_function(query, response):
    """
    Structural organization scorer. Rewards good formatting (headers, lists, paragraphs)
    but does NOT penalize short, direct, well-written prose responses when query
    doesn't demand structure.
    """
    try:
        import re
        import math

        if not response or not isinstance(response, str):
            return 0.0
        if not query:
            query = ""

        r = response.strip()
        if len(r) < 5:
            return 0.5
        rl = r.lower()
        ql = str(query).lower()

        lines = r.split('\n')
        non_empty = [l for l in lines if l.strip()]
        words = r.split()
        wc = len(words)

        score = 40.0

        short_req = bool(re.search(
            r'\b(very short|concise|brief|raw message|one sentence|short response|in (?:a |one )?sentence)\b',
            ql))

        # === 1. Headers ===
        md_headers = len(re.findall(r'(?:^|\n)#{1,6}\s+', r))
        bold_headers = len(re.findall(r'(?:^|\n)\s*\*\*[^*]+\*\*\s*:?\s*$', r))
        total_headers = md_headers + bold_headers

        if not short_req:
            if total_headers >= 1:
                score += 4
            if total_headers >= 3:
                score += 4
            if total_headers >= 5:
                score += 2

        # === 2. Lists ===
        numbered = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s+\S', r))
        bullets = len(re.findall(r'(?:^|\n)\s*[-*•]\s+\S', r))
        list_items = numbered + bullets

        if not short_req:
            if list_items >= 2:
                score += 4
            if list_items >= 5:
                score += 4
            if list_items >= 8:
                score += 2
        elif list_items > 0 and short_req:
            score -= 3  # short requested, lists are wrong format

        # === 3. Bold/emphasis ===
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', r))
        if not short_req:
            score += min(bold_count, 8) * 0.5

        # === 4. Paragraphs ===
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', r) if p.strip()]
        n_para = len(paragraphs)

        if not short_req:
            if n_para >= 2:
                score += 3
            if n_para >= 4:
                score += 3
        # For short requests, single paragraph is fine

        # === 5. Wall-of-text penalty ===
        if wc > 150 and n_para == 1 and list_items == 0 and total_headers == 0:
            score -= 8
        if wc > 300 and n_para <= 2 and list_items == 0:
            score -= 5

        # === 6. Whitespace usage ===
        if len(lines) > 5:
            blank = sum(1 for l in lines if not l.strip())
            blank_ratio = blank / len(lines)
            if 0.1 <= blank_ratio <= 0.4:
                score += 3
            elif blank_ratio == 0 and wc > 100:
                score -= 2

        # === 7. List consistency ===
        if numbered >= 3:
            nums = [int(m) for m in re.findall(r'(?:^|\n)\s*(\d+)[.)]', r)]
            if nums and all(nums[i] <= nums[i+1] for i in range(len(nums)-1)):
                score += 2

        # === 8. Sentence-level structure for plain prose ===
        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        n_sent = len(sentences)

        if n_sent >= 3:
            sl = [len(s.split()) for s in sentences]
            avg_sl = sum(sl) / len(sl)
            if 8 <= avg_sl <= 30:
                score += 3
            elif avg_sl < 4 or avg_sl > 50:
                score -= 2

        # === 9. Code blocks ===
        code = len(re.findall(r'```', r))
        if code >= 2:
            score += 2

        # === 10. Truncation ===
        last_char = r.rstrip()[-1] if r.rstrip() else ''
        if last_char not in '.!?"\')]>*}':
            score -= 6

        # === 11. Inline formatting variety ===
        italic = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', r))
        code_span = len(re.findall(r'`[^`]+`', r))
        inline = bold_count + italic + code_span
        if inline >= 3 and not short_req:
            score += 2

        # === 12. Short response handling ===
        if short_req:
            if wc <= 30 and r.rstrip().endswith(('.','!','?','"',"'")):
                score += 10  # nailed the brief
            elif wc <= 70:
                score += 2
            else:
                score -= 8

        # === 13. Length-appropriate ===
        if not short_req:
            if wc >= 50:
                score += 2
            if wc >= 150:
                score += 2
            if wc < 15:
                score -= 4

        # === 14. Anti-overformatting for tiny content ===
        if wc < 50 and (md_headers > 2 or list_items > 8):
            score -= 3  # overdoing structure for short content

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 25.0
