def judging_function(query, response):
    """
    Denoising specialist: rewards directness on direct questions, calibrates
    hedging vs confidence appropriately, and applies a strong anti-truncation
    gate. Targets failure cases where verbose hedging or truncated long
    responses were preferred over crisp direct answers.
    """
    try:
        import re
        import math

        if not query or not response or not isinstance(response, str):
            return 0.0
        q = str(query).strip()
        r = str(response).strip()
        if len(r) < 5:
            return 5.0
        ql = q.lower()
        rl = r.lower()

        wc = len(r.split())
        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        n_sent = max(len(sentences), 1)

        score = 50.0

        # ============================================================
        # 1. STRONG TRUNCATION GATE
        # ============================================================
        is_truncated = False
        truncation_severity = 0

        stripped = r.rstrip()
        if stripped:
            lc = stripped[-1]
            # Hard truncation signals
            if lc in ',:;-' or lc.isalpha():
                # Most truncation ends mid-word/sentence
                if lc.isalpha() and len(stripped) > 50:
                    is_truncated = True
                    truncation_severity = 2
                elif lc in ',:;-':
                    is_truncated = True
                    truncation_severity = 2

            last_40 = stripped[-40:].lower()
            if re.search(r'\b(the|a|an|and|or|but|to|of|in|for|with|is|are|was|were|that|this|will|can|on)\s*$',
                         last_40):
                is_truncated = True
                truncation_severity = max(truncation_severity, 2)

            # Soft truncation: no terminal punct but maybe just informal end
            if lc not in '.!?"\')]>*}:' and not is_truncated:
                is_truncated = True
                truncation_severity = 1

        if truncation_severity == 2:
            score -= 14
        elif truncation_severity == 1:
            score -= 5
        else:
            score += 2

        # ============================================================
        # 2. DIRECTNESS DETECTION
        # ============================================================
        # Is this a direct factual / yes-no / why question?
        is_yesno = bool(re.search(r'^\s*(do|does|did|is|are|was|were|can|could|would|should|will|has|have|had)\b', ql))
        is_what_simple = bool(re.search(r'^\s*(what|who|when|where) (?:is|are|was|were)\b', ql))
        is_why = bool(re.search(r'^\s*why\b', ql))

        # Direct answer at start?
        first_30 = rl[:60]
        direct_start = bool(re.search(
            r'^(yes|no|the |a |an |[a-z]+ (?:is|are|was|were))', first_30))

        if (is_yesno or is_what_simple) and direct_start:
            score += 5

        # ============================================================
        # 3. HEDGING CALIBRATION
        # ============================================================
        hedges = ['perhaps','possibly','maybe','might be','could be','not necessarily',
                  'it depends','arguably','not sure','hard to say','i think','i believe',
                  'probably','potentially']
        hedge_count = sum(rl.count(h) for h in hedges)

        if is_why or is_what_simple:
            # On direct questions, excessive hedging hurts
            if hedge_count >= 4:
                score -= 5
            elif hedge_count >= 2:
                score -= 1
        else:
            # For complex/opinion questions, moderate hedging is fine
            if 1 <= hedge_count <= 4:
                score += 2

        # ============================================================
        # 4. AVOID OVER-VERBOSE WHEN QUERY IS SIMPLE
        # ============================================================
        q_wc = len(q.split())
        # Simple short query (under 15 words) doesn't need 500+ word response
        if q_wc < 15 and wc > 250 and not re.search(r'\b(detailed|comprehensive|thorough|explain in detail)\b', ql):
            # Some buffer - not all long responses are bad
            score -= 4

        # If concise/short requested, definitely penalize
        wants_brief = bool(re.search(r'\b(very short|concise|brief|one sentence|raw message)\b', ql))
        if wants_brief:
            if wc <= 30:
                score += 12
            elif wc > 100:
                score -= 12

        # ============================================================
        # 5. DEFINITIVE STATEMENTS bonus for factual questions
        # ============================================================
        if is_yesno or is_what_simple or is_why:
            # First sentence quality
            if sentences:
                first_sent = sentences[0]
                fl = len(first_sent.split())
                if 5 <= fl <= 40 and re.search(r'\b(is|are|was|were|will|can)\b', first_sent.lower()):
                    score += 3

        # ============================================================
        # 6. BROAD QUALITY (so program votes on normal cases too)
        # ============================================================
        # Specificity
        numbers = len(re.findall(r'\b\d+\b', r))
        score += min(numbers, 6) * 0.4

        # Named entities (proper nouns)
        proper = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', r)
        score += min(len(proper), 8) * 0.4

        # Examples / explanations
        depth = sum(rl.count(m) for m in
            ['because','therefore','for example','for instance','such as','specifically'])
        score += min(depth, 6) * 0.7

        # Structure for complex queries
        if wc > 80:
            list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', r))
            if list_items >= 3:
                score += 3

        # ============================================================
        # 7. PENALIZE FILLER / EXCESS ENTHUSIASM
        # ============================================================
        filler = ['great question','that\'s a great','awesome!','absolutely!','sure!',
                  'what a great question','wonderful question','interesting question']
        if any(f in rl[:100] for f in filler):
            score -= 2

        # ============================================================
        # 8. PENALIZE ECHOING THE QUERY
        # ============================================================
        # If response is mostly verbatim repetition of query
        q_clean = re.sub(r'[^\w\s]', '', ql)
        r_clean = re.sub(r'[^\w\s]', '', rl[:len(q_clean)+30])
        if len(q_clean) > 20 and q_clean[:100] in r_clean:
            score -= 8

        # ============================================================
        # 9. HONEST UNCERTAINTY BONUS for ambiguous queries
        # ============================================================
        # If query asks about "latest" / "current" / specific recent events
        if re.search(r'\b(latest|recent|current|happening|today)\b', ql):
            if any(m in rl for m in ['not aware','don\'t have','knowledge cutoff','as of my']):
                score += 4

        # ============================================================
        # 10. AVOID PENALIZING HONEST "NO" ANSWERS
        # ============================================================
        # "There is no X" / "I'm not aware of any" are valid direct answers
        if re.search(r'^(there (?:is|are) no|i\'m not aware|no,|i am not aware)', rl[:50]):
            score += 3  # honest direct negation

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 50.0
