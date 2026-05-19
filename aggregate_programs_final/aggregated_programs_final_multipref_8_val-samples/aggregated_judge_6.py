def judging_function(query, response):
    """
    Denoising specialist: rewards responses that MATCH the query's explicit
    or implicit constraints (length, format, language, item count, etc.).
    Targets failure modes where verbose responses win over concise ones
    despite the query asking for brevity.
    """
    try:
        import re

        if not query or not response or not isinstance(response, str):
            return 0.0
        q = str(query).strip()
        r = str(response).strip()
        if len(r) < 3:
            return 5.0
        ql = q.lower()
        rl = r.lower()

        wc = len(r.split())
        words = r.split()

        score = 50.0

        # ============================================================
        # 1. EXPLICIT BREVITY REQUEST
        # ============================================================
        brief_patterns = [
            r'\bvery short\b', r'\bconcise\b', r'\bbrief\b', r'\bshort response\b',
            r'\bone sentence\b', r'\bin a sentence\b', r'\braw message\b',
            r'\bdon\'?t say\b', r'\bonly return\b', r'\bsuccinct\b',
            r'\bshort (?:and|but) (?:clear|sweet)\b', r'\bto the point\b',
        ]
        wants_brief = any(re.search(p, ql) for p in brief_patterns)

        if wants_brief:
            if wc <= 25:
                score += 18
            elif wc <= 50:
                score += 8
            elif wc <= 100:
                score -= 6
            elif wc <= 200:
                score -= 14
            else:
                score -= 20

            # Penalize meta-commentary in brief responses
            meta = ['here is','hey here','here\'s the','let me know','i hope','hope this',
                    'as requested','as you asked']
            for m in meta:
                if rl.startswith(m):
                    score -= 6
                    break

        # ============================================================
        # 2. EXPLICIT LENGTH/DEPTH REQUEST
        # ============================================================
        wants_long = bool(re.search(
            r'\b(detailed|comprehensive|thorough|in.depth|elaborate|explain in detail|2000 characters?|long)\b',
            ql))
        if wants_long:
            if wc >= 200:
                score += 8
            elif wc >= 100:
                score += 3
            else:
                score -= 6

        # ============================================================
        # 3. EXPLICIT COUNT REQUEST (e.g., "name 4 colors")
        # ============================================================
        count_match = re.search(r'\b(?:name|list|give|provide|suggest|create)\s+(?:me\s+)?(\d+|two|three|four|five|six|seven|eight|nine|ten)\b', ql)
        if count_match:
            num_map = {'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
            n_str = count_match.group(1)
            requested = num_map.get(n_str, None)
            if requested is None:
                try:
                    requested = int(n_str)
                except:
                    requested = None
            if requested:
                # Count items in response
                items = len(re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+\S', r))
                if items == 0:
                    items = len(re.findall(r'[.!?,;\n]', r))  # rough
                if abs(items - requested) <= 1:
                    score += 6

        # ============================================================
        # 4. CONSTRAINT-FOLLOWING (e.g., "starts with letters a/b/c/d")
        # ============================================================
        letter_constraint = re.search(r'start[s]?\s+with\s+([a-z](?:\s*[,/]\s*[a-z])+|[a-z](?:\s+(?:or|,|and)\s+[a-z])+)', ql)
        if letter_constraint:
            constraint_str = letter_constraint.group(1)
            allowed_letters = set(re.findall(r'[a-z]', constraint_str))
            # Extract items from response
            response_items = re.findall(r'(?:^|\n)\s*\d+[.)]\s*([A-Za-z][A-Za-z\s\-]+)', r)
            if not response_items:
                response_items = re.findall(r'(?:^|\n)\s*[-*•]\s*([A-Za-z][A-Za-z\s\-]+)', r)
            if response_items:
                matched = sum(1 for it in response_items if it.strip() and it.strip()[0].lower() in allowed_letters)
                total = len(response_items)
                if total > 0:
                    match_ratio = matched / total
                    score += match_ratio * 15 - 5

                # Bonus for covering different letters
                covered_letters = set()
                for it in response_items:
                    if it.strip():
                        covered_letters.add(it.strip()[0].lower())
                covered_count = len(covered_letters & allowed_letters)
                if covered_count >= len(allowed_letters):
                    score += 5

        # ============================================================
        # 5. LANGUAGE-MATCHING
        # ============================================================
        # If query is in English and response is in another language unexpectedly, penalize.
        # But if query explicitly requested another language, reward.
        non_english_chars = len(re.findall(r'[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]', r))
        wants_other_lang = bool(re.search(r'\b(in chinese|in spanish|in french|in german|in japanese|in russian)\b', ql))
        if non_english_chars > 30 and not wants_other_lang:
            # Check if query had non-english chars too
            q_non_english = len(re.findall(r'[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]', q))
            if q_non_english < 5:
                score -= 8

        # ============================================================
        # 6. NON-ECHO CHECK
        # ============================================================
        # Penalize responses that just echo the query back
        if wc > 8:
            # Check if response is mostly the same as the query
            q_set = set(re.findall(r'\b\w+\b', ql))
            r_set = set(re.findall(r'\b\w+\b', rl))
            if q_set and len(q_set & r_set) / len(q_set) > 0.85 and len(r_set - q_set) < 5:
                score -= 12

        # ============================================================
        # 7. COMPLETENESS - not truncated
        # ============================================================
        stripped = r.rstrip()
        if stripped:
            lc = stripped[-1]
            if lc not in '.!?"\')]>*}':
                score -= 5

        # ============================================================
        # 8. INSTRUCTION FOLLOWING - "DON'T SAY X"
        # ============================================================
        dont_say = re.search(r'do(?:\s|n\'?)?t\s+say\s+[\"\'](.+?)[\"\']', ql)
        if dont_say:
            forbidden = dont_say.group(1).lower()
            if forbidden[:20] in rl[:100]:
                score -= 10

        # ============================================================
        # 9. FORMAT-MATCHING (email, list, recipe, etc.)
        # ============================================================
        wants_email = bool(re.search(r'\b(email|message|letter)\b', ql))
        if wants_email:
            has_greeting = bool(re.search(r'^(hi|hey|hello|dear|greetings)\b', rl[:50]))
            has_signoff = bool(re.search(r'\b(sincerely|best|regards|cheers|thanks|yours)\b', rl[-150:]))
            if has_greeting:
                score += 2
            if has_signoff:
                score += 2
            if re.search(r'^subject:', rl, re.MULTILINE):
                score += 3

        wants_recipe = bool(re.search(r'\b(recipe|how to make|how to bake|how to cook)\b', ql))
        if wants_recipe:
            has_ingredients = bool(re.search(r'\bingredients?\b', rl))
            has_steps = bool(re.search(r'\b(steps?|directions?|instructions?)\b|(?:^|\n)\s*\d+[.)]', rl))
            if has_ingredients:
                score += 3
            if has_steps:
                score += 3

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 50.0
