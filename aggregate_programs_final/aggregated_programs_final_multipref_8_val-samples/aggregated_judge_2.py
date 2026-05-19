def judging_function(query, response):
    """
    Evidence density and specificity scorer with anti-fabrication gating.
    Rewards concrete details, named entities, numbers with units, but penalizes
    suspiciously over-precise unsourced statistics and vague filler.
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not query or not isinstance(response, str):
            return 0.0

        r = response.strip()
        if len(r) < 5:
            return 0.0
        rl = r.lower()
        q = str(query).strip()
        ql = q.lower()

        words = r.split()
        wc = len(words)
        if wc < 3:
            return 5.0

        sentences = re.split(r'[.!?]+', r)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 4]
        n_sent = max(len(sentences), 1)

        score = 30.0

        # === 1. Numbers with units (precise quantitative info) ===
        units_pat = r'\b\d+(?:\.\d+)?\s*(?:kg|lb|g|mg|oz|m|km|mi|miles|ft|cm|mm|°[CF]|degrees?|%|percent|hours?|minutes?|seconds?|days?|weeks?|months?|years?|mph|km/h|cups?|tbsp|tsp|calories|watts?|volts?|GB|MB)\b'
        units = re.findall(units_pat, r, re.IGNORECASE)
        score += min(len(units), 8) * 2.0

        # Dates
        years = re.findall(r'\b(?:1[5-9]\d{2}|20[0-2]\d)\b', r)
        score += min(len(years), 5) * 1.5

        # Currency
        currency = re.findall(r'\$\d[\d,]*\.?\d*', r)
        score += min(len(currency), 4) * 1.5

        # Standalone numbers (less weight)
        all_nums = re.findall(r'\b\d+\.?\d*\b', r)
        score += min(len(all_nums), 10) * 0.4

        # === 2. Named entities (capitalized mid-sentence) ===
        named_multi = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', r)
        score += min(len(named_multi), 8) * 1.2

        mid_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}\b', r)
        common_caps = {'The','This','That','These','However','Therefore','First','Second',
                       'Third','Finally','Additionally','Also','And','But','For','While'}
        meaningful_caps = [w for w in mid_caps if w not in common_caps]
        score += min(len(meaningful_caps), 12) * 0.5

        # === 3. Example markers ===
        example_markers = ['for example','for instance','such as','specifically',
                          'in particular','namely','e.g.','i.e.','including','like the']
        ex_count = sum(rl.count(m) for m in example_markers)
        score += min(ex_count, 6) * 1.2

        # === 4. Vagueness penalty ===
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|numerous|different)\b',
            r'\bvarious factors\b', r'\bgenerally speaking\b', r'\ba lot of\b',
            r'\band so on\b', r'\bet cetera\b', r'\bvarious\b', r'\bnumerous\b',
            r'\bsort of\b', r'\bkind of\b', r'\bmore or less\b',
            r'\bin some cases\b', r'\bfor the most part\b'
        ]
        vague_count = 0
        for p in vague_patterns:
            vague_count += len(re.findall(p, rl))
        score -= min(vague_count, 8) * 1.5

        # Filler/excited openers
        filler = ['great question','that\'s a great','awesome!','absolutely!','sure!',
                  'i\'m glad you asked','let me','wonderful','fantastic','amazing']
        filler_count = sum(rl.count(f) for f in filler)
        score -= min(filler_count, 4) * 1.0

        # === 5. Content word density ===
        stop = {'a','an','the','is','are','was','were','be','been','being','have','has','had',
                'do','does','did','will','would','could','should','may','might','can','to',
                'of','in','for','on','with','at','by','from','as','into','through','and','or',
                'but','if','that','this','these','those','it','its','you','your','we','our',
                'they','their','i','me','my','also','so','not','no'}
        clean = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words]
        clean = [w for w in clean if w]
        content = [w for w in clean if w not in stop and len(w) > 2]
        if clean:
            ratio = len(content) / len(clean)
            score += ratio * 10 - 3

        # === 6. Unique vocabulary ===
        if content:
            uniqueness = len(set(content)) / len(content)
            score += uniqueness * 6

        # === 7. ANTI-FABRICATION GATE ===
        # Penalize suspiciously precise unsourced statistics
        fake_precise = re.findall(r'\b\d{2,3}\.\d{2,}\s*%', r)
        score -= len(fake_precise) * 4

        # "Exactly X%" without citation
        absolute_fake = re.findall(r'exactly\s+\d+\.?\d*\s*%', rl)
        score -= len(absolute_fake) * 3

        # Sensationalist language
        sensational = ['shocking','mind-blowing','they don\'t want you to know',
                      'cover-up','conspiracy','doctors hate']
        score -= sum(rl.count(s) for s in sensational) * 4

        # === 8. Honest acknowledgment bonus ===
        # When response acknowledges limits / no scientific evidence on dubious claims
        honesty_markers = ['no scientific evidence','no evidence to support',
                          'consult with a healthcare','consult a doctor',
                          'not aware of','i don\'t have information',
                          'recommend consulting','speak with a professional']
        honesty_count = sum(rl.count(h) for h in honesty_markers)
        score += min(honesty_count, 3) * 1.5

        # === 9. Structure that supports evidence delivery ===
        list_items = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s+\S', r))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-*•]\s+\S', r))
        bold = len(re.findall(r'\*\*[^*]+\*\*', r))
        score += min(list_items + bullet_items, 10) * 0.4
        score += min(bold, 8) * 0.4

        # === 10. Parenthetical clarifications (signal of precision) ===
        parens = re.findall(r'\([^)]{3,80}\)', r)
        score += min(len(parens), 5) * 0.7

        # === 11. URLs and references ===
        urls = re.findall(r'https?://\S+', r)
        score += min(len(urls), 3) * 1.5
        refs = sum(rl.count(t) for t in ['according to','research shows','studies show','reported by'])
        score += min(refs, 4) * 1.2

        # === 12. Length calibration (avoid penalizing concise answers) ===
        if 30 <= wc <= 500:
            score += 3
        elif wc < 15:
            score -= 3
        elif wc > 700:
            score += 1  # neutral for very long

        # === 13. Truncation penalty ===
        last = r.rstrip()
        if last and last[-1] not in '.!?"\')]:>*':
            score -= 6

        return round(max(0.0, min(100.0, score)), 2)

    except Exception:
        return 25.0
