def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a substantially different approach:
    - Named entity density (capitalized multi-word phrases, proper nouns)
    - Citation/reference pattern detection
    - Specificity scoring (dates, numbers, named sources)
    - Hedging calibration (appropriate uncertainty vs overconfidence)
    - Anti-hallucination checks (sensationalism, conspiracy markers, unsourced absolutes)
    - Discourse structure analysis (causal connectives, evidence framing)
    - Information density via unique content word ratio
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0

        resp = response.strip()
        if len(resp) < 10:
            return 0.5

        words = resp.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if s.strip()]
        sent_count = max(len(sentences), 1)

        score = 50.0  # Start at midpoint

        # ============================================================
        # 1. NAMED ENTITY DENSITY (capitalized phrases not at sentence start)
        # ============================================================
        # Find capitalized words that aren't at the start of sentences
        cap_pattern = re.findall(r'(?<=[a-z,;:] )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', resp)
        # Also find standalone proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', resp)
        # Filter out common sentence starters
        common_starters = {'The', 'This', 'That', 'These', 'Those', 'There', 'Here',
                          'However', 'But', 'And', 'Also', 'While', 'When', 'Where',
                          'What', 'How', 'Why', 'Who', 'Which', 'Some', 'Many', 'Most',
                          'For', 'From', 'Its', 'Being', 'Can', 'Could', 'Would',
                          'Should', 'May', 'Might', 'Will', 'Has', 'Have', 'Had',
                          'Not', 'Just', 'Very', 'More', 'Less', 'Much', 'Any',
                          'All', 'Each', 'Every', 'Both', 'Either', 'Neither',
                          'Yes', 'No', 'Sure', 'Well', 'Now', 'Then', 'Still',
                          'Got', 'Get', 'Let', 'Try', 'See', 'One', 'Two',
                          'Please', 'Thanks', 'Welcome', 'Hey', 'Hello',
                          'Feeling', 'Trying', 'Stepping', 'Looking', 'Going',
                          'Do', 'Did', 'Does', 'Are', 'Is', 'Was', 'Were',
                          'If', 'So', 'As', 'Or', 'In', 'On', 'At', 'To',
                          'My', 'Your', 'Our', 'His', 'Her', 'It', 'We', 'They'}
        
        meaningful_proper = [p for p in proper_nouns if p not in common_starters]
        entity_density = len(meaningful_proper) / max(word_count, 1)
        # Reward moderate entity density (0.02-0.10 is good range)
        if entity_density > 0.01:
            score += min(entity_density * 80, 8.0)

        # ============================================================
        # 2. SPECIFIC REFERENCE PATTERNS
        # ============================================================
        # Books, papers, titles (italics in markdown, quotes, asterisks)
        title_refs = len(re.findall(r'\*[A-Z][^*]{3,60}\*', resp))
        quoted_refs = len(re.findall(r'"[A-Z][^"]{3,80}"', resp))
        
        # User references (u/, @, by [Name])
        user_refs = len(re.findall(r'(?:u/|@|by\s+)[A-Z]\w+', resp))
        
        # URL or link patterns
        url_refs = len(re.findall(r'https?://|www\.|\.com|\.org|\.edu', resp))
        
        # "according to", "as described by", "as noted by"
        attribution_phrases = len(re.findall(
            r'(?:according to|as (?:described|noted|mentioned|stated|argued|explained) (?:by|in))',
            resp, re.IGNORECASE))
        
        ref_score = (title_refs * 2.5 + quoted_refs * 2.0 + user_refs * 1.5 + 
                     url_refs * 1.0 + attribution_phrases * 2.0)
        score += min(ref_score, 10.0)

        # ============================================================
        # 3. SPECIFICITY: dates, numbers, measurements
        # ============================================================
        # Year patterns
        years = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', resp)
        # Specific numbers (not just 1, 2, 3)
        specific_numbers = re.findall(r'\b\d{2,}\b', resp)
        # Percentages
        percentages = re.findall(r'\d+%', resp)
        # Measurements
        measurements = re.findall(r'\d+\s*(?:mg|kg|lb|oz|cm|mm|km|miles?|hours?|minutes?|days?|years?|months?|weeks?)\b', resp, re.IGNORECASE)
        
        specificity_count = len(years) * 1.5 + len(specific_numbers) * 0.5 + len(percentages) * 1.0 + len(measurements) * 1.0
        specificity_density = specificity_count / max(word_count, 1)
        
        # Reward specificity but cap it (too many unsourced numbers is suspicious)
        if specificity_density < 0.05:
            score += min(specificity_count * 1.2, 8.0)
        else:
            # Excessive numbers without context might indicate hallucination
            score += 3.0  # Still some credit but reduced

        # ============================================================
        # 4. HEDGING CALIBRATION
        # ============================================================
        # Appropriate hedging words
        hedging_words = re.findall(
            r'\b(?:possibly|perhaps|likely|unlikely|might|may|could|tends? to|'
            r'generally|typically|often|sometimes|usually|approximately|roughly|'
            r'it seems|appears to|suggests?|in my (?:experience|opinion)|'
            r'I think|I believe|arguably|plausibly|probably|'
            r'one (?:possibility|interpretation)|as far as I know|'
            r'to my knowledge|from what I|if I recall)\b',
            resp, re.IGNORECASE)
        
        hedge_ratio = len(hedging_words) / max(sent_count, 1)
        
        # Overconfidence markers (absolute claims without evidence framing)
        absolute_markers = re.findall(
            r'\b(?:always|never|definitely|certainly|absolutely|undoubtedly|'
            r'without a doubt|100%|guaranteed|proven fact|everyone knows|'
            r'obviously|clearly everyone|no one can deny|it is known that|'
            r'the truth is|the fact is that)\b',
            resp, re.IGNORECASE)
        
        absolute_ratio = len(absolute_markers) / max(sent_count, 1)
        
        # Good hedging: some but not excessive
        if 0.05 <= hedge_ratio <= 0.8:
            score += min(hedge_ratio * 8, 6.0)
        elif hedge_ratio > 0.8:
            score -= 2.0  # Over-hedging reduces confidence
        
        # Penalize overconfidence
        score -= min(absolute_ratio * 5, 6.0)

        # ============================================================
        # 5. ANTI-HALLUCINATION / SENSATIONALISM CHECK
        # ============================================================
        sensational_patterns = re.findall(
            r'\b(?:shocking|mind-blowing|unbelievable|they don\'?t want you to know|'
            r'exposed|cover-?up|conspiracy|wake up|sheeple|mainstream media lies|'
            r'big pharma|deep state|secret(?:ly)?(?:\s+agenda)?|hidden truth|'
            r'bombshell|explosive|jaw-?dropping|insane(?:ly)?|'
            r'you won\'?t believe|EXPOSED|TRUTH|WAKE UP)\b',
            resp, re.IGNORECASE)
        
        score -= len(sensational_patterns) * 4.0

        # Check for suspiciously precise statistics without sources
        unsourced_precise = re.findall(r'\b\d{1,3}\.\d{1,2}%\b', resp)
        # If there are precise percentages but no attribution, penalize slightly
        if len(unsourced_precise) > 0 and attribution_phrases == 0 and title_refs == 0:
            score -= len(unsourced_precise) * 1.5

        # ============================================================
        # 6. DISCOURSE STRUCTURE: causal/evidence connectives
        # ============================================================
        evidence_connectives = re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as a result|'
            r'this (?:means|implies|suggests|indicates)|for (?:example|instance)|'
            r'such as|specifically|in particular|namely|e\.g\.|i\.e\.|'
            r'the reason|due to|owing to|given that|considering|'
            r'on the other hand|however|although|while|whereas|'
            r'in contrast|similarly|moreover|furthermore|additionally|'
            r'first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|'
            r'in other words|to put it|that is to say)\b',
            resp, re.IGNORECASE)
        
        connective_density = len(evidence_connectives) / max(sent_count, 1)
        # Reward structured reasoning
        score += min(connective_density * 5, 7.0)

        # ============================================================
        # 7. INFORMATION DENSITY (unique content words / total words)
        # ============================================================
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
                     'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
                     'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                     'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
                     'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                     'he', 'his', 'she', 'her', 'they', 'them', 'their', 'this',
                     'that', 'these', 'those', 'what', 'which', 'who', 'whom',
                     'when', 'where', 'why', 'how', 'if', 'then', 'else', 'up',
                     'out', 'about', 'over', 'down', 'off', 'again', 'there',
                     'here', 'once', 'am', 'an'}
        
        lower_words = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words]
        content_words = [w for w in lower_words if w and w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        if len(content_words) > 0:
            content_diversity = len(unique_content) / len(content_words)
            # Reward moderate diversity (0.4-0.8 is good)
            if 0.35 <= content_diversity <= 0.85:
                score += content_diversity * 5
            elif content_diversity > 0.85:
                score += 3.0  # Very diverse but might be scattered
        
        # ============================================================
        # 8. RESPONSE SUBSTANTIVENESS
        # ============================================================
        # Reward responses that are substantive but not just padding
        if word_count < 15:
            score -= 5.0
        elif 15 <= word_count < 30:
            score += 1.0
        elif 30 <= word_count < 80:
            score += 3.0
        elif 80 <= word_count < 250:
            score += 5.0
        elif word_count >= 250:
            score += 4.0  # Slightly less for very long (might be padding)

        # ============================================================
        # 9. QUERY-RESPONSE TOPICAL ALIGNMENT (using content word Jaccard)
        # ============================================================
        query_lower = [w.lower().strip('.,;:!?()[]{}"\'-') for w in query.split()]
        query_content = set(w for w in query_lower if w and w not in stop_words and len(w) > 2)
        
        if query_content and unique_content:
            overlap = query_content.intersection(unique_content)
            # Use a modified Jaccard that doesn't over-penalize long responses
            relevance = len(overlap) / max(len(query_content), 1)
            score += relevance * 6.0

        # ============================================================
        # 10. EXPLANATORY DEPTH MARKERS
        # ============================================================
        # Conditional/nuanced reasoning
        conditional_patterns = re.findall(
            r'\b(?:if you|depending on|it depends|in (?:this|that) case|'
            r'the trade-?off|on one hand|the (?:key|important|crucial) (?:thing|point|aspect)|'
            r'worth noting|keep in mind|bear in mind|'
            r'the difference (?:is|between)|distinction between|'
            r'in practice|in theory|essentially|fundamentally)\b',
            resp, re.IGNORECASE)
        
        score += min(len(conditional_patterns) * 1.8, 6.0)

        # ============================================================
        # 11. PERSONAL EXPERIENCE / ANECDOTAL EVIDENCE (can be good in context)
        # ============================================================
        personal_exp = re.findall(
            r'\b(?:in my experience|I\'?ve (?:seen|found|noticed|worked|had)|'
            r'from my|when I was|I used to|I currently|I work(?:ed)?)\b',
            resp, re.IGNORECASE)
        
        # Personal experience is mildly positive (shows real knowledge)
        score += min(len(personal_exp) * 1.0, 3.0)

        # ============================================================
        # 12. CODE/TECHNICAL CONTENT BONUS
        # ============================================================
        has_code = bool(re.search(r'```', resp))
        has_technical = bool(re.search(r'(?:SELECT|FROM|WHERE|JOIN|CREATE|INSERT|def |class |import |function )', resp))
        
        # Check if query seems technical
        query_is_technical = bool(re.search(r'(?:SQL|code|program|function|table|CREATE|SELECT|API|algorithm)', query, re.IGNORECASE))
        
        if query_is_technical and (has_code or has_technical):
            score += 4.0

        # ============================================================
        # 13. PENALIZE GENERIC / BOT-LIKE RESPONSES
        # ============================================================
        generic_openers = re.findall(
            r'^(?:Sure,? I can help|I\'?d be happy to|Great question|'
            r'That\'?s a (?:great|good|interesting) question|'
            r'Welcome to|Please read our rules)',
            resp, re.IGNORECASE)
        
        # Mild penalty for generic openers (they waste space)
        score -= len(generic_openers) * 1.5

        # Penalize meta-responses that don't answer
        meta_non_answers = re.findall(
            r'(?:your (?:question|query) (?:is|was)|I (?:cannot|can\'t) (?:answer|help)|'
            r'please (?:clarify|provide more)|do not fear|'
            r'can you (?:please )?describe)',
            resp, re.IGNORECASE)
        
        score -= len(meta_non_answers) * 3.0

        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        
        # Scale to 0-10 for output
        final_score = score / 10.0
        
        return round(final_score, 2)

    except Exception:
        return 5.0