def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a different approach:
    - Citation/reference pattern detection (specific names, dates, numbers, URLs)
    - Hallucination red-flag detection (overly precise unsourced stats, absolute claims)
    - Appropriate hedging vs over-hedging balance
    - Sensationalism and conspiracy language penalties
    - Structural credibility signals (organized reasoning, step-by-step)
    - Specificity-to-vagueness ratio
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower().strip()
        query_lower = query.lower().strip()
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        score = 50.0  # Start at midpoint
        
        # ============================================================
        # FEATURE 1: Verifiable fact indicators (dates, numbers, names)
        # ============================================================
        
        # Count date patterns (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
        date_pattern = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', response)
        
        # Count specific numbers with units (measurements, quantities)
        number_with_unit = re.findall(
            r'\b\d+\.?\d*\s*(?:kg|lb|lbs|m|km|miles|feet|ft|cm|mm|inches|'
            r'hours|minutes|seconds|degrees|°|percent|%|mph|m/s|liters|gallons|'
            r'cups|tbsp|tsp|oz|grams|mg|ml)\b', resp_lower
        )
        
        # Count standalone meaningful numbers
        all_numbers = re.findall(r'\b\d+\.?\d*\b', response)
        
        # Ratio of factual anchors per 100 words
        factual_anchors = len(year_pattern) + len(date_pattern) * 2 + len(number_with_unit) * 1.5
        factual_density = factual_anchors / max(word_count, 1) * 100
        score += min(factual_density * 2.0, 8.0)
        
        # ============================================================
        # FEATURE 2: Hallucination red-flags
        # ============================================================
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d{2,}\s*%', response)
        score -= len(precise_stats) * 2.0
        
        # Absolute claims without evidence
        absolute_phrases = [
            'it is a fact that', 'it is proven that', 'everyone knows',
            'it is well known that', 'studies have shown that',  # without citation
            'research proves', 'science has proven', 'always causes',
            'never fails to', 'guaranteed to', 'without exception',
            'there is no doubt', 'undeniably', 'irrefutably',
            '100% of', '100 percent of', 'all experts agree',
            'no one has ever', 'every single', 'in all cases',
            'has been conclusively', 'definitively proven'
        ]
        
        absolute_count = 0
        for phrase in absolute_phrases:
            absolute_count += resp_lower.count(phrase)
        score -= absolute_count * 3.0
        
        # ============================================================
        # FEATURE 3: Appropriate hedging (balanced, not over/under)
        # ============================================================
        
        hedging_words = [
            'may', 'might', 'could', 'possibly', 'perhaps', 'likely',
            'generally', 'typically', 'often', 'usually', 'tends to',
            'in many cases', 'it appears', 'it seems', 'suggest',
            'approximately', 'roughly', 'around', 'about',
            'depending on', 'in some cases', 'can vary',
            'not necessarily', 'it depends', 'consider'
        ]
        
        hedge_count = 0
        for h in hedging_words:
            hedge_count += resp_lower.count(h)
        
        # Optimal hedging: some is good, too much is bad
        hedge_ratio = hedge_count / max(word_count, 1) * 100
        if hedge_ratio < 0.5:
            score += 0  # No hedging - neutral
        elif hedge_ratio < 3.0:
            score += 4.0  # Good hedging
        elif hedge_ratio < 5.0:
            score += 2.0  # Moderate hedging
        else:
            score -= 2.0  # Over-hedging suggests uncertainty
        
        # ============================================================
        # FEATURE 4: Sensationalism and conspiracy penalties
        # ============================================================
        
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'insane',
            'you won\'t believe', 'jaw-dropping', 'bombshell',
            'explosive', 'devastating', 'terrifying', 'horrifying',
            'outrageous', 'scandalous', 'breaking'
        ]
        
        conspiracy_words = [
            'cover-up', 'coverup', 'they don\'t want you to know',
            'mainstream media', 'big pharma', 'deep state', 'wake up',
            'sheeple', 'new world order', 'illuminati', 'suppressed',
            'hidden truth', 'the elite', 'controlled by', 'puppet',
            'false flag', 'hoax', 'plandemic', 'truth movement'
        ]
        
        sensational_count = sum(1 for w in sensational_words if w in resp_lower)
        conspiracy_count = sum(1 for w in conspiracy_words if w in resp_lower)
        
        score -= sensational_count * 4.0
        score -= conspiracy_count * 6.0
        
        # ============================================================
        # FEATURE 5: Structural credibility signals
        # ============================================================
        
        # Numbered steps or lists (shows organized thinking)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        has_numbered_list = len(numbered_items) >= 2
        
        # Bold/markdown formatting (shows structure)
        bold_items = re.findall(r'\*\*[^*]+\*\*', response)
        has_bold = len(bold_items) >= 1
        
        # Paragraphs (shows developed reasoning)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        para_count = len(paragraphs)
        
        # Reward structured responses
        if has_numbered_list:
            score += 3.0
        if has_bold:
            score += 2.0
        if para_count >= 2:
            score += min(para_count * 0.5, 3.0)
        
        # ============================================================
        # FEATURE 6: Specificity-to-vagueness ratio
        # ============================================================
        
        vague_words = [
            'stuff', 'things', 'something', 'somehow', 'kind of',
            'sort of', 'like', 'basically', 'whatever', 'etc',
            'and so on', 'and stuff', 'you know', 'i guess',
            'pretty much', 'more or less'
        ]
        
        specific_indicators = [
            'specifically', 'in particular', 'for example', 'for instance',
            'such as', 'including', 'namely', 'e.g.', 'i.e.',
            'according to', 'based on', 'defined as'
        ]
        
        vague_count = sum(1 for v in vague_words if v in resp_lower)
        specific_count = sum(1 for s in specific_indicators if s in resp_lower)
        
        score += specific_count * 2.0
        score -= vague_count * 1.5
        
        # ============================================================
        # FEATURE 7: Sentence complexity and variety
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences > 0:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sentence_lengths) / len(sentence_lengths)
            
            # Good range: 10-25 words per sentence
            if 10 <= avg_sent_len <= 25:
                score += 3.0
            elif 7 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
                score += 1.0
            else:
                score -= 1.0
            
            # Sentence length variety (std dev)
            if len(sentence_lengths) > 1:
                mean_sl = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((x - mean_sl) ** 2 for x in sentence_lengths) / len(sentence_lengths)
                std_sl = math.sqrt(variance)
                # Some variety is good
                if 3 <= std_sl <= 12:
                    score += 2.0
        
        # ============================================================
        # FEATURE 8: Vocabulary richness (type-token ratio)
        # ============================================================
        
        clean_words = re.findall(r'[a-z]+', resp_lower)
        if len(clean_words) > 10:
            unique_words = len(set(clean_words))
            # Use root TTR to normalize for length
            root_ttr = unique_words / math.sqrt(len(clean_words))
            # Typical good range: 5-10
            if root_ttr > 6:
                score += 2.0
            elif root_ttr > 5:
                score += 1.0
        
        # ============================================================
        # FEATURE 9: Response completeness and engagement
        # ============================================================
        
        # Check if response seems cut off (ends mid-sentence without punctuation)
        stripped_resp = response.rstrip()
        if stripped_resp and stripped_resp[-1] not in '.!?:"\')]}':
            score -= 2.0  # Likely truncated
        
        # Opening quality - does it address the query directly?
        # Check first sentence relevance by word overlap with query
        query_words = set(re.findall(r'[a-z]+', query_lower))
        query_words -= {'a', 'the', 'is', 'are', 'was', 'were', 'do', 'does',
                       'did', 'can', 'could', 'will', 'would', 'should', 'have',
                       'has', 'had', 'be', 'been', 'being', 'to', 'of', 'in',
                       'for', 'on', 'with', 'at', 'by', 'from', 'an', 'and',
                       'or', 'but', 'not', 'no', 'if', 'how', 'what', 'when',
                       'where', 'who', 'why', 'which', 'that', 'this', 'it',
                       'i', 'you', 'we', 'they', 'my', 'your', 'me', 'im'}
        
        if query_words and sentences:
            first_sent_words = set(re.findall(r'[a-z]+', sentences[0].lower()))
            overlap = len(query_words & first_sent_words)
            overlap_ratio = overlap / max(len(query_words), 1)
            score += overlap_ratio * 5.0
        
        # ============================================================
        # FEATURE 10: Causal/logical connectors (reasoning quality)
        # ============================================================
        
        logical_connectors = [
            'because', 'therefore', 'however', 'although', 'moreover',
            'furthermore', 'consequently', 'as a result', 'in addition',
            'on the other hand', 'nevertheless', 'in contrast',
            'similarly', 'meanwhile', 'thus', 'hence', 'accordingly',
            'while', 'whereas', 'since', 'given that', 'due to'
        ]
        
        connector_count = sum(1 for c in logical_connectors if c in resp_lower)
        score += min(connector_count * 1.0, 5.0)
        
        # ============================================================
        # FEATURE 11: Exclamation mark density (informal/sensational)
        # ============================================================
        
        exclamation_count = response.count('!')
        excl_ratio = exclamation_count / max(num_sentences, 1)
        
        # A few exclamations are fine (engagement), too many is sensational
        if excl_ratio > 0.5:
            score -= 3.0
        elif excl_ratio > 0.3:
            score -= 1.0
        
        # ============================================================
        # FEATURE 12: Response length adequacy
        # ============================================================
        
        # Very short responses are usually less informative
        if word_count < 20:
            score -= 5.0
        elif word_count < 50:
            score -= 2.0
        elif word_count >= 80:
            score += 2.0
        elif word_count >= 120:
            score += 3.0
        
        # ============================================================
        # FEATURE 13: First-person hedging vs authoritative tone
        # ============================================================
        
        # "I think", "I believe" - shows opinion vs fact distinction
        opinion_markers = re.findall(r'\bi (?:think|believe|feel|suppose|imagine)\b', resp_lower)
        # A small amount is good (honest), too much undermines credibility
        if 0 < len(opinion_markers) <= 2:
            score += 1.0
        elif len(opinion_markers) > 3:
            score -= 1.0
        
        # ============================================================
        # FEATURE 14: Colon usage (explanatory depth)
        # ============================================================
        
        colon_count = response.count(':')
        if colon_count >= 2:
            score += 1.5
        
        # ============================================================
        # FEATURE 15: Parenthetical clarifications
        # ============================================================
        
        parentheticals = re.findall(r'\([^)]{3,}\)', response)
        if 1 <= len(parentheticals) <= 5:
            score += 2.0  # Shows additional context/clarification
        
        # Clamp score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0