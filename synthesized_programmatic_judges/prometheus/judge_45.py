def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, signal-to-noise ratio,
    and structural coherence metrics. This variant focuses on:
    - Information density (unique content words per total words ratio)
    - Filler/weasel word penalization
    - Sentence-level clarity (avoiding run-ons and fragments)
    - Directness scoring (how quickly the response addresses the query)
    - Repetition detection via n-gram analysis
    - Coherence via topic word consistency
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # =====================
        # 1. FILLER / WEASEL WORD PENALTY
        # =====================
        filler_phrases = [
            'kind of', 'sort of', 'you know', 'i mean', 'like', 'basically',
            'actually', 'literally', 'honestly', 'frankly', 'really', 'very',
            'just', 'quite', 'pretty much', 'more or less', 'in a way',
            'to be honest', 'at the end of the day', 'when all is said and done',
            'it goes without saying', 'needless to say', 'as a matter of fact',
            'the thing is', 'the fact of the matter', 'in my opinion',
            'i think that', 'it seems like', 'it would appear',
            'hmm', 'well', 'so yeah', 'right', 'anyway', 'anyhow',
            'nifty', 'stuff', 'things get wild', 'where to start'
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for phrase in filler_phrases:
            filler_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', response_lower))
        
        filler_ratio = filler_count / total_words
        filler_score = max(0, 1.0 - filler_ratio * 8)  # Penalize heavily
        
        # =====================
        # 2. HEDGING LANGUAGE DENSITY (different from variant 3 which just counts hedging)
        # Here we compute positional hedging - hedging at the START is worse
        # =====================
        hedge_words = [
            'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably',
            'somewhat', 'apparently', 'supposedly', 'arguably'
        ]
        
        # Check if first sentence contains hedging
        first_sentence_words = re.findall(r'[a-zA-Z]+', sentences[0].lower()) if sentences else []
        first_sent_hedge = sum(1 for w in first_sentence_words if w in hedge_words)
        hedge_opening_penalty = min(first_sent_hedge * 0.15, 0.4)
        
        # Overall hedge density
        total_hedge = sum(1 for w in words if w in hedge_words)
        hedge_density = total_hedge / total_words
        hedge_score = max(0, 1.0 - hedge_density * 12 - hedge_opening_penalty)
        
        # =====================
        # 3. N-GRAM REPETITION (trigram and 4-gram overlap)
        # =====================
        def get_ngrams(word_list, n):
            return [tuple(word_list[i:i+n]) for i in range(len(word_list) - n + 1)]
        
        trigrams = get_ngrams(words, 3)
        fourgrams = get_ngrams(words, 4)
        
        if len(trigrams) > 1:
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_repetition = repeated_trigrams / len(trigrams)
        else:
            trigram_repetition = 0
        
        if len(fourgrams) > 1:
            fourgram_counts = Counter(fourgrams)
            repeated_fourgrams = sum(c - 1 for c in fourgram_counts.values() if c > 1)
            fourgram_repetition = repeated_fourgrams / len(fourgrams)
        else:
            fourgram_repetition = 0
        
        repetition_penalty = trigram_repetition * 0.4 + fourgram_repetition * 0.6
        repetition_score = max(0, 1.0 - repetition_penalty * 5)
        
        # =====================
        # 4. INFORMATION DENSITY via content word ratio
        # =====================
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'should', 'may', 'might', 'must', 'can', 'could', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
            'don', 'now', 'and', 'but', 'or', 'if', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'up', 'about', 'also', 'get', 'got'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        
        # Unique content words ratio (type-token for content words)
        unique_content = len(set(content_words))
        content_ttr = unique_content / len(content_words) if content_words else 0
        
        # Combined information density
        info_density_score = (content_ratio * 0.5 + content_ttr * 0.5)
        # Normalize to 0-1 range (typical content_ratio ~0.3-0.6, ttr ~0.4-0.9)
        info_density_score = min(1.0, info_density_score / 0.6)
        
        # =====================
        # 5. SENTENCE QUALITY - variance in sentence length (too uniform = robotic, too varied = chaotic)
        # Also check for overly long sentences and fragments
        # =====================
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            sent_word_counts.append(len(s_words))
        
        avg_sent_len = sum(sent_word_counts) / num_sentences if num_sentences > 0 else 0
        
        # Ideal average sentence length: 12-22 words
        if 12 <= avg_sent_len <= 22:
            sent_len_score = 1.0
        elif avg_sent_len < 5:
            sent_len_score = 0.3
        elif avg_sent_len < 12:
            sent_len_score = 0.5 + 0.5 * (avg_sent_len - 5) / 7
        elif avg_sent_len <= 30:
            sent_len_score = 1.0 - 0.5 * (avg_sent_len - 22) / 8
        else:
            sent_len_score = 0.3
        
        # Check for very long sentences (run-ons)
        long_sent_count = sum(1 for c in sent_word_counts if c > 35)
        long_sent_penalty = long_sent_count / num_sentences * 0.3
        
        # Check for fragments (very short sentences, < 4 words, excluding headers)
        fragment_count = sum(1 for c in sent_word_counts if 0 < c < 3)
        fragment_penalty = fragment_count / num_sentences * 0.2
        
        sentence_quality_score = max(0, sent_len_score - long_sent_penalty - fragment_penalty)
        
        # =====================
        # 6. DIRECTNESS - Does the response address the query quickly?
        # Check if first sentence contains query-relevant words
        # =====================
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - stop_words
        query_content = {w for w in query_words if len(w) > 2}
        
        if query_content and first_sentence_words:
            first_sent_content = set(first_sentence_words) - stop_words
            query_overlap_first = len(first_sent_content & query_content) / max(len(query_content), 1)
        else:
            query_overlap_first = 0.5  # neutral if no query
        
        # Also check: does the response start with dismissive/vague openings?
        dismissive_openings = [
            r'^(well|so|hmm|uh|um|okay so|ok so)',
            r'^(that\'s a)',
            r'^(where to start)',
        ]
        opening_penalty = 0
        first_50 = response_lower[:50]
        for pattern in dismissive_openings:
            if re.search(pattern, first_50):
                opening_penalty += 0.1
        
        directness_score = min(1.0, query_overlap_first * 1.5) - opening_penalty
        directness_score = max(0, directness_score)
        
        # =====================
        # 7. STRUCTURAL ORGANIZATION
        # Check for numbered lists, clear structure, paragraph breaks
        # =====================
        has_numbered_list = bool(re.search(r'\d+[\.\)]\s', response))
        has_paragraph_breaks = response.count('\n\n') >= 1
        has_colon_structure = bool(re.search(r':\s', response))
        
        structure_score = 0.5  # baseline
        if has_numbered_list:
            structure_score += 0.25
        if has_paragraph_breaks:
            structure_score += 0.15
        if has_colon_structure:
            structure_score += 0.1
        structure_score = min(1.0, structure_score)
        
        # =====================
        # 8. EMPATHY / ENGAGEMENT DETECTION (for emotional queries)
        # =====================
        emotional_query_words = {'feeling', 'feel', 'frustrated', 'sad', 'stress', 'stressed',
                                  'lonely', 'loneliness', 'heartbroken', 'devastated', 'upset',
                                  'angry', 'anxious', 'worried', 'struggling', 'difficult',
                                  'breakup', 'passed', 'died', 'death'}
        
        query_is_emotional = bool(query_content & emotional_query_words)
        
        empathy_phrases = [
            r"i('m| am) sorry", r"i understand", r"that('s| is) (completely |totally |absolutely )?understandable",
            r"it('s| is) (perfectly |completely )?(okay|ok|fine|natural|normal)",
            r"i can (see|hear|imagine|understand)", r"your feelings",
            r"give yourself", r"take a moment", r"it('s| is) important to"
        ]
        
        empathy_count = 0
        for pattern in empathy_phrases:
            if re.search(pattern, response_lower):
                empathy_count += 1
        
        if query_is_emotional:
            empathy_score = min(1.0, empathy_count * 0.25 + 0.2)
        else:
            empathy_score = 0.6  # neutral for non-emotional queries
        
        # =====================
        # 9. NEGATIVE TONE / DISMISSIVENESS DETECTION
        # =====================
        dismissive_phrases = [
            r"you should be able to",
            r"just (keep|try|do|get|read|remember)",
            r"you need to get",
            r"it('s| is) (just|only) a",
            r"maybe you('re| are) (just|not)",
            r"get yourself together",
            r"don('t| not) let it",
            r"that('s| is) a bummer",
            r"nothing wrong with",
        ]
        
        dismissive_count = 0
        for pattern in dismissive_phrases:
            if re.search(pattern, response_lower):
                dismissive_count += 1
        
        dismissive_penalty = min(dismissive_count * 0.12, 0.5)
        tone_score = max(0, 1.0 - dismissive_penalty)
        
        # =====================
        # 10. ACTIONABILITY - Does response provide concrete, actionable content?
        # =====================
        action_indicators = [
            r'\b(first|second|third|next|then|finally|start by|begin with)\b',
            r'\b(try|consider|ensure|make sure|remember to|don\'t forget)\b',
            r'\b(step \d|tip \d)\b',
            r'\b(for example|for instance|such as|e\.g\.)\b',
            r'\b(specifically|in particular|namely)\b',
        ]
        
        action_count = 0
        for pattern in action_indicators:
            action_count += len(re.findall(pattern, response_lower))
        
        actionability_score = min(1.0, 0.3 + action_count * 0.1)
        
        # =====================
        # 11. NEGATIVE CAPABILITY LANGUAGE (saying what CAN'T be done)
        # =====================
        negative_capability = [
            r"(can't|cannot|won't|unable to|might not|may not|probably won't)",
            r"(it (might|may|could) not)",
            r"(not (be able|have the ability))",
        ]
        
        neg_cap_count = 0
        for pattern in negative_capability:
            neg_cap_count += len(re.findall(pattern, response_lower))
        
        # Excessive negative capability language reduces clarity
        neg_cap_penalty = min(neg_cap_count * 0.08, 0.4)
        positivity_score = max(0, 1.0 - neg_cap_penalty)
        
        # =====================
        # COMPOSITE SCORING
        # =====================
        # Weight the components
        weights = {
            'filler': 0.12,
            'hedge': 0.08,
            'repetition': 0.10,
            'info_density': 0.12,
            'sentence_quality': 0.10,
            'directness': 0.10,
            'structure': 0.08,
            'empathy': 0.08,
            'tone': 0.08,
            'actionability': 0.08,
            'positivity': 0.06,
        }
        
        scores = {
            'filler': filler_score,
            'hedge': hedge_score,
            'repetition': repetition_score,
            'info_density': info_density_score,
            'sentence_quality': sentence_quality_score,
            'directness': directness_score,
            'structure': structure_score,
            'empathy': empathy_score,
            'tone': tone_score,
            'actionability': actionability_score,
            'positivity': positivity_score,
        }
        
        composite = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 1-5 range
        final_score = 1.0 + composite * 4.0
        
        # Apply length sanity check: very short responses (< 50 chars) get capped
        if len(response) < 50:
            final_score = min(final_score, 2.0)
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5