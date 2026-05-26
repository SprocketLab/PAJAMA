def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a substantially different approach:
    - N-gram based analysis for detecting specific factual patterns
    - Discourse coherence scoring via transition/connective analysis
    - Empathy and acknowledgment pattern detection
    - Specificity vs vagueness ratio
    - Response-query semantic alignment via character/word n-gram overlap
    - Structural sophistication scoring
    - Red flag detection for hallucination patterns
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 1.0
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        # Tokenize
        def tokenize(text):
            return re.findall(r'\b[a-z]+(?:\'[a-z]+)?\b', text.lower())
        
        response_tokens = tokenize(response)
        query_tokens = tokenize(query)
        
        if len(response_tokens) < 3:
            return 1.0
        
        response_sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(response_sentences), 1)
        
        score = 0.0
        
        # ============================================================
        # 1. DISCOURSE CONNECTIVES & COHERENCE (unique feature)
        # ============================================================
        # Measure logical flow through discourse markers
        causal_connectives = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'in order to',
            'this means', 'which means', 'leading to', 'resulting in'
        ]
        additive_connectives = [
            'furthermore', 'moreover', 'additionally', 'in addition',
            'also', 'besides', 'what\'s more', 'not only'
        ]
        contrastive_connectives = [
            'however', 'although', 'nevertheless', 'on the other hand',
            'despite', 'whereas', 'while', 'but', 'yet', 'conversely',
            'in contrast', 'instead'
        ]
        temporal_connectives = [
            'first', 'then', 'next', 'finally', 'subsequently',
            'before', 'after', 'meanwhile', 'once', 'when'
        ]
        
        causal_count = sum(1 for c in causal_connectives if c in response_lower)
        additive_count = sum(1 for c in additive_connectives if c in response_lower)
        contrastive_count = sum(1 for c in contrastive_connectives if c in response_lower)
        temporal_count = sum(1 for c in temporal_connectives if c in response_lower)
        
        # Variety of connective types used (0-4)
        connective_variety = sum(1 for c in [causal_count, additive_count, contrastive_count, temporal_count] if c > 0)
        total_connectives = causal_count + additive_count + contrastive_count + temporal_count
        
        # Connective density per sentence
        connective_density = total_connectives / num_sentences
        
        # Score: reward variety and moderate density
        coherence_score = min(connective_variety * 1.5, 6.0) + min(connective_density * 2.0, 4.0)
        score += coherence_score  # max ~10
        
        # ============================================================
        # 2. ACKNOWLEDGMENT & EMPATHY PATTERNS (unique combination)
        # ============================================================
        acknowledgment_phrases = [
            r'\bi (?:can |do )?(?:understand|hear|see|recognize|appreciate)\b',
            r'\bthat\'s (?:completely |totally |perfectly )?(?:understandable|normal|okay|fine|natural|valid)\b',
            r'\bit\'s (?:completely |totally |perfectly )?(?:understandable|normal|okay|fine|natural|valid)\b',
            r'\bi\'m (?:genuinely |truly |really )?sorry\b',
            r'\byour (?:feelings?|concerns?|frustration|experience|situation)\b',
            r'\bcompletely understandable\b',
            r'\bperfectly (?:fine|okay|normal|natural)\b',
            r'\bgive yourself\b',
            r'\bit\'s okay to\b',
            r'\bwe (?:value|appreciate|understand)\b',
        ]
        
        ack_count = 0
        for pattern in acknowledgment_phrases:
            if re.search(pattern, response_lower):
                ack_count += 1
        
        empathy_score = min(ack_count * 2.0, 10.0)
        score += empathy_score  # max ~10
        
        # ============================================================
        # 3. SPECIFICITY ANALYSIS via character trigram diversity
        # ============================================================
        # More specific/detailed responses have richer character trigram distributions
        def char_trigrams(text):
            text = re.sub(r'\s+', ' ', text.lower())
            return [text[i:i+3] for i in range(len(text)-2)]
        
        trigrams = char_trigrams(response)
        if trigrams:
            trigram_counts = Counter(trigrams)
            unique_trigrams = len(trigram_counts)
            total_trigrams = len(trigrams)
            trigram_diversity = unique_trigrams / total_trigrams if total_trigrams > 0 else 0
            
            # Higher diversity in moderate-length responses = more specific content
            # Very short or very long texts naturally have different baselines
            length_factor = min(len(response_tokens) / 50.0, 1.0)
            specificity_score = trigram_diversity * length_factor * 8.0
        else:
            specificity_score = 0.0
        
        score += min(specificity_score, 8.0)  # max ~8
        
        # ============================================================
        # 4. QUERY-RESPONSE ALIGNMENT via word bigram overlap
        # ============================================================
        def word_bigrams(tokens):
            return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        
        query_bigrams = set(word_bigrams(query_tokens))
        response_bigrams = set(word_bigrams(response_tokens))
        
        # Unigram content word overlap (filter stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'they', 'them', 'their', 'about', 'up', 'out',
        }
        
        query_content = set(w for w in query_tokens if w not in stopwords and len(w) > 2)
        response_content = set(w for w in response_tokens if w not in stopwords and len(w) > 2)
        
        if query_content:
            content_overlap = len(query_content & response_content) / len(query_content)
        else:
            content_overlap = 0.5
        
        # Bigram overlap for deeper alignment
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        alignment_score = content_overlap * 6.0 + bigram_overlap * 4.0
        score += min(alignment_score, 10.0)  # max ~10
        
        # ============================================================
        # 5. STRUCTURAL SOPHISTICATION
        # ============================================================
        # Paragraph count
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Numbered/ordered lists (shows structured thinking)
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        
        # Sentence length variance (good writing has varied sentence lengths)
        sent_lengths = [len(tokenize(s)) for s in response_sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len)**2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variance is good (not all same length, not wildly different)
            length_variance_score = min(std_dev / 3.0, 1.0) * 3.0
        else:
            length_variance_score = 0.0
        
        # Multi-paragraph bonus
        paragraph_score = min(num_paragraphs * 1.0, 3.0)
        
        # Numbered list bonus (structured advice)
        list_score = min(numbered_items * 0.8, 3.0)
        
        structure_score = paragraph_score + list_score + length_variance_score
        score += min(structure_score, 8.0)  # max ~8
        
        # ============================================================
        # 6. APPROPRIATE HEDGING & EPISTEMIC MARKERS
        # ============================================================
        appropriate_hedging = [
            'it\'s possible', 'it may', 'it might', 'it could',
            'generally', 'typically', 'usually', 'often',
            'in many cases', 'tends to', 'can be', 'may help',
            'consider', 'you might', 'you could', 'you may',
            'one approach', 'one way', 'perhaps', 'potentially',
            'it seems', 'it appears', 'likely', 'unlikely',
        ]
        
        hedge_count = sum(1 for h in appropriate_hedging if h in response_lower)
        hedging_score = min(hedge_count * 1.2, 5.0)
        score += hedging_score  # max ~5
        
        # ============================================================
        # 7. RED FLAGS / PENALTIES
        # ============================================================
        penalties = 0.0
        
        # Dismissive language
        dismissive_patterns = [
            r'\bjust (?:get over|deal with|move on|forget|stop)\b',
            r'\byou(?:\'re| are) (?:just |probably )?not (?:using|doing)\b',
            r'\bmaybe you(?:\'re| are) just\b',
            r'\bit\'s (?:just|only) a\b',
            r'\bget yourself together\b',
            r'\byou need to get\b',
            r'\byou should be able to\b',
        ]
        for pattern in dismissive_patterns:
            if re.search(pattern, response_lower):
                penalties += 3.0
        
        # Unsupported absolute claims
        absolute_patterns = [
            r'\b(?:always|never|every single|without exception|guaranteed|100%|absolutely certain)\b',
            r'\beveryone knows\b',
            r'\bit is (?:a )?fact that\b',
            r'\bundeniably\b',
            r'\bwithout a doubt\b',
        ]
        abs_count = sum(1 for p in absolute_patterns if re.search(p, response_lower))
        penalties += abs_count * 1.5
        
        # Hallucination red flags: overly precise unsourced statistics
        fake_stats = re.findall(r'\b\d{2,3}(?:\.\d+)?%\b', response)
        if fake_stats:
            penalties += len(fake_stats) * 1.0
        
        # Very short responses (likely low quality)
        if len(response_tokens) < 20:
            penalties += 5.0
        elif len(response_tokens) < 40:
            penalties += 2.0
        
        # Repetitive phrasing (same sentence starter repeated)
        if len(response_sentences) >= 3:
            starters = [s.split()[:2] if len(s.split()) >= 2 else s.split()[:1] for s in response_sentences]
            starter_strings = [' '.join(s).lower() for s in starters]
            starter_counts = Counter(starter_strings)
            max_repeat = max(starter_counts.values())
            if max_repeat >= 3:
                penalties += (max_repeat - 2) * 1.5
        
        # Condescending tone
        condescending = [
            r'\bcarefully read\b',
            r'\byou(?:\'re| are) (?:probably )?(?:just )?not (?:understanding|getting)\b',
            r'\bas i (?:already |just )?(?:said|mentioned|explained)\b',
        ]
        for pattern in condescending:
            if re.search(pattern, response_lower):
                penalties += 2.5
        
        score -= penalties
        
        # ============================================================
        # 8. RESPONSE COMPLETENESS INDICATOR
        # ============================================================
        # Check if response addresses the query's implicit needs
        # by looking for action-oriented language
        action_words = [
            'try', 'start', 'begin', 'consider', 'remember', 'keep',
            'make sure', 'ensure', 'focus', 'practice', 'explore',
            'take', 'use', 'apply', 'implement', 'create', 'develop',
            'here are', 'here\'s', 'follow', 'step'
        ]
        action_count = sum(1 for a in action_words if a in response_lower)
        actionability_score = min(action_count * 0.8, 5.0)
        score += actionability_score  # max ~5
        
        # ============================================================
        # 9. TONE APPROPRIATENESS
        # ============================================================
        # Second-person engagement (addressing the user directly)
        second_person = len(re.findall(r'\byou(?:r|\'re|\'ll|\'ve)?\b', response_lower))
        second_person_density = second_person / num_sentences
        
        # Moderate second-person usage is good (engaging but not preachy)
        if 0.3 <= second_person_density <= 3.0:
            tone_score = 3.0
        elif second_person_density > 3.0:
            tone_score = 1.5  # possibly preachy
        else:
            tone_score = 1.0
        
        # First person plural "we" (collaborative tone)
        we_count = len(re.findall(r'\bwe\b|\blet\'s\b|\bour\b', response_lower))
        if we_count > 0:
            tone_score += min(we_count * 0.5, 2.0)
        
        score += min(tone_score, 5.0)  # max ~5
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Theoretical max is roughly: 10 + 10 + 8 + 10 + 8 + 5 + 5 + 5 = 61
        # Normalize to 1-5 scale
        
        # Clamp score
        score = max(score, 0.0)
        
        # Map to 1-5 range
        # Use a sigmoid-like mapping for better discrimination
        normalized = score / 50.0  # rough normalization to 0-1ish range
        normalized = min(max(normalized, 0.0), 1.0)
        
        # Apply slight curve for better discrimination
        final_score = 1.0 + normalized * 4.0
        
        # Clamp to 1-5
        final_score = max(1.0, min(5.0, round(final_score, 2)))
        
        return final_score
        
    except Exception:
        return 2.5