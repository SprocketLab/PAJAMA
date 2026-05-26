def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant focuses on:
    - Citation/reference patterns (specific names, dates, numbers as credibility signals)
    - Hallucination red-flags (overly precise unsourced stats, absolute claims)
    - Appropriate hedging vs overconfidence
    - Sensationalism/conspiracy language detection
    - Structured reasoning and specificity
    - Empathy and acknowledgment patterns (contextual quality)
    - Action-oriented specificity (concrete actionable advice)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        word_count = len(words)
        
        if word_count < 3:
            return 0.5
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 5.0  # Start at midpoint
        
        # ============================================================
        # 1. ACKNOWLEDGMENT & EMPATHY DETECTION
        #    Responses that acknowledge the user's situation tend to be higher quality
        # ============================================================
        empathy_phrases = [
            r"\bi can see\b", r"\bi can hear\b", r"\bi understand\b",
            r"\bthat'?s? (completely |totally |absolutely )?understandable\b",
            r"\bit'?s? (perfectly |completely |absolutely )?(ok|okay|fine|normal|natural)\b",
            r"\bi'?m (genuinely |truly |really )?sorry\b",
            r"\bcompletely understandable\b", r"\btotally understandable\b",
            r"\bgive yourself\b", r"\bpermission to\b",
            r"\bit'?s? natural to\b", r"\bit'?s? okay to\b",
        ]
        empathy_count = sum(1 for pat in empathy_phrases if re.search(pat, response_lower))
        empathy_score = min(empathy_count * 0.35, 1.4)
        score += empathy_score
        
        # ============================================================
        # 2. STRUCTURED SPECIFICITY
        #    Concrete, actionable advice with structure signals quality
        # ============================================================
        # Numbered steps or enumerated advice
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_structure = numbered_items >= 2
        
        # Colon-based explanations (e.g., "Backstory: knowing their past...")
        colon_explanations = len(re.findall(r'\w+:\s+\w', response))
        
        # Action verbs that indicate concrete advice
        action_verbs = [
            r"\btry to\b", r"\bstart with\b", r"\bbegin by\b", r"\bconsider\b",
            r"\bremember to\b", r"\bmake sure\b", r"\bdon'?t forget\b",
            r"\bexplore\b", r"\bdemonstrate\b", r"\bpractice\b",
            r"\btackle\b", r"\bbreak (it |your |the )?down\b",
            r"\bfocus on\b", r"\bkeep in mind\b", r"\btake a moment\b",
        ]
        action_count = sum(1 for pat in action_verbs if re.search(pat, response_lower))
        
        structure_score = 0.0
        if has_structure:
            structure_score += 0.5
        structure_score += min(colon_explanations * 0.1, 0.3)
        structure_score += min(action_count * 0.15, 0.6)
        score += structure_score
        
        # ============================================================
        # 3. DISMISSIVE / LOW-EFFORT LANGUAGE DETECTION (penalize)
        # ============================================================
        dismissive_phrases = [
            r"\bjust (get over|deal with|move on|keep|handle)\b",
            r"\byou should be able to\b",
            r"\bmaybe you'?re? just not\b",
            r"\bthat'?s? a bummer\b",
            r"\bit'?s? just a\b",
            r"\bget yourself together\b",
            r"\bget rid of\b.*\bnegative\b",
            r"\byou need to get\b",
            r"\bjust remember\b",
            r"\bkeep trying\b",
            r"\byou'?ll get there\b",
        ]
        dismissive_count = sum(1 for pat in dismissive_phrases if re.search(pat, response_lower))
        score -= dismissive_count * 0.4
        
        # ============================================================
        # 4. ABSOLUTE / OVERCONFIDENT CLAIMS (hallucination red flags)
        # ============================================================
        absolute_phrases = [
            r"\balways\b", r"\bnever\b", r"\beveryone knows\b",
            r"\bit'?s? (a )?fact that\b", r"\bundeniably\b",
            r"\bwithout (a )?doubt\b", r"\b100\s*%\b", r"\bguaranteed\b",
            r"\bno one (can|will|has)\b", r"\bimpossible\b",
            r"\bobviously\b", r"\bclearly\b",
        ]
        absolute_count = sum(1 for pat in absolute_phrases if re.search(pat, response_lower))
        # Mild penalty - some absolutes are fine, many are red flags
        score -= max(0, (absolute_count - 1)) * 0.2
        
        # ============================================================
        # 5. APPROPRIATE HEDGING (positive signal for uncertain claims)
        # ============================================================
        hedging_phrases = [
            r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\bpossibly\b",
            r"\bgenerally\b", r"\btypically\b", r"\busually\b",
            r"\bit seems\b", r"\bit appears\b", r"\btend to\b",
            r"\bin many cases\b", r"\boften\b", r"\bsome\b",
            r"\bcan be\b", r"\bmay\b",
        ]
        hedge_count = sum(1 for pat in hedging_phrases if re.search(pat, response_lower))
        # Moderate hedging is good; excessive hedging reduces confidence
        if hedge_count <= 5:
            score += hedge_count * 0.08
        else:
            score += 0.4 - (hedge_count - 5) * 0.05
        
        # ============================================================
        # 6. SENSATIONALISM / CONSPIRACY DETECTION (penalize)
        # ============================================================
        sensational_words = [
            r"\bshocking\b", r"\bunbelievable\b", r"\bmind-?blowing\b",
            r"\bthey don'?t want you to know\b", r"\bhidden truth\b",
            r"\bwake up\b", r"\bsheeple\b", r"\bconspiracy\b",
            r"\bcover-?up\b", r"\bsecret(ly)?\b", r"\bexposed\b",
            r"\bbombshell\b", r"\binsane\b", r"\bcrazy\b",
        ]
        sensational_count = sum(1 for pat in sensational_words if re.search(pat, response_lower))
        score -= sensational_count * 0.5
        
        # ============================================================
        # 7. RESPONSE RELEVANCE via SEMANTIC OVERLAP
        #    Using bigram and trigram overlap, not just unigram
        # ============================================================
        def get_ngrams(text, n):
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            # Filter stopwords for content ngrams
            stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'its', 'this', 'that', 'and', 'or', 'but', 'if', 'not',
                     'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
                     'us', 'them', 'my', 'your', 'his', 'our', 'their'}
            content_tokens = [t for t in tokens if t not in stops and len(t) > 2]
            if len(content_tokens) < n:
                return set()
            return set(tuple(content_tokens[i:i+n]) for i in range(len(content_tokens) - n + 1))
        
        # Content word overlap
        query_content = get_ngrams(query, 1)
        resp_content = get_ngrams(response, 1)
        if query_content:
            unigram_overlap = len(query_content & resp_content) / len(query_content)
        else:
            unigram_overlap = 0.5
        
        # Bigram overlap for deeper relevance
        query_bigrams = get_ngrams(query, 2)
        resp_bigrams = get_ngrams(response, 2)
        if query_bigrams:
            bigram_overlap = len(query_bigrams & resp_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        relevance_score = unigram_overlap * 0.6 + bigram_overlap * 0.8
        score += min(relevance_score, 1.2)
        
        # ============================================================
        # 8. RESPONSE COMPLETENESS & DEPTH
        # ============================================================
        # Average sentence length (too short = shallow, too long = rambling)
        avg_sent_len = word_count / num_sentences
        if 10 <= avg_sent_len <= 25:
            depth_score = 0.4
        elif 8 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
            depth_score = 0.2
        else:
            depth_score = -0.2
        
        # Sufficient length for substantive response
        if word_count >= 50:
            depth_score += 0.3
        elif word_count >= 30:
            depth_score += 0.15
        
        # Multiple distinct points/ideas (measured by sentence diversity)
        if num_sentences >= 4:
            depth_score += 0.3
        elif num_sentences >= 2:
            depth_score += 0.15
        
        score += depth_score
        
        # ============================================================
        # 9. CONVERSATIONAL APPROPRIATENESS
        #    Does the response address the user directly and constructively?
        # ============================================================
        # Direct address
        direct_address = len(re.findall(r'\byou(?:r|\'re|\'ve|\'ll)?\b', response_lower))
        if 2 <= direct_address <= 15:
            score += 0.2
        
        # Questions back to user (shows engagement, clarification)
        questions = len(re.findall(r'\?', response))
        if 1 <= questions <= 3:
            score += 0.15
        
        # ============================================================
        # 10. NEGATIVE CAPABILITY SIGNALS
        #     Responses that admit limitations or suggest seeking help
        # ============================================================
        capability_phrases = [
            r"\bwithout (further|more) (details|information|context)\b",
            r"\bcould you (provide|give|share|clarify)\b",
            r"\bcan you (tell|share|provide|give)\b",
            r"\bdon'?t hesitate to\b", r"\bfeel free to\b",
            r"\bseek(ing)? (professional |expert )?help\b",
            r"\bthere'?s? no (harm|shame) in\b",
            r"\bask for help\b",
        ]
        capability_count = sum(1 for pat in capability_phrases if re.search(pat, response_lower))
        score += min(capability_count * 0.25, 0.6)
        
        # ============================================================
        # 11. TONE CONSISTENCY
        #     Detect jarring tone shifts or inappropriate casualness for serious topics
        # ============================================================
        # Detect if query is about a serious/emotional topic
        serious_indicators = [
            r"\bstress\b", r"\bfrustrat\w+\b", r"\bsad\b", r"\blonely\b",
            r"\bgriev\w+\b", r"\bbreakup\b", r"\bbreak-?up\b", r"\bpassed away\b",
            r"\bdeadline\b", r"\bstruggl\w+\b", r"\bdevast\w+\b", r"\bheartbroken\b",
            r"\bdespair\b", r"\bexhaust\w+\b", r"\bangry\b", r"\bupset\b",
        ]
        is_serious = sum(1 for pat in serious_indicators if re.search(pat, query_lower)) >= 1
        
        if is_serious:
            # Penalize overly casual/dismissive tone in serious contexts
            casual_in_serious = [
                r"\bbummer\b", r"\bno biggie\b", r"\bchill\b",
                r"\bwhatever\b", r"\bjust a\b", r"\bget over it\b",
            ]
            casual_count = sum(1 for pat in casual_in_serious if re.search(pat, response_lower))
            score -= casual_count * 0.5
        
        # ============================================================
        # 12. FACTUAL SPECIFICITY INDICATORS
        #     Specific details (without being unsourced absolute claims)
        # ============================================================
        # Specific quantities/measurements (shows precision)
        specific_numbers = len(re.findall(r'\b\d+\.?\d*\s*(?:pounds?|ounces?|cups?|minutes?|hours?|degrees?|percent|%|ml|kg|lb|oz)\b', response_lower))
        score += min(specific_numbers * 0.1, 0.3)
        
        # Named entities / proper nouns (capitalized words not at sentence start)
        proper_nouns = len(re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response))
        score += min(proper_nouns * 0.05, 0.2)
        
        # ============================================================
        # 13. NEGATIVE PATTERN: VAGUE FILLER
        # ============================================================
        filler_phrases = [
            r"\bkind of\b", r"\bsort of\b", r"\blike,?\s", r"\byou know\b",
            r"\bwhere to start\b", r"\bhmm\b", r"\bwell,\b",
            r"\bthat'?s? right\b",
        ]
        filler_count = sum(1 for pat in filler_phrases if re.search(pat, response_lower))
        score -= filler_count * 0.15
        
        # ============================================================
        # 14. CONTRADICTION DETECTION (simple)
        #     Responses that say opposite things are unreliable
        # ============================================================
        has_but_patterns = len(re.findall(r'\bbut\b.*\bbut\b', response_lower))
        if has_but_patterns > 2:
            score -= 0.3
        
        # Negation of own statements
        negation_pairs = [
            (r"\bcan\b", r"\bcan'?t\b"), (r"\bwill\b", r"\bwon'?t\b"),
            (r"\bshould\b", r"\bshouldn'?t\b"),
        ]
        contradiction_count = sum(
            1 for pos, neg in negation_pairs
            if re.search(pos, response_lower) and re.search(neg, response_lower)
        )
        # Mild penalty - some contradiction is natural in nuanced discussion
        if contradiction_count > 1:
            score -= 0.2
        
        # ============================================================
        # 15. PROBABILISTIC / UNCERTAIN LANGUAGE (for factual claims)
        #     Appropriate uncertainty markers
        # ============================================================
        uncertainty_markers = [
            r"\bprobably\b", r"\blikely\b", r"\bunlikely\b",
            r"\bit depends\b", r"\bit varies\b", r"\bin general\b",
            r"\bfor the most part\b",
        ]
        uncertainty_count = sum(1 for pat in uncertainty_markers if re.search(pat, response_lower))
        score += min(uncertainty_count * 0.1, 0.3)
        
        # ============================================================
        # 16. "MIGHT NOT" / INABILITY PATTERNS
        #     Responses expressing inability/limitation of the system itself
        # ============================================================
        inability_phrases = [
            r"\bmight not be able to\b", r"\bprobably won'?t\b",
            r"\bmay not have the ability\b", r"\bit might not\b",
            r"\bit probably\b.*\bnot\b",
        ]
        inability_count = sum(1 for pat in inability_phrases if re.search(pat, response_lower))
        # In the context of describing solutions, expressing inability is negative
        if inability_count >= 2:
            score -= 0.4
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Clamp to 1-5 range to match the scoring examples
        score = max(1.0, min(5.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 3.0