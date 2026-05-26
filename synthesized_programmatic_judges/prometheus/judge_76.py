def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    
    This variant analyzes responses at the sentence level, computing an "information density"
    score per sentence, then aggregates. It focuses on:
    1. Sentence-level specificity scoring (ratio of informative vs filler sentences)
    2. Named entity density via capitalized multi-word detection
    3. Numeric/quantitative expression detection
    4. Actionable language detection (imperative verbs, step-by-step structure)
    5. Filler/vagueness penalty using trigram and phrase patterns
    6. Lexical sophistication (avg word length, rare word ratio)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        words = response.split()
        word_count = len(words)
        if word_count < 3:
            return 0.5
        
        # ============================================================
        # 1. SENTENCE-LEVEL SPECIFICITY SCORING
        # ============================================================
        # For each sentence, compute how "specific" it is
        
        # Vague/filler trigrams and phrases (scored per sentence)
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are many\b', r'\bthere are various\b',
            r'\bin general\b', r'\bgenerally speaking\b', r'\bit\'s important to\b',
            r'\byou should\b', r'\byou could\b', r'\byou might\b',
            r'\bkeep in mind\b', r'\bas you know\b', r'\bas we know\b',
            r'\bneedless to say\b', r'\bit goes without saying\b',
            r'\bat the end of the day\b', r'\ball in all\b',
            r'\bfor the most part\b', r'\bmore or less\b',
            r'\bkind of\b', r'\bsort of\b', r'\bpretty much\b',
            r'\bstuff like that\b', r'\bthings like that\b',
            r'\band so on\b', r'\betc\b', r'\band whatnot\b',
            r'\bmaybe\b', r'\bperhaps\b', r'\bprobably\b',
            r'\bi think\b', r'\bi guess\b', r'\bi suppose\b',
            r'\bjust\b.*\bjust\b',  # repeated "just"
        ]
        
        # Specific/concrete indicators per sentence
        specific_indicators = [
            r'\b\d+[\.\d]*\s*(%|percent|pounds?|kg|miles?|km|hours?|minutes?|seconds?|dollars?|euros?|ounces?|cups?|tablespoons?|teaspoons?|degrees?|watts?|volts?|GB|MB|TB)\b',
            r'\b\d+[\.\d]*\b',  # any number
            r'\b(first|second|third|fourth|fifth|step \d|phase \d)\b',
            r'\b(specifically|precisely|exactly|concretely|notably)\b',
            r'\b(for example|for instance|such as|e\.g\.|i\.e\.)\b',
            r'\b(because|since|due to|as a result|therefore|consequently|thus)\b',
            r'\b(according to|research shows|studies show|data shows|evidence suggests)\b',
        ]
        
        sentence_scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            sent_words = sent.split()
            sent_word_count = len(sent_words)
            if sent_word_count < 2:
                continue
            
            score = 0.0
            
            # Check for specific indicators
            specific_count = 0
            for pattern in specific_indicators:
                matches = re.findall(pattern, sent_lower)
                specific_count += len(matches)
            
            # Check for vague phrases
            vague_count = 0
            for pattern in vague_phrases:
                if re.search(pattern, sent_lower):
                    vague_count += 1
            
            # Average word length in sentence (longer words tend to be more specific)
            avg_wl = sum(len(w) for w in sent_words) / max(sent_word_count, 1)
            
            # Capitalized words (potential named entities, excluding sentence start)
            cap_words = sum(1 for w in sent_words[1:] if w and w[0].isupper() and len(w) > 1 and not w.isupper())
            
            # Sentence specificity score
            score = (specific_count * 2.0 + cap_words * 1.5 + max(0, avg_wl - 4.0) * 0.5) - (vague_count * 2.0)
            
            # Normalize by sentence length (density)
            density = score / max(sent_word_count, 1) * 10
            sentence_scores.append(density)
        
        avg_sentence_specificity = sum(sentence_scores) / max(len(sentence_scores), 1) if sentence_scores else 0
        
        # ============================================================
        # 2. NAMED ENTITY / PROPER NOUN DENSITY
        # ============================================================
        # Detect capitalized sequences (excluding sentence starts)
        # Use a different approach: look for capitalized words not at sentence boundaries
        
        all_words = response.split()
        proper_noun_count = 0
        for i, w in enumerate(all_words):
            if i == 0:
                continue
            # Check if previous char was not a sentence ender
            clean_w = re.sub(r'[^a-zA-Z]', '', w)
            if not clean_w:
                continue
            if clean_w[0].isupper() and len(clean_w) > 1 and not clean_w.isupper():
                # Check if preceded by sentence boundary
                prev_word = all_words[i-1] if i > 0 else ''
                if not prev_word.endswith(('.', '!', '?', ':')):
                    proper_noun_count += 1
        
        entity_density = proper_noun_count / max(word_count, 1) * 100
        
        # ============================================================
        # 3. QUANTITATIVE EXPRESSION DETECTION (more sophisticated)
        # ============================================================
        # Look for numbers with context (not just bare numbers)
        
        quant_patterns = [
            r'\b\d+\.?\d*\s*(percent|%|pounds?|lbs?|kg|kilograms?|grams?|oz|ounces?|cups?|tbsp|tsp|tablespoons?|teaspoons?|ml|liters?|gallons?)',
            r'\b\d+\.?\d*\s*(miles?|km|kilometers?|meters?|feet|inches?|cm|mm)',
            r'\b\d+\.?\d*\s*(hours?|minutes?|seconds?|days?|weeks?|months?|years?)',
            r'\b\d+\.?\d*\s*(dollars?|\$|euros?|€|£)',
            r'\b\d+\.?\d*\s*(MB|GB|TB|KB|MHz|GHz|watts?|volts?|amps?)',
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand|million)\s+\w+',
            r'\b\d+\s*[-–]\s*\d+\b',  # ranges like 5-10
        ]
        
        quant_count = 0
        resp_lower = response.lower()
        for pattern in quant_patterns:
            quant_count += len(re.findall(pattern, resp_lower))
        
        # Also count bare numbers but with less weight
        bare_numbers = re.findall(r'\b\d+\.?\d*\b', response)
        bare_number_count = len(bare_numbers)
        
        quant_score = (quant_count * 3.0 + bare_number_count * 0.8) / max(word_count, 1) * 100
        
        # ============================================================
        # 4. ACTIONABLE LANGUAGE / IMPERATIVE DETECTION
        # ============================================================
        
        action_verbs = [
            r'\b(start|begin|open|create|write|type|click|select|choose|pick|grab|take|place|put|set|add|remove|delete|insert|apply|use|try|check|verify|ensure|confirm|test|run|execute|install|download|upload|save|load|configure|adjust|modify|change|update|replace|combine|mix|stir|heat|cook|bake|pour|cut|slice|chop)\b'
        ]
        
        action_count = 0
        for pattern in action_verbs:
            action_count += len(re.findall(pattern, resp_lower))
        
        action_density = action_count / max(word_count, 1) * 100
        
        # ============================================================
        # 5. STRUCTURAL ORGANIZATION SIGNALS
        # ============================================================
        
        # Numbered items (1. 2. 3. or 1) 2) 3))
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        
        # Colon-based headers/labels
        colon_labels = len(re.findall(r'\b[A-Z][a-zA-Z\s]{2,25}:', response))
        
        # Transition/sequencing words
        sequencing = len(re.findall(r'\b(first|firstly|second|secondly|third|thirdly|next|then|finally|lastly|additionally|furthermore|moreover|also|in addition)\b', resp_lower))
        
        structure_score = (numbered_items * 2.5 + colon_labels * 2.0 + sequencing * 1.0)
        
        # ============================================================
        # 6. VAGUENESS / HEDGING PENALTY (global level)
        # ============================================================
        
        hedge_words = [
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could',
            'somewhat', 'somehow', 'something', 'somewhere', 'sometimes',
            'often', 'usually', 'generally', 'typically', 'normally',
            'basically', 'essentially', 'virtually', 'practically',
            'stuff', 'things', 'whatever', 'whatnot', 'etc'
        ]
        
        resp_words_lower = [w.lower().strip('.,!?;:()[]"\'') for w in words]
        hedge_count = sum(1 for w in resp_words_lower if w in hedge_words)
        hedge_ratio = hedge_count / max(word_count, 1)
        
        # Dismissive / low-effort phrases
        dismissive_patterns = [
            r'\bjust\s+(do|try|get|go|keep|make|be)\b',
            r'\byou\'ll be fine\b', r'\bdon\'t worry\b',
            r'\bit\'s (just|only|simply)\b',
            r'\bthat\'s (just|about) (it|all)\b',
            r'\bwhatever works\b', r'\bno big deal\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, resp_lower))
        
        vagueness_penalty = hedge_ratio * 15 + dismissive_count * 1.5
        
        # ============================================================
        # 7. LEXICAL SOPHISTICATION
        # ============================================================
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        # Unique word ratio (type-token ratio)
        unique_words = set(w.lower().strip('.,!?;:()[]"\'') for w in words)
        ttr = len(unique_words) / max(word_count, 1)
        
        # Long words ratio (>= 8 chars, often more specific/technical)
        long_words = sum(1 for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) >= 8)
        long_word_ratio = long_words / max(word_count, 1)
        
        # ============================================================
        # 8. CAUSAL / EXPLANATORY DEPTH
        # ============================================================
        
        causal_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\bdue to\b', r'\bas a result\b',
            r'\btherefore\b', r'\bconsequently\b', r'\bthus\b', r'\bhence\b',
            r'\bthis means\b', r'\bthis leads to\b', r'\bwhich causes\b',
            r'\bwhich results in\b', r'\bthe reason\b', r'\bthis is why\b',
            r'\bin order to\b', r'\bso that\b',
        ]
        
        causal_count = 0
        for pattern in causal_patterns:
            causal_count += len(re.findall(pattern, resp_lower))
        
        causal_score = causal_count / max(len(sentences), 1) * 10
        
        # ============================================================
        # 9. EMPATHY / ENGAGEMENT SIGNALS (relevant for conversational queries)
        # ============================================================
        
        empathy_patterns = [
            r'\bi (understand|hear|see|recognize|acknowledge)\b',
            r'\bit\'s (completely|perfectly|absolutely|totally|entirely) (normal|natural|understandable|okay|fine|valid)\b',
            r'\byour feelings\b', r'\byour experience\b', r'\byour situation\b',
            r'\bi\'m (sorry|here)\b', r'\bthat must be\b',
            r'\bcompletely understandable\b', r'\btotally understandable\b',
        ]
        
        empathy_count = 0
        for pattern in empathy_patterns:
            empathy_count += len(re.findall(pattern, resp_lower))
        
        # ============================================================
        # 10. RESPONSE LENGTH FACTOR (with diminishing returns)
        # ============================================================
        
        # Moderate length is good, but very short is bad
        length_factor = math.log(max(word_count, 1) + 1) / math.log(200)
        length_factor = min(length_factor, 1.2)  # cap
        
        # ============================================================
        # 11. QUERY-RESPONSE RELEVANCE (content word overlap)
        # ============================================================
        
        # Extract content words from query (>= 4 chars)
        query_words = set(w.lower().strip('.,!?;:()[]"\'') for w in query.split() if len(w) >= 4)
        resp_word_set = set(resp_words_lower)
        
        if query_words:
            relevance = len(query_words & resp_word_set) / max(len(query_words), 1)
        else:
            relevance = 0.5
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        # Weighted combination of all features
        score = 0.0
        
        # Base score from sentence-level specificity (range roughly -5 to 5)
        score += avg_sentence_specificity * 3.0
        
        # Entity density bonus (0-5 typically)
        score += min(entity_density * 1.5, 5.0)
        
        # Quantitative expression bonus
        score += min(quant_score * 2.0, 8.0)
        
        # Action density
        score += min(action_density * 0.8, 5.0)
        
        # Structure score
        score += min(structure_score * 0.4, 6.0)
        
        # Causal/explanatory depth
        score += min(causal_score * 1.5, 5.0)
        
        # Lexical sophistication
        score += (avg_word_length - 3.5) * 1.5  # reward longer avg words
        score += long_word_ratio * 10
        score += ttr * 3.0  # vocabulary diversity
        
        # Empathy bonus (small, for conversational contexts)
        score += min(empathy_count * 1.0, 3.0)
        
        # Length factor
        score *= max(length_factor, 0.3)
        
        # Relevance bonus
        score += relevance * 3.0
        
        # Vagueness penalty
        score -= vagueness_penalty
        
        # Dismissive penalty
        score -= dismissive_count * 1.0
        
        # ============================================================
        # NORMALIZE TO 1-5 RANGE
        # ============================================================
        
        # Empirical range adjustment: raw scores typically range from -5 to 25
        # Map to 1-5
        normalized = 1.0 + (score + 5.0) / 30.0 * 4.0
        
        # Clamp
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5