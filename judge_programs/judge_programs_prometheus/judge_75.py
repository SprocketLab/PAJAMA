def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    Analyzes each sentence for concrete evidence markers and penalizes vague/filler patterns.
    Uses a fundamentally different approach: sentence-by-sentence scoring with 
    information-theoretic density measures.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        # ---- FEATURE 1: Sentence-level specificity scoring ----
        # For each sentence, compute a specificity score based on presence of concrete markers
        
        # Concrete evidence patterns (regex-based)
        number_pattern = re.compile(r'\b\d+[\d,.]*\b')
        percentage_pattern = re.compile(r'\b\d+(\.\d+)?%')
        measurement_pattern = re.compile(r'\b\d+\s*(pounds?|lbs?|kg|grams?|oz|ounces?|miles?|km|feet|inches?|cm|mm|hours?|minutes?|seconds?|days?|weeks?|months?|years?|degrees?|celsius|fahrenheit)\b', re.IGNORECASE)
        proper_noun_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b')
        quoted_pattern = re.compile(r'["\'].*?["\']')
        parenthetical_pattern = re.compile(r'\(.*?\)')
        colon_definition_pattern = re.compile(r':\s+\w')
        technical_term_pattern = re.compile(r'\b[a-z]+(?:tion|sion|ment|ness|ity|ance|ence|ism|ist|ous|ive|ful|less|able|ible|ical|ology|ography|ometry)\b', re.IGNORECASE)
        
        # Action/instruction verbs that indicate concrete advice
        action_verbs = re.compile(r'\b(take|grab|heat|cook|add|remove|place|set|turn|start|begin|open|close|click|type|enter|select|choose|apply|use|try|write|read|call|send|check|verify|ensure|maintain|keep|hold|break|split|divide|combine|mix|stir|pour|cut|slice)\b', re.IGNORECASE)
        
        # Vagueness indicators
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are (many|various|several|some)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bfor the most part\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bmight not\b', r'\bmay not\b', r'\bwon\'t be able\b',
            r'\bcan\'t really\b', r'\bnot really\b',
            r'\bjust\s+(keep|try|do|get|go|be)\b',
            r'\byou should be able\b', r'\byou could try\b',
            r'\bor something\b', r'\bor whatever\b',
            r'\bstuff like that\b', r'\band things\b',
            r'\byou know\b', r'\bI guess\b', r'\bI think\b',
            r'\bnot sure\b', r'\bdon\'t know\b',
        ]
        vague_patterns = [re.compile(p, re.IGNORECASE) for p in vague_phrases]
        
        # Dismissive/unhelpful patterns
        dismissive_phrases = [
            r'\bjust\s+get over\b', r'\bjust\s+move on\b',
            r'\bjust\s+deal with\b', r'\bit\'s not that\b',
            r'\byou\'re just\b', r'\bjust\s+remember\b',
            r'\bstop\s+(being|feeling|thinking)\b',
        ]
        dismissive_patterns = [re.compile(p, re.IGNORECASE) for p in dismissive_phrases]
        
        sentence_scores = []
        
        for sent in sentences:
            score = 0.0
            words = sent.split()
            word_count = len(words)
            
            if word_count < 2:
                sentence_scores.append(0.0)
                continue
            
            # Numbers and measurements
            nums = number_pattern.findall(sent)
            score += min(len(nums) * 1.5, 4.0)
            
            percs = percentage_pattern.findall(sent)
            score += len(percs) * 2.0
            
            measures = measurement_pattern.findall(sent)
            score += len(measures) * 2.0
            
            # Proper nouns (named entities proxy)
            proper_nouns = proper_noun_pattern.findall(sent)
            # Filter out sentence-start words
            if proper_nouns and sent.startswith(proper_nouns[0]):
                proper_nouns = proper_nouns[1:]
            score += min(len(proper_nouns) * 0.8, 3.0)
            
            # Quoted text
            quotes = quoted_pattern.findall(sent)
            score += len(quotes) * 1.0
            
            # Parenthetical details
            parens = parenthetical_pattern.findall(sent)
            score += len(parens) * 1.0
            
            # Technical terms
            tech_terms = technical_term_pattern.findall(sent)
            score += min(len(tech_terms) * 0.3, 2.0)
            
            # Action verbs (concrete instructions)
            actions = action_verbs.findall(sent)
            score += min(len(actions) * 0.6, 2.5)
            
            # Colon definitions
            if colon_definition_pattern.search(sent):
                score += 0.8
            
            # Penalize vagueness
            for vp in vague_patterns:
                if vp.search(sent):
                    score -= 1.2
            
            # Penalize dismissiveness
            for dp in dismissive_patterns:
                if dp.search(sent):
                    score -= 2.0
            
            # Normalize by sentence length (reward density)
            if word_count > 0:
                density = score / max(word_count, 5) * 10
                score = score * 0.6 + density * 0.4
            
            sentence_scores.append(score)
        
        avg_sentence_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0
        max_sentence_score = max(sentence_scores) if sentence_scores else 0
        
        # ---- FEATURE 2: Structural organization (numbered/lettered steps) ----
        numbered_steps = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+\w', response_clean))
        lettered_steps = len(re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s+\w', response_clean))
        dash_bullets = len(re.findall(r'(?:^|\n)\s*[-•*]\s+\w', response_clean))
        
        structure_score = min((numbered_steps * 1.5 + lettered_steps * 1.0 + dash_bullets * 0.8), 6.0)
        
        # ---- FEATURE 3: Information density via unique content words ratio ----
        words_lower = re.findall(r'\b[a-z]+\b', response_clean.lower())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'also', 'now', 'then', 'here', 'there', 'when', 'where',
            'why', 'how', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
            'their', 'if', 'about', 'up', 'out', 'over', 'down', 'off',
        }
        
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 2]
        total_words = len(words_lower)
        
        if total_words > 0:
            content_ratio = len(content_words) / total_words
            unique_content = len(set(content_words))
            # Hapax legomena ratio (words appearing only once) - measures information richness
            word_freq = Counter(content_words)
            hapax = sum(1 for w, c in word_freq.items() if c == 1)
            hapax_ratio = hapax / max(unique_content, 1)
            
            info_density_score = content_ratio * 3.0 + min(unique_content / 15.0, 4.0) + hapax_ratio * 1.5
        else:
            info_density_score = 0
        
        # ---- FEATURE 4: Engagement and empathy markers (for support-type queries) ----
        empathy_patterns = [
            r'\bi (can )?(see|hear|understand|sense)\b',
            r'\bit\'s (completely |perfectly |totally )?(understandable|okay|natural|normal|fine)\b',
            r'\byour (feelings?|emotions?|experience|situation|concerns?)\b',
            r'\blet\'s\b',
            r'\btogether\b',
            r'\bwe (can|will|should)\b',
        ]
        
        empathy_score = 0
        for ep in empathy_patterns:
            if re.search(ep, response_clean, re.IGNORECASE):
                empathy_score += 0.7
        empathy_score = min(empathy_score, 3.0)
        
        # ---- FEATURE 5: Conditional/hedging language density (negative signal) ----
        hedge_words = re.findall(r'\b(might|maybe|perhaps|possibly|probably|could be|seems|appears|somewhat|fairly|rather|quite|pretty much)\b', response_clean, re.IGNORECASE)
        hedge_penalty = min(len(hedge_words) * 0.5, 4.0)
        
        # ---- FEATURE 6: Negative capability language (negative signal) ----
        negative_capability = re.findall(r'\b(can\'t|cannot|won\'t|unable|not able|might not|may not|probably won\'t)\b', response_clean, re.IGNORECASE)
        neg_cap_penalty = min(len(negative_capability) * 0.8, 3.0)
        
        # ---- FEATURE 7: Response completeness proxy ----
        # Longer responses with maintained quality indicate more thorough answers
        length_bonus = min(math.log(max(total_words, 1) + 1) / math.log(200), 1.5)
        
        # ---- FEATURE 8: Specificity through "for example", "for instance", "such as" ----
        example_markers = len(re.findall(r'\b(for example|for instance|such as|e\.g\.|i\.e\.|specifically|in particular|namely)\b', response_clean, re.IGNORECASE))
        example_score = min(example_markers * 1.2, 3.0)
        
        # ---- FEATURE 9: Cause-effect and logical connectors (indicates reasoning depth) ----
        logical_connectors = len(re.findall(r'\b(because|therefore|thus|hence|consequently|as a result|this means|which leads|so that|in order to|due to|since)\b', response_clean, re.IGNORECASE))
        logic_score = min(logical_connectors * 0.6, 2.5)
        
        # ---- FEATURE 10: Query-response alignment (addressing the actual question) ----
        query_content = [w for w in re.findall(r'\b[a-z]+\b', query.lower()) if w not in stop_words and len(w) > 3]
        response_content_set = set(content_words)
        
        if query_content:
            overlap = sum(1 for w in query_content if w in response_content_set)
            alignment_score = min(overlap / max(len(set(query_content)), 1) * 3.0, 3.0)
        else:
            alignment_score = 1.0
        
        # ---- COMBINE ALL FEATURES ----
        raw_score = (
            avg_sentence_score * 2.0      # Sentence-level evidence density
            + max_sentence_score * 0.5     # Best sentence quality
            + structure_score * 0.8        # Organizational structure
            + info_density_score * 0.7     # Information richness
            + empathy_score * 0.6          # Engagement (for support queries)
            + example_score * 0.9          # Explicit examples
            + logic_score * 0.6            # Reasoning depth
            + alignment_score * 0.5        # Query relevance
            + length_bonus * 0.8           # Completeness
            - hedge_penalty * 0.7          # Hedging penalty
            - neg_cap_penalty * 0.8        # Negative capability penalty
        )
        
        # Scale to 1-5 range
        # Empirical calibration: raw scores typically range from about -2 to 15
        scaled = 1.0 + (raw_score + 2.0) * (4.0 / 17.0)
        
        # Clamp to [0.5, 5.5]
        final_score = max(0.5, min(5.5, scaled))
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5