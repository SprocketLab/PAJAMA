def judging_function(query, response):
    """
    Evaluate response quality based on Evidence Density and Specificity.
    
    Focuses on concrete evidence: specific examples, data points, named entities,
    precise numbers, real-world references, and actionable details.
    Penalizes vague, generic filler language.
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        words = response.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # 1. SPECIFICITY INDICATORS (positive signals)
        # ============================================================
        
        # 1a. Numbers and quantities (concrete data points)
        numbers = re.findall(r'\b\d+[\.,]?\d*%?\b', response)
        number_count = len(numbers)
        number_score = min(number_count * 2.0, 15.0)
        
        # 1b. Named entities - capitalized words not at sentence start
        sentences = re.split(r'[.!?]+', response)
        named_entity_count = 0
        for sent in sentences:
            sent = sent.strip()
            sent_words = sent.split()
            if len(sent_words) > 1:
                for w in sent_words[1:]:
                    clean_w = w.strip(string.punctuation)
                    if clean_w and clean_w[0].isupper() and len(clean_w) > 1 and not clean_w.isupper():
                        named_entity_count += 1
        named_entity_score = min(named_entity_count * 1.5, 12.0)
        
        # 1c. Specific/precise language markers
        specific_markers = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b',
            r'\bin particular\b', r'\be\.g\.\b', r'\bi\.e\.\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies\b',
            r'\bdata\b', r'\bevidence\b', r'\bpercent\b', r'\bpercentage\b',
            r'\bmillion\b', r'\bbillion\b', r'\bthousand\b',
            r'\bexactly\b', r'\bprecisely\b',
        ]
        specificity_count = 0
        response_lower = response.lower()
        for pattern in specific_markers:
            matches = re.findall(pattern, response_lower)
            specificity_count += len(matches)
        specificity_score = min(specificity_count * 2.5, 15.0)
        
        # 1d. Enumerations and lists (structured concrete info)
        list_patterns = [
            r'\b(?:first|second|third|fourth|fifth)\b',
            r'\b(?:1\)|2\)|3\)|4\)|5\))',
            r'\b(?:1\.|2\.|3\.|4\.|5\.)',
            r'(?:^|\n)\s*[-•*]\s+',
            r'\b(?:step \d+)\b',
        ]
        list_count = 0
        for pattern in list_patterns:
            list_count += len(re.findall(pattern, response_lower))
        list_score = min(list_count * 2.0, 10.0)
        
        # 1e. Technical/domain-specific vocabulary density
        # Words that are longer and less common tend to be more specific
        technical_word_count = 0
        for w in words:
            clean = w.strip(string.punctuation).lower()
            if len(clean) >= 8:  # longer words tend to be more specific
                technical_word_count += 1
            elif len(clean) >= 6:
                technical_word_count += 0.3
        tech_density = technical_word_count / max(word_count, 1)
        tech_score = min(tech_density * 30, 12.0)
        
        # 1f. Concrete nouns and action verbs (approximation)
        concrete_indicators = [
            r'\bprocess\b', r'\bmethod\b', r'\btechnique\b', r'\bsystem\b',
            r'\btool\b', r'\bplatform\b', r'\bdevice\b', r'\bsoftware\b',
            r'\bserver\b', r'\bdatabase\b', r'\bprotocol\b', r'\balgorithm\b',
            r'\bframework\b', r'\bstructure\b', r'\bcomponent\b', r'\bmodule\b',
            r'\bfeature\b', r'\bfunction\b', r'\bparameter\b', r'\bvariable\b',
            r'\bcategory\b', r'\btype\b', r'\bclass\b', r'\bgroup\b',
        ]
        concrete_count = 0
        for pattern in concrete_indicators:
            concrete_count += len(re.findall(pattern, response_lower))
        concrete_score = min(concrete_count * 1.5, 8.0)
        
        # 1g. Comparative and descriptive adjectives (adds specificity)
        descriptive_patterns = [
            r'\bmore\s+\w+\s+than\b', r'\bless\s+\w+\s+than\b',
            r'\b\w+er\s+than\b', r'\bthe\s+most\b', r'\bthe\s+least\b',
            r'\bunlike\b', r'\bwhereas\b', r'\bwhile\b', r'\bin contrast\b',
            r'\bon the other hand\b', r'\bcompared to\b',
        ]
        comparative_count = 0
        for pattern in descriptive_patterns:
            comparative_count += len(re.findall(pattern, response_lower))
        comparative_score = min(comparative_count * 2.0, 8.0)
        
        # ============================================================
        # 2. VAGUENESS PENALTIES (negative signals)
        # ============================================================
        
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bthere are various\b', r'\bvarious factors\b',
            r'\bin many ways\b', r'\bin some ways\b', r'\bin various ways\b',
            r'\bgenerally speaking\b', r'\boverall\b',
            r'\bkind of\b', r'\bsort of\b',
            r'\bmore or less\b', r'\bto some extent\b',
            r'\ba lot of\b', r'\blots of\b',
            r'\bthings\b', r'\bstuff\b',
            r'\bbasically\b', r'\bessentially\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bsomewhat\b', r'\bsomehow\b',
            r'\band so on\b', r'\betc\b', r'\band more\b',
            r'\bin general\b', r'\btypically\b',
            r'\bcan be\b', r'\bmight be\b', r'\bcould be\b',
        ]
        vague_count = 0
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response_lower))
        
        # Normalize vagueness by word count
        vague_density = vague_count / max(word_count, 1)
        vague_penalty = min(vague_density * 100, 20.0)
        
        # ============================================================
        # 3. REPETITION PENALTY
        # ============================================================
        
        # Penalize repeated words/phrases (sign of low-quality padding)
        word_list = [w.strip(string.punctuation).lower() for w in words if len(w.strip(string.punctuation)) > 3]
        if word_list:
            word_freq = Counter(word_list)
            total_content_words = len(word_list)
            unique_content_words = len(word_freq)
            
            if total_content_words > 0:
                uniqueness_ratio = unique_content_words / total_content_words
            else:
                uniqueness_ratio = 0
            
            # High repetition = low uniqueness ratio
            if uniqueness_ratio < 0.3:
                repetition_penalty = 15.0
            elif uniqueness_ratio < 0.5:
                repetition_penalty = 8.0
            elif uniqueness_ratio < 0.6:
                repetition_penalty = 3.0
            else:
                repetition_penalty = 0.0
            
            # Check for exact phrase repetition
            bigrams = [' '.join(word_list[i:i+2]) for i in range(len(word_list)-1)]
            bigram_freq = Counter(bigrams)
            repeated_bigrams = sum(1 for bg, cnt in bigram_freq.items() if cnt > 2)
            repetition_penalty += min(repeated_bigrams * 3.0, 10.0)
        else:
            repetition_penalty = 5.0
            uniqueness_ratio = 0
        
        # ============================================================
        # 4. INFORMATION DENSITY
        # ============================================================
        
        # Count unique meaningful information units (approximated by unique content words)
        # relative to total length
        if word_count > 0:
            # Unique content words per sentence
            sentence_count = max(len([s for s in sentences if s.strip()]), 1)
            info_per_sentence = len(set(word_list)) / sentence_count
            info_density_score = min(info_per_sentence * 1.2, 10.0)
        else:
            info_density_score = 0.0
        
        # ============================================================
        # 5. RESPONSE LENGTH (moderate bonus for substantive responses)
        # ============================================================
        
        # Reward adequate length but with diminishing returns
        if word_count < 5:
            length_score = -5.0
        elif word_count < 15:
            length_score = 0.0
        elif word_count < 30:
            length_score = 3.0
        elif word_count < 60:
            length_score = 5.0
        elif word_count < 120:
            length_score = 7.0
        elif word_count < 200:
            length_score = 8.0
        else:
            length_score = 8.0  # cap it
        
        # ============================================================
        # 6. STRUCTURAL QUALITY
        # ============================================================
        
        # Multiple sentences suggest more thorough explanation
        real_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_variety_score = min(len(real_sentences) * 1.0, 6.0)
        
        # Presence of causal/explanatory connectors (shows reasoning with evidence)
        causal_connectors = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bwhich leads\b',
            r'\bcaused by\b', r'\bleads to\b', r'\bresults in\b',
        ]
        causal_count = 0
        for pattern in causal_connectors:
            causal_count += len(re.findall(pattern, response_lower))
        causal_score = min(causal_count * 2.0, 8.0)
        
        # ============================================================
        # 7. DETAIL ELABORATION (does the response expand on points?)
        # ============================================================
        
        # Parenthetical details, appositives, and elaborations
        elaboration_patterns = [
            r'\([^)]+\)',  # parenthetical info
            r'\b(?:which|that|where|when|who)\s+\w+',  # relative clauses
            r',\s*(?:which|that|including|such as)',  # appositives
        ]
        elaboration_count = 0
        for pattern in elaboration_patterns:
            elaboration_count += len(re.findall(pattern, response_lower))
        elaboration_score = min(elaboration_count * 1.5, 8.0)
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        total_positive = (
            number_score +          # up to 15
            named_entity_score +    # up to 12
            specificity_score +     # up to 15
            list_score +            # up to 10
            tech_score +            # up to 12
            concrete_score +        # up to 8
            comparative_score +     # up to 8
            info_density_score +    # up to 10
            length_score +          # up to 8
            sentence_variety_score +# up to 6
            causal_score +          # up to 8
            elaboration_score       # up to 8
        )
        # Max positive ~120
        
        total_negative = (
            vague_penalty +         # up to 20
            repetition_penalty      # up to 25
        )
        # Max negative ~45
        
        raw_score = total_positive - total_negative
        
        # Normalize to 0-100 range
        # Typical range: -20 to 80
        normalized = max(0.0, min(100.0, (raw_score + 10) * 1.0))
        
        # Apply a slight sigmoid-like compression to spread scores
        # centered around 30
        centered = normalized - 30
        compressed = 50 + 50 * (centered / math.sqrt(centered**2 + 900))
        
        return round(compressed, 2)
        
    except Exception as e:
        # Never crash - return a neutral score on error
        try:
            if response and len(response.strip()) > 0:
                return 25.0
            return 0.0
        except:
            return 0.0