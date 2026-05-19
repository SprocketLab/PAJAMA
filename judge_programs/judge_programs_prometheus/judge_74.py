def judging_function(query, response):
    """
    Evaluates evidence density and specificity in LLM responses.
    
    This variant uses a sentence-level analysis approach:
    - Analyzes each sentence for concrete evidence markers
    - Computes an "evidence density ratio" (proportion of sentences with evidence)
    - Measures specificity through named patterns, numerical data, and actionable language
    - Penalizes vague hedging language at the sentence level
    
    Different from Variant 1 (which uses vocabulary diversity, bullet/list detection, 
    word overlap, concreteness) by focusing on sentence-level evidence classification
    and hedging penalty ratios.
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
        
        num_sentences = len(sentences)
        
        # === FEATURE 1: Sentence-level evidence classification ===
        # For each sentence, determine if it contains concrete evidence
        
        # Patterns indicating concrete/specific content
        number_pattern = re.compile(r'\b\d+\.?\d*\b')
        specific_action_verbs = re.compile(
            r'\b(start|begin|take|use|apply|create|build|write|read|call|send|'
            r'open|close|add|remove|set|check|run|install|click|select|choose|'
            r'type|enter|press|drag|navigate|download|upload|configure|enable|'
            r'disable|implement|execute|measure|calculate|compare|identify|'
            r'break down|divide|organize|prioritize|schedule|allocate|assign|'
            r'heat|cook|brown|stir|pour|mix|chop|slice|grab|whip)\b',
            re.IGNORECASE
        )
        sequence_markers = re.compile(
            r'\b(first|second|third|then|next|after that|finally|step \d|'
            r'once|before|afterward|subsequently|meanwhile|lastly|now)\b',
            re.IGNORECASE
        )
        causal_connectors = re.compile(
            r'\b(because|since|therefore|thus|consequently|as a result|'
            r'due to|this means|which leads to|so that|in order to|'
            r'this helps|this allows|this enables|this ensures)\b',
            re.IGNORECASE
        )
        example_markers = re.compile(
            r'\b(for example|for instance|such as|like|e\.g\.|i\.e\.|'
            r'specifically|in particular|namely|consider|imagine|suppose)\b',
            re.IGNORECASE
        )
        
        evidence_sentences = 0
        total_evidence_strength = 0.0
        
        for sent in sentences:
            sent_score = 0.0
            
            # Numbers present
            nums = number_pattern.findall(sent)
            if nums:
                sent_score += 0.4 + 0.1 * min(len(nums), 3)
            
            # Specific action verbs
            actions = specific_action_verbs.findall(sent)
            if actions:
                sent_score += 0.3 + 0.05 * min(len(actions), 4)
            
            # Sequence markers
            if sequence_markers.search(sent):
                sent_score += 0.25
            
            # Causal reasoning
            if causal_connectors.search(sent):
                sent_score += 0.3
            
            # Examples
            if example_markers.search(sent):
                sent_score += 0.35
            
            if sent_score > 0.2:
                evidence_sentences += 1
            
            total_evidence_strength += sent_score
        
        evidence_density_ratio = evidence_sentences / max(num_sentences, 1)
        avg_evidence_strength = total_evidence_strength / max(num_sentences, 1)
        
        # === FEATURE 2: Vagueness / hedging penalty at sentence level ===
        vague_patterns = re.compile(
            r'\b(many people|some people|it depends|various factors|'
            r'there are many|there are various|in general|generally speaking|'
            r'it might|it could be|maybe|perhaps|possibly|probably|'
            r'kind of|sort of|more or less|to some extent|in some ways|'
            r'a lot of|a bunch of|stuff|things like that|or something|'
            r'whatever|you know|just|basically|literally|honestly)\b',
            re.IGNORECASE
        )
        
        dismissive_patterns = re.compile(
            r'\b(just do|just try|just get|just keep|just remember|'
            r'you should be able|it\'s not that|no big deal|'
            r'get over it|move on|deal with it|toughen up|'
            r"that's life|part of life|nothing wrong)\b",
            re.IGNORECASE
        )
        
        vague_sentences = 0
        dismissive_count = 0
        
        for sent in sentences:
            vague_matches = vague_patterns.findall(sent)
            if len(vague_matches) >= 2:
                vague_sentences += 1
            elif len(vague_matches) == 1 and not number_pattern.search(sent) and not specific_action_verbs.search(sent):
                vague_sentences += 0.5
            
            if dismissive_patterns.search(sent):
                dismissive_count += 1
        
        vagueness_ratio = vague_sentences / max(num_sentences, 1)
        dismissive_ratio = dismissive_count / max(num_sentences, 1)
        
        # === FEATURE 3: Structural depth indicators ===
        # Numbered lists or structured enumeration
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean))
        lettered_items = len(re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s', response_clean))
        dash_items = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response_clean))
        
        structure_count = numbered_items + lettered_items + dash_items
        structure_score = min(structure_count * 0.15, 1.0)
        
        # === FEATURE 4: Engagement and empathy depth (for emotional queries) ===
        emotional_query_words = re.compile(
            r'\b(feel|feeling|emotion|stress|frustrat|sad|angry|upset|'
            r'heartbroken|devastat|lonely|loneliness|despair|grief|'
            r'anxious|anxiety|worried|overwhelm|exhaust|tired|down|'
            r'struggling|difficult|hard time|breakup|loss|passed away)\b',
            re.IGNORECASE
        )
        
        is_emotional_query = len(emotional_query_words.findall(query)) >= 1
        
        empathy_markers = re.compile(
            r'\b(I understand|I can see|I hear|I\'m sorry|it\'s okay|'
            r'it\'s completely|it\'s perfectly|it\'s natural|it\'s understandable|'
            r'that\'s understandable|completely understandable|totally understandable|'
            r'give yourself|allow yourself|be kind to yourself|'
            r'your feelings are valid|it\'s normal to feel|'
            r'take a moment|take your time|no rush)\b',
            re.IGNORECASE
        )
        
        empathy_count = len(empathy_markers.findall(response_clean))
        
        # === FEATURE 5: Specificity through unique content words ===
        words = re.findall(r'[a-zA-Z]+', response_clean.lower())
        
        # Filter out very common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or',
            'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its',
            'you', 'your', 'we', 'our', 'they', 'their', 'i', 'me', 'my',
            'he', 'she', 'him', 'her', 'his', 'what', 'which', 'who', 'whom',
            'about', 'up', 'also', 'them', 'us', 'am', 'get', 'got', 'make',
            'made', 'go', 'going', 'come', 'take', 'know', 'see', 'think',
            'say', 'said', 'like', 'well', 'back', 'much', 'even', 'still',
            'way', 'new', 'one', 'two', 'time', 'long', 'good', 'right'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if content_words:
            word_freq = Counter(content_words)
            # Hapax legomena ratio (words appearing once) - indicates specificity
            hapax = sum(1 for w, c in word_freq.items() if c == 1)
            hapax_ratio = hapax / len(word_freq) if word_freq else 0
            
            # Average word length of content words (longer = more specific/technical)
            avg_content_word_len = sum(len(w) for w in content_words) / len(content_words)
            
            # Content word density
            content_density = len(content_words) / max(len(words), 1)
        else:
            hapax_ratio = 0
            avg_content_word_len = 0
            content_density = 0
        
        # === FEATURE 6: Query-response relevance through topic coverage ===
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_content = set(w for w in query_words if w not in stop_words and len(w) > 2)
        response_content = set(content_words)
        
        if query_content:
            topic_coverage = len(query_content & response_content) / len(query_content)
        else:
            topic_coverage = 0.5
        
        # === FEATURE 7: Response completeness / length adequacy ===
        word_count = len(words)
        # Moderate length is good; too short is bad
        if word_count < 20:
            length_score = 0.2
        elif word_count < 40:
            length_score = 0.5
        elif word_count < 80:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # === FEATURE 8: Conditional/qualifying language (shows nuance vs dismissiveness) ===
        nuance_patterns = re.compile(
            r'\b(however|although|while|on the other hand|that said|'
            r'keep in mind|remember that|it\'s important to|note that|'
            r'be aware|consider|worth noting|crucial|essential|key)\b',
            re.IGNORECASE
        )
        nuance_count = len(nuance_patterns.findall(response_clean))
        nuance_score = min(nuance_count * 0.12, 0.6)
        
        # === SCORING FORMULA ===
        # Weighted combination of all features
        
        score = 0.0
        
        # Evidence density (0-2.5 points)
        score += evidence_density_ratio * 1.5
        score += min(avg_evidence_strength, 1.0) * 1.0
        
        # Structure bonus (0-0.8 points)
        score += structure_score * 0.8
        
        # Specificity through vocabulary (0-1.5 points)
        score += min(hapax_ratio, 0.85) * 0.6
        score += min((avg_content_word_len - 3) / 5, 1.0) * 0.5 if avg_content_word_len > 3 else 0
        score += content_density * 0.4
        
        # Topic relevance (0-0.8 points)
        score += topic_coverage * 0.8
        
        # Length adequacy (0-0.6 points)
        score += length_score * 0.6
        
        # Nuance (0-0.6 points)
        score += nuance_score
        
        # Empathy bonus for emotional queries (0-0.7 points)
        if is_emotional_query:
            empathy_score = min(empathy_count * 0.2, 0.7)
            score += empathy_score
        
        # === PENALTIES ===
        # Vagueness penalty (0-1.5 points deducted)
        score -= vagueness_ratio * 1.5
        
        # Dismissiveness penalty (0-1.0 points deducted)
        score -= dismissive_ratio * 1.0
        
        # Penalty for negation-heavy responses ("can't", "won't", "not able", "might not")
        negation_inability = len(re.findall(
            r'\b(can\'t|cannot|won\'t|not able|might not|may not|probably won\'t|'
            r'unlikely to|unable to|doesn\'t have|don\'t have)\b',
            response_clean, re.IGNORECASE
        ))
        score -= min(negation_inability * 0.15, 0.8)
        
        # Normalize to 1-5 scale
        # Raw score typically ranges from about -1 to 6
        # Map to 1-5
        normalized = 1.0 + (score / 5.5) * 4.0
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        return 2.5