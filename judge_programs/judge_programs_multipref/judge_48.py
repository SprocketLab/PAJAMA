def judging_function(query, response):
    """
    Evaluate clarity and conciseness of an LLM response.
    Uses a signal-to-noise ratio approach: measures how much useful, structured
    information is conveyed per unit of text, penalizing bloat and rewarding
    efficient communication.
    
    VARIANT 8: Signal-to-noise ratio + structural efficiency approach
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not query:
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 5:
            return 0.0
        
        # ============================================================
        # COMPONENT 1: Lexical Efficiency (unique info per word)
        # ============================================================
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words) if words else 1
        
        # Type-token ratio (vocabulary richness) - but adjusted for length
        unique_words = len(set(words)) if words else 0
        # Use root TTR to normalize for text length
        root_ttr = unique_words / math.sqrt(total_words) if total_words > 0 else 0
        # Normalize to 0-1 range (typical root TTR is 3-10)
        lexical_efficiency = min(root_ttr / 8.0, 1.0)
        
        # ============================================================
        # COMPONENT 2: Sentence-level conciseness
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length in words (sweet spot: 10-20 words)
        sentence_lengths = []
        for sent in sentences:
            sent_words = re.findall(r'[a-zA-Z]+', sent)
            sentence_lengths.append(len(sent_words))
        
        avg_sent_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 15
        
        # Penalize very long sentences (>30 words) and very short (<5 words)
        # Optimal around 12-20 words
        if avg_sent_len < 5:
            sent_score = 0.4
        elif avg_sent_len <= 20:
            sent_score = 0.7 + 0.3 * (1.0 - abs(avg_sent_len - 15) / 15)
        elif avg_sent_len <= 30:
            sent_score = 0.7 - 0.3 * ((avg_sent_len - 20) / 10)
        else:
            sent_score = 0.3
        
        # ============================================================
        # COMPONENT 3: Structural organization signals
        # ============================================================
        # Detect formatting elements that aid clarity
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_bullet_list = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,4}\s', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_colon_structure = bool(re.search(r'\w+:\s', response))
        
        structure_signals = sum([
            has_numbered_list * 1.5,
            has_bullet_list * 1.0,
            has_headers * 1.5,
            has_bold * 1.0,
            has_colon_structure * 0.5
        ])
        
        # Scale structure score based on response length (longer responses benefit more from structure)
        if total_words > 50:
            structure_score = min(structure_signals / 4.0, 1.0)
        elif total_words > 20:
            structure_score = min(structure_signals / 5.0, 0.8)
        else:
            # Short responses don't need much structure
            structure_score = 0.6 + min(structure_signals / 10.0, 0.3)
        
        # ============================================================
        # COMPONENT 4: Filler and hedging penalty
        # ============================================================
        filler_phrases = [
            r'\bbasically\b', r'\bactually\b', r'\bin fact\b',
            r'\bit is important to note that\b', r'\bit should be noted that\b',
            r'\bit is worth mentioning that\b', r'\bneedless to say\b',
            r'\bas a matter of fact\b', r'\bin other words\b',
            r'\bthat being said\b', r'\bhaving said that\b',
            r'\bat the end of the day\b', r'\ball things considered\b',
            r'\bto be honest\b', r'\bif you will\b',
            r'\bso to speak\b', r'\bas it were\b',
            r'\bfor what it\'s worth\b', r'\bin any case\b',
            r'\bwith that being said\b', r'\bit goes without saying\b',
            r'\bthe fact of the matter is\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / max(num_sentences, 1)
        filler_penalty = max(0, 1.0 - filler_ratio * 0.4)
        
        # ============================================================
        # COMPONENT 5: Redundancy detection (n-gram repetition)
        # ============================================================
        # Check for repeated trigrams and 4-grams
        if len(words) >= 4:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_redundancy = repeated_trigrams / max(len(trigrams), 1)
            
            fourgrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            fourgram_counts = Counter(fourgrams)
            repeated_fourgrams = sum(1 for c in fourgram_counts.values() if c > 1)
            fourgram_redundancy = repeated_fourgrams / max(len(fourgrams), 1)
            
            redundancy_penalty = max(0, 1.0 - (trigram_redundancy * 2.0 + fourgram_redundancy * 4.0))
        else:
            redundancy_penalty = 0.8
        
        # ============================================================
        # COMPONENT 6: Opening directness (does it get to the point?)
        # ============================================================
        # Check if response starts with fluff vs. substance
        first_100_chars = response_lower[:150]
        
        fluff_openers = [
            r'^(great question|that\'s a great|what a great|excellent question|good question)',
            r'^(oh,?\s|well,?\s|so,?\s|hmm)',
            r'^(i\'m glad you asked|thank you for asking|thanks for asking)',
            r'^(that\'s an? (interesting|excellent|wonderful|fantastic))',
            r'^(a classic)',
        ]
        
        opener_penalty = 1.0
        for pattern in fluff_openers:
            if re.search(pattern, first_100_chars.strip()):
                opener_penalty = 0.85
                break
        
        # Reward responses that start directly with content
        # Check if first sentence contains query-relevant words
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                      'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some', 'them',
                      'than', 'its', 'over', 'also', 'that', 'this', 'with', 'from', 'what',
                      'how', 'why', 'when', 'where', 'which', 'who', 'will', 'would', 'could',
                      'should', 'about', 'into', 'does', 'did', 'being'}
        query_content_words = query_words - stop_words
        
        if sentences and query_content_words:
            first_sent_words = set(re.findall(r'[a-zA-Z]{3,}', sentences[0].lower()))
            relevance_in_opener = len(first_sent_words & query_content_words) / max(len(query_content_words), 1)
            directness_score = min(0.5 + relevance_in_opener, 1.0)
        else:
            directness_score = 0.5
        
        # ============================================================
        # COMPONENT 7: Information density (content words ratio)
        # ============================================================
        function_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                         'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                         'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                         'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
                         'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
                         'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
                         'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'it', 'its',
                         'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                         'she', 'her', 'they', 'them', 'their', 'also', 'about'}
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        content_ratio = len(content_words) / max(total_words, 1)
        # Ideal content ratio is around 0.45-0.65
        info_density = min(content_ratio / 0.55, 1.0)
        
        # ============================================================
        # COMPONENT 8: Appropriate length for query type
        # ============================================================
        query_lower = query.lower()
        
        # Detect query complexity
        is_simple_question = bool(re.search(r'^(what is|who is|when did|where is|is it|do you|can you|does)', query_lower))
        is_howto = bool(re.search(r'(how (can|do|to|should)|steps|guide|help me|teach me|learn)', query_lower))
        is_opinion = bool(re.search(r'(do you think|should|opinion|believe|feel about)', query_lower))
        is_creative = bool(re.search(r'(write|create|suggest|ideas|recipe|story)', query_lower))
        
        # Expected word count ranges
        if is_simple_question:
            ideal_min, ideal_max = 30, 150
        elif is_howto:
            ideal_min, ideal_max = 80, 300
        elif is_creative:
            ideal_min, ideal_max = 60, 250
        else:
            ideal_min, ideal_max = 40, 250
        
        if total_words < ideal_min:
            length_score = 0.5 + 0.5 * (total_words / ideal_min)
        elif total_words <= ideal_max:
            length_score = 1.0
        else:
            overshoot = (total_words - ideal_max) / ideal_max
            length_score = max(0.4, 1.0 - overshoot * 0.5)
        
        # ============================================================
        # COMPONENT 9: Clarity markers (transition words, logical connectors)
        # ============================================================
        clarity_markers = [
            r'\bfirst(ly)?\b', r'\bsecond(ly)?\b', r'\bthird(ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bin summary\b',
            r'\bfor example\b', r'\bsuch as\b', r'\bspecifically\b',
            r'\bhowever\b', r'\btherefore\b', r'\bconsequently\b',
            r'\bin contrast\b', r'\badditionally\b', r'\bmoreover\b',
            r'\bhere\'s\b', r'\bhere are\b', r'\bstep \d\b',
        ]
        
        clarity_count = 0
        for pattern in clarity_markers:
            clarity_count += len(re.findall(pattern, response_lower))
        
        # Normalize by number of sentences
        clarity_ratio = clarity_count / max(num_sentences, 1)
        clarity_marker_score = min(0.5 + clarity_ratio * 0.5, 1.0)
        
        # ============================================================
        # COMPONENT 10: Sentence length variance (good writing has varied rhythm)
        # ============================================================
        if len(sentence_lengths) > 2:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Good writing has CV around 0.3-0.6
            if cv < 0.1:
                rhythm_score = 0.5  # Too monotonous
            elif cv <= 0.6:
                rhythm_score = 0.7 + 0.3 * min(cv / 0.4, 1.0)
            else:
                rhythm_score = max(0.5, 1.0 - (cv - 0.6) * 0.5)
        else:
            rhythm_score = 0.6
        
        # ============================================================
        # COMBINE ALL COMPONENTS with weights
        # ============================================================
        # Weighted combination emphasizing the most important clarity/conciseness signals
        score = (
            lexical_efficiency * 1.2 +      # Vocabulary richness
            sent_score * 1.5 +               # Sentence length appropriateness
            structure_score * 2.0 +          # Structural organization
            filler_penalty * 1.0 +           # Filler words penalty
            redundancy_penalty * 1.5 +       # Repetition penalty
            opener_penalty * 0.8 +           # Opening directness
            directness_score * 1.2 +         # Query relevance in opening
            info_density * 1.0 +             # Content word ratio
            length_score * 1.3 +             # Appropriate length
            clarity_marker_score * 1.0 +     # Logical connectors
            rhythm_score * 0.5              # Sentence rhythm variety
        )
        
        total_weight = 1.2 + 1.5 + 2.0 + 1.0 + 1.5 + 0.8 + 1.2 + 1.0 + 1.3 + 1.0 + 0.5
        
        # Normalize to 0-100 scale
        normalized_score = (score / total_weight) * 100
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Center around 70 (most responses are decent)
        centered = (normalized_score - 50) / 25
        transformed = 50 + 50 * (2 / (1 + math.exp(-centered)) - 1)
        
        return round(max(0, min(100, transformed)), 2)
        
    except Exception as e:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 10:
                return 45.0
            return 20.0
        except:
            return 30.0