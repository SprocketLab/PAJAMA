def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-based approach
    that focuses on detecting information-carrying tokens, proper nouns,
    numeric references, and penalizing empty/repetitive/vague content.
    
    This variant uses:
    - Named entity density (capitalized multi-word sequences)
    - Numeric/quantitative expression detection
    - Unique information token ratio (type-token ratio on content words)
    - Sentence-level information variance
    - Repetition penalty via compression ratio
    - Vague filler phrase density penalty
    - Response completeness relative to query
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        resp = response.strip()
        query_text = (query or "").strip()
        
        # Very short responses are almost always low quality
        resp_words = resp.split()
        word_count = len(resp_words)
        
        if word_count <= 2:
            return 0.5
        
        # === Feature 1: Named Entity Density ===
        # Detect sequences of capitalized words (likely proper nouns, places, names)
        named_entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        named_entities = re.findall(named_entity_pattern, resp)
        # Also single capitalized words not at sentence start
        sentences = re.split(r'[.!?]\s+', resp)
        mid_sentence_caps = 0
        for sent in sentences:
            words_in_sent = sent.split()
            for i, w in enumerate(words_in_sent):
                if i > 0 and len(w) > 1 and w[0].isupper() and w[1:].islower() and w.isalpha():
                    mid_sentence_caps += 1
        
        ne_count = len(named_entities) + mid_sentence_caps
        ne_density = ne_count / max(word_count, 1) * 10  # scale up
        ne_score = min(ne_density * 3, 3.0)
        
        # === Feature 2: Numeric/Quantitative Expression Density ===
        # Detect numbers, percentages, dates, measurements
        numeric_patterns = [
            r'\b\d+[,.]?\d*\s*%',          # percentages
            r'\b\d{4}\b',                    # years
            r'\b\d+[,.]?\d*\s*(million|billion|trillion|thousand|hundred)',  # large numbers
            r'\$\s*\d+',                     # dollar amounts
            r'\b\d+\s*(km|miles|meters|feet|inches|pounds|kg|lbs|oz|mg|gb|mb|tb)',  # measurements
            r'\b\d+:\d+',                    # times
            r'\b\d+/\d+',                    # fractions/dates
            r'\b\d+\.\d+\b',                # decimals
            r'\b\d+\b',                      # plain numbers
        ]
        
        numeric_matches = 0
        for pat in numeric_patterns:
            numeric_matches += len(re.findall(pat, resp, re.IGNORECASE))
        
        numeric_density = numeric_matches / max(word_count, 1) * 10
        numeric_score = min(numeric_density * 2.5, 2.5)
        
        # === Feature 3: Compression Ratio (repetition penalty) ===
        # If the response has lots of repeated phrases, it's low quality
        # Use character-level bigram repetition as a proxy
        resp_lower = resp.lower()
        
        # Word-level: check repeated trigrams
        lower_words = resp_lower.split()
        if len(lower_words) >= 3:
            trigrams = [tuple(lower_words[i:i+3]) for i in range(len(lower_words)-2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            unique_trigrams = len(trigram_counts)
            trigram_ratio = unique_trigrams / max(total_trigrams, 1)
        else:
            trigram_ratio = 1.0
        
        # Penalize highly repetitive text
        repetition_penalty = 0.0
        if trigram_ratio < 0.3:
            repetition_penalty = 3.0
        elif trigram_ratio < 0.5:
            repetition_penalty = 2.0
        elif trigram_ratio < 0.7:
            repetition_penalty = 1.0
        
        # === Feature 4: Vague Filler Phrase Density ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are many\b', r'\bthere are various\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bit is important to\b',
            r'\bas we all know\b', r'\bneedless to say\b', r'\bit goes without saying\b',
            r'\bfor the most part\b', r'\bin some cases\b', r'\bin many cases\b',
            r'\bcan be\b', r'\bmight be\b', r'\bcould be\b',
            r'\band so on\b', r'\betc\.?\b', r'\band more\b',
            r'\ba lot of\b', r'\bkind of\b', r'\bsort of\b',
            r'\bbasically\b', r'\bessentially\b', r'\bsomewhat\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bpossibly\b',
            r'\btend to\b', r'\btends to\b',
            r'\bthings like\b', r'\bstuff like\b',
        ]
        
        vague_count = 0
        for vp in vague_phrases:
            vague_count += len(re.findall(vp, resp_lower))
        
        vague_density = vague_count / max(word_count, 1) * 100
        vague_penalty = min(vague_density * 0.5, 2.0)
        
        # === Feature 5: Unique Content Word Ratio (Type-Token on content words) ===
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'that', 'this', 'these', 'those', 'it',
            'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        }
        
        content_words = [w for w in re.findall(r'[a-z]+', resp_lower) if w not in stop_words and len(w) > 2]
        if len(content_words) > 0:
            unique_content = len(set(content_words))
            total_content = len(content_words)
            # Type-token ratio (adjusted for length)
            ttr = unique_content / max(total_content, 1)
            # For longer texts, TTR naturally drops, so use root TTR
            root_ttr = unique_content / max(math.sqrt(total_content), 1)
            content_richness = min(root_ttr / 3.0, 1.0)  # normalize to 0-1
        else:
            content_richness = 0.0
        
        content_score = content_richness * 2.5
        
        # === Feature 6: Sentence-level Information Variance ===
        # Good responses have sentences that each add new information
        sentence_splits = re.split(r'[.!?\n]+', resp)
        sentence_splits = [s.strip() for s in sentence_splits if len(s.strip()) > 5]
        
        if len(sentence_splits) >= 2:
            sentence_word_sets = []
            for s in sentence_splits:
                s_words = set(re.findall(r'[a-z]+', s.lower())) - stop_words
                sentence_word_sets.append(s_words)
            
            # Measure how much new content each sentence adds
            seen_words = set()
            new_word_ratios = []
            for ws in sentence_word_sets:
                if len(ws) > 0:
                    new_words = ws - seen_words
                    new_word_ratios.append(len(new_words) / len(ws))
                    seen_words |= ws
            
            if new_word_ratios:
                avg_new_ratio = sum(new_word_ratios) / len(new_word_ratios)
            else:
                avg_new_ratio = 0.0
            
            info_progression_score = avg_new_ratio * 2.0
        else:
            # Single sentence - slight penalty for lack of elaboration but not huge
            info_progression_score = 0.5 if word_count > 5 else 0.0
        
        # === Feature 7: Specific Detail Indicators ===
        # Look for patterns that indicate specific, concrete information
        specificity_patterns = [
            r'\b(?:called|named|known as|titled|referred to as)\b',  # naming things
            r'\b(?:located in|found in|based in|situated in)\b',     # locations
            r'\b(?:published|written|authored|created|founded|established)\b',  # creation
            r'\b(?:according to|reported by|stated by|cited by)\b',  # attribution
            r'\b(?:for example|for instance|such as|including|specifically|namely)\b',  # examples
            r'\b(?:first|second|third|fourth|fifth)\b',             # ordinals (structured info)
            r'"[^"]{3,}"',                                           # quoted text
            r"'[^']{3,}'",                                           # quoted text
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',                      # proper names
        ]
        
        specificity_count = 0
        for sp in specificity_patterns:
            specificity_count += len(re.findall(sp, resp))
        
        specificity_density = specificity_count / max(word_count, 1) * 20
        specificity_score = min(specificity_density * 1.5, 2.0)
        
        # === Feature 8: Response Relevance (lightweight) ===
        # Check if response shares content words with query
        query_content = set(re.findall(r'[a-z]+', query_text.lower())) - stop_words
        resp_content = set(re.findall(r'[a-z]+', resp_lower)) - stop_words
        
        if len(query_content) > 0 and len(resp_content) > 0:
            overlap = len(query_content & resp_content)
            relevance = overlap / max(len(query_content), 1)
            relevance_score = min(relevance, 1.0) * 1.0
        else:
            relevance_score = 0.3  # neutral
        
        # === Feature 9: Length adequacy ===
        # Very short responses rarely have evidence density
        # But extremely long repetitive responses are bad too
        if word_count < 5:
            length_factor = 0.3
        elif word_count < 10:
            length_factor = 0.6
        elif word_count < 20:
            length_factor = 0.8
        elif word_count < 200:
            length_factor = 1.0
        elif word_count < 500:
            length_factor = 0.95
        else:
            length_factor = 0.85  # very long might be padded
        
        # === Feature 10: Garbage/noise detection ===
        # Detect HTML tags, code artifacts, excessive special characters
        html_tags = len(re.findall(r'<[^>]+>', resp))
        code_indicators = len(re.findall(r'(?:import |def |class |function |var |let |const )', resp))
        
        noise_penalty = 0.0
        # If response has lots of HTML/code but query doesn't ask for it
        query_asks_code = bool(re.search(r'\b(?:code|html|program|script|function|tag)\b', query_text.lower()))
        if not query_asks_code:
            if html_tags > 3:
                noise_penalty += 1.0
            if code_indicators > 2:
                noise_penalty += 1.0
        
        # Detect if response is mostly punctuation or special chars
        alpha_chars = sum(1 for c in resp if c.isalpha())
        total_chars = len(resp)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.4:
                noise_penalty += 1.5
        
        # === Combine all features ===
        raw_score = (
            ne_score                    # 0-3: named entities
            + numeric_score             # 0-2.5: numbers/data
            + content_score             # 0-2.5: content richness
            + info_progression_score    # 0-2: information progression
            + specificity_score         # 0-2: specificity indicators
            + relevance_score           # 0-1: query relevance
            - repetition_penalty        # 0-3: repetition
            - vague_penalty             # 0-2: vagueness
            - noise_penalty             # 0-3.5: noise
        )
        
        # Apply length factor
        raw_score *= length_factor
        
        # Normalize to 0-10 range
        # Theoretical max is about 13, theoretical min is about -8.5
        # Map to 0-10
        final_score = max(0.0, min(10.0, raw_score + 2.0))
        
        # Round to 1 decimal
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0