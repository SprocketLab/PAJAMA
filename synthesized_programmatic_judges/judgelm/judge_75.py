def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    focused on detecting concrete linguistic markers vs. vague hedging language.
    
    Algorithm: Combines detection of specific evidence markers (numbers, proper nouns,
    technical terms, precise references) with penalization of vagueness indicators,
    using a sentence-level analysis approach rather than document-level statistics.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 2:
            return 0.0
        
        query = query.strip() if query else ""
        
        # Split into sentences for sentence-level analysis
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        words = re.findall(r'\b\w+\b', response)
        word_count = len(words)
        
        if word_count < 2:
            return 0.5
        
        # ---- FEATURE 1: Named Entity Density ----
        # Detect capitalized multi-word phrases (likely proper nouns/named entities)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response)
        # Single capitalized words not at sentence start
        # Find words that are capitalized but not at the very start of a sentence
        cap_words_in_middle = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response)
        
        named_entity_score = min((len(named_entities) * 3 + len(cap_words_in_middle)) / max(word_count, 1) * 50, 10)
        
        # ---- FEATURE 2: Numeric Precision ----
        # Detect various number formats
        percentages = re.findall(r'\d+\.?\d*\s*%', response)
        years = re.findall(r'\b(?:1[0-9]{3}|2[0-9]{3})\b', response)
        money = re.findall(r'[\$€£]\s*\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s*(?:dollars|euros|pounds|USD|EUR)', response)
        plain_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', response)
        fractions = re.findall(r'\b\d+/\d+\b', response)
        
        numeric_items = len(percentages) * 2 + len(years) * 1.5 + len(money) * 2 + len(plain_numbers) * 0.5 + len(fractions) * 1.5
        numeric_score = min(numeric_items / max(len(sentences), 1) * 3, 10)
        
        # ---- FEATURE 3: Vagueness Penalty ----
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are many\b', r'\bthere are various\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bsome say\b',
            r'\bmost people\b', r'\boften times\b', r'\bit varies\b',
            r'\ba lot of\b', r'\bkind of\b', r'\bsort of\b',
            r'\bmore or less\b', r'\bto some extent\b', r'\bin some cases\b',
            r'\bcan be\b', r'\bmight be\b', r'\bcould be\b',
            r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b',
            r'\bsomewhat\b', r'\bsomehow\b',
            r'\betc\.?\b', r'\band so on\b', r'\band more\b',
            r'\bamong others\b', r'\bthings like that\b',
        ]
        
        vague_count = 0
        response_lower = response.lower()
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response_lower))
        
        vagueness_penalty = min(vague_count / max(len(sentences), 1) * 2.5, 8)
        
        # ---- FEATURE 4: Specificity Markers ----
        # Technical/specific language indicators
        specific_patterns = [
            r'\b(?:specifically|precisely|exactly|approximately|roughly)\b',
            r'\b(?:according to|based on|as reported by|as stated in)\b',
            r'\b(?:for example|for instance|such as|e\.g\.|i\.e\.)\b',
            r'\b(?:known as|called|named|titled|referred to as)\b',
            r'\b(?:located in|located at|found in|based in)\b',
            r'\b(?:published|authored|written by|created by|developed by)\b',
            r'\b(?:measured|calculated|estimated at|valued at)\b',
            r'\b(?:between \d+ and \d+)\b',
            r'\b(?:from \d+ to \d+)\b',
        ]
        
        specificity_count = 0
        for pattern in specific_patterns:
            specificity_count += len(re.findall(pattern, response_lower))
        
        specificity_score = min(specificity_count / max(len(sentences), 1) * 5, 10)
        
        # ---- FEATURE 5: Sentence Information Density ----
        # Average unique content words per sentence (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        }
        
        content_densities = []
        for sent in sentences:
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            content_words = [w for w in sent_words if w not in stop_words and len(w) > 2]
            unique_content = set(content_words)
            if len(sent_words) > 0:
                density = len(unique_content) / len(sent_words)
                content_densities.append(density)
        
        avg_content_density = sum(content_densities) / len(content_densities) if content_densities else 0
        content_density_score = avg_content_density * 12  # Scale to ~0-8 range
        
        # ---- FEATURE 6: Repetition Penalty ----
        # Detect repeated phrases (3-grams)
        if word_count >= 6:
            lower_words = [w.lower() for w in words]
            trigrams = [' '.join(lower_words[i:i+3]) for i in range(len(lower_words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            repetition_penalty = min(repetition_ratio * 15, 6)
        else:
            repetition_penalty = 0
        
        # ---- FEATURE 7: Response Relevance (lightweight) ----
        query_words = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
        response_words = set(re.findall(r'\b\w+\b', response_lower)) - stop_words
        
        if query_words:
            relevance = len(query_words & response_words) / len(query_words)
        else:
            relevance = 0.5
        
        relevance_score = relevance * 3  # 0-3 range
        
        # ---- FEATURE 8: Structural Completeness ----
        # Check if response seems complete vs truncated or minimal
        ends_properly = response.rstrip()[-1] in '.!?")\']' if response.rstrip() else False
        has_reasonable_length = 10 <= word_count <= 500
        
        completeness_score = 0
        if ends_properly:
            completeness_score += 1.0
        if has_reasonable_length:
            completeness_score += 1.0
        if word_count >= 20:
            completeness_score += 0.5
        
        # ---- FEATURE 9: Garbage/Off-topic Detection ----
        # Detect HTML, code blocks, excessive formatting that isn't content
        html_tags = len(re.findall(r'<[^>]+>', response))
        code_indicators = len(re.findall(r'(?:import |def |class |function |var |let |const )', response))
        
        # Check if response is mostly code/HTML when query doesn't ask for it
        query_asks_code = bool(re.search(r'\b(?:code|html|program|script|function|tag)\b', query.lower()))
        
        garbage_penalty = 0
        if not query_asks_code:
            garbage_penalty = min((html_tags + code_indicators * 2) / max(len(sentences), 1) * 2, 5)
        
        # ---- FEATURE 10: Unique Long Words (domain-specific terminology) ----
        long_words = [w for w in words if len(w) >= 8]
        unique_long = set(w.lower() for w in long_words)
        terminology_score = min(len(unique_long) / max(word_count, 1) * 30, 5)
        
        # ---- COMBINE SCORES ----
        raw_score = (
            named_entity_score * 1.2 +      # Named entities are strong evidence
            numeric_score * 1.5 +             # Numbers are very concrete
            specificity_score * 1.3 +         # Specificity markers
            content_density_score * 1.0 +     # Information density per sentence
            relevance_score * 1.0 +           # Query relevance
            completeness_score * 0.8 +        # Structural completeness
            terminology_score * 0.7 -         # Domain terminology
            vagueness_penalty * 1.5 -         # Vagueness hurts
            repetition_penalty * 1.2 -        # Repetition hurts
            garbage_penalty * 1.3             # Off-topic garbage hurts
        )
        
        # Length adjustment: very short responses get penalized, but not linearly
        # This uses a sigmoid-like curve
        length_factor = 1.0 - 1.0 / (1.0 + math.exp((word_count - 5) * 0.5))
        
        # Very long responses with lots of repetition should be penalized
        if word_count > 200:
            unique_ratio = len(set(w.lower() for w in words)) / word_count
            if unique_ratio < 0.4:
                length_factor *= 0.6
        
        raw_score *= length_factor
        
        # Normalize to 0-10 range
        # Empirically, raw scores tend to range from about -5 to 20
        normalized = (raw_score + 3) / 2.5
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            words = response.split() if response else []
            return min(max(len(words) / 20.0, 0.5), 5.0)
        except Exception:
            return 2.0