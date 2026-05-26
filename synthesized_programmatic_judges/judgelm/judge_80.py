def judging_function(query, response):
    """
    Evaluate response quality based on evidence density and specificity.
    
    This variant uses a unique approach based on:
    1. Named entity density (capitalized multi-word phrases)
    2. Numeric/quantitative information density
    3. Specificity markers (precise language patterns)
    4. Anti-vagueness scoring (penalizing weasel words and filler)
    5. Information-to-noise ratio using sentence-level analysis
    6. Structural coherence (complete sentences vs fragments)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 2:
            return 0.0
        
        words = response_stripped.split()
        word_count = len(words)
        
        if word_count < 2:
            return 0.5
        
        # ---- 1. Named Entity Density ----
        # Look for capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]+', response_stripped)
        named_entity_count = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = sent.split()
            # Skip first word (sentence starter)
            for i, w in enumerate(sent_words):
                if i == 0:
                    continue
                # Capitalized word not after common punctuation
                clean_w = re.sub(r'[^a-zA-Z]', '', w)
                if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                    named_entity_count += 1
        
        # Also detect multi-word proper nouns (consecutive capitalized words)
        proper_noun_pattern = re.findall(r'(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)', response_stripped)
        multi_word_entities = len(proper_noun_pattern)
        
        entity_score = min(10, (named_entity_count * 1.5 + multi_word_entities * 3.0))
        
        # ---- 2. Numeric/Quantitative Information ----
        # Find numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', response_stripped)
        percentages = re.findall(r'\d+\.?\d*\s*%', response_stripped)
        years = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', response_stripped)
        measurements = re.findall(r'\b\d+\.?\d*\s*(?:km|mi|lb|kg|ft|m|cm|mm|mph|kph|GB|MB|TB|Hz|GHz|MHz)\b', response_stripped, re.IGNORECASE)
        currency = re.findall(r'[\$€£¥]\s*\d[\d,]*\.?\d*|\d[\d,]*\.?\d*\s*(?:dollars|euros|pounds|yen|cents)', response_stripped, re.IGNORECASE)
        
        numeric_items = len(numbers) + len(percentages) * 2 + len(years) * 1.5 + len(measurements) * 2 + len(currency) * 2
        numeric_score = min(10, numeric_items * 1.5)
        
        # ---- 3. Specificity Markers ----
        # Words/phrases indicating precise, specific information
        specificity_patterns = [
            r'\bspecifically\b', r'\bin particular\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bincluding\b', r'\baccording to\b', r'\bbased on\b',
            r'\blocated (?:in|at|near)\b', r'\bfounded (?:in|by)\b',
            r'\bknown as\b', r'\bcalled\b', r'\bentitled\b',
            r'\bpublished\b', r'\breleased\b', r'\bdeveloped by\b',
            r'\bcreated by\b', r'\binvented by\b', r'\bdiscovered\b',
            r'\bexactly\b', r'\bprecisely\b', r'\bapproximately\b',
        ]
        specificity_count = 0
        response_lower = response_stripped.lower()
        for pat in specificity_patterns:
            specificity_count += len(re.findall(pat, response_lower))
        
        specificity_score = min(10, specificity_count * 2.0)
        
        # ---- 4. Anti-Vagueness (Weasel Words / Filler Penalty) ----
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are (?:many|various|several|some)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bkind of\b',
            r'\bsort of\b', r'\bmore or less\b', r'\bto some extent\b',
            r'\bquite a few\b', r'\ba number of\b', r'\ba lot of\b',
            r'\bmost people\b', r'\beveryone knows\b', r'\bit is said\b',
            r'\bthey say\b', r'\bsome say\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bmight be\b',
            r'\bcould be\b', r'\btend to\b', r'\boften\b',
            r'\bsometimes\b', r'\busually\b', r'\bnormally\b',
            r'\bcan vary\b', r'\bvaries\b', r'\bdepending on\b',
            r'\bin some cases\b', r'\bin certain cases\b',
        ]
        vague_count = 0
        for pat in vague_patterns:
            vague_count += len(re.findall(pat, response_lower))
        
        # Normalize by word count
        vague_density = vague_count / max(1, word_count) * 100
        vague_penalty = min(5, vague_density * 3)
        
        # ---- 5. Information-to-Noise Ratio (Sentence-level) ----
        # Analyze each sentence for content density
        non_empty_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 2]
        
        content_words_ratio = 0
        if non_empty_sentences:
            # Common stop/function words
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'while',
                'that', 'this', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they',
                'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
                'what', 'which', 'who', 'whom', 'whose', 'about', 'up', 'also',
            }
            
            total_words_in_sent = 0
            content_words_in_sent = 0
            for sent in non_empty_sentences:
                sw = re.findall(r'[a-zA-Z]+', sent.lower())
                total_words_in_sent += len(sw)
                content_words_in_sent += sum(1 for w in sw if w not in stop_words and len(w) > 2)
            
            if total_words_in_sent > 0:
                content_words_ratio = content_words_in_sent / total_words_in_sent
        
        info_density_score = content_words_ratio * 10
        
        # ---- 6. Structural Coherence ----
        # Complete sentences, proper structure
        complete_sentences = len([s for s in non_empty_sentences if len(s.strip().split()) >= 4])
        
        # Check for repetition (bad sign)
        sentence_texts = [s.strip().lower() for s in non_empty_sentences if s.strip()]
        unique_sentences = len(set(sentence_texts))
        total_sentences = max(1, len(sentence_texts))
        repetition_ratio = unique_sentences / total_sentences
        
        # Check for excessive repetition of words
        word_list_lower = [w.lower() for w in re.findall(r'[a-zA-Z]+', response_stripped)]
        if word_list_lower:
            word_freq = Counter(word_list_lower)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            word_repetition_ratio = most_common_freq / len(word_list_lower)
        else:
            word_repetition_ratio = 0
        
        coherence_score = min(10, complete_sentences * 1.5) * repetition_ratio
        
        # Penalize high word repetition (indicates low-quality content)
        if word_repetition_ratio > 0.15:
            coherence_score *= 0.7
        
        # ---- 7. Relevance to Query ----
        # Check if response uses terms from the query (basic relevance)
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        response_words_set = set(word_list_lower)
        
        stop_words_set = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'can', 'you',
                          'what', 'how', 'where', 'when', 'why', 'who', 'which', 'this',
                          'that', 'with', 'for', 'from', 'about', 'have', 'has', 'had',
                          'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        query_content_words = query_words - stop_words_set
        if query_content_words:
            overlap = len(query_content_words & response_words_set) / len(query_content_words)
        else:
            overlap = 0.5
        
        relevance_score = overlap * 5  # Max 5 from this
        
        # ---- 8. Response Length Adequacy ----
        # Very short responses are usually low quality, but padding is bad too
        if word_count < 5:
            length_factor = 0.3
        elif word_count < 10:
            length_factor = 0.5
        elif word_count < 20:
            length_factor = 0.7
        elif word_count < 50:
            length_factor = 0.85
        elif word_count <= 200:
            length_factor = 1.0
        else:
            # Slight penalty for very long responses (might be padding)
            length_factor = max(0.7, 1.0 - (word_count - 200) / 1000)
        
        # ---- 9. Detect garbage/off-topic content ----
        # Check for HTML tags, code blocks, excessive special characters
        html_tags = len(re.findall(r'<[^>]+>', response_stripped))
        code_indicators = len(re.findall(r'(?:import |def |class |print\(|return )', response_stripped))
        
        # If the query doesn't ask for code/HTML but response has lots of it
        query_asks_code = bool(re.search(r'\b(?:code|html|program|script|function|tag)\b', query.lower()))
        
        garbage_penalty = 0
        if not query_asks_code:
            if html_tags > 3:
                garbage_penalty += min(3, html_tags * 0.5)
            if code_indicators > 2:
                garbage_penalty += min(3, code_indicators * 0.5)
        
        # Detect if response just repeats the query
        query_stripped = query.strip().lower()
        resp_lower_stripped = response_stripped.lower().strip()
        if resp_lower_stripped.startswith(query_stripped) and len(resp_lower_stripped) < len(query_stripped) * 1.5:
            garbage_penalty += 3
        
        # ---- 10. Unique vocabulary richness ----
        if word_list_lower and len(word_list_lower) > 5:
            vocab_richness = len(set(word_list_lower)) / len(word_list_lower)
        else:
            vocab_richness = 0.5
        
        vocab_score = vocab_richness * 5  # Max ~5
        
        # ---- Combine all scores ----
        raw_score = (
            entity_score * 0.20 +        # Named entities
            numeric_score * 0.15 +         # Numbers and data
            specificity_score * 0.15 +     # Specificity markers
            info_density_score * 0.15 +    # Content word density
            coherence_score * 0.10 +       # Structural coherence
            relevance_score * 0.10 +       # Query relevance
            vocab_score * 0.15             # Vocabulary richness
        )
        
        # Apply penalties
        raw_score -= vague_penalty * 0.5
        raw_score -= garbage_penalty
        
        # Apply length factor
        raw_score *= length_factor
        
        # Apply repetition penalty
        if repetition_ratio < 0.5:
            raw_score *= 0.5
        
        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        # Ensure very short, contentless responses score low
        if word_count <= 3 and entity_score == 0 and numeric_score == 0:
            final_score = min(final_score, 1.5)
        
        # Single word or near-empty
        if word_count <= 1:
            final_score = min(final_score, 0.5)
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            if response and len(response.strip()) > 0:
                wc = len(response.strip().split())
                return min(5.0, max(0.5, wc / 20.0 * 5.0))
            return 0.0
        except Exception:
            return 0.0