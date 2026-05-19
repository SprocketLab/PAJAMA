def judging_function(query, response):
    """
    Evaluates response quality based on Evidence Density and Specificity.
    
    Focuses on: specific examples, concrete data points, named entities,
    precise numbers, real-world references, and actionable details.
    Penalizes vague, generic filler responses.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        query = query.strip() if query else ""
        
        if len(response) == 0:
            return 0.0
        
        # Very short responses are almost always low quality
        if len(response) < 5:
            return 0.5
        
        score = 0.0
        words = response.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # ============================================================
        # FEATURE 1: Named Entities / Proper Nouns Detection
        # Words starting with uppercase that aren't sentence starters
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentence_starters = set()
        for sent in sentences:
            sent = sent.strip()
            if sent:
                first_word = sent.split()[0] if sent.split() else ""
                sentence_starters.add(first_word)
        
        capitalized_words = []
        for i, w in enumerate(words):
            clean_w = w.strip(string.punctuation)
            if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                if clean_w not in sentence_starters or i > 0:
                    capitalized_words.append(clean_w)
        
        # Filter out common non-entity capitalized words
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'It', 'Is', 'Are', 
                       'Was', 'Were', 'Has', 'Have', 'Had', 'Will', 'Would', 'Could',
                       'Should', 'May', 'Might', 'Can', 'Do', 'Does', 'Did', 'A', 'An',
                       'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From',
                       'But', 'And', 'Or', 'Not', 'No', 'Yes', 'If', 'So', 'As',
                       'He', 'She', 'We', 'They', 'You', 'I', 'My', 'Your', 'His',
                       'Her', 'Its', 'Our', 'Their', 'Some', 'Many', 'Most', 'All',
                       'Each', 'Every', 'Any', 'Both', 'Few', 'Several', 'Such',
                       'Here', 'There', 'Where', 'When', 'How', 'What', 'Who', 'Why',
                       'However', 'Also', 'Although', 'Because', 'Since', 'While',
                       'After', 'Before', 'During', 'Until', 'Unless', 'Once',
                       'Sure', 'Comment', 'Output', 'Input', 'Question', 'Answer',
                       'Note', 'String', 'Percussion', 'Determine', 'Identify'}
        
        named_entities = [w for w in capitalized_words if w not in common_caps]
        # Multi-word entities (consecutive capitalized words)
        entity_count = len(set(named_entities))
        named_entity_density = entity_count / max(word_count, 1)
        named_entity_score = min(named_entity_density * 40, 2.0)
        score += named_entity_score
        
        # ============================================================
        # FEATURE 2: Numbers and Quantitative Data
        # ============================================================
        # Match various number patterns: integers, decimals, percentages, dates, ranges
        number_patterns = re.findall(
            r'\b\d[\d,]*\.?\d*%?\b|\b\d{4}\b|\$[\d,]+\.?\d*|\b\d+(?:st|nd|rd|th)\b',
            response
        )
        number_count = len(number_patterns)
        number_density = number_count / max(word_count, 1)
        number_score = min(number_density * 30, 1.5)
        score += number_score
        
        # Bonus for having at least some numbers
        if number_count >= 1:
            score += 0.3
        if number_count >= 3:
            score += 0.3
        
        # ============================================================
        # FEATURE 3: Specific Location/Place References
        # ============================================================
        location_indicators = [
            r'\b(?:in|at|from|near|of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'\b(?:New York|Los Angeles|London|Tokyo|Paris|Berlin|Moscow|Beijing|'
            r'Washington|Chicago|San Francisco|Boston|Seattle|Cooperstown|'
            r'California|Texas|Florida|England|France|Germany|Japan|China|India|'
            r'Africa|Europe|Asia|America|Australia)\b',
        ]
        location_count = 0
        for pattern in location_indicators:
            location_count += len(re.findall(pattern, response))
        location_score = min(location_count * 0.3, 1.5)
        score += location_score
        
        # ============================================================
        # FEATURE 4: Specificity Vocabulary (concrete vs vague language)
        # ============================================================
        response_lower = response.lower()
        
        # Vague/filler phrases to penalize
        vague_phrases = [
            'many people think', 'it depends', 'there are various factors',
            'there are many', 'various reasons', 'a lot of people',
            'some people say', 'it is important to', 'in general',
            'you can find', 'there are several', 'it varies',
            'many factors', 'depends on many', 'a number of',
            'it is difficult to say', 'hard to say', 'could be many',
            'various ways', 'many ways', 'different ways',
            'you should consider', 'keep in mind', 'it is worth noting',
            'generally speaking', 'broadly speaking', 'in most cases',
            'tends to vary', 'subjective', 'depending on interpretation',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += response_lower.count(phrase)
        
        vague_penalty = min(vague_count * 0.5, 2.0)
        score -= vague_penalty
        
        # ============================================================
        # FEATURE 5: Technical/Domain-Specific Terms
        # ============================================================
        # Words that are longer and less common tend to be more specific
        words_clean = [w.strip(string.punctuation).lower() for w in words if w.strip(string.punctuation)]
        
        # Average word length as proxy for specificity
        if words_clean:
            avg_word_len = sum(len(w) for w in words_clean) / len(words_clean)
            # Reward slightly longer average word length (more specific vocabulary)
            if avg_word_len > 5.0:
                score += min((avg_word_len - 5.0) * 0.3, 0.6)
        
        # Unique word ratio (vocabulary richness)
        if words_clean:
            unique_ratio = len(set(words_clean)) / len(words_clean)
            score += unique_ratio * 0.8
        
        # ============================================================
        # FEATURE 6: Structural Evidence (lists, enumerations, examples)
        # ============================================================
        # Bullet points, numbered lists, colons introducing examples
        list_markers = len(re.findall(r'(?:^|\n)\s*[\-\*•]\s', response))
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        example_markers = len(re.findall(r'\b(?:for example|for instance|such as|e\.g\.|i\.e\.|specifically|namely|including)\b', response_lower))
        
        structure_score = min((list_markers + numbered_items) * 0.2 + example_markers * 0.4, 1.5)
        score += structure_score
        
        # ============================================================
        # FEATURE 7: Quotation marks (citing specific text/quotes)
        # ============================================================
        quotes = re.findall(r'["\u201c\u201d\u2018\u2019].*?["\u201c\u201d\u2018\u2019]', response)
        meaningful_quotes = [q for q in quotes if len(q) > 5]
        quote_score = min(len(meaningful_quotes) * 0.3, 0.6)
        score += quote_score
        
        # ============================================================
        # FEATURE 8: Response Length and Content Density
        # ============================================================
        # Reasonable length bonus (not too short, not excessively padded)
        if word_count >= 10:
            score += 0.8
        if word_count >= 25:
            score += 0.5
        if word_count >= 50:
            score += 0.3
        if word_count < 5:
            score -= 1.5
        
        # ============================================================
        # FEATURE 9: Relevance to Query
        # ============================================================
        if query:
            query_words = set(query.lower().split())
            # Remove stop words from query
            stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'as', 'into', 'through', 'during', 'before', 'after',
                         'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
                         'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we',
                         'they', 'this', 'that', 'these', 'those', 'what', 'which',
                         'who', 'whom', 'how', 'where', 'when', 'why',
                         'if', 'then', 'than', 'also', 'just', 'about', 'up', 'out'}
            
            query_content_words = query_words - stop_words
            if query_content_words:
                response_words_set = set(words_clean)
                overlap = len(query_content_words & response_words_set)
                relevance_ratio = overlap / len(query_content_words)
                score += relevance_ratio * 1.5
        
        # ============================================================
        # FEATURE 10: Detect Garbage / Repetition / HTML/Code Noise
        # ============================================================
        # Repetition detection
        if word_count > 10:
            bigrams = [' '.join(words_clean[i:i+2]) for i in range(len(words_clean)-1)]
            if bigrams:
                bigram_counts = Counter(bigrams)
                most_common_freq = bigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(bigrams)
                if repetition_ratio > 0.15:
                    score -= min(repetition_ratio * 5, 2.0)
        
        # HTML/code noise detection
        html_tags = len(re.findall(r'<[^>]+>', response))
        if html_tags > 2:
            score -= min(html_tags * 0.2, 1.5)
        
        # Code block detection (not always bad, but if query doesn't ask for code)
        code_indicators = len(re.findall(r'(?:import |def |class |print\(|return )', response))
        query_asks_code = bool(re.search(r'\b(?:code|program|function|script|implement|write)\b', query.lower())) if query else False
        if code_indicators > 2 and not query_asks_code:
            score -= min(code_indicators * 0.3, 1.0)
        
        # ============================================================
        # FEATURE 11: Actionable Details
        # ============================================================
        actionable_patterns = [
            r'\b(?:visit|go to|check|look at|search for|read|call|contact|use|try|apply|download)\b',
            r'\b(?:step \d|first|second|third|next|then|finally|start by|begin with)\b',
        ]
        actionable_count = 0
        for pattern in actionable_patterns:
            actionable_count += len(re.findall(pattern, response_lower))
        actionable_score = min(actionable_count * 0.2, 1.0)
        score += actionable_score
        
        # ============================================================
        # FEATURE 12: Direct Answer Quality
        # ============================================================
        # Bonus for responses that get to the point quickly
        # Check if first sentence contains substantive content
        first_sentence = sentences[0].strip() if sentences else ""
        if first_sentence:
            first_words = first_sentence.split()
            if len(first_words) >= 3:
                # Check for named entities or numbers in first sentence
                first_caps = sum(1 for w in first_words[1:] if w[0].isupper() and w.strip(string.punctuation) not in common_caps) if len(first_words) > 1 else 0
                first_nums = len(re.findall(r'\d+', first_sentence))
                if first_caps > 0 or first_nums > 0:
                    score += 0.5
        
        # ============================================================
        # FEATURE 13: Completeness - Does response trail off?
        # ============================================================
        if response.rstrip()[-1:] not in '.!?")\']}>':
            # Response might be cut off
            if len(response) > 100:
                score -= 0.3
        
        # ============================================================
        # Normalize to 0-10 range
        # ============================================================
        # Base adjustment: shift score to center around 5
        score += 3.0
        
        # Clamp to [0.5, 10]
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return middle score on any error
        try:
            if not response or len(response.strip()) < 3:
                return 1.0
            return 4.0
        except:
            return 4.0