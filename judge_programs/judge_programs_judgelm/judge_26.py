def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a pattern-based approach
    that analyzes sentence structure, specificity signals, and reliability markers.
    
    This variant focuses on:
    - Named entity density (capitalized multi-word phrases)
    - Numeric/date specificity patterns
    - Citation and source reference patterns
    - Hallucination red flags (absolute claims, unsourced precision)
    - Appropriate uncertainty language
    - Sensationalism/conspiracy detection
    - Response coherence and structure quality
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses are generally low quality
        if len(response_stripped) < 5:
            return 0.5
        
        score = 5.0  # Start at midpoint
        
        words = response_stripped.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.5
        
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Named Entity Density (capitalized phrases)
        # Look for sequences of capitalized words that suggest proper nouns
        # ============================================================
        # Find capitalized word sequences (2+ words) not at sentence starts
        named_entity_pattern = re.findall(
            r'(?<=[.!?]\s|,\s|;\s|\()\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 
            response_stripped
        )
        # Also find standalone proper nouns mid-sentence
        mid_sentence_caps = re.findall(
            r'(?<=\s)([A-Z][a-z]{2,})', 
            ' '.join(response_stripped.split()[1:])  # skip first word
        )
        # Filter out common sentence starters
        common_starters = {'The', 'This', 'That', 'These', 'Those', 'However', 
                          'Moreover', 'Furthermore', 'Also', 'But', 'And', 'Or',
                          'While', 'When', 'Where', 'What', 'How', 'Why', 'Who',
                          'Once', 'After', 'Before', 'Since', 'Although', 'Though',
                          'Yet', 'Still', 'Here', 'There', 'Some', 'Many', 'Most',
                          'Each', 'Every', 'Any', 'All', 'Such', 'Other', 'Another',
                          'Sure', 'Yes', 'No', 'Comment', 'Note', 'Output', 'Input',
                          'Question', 'Answer', 'String', 'Percussion', 'Determine',
                          'Identify'}
        
        proper_nouns = [w for w in mid_sentence_caps if w not in common_starters]
        entity_density = len(proper_nouns) / max(word_count, 1)
        
        # Named entities boost (capped)
        entity_score = min(entity_density * 15, 1.5)
        score += entity_score
        
        # Multi-word named entities are stronger signals
        multi_word_entities = len(named_entity_pattern)
        score += min(multi_word_entities * 0.3, 1.0)
        
        # ============================================================
        # FEATURE 2: Numeric and Date Specificity
        # ============================================================
        # Dates (years, full dates)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', response_stripped)
        full_date_pattern = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            response_stripped
        )
        
        # Numbers with context (e.g., "9,733,557" or "37.7749 degrees")
        specific_numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', response_stripped)
        
        # Score for dates
        date_score = min(len(year_pattern) * 0.2 + len(full_date_pattern) * 0.4, 1.0)
        score += date_score
        
        # Score for numeric specificity (moderate - too many unsourced numbers is bad)
        num_count = len(specific_numbers)
        if num_count > 0 and num_count <= 5:
            score += min(num_count * 0.15, 0.6)
        elif num_count > 10:
            # Excessive unsourced numbers could be hallucination
            score -= 0.3
        
        # ============================================================
        # FEATURE 3: Citation and Source References
        # ============================================================
        citation_patterns = [
            r'according to\b',
            r'published (?:in|by)\b',
            r'(?:a |the )?study (?:by|from|in)\b',
            r'research (?:by|from|shows|suggests)\b',
            r'\b(?:source|reference|cited)\b',
            r'(?:University|Institute|Organization|Foundation)\b',
            r'(?:Journal|Magazine|Times|Post|Review)\b',
            r'(?:written|authored|compiled) by\b',
            r'(?:Hall of Fame|Museum|Library|Archive)\b',
        ]
        
        citation_count = 0
        for pat in citation_patterns:
            citation_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        score += min(citation_count * 0.25, 1.0)
        
        # ============================================================
        # FEATURE 4: Appropriate Uncertainty/Hedging Language
        # (Different from Variant 1 - uses ratio-based scoring and contextual analysis)
        # ============================================================
        uncertainty_phrases = [
            r'\bit is difficult to\b',
            r'\bcan be subjective\b',
            r'\bdepending on\b',
            r'\bgenerally\b',
            r'\btypically\b',
            r'\boften\b',
            r'\busually\b',
            r'\bmay (?:vary|differ|depend)\b',
            r'\bis (?:considered|regarded|known)\b',
            r'\bnot without controversy\b',
            r'\bhas been (?:criticized|debated|questioned)\b',
            r'\bapproximately\b',
            r'\babout\b',
            r'\bestimated\b',
            r'\bin some cases\b',
            r'\bcan vary\b',
            r'\btends to\b',
        ]
        
        uncertainty_count = 0
        for pat in uncertainty_phrases:
            uncertainty_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        # Moderate hedging is good; excessive hedging reduces confidence
        if uncertainty_count > 0 and uncertainty_count <= 4:
            score += uncertainty_count * 0.2
        elif uncertainty_count > 4:
            score += 0.8 - (uncertainty_count - 4) * 0.1
        
        # ============================================================
        # FEATURE 5: Hallucination Red Flags
        # ============================================================
        absolute_claims = re.findall(
            r'\b(?:always|never|every single|without exception|100%|guaranteed|undeniable|irrefutable|unquestionable)\b',
            response_stripped, re.IGNORECASE
        )
        score -= len(absolute_claims) * 0.3
        
        # Sensationalism indicators
        sensational_words = re.findall(
            r'\b(?:shocking|unbelievable|mind-blowing|explosive|bombshell|devastating|incredible truth|they don\'t want you to know|cover-up|conspiracy|suppressed|hidden truth|wake up|sheeple)\b',
            response_stripped, re.IGNORECASE
        )
        score -= len(sensational_words) * 0.5
        
        # Overly precise statistics without sources
        precise_stats = re.findall(r'\b\d{1,3}\.\d{2,}%\b', response_stripped)
        if precise_stats and citation_count == 0:
            score -= len(precise_stats) * 0.3
        
        # ============================================================
        # FEATURE 6: Response Structure and Coherence
        # ============================================================
        
        # Average sentence length (moderate is good)
        avg_words_per_sentence = word_count / sentence_count
        if 8 <= avg_words_per_sentence <= 25:
            score += 0.5
        elif avg_words_per_sentence < 4:
            score -= 0.5
        elif avg_words_per_sentence > 40:
            score -= 0.3
        
        # Response length adequacy relative to query complexity
        query_words = query.split()
        query_word_count = len(query_words)
        
        # Very short responses to complex queries
        length_ratio = word_count / max(query_word_count, 1)
        if length_ratio < 0.3:
            score -= 1.5
        elif length_ratio < 1.0:
            score -= 0.5
        elif 1.5 <= length_ratio <= 15:
            score += 0.5
        
        # Minimum substantive length
        if word_count < 3:
            score -= 2.0
        elif word_count < 10:
            score -= 0.5
        elif word_count >= 20:
            score += 0.3
        
        # ============================================================
        # FEATURE 7: Repetition Detection (hallucination signal)
        # ============================================================
        # Check for repeated phrases (3-gram repetition)
        if word_count >= 6:
            trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            if repeated_trigrams > 0:
                score -= min(repeated_trigrams * 0.4, 2.0)
        
        # Check for repeated sentences
        sentence_lower = [s.lower().strip() for s in sentences if len(s.strip()) > 10]
        if sentence_lower:
            unique_ratio = len(set(sentence_lower)) / len(sentence_lower)
            if unique_ratio < 0.5:
                score -= 1.5
            elif unique_ratio < 0.8:
                score -= 0.5
        
        # ============================================================
        # FEATURE 8: Relevance to Query (topic alignment)
        # ============================================================
        # Extract content words from query and check overlap
        stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
                      'they', 'them', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'to',
                      'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'between',
                      'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
                      'very', 'just', 'about', 'this', 'that', 'these', 'those', 'what',
                      'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'if', 'then',
                      'there', 'here', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
                      'also', 'make', 'like', 'know', 'get', 'give', 'go', 'come', 'take',
                      'want', 'tell', 'find', 'ask', 'use', 'say', 'help', 'think', 'new',
                      'different', 'way', 'many', 'much', 'well'}
        
        query_content_words = set(w.lower().strip('.,!?;:()[]{}"\'-') 
                                  for w in query_words 
                                  if w.lower().strip('.,!?;:()[]{}"\'-') not in stop_words
                                  and len(w.strip('.,!?;:()[]{}"\'-')) > 2)
        
        response_words_lower = set(w.lower().strip('.,!?;:()[]{}"\'-') for w in words)
        
        if query_content_words:
            topic_overlap = len(query_content_words & response_words_lower) / len(query_content_words)
            if topic_overlap >= 0.4:
                score += 0.8
            elif topic_overlap >= 0.2:
                score += 0.4
            elif topic_overlap == 0:
                score -= 1.0
        
        # ============================================================
        # FEATURE 9: Off-topic / Garbage Detection
        # ============================================================
        # HTML/code in non-code queries
        code_query = bool(re.search(r'\b(?:code|html|python|javascript|program|function|tag)\b', 
                                     query, re.IGNORECASE))
        
        html_tags = re.findall(r'<[^>]+>', response_stripped)
        code_blocks = re.findall(r'(?:import |def |class |function |var |let |const )', response_stripped)
        
        if not code_query:
            if len(html_tags) > 3:
                score -= 1.0
            if len(code_blocks) > 2:
                score -= 1.0
        
        # Random/garbage text detection
        # High ratio of special characters
        special_chars = sum(1 for c in response_stripped if c in '#$%^&*{}[]|\\<>')
        special_ratio = special_chars / max(len(response_stripped), 1)
        if special_ratio > 0.1:
            score -= 1.0
        
        # ============================================================
        # FEATURE 10: Explanatory Quality
        # ============================================================
        # Causal/explanatory connectors suggest structured reasoning
        explanatory_patterns = [
            r'\bbecause\b',
            r'\btherefore\b',
            r'\bas a result\b',
            r'\bthis (?:means|implies|suggests|indicates)\b',
            r'\bfor (?:example|instance)\b',
            r'\bsuch as\b',
            r'\bin other words\b',
            r'\bspecifically\b',
            r'\bincluding\b',
            r'\bhowever\b',
            r'\bon the other hand\b',
            r'\bwhile\b',
            r'\balthough\b',
            r'\bdespite\b',
        ]
        
        explanatory_count = 0
        for pat in explanatory_patterns:
            explanatory_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        score += min(explanatory_count * 0.15, 0.8)
        
        # ============================================================
        # FEATURE 11: Question echoing without answering
        # ============================================================
        # If response just repeats the query without adding info
        query_lower = query.lower().strip()
        response_lower = response_stripped.lower().strip()
        
        if len(query_lower) > 10 and query_lower in response_lower:
            # Query is contained in response - check if much else is added
            remaining = response_lower.replace(query_lower, '').strip()
            if len(remaining) < 20:
                score -= 1.0
        
        # ============================================================
        # FEATURE 12: Completeness signals
        # ============================================================
        # Truncated response detection
        if response_stripped[-1] not in '.!?"\')]}' and word_count > 10:
            # Might be truncated, slight penalty
            score -= 0.3
        
        # Response that trails off with repetitive content
        if word_count > 20:
            last_quarter = words[-(word_count // 4):]
            first_quarter = words[:word_count // 4]
            last_set = set(w.lower() for w in last_quarter)
            first_set = set(w.lower() for w in first_quarter)
            if last_set and first_set:
                overlap = len(last_set & first_set) / max(len(last_set), 1)
                if overlap > 0.8 and word_count > 30:
                    score -= 0.5
        
        # ============================================================
        # FEATURE 13: Factual language patterns (verb tenses, definitive statements)
        # ============================================================
        # Informative verb patterns
        informative_verbs = re.findall(
            r'\b(?:contains|consists|comprises|refers to|defined as|known as|located in|founded|established|created|developed|invented|discovered|published|written)\b',
            response_stripped, re.IGNORECASE
        )
        score += min(len(informative_verbs) * 0.2, 0.8)
        
        # ============================================================
        # Clamp final score
        # ============================================================
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 3.0