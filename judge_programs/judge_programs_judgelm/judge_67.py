def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant focuses on:
    - Information density and signal-to-noise ratio
    - Hierarchical structure detection (nested organization)
    - Repetition/redundancy detection
    - Sentence length variance (good writing mixes lengths)
    - Response completeness (not truncated)
    - Clean formatting vs noisy/broken formatting
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query else ""
        
        score = 5.0  # Start at midpoint
        
        # === 1. Response length adequacy ===
        resp_len = len(response)
        word_count = len(response.split())
        
        # Very short responses - penalize heavily
        if word_count <= 2:
            score -= 3.0
        elif word_count <= 5:
            score -= 1.5
        elif word_count >= 15:
            score += 0.5
        
        # === 2. Sentence analysis ===
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Sentence length variance - good writing has varied sentence lengths
        if num_sentences >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len if mean_len > 0 else 0
                # Coefficient of variation between 0.3 and 0.8 is ideal
                if 0.2 <= cv <= 1.0:
                    score += 0.8
                elif cv > 0:
                    score += 0.3
        
        # === 3. Repetition / redundancy detection ===
        # Check for repeated lines or phrases
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if len(lines) >= 2:
            line_counter = Counter(lines)
            most_common_count = line_counter.most_common(1)[0][1] if line_counter else 1
            if most_common_count > 2:
                score -= min(3.0, most_common_count * 0.7)
            elif most_common_count > 1:
                score -= 0.5
        
        # Check for repeated n-grams (trigrams)
        words_lower = response.lower().split()
        if len(words_lower) >= 6:
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                repetition_ratio = repeated_trigrams / max(len(trigram_counts), 1)
                if repetition_ratio > 0.15:
                    score -= 2.0
                elif repetition_ratio > 0.05:
                    score -= 1.0
        
        # === 4. Structural hierarchy detection ===
        # Detect different levels of organization
        has_headers = bool(re.search(r'(^|\n)(#{1,6}\s|[A-Z][A-Za-z\s]{2,30}:\s*\n|[A-Z][A-Za-z\s]{2,30}:\s*$)', response, re.MULTILINE))
        has_numbered_list = bool(re.search(r'(^|\n)\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bullet_list = bool(re.search(r'(^|\n)\s*[-*•]\s', response, re.MULTILINE))
        has_labeled_items = bool(re.search(r'(^|\n)\s*[A-Z][\w\s]+:\s+\S', response, re.MULTILINE))
        
        structure_elements = sum([has_headers, has_numbered_list, has_bullet_list, has_labeled_items])
        
        # Reward structural elements proportional to response length
        if word_count > 20:
            score += min(1.5, structure_elements * 0.5)
        elif structure_elements > 0:
            score += 0.3
        
        # === 5. Paragraph structure analysis ===
        # Split by double newlines or single newlines with content
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if word_count > 50:
            # Long responses should have paragraph breaks
            if num_paragraphs == 1:
                # Wall of text penalty
                score -= 1.5
            elif num_paragraphs >= 2:
                # Good paragraph separation
                avg_para_words = word_count / num_paragraphs
                if 15 <= avg_para_words <= 80:
                    score += 1.0
                elif avg_para_words > 5:
                    score += 0.3
        
        # === 6. Signal-to-noise ratio ===
        # Detect noisy/broken content
        noise_patterns = [
            (r'(Input:|Output:){2,}', -1.5),  # Repeated Input/Output markers
            (r'(Question:|Answer:){2,}', -1.0),  # Repeated Q&A markers
            (r'\n\s*\n\s*\n\s*\n', -0.5),  # Excessive blank lines
            (r'(.)\1{5,}', -1.0),  # Repeated characters (e.g., "......")
            (r'```[\s\S]{0,10}```', -0.3),  # Empty code blocks
            (r'<[^>]+>[^<]*<[^>]+>', -0.3),  # Raw HTML tags (unless query asks for HTML)
        ]
        
        for pattern, penalty in noise_patterns:
            if re.search(pattern, response):
                # Don't penalize HTML if query asks for HTML
                if '<' in pattern and ('html' in query.lower() or 'tag' in query.lower()):
                    continue
                score += penalty
        
        # === 7. Completeness detection ===
        # Check if response appears truncated
        truncation_indicators = [
            response.endswith('...'),
            response.endswith('..'),
            response[-1:] not in '.!?"\')>}' + '\n' and word_count > 30 and not has_bullet_list and not has_numbered_list,
            bool(re.search(r'\b(the|a|an|is|are|was|were|to|of|in|and|or|but|for|with)\s*$', response, re.IGNORECASE)),
        ]
        
        truncation_count = sum(truncation_indicators)
        if truncation_count >= 2:
            score -= 1.0
        elif truncation_count == 1:
            score -= 0.3
        
        # === 8. Content coherence - topic sentence detection ===
        if num_sentences >= 2:
            # Check if first sentence relates to query
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            first_sent_words = set(re.findall(r'\b\w{3,}\b', sentences[0].lower()))
            overlap = len(query_words & first_sent_words)
            if overlap >= 2:
                score += 0.5
            elif overlap >= 1:
                score += 0.2
        
        # === 9. Clean formatting bonus ===
        # Responses that are well-proportioned and clean
        if 10 <= word_count <= 300:
            # Check for consistent formatting
            line_lengths = [len(l) for l in lines if l]
            if line_lengths:
                max_line = max(line_lengths)
                min_line = min(line_lengths) if len(line_lengths) > 1 else max_line
                
                # Not all lines the same length (not just copy-paste)
                if len(set(line_lengths)) > 1 or len(line_lengths) == 1:
                    score += 0.3
        
        # === 10. Penalize responses that are just echoing/repeating the query ===
        if query and response:
            query_norm = query.lower().strip()
            resp_norm = response.lower().strip()
            if resp_norm.startswith(query_norm) and len(resp_norm) < len(query_norm) * 1.5:
                score -= 1.5
        
        # === 11. Detect "garbage" responses ===
        # Single word or single character responses
        if response.strip() in ['.', ',', '-', '!', '?', 'no', 'yes', 'n/a']:
            return 1.0
        
        # Mostly non-alphanumeric
        alnum_chars = sum(1 for c in response if c.isalnum())
        total_chars = max(len(response), 1)
        alnum_ratio = alnum_chars / total_chars
        if alnum_ratio < 0.3 and word_count < 10:
            score -= 2.0
        
        # === 12. Appropriate response length relative to query complexity ===
        query_word_count = len(query.split()) if query else 0
        
        # Complex queries (longer) deserve more detailed responses
        if query_word_count > 20 and word_count < 10:
            score -= 1.0
        
        # Simple queries are fine with concise answers
        if query_word_count <= 10 and 5 <= word_count <= 50:
            score += 0.3
        
        # === 13. Detect off-topic rambling ===
        # If response has question-answer pairs that seem auto-generated
        qa_pattern_count = len(re.findall(r'(Question:|Q:|Input:|Output:)', response))
        if qa_pattern_count > 3:
            score -= 2.0
        elif qa_pattern_count > 1 and 'Question' not in query and 'Input' not in query:
            score -= 1.0
        
        # Clamp score to [0, 10]
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 3.0