def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant focuses on:
    - Information density and segmentation ratio
    - Transition/connector word usage for logical flow
    - Hierarchical depth detection (nested structures)
    - Visual variety score (mixing of different formatting elements)
    - Sentence-level structural patterns (topic sentences, supporting details)
    - Redundancy/repetition penalty
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        score = 0.0
        
        # === 1. INFORMATION DENSITY & SEGMENTATION RATIO ===
        # Measure how well content is broken into digestible segments
        # Count distinct "segments" (separated by newlines, punctuation breaks, list items)
        segments = re.split(r'\n\s*\n|\n(?=[\-\*\d•►▪])|(?<=\.)\s{2,}', response)
        segments = [s.strip() for s in segments if s.strip()]
        num_segments = len(segments)
        
        total_chars = len(response)
        total_words = len(response.split())
        
        if total_words == 0:
            return 0.0
        
        # Average segment length in words - sweet spot is 20-80 words per segment
        avg_segment_words = total_words / max(num_segments, 1)
        if 15 <= avg_segment_words <= 80:
            score += 8.0
        elif 10 <= avg_segment_words <= 120:
            score += 5.0
        elif avg_segment_words > 120:
            score += 1.0  # wall of text penalty
        else:
            score += 3.0  # too fragmented
        
        # Bonus for having multiple segments (indicates structure)
        if num_segments >= 4:
            score += 6.0
        elif num_segments >= 3:
            score += 4.0
        elif num_segments >= 2:
            score += 2.0
        
        # === 2. TRANSITION & CONNECTOR WORDS (logical flow) ===
        transition_patterns = [
            r'\b(however|moreover|furthermore|additionally|consequently)\b',
            r'\b(in addition|on the other hand|as a result|for example|for instance)\b',
            r'\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly|next|then)\b',
            r'\b(in contrast|similarly|likewise|nevertheless|nonetheless)\b',
            r'\b(therefore|thus|hence|accordingly|meanwhile)\b',
            r'\b(in summary|in conclusion|to summarize|overall|in particular)\b',
            r'\b(specifically|notably|importantly|significantly)\b',
            r'\b(while|whereas|although|despite|instead)\b',
        ]
        
        response_lower = response.lower()
        transition_count = 0
        for pattern in transition_patterns:
            transition_count += len(re.findall(pattern, response_lower))
        
        # Normalize by word count - ideal is roughly 1 transition per 30-50 words
        transition_density = transition_count / max(total_words, 1) * 100
        if 1.5 <= transition_density <= 6.0:
            score += 8.0
        elif 0.5 <= transition_density <= 8.0:
            score += 5.0
        elif transition_count > 0:
            score += 2.0
        
        # === 3. HIERARCHICAL DEPTH DETECTION ===
        # Look for nested structures, indentation levels, sub-items
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        indent_levels = set()
        for line in non_empty_lines:
            leading_spaces = len(line) - len(line.lstrip())
            # Also check for tab-based indentation
            leading_tabs = len(line) - len(line.lstrip('\t'))
            indent_level = leading_spaces // 2 + leading_tabs
            indent_levels.add(min(indent_level, 5))
        
        # Headers at different levels (markdown style)
        header_levels = set()
        for line in non_empty_lines:
            stripped = line.strip()
            h_match = re.match(r'^(#{1,6})\s', stripped)
            if h_match:
                header_levels.add(len(h_match.group(1)))
            # Also detect bold headers or colon-terminated headers
            if re.match(r'^\*\*[^*]+\*\*\s*$', stripped):
                header_levels.add(1)
            if re.match(r'^[A-Z][^.!?]*:\s*$', stripped):
                header_levels.add(1)
        
        hierarchy_depth = len(indent_levels) + len(header_levels)
        if hierarchy_depth >= 4:
            score += 8.0
        elif hierarchy_depth >= 3:
            score += 6.0
        elif hierarchy_depth >= 2:
            score += 4.0
        elif hierarchy_depth >= 1:
            score += 1.0
        
        # === 4. VISUAL VARIETY SCORE ===
        # Count distinct formatting elements used
        formatting_elements = 0
        
        # Numbered lists (various formats)
        has_numbered = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        if has_numbered:
            formatting_elements += 1
        
        # Bullet points (various styles)
        has_bullets = bool(re.search(r'(?:^|\n)\s*[\-\*•►▪◦‣]\s', response))
        if has_bullets:
            formatting_elements += 1
        
        # Headers (markdown or otherwise)
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,6}\s', response)) or \
                      bool(re.search(r'(?:^|\n)\s*\*\*[^*]+\*\*\s*(?:\n|$)', response))
        if has_headers:
            formatting_elements += 1
        
        # Bold text (inline, not headers)
        has_bold_inline = bool(re.search(r'(?<!\n)\*\*[^*]+\*\*(?!\s*$)', response))
        if has_bold_inline:
            formatting_elements += 1
        
        # Italic text
        has_italic = bool(re.search(r'(?<!\*)\*[^*\n]+\*(?!\*)', response))
        if has_italic:
            formatting_elements += 1
        
        # Code blocks or inline code
        has_code = bool(re.search(r'`[^`]+`|```', response))
        if has_code:
            formatting_elements += 1
        
        # Paragraph breaks (double newline)
        has_para_breaks = bool(re.search(r'\n\s*\n', response))
        if has_para_breaks:
            formatting_elements += 1
        
        # Colon-based definitions or key-value pairs
        has_definitions = bool(re.search(r'(?:^|\n)[A-Z][^:.\n]{2,30}:\s+\S', response))
        if has_definitions:
            formatting_elements += 1
        
        # Score visual variety
        score += min(formatting_elements * 2.0, 10.0)
        
        # === 5. SENTENCE-LEVEL STRUCTURAL PATTERNS ===
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences >= 2:
            # Sentence length variance - good writing has varied sentence lengths
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            if 0.2 <= cv <= 0.8:
                score += 6.0  # Good variety
            elif 0.1 <= cv <= 1.0:
                score += 3.0
            else:
                score += 1.0
            
            # Check if first sentence of each paragraph acts as topic sentence
            # (typically shorter or contains key terms from query)
            if num_segments >= 2:
                query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
                topic_sentence_score = 0
                for seg in segments:
                    seg_sentences = re.split(r'(?<=[.!?])\s+', seg)
                    if seg_sentences:
                        first_sent = seg_sentences[0].lower()
                        first_words = set(re.findall(r'\b\w{4,}\b', first_sent))
                        if query_words and first_words & query_words:
                            topic_sentence_score += 1
                
                score += min(topic_sentence_score * 1.5, 5.0)
        else:
            # Very few sentences
            score += 1.0
        
        # === 6. REDUNDANCY / REPETITION PENALTY ===
        # Detect repeated phrases (n-grams)
        words = re.findall(r'\b\w+\b', response_lower)
        
        if len(words) >= 6:
            # Check trigram repetition
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            
            # Check for repeated sentences
            sentence_set = set()
            duplicate_sentences = 0
            for s in sentences:
                normalized = re.sub(r'\s+', ' ', s.lower().strip())
                if normalized in sentence_set:
                    duplicate_sentences += 1
                sentence_set.add(normalized)
            
            # Check word-level repetition (same word used excessively)
            if len(words) > 10:
                word_counts = Counter(words)
                # Exclude common stop words
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                             'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                             'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                             'and', 'or', 'but', 'not', 'that', 'this', 'it', 'its',
                             'as', 'if', 'than', 'so', 'no', 'up', 'out', 'about'}
                content_words = {w: c for w, c in word_counts.items() 
                               if w not in stop_words and len(w) > 2}
                if content_words:
                    max_freq = max(content_words.values())
                    if max_freq > len(words) * 0.15:
                        score -= 5.0  # Excessive repetition of single word
            
            repetition_penalty = repeated_trigrams * 1.5 + duplicate_sentences * 3.0
            score -= min(repetition_penalty, 12.0)
        
        # === 7. RESPONSE LENGTH APPROPRIATENESS ===
        # Reward responses that are substantive but not bloated
        query_words_count = len(query.split()) if query else 5
        
        # Relative length - response should be meaningfully longer than query for most tasks
        length_ratio = total_words / max(query_words_count, 1)
        
        if 2.0 <= length_ratio <= 30.0:
            score += 4.0
        elif 1.0 <= length_ratio <= 50.0:
            score += 2.0
        elif length_ratio < 0.5:
            score -= 2.0  # Too short
        
        # Absolute length bonuses
        if 30 <= total_words <= 500:
            score += 3.0
        elif 15 <= total_words <= 30:
            score += 1.0
        elif total_words < 10:
            score -= 3.0
        
        # === 8. OPENING AND CLOSING STRUCTURE ===
        # Good responses often have a clear opening and closing
        first_sentence = sentences[0] if sentences else ""
        last_sentence = sentences[-1] if sentences else ""
        
        # Check for a clear introductory/framing sentence
        intro_patterns = [
            r'^(here|the|this|there|when|a |an )',
            r'^(to |in order|the following|below)',
        ]
        has_intro = any(re.match(p, first_sentence.lower()) for p in intro_patterns)
        if has_intro and num_sentences >= 3:
            score += 2.0
        
        # Check for concluding language
        conclusion_patterns = [
            r'\b(in conclusion|overall|in summary|to summarize|finally)\b',
            r'\b(therefore|thus|as a result|this shows|this demonstrates)\b',
        ]
        has_conclusion = any(re.search(p, last_sentence.lower()) for p in conclusion_patterns)
        if has_conclusion and num_sentences >= 3:
            score += 2.0
        
        # === 9. WALL OF TEXT PENALTY ===
        # If response is long but has no structural breaks
        if total_words > 50 and num_segments == 1 and not has_bullets and not has_numbered and not has_headers:
            # It's a wall of text
            wall_penalty = min((total_words - 50) / 30.0, 8.0)
            score -= wall_penalty
        
        # === NORMALIZE FINAL SCORE ===
        # Clamp to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 5.0