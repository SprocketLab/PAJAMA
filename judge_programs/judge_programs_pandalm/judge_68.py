def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality.
    
    This variant focuses on:
    - Information density and segmentation ratio
    - Hierarchical depth detection (nested structures)
    - Sentence-level coherence flow (topic sentence patterns)
    - Visual separation metrics (whitespace-to-content ratio)
    - Structural diversity index (variety of formatting elements used)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        score = 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(lines)
        non_empty_count = len(non_empty_lines)
        
        if non_empty_count == 0:
            return 0.0
        
        # ---- Feature 1: Structural Diversity Index ----
        # Count how many DIFFERENT structural element types are present
        structural_elements = set()
        
        # Check for numbered lists (various formats)
        if re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE):
            structural_elements.add('numbered_list')
        
        # Check for bullet points (various markers)
        if re.search(r'^\s*[-•*▪◦►]\s', response, re.MULTILINE):
            structural_elements.add('bullet_list')
        
        # Check for headers (markdown or all-caps short lines)
        if re.search(r'^\s*#{1,6}\s+\S', response, re.MULTILINE):
            structural_elements.add('markdown_header')
        if re.search(r'^[A-Z][A-Z\s]{3,40}:?\s*$', response, re.MULTILINE):
            structural_elements.add('caps_header')
        
        # Check for bold/italic emphasis
        if re.search(r'\*\*[^*]+\*\*', response) or re.search(r'__[^_]+__', response):
            structural_elements.add('bold')
        if re.search(r'(?<!\*)\*[^*]+\*(?!\*)', response) or re.search(r'(?<!_)_[^_]+_(?!_)', response):
            structural_elements.add('italic')
        
        # Check for code blocks or inline code
        if re.search(r'```', response) or re.search(r'`[^`]+`', response):
            structural_elements.add('code')
        
        # Check for colons used as label-value pairs
        if re.search(r'^\s*\w[\w\s]{0,30}:\s+\S', response, re.MULTILINE):
            structural_elements.add('label_value')
        
        # Check for parenthetical asides
        if re.search(r'\([^)]{5,}\)', response):
            structural_elements.add('parenthetical')
        
        # Check for quotation marks (quoted material)
        if re.search(r'"[^"]{10,}"', response):
            structural_elements.add('quotation')
        
        diversity_score = min(len(structural_elements) * 1.5, 10.0)
        score += diversity_score
        
        # ---- Feature 2: Segmentation Ratio ----
        # Ratio of meaningful segments to total content length
        # Good responses break content into digestible chunks
        
        words = response.split()
        word_count = len(words)
        
        if word_count < 3:
            return max(1.0, score)
        
        # Count distinct segments (separated by blank lines, list items, headers)
        segments = re.split(r'\n\s*\n|\n\s*[-•*]\s|\n\s*\d+[\.\)]\s|\n\s*#{1,6}\s', response)
        segments = [s.strip() for s in segments if s.strip()]
        num_segments = len(segments)
        
        # Ideal: roughly one segment per 30-80 words for longer responses
        if word_count > 50:
            ideal_segments = word_count / 50.0
            seg_ratio = num_segments / ideal_segments if ideal_segments > 0 else 0
            # Score peaks at ratio ~1.0, decays for too few or too many
            segmentation_score = 10.0 * math.exp(-0.5 * (seg_ratio - 1.0) ** 2)
        elif word_count > 20:
            # For medium responses, having 2-3 segments is good
            if num_segments >= 2:
                segmentation_score = 7.0
            else:
                segmentation_score = 4.0
        else:
            # Short responses don't need segmentation
            segmentation_score = 5.0
        
        score += segmentation_score
        
        # ---- Feature 3: Sentence-level coherence flow ----
        # Check for topic sentences, connective tissue between ideas
        sentences = re.split(r'[.!?]+\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences >= 2:
            # Check sentence length variance (moderate variance = good structure)
            sent_lengths = [len(s.split()) for s in sentences]
            avg_len = sum(sent_lengths) / len(sent_lengths)
            
            if avg_len > 0:
                variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                cv = math.sqrt(variance) / avg_len  # coefficient of variation
                
                # Moderate CV (0.2-0.6) suggests varied but controlled sentence structure
                if 0.15 <= cv <= 0.7:
                    flow_score = 8.0
                elif cv < 0.15:
                    # Very uniform - monotonous
                    flow_score = 4.0
                else:
                    # Very high variance - chaotic
                    flow_score = 3.0
            else:
                flow_score = 2.0
            
            # Bonus for sentences that start with different words (variety)
            first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
            unique_starters = len(set(first_words))
            starter_ratio = unique_starters / len(first_words) if first_words else 0
            flow_score += starter_ratio * 3.0
            
        else:
            flow_score = 3.0
        
        score += min(flow_score, 12.0)
        
        # ---- Feature 4: Visual Separation Quality ----
        # Analyze whitespace patterns - not just presence but quality
        
        if total_lines > 1:
            empty_lines = total_lines - non_empty_count
            empty_ratio = empty_lines / total_lines
            
            # Check for consistent indentation patterns
            indentation_levels = set()
            for line in non_empty_lines:
                leading = len(line) - len(line.lstrip())
                indentation_levels.add(leading)
            
            # Multiple indentation levels suggest hierarchical structure
            indent_depth = len(indentation_levels)
            
            visual_score = 0.0
            
            # Reward moderate whitespace (not too much, not too little)
            if 0.1 <= empty_ratio <= 0.4:
                visual_score += 5.0
            elif 0.0 < empty_ratio < 0.1:
                visual_score += 2.0
            elif empty_ratio > 0.4:
                visual_score += 1.0  # Too much whitespace
            else:
                visual_score += 1.0  # No whitespace at all
            
            # Reward hierarchical indentation
            if indent_depth >= 3:
                visual_score += 4.0
            elif indent_depth == 2:
                visual_score += 2.5
            else:
                visual_score += 0.5
            
        else:
            # Single line - check if it's a wall of text
            if word_count > 80:
                visual_score = 0.0  # Wall of text penalty
            elif word_count > 40:
                visual_score = 2.0
            else:
                visual_score = 4.0  # Short single-line is acceptable
        
        score += min(visual_score, 10.0)
        
        # ---- Feature 5: Information Density Distribution ----
        # Check if information is evenly distributed or front/back-loaded
        
        if num_sentences >= 4:
            third = num_sentences // 3
            if third > 0:
                first_third_avg = sum(len(s.split()) for s in sentences[:third]) / third
                mid_third_avg = sum(len(s.split()) for s in sentences[third:2*third]) / third
                last_third_avg = sum(len(s.split()) for s in sentences[2*third:]) / max(1, num_sentences - 2*third)
                
                avgs = [first_third_avg, mid_third_avg, last_third_avg]
                overall_avg = sum(avgs) / 3
                
                if overall_avg > 0:
                    # Check balance - lower deviation = more balanced
                    deviation = sum(abs(a - overall_avg) for a in avgs) / (3 * overall_avg)
                    density_score = max(0, 8.0 * (1.0 - deviation))
                else:
                    density_score = 2.0
            else:
                density_score = 4.0
        else:
            density_score = 4.0
        
        score += density_score
        
        # ---- Feature 6: Completeness Signals ----
        # Check for structural completeness (intro, body, conclusion patterns)
        
        completeness_score = 0.0
        
        # Opening patterns
        opening_patterns = [
            r'^(here|the following|below|this|let me|i\'ll|to|in order)',
            r'^(there are|we can|you can|one|first)',
        ]
        first_line_lower = non_empty_lines[0].lower().strip() if non_empty_lines else ''
        for pat in opening_patterns:
            if re.match(pat, first_line_lower):
                completeness_score += 1.5
                break
        
        # Closing patterns
        if non_empty_lines:
            last_line_lower = non_empty_lines[-1].lower().strip()
            closing_patterns = [
                r'(in conclusion|overall|in summary|to summarize|finally|in short)',
                r'(these|this|such|the above|together)',
            ]
            for pat in closing_patterns:
                if re.search(pat, last_line_lower):
                    completeness_score += 1.5
                    break
        
        # Multi-paragraph structure suggests completeness
        paragraph_breaks = response.count('\n\n')
        if paragraph_breaks >= 2:
            completeness_score += 2.0
        elif paragraph_breaks == 1:
            completeness_score += 1.0
        
        score += min(completeness_score, 5.0)
        
        # ---- Feature 7: Repetition Penalty ----
        # Penalize repetitive content (indicates poor organization)
        
        if word_count > 10:
            word_list = [w.lower().strip('.,!?;:()[]"\'') for w in words]
            word_freq = Counter(word_list)
            
            # Remove common stop words from consideration
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'and', 'or', 'but', 'not', 'that', 'this', 'it', 'its',
                         'as', 'if', 'than', 'so', 'no', 'up', 'out', 'about'}
            
            content_words = {w: c for w, c in word_freq.items() 
                           if w not in stop_words and len(w) > 2}
            
            if content_words:
                max_freq = max(content_words.values())
                total_content = sum(content_words.values())
                
                # High concentration of any single content word = repetitive
                max_ratio = max_freq / total_content if total_content > 0 else 0
                
                if max_ratio > 0.3:
                    score -= 8.0  # Heavy repetition penalty
                elif max_ratio > 0.2:
                    score -= 4.0
                elif max_ratio > 0.15:
                    score -= 2.0
        
        # ---- Feature 8: Response Length Appropriateness ----
        # Based on query complexity, evaluate if response length is appropriate
        
        query_words = len(query.split()) if query else 5
        
        # Longer/complex queries deserve longer, more structured responses
        length_ratio = word_count / max(query_words, 1)
        
        if length_ratio < 0.5:
            length_score = 1.0  # Too short
        elif 0.5 <= length_ratio <= 1.5:
            length_score = 4.0  # Adequate
        elif 1.5 < length_ratio <= 4.0:
            length_score = 6.0  # Good elaboration
        elif 4.0 < length_ratio <= 8.0:
            length_score = 5.0  # Might be verbose
        else:
            length_score = 3.0  # Likely too verbose
        
        score += length_score
        
        # Normalize to 0-100 range
        # Max theoretical: ~10 + 10 + 12 + 10 + 8 + 5 + 6 = ~61, min ~0
        score = max(0.0, score)
        normalized = min(100.0, (score / 55.0) * 100.0)
        
        return round(normalized, 2)
        
    except Exception:
        return 25.0