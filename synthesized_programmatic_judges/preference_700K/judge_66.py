def judging_function(query, response):
    """
    Evaluates structural organization and formatting of a response.
    
    This variant focuses on:
    - Information density distribution (entropy of content across sections)
    - Visual hierarchy depth (nested structures, indentation levels)
    - Sentence length variance as a proxy for structural diversity
    - Ratio of structural markers to content
    - Readability flow metrics (sentence-to-sentence coherence signals)
    - Code block and special formatting detection
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        score = 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # === 1. VISUAL HIERARCHY DEPTH ANALYSIS (0-15 points) ===
        # Measure indentation levels and nesting depth
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                # Normalize tabs
                leading_spaces += line.count('\t') * 4
                indent_levels.add(leading_spaces)
        
        hierarchy_depth = len(indent_levels)
        hierarchy_score = min(hierarchy_depth * 2.5, 15.0)
        score += hierarchy_score
        
        # === 2. SENTENCE LENGTH VARIANCE (0-12 points) ===
        # Good writing mixes short and long sentences; structured content
        # tends to have more variance due to headers, list items, paragraphs
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
                std_dev = math.sqrt(variance)
                # Coefficient of variation
                cv = std_dev / mean_len if mean_len > 0 else 0
                # Moderate CV (0.3-0.8) is ideal - indicates structural variety
                if cv < 0.1:
                    variance_score = 2.0
                elif cv < 0.3:
                    variance_score = 5.0
                elif cv < 0.6:
                    variance_score = 10.0
                elif cv < 1.0:
                    variance_score = 12.0
                else:
                    variance_score = 8.0
            else:
                variance_score = 1.0
        elif len(sentences) == 2:
            variance_score = 3.0
        else:
            variance_score = 1.0
        
        score += variance_score
        
        # === 3. CONTENT DISTRIBUTION ENTROPY (0-15 points) ===
        # Split response into chunks and measure how evenly content is distributed
        # Well-organized responses distribute information more evenly
        if len(non_empty_lines) >= 2:
            # Split into segments by blank lines (paragraphs/sections)
            segments = []
            current_segment = []
            for line in lines:
                if line.strip() == '':
                    if current_segment:
                        segments.append(' '.join(current_segment))
                        current_segment = []
                else:
                    current_segment.append(line.strip())
            if current_segment:
                segments.append(' '.join(current_segment))
            
            if len(segments) >= 2:
                seg_lengths = [len(s) for s in segments]
                total_seg = sum(seg_lengths)
                if total_seg > 0:
                    probs = [l / total_seg for l in seg_lengths if l > 0]
                    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    # Bonus for having multiple well-balanced segments
                    segment_count_bonus = min(len(segments) * 0.5, 3.0)
                    entropy_score = normalized_entropy * 12.0 + segment_count_bonus
                else:
                    entropy_score = 2.0
            else:
                # Single block of text - poor organization for longer responses
                if total_chars > 200:
                    entropy_score = 1.0
                else:
                    entropy_score = 5.0  # Short responses don't need multiple paragraphs
        else:
            entropy_score = 2.0
        
        score += min(entropy_score, 15.0)
        
        # === 4. STRUCTURAL MARKER DENSITY (0-15 points) ===
        # Count different types of structural markers and their variety
        marker_types = {
            'numbered_list': len(re.findall(r'(?m)^\s*\d+[\.\)]\s', response)),
            'bullet_dash': len(re.findall(r'(?m)^\s*[-–—]\s', response)),
            'bullet_star': len(re.findall(r'(?m)^\s*[*•·]\s', response)),
            'markdown_header': len(re.findall(r'(?m)^#{1,6}\s', response)),
            'bold_text': len(re.findall(r'\*\*[^*]+\*\*', response)),
            'italic_text': len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response)),
            'code_block': len(re.findall(r'```', response)) // 2,
            'inline_code': len(re.findall(r'`[^`]+`', response)),
            'colon_header': len(re.findall(r'(?m)^[A-Z][^.!?\n]{2,40}:\s*$', response)),
            'parenthetical': len(re.findall(r'\([^)]{5,}\)', response)),
            'letter_list': len(re.findall(r'(?m)^\s*[a-zA-Z][\.\)]\s', response)),
        }
        
        total_markers = sum(marker_types.values())
        types_used = sum(1 for v in marker_types.values() if v > 0)
        
        # Reward both quantity and variety of markers
        marker_quantity_score = min(total_markers * 0.8, 8.0)
        marker_variety_score = min(types_used * 1.5, 7.0)
        marker_score = marker_quantity_score + marker_variety_score
        
        score += min(marker_score, 15.0)
        
        # === 5. LINE-LEVEL STRUCTURAL PATTERNS (0-12 points) ===
        # Analyze patterns in line lengths - structured content has recognizable patterns
        if len(non_empty_lines) >= 3:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            
            # Check for "short-long" alternation (headers followed by content)
            short_threshold = max(30, sum(line_lengths) / len(line_lengths) * 0.4)
            pattern_changes = 0
            for i in range(1, len(line_lengths)):
                prev_short = line_lengths[i-1] < short_threshold
                curr_short = line_lengths[i] < short_threshold
                if prev_short != curr_short:
                    pattern_changes += 1
            
            pattern_ratio = pattern_changes / (len(line_lengths) - 1) if len(line_lengths) > 1 else 0
            # Higher pattern ratio = more structural alternation
            pattern_score = min(pattern_ratio * 15.0, 8.0)
            
            # Check for consistent list item lengths (sign of organized lists)
            if total_markers > 2:
                list_line_lengths = []
                for line in non_empty_lines:
                    stripped = line.strip()
                    if re.match(r'^[\d\-*•·]\s|^\d+[\.\)]\s', stripped):
                        list_line_lengths.append(len(stripped))
                
                if len(list_line_lengths) >= 2:
                    list_mean = sum(list_line_lengths) / len(list_line_lengths)
                    if list_mean > 0:
                        list_cv = math.sqrt(sum((l - list_mean)**2 for l in list_line_lengths) / len(list_line_lengths)) / list_mean
                        # Consistent list items (low CV) get bonus
                        consistency_bonus = max(0, 4.0 - list_cv * 4.0)
                        pattern_score += consistency_bonus
            
            score += min(pattern_score, 12.0)
        else:
            score += 1.0
        
        # === 6. RESPONSE LENGTH APPROPRIATENESS (0-8 points) ===
        # Longer responses need more structure; short responses get a pass
        query_len = len(query) if query else 50
        resp_words = len(response.split())
        
        if resp_words < 20:
            # Very short - structure matters less but shouldn't be penalized too much
            length_score = 4.0
        elif resp_words < 50:
            # Short-medium: basic structure sufficient
            length_score = 5.0 if len(segments if 'segments' in dir() else []) >= 1 else 3.0
            try:
                if len(segments) >= 2:
                    length_score = 6.0
            except:
                pass
        elif resp_words < 150:
            # Medium: should have some structure
            has_structure = total_markers > 0 or (len(segments) >= 2 if 'segments' in dir() else False)
            try:
                has_structure = total_markers > 0 or len(segments) >= 2
            except:
                has_structure = total_markers > 0
            length_score = 7.0 if has_structure else 3.0
        else:
            # Long: definitely needs structure
            try:
                has_good_structure = (total_markers > 2 or len(segments) >= 3) and types_used >= 1
            except:
                has_good_structure = total_markers > 2
            length_score = 8.0 if has_good_structure else 2.0
        
        score += length_score
        
        # === 7. OPENING AND CLOSING QUALITY (0-8 points) ===
        # Check if response has a clear opening statement and closing
        first_line = non_empty_lines[0].strip() if non_empty_lines else ""
        last_line = non_empty_lines[-1].strip() if non_empty_lines else ""
        
        opening_score = 0.0
        # Good openings: direct address, clear topic sentence, contextual setup
        if len(first_line) > 15:
            opening_score += 2.0
        # Check if first line is a clear introductory sentence (not a list item)
        if first_line and not re.match(r'^[\d\-*•]', first_line):
            opening_score += 1.5
        # Check for engagement markers
        if re.search(r'^(Sure|Yes|No|Great|Absolutely|Well|So|To|The|In|As|When|If|This)', first_line, re.I):
            opening_score += 0.5
        
        closing_score = 0.0
        if len(last_line) > 15:
            closing_score += 1.5
        # Check for concluding signals
        if re.search(r'(overall|in summary|hope|conclusion|finally|in short|ultimately|good luck)', last_line, re.I):
            closing_score += 1.5
        # Last line ends with proper punctuation
        if last_line and last_line[-1] in '.!?)':
            closing_score += 1.0
        
        score += min(opening_score + closing_score, 8.0)
        
        # === 8. WALL-OF-TEXT PENALTY (0 to -10 points) ===
        # Penalize responses that are one big block with no breaks
        if total_chars > 300:
            blank_line_count = sum(1 for l in lines if l.strip() == '')
            lines_per_break = len(non_empty_lines) / (blank_line_count + 1)
            
            if blank_line_count == 0 and total_chars > 500:
                score -= 8.0
            elif blank_line_count == 0 and total_chars > 300:
                score -= 4.0
            elif lines_per_break > 15:
                score -= 3.0
        
        # === 9. SPECIAL FORMAT DETECTION (0-10 points) ===
        # Detect and reward well-formatted special content
        special_score = 0.0
        
        # Code blocks with language specification
        if re.search(r'```\w+', response):
            special_score += 3.0
        elif '```' in response:
            special_score += 1.5
        
        # Tables (markdown style)
        if re.search(r'\|.*\|.*\|', response):
            special_score += 3.0
        
        # Quotations
        if re.search(r'(?m)^>\s', response):
            special_score += 2.0
        
        # URLs/references formatted properly
        if re.search(r'\[.*?\]\(.*?\)', response):
            special_score += 2.0
        
        # Emphasis for key terms
        if marker_types.get('bold_text', 0) >= 1:
            special_score += 1.5
        if marker_types.get('italic_text', 0) >= 1:
            special_score += 1.0
        
        score += min(special_score, 10.0)
        
        # Normalize to 0-10 range
        # Max theoretical: 15+12+15+15+12+8+8+10 = 95, but practical max ~70
        normalized_score = max(0.0, min(10.0, score / 7.5))
        
        return round(normalized_score, 2)
    
    except Exception:
        # Fallback: return a middle-ground score
        try:
            if response and len(response) > 50:
                return 3.0
            return 1.5
        except:
            return 1.0