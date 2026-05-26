def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    
    This variant focuses on a HIERARCHICAL STRUCTURE ANALYSIS approach:
    - Analyzes the response as a tree of structural elements (sections, subsections, items)
    - Measures structural depth and breadth
    - Evaluates visual rhythm (alternation of different element types)
    - Computes a "scanability" score based on entry points for the eye
    - Measures proportionality of sections
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(text)
        
        score = 0.0
        
        # ============================================================
        # 1. STRUCTURAL ELEMENT CLASSIFICATION
        # Classify each line into a type to analyze the "rhythm"
        # ============================================================
        
        LINE_HEADER = 'header'
        LINE_BULLET = 'bullet'
        LINE_NUMBERED = 'numbered'
        LINE_CODE = 'code'
        LINE_QUOTE = 'quote'
        LINE_BLANK = 'blank'
        LINE_TEXT = 'text'
        LINE_BOLD_START = 'bold_start'  # line starting with bold text (pseudo-header)
        
        def classify_line(line):
            stripped = line.strip()
            if not stripped:
                return LINE_BLANK
            if re.match(r'^#{1,6}\s', stripped):
                return LINE_HEADER
            if re.match(r'^(\*\*|__)[A-Z]', stripped) and len(stripped) < 120:
                return LINE_BOLD_START
            if re.match(r'^[-*+•–—]\s', stripped):
                return LINE_BULLET
            if re.match(r'^\d+[\.\)]\s', stripped):
                return LINE_NUMBERED
            if stripped.startswith('```') or stripped.startswith('~~~'):
                return LINE_CODE
            if stripped.startswith('>'):
                return LINE_QUOTE
            if re.match(r'^[A-Z][^.!?]*:\s*$', stripped) and len(stripped) < 80:
                return LINE_HEADER  # colon-terminated headers
            return LINE_TEXT
        
        classified = [classify_line(l) for l in lines]
        
        # ============================================================
        # 2. STRUCTURAL DEPTH SCORE
        # Measures how many levels of hierarchy exist
        # ============================================================
        
        hierarchy_types = set()
        for c in classified:
            if c in (LINE_HEADER, LINE_BOLD_START):
                hierarchy_types.add('header')
            elif c in (LINE_BULLET, LINE_NUMBERED):
                hierarchy_types.add('list')
            elif c == LINE_CODE:
                hierarchy_types.add('code')
            elif c == LINE_QUOTE:
                hierarchy_types.add('quote')
        
        # Check for nested lists (indented bullets)
        has_nested = any(
            re.match(r'^(\s{2,}|\t+)[-*+•]\s', l) or re.match(r'^(\s{2,}|\t+)\d+[\.\)]\s', l)
            for l in lines
        )
        if has_nested:
            hierarchy_types.add('nested_list')
        
        # Check for markdown headers at different levels
        header_levels = set()
        for l in lines:
            m = re.match(r'^(#{1,6})\s', l.strip())
            if m:
                header_levels.add(len(m.group(1)))
        if len(header_levels) > 1:
            hierarchy_types.add('multi_level_headers')
        
        depth_score = min(len(hierarchy_types) * 2.5, 10.0)
        
        # ============================================================
        # 3. VISUAL RHYTHM SCORE
        # Good formatting alternates between different element types
        # creating a visual rhythm. Monotonous = bad.
        # ============================================================
        
        non_blank_classes = [c for c in classified if c != LINE_BLANK]
        
        rhythm_score = 0.0
        if len(non_blank_classes) > 2:
            transitions = 0
            for i in range(1, len(non_blank_classes)):
                if non_blank_classes[i] != non_blank_classes[i-1]:
                    transitions += 1
            
            transition_rate = transitions / (len(non_blank_classes) - 1) if len(non_blank_classes) > 1 else 0
            # Optimal transition rate is around 0.3-0.6 (not too monotonous, not too chaotic)
            if transition_rate < 0.1:
                rhythm_score = 2.0
            elif transition_rate < 0.3:
                rhythm_score = 5.0 + transition_rate * 10
            elif transition_rate <= 0.65:
                rhythm_score = 8.0 + (transition_rate - 0.3) * 5
            else:
                rhythm_score = max(5.0, 10.0 - (transition_rate - 0.65) * 15)
        else:
            rhythm_score = 3.0
        
        # ============================================================
        # 4. SCANABILITY SCORE
        # How many "entry points" does the eye have?
        # Entry points: headers, bold text, list markers, code blocks,
        # paragraph breaks, emphasized words
        # ============================================================
        
        entry_points = 0
        
        # Headers
        entry_points += sum(1 for c in classified if c in (LINE_HEADER, LINE_BOLD_START)) * 3
        
        # List items
        entry_points += sum(1 for c in classified if c in (LINE_BULLET, LINE_NUMBERED)) * 1.5
        
        # Bold/italic inline markers
        bold_inline = len(re.findall(r'\*\*[^*]+\*\*', text)) + len(re.findall(r'__[^_]+__', text))
        italic_inline = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', text))
        entry_points += bold_inline * 1.5 + italic_inline * 0.5
        
        # Code inline
        code_inline = len(re.findall(r'`[^`]+`', text))
        entry_points += code_inline * 0.5
        
        # Paragraph breaks (blank lines between text)
        para_breaks = sum(1 for i in range(1, len(classified)-1) 
                         if classified[i] == LINE_BLANK and 
                         classified[i-1] != LINE_BLANK)
        entry_points += para_breaks * 2
        
        # Normalize by response length (per 100 chars)
        entry_density = entry_points / max(total_chars / 100, 1)
        
        # Sweet spot: enough entry points but not overwhelming
        if entry_density < 0.2:
            scan_score = 2.0 + entry_density * 15
        elif entry_density < 1.5:
            scan_score = 5.0 + min((entry_density - 0.2) * 4, 5.0)
        elif entry_density < 4.0:
            scan_score = 10.0
        else:
            scan_score = max(5.0, 10.0 - (entry_density - 4.0) * 1.5)
        
        # ============================================================
        # 5. PARAGRAPH PROPORTIONALITY
        # Segments should be roughly balanced in size
        # ============================================================
        
        # Split into segments (by blank lines or structural markers)
        segments = []
        current_seg = []
        for i, line in enumerate(lines):
            cl = classified[i]
            if cl == LINE_BLANK:
                if current_seg:
                    segments.append('\n'.join(current_seg))
                    current_seg = []
            elif cl in (LINE_HEADER, LINE_BOLD_START) and current_seg:
                segments.append('\n'.join(current_seg))
                current_seg = [line]
            else:
                current_seg.append(line)
        if current_seg:
            segments.append('\n'.join(current_seg))
        
        proportion_score = 5.0  # default
        if len(segments) >= 2:
            seg_lengths = [len(s.strip()) for s in segments if s.strip()]
            if seg_lengths:
                avg_len = sum(seg_lengths) / len(seg_lengths)
                if avg_len > 0:
                    # Coefficient of variation
                    variance = sum((l - avg_len)**2 for l in seg_lengths) / len(seg_lengths)
                    cv = math.sqrt(variance) / avg_len
                    
                    # Some variation is fine, extreme variation is bad
                    if cv < 0.3:
                        proportion_score = 9.0
                    elif cv < 0.6:
                        proportion_score = 7.0
                    elif cv < 1.0:
                        proportion_score = 5.0
                    elif cv < 1.5:
                        proportion_score = 3.5
                    else:
                        proportion_score = 2.0
                
                # Bonus for having multiple segments (not a wall of text)
                num_segs = len(seg_lengths)
                if num_segs >= 3:
                    proportion_score += 1.5
                elif num_segs >= 2:
                    proportion_score += 0.5
                proportion_score = min(proportion_score, 10.0)
        elif len(segments) == 1:
            # Single block - check if it's a wall of text
            if total_chars > 400:
                proportion_score = 2.0
            elif total_chars > 200:
                proportion_score = 3.5
            else:
                proportion_score = 5.0
        
        # ============================================================
        # 6. WALL-OF-TEXT PENALTY
        # Detect long unbroken stretches of text
        # ============================================================
        
        wall_penalty = 0.0
        
        # Check for very long lines (no line breaks in long text)
        max_line_len = max((len(l) for l in non_empty_lines), default=0)
        if max_line_len > 500:
            wall_penalty += 3.0
        elif max_line_len > 300:
            wall_penalty += 1.5
        elif max_line_len > 200:
            wall_penalty += 0.5
        
        # Check ratio of blank lines to non-blank lines
        blank_count = sum(1 for c in classified if c == LINE_BLANK)
        if len(non_empty_lines) > 3 and blank_count == 0:
            wall_penalty += 1.5
        
        # Long response with no structural elements
        structural_lines = sum(1 for c in classified if c in (LINE_HEADER, LINE_BULLET, LINE_NUMBERED, LINE_BOLD_START, LINE_CODE, LINE_QUOTE))
        if total_chars > 300 and structural_lines == 0 and blank_count < 2:
            wall_penalty += 2.5
        
        # ============================================================
        # 7. LOGICAL GROUPING INDICATORS
        # Check for topic sentences, transition patterns
        # ============================================================
        
        grouping_score = 5.0
        
        # Check if paragraphs start with different topics (diverse first words)
        para_starters = []
        for seg in segments:
            first_line = seg.strip().split('\n')[0].strip()
            # Get first meaningful word
            words = re.findall(r'[A-Za-z]+', first_line)
            if words:
                para_starters.append(words[0].lower())
        
        if len(para_starters) >= 2:
            unique_ratio = len(set(para_starters)) / len(para_starters)
            if unique_ratio > 0.7:
                grouping_score += 2.0  # diverse paragraph starts
            elif unique_ratio < 0.3:
                grouping_score -= 1.0  # repetitive starts
        
        # Check for connective/transition words at paragraph starts
        connectives = {'however', 'moreover', 'furthermore', 'additionally', 'conversely',
                       'nevertheless', 'therefore', 'consequently', 'meanwhile', 'similarly',
                       'in', 'on', 'for', 'first', 'second', 'third', 'finally', 'lastly',
                       'also', 'that', 'another', 'next', 'then', 'overall', 'to'}
        
        connective_count = sum(1 for s in para_starters if s in connectives)
        if len(para_starters) > 2 and connective_count >= 1:
            grouping_score += 1.5
        
        grouping_score = min(max(grouping_score, 0), 10.0)
        
        # ============================================================
        # 8. RESPONSE LENGTH APPROPRIATENESS
        # Very short responses get a mild baseline (formatting matters less)
        # Very long responses need more structure
        # ============================================================
        
        length_factor = 1.0
        query_complexity = len(query.split()) if query else 10
        
        if total_chars < 50:
            # Very short - formatting barely matters, give moderate baseline
            length_factor = 0.6
        elif total_chars < 150:
            length_factor = 0.8
        elif total_chars > 1000:
            # Long responses need structure more
            if structural_lines < 2 and blank_count < 3:
                length_factor = 0.7  # penalize long unstructured
            else:
                length_factor = 1.1  # reward long structured
        
        # ============================================================
        # 9. CODE BLOCK FORMATTING (for technical responses)
        # ============================================================
        
        code_bonus = 0.0
        has_code_fence = '```' in text
        if has_code_fence:
            # Properly fenced code blocks
            fences = text.count('```')
            if fences >= 2 and fences % 2 == 0:
                code_bonus = 2.0
                # Check for language annotation
                if re.search(r'```\w+', text):
                    code_bonus += 1.0
            else:
                code_bonus = 0.5  # incomplete fencing
        
        # ============================================================
        # 10. EMPHASIS AND INLINE FORMATTING
        # ============================================================
        
        inline_format_score = 0.0
        if bold_inline > 0:
            inline_format_score += min(bold_inline * 0.8, 3.0)
        if italic_inline > 0:
            inline_format_score += min(italic_inline * 0.5, 2.0)
        if code_inline > 0:
            inline_format_score += min(code_inline * 0.3, 1.5)
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        # Weighted combination
        raw_score = (
            depth_score * 0.15 +
            rhythm_score * 0.12 +
            scan_score * 0.20 +
            proportion_score * 0.18 +
            grouping_score * 0.10 +
            inline_format_score * 0.05 +
            code_bonus * 0.05
        )
        
        # Apply penalties
        raw_score -= wall_penalty
        
        # Apply length factor
        raw_score *= length_factor
        
        # Add a small baseline for non-empty responses
        raw_score += 1.5
        
        # Bonus for responses that have a clear opening and closing pattern
        if len(segments) >= 3:
            first_seg = segments[0].strip()
            last_seg = segments[-1].strip()
            # Opening is shorter (intro), closing is shorter (conclusion)
            mid_avg = sum(len(s.strip()) for s in segments[1:-1]) / max(len(segments) - 2, 1)
            if len(first_seg) < mid_avg * 1.5 and len(last_seg) < mid_avg * 2:
                raw_score += 0.5
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 3.0