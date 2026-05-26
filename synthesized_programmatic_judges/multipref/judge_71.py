def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical document structure model approach.
    
    This variant uses a fundamentally different approach: it models the response
    as a hierarchical document tree, analyzing nesting depth, structural variety,
    visual rhythm (alternation of element types), and spatial balance between
    sections. Rather than counting individual features, it evaluates the
    compositional architecture of the response.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        import re
        from collections import Counter
        
        lines = response.split('\n')
        
        # === PHASE 1: Classify each line into a structural element type ===
        element_types = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                element_types.append('blank')
            elif re.match(r'^#{1,6}\s+', stripped):
                element_types.append('header')
            elif re.match(r'^(\*\*|__).+(\*\*|__)\s*$', stripped) and len(stripped) < 100:
                element_types.append('bold_header')
            elif re.match(r'^[-*•]\s+', stripped):
                element_types.append('bullet')
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                element_types.append('numbered')
            elif re.match(r'^[a-zA-Z][\.\)]\s+', stripped):
                element_types.append('lettered')
            elif re.match(r'^\s{2,}[-*•]\s+', line):
                element_types.append('nested_bullet')
            elif re.match(r'^\s{2,}\d+[\.\)]\s+', line):
                element_types.append('nested_numbered')
            elif re.match(r'^(---+|===+|\*\*\*+)\s*$', stripped):
                element_types.append('separator')
            elif re.match(r'^```', stripped):
                element_types.append('code_fence')
            elif re.match(r'^\|.*\|', stripped):
                element_types.append('table_row')
            elif stripped.startswith('>'):
                element_types.append('blockquote')
            elif stripped.endswith(':') and len(stripped) < 80:
                element_types.append('label_line')
            else:
                element_types.append('prose')
        
        # === PHASE 2: Compute structural variety score ===
        # How many distinct non-blank element types are used?
        non_blank_types = [t for t in element_types if t != 'blank']
        if not non_blank_types:
            return 1.0
        
        unique_types = set(non_blank_types)
        type_variety = len(unique_types)
        
        # Diminishing returns on variety
        import math
        variety_score = min(1.0, math.log2(1 + type_variety) / 3.0)
        
        # === PHASE 3: Visual rhythm analysis ===
        # Measure the alternation pattern of element types (not counting blanks)
        # Good structure alternates between different element types
        transitions = 0
        total_adjacent = 0
        for i in range(1, len(non_blank_types)):
            total_adjacent += 1
            if non_blank_types[i] != non_blank_types[i-1]:
                transitions += 1
        
        rhythm_score = (transitions / total_adjacent) if total_adjacent > 0 else 0.0
        # Moderate rhythm is best (not too chaotic, not monotone)
        # Optimal around 0.3-0.6
        rhythm_quality = 1.0 - 2.0 * abs(rhythm_score - 0.45)
        rhythm_quality = max(0.0, min(1.0, rhythm_quality))
        
        # === PHASE 4: Segment balance analysis ===
        # Split response into segments at headers/blank lines
        segments = []
        current_segment_len = 0
        for et, line in zip(element_types, lines):
            if et in ('header', 'bold_header', 'separator', 'label_line') or et == 'blank':
                if current_segment_len > 0:
                    segments.append(current_segment_len)
                current_segment_len = 0
            else:
                current_segment_len += len(line.strip())
        if current_segment_len > 0:
            segments.append(current_segment_len)
        
        # Calculate balance using coefficient of variation
        if len(segments) >= 2:
            mean_seg = sum(segments) / len(segments)
            if mean_seg > 0:
                variance = sum((s - mean_seg)**2 for s in segments) / len(segments)
                cv = math.sqrt(variance) / mean_seg
                # Lower CV = more balanced
                balance_score = max(0.0, 1.0 - cv * 0.5)
            else:
                balance_score = 0.5
        elif len(segments) == 1:
            # Single block - penalize if response is long (should be broken up)
            total_len = len(response)
            if total_len > 500:
                balance_score = 0.2
            elif total_len > 200:
                balance_score = 0.4
            else:
                balance_score = 0.7
        else:
            balance_score = 0.3
        
        # === PHASE 5: Whitespace architecture ===
        # Analyze the pattern of blank lines as structural separators
        blank_count = element_types.count('blank')
        total_lines = len(element_types)
        blank_ratio = blank_count / total_lines if total_lines > 0 else 0
        
        # Optimal blank ratio is around 15-30%
        if blank_ratio < 0.05:
            whitespace_score = 0.2
        elif blank_ratio < 0.10:
            whitespace_score = 0.5
        elif blank_ratio <= 0.35:
            whitespace_score = 1.0
        elif blank_ratio <= 0.50:
            whitespace_score = 0.6
        else:
            whitespace_score = 0.3
        
        # Check for consecutive blank lines (excessive spacing)
        consecutive_blanks = max(
            (len(list(g)) for k, g in __import__('itertools').groupby(element_types) if k == 'blank'),
            default=0
        )
        if consecutive_blanks > 2:
            whitespace_score *= 0.7
        
        # === PHASE 6: Hierarchical depth analysis ===
        # Check for nesting (sub-items, indentation levels)
        has_nesting = any(t.startswith('nested_') for t in element_types)
        has_headers = 'header' in unique_types or 'bold_header' in unique_types
        has_lists = 'bullet' in unique_types or 'numbered' in unique_types
        has_prose = 'prose' in unique_types
        
        # Count distinct header levels
        header_levels = set()
        for line in lines:
            m = re.match(r'^(#{1,6})\s+', line.strip())
            if m:
                header_levels.add(len(m.group(1)))
        
        depth_score = 0.0
        if has_headers:
            depth_score += 0.3
        if has_lists:
            depth_score += 0.25
        if has_nesting:
            depth_score += 0.2
        if len(header_levels) >= 2:
            depth_score += 0.15
        if has_prose and has_lists:
            depth_score += 0.1  # Mix of prose and structured content
        depth_score = min(1.0, depth_score)
        
        # === PHASE 7: Inline formatting density ===
        # Check for bold, italic, inline code as emphasis tools
        bold_matches = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_matches = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        inline_code = len(re.findall(r'`[^`]+`', response))
        
        total_words = len(response.split())
        if total_words > 0:
            formatting_density = (bold_matches + italic_matches + inline_code) / (total_words / 10.0)
        else:
            formatting_density = 0
        
        # Optimal density around 0.3-1.5 per 10 words
        if formatting_density < 0.05:
            inline_score = 0.2
        elif formatting_density < 0.3:
            inline_score = 0.5
        elif formatting_density <= 2.0:
            inline_score = 1.0
        else:
            inline_score = 0.6  # Over-formatted
        
        # === PHASE 8: Opening and closing structure ===
        # Good responses often have an intro paragraph and a conclusion
        non_blank_lines = [(i, et) for i, et in enumerate(element_types) if et != 'blank']
        
        intro_score = 0.0
        if non_blank_lines:
            first_type = non_blank_lines[0][1]
            if first_type == 'prose':
                # Starts with prose intro before diving into structure
                intro_score = 0.7
                # Check if intro is followed by structured content
                if len(non_blank_lines) > 1 and non_blank_lines[1][1] in ('header', 'bold_header', 'numbered', 'bullet', 'label_line'):
                    intro_score = 1.0
            elif first_type in ('header', 'bold_header'):
                intro_score = 0.5
            elif first_type in ('numbered', 'bullet'):
                intro_score = 0.3  # Jumps right into list without intro
        
        # === PHASE 9: Wall-of-text detection (different from paragraph counting) ===
        # Measure longest unbroken prose run
        max_prose_run = 0
        current_run = 0
        for et in element_types:
            if et == 'prose':
                current_run += 1
            else:
                max_prose_run = max(max_prose_run, current_run)
                current_run = 0
        max_prose_run = max(max_prose_run, current_run)
        
        # Also measure by character length of longest unbroken text block
        text_blocks = re.split(r'\n\s*\n', response)
        max_block_len = max((len(b.strip()) for b in text_blocks if b.strip()), default=0)
        
        wall_penalty = 0.0
        if max_prose_run > 15:
            wall_penalty += 0.3
        elif max_prose_run > 8:
            wall_penalty += 0.15
        
        if max_block_len > 800:
            wall_penalty += 0.3
        elif max_block_len > 500:
            wall_penalty += 0.15
        
        wall_penalty = min(0.5, wall_penalty)
        
        # === PHASE 10: Response length appropriateness ===
        query_len = len(query.strip()) if query else 0
        resp_len = len(response)
        
        # Very short responses for complex queries
        length_factor = 1.0
        if resp_len < 50:
            length_factor = 0.5
        elif resp_len < 100:
            length_factor = 0.7
        
        # === COMBINE SCORES ===
        # Weighted combination with emphasis on architectural qualities
        raw_score = (
            variety_score * 1.5 +
            rhythm_quality * 0.8 +
            balance_score * 1.0 +
            whitespace_score * 1.2 +
            depth_score * 1.8 +
            inline_score * 0.8 +
            intro_score * 0.9 +
            (1.0 - wall_penalty) * 1.0
        )
        
        max_possible = 1.5 + 0.8 + 1.0 + 1.2 + 1.8 + 0.8 + 0.9 + 1.0  # = 9.0
        
        normalized = (raw_score / max_possible) * 10.0 * length_factor
        
        # Bonus for responses that demonstrate clear structural intent
        structural_elements = sum(1 for t in non_blank_types if t not in ('prose',))
        structural_ratio = structural_elements / len(non_blank_types) if non_blank_types else 0
        
        # Bonus for having a good mix (not all structure, not all prose)
        if 0.2 <= structural_ratio <= 0.8:
            mix_bonus = 0.5
        elif 0.1 <= structural_ratio <= 0.9:
            mix_bonus = 0.25
        elif structural_ratio > 0.0:
            mix_bonus = 0.1
        else:
            mix_bonus = 0.0
        
        final_score = normalized + mix_bonus
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Minimal fallback
            if not response or len(response.strip()) < 10:
                return 1.0
            has_structure = bool(re.search(r'(#{1,6}\s|\d+\.\s|[-*]\s|\*\*)', response))
            has_breaks = '\n\n' in response
            return 5.0 if (has_structure and has_breaks) else 3.0
        except Exception:
            return 3.0