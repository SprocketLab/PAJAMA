def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical document structure analysis approach.
    
    This variant focuses on:
    - Document structure tree depth and breadth analysis
    - Visual rhythm (alternation between different element types)
    - Information density distribution across sections
    - Structural consistency and symmetry
    - Readability flow scoring based on element sequencing
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response = response.rstrip()
        lines = response.split('\n')
        
        # === Phase 1: Classify each line into a structural element type ===
        element_types = []  # list of (type, indent_level, content_length)
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                element_types.append(('blank', 0, 0))
                continue
            
            # Measure indentation
            indent = len(line) - len(line.lstrip())
            indent_level = indent // 2  # normalize to levels
            clen = len(stripped)
            
            # Classify line type
            # Markdown headers
            if re.match(r'^#{1,6}\s+', stripped):
                depth = len(re.match(r'^(#+)', stripped).group(1))
                element_types.append(('header', depth, clen))
            # Bold headers (standalone bold text, typically short)
            elif re.match(r'^\*\*[^*]+\*\*[:\s]*$', stripped) and clen < 120:
                element_types.append(('bold_header', 1, clen))
            # Numbered list items
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                element_types.append(('numbered', indent_level, clen))
            # Bullet list items
            elif re.match(r'^[-*•+]\s+', stripped):
                element_types.append(('bullet', indent_level, clen))
            # Lettered list items
            elif re.match(r'^[a-zA-Z][\.\)]\s+', stripped):
                element_types.append(('lettered', indent_level, clen))
            # Code block markers
            elif stripped.startswith('```'):
                element_types.append(('code_fence', 0, clen))
            # Separator/divider lines
            elif re.match(r'^[-=_*]{3,}\s*$', stripped):
                element_types.append(('separator', 0, clen))
            # Short lines that look like labels/titles (under 60 chars, no period at end)
            elif clen < 60 and not stripped.endswith('.') and not stripped.endswith(',') and ':' in stripped:
                element_types.append(('label', indent_level, clen))
            else:
                element_types.append(('prose', indent_level, clen))
        
        # === Phase 2: Compute structural diversity and rhythm ===
        non_blank_types = [t for t, _, _ in element_types if t != 'blank']
        
        if not non_blank_types:
            return 0.5
        
        # Structural diversity: how many different element types are used
        unique_types = set(non_blank_types)
        structural_types = {'header', 'bold_header', 'numbered', 'bullet', 'lettered', 
                           'code_fence', 'separator', 'label'}
        formatting_types_used = unique_types & structural_types
        
        # Diversity score: reward variety of structural elements
        diversity_score = min(len(formatting_types_used) / 3.0, 1.0) * 20
        
        # === Phase 3: Visual rhythm analysis ===
        # Good documents alternate between different element types
        # Bad documents are monotonous (all prose or all bullets)
        transitions = 0
        meaningful_transitions = 0
        for i in range(1, len(non_blank_types)):
            if non_blank_types[i] != non_blank_types[i-1]:
                transitions += 1
                # Especially reward transitions between structural and content types
                prev_is_struct = non_blank_types[i-1] in structural_types
                curr_is_struct = non_blank_types[i] in structural_types
                if prev_is_struct != curr_is_struct:
                    meaningful_transitions += 1
        
        if len(non_blank_types) > 1:
            rhythm_ratio = transitions / (len(non_blank_types) - 1)
            meaningful_ratio = meaningful_transitions / (len(non_blank_types) - 1)
        else:
            rhythm_ratio = 0
            meaningful_ratio = 0
        
        rhythm_score = (rhythm_ratio * 10) + (meaningful_ratio * 8)
        
        # === Phase 4: Blank line usage analysis (visual breathing room) ===
        total_lines = len(element_types)
        blank_count = sum(1 for t, _, _ in element_types if t == 'blank')
        
        if total_lines > 0:
            blank_ratio = blank_count / total_lines
        else:
            blank_ratio = 0
        
        # Optimal blank ratio is around 0.15-0.35 for well-formatted text
        if 0.1 <= blank_ratio <= 0.4:
            whitespace_score = 10
        elif 0.05 <= blank_ratio < 0.1 or 0.4 < blank_ratio <= 0.5:
            whitespace_score = 6
        elif blank_ratio < 0.05:
            # Wall of text or very short
            whitespace_score = 2 if total_lines > 5 else 5
        else:
            whitespace_score = 3
        
        # === Phase 5: Hierarchical depth analysis ===
        # Check if there's a proper hierarchy (headers -> content -> sub-items)
        hierarchy_depth = 0
        header_levels_seen = set()
        has_nested_lists = False
        
        for t, level, _ in element_types:
            if t == 'header':
                header_levels_seen.add(level)
            if t in ('bullet', 'numbered', 'lettered') and level > 0:
                has_nested_lists = True
        
        hierarchy_depth = len(header_levels_seen)
        if has_nested_lists:
            hierarchy_depth += 1
        
        hierarchy_score = min(hierarchy_depth * 4, 12)
        
        # === Phase 6: Section symmetry analysis ===
        # Split response by headers and check if sections are roughly balanced
        sections = []
        current_section_lines = 0
        
        for t, _, clen in element_types:
            if t in ('header', 'bold_header') and current_section_lines > 0:
                sections.append(current_section_lines)
                current_section_lines = 0
            if t != 'blank':
                current_section_lines += 1
        if current_section_lines > 0:
            sections.append(current_section_lines)
        
        symmetry_score = 0
        if len(sections) >= 2:
            avg_section = sum(sections) / len(sections)
            if avg_section > 0:
                variance = sum((s - avg_section)**2 for s in sections) / len(sections)
                cv = math.sqrt(variance) / avg_section  # coefficient of variation
                # Lower CV = more balanced sections
                symmetry_score = max(0, 8 * (1 - min(cv, 1.5) / 1.5))
            # Bonus for having multiple sections
            symmetry_score += min(len(sections) - 1, 4) * 1.5
        
        # === Phase 7: Content density distribution ===
        # Check that content lengths vary appropriately (not all same length)
        content_lengths = [clen for t, _, clen in element_types if t != 'blank' and clen > 0]
        
        density_score = 0
        if content_lengths:
            avg_len = sum(content_lengths) / len(content_lengths)
            
            # Check for varied line lengths (sign of mixed formatting)
            short_lines = sum(1 for l in content_lengths if l < 40)
            medium_lines = sum(1 for l in content_lengths if 40 <= l < 120)
            long_lines = sum(1 for l in content_lengths if l >= 120)
            
            total_content = len(content_lengths)
            categories_present = sum(1 for c in [short_lines, medium_lines, long_lines] if c > 0)
            
            # Having mix of line lengths suggests formatting variety
            density_score = categories_present * 3
            
            # Penalize if everything is very long (wall of text paragraphs)
            if total_content > 0 and long_lines / total_content > 0.6:
                density_score -= 3
        
        # === Phase 8: Opening and closing structure ===
        # Good responses often start with an intro and end with a conclusion/summary
        opening_score = 0
        
        # Check if response starts with a brief intro before diving into lists
        first_content_types = [t for t, _, _ in element_types if t != 'blank'][:3]
        
        if first_content_types:
            # Starting with prose before structured content is good
            if first_content_types[0] == 'prose':
                if len(first_content_types) > 1 and first_content_types[1] in structural_types:
                    opening_score = 5  # Intro then structure
                else:
                    opening_score = 3  # Just prose
            elif first_content_types[0] in ('header', 'bold_header'):
                opening_score = 4  # Starting with a header is fine
            elif first_content_types[0] in ('numbered', 'bullet'):
                opening_score = 2  # Jumping straight to list
        
        # === Phase 9: List quality analysis ===
        # Check for consistent list usage
        list_items = [(t, level, clen) for t, level, clen in element_types 
                      if t in ('numbered', 'bullet', 'lettered')]
        
        list_score = 0
        if list_items:
            # Having lists is good
            list_score += 3
            
            # Check consistency of list item lengths
            list_lengths = [clen for _, _, clen in list_items]
            if len(list_lengths) >= 3:
                avg_list_len = sum(list_lengths) / len(list_lengths)
                if avg_list_len > 0:
                    list_cv = math.sqrt(sum((l - avg_list_len)**2 for l in list_lengths) / len(list_lengths)) / avg_list_len
                    # Moderate consistency is good (not too uniform, not too varied)
                    if list_cv < 0.5:
                        list_score += 4
                    elif list_cv < 1.0:
                        list_score += 2
            
            # Bonus for numbered lists (show ordering/priority)
            if any(t == 'numbered' for t, _, _ in list_items):
                list_score += 2
        
        # === Phase 10: Bold/emphasis usage ===
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        emphasis_score = 0
        if bold_count > 0:
            # Some bold is good for highlighting key terms
            emphasis_score = min(bold_count * 1.0, 6)
        
        # === Phase 11: Response length appropriateness ===
        query_len = len(query) if query else 10
        response_len = len(response)
        
        length_score = 0
        # Very short responses for complex queries are bad
        if response_len < 50:
            length_score = -3
        elif response_len < 100:
            length_score = 0
        else:
            length_score = 2
        
        # === Combine all scores ===
        total = (
            diversity_score +       # 0-20: variety of formatting elements
            rhythm_score +           # 0-18: alternation between element types
            whitespace_score +       # 2-10: appropriate use of blank lines
            hierarchy_score +        # 0-12: depth of document hierarchy
            symmetry_score +         # 0-14: balance between sections
            density_score +          # -3 to 9: varied content density
            opening_score +          # 0-5: good intro structure
            list_score +             # 0-9: quality list usage
            emphasis_score +         # 0-6: bold/emphasis usage
            length_score             # -3 to 2: length appropriateness
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~105, typical good response ~60-80
        # Normalize so scores spread across 0-100
        normalized = max(0, min(100, total * 1.1))
        
        return round(normalized, 2)
        
    except Exception:
        return 25.0