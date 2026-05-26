def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant focuses on a HIERARCHICAL STRUCTURE ANALYSIS approach:
    - Analyzes the depth and consistency of structural hierarchy
    - Measures visual rhythm (alternation between different element types)
    - Evaluates proportionality of sections
    - Scores structural "completeness" (intro, body, conclusion patterns)
    - Uses a state-machine approach to track formatting transitions
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
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Line-type state machine and transition analysis
        # Classify each line into a type, then analyze transitions
        # ============================================================
        
        def classify_line(line):
            stripped = line.strip()
            if not stripped:
                return 'blank'
            # Markdown header
            if re.match(r'^#{1,6}\s+', stripped):
                return 'header'
            # Bold header-like (e.g., **Step 1: ...**)
            if re.match(r'^\*\*[^*]+\*\*\s*$', stripped) or re.match(r'^\*\*[^*]+\*\*:?\s*$', stripped):
                return 'bold_header'
            # Numbered list item
            if re.match(r'^\d+[\.\)]\s+', stripped):
                return 'numbered'
            # Bullet list item
            if re.match(r'^[-*•]\s+', stripped):
                return 'bullet'
            # Sub-item (indented bullet or number)
            if re.match(r'^\s{2,}[-*•]\s+', line) or re.match(r'^\s{2,}\d+[\.\)]\s+', line):
                return 'sub_item'
            # Code block marker
            if stripped.startswith('```'):
                return 'code_fence'
            # Separator/horizontal rule
            if re.match(r'^[-=_]{3,}\s*$', stripped):
                return 'separator'
            # Short line that looks like a label/intro
            if len(stripped) < 60 and stripped.endswith(':'):
                return 'label'
            return 'prose'
        
        line_types = [classify_line(l) for l in lines]
        non_blank_types = [t for t in line_types if t != 'blank']
        
        # Count transitions between different non-blank types
        transitions = 0
        transition_pairs = Counter()
        for i in range(1, len(non_blank_types)):
            if non_blank_types[i] != non_blank_types[i-1]:
                transitions += 1
                pair = (non_blank_types[i-1], non_blank_types[i])
                transition_pairs[pair] += 1
        
        # Good transitions: header->prose, header->numbered, prose->numbered, etc.
        good_transitions = {
            ('header', 'prose'), ('header', 'numbered'), ('header', 'bullet'),
            ('bold_header', 'prose'), ('bold_header', 'numbered'), ('bold_header', 'bullet'),
            ('prose', 'numbered'), ('prose', 'bullet'), ('prose', 'header'),
            ('prose', 'bold_header'), ('numbered', 'prose'), ('bullet', 'prose'),
            ('numbered', 'header'), ('bullet', 'header'), ('numbered', 'bold_header'),
            ('label', 'numbered'), ('label', 'bullet'), ('label', 'prose'),
            ('numbered', 'sub_item'), ('bullet', 'sub_item'), ('sub_item', 'numbered'),
            ('sub_item', 'bullet'), ('prose', 'label'),
        }
        
        good_trans_count = sum(v for k, v in transition_pairs.items() if k in good_transitions)
        total_trans = sum(transition_pairs.values())
        
        if total_trans > 0:
            trans_quality = good_trans_count / total_trans
            # Reward variety of transitions (structural richness)
            unique_types = len(set(non_blank_types))
            transition_score = trans_quality * min(unique_types / 3.0, 1.5)
            score += transition_score * 8  # up to ~12
        
        # ============================================================
        # FEATURE 2: Structural completeness (intro-body-conclusion)
        # ============================================================
        
        type_counts = Counter(non_blank_types)
        has_headers = type_counts.get('header', 0) + type_counts.get('bold_header', 0) > 0
        has_lists = type_counts.get('numbered', 0) + type_counts.get('bullet', 0) > 0
        has_prose = type_counts.get('prose', 0) > 0
        has_sub_items = type_counts.get('sub_item', 0) > 0
        
        # Check if response starts with prose (intro) before structured content
        intro_detected = False
        if len(non_blank_types) >= 3:
            if non_blank_types[0] == 'prose':
                intro_detected = True
                score += 3.0
        
        # Check for structured body
        structured_lines = (type_counts.get('numbered', 0) + type_counts.get('bullet', 0) + 
                          type_counts.get('header', 0) + type_counts.get('bold_header', 0) +
                          type_counts.get('sub_item', 0))
        
        if len(non_blank_types) > 0:
            structure_ratio = structured_lines / len(non_blank_types)
            # Sweet spot: 30-70% structured content (mix of prose and structure)
            if 0.2 <= structure_ratio <= 0.8:
                score += 6.0
            elif 0.1 <= structure_ratio <= 0.9:
                score += 3.5
            elif structure_ratio > 0:
                score += 1.5
        
        # ============================================================
        # FEATURE 3: Hierarchical depth analysis
        # ============================================================
        
        # Measure nesting levels present
        hierarchy_levels = set()
        
        if has_headers:
            hierarchy_levels.add('h')
            # Check for multiple header levels
            h_levels = set()
            for l in lines:
                m = re.match(r'^(#{1,6})\s+', l.strip())
                if m:
                    h_levels.add(len(m.group(1)))
            if len(h_levels) > 1:
                hierarchy_levels.add('h_multi')
        
        if type_counts.get('bold_header', 0) > 0:
            hierarchy_levels.add('bold_h')
        
        if has_lists:
            hierarchy_levels.add('list')
        
        if has_sub_items:
            hierarchy_levels.add('sub')
        
        if has_prose:
            hierarchy_levels.add('prose')
        
        depth_score = len(hierarchy_levels)
        score += min(depth_score * 2.0, 10.0)
        
        # ============================================================
        # FEATURE 4: Visual rhythm - blank line usage patterns
        # ============================================================
        
        blank_lines = [i for i, t in enumerate(line_types) if t == 'blank']
        total_lines = len(lines)
        
        if total_lines > 3:
            blank_ratio = len(blank_lines) / total_lines
            # Good whitespace usage: 15-40% blank lines
            if 0.1 <= blank_ratio <= 0.45:
                score += 5.0
            elif 0.05 <= blank_ratio <= 0.55:
                score += 2.5
            elif blank_ratio > 0:
                score += 1.0
            
            # Check for regular spacing (blank lines between sections)
            if len(blank_lines) >= 2:
                gaps = [blank_lines[i+1] - blank_lines[i] for i in range(len(blank_lines)-1)]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    variance = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
                    # Lower variance = more regular spacing
                    regularity = 1.0 / (1.0 + math.sqrt(variance) / max(avg_gap, 1))
                    score += regularity * 4.0
        
        # ============================================================
        # FEATURE 5: Section proportionality
        # ============================================================
        
        # Split response into sections by headers or blank lines
        sections = []
        current_section = []
        for line in lines:
            stripped = line.strip()
            cls = classify_line(line)
            if cls in ('header', 'bold_header', 'separator') and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            elif not stripped and current_section and len(current_section) > 2:
                # Large blank-line separated blocks count as sections
                current_section.append(line)
            else:
                current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
        
        if len(sections) >= 2:
            section_lengths = [len(s.strip()) for s in sections if s.strip()]
            if section_lengths:
                avg_len = sum(section_lengths) / len(section_lengths)
                if avg_len > 0:
                    # Coefficient of variation - lower is more balanced
                    cv = math.sqrt(sum((l - avg_len)**2 for l in section_lengths) / len(section_lengths)) / avg_len
                    balance_score = max(0, 1.0 - cv * 0.5)
                    score += balance_score * 4.0
                
                # Reward having multiple sections
                section_count_bonus = min(len(section_lengths) / 3.0, 1.5)
                score += section_count_bonus * 3.0
        
        # ============================================================
        # FEATURE 6: Inline formatting richness
        # ============================================================
        
        # Bold text usage (not as headers)
        bold_inline = len(re.findall(r'(?<!\n)\*\*[^*\n]+\*\*', response))
        # Inline code
        inline_code = len(re.findall(r'`[^`\n]+`', response))
        # Colons used as delimiters in structured content
        colon_labels = len(re.findall(r'\*\*[^*]+\*\*:', response))
        
        formatting_richness = min(bold_inline * 0.3 + inline_code * 0.2 + colon_labels * 0.4, 6.0)
        score += formatting_richness
        
        # ============================================================
        # FEATURE 7: Paragraph quality (for prose sections)
        # ============================================================
        
        # Split by double newlines to find paragraphs
        paragraphs = re.split(r'\n\s*\n', response)
        prose_paragraphs = []
        for p in paragraphs:
            p_stripped = p.strip()
            if len(p_stripped) > 50:
                # Check if it's mostly prose (not a list)
                p_lines = p_stripped.split('\n')
                prose_lines = sum(1 for l in p_lines if classify_line(l) == 'prose')
                if prose_lines >= len(p_lines) * 0.5:
                    prose_paragraphs.append(p_stripped)
        
        if prose_paragraphs:
            para_lengths = [len(p) for p in prose_paragraphs]
            # Good paragraph length: 100-500 chars
            good_paras = sum(1 for l in para_lengths if 80 <= l <= 600)
            if para_lengths:
                para_quality = good_paras / len(para_lengths)
                score += para_quality * 3.0
        
        # ============================================================
        # FEATURE 8: Wall-of-text penalty
        # ============================================================
        
        # Detect wall of text: long response with few structural elements
        if total_chars > 200:
            structural_density = structured_lines / max(len(non_empty_lines), 1)
            blank_density = len(blank_lines) / max(total_lines, 1)
            
            if structural_density < 0.05 and blank_density < 0.05 and len(non_empty_lines) > 5:
                # Heavy wall-of-text penalty
                score -= 8.0
            elif structural_density < 0.1 and blank_density < 0.1:
                score -= 3.0
        
        # ============================================================
        # FEATURE 9: Opening engagement quality
        # ============================================================
        
        if non_empty_lines:
            first_line = non_empty_lines[0].strip()
            # Engaging opener that's not just jumping into a list
            if (len(first_line) > 30 and 
                classify_line(first_line) == 'prose' and
                not first_line.startswith('1.') and
                not first_line.startswith('-')):
                score += 2.0
                # Extra for conversational/engaging tone
                engagement_words = ['great', 'certainly', 'absolutely', 'excellent', 'happy to', 
                                   'let me', "let's", 'here', 'sure', 'of course']
                if any(w in first_line.lower() for w in engagement_words):
                    score += 1.0
        
        # ============================================================
        # FEATURE 10: Response length appropriateness relative to query
        # ============================================================
        
        query_len = len(query) if query else 1
        response_len = len(response)
        ratio = response_len / max(query_len, 1)
        
        # Responses should generally be longer than queries but not absurdly so
        if 2 <= ratio <= 50:
            score += 2.0
        elif 1 <= ratio <= 100:
            score += 1.0
        
        # ============================================================
        # FEATURE 11: Consistency of list formatting
        # ============================================================
        
        numbered_items = re.findall(r'^(\d+)[\.\)]\s+', response, re.MULTILINE)
        if len(numbered_items) >= 2:
            # Check if numbers are sequential
            nums = [int(n) for n in numbered_items]
            sequential = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
            starts_at_one = nums[0] == 1
            if sequential and starts_at_one:
                score += 3.0
            elif sequential:
                score += 1.5
        
        bullet_items = re.findall(r'^[-*•]\s+', response, re.MULTILINE)
        if len(bullet_items) >= 2:
            # Consistent bullet character
            bullet_chars = [b[0] for b in bullet_items]
            if len(set(bullet_chars)) == 1:
                score += 1.5
            else:
                score += 0.5
        
        # ============================================================
        # Normalize to 0-100 range
        # ============================================================
        
        # Theoretical max is roughly ~65, typical good response ~35-50
        # Normalize so good responses score 60-85, great ones 85-100
        final_score = max(0.0, min(score, 60.0))
        final_score = (final_score / 60.0) * 100.0
        
        return round(final_score, 2)
        
    except Exception:
        return 25.0