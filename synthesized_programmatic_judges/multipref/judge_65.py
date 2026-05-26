def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    Higher scores indicate better structural quality.
    
    This variant focuses on a feature-weighted scoring approach analyzing:
    - Markdown formatting elements (headers, bold, lists)
    - Paragraph structure and whitespace usage
    - Logical flow indicators (transition words, topic sentences)
    - Anti-patterns (wall of text, poor organization)
    
    Returns a score from 0 to 100.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        score = 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(lines)
        non_empty_count = len(non_empty_lines)
        
        # =====================================================
        # FEATURE 1: Header detection and quality (0-15 points)
        # =====================================================
        header_score = 0.0
        
        # Markdown headers (### style)
        md_headers = re.findall(r'^#{1,6}\s+.+', response, re.MULTILINE)
        num_md_headers = len(md_headers)
        
        # Bold headers (lines that are primarily bold text, acting as headers)
        bold_header_lines = re.findall(r'^\s*\*\*[^*]+\*\*\s*$', response, re.MULTILINE)
        num_bold_headers = len(bold_header_lines)
        
        # Combined header count
        total_headers = num_md_headers + num_bold_headers
        
        if total_headers >= 1:
            header_score += 5.0
        if total_headers >= 2:
            header_score += 4.0
        if total_headers >= 3:
            header_score += 3.0
        if total_headers >= 5:
            header_score += 3.0
        
        # Cap at 15
        header_score = min(header_score, 15.0)
        score += header_score
        
        # =====================================================
        # FEATURE 2: List usage and quality (0-20 points)
        # =====================================================
        list_score = 0.0
        
        # Numbered lists (1. 2. etc.)
        numbered_items = re.findall(r'^\s*\d+[\.\)]\s+\S', response, re.MULTILINE)
        num_numbered = len(numbered_items)
        
        # Bullet points (-, *, •)
        bullet_items = re.findall(r'^\s*[-*•]\s+\S', response, re.MULTILINE)
        num_bullets = len(bullet_items)
        
        # Sub-items (indented list items)
        sub_items = re.findall(r'^\s{2,}[-*•]\s+\S', response, re.MULTILINE)
        num_sub_items = len(sub_items)
        
        total_list_items = num_numbered + num_bullets
        
        if total_list_items >= 1:
            list_score += 4.0
        if total_list_items >= 3:
            list_score += 4.0
        if total_list_items >= 5:
            list_score += 4.0
        if total_list_items >= 8:
            list_score += 3.0
        
        # Bonus for sub-items (hierarchical organization)
        if num_sub_items >= 1:
            list_score += 2.5
        if num_sub_items >= 3:
            list_score += 2.5
        
        list_score = min(list_score, 20.0)
        score += list_score
        
        # =====================================================
        # FEATURE 3: Bold/emphasis usage (0-12 points)
        # =====================================================
        bold_score = 0.0
        
        # Bold text instances (not counting bold headers already counted)
        bold_instances = re.findall(r'\*\*[^*]+\*\*', response)
        num_bold = len(bold_instances)
        
        # Inline bold within paragraphs/list items (key term highlighting)
        inline_bold = 0
        for line in non_empty_lines:
            stripped = line.strip()
            # Line has bold but also has other text (not pure bold header)
            if '**' in stripped and not re.match(r'^\s*\*\*[^*]+\*\*\s*$', stripped):
                inline_bold += 1
        
        if num_bold >= 1:
            bold_score += 3.0
        if num_bold >= 3:
            bold_score += 3.0
        if num_bold >= 5:
            bold_score += 2.0
        if inline_bold >= 2:
            bold_score += 2.0
        if inline_bold >= 4:
            bold_score += 2.0
        
        bold_score = min(bold_score, 12.0)
        score += bold_score
        
        # =====================================================
        # FEATURE 4: Paragraph structure (0-15 points)
        # =====================================================
        para_score = 0.0
        
        # Count paragraphs (blocks of text separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Multiple paragraphs indicate good structure
        if num_paragraphs >= 2:
            para_score += 4.0
        if num_paragraphs >= 3:
            para_score += 3.0
        if num_paragraphs >= 4:
            para_score += 3.0
        if num_paragraphs >= 6:
            para_score += 2.0
        
        # Check for opening/intro paragraph followed by structured content
        if num_paragraphs >= 2:
            first_para = paragraphs[0]
            # Short intro paragraph (1-3 sentences) is good
            first_sentences = re.split(r'[.!?]+', first_para)
            first_sentences = [s for s in first_sentences if s.strip()]
            if 1 <= len(first_sentences) <= 4:
                para_score += 3.0
        
        para_score = min(para_score, 15.0)
        score += para_score
        
        # =====================================================
        # FEATURE 5: Whitespace and visual separation (0-10 points)
        # =====================================================
        ws_score = 0.0
        
        # Count blank lines
        blank_lines = total_lines - non_empty_count
        
        if non_empty_count > 0:
            blank_ratio = blank_lines / max(non_empty_count, 1)
        else:
            blank_ratio = 0
        
        # Good whitespace ratio (some blank lines for separation)
        if 0.1 <= blank_ratio <= 1.5:
            ws_score += 5.0
        elif blank_ratio > 0:
            ws_score += 2.0
        
        # Variety in line lengths suggests structure (headers, list items, paragraphs)
        if non_empty_count >= 3:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            avg_len = sum(line_lengths) / len(line_lengths)
            if avg_len > 0:
                variance = sum((l - avg_len) ** 2 for l in line_lengths) / len(line_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / avg_len  # coefficient of variation
                
                # Higher variation suggests mixed formatting (headers, bullets, paragraphs)
                if cv > 0.3:
                    ws_score += 3.0
                elif cv > 0.15:
                    ws_score += 1.5
        
        # Penalize single-block responses with no line breaks
        if total_lines <= 2 and len(response) > 200:
            ws_score -= 3.0
        
        ws_score = max(0, min(ws_score, 10.0))
        score += ws_score
        
        # =====================================================
        # FEATURE 6: Transition and flow words (0-8 points)
        # =====================================================
        flow_score = 0.0
        
        response_lower = response.lower()
        
        # Transition phrases indicating logical flow
        transitions = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bfinally\b', r'\bin addition\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin summary\b', r'\bin conclusion\b', r'\bto summarize\b',
            r'\bas a result\b', r'\btherefore\b', r'\bconsequently\b',
            r'\bhere are\b', r'\blet\'s\b', r'\bstep \d+\b',
            r'\bnext\b', r'\bthen\b', r'\balso\b',
            r'\bimportantly\b', r'\bnotably\b', r'\bkey\b',
        ]
        
        transition_count = 0
        for pattern in transitions:
            matches = re.findall(pattern, response_lower)
            transition_count += len(matches)
        
        if transition_count >= 1:
            flow_score += 2.0
        if transition_count >= 3:
            flow_score += 2.0
        if transition_count >= 5:
            flow_score += 2.0
        if transition_count >= 8:
            flow_score += 2.0
        
        flow_score = min(flow_score, 8.0)
        score += flow_score
        
        # =====================================================
        # FEATURE 7: Structural patterns and templates (0-10 points)
        # =====================================================
        template_score = 0.0
        
        # Check for "Step X:" or "Step X." patterns
        steps = re.findall(r'step\s+\d+', response_lower)
        if len(steps) >= 2:
            template_score += 3.0
        
        # Check for labeled sections (e.g., "Ingredients:", "Instructions:", "Note:")
        labeled_sections = re.findall(r'^[A-Z][A-Za-z\s]+:\s*$', response, re.MULTILINE)
        # Also markdown-style labeled sections
        md_labeled = re.findall(r'^#{1,6}\s+\d*\.?\s*\*?\*?[A-Z]', response, re.MULTILINE)
        
        section_count = len(labeled_sections) + len(md_labeled)
        if section_count >= 1:
            template_score += 2.0
        if section_count >= 3:
            template_score += 2.0
        
        # Check for code blocks (indicates technical formatting awareness)
        code_blocks = re.findall(r'```', response)
        if len(code_blocks) >= 2:
            template_score += 2.0
        
        # Check for separator patterns (---, ***, ___)
        separators = re.findall(r'^[-*_]{3,}\s*$', response, re.MULTILINE)
        if separators:
            template_score += 1.0
        
        template_score = min(template_score, 10.0)
        score += template_score
        
        # =====================================================
        # FEATURE 8: Wall-of-text penalty (0 to -15 points)
        # =====================================================
        wall_penalty = 0.0
        
        # Check for very long lines without breaks
        long_lines = [l for l in non_empty_lines if len(l.strip()) > 300]
        if long_lines:
            wall_penalty -= 3.0 * min(len(long_lines), 3)
        
        # Single paragraph with lots of text
        if num_paragraphs == 1 and len(response) > 300:
            wall_penalty -= 5.0
        if num_paragraphs == 1 and len(response) > 600:
            wall_penalty -= 5.0
        
        # No formatting at all in a long response
        if len(response) > 200 and total_headers == 0 and total_list_items == 0 and num_bold == 0 and num_paragraphs <= 1:
            wall_penalty -= 5.0
        
        wall_penalty = max(wall_penalty, -15.0)
        score += wall_penalty
        
        # =====================================================
        # FEATURE 9: Response length appropriateness (0-5 points)
        # =====================================================
        len_score = 0.0
        resp_len = len(response)
        
        # Very short responses get less credit for structure (less opportunity)
        # But if they're well-formatted for their size, that's fine
        if resp_len >= 100:
            len_score += 1.0
        if resp_len >= 200:
            len_score += 1.0
        if resp_len >= 400:
            len_score += 1.5
        if resp_len >= 600:
            len_score += 1.5
        
        # Very long responses without structure get penalized elsewhere
        len_score = min(len_score, 5.0)
        score += len_score
        
        # =====================================================
        # FEATURE 10: Consistency of formatting (0-5 points)
        # =====================================================
        consistency_score = 0.0
        
        # If using numbered lists, check if numbers are sequential
        if num_numbered >= 2:
            numbers = re.findall(r'^\s*(\d+)[\.\)]\s', response, re.MULTILINE)
            if numbers:
                nums = [int(n) for n in numbers]
                # Check if roughly sequential
                is_sequential = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
                if is_sequential:
                    consistency_score += 2.5
                else:
                    consistency_score += 1.0
        
        # If using headers, check if they follow a consistent pattern
        if num_md_headers >= 2:
            header_levels = [len(re.match(r'^(#+)', h).group(1)) for h in md_headers]
            # Consistent header levels is good
            if len(set(header_levels)) <= 2:
                consistency_score += 2.5
            else:
                consistency_score += 1.0
        
        consistency_score = min(consistency_score, 5.0)
        score += consistency_score
        
        # =====================================================
        # Final normalization and clamping
        # =====================================================
        # Maximum theoretical score: 15+20+12+15+10+8+10+0+5+5 = 100
        # Minimum: 0 - 15 = -15 (clamp to 0)
        
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 0:
                return 25.0
            return 0.0
        except:
            return 0.0