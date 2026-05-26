def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    
    This variant focuses on:
    - Hierarchical depth analysis (nested structure detection)
    - Information density distribution (entropy-like measure across sections)
    - Visual scanning patterns (how easy it is to scan/skim)
    - Structural variety score (mix of different formatting elements)
    - Proportion-based formatting metrics (ratio of formatted vs unformatted content)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        score = 0.0
        
        # === 1. HIERARCHICAL DEPTH ANALYSIS ===
        # Detect different levels of hierarchy and reward depth
        hierarchy_elements = {
            'h1_markdown': re.findall(r'^#{1}\s+\S', response, re.MULTILINE),
            'h2_markdown': re.findall(r'^#{2}\s+\S', response, re.MULTILINE),
            'h3_markdown': re.findall(r'^#{3,}\s+\S', response, re.MULTILINE),
            'bold_headers': re.findall(r'^\*\*[^*]+\*\*\s*$', response, re.MULTILINE),
            'numbered_top': re.findall(r'^\d+\.\s+', response, re.MULTILINE),
            'numbered_sub': re.findall(r'^\s+\d+\.\s+', response, re.MULTILINE),
            'bullet_top': re.findall(r'^[-*•]\s+', response, re.MULTILINE),
            'bullet_sub': re.findall(r'^\s+[-*•]\s+', response, re.MULTILINE),
            'lettered': re.findall(r'^\s*[a-z]\)\s+', response, re.MULTILINE),
        }
        
        distinct_levels = sum(1 for v in hierarchy_elements.values() if len(v) > 0)
        hierarchy_score = min(distinct_levels * 2.5, 15.0)
        score += hierarchy_score
        
        # === 2. STRUCTURAL VARIETY INDEX ===
        # Measure the diversity of formatting techniques used
        formatting_types_present = []
        
        # Check for bold text (not just headers)
        bold_inline = re.findall(r'\*\*[^*\n]{2,60}\*\*', response)
        if bold_inline:
            formatting_types_present.append('bold')
        
        # Check for inline code
        if re.search(r'`[^`]+`', response):
            formatting_types_present.append('inline_code')
        
        # Check for code blocks
        if re.search(r'```', response):
            formatting_types_present.append('code_block')
        
        # Check for markdown headers
        if re.search(r'^#{1,6}\s', response, re.MULTILINE):
            formatting_types_present.append('headers')
        
        # Check for numbered lists
        if re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE):
            formatting_types_present.append('numbered_list')
        
        # Check for bullet lists
        if re.search(r'^\s*[-*•]\s', response, re.MULTILINE):
            formatting_types_present.append('bullet_list')
        
        # Check for colons used as label-value pairs
        colon_pairs = re.findall(r'^[-*•\d.)\s]*\*?\*?[A-Z][^:]{2,30}:\*?\*?\s', response, re.MULTILINE)
        if len(colon_pairs) >= 2:
            formatting_types_present.append('label_value')
        
        # Check for parenthetical clarifications
        if re.search(r'\([^)]{5,}\)', response):
            formatting_types_present.append('parentheticals')
        
        variety_score = min(len(formatting_types_present) * 2.0, 12.0)
        score += variety_score
        
        # === 3. VISUAL SCANABILITY ANALYSIS ===
        # Measure how easy it is to visually scan the response
        
        # 3a. Line length variance - good formatting has varied line lengths
        line_lengths = [len(l) for l in non_empty_lines]
        if len(line_lengths) > 2:
            mean_len = sum(line_lengths) / len(line_lengths)
            variance = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            # Higher std_dev means more visual variety (headers, lists, paragraphs mixed)
            scanability_variance = min(std_dev / 15.0, 5.0)
        else:
            scanability_variance = 0.0
        
        # 3b. Short lines ratio (lines < 80 chars that aren't empty) - indicates list items, headers
        if non_empty_lines:
            short_lines = sum(1 for l in non_empty_lines if len(l.strip()) < 80)
            short_ratio = short_lines / len(non_empty_lines)
            # We want a balanced ratio - not all short (too sparse) nor all long (wall of text)
            if 0.2 <= short_ratio <= 0.8:
                scanability_short = 5.0
            elif short_ratio > 0.8:
                scanability_short = 3.0
            else:
                scanability_short = 1.0
        else:
            scanability_short = 0.0
        
        # 3c. "Entry points" - lines that start with visual markers
        entry_points = 0
        for line in non_empty_lines:
            stripped = line.strip()
            if re.match(r'^(#{1,6}\s|\d+[\.\)]\s|[-*•]\s|\*\*)', stripped):
                entry_points += 1
        
        if non_empty_lines:
            entry_ratio = entry_points / len(non_empty_lines)
            entry_score = min(entry_ratio * 12.0, 8.0)
        else:
            entry_score = 0.0
        
        score += scanability_variance + scanability_short + entry_score
        
        # === 4. INFORMATION DENSITY DISTRIBUTION ===
        # Split response into chunks and measure if content is evenly distributed
        # (vs front-loaded or back-loaded)
        
        if total_chars > 100:
            # Split into ~4 equal chunks
            chunk_size = total_chars // 4
            chunks = []
            for i in range(4):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < 3 else total_chars
                chunks.append(response[start:end])
            
            # Count formatting elements per chunk
            chunk_format_counts = []
            for chunk in chunks:
                count = 0
                count += len(re.findall(r'#{1,6}\s', chunk))
                count += len(re.findall(r'\d+[\.\)]\s', chunk))
                count += len(re.findall(r'[-*•]\s', chunk))
                count += len(re.findall(r'\*\*', chunk)) // 2
                chunk_format_counts.append(count)
            
            total_format = sum(chunk_format_counts)
            if total_format > 0:
                # Calculate entropy of distribution
                probs = [(c / total_format) if total_format > 0 else 0.25 for c in chunk_format_counts]
                entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
                max_entropy = math.log2(4)  # 2.0
                # Normalized entropy: higher = more evenly distributed
                distribution_score = (entropy / max_entropy) * 8.0
            else:
                distribution_score = 0.0
        else:
            distribution_score = 0.0
        
        score += distribution_score
        
        # === 5. PARAGRAPH STRUCTURE QUALITY ===
        # Analyze paragraph breaks and their appropriateness
        
        # Split by double newlines to find paragraphs/sections
        sections = re.split(r'\n\s*\n', response)
        sections = [s.strip() for s in sections if s.strip()]
        num_sections = len(sections)
        
        # Reward having multiple sections (but not too fragmented)
        if num_sections == 1:
            section_score = 0.0  # Wall of text
        elif 2 <= num_sections <= 4:
            section_score = 4.0
        elif 5 <= num_sections <= 10:
            section_score = 7.0
        elif 11 <= num_sections <= 20:
            section_score = 5.0
        else:
            section_score = 3.0  # Too fragmented
        
        score += section_score
        
        # === 6. OPENING AND CLOSING STRUCTURE ===
        # Good responses often have an intro and structured body
        
        # Check if response starts with an introductory sentence (not immediately a list)
        first_line = non_empty_lines[0].strip() if non_empty_lines else ""
        starts_with_intro = False
        if first_line and not re.match(r'^(\d+[\.\)]\s|[-*•]\s|#{1,6}\s)', first_line):
            if len(first_line) > 30:
                starts_with_intro = True
        
        # Check if formatting appears after intro (structured body)
        has_structured_body = False
        if len(non_empty_lines) > 2:
            body_lines = non_empty_lines[1:]
            for line in body_lines:
                if re.match(r'^\s*(#{1,6}\s|\d+[\.\)]\s|[-*•]\s|\*\*)', line.strip()):
                    has_structured_body = True
                    break
        
        if starts_with_intro and has_structured_body:
            intro_body_score = 8.0
        elif starts_with_intro:
            intro_body_score = 3.0
        elif has_structured_body:
            intro_body_score = 5.0
        else:
            intro_body_score = 1.0
        
        score += intro_body_score
        
        # === 7. WALL-OF-TEXT PENALTY ===
        # Penalize responses that are one giant paragraph
        
        if num_sections == 1 and total_chars > 200:
            # Check if it's truly a wall of text (no formatting at all)
            any_formatting = bool(re.search(r'(#{1,6}\s|\d+[\.\)]\s|[-*•]\s|\*\*)', response))
            if not any_formatting:
                wall_penalty = -10.0
            else:
                wall_penalty = -3.0
        else:
            wall_penalty = 0.0
        
        score += wall_penalty
        
        # === 8. CONSISTENT FORMATTING PATTERN ===
        # Reward responses that use consistent formatting (e.g., all list items formatted same way)
        
        # Check numbered list consistency
        numbered_items = re.findall(r'^(\d+)[\.\)]\s', response, re.MULTILINE)
        if len(numbered_items) >= 2:
            # Check if numbers are sequential
            nums = [int(n) for n in numbered_items]
            is_sequential = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
            if is_sequential:
                consistency_score = min(len(numbered_items) * 0.8, 5.0)
            else:
                consistency_score = 1.0
        else:
            consistency_score = 0.0
        
        # Check bullet list consistency
        bullet_items = re.findall(r'^\s*[-*•]\s', response, re.MULTILINE)
        if len(bullet_items) >= 2:
            consistency_score = max(consistency_score, min(len(bullet_items) * 0.7, 5.0))
        
        score += consistency_score
        
        # === 9. CONTENT-TO-CHROME RATIO ===
        # Ensure formatting serves the content (not over-formatted for short responses)
        
        format_chars = 0
        format_chars += len(re.findall(r'[#*\-•]', response))
        content_chars = total_chars - format_chars
        
        if total_chars > 0:
            chrome_ratio = format_chars / total_chars
            if 0.02 <= chrome_ratio <= 0.15:
                ratio_score = 4.0  # Good balance
            elif chrome_ratio < 0.02:
                ratio_score = 1.0  # Too little formatting
            else:
                ratio_score = 2.0  # Over-formatted
        else:
            ratio_score = 0.0
        
        score += ratio_score
        
        # === 10. RESPONSE LENGTH CONTEXT ADJUSTMENT ===
        # Longer responses benefit more from formatting
        if total_chars > 500 and distinct_levels < 2 and len(formatting_types_present) < 2:
            score -= 5.0  # Long response with no formatting
        
        # Very short responses shouldn't be penalized too much for lack of formatting
        if total_chars < 150:
            score = max(score, 20.0) if num_sections <= 2 else score
        
        # Normalize to 0-100 range
        # Max theoretical: ~15 + 12 + 5 + 5 + 8 + 8 + 7 + 8 + 0 + 5 + 4 = ~77
        score = max(0.0, min(score, 80.0))
        normalized_score = (score / 80.0) * 100.0
        
        return round(normalized_score, 2)
        
    except Exception as e:
        return 25.0  # Default middle-low score on error