def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical document structure analysis approach.
    
    This variant focuses on:
    - Document tree depth (nested structures, indentation levels)
    - Visual rhythm analysis (alternating patterns of dense/sparse content)
    - Structural diversity score (variety of formatting elements used)
    - Content-to-chrome ratio (meaningful content vs formatting overhead)
    - Semantic segmentation quality (how well content is chunked)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        import re
        import math
        from collections import Counter
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # ============================================================
        # 1. STRUCTURAL DIVERSITY INDEX (0-15)
        # Count how many DIFFERENT formatting mechanisms are used
        # ============================================================
        formatting_types_found = set()
        
        # Check for markdown headers
        if re.search(r'^#{1,6}\s+\S', response, re.MULTILINE):
            formatting_types_found.add('md_header')
        
        # Check for bold text
        if re.search(r'\*\*[^*]+\*\*', response) or re.search(r'__[^_]+__', response):
            formatting_types_found.add('bold')
        
        # Check for italic text
        if re.search(r'(?<!\*)\*(?!\*)[^*]+(?<!\*)\*(?!\*)', response):
            formatting_types_found.add('italic')
        
        # Check for numbered lists
        if re.search(r'^\s*\d+[\.\)]\s+\S', response, re.MULTILINE):
            formatting_types_found.add('numbered_list')
        
        # Check for bullet lists (various markers)
        if re.search(r'^\s*[-*•◦▪►→]\s+\S', response, re.MULTILINE):
            formatting_types_found.add('bullet_list')
        
        # Check for code blocks
        if re.search(r'```', response):
            formatting_types_found.add('code_block')
        
        # Check for inline code
        if re.search(r'`[^`]+`', response):
            formatting_types_found.add('inline_code')
        
        # Check for colon-based definitions/labels
        if re.search(r'^[A-Z][^:]{2,30}:\s+\S', response, re.MULTILINE):
            formatting_types_found.add('label_value')
        
        # Check for parenthetical asides
        if re.search(r'\([^)]{10,}\)', response):
            formatting_types_found.add('parenthetical')
        
        # Check for quotation marks or block quotes
        if re.search(r'^>\s+', response, re.MULTILINE):
            formatting_types_found.add('blockquote')
        
        # Check for URLs/links
        if re.search(r'\[([^\]]+)\]\(([^)]+)\)', response):
            formatting_types_found.add('link')
        
        # Check for tables
        if re.search(r'\|.*\|.*\|', response):
            formatting_types_found.add('table')
        
        # Check for em-dash or en-dash usage for structure
        if re.search(r'[—–]\s', response):
            formatting_types_found.add('dash_structure')
        
        diversity_count = len(formatting_types_found)
        # Diminishing returns: first few types matter more
        diversity_score = min(15, diversity_count * 3.5 - max(0, diversity_count - 3) * 1.0)
        diversity_score = max(0, diversity_score)
        
        # ============================================================
        # 2. VISUAL RHYTHM ANALYSIS (0-20)
        # Analyze the pattern of line lengths to detect good visual structure
        # ============================================================
        line_lengths = [len(l) for l in lines]
        
        if len(line_lengths) < 2:
            rhythm_score = 2.0
        else:
            # Categorize each line: empty(0), short(<40), medium(40-100), long(>100)
            categories = []
            for ll in line_lengths:
                stripped = lines[len(categories)].strip() if len(categories) < len(lines) else ''
                if ll == 0 or (stripped == ''):
                    categories.append('E')  # empty
                elif ll < 40:
                    categories.append('S')  # short
                elif ll < 100:
                    categories.append('M')  # medium
                else:
                    categories.append('L')  # long
            
            cat_str = ''.join(categories)
            
            # Count transitions between categories (more transitions = more visual variety)
            transitions = sum(1 for i in range(1, len(categories)) if categories[i] != categories[i-1])
            transition_rate = transitions / max(1, len(categories) - 1)
            
            # Penalize all-same patterns (wall of text)
            unique_cats = len(set(categories))
            
            # Check for paragraph breaks (empty lines between content)
            para_breaks = cat_str.count('E')
            content_lines_count = len([c for c in categories if c != 'E'])
            
            # Good rhythm: mix of categories with periodic breaks
            rhythm_score = 0.0
            rhythm_score += min(6, transition_rate * 12)  # reward transitions
            rhythm_score += min(4, unique_cats * 1.5)  # reward variety
            
            # Reward paragraph breaks proportional to content
            if content_lines_count > 0:
                break_ratio = para_breaks / content_lines_count
                # Sweet spot: ~0.15-0.4 break ratio
                if 0.1 <= break_ratio <= 0.5:
                    rhythm_score += 5
                elif 0.05 <= break_ratio < 0.1 or 0.5 < break_ratio <= 0.7:
                    rhythm_score += 3
                elif break_ratio > 0:
                    rhythm_score += 1
            
            # Detect repeating structural patterns (e.g., header-content-break)
            # Look for 2-3 length repeating motifs
            if len(cat_str) >= 6:
                for motif_len in [2, 3, 4]:
                    motifs = [cat_str[i:i+motif_len] for i in range(0, len(cat_str) - motif_len + 1, motif_len)]
                    if len(motifs) >= 2:
                        most_common_motif = Counter(motifs).most_common(1)[0]
                        if most_common_motif[1] >= 2:
                            rhythm_score += min(3, most_common_motif[1] * 0.8)
                            break
            
            rhythm_score = min(20, max(0, rhythm_score))
        
        # ============================================================
        # 3. HIERARCHICAL DEPTH ANALYSIS (0-15)
        # Measure indentation levels and nesting depth
        # ============================================================
        indent_levels = set()
        for line in non_empty_lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            indent_levels.add(indent)
        
        # Count distinct indentation levels
        num_indent_levels = len(indent_levels)
        
        # Check for nested list items
        nested_items = 0
        for line in lines:
            if re.match(r'^\s{2,}[-*•]\s', line) or re.match(r'^\s{2,}\d+[\.\)]\s', line):
                nested_items += 1
        
        # Check for header hierarchy (different header levels)
        header_levels = set()
        for line in lines:
            hm = re.match(r'^(#{1,6})\s', line.strip())
            if hm:
                header_levels.add(len(hm.group(1)))
        
        depth_score = 0.0
        depth_score += min(5, (num_indent_levels - 1) * 2)  # indent variety
        depth_score += min(4, nested_items * 1.5)  # nested items
        depth_score += min(4, len(header_levels) * 2.5)  # header hierarchy
        
        # Bonus for having both headers and lists (clear hierarchy)
        has_headers = len(header_levels) > 0 or bool(re.search(r'^[A-Z][^.!?]{3,40}$', response, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*•]\s|\d+[\.\)]\s', response, re.MULTILINE))
        if has_headers and has_lists:
            depth_score += 3
        
        depth_score = min(15, max(0, depth_score))
        
        # ============================================================
        # 4. SEMANTIC CHUNKING QUALITY (0-20)
        # Analyze how well content is segmented into meaningful chunks
        # ============================================================
        
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        chunk_score = 0.0
        
        if num_paragraphs == 1:
            # Single block - check if it's short enough to not need structure
            if total_chars < 200:
                chunk_score = 8.0  # acceptable for short responses
            elif total_chars < 400:
                chunk_score = 4.0
            else:
                chunk_score = 1.0  # wall of text penalty
        else:
            # Multiple paragraphs - analyze their sizes
            para_lengths = [len(p) for p in paragraphs]
            
            if para_lengths:
                avg_para_len = sum(para_lengths) / len(para_lengths)
                
                # Ideal paragraph size: 80-300 chars
                if 60 <= avg_para_len <= 350:
                    chunk_score += 7
                elif 40 <= avg_para_len <= 500:
                    chunk_score += 4
                else:
                    chunk_score += 2
                
                # Consistency of paragraph sizes (coefficient of variation)
                if len(para_lengths) >= 2:
                    mean_pl = sum(para_lengths) / len(para_lengths)
                    variance = sum((x - mean_pl) ** 2 for x in para_lengths) / len(para_lengths)
                    std_pl = math.sqrt(variance)
                    cv = std_pl / max(1, mean_pl)
                    
                    # Some variation is good (0.3-0.7), too much or too little is bad
                    if 0.2 <= cv <= 0.8:
                        chunk_score += 5
                    elif cv < 0.2:
                        chunk_score += 3  # too uniform
                    elif cv <= 1.2:
                        chunk_score += 3
                    else:
                        chunk_score += 1  # too varied
                
                # Reward appropriate number of paragraphs for content length
                expected_paras = max(1, total_chars / 250)
                para_ratio = num_paragraphs / expected_paras
                if 0.5 <= para_ratio <= 2.0:
                    chunk_score += 5
                elif 0.3 <= para_ratio <= 3.0:
                    chunk_score += 3
                else:
                    chunk_score += 1
                
                # Check if paragraphs start with topic sentences (capitalized, declarative)
                topic_sentence_count = 0
                for p in paragraphs:
                    first_line = p.split('\n')[0].strip()
                    # Starts with capital, is a reasonable sentence
                    if (first_line and first_line[0].isupper() and 
                        len(first_line) > 20 and 
                        not first_line.startswith(('-', '*', '•', '#', '`'))):
                        topic_sentence_count += 1
                
                if num_paragraphs > 0:
                    topic_ratio = topic_sentence_count / num_paragraphs
                    chunk_score += min(3, topic_ratio * 4)
        
        chunk_score = min(20, max(0, chunk_score))
        
        # ============================================================
        # 5. INFORMATION DENSITY DISTRIBUTION (0-15)
        # Check if information is evenly distributed, not front/back-loaded
        # ============================================================
        
        density_score = 0.0
        
        if total_chars > 100 and len(non_empty_lines) >= 3:
            # Split response into thirds and compare information density
            third = total_chars // 3
            parts = [response[:third], response[third:2*third], response[2*third:]]
            
            # Use word count per character as a rough density proxy
            part_densities = []
            for part in parts:
                words = len(part.split())
                chars = max(1, len(part))
                # Count "information markers": numbers, proper nouns, technical terms
                info_markers = len(re.findall(r'\b[A-Z][a-z]+\b', part))
                info_markers += len(re.findall(r'\b\d+\b', part))
                density = (words + info_markers * 0.5) / chars
                part_densities.append(density)
            
            if all(d > 0 for d in part_densities):
                max_d = max(part_densities)
                min_d = min(part_densities)
                balance = min_d / max(0.001, max_d)
                
                # Well-balanced distribution
                if balance > 0.6:
                    density_score += 8
                elif balance > 0.4:
                    density_score += 5
                elif balance > 0.2:
                    density_score += 3
                else:
                    density_score += 1
            
            # Check for conclusion/summary presence (structural completeness)
            last_para = paragraphs[-1] if paragraphs else ''
            conclusion_markers = ['in summary', 'overall', 'in conclusion', 'to summarize',
                                  'ultimately', 'in short', 'the key', 'to sum up',
                                  'hope this', 'good luck', 'let me know']
            has_conclusion = any(marker in last_para.lower() for marker in conclusion_markers)
            if has_conclusion:
                density_score += 3
            
            # Check for introduction/context setting
            first_para = paragraphs[0] if paragraphs else ''
            if len(first_para) > 30 and len(paragraphs) > 1:
                # First paragraph should be shorter or equal to body paragraphs
                if len(first_para) < sum(len(p) for p in paragraphs) / len(paragraphs) * 1.5:
                    density_score += 2
            
            # Reward responses that have clear opening
            opening_patterns = [r'^(Sure|Great|Yes|No|Here|Let me|To answer|The |This |In |First)',
                               r'^(Essentially|Basically|Simply|Generally|Typically)']
            for pat in opening_patterns:
                if re.match(pat, response.strip()):
                    density_score += 1
                    break
        else:
            density_score = 5.0  # neutral for short responses
        
        density_score = min(15, max(0, density_score))
        
        # ============================================================
        # 6. RESPONSE LENGTH APPROPRIATENESS (0-10)
        # Longer, well-structured responses tend to score higher
        # ============================================================
        length_score = 0.0
        
        word_count = len(response.split())
        
        if word_count < 15:
            length_score = 1.0
        elif word_count < 30:
            length_score = 3.0
        elif word_count < 60:
            length_score = 5.0
        elif word_count < 120:
            length_score = 7.0
        elif word_count < 300:
            length_score = 9.0
        elif word_count < 600:
            length_score = 10.0
        else:
            length_score = 8.0  # very long might be verbose
        
        # But penalize long responses without structure
        if word_count > 80 and num_paragraphs <= 1 and diversity_count == 0:
            length_score *= 0.5
        
        # ============================================================
        # 7. SENTENCE-LEVEL STRUCTURE (0-5)
        # Check sentence length variation within paragraphs
        # ============================================================
        sentences = re.split(r'[.!?]+\s+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        sentence_score = 0.0
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            if mean_sl > 0:
                var_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
                cv_sl = math.sqrt(var_sl) / mean_sl
                # Good sentence variety: cv between 0.3 and 0.8
                if 0.25 <= cv_sl <= 0.9:
                    sentence_score = 5
                elif 0.15 <= cv_sl <= 1.2:
                    sentence_score = 3
                else:
                    sentence_score = 1.5
        elif len(sentences) >= 1:
            sentence_score = 2.0
        
        # ============================================================
        # FINAL SCORE COMPOSITION
        # ============================================================
        total = (
            diversity_score +      # 0-15
            rhythm_score +         # 0-20
            depth_score +          # 0-15
            chunk_score +          # 0-20
            density_score +        # 0-15
            length_score +         # 0-10
            sentence_score         # 0-5
        )
        # Max possible: 100
        
        # Normalize to 0-10 scale
        final_score = total / 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0