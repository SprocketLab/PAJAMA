def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant focuses on a HIERARCHICAL STRUCTURE ANALYSIS approach:
    - Measures the depth and consistency of structural hierarchy
    - Analyzes visual rhythm (alternation between different element types)
    - Scores the "scanability" of the response (how easy it is to skim)
    - Evaluates proportionality of structural elements relative to content length
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 0.0
        
        import re
        from collections import Counter
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        total_lines = len(non_empty_lines)
        
        if total_lines == 0:
            return 0.0
        
        # === FEATURE 1: Structural Hierarchy Depth Score ===
        # Classify each line into a structural type
        LINE_TYPES = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                LINE_TYPES.append('blank')
            elif re.match(r'^#{1,6}\s+', stripped):
                level = len(re.match(r'^(#{1,6})\s+', stripped).group(1))
                LINE_TYPES.append(f'header_{level}')
            elif re.match(r'^\*\*[^*]+\*\*\s*$', stripped):
                LINE_TYPES.append('bold_header')
            elif re.match(r'^(\d+)\.\s+\*\*', stripped):
                LINE_TYPES.append('numbered_bold')
            elif re.match(r'^(\d+)\.\s+', stripped):
                LINE_TYPES.append('numbered')
            elif re.match(r'^[-*•]\s+\*\*', stripped):
                LINE_TYPES.append('bullet_bold')
            elif re.match(r'^[-*•]\s+', stripped):
                LINE_TYPES.append('bullet')
            elif re.match(r'^\s{2,}[-*•]\s+', line):
                LINE_TYPES.append('sub_bullet')
            elif re.match(r'^\s{2,}\d+\.\s+', line):
                LINE_TYPES.append('sub_numbered')
            elif stripped.startswith('```'):
                LINE_TYPES.append('code_fence')
            elif re.match(r'^\s{4,}', line) and len(stripped) > 0:
                LINE_TYPES.append('indented')
            else:
                LINE_TYPES.append('prose')
        
        # Count unique structural types (excluding blank and prose)
        structural_types = [t for t in LINE_TYPES if t not in ('blank', 'prose')]
        unique_structural = len(set(structural_types))
        
        # Hierarchy depth: how many distinct levels of structure
        hierarchy_depth = min(unique_structural, 5)
        hierarchy_score = hierarchy_depth * 3.0  # 0-15
        
        # === FEATURE 2: Visual Rhythm Score ===
        # Measures how well the response alternates between structure and content
        # Good rhythm: header -> content -> list -> content -> header -> content
        # Bad rhythm: prose prose prose prose (monotonous)
        
        simplified_types = []
        for t in LINE_TYPES:
            if t == 'blank':
                simplified_types.append('space')
            elif t == 'prose':
                simplified_types.append('text')
            else:
                simplified_types.append('structure')
        
        # Count transitions between different simplified types
        transitions = 0
        for i in range(1, len(simplified_types)):
            if simplified_types[i] != simplified_types[i-1]:
                transitions += 1
        
        max_transitions = max(len(simplified_types) - 1, 1)
        rhythm_ratio = transitions / max_transitions if max_transitions > 0 else 0
        
        # Ideal rhythm ratio is around 0.4-0.7 (not too choppy, not too monotonous)
        if rhythm_ratio < 0.1:
            rhythm_score = 1.0
        elif rhythm_ratio < 0.25:
            rhythm_score = 4.0
        elif rhythm_ratio < 0.4:
            rhythm_score = 7.0
        elif rhythm_ratio <= 0.7:
            rhythm_score = 10.0
        elif rhythm_ratio <= 0.85:
            rhythm_score = 7.0
        else:
            rhythm_score = 5.0
        
        # === FEATURE 3: Scanability Score ===
        # How easy is it to skim the response and extract key information?
        
        # Count "anchor points" - bold text, headers, numbered items, bullets
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        header_count = len(re.findall(r'^#{1,6}\s+', response, re.MULTILINE))
        numbered_count = len(re.findall(r'^\s*\d+\.\s+', response, re.MULTILINE))
        bullet_count = len(re.findall(r'^\s*[-*•]\s+', response, re.MULTILINE))
        
        total_anchors = bold_count + header_count * 2 + numbered_count + bullet_count
        
        # Anchors per 100 characters (normalized)
        anchor_density = (total_anchors / max(total_chars, 1)) * 100
        
        # Ideal anchor density depends on response length
        if total_chars < 200:
            ideal_density = 1.0
        elif total_chars < 500:
            ideal_density = 2.0
        else:
            ideal_density = 2.5
        
        density_diff = abs(anchor_density - ideal_density)
        if density_diff < 0.5:
            scanability_score = 12.0
        elif density_diff < 1.0:
            scanability_score = 10.0
        elif density_diff < 2.0:
            scanability_score = 7.0
        elif density_diff < 3.0:
            scanability_score = 4.0
        else:
            scanability_score = 2.0
        
        # Bonus for having anchors at all vs pure text
        if total_anchors == 0:
            scanability_score = max(scanability_score - 5, 0)
        
        # === FEATURE 4: Proportionality Score ===
        # Are structural elements well-proportioned to content?
        
        # Check paragraph lengths (text between structural elements)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if len(paragraphs) >= 2:
            para_lengths = [len(p) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            
            # Check variance - are paragraphs roughly similar in length?
            if avg_para_len > 0:
                variance = sum((l - avg_para_len) ** 2 for l in para_lengths) / len(para_lengths)
                cv = (variance ** 0.5) / avg_para_len  # coefficient of variation
            else:
                cv = 0
            
            # Some variation is good (different section sizes), but extreme is bad
            if cv < 0.3:
                proportion_score = 6.0  # very uniform, slightly boring but ok
            elif cv < 0.8:
                proportion_score = 8.0  # good variety
            elif cv < 1.5:
                proportion_score = 5.0  # getting unbalanced
            else:
                proportion_score = 2.0  # very unbalanced
            
            # Bonus for having multiple paragraphs (vs wall of text)
            num_paras = len(paragraphs)
            if total_chars > 300:
                if num_paras >= 4:
                    proportion_score += 4.0
                elif num_paras >= 3:
                    proportion_score += 3.0
                elif num_paras >= 2:
                    proportion_score += 1.5
                else:
                    proportion_score -= 2.0
            elif total_chars > 150:
                if num_paras >= 2:
                    proportion_score += 2.0
        else:
            # Single block of text
            if total_chars > 300:
                proportion_score = 1.0  # wall of text penalty
            elif total_chars > 150:
                proportion_score = 4.0
            else:
                proportion_score = 6.0  # short response, single paragraph is fine
        
        # === FEATURE 5: Opening and Closing Structure ===
        # Good responses often have an intro, body, and conclusion-like structure
        
        framing_score = 0.0
        
        # Check for an introductory line/paragraph (short, before lists/structure)
        if len(paragraphs) >= 2:
            first_para = paragraphs[0]
            has_structural = bool(re.search(r'(^\d+\.|^[-*•]|^#{1,6}\s)', first_para, re.MULTILINE))
            if not has_structural and len(first_para) < 300:
                framing_score += 3.0  # good intro paragraph
        
        # Check if response starts with a direct, engaging opening
        first_line = non_empty_lines[0] if non_empty_lines else ""
        if first_line and not re.match(r'^\d+\.\s|^[-*•]\s|^#{1,6}\s', first_line.strip()):
            framing_score += 1.0  # starts with prose, not jumping into a list
        
        # === FEATURE 6: Consistent List Formatting ===
        # If lists are used, are they consistent?
        
        consistency_score = 0.0
        
        # Check numbered list consistency
        numbered_items = re.findall(r'^(\s*)\d+\.\s+', response, re.MULTILINE)
        if len(numbered_items) >= 2:
            # Check if indentation is consistent
            indents = [len(n) for n in numbered_items]
            if len(set(indents)) <= 2:  # at most 2 indent levels
                consistency_score += 3.0
            else:
                consistency_score += 1.0
        
        # Check bullet consistency
        bullet_items = re.findall(r'^(\s*)([-*•])\s+', response, re.MULTILINE)
        if len(bullet_items) >= 2:
            markers = [b[1] for b in bullet_items]
            # Consistent marker usage
            if len(set(markers)) == 1:
                consistency_score += 3.0
            else:
                consistency_score += 1.0
        
        # If no lists, give a neutral score for short responses, slight penalty for long ones
        if numbered_count == 0 and bullet_count == 0:
            if total_chars > 400:
                consistency_score = -1.0  # long response with no lists at all
            else:
                consistency_score = 1.0
        
        # === FEATURE 7: Line Length Distribution (readability) ===
        # Extremely long lines hurt readability
        
        line_lengths = [len(l) for l in non_empty_lines]
        avg_line_len = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        max_line_len = max(line_lengths) if line_lengths else 0
        
        readability_score = 5.0
        # Penalize very long unbroken lines
        long_lines = sum(1 for l in line_lengths if l > 200)
        if long_lines > 0:
            readability_score -= min(long_lines * 1.5, 4.0)
        
        # Reward variety in line lengths (indicates mixed formatting)
        if len(line_lengths) >= 3:
            unique_length_buckets = len(set(l // 20 for l in line_lengths))
            if unique_length_buckets >= 4:
                readability_score += 2.0
            elif unique_length_buckets >= 3:
                readability_score += 1.0
        
        # === FEATURE 8: Markdown/Formatting Sophistication ===
        # Use of advanced formatting features
        
        sophistication_score = 0.0
        
        # Nested structures (sub-items under items)
        sub_items = len(re.findall(r'^\s{2,}[-*•]\s+', response, re.MULTILINE))
        sub_items += len(re.findall(r'^\s{2,}\d+\.\s+', response, re.MULTILINE))
        if sub_items > 0:
            sophistication_score += min(sub_items, 3) * 1.0
        
        # Bold within list items (labeled lists)
        bold_in_lists = len(re.findall(r'^\s*(\d+\.|\s*[-*•])\s+\*\*[^*]+\*\*', response, re.MULTILINE))
        if bold_in_lists >= 2:
            sophistication_score += min(bold_in_lists, 4) * 1.5
        
        # Headers with markdown
        if header_count >= 1:
            sophistication_score += min(header_count, 3) * 1.5
        
        # Code blocks
        code_blocks = len(re.findall(r'```', response))
        if code_blocks >= 2:
            sophistication_score += 2.0
        
        # LaTeX/math formatting
        math_blocks = len(re.findall(r'\\\[|\\\(|\\frac|\\text', response))
        if math_blocks > 0:
            sophistication_score += min(math_blocks, 3) * 0.5
        
        sophistication_score = min(sophistication_score, 10.0)
        
        # === FEATURE 9: Content-to-Query Appropriateness of Structure ===
        # Short factual queries might not need heavy formatting
        
        query_len = len(query.strip())
        query_words = len(query.strip().split())
        
        # Detect if query is asking for a list/multiple items
        list_query = bool(re.search(r'\b(list|ideas|suggestions|steps|ways|tips|reasons|examples|options|things)\b', query.lower()))
        how_query = bool(re.search(r'\b(how|explain|describe|what are)\b', query.lower()))
        
        appropriateness_bonus = 0.0
        
        if list_query and (numbered_count >= 2 or bullet_count >= 2):
            appropriateness_bonus = 3.0
        elif how_query and (numbered_count >= 2 or header_count >= 1):
            appropriateness_bonus = 2.0
        elif not list_query and not how_query and total_chars < 300:
            # Short factual answer - don't penalize lack of structure
            if total_anchors == 0:
                appropriateness_bonus = 2.0  # simple answer is fine
        
        # === COMBINE ALL SCORES ===
        
        raw_score = (
            hierarchy_score * 1.0 +      # 0-15
            rhythm_score * 1.0 +          # 0-10
            scanability_score * 1.0 +     # 0-12
            proportion_score * 1.0 +      # ~-1 to 12
            framing_score * 1.0 +         # 0-4
            consistency_score * 1.0 +     # -1 to 6
            readability_score * 1.0 +     # ~1 to 7
            sophistication_score * 1.0 +  # 0-10
            appropriateness_bonus * 1.0   # 0-3
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~79, typical range 5-60
        normalized = max(0.0, min(100.0, raw_score * 1.4))
        
        return round(normalized, 2)
        
    except Exception:
        return 25.0  # neutral fallback score