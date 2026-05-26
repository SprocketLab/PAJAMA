def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical document structure analysis approach.
    
    This variant uses a fundamentally different approach: it models the response
    as a tree-like document structure, analyzing nesting depth, structural variety,
    visual rhythm (alternating patterns of structure types), and the ratio of
    "structured segments" to "unstructured prose blobs". It also measures
    structural coherence via segment length variance analysis.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        import re
        import math
        from collections import Counter
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        total_lines = len(non_empty_lines)
        
        if total_lines == 0:
            return 0.5
        
        # === SEGMENT CLASSIFICATION ===
        # Classify each non-empty line into a structural type
        # Types: header, numbered_item, bullet_item, colon_def, quote, prose, short_phrase
        
        def classify_line(line):
            stripped = line.strip()
            # Header patterns (markdown or all-caps short lines)
            if re.match(r'^#{1,6}\s+\S', stripped):
                return 'header'
            if re.match(r'^[A-Z][A-Za-z\s]{2,40}:$', stripped):
                return 'header'
            if len(stripped) < 60 and stripped.isupper() and len(stripped.split()) >= 2:
                return 'header'
            if re.match(r'^\*\*[^*]+\*\*$', stripped):
                return 'header'
            # Numbered items
            if re.match(r'^\d{1,3}[\.\)]\s+\S', stripped):
                return 'numbered_item'
            if re.match(r'^[a-zA-Z][\.\)]\s+\S', stripped):
                return 'numbered_item'
            if re.match(r'^(?:step|phase|stage|part|point)\s+\d/i', stripped, re.IGNORECASE):
                return 'numbered_item'
            # Bullet items
            if re.match(r'^[\-\*\•\◦\▪\►\→\➤\●]\s+\S', stripped):
                return 'bullet_item'
            if re.match(r'^\s{2,}[\-\*\•]\s+\S', line):  # indented bullets
                return 'bullet_item'
            # Definition/key-value patterns
            if re.match(r'^[A-Z][^:]{2,30}:\s+\S', stripped) and len(stripped) > 15:
                return 'colon_def'
            # Quote
            if re.match(r'^["\'>]', stripped):
                return 'quote'
            # Short phrase (likely a label or sub-header)
            if len(stripped.split()) <= 5 and not stripped.endswith('.') and len(stripped) < 40:
                return 'short_phrase'
            # Default: prose
            return 'prose'
        
        classifications = [classify_line(l) for l in non_empty_lines]
        type_counts = Counter(classifications)
        
        # === 1. STRUCTURAL VARIETY SCORE ===
        # How many different structural types are used?
        distinct_types = len([t for t, c in type_counts.items() if c > 0])
        structural_types_used = set(type_counts.keys())
        
        # Bonus for having a good mix; penalize mono-type responses
        variety_score = min(distinct_types / 3.0, 1.0)  # max at 3+ types
        
        # Extra bonus for having both organizational elements and content
        has_organizational = bool(structural_types_used & {'header', 'numbered_item', 'bullet_item', 'colon_def'})
        has_content = 'prose' in structural_types_used or total_lines > 3
        if has_organizational and has_content:
            variety_score = min(variety_score + 0.2, 1.0)
        
        # === 2. STRUCTURED RATIO ===
        # What fraction of lines are "structured" (non-plain-prose)?
        structured_types = {'header', 'numbered_item', 'bullet_item', 'colon_def', 'short_phrase'}
        structured_count = sum(1 for c in classifications if c in structured_types)
        
        if total_lines > 0:
            structured_ratio = structured_count / total_lines
        else:
            structured_ratio = 0.0
        
        # Ideal ratio is around 0.3-0.7 (mix of structure and prose)
        # Pure structure or pure prose are both suboptimal
        if structured_ratio <= 0.0:
            ratio_score = 0.1
        elif structured_ratio < 0.15:
            ratio_score = 0.3
        elif structured_ratio <= 0.7:
            ratio_score = 0.7 + 0.3 * (structured_ratio / 0.7)
        elif structured_ratio <= 0.85:
            ratio_score = 0.9
        else:
            ratio_score = 0.75  # too much structure, not enough prose
        
        # === 3. VISUAL RHYTHM ANALYSIS ===
        # Measure how the structural types alternate/flow
        # Good responses have patterns like: prose -> header -> items -> prose -> header -> items
        # Bad responses are monotonous: prose -> prose -> prose -> prose
        
        if len(classifications) >= 3:
            transitions = 0
            for i in range(1, len(classifications)):
                if classifications[i] != classifications[i-1]:
                    transitions += 1
            rhythm_score = min(transitions / (len(classifications) - 1) * 1.5, 1.0)
        elif len(classifications) == 2:
            rhythm_score = 0.5 if classifications[0] != classifications[1] else 0.2
        else:
            rhythm_score = 0.2
        
        # === 4. SEGMENT LENGTH VARIANCE (VISUAL BALANCE) ===
        # Analyze the lengths of consecutive same-type blocks
        # Well-organized text has balanced segment sizes
        
        segments = []
        if classifications:
            current_type = classifications[0]
            current_length = 1
            for i in range(1, len(classifications)):
                if classifications[i] == current_type:
                    current_length += 1
                else:
                    segments.append((current_type, current_length))
                    current_type = classifications[i]
                    current_length = 1
            segments.append((current_type, current_length))
        
        num_segments = len(segments)
        
        if num_segments >= 3:
            segment_lengths = [s[1] for s in segments]
            mean_len = sum(segment_lengths) / len(segment_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len)**2 for l in segment_lengths) / len(segment_lengths)
                cv = math.sqrt(variance) / mean_len if mean_len > 0 else 0
                # Lower CV = more balanced, but some variance is okay
                balance_score = max(0, 1.0 - cv * 0.3)
            else:
                balance_score = 0.5
        elif num_segments == 2:
            balance_score = 0.6
        else:
            balance_score = 0.3  # single block = poorly organized
        
        # === 5. PARAGRAPH QUALITY (via blank line analysis) ===
        # Instead of counting paragraphs, analyze the "breathing room" pattern
        
        # Count blank lines and their positions
        blank_positions = []
        for i, line in enumerate(lines):
            if not line.strip():
                blank_positions.append(i)
        
        num_blanks = len(blank_positions)
        
        # Calculate inter-blank-line distances (paragraph sizes)
        if num_blanks > 0:
            # Distances between consecutive blank lines
            distances = []
            prev = -1
            for pos in blank_positions:
                distances.append(pos - prev - 1)
                prev = pos
            distances.append(len(lines) - prev - 1)
            distances = [d for d in distances if d > 0]
            
            if distances:
                avg_para_size = sum(distances) / len(distances)
                # Ideal paragraph size: 2-6 lines
                if 2 <= avg_para_size <= 6:
                    para_quality = 1.0
                elif 1 <= avg_para_size < 2:
                    para_quality = 0.7
                elif 6 < avg_para_size <= 10:
                    para_quality = 0.6
                else:
                    para_quality = 0.3
            else:
                para_quality = 0.3
        else:
            # No blank lines at all
            if total_lines <= 3:
                para_quality = 0.5  # short response, okay
            else:
                para_quality = 0.15  # wall of text
        
        # === 6. INDENTATION/NESTING DEPTH ===
        # Check for hierarchical nesting (indented sub-items)
        
        indent_levels = []
        for line in non_empty_lines:
            leading_spaces = len(line) - len(line.lstrip())
            indent_level = leading_spaces // 2  # normalize to 2-space indents
            indent_levels.append(indent_level)
        
        max_indent = max(indent_levels) if indent_levels else 0
        unique_indents = len(set(indent_levels))
        
        if unique_indents >= 3:
            nesting_score = 0.9
        elif unique_indents == 2:
            nesting_score = 0.6
        else:
            nesting_score = 0.3
        
        # Boost if nesting is used with list items
        if max_indent >= 2 and has_organizational:
            nesting_score = min(nesting_score + 0.2, 1.0)
        
        # === 7. LINE LENGTH DISTRIBUTION ===
        # Well-formatted text has varied line lengths (headers short, prose medium, etc.)
        # Wall-of-text has uniformly long lines
        
        line_lengths = [len(l.strip()) for l in non_empty_lines]
        if line_lengths:
            avg_line_len = sum(line_lengths) / len(line_lengths)
            
            # Categorize lines by length
            short_lines = sum(1 for l in line_lengths if l < 40)
            medium_lines = sum(1 for l in line_lengths if 40 <= l < 100)
            long_lines = sum(1 for l in line_lengths if l >= 100)
            
            # Good formatting has a mix of line lengths
            length_categories_used = sum(1 for count in [short_lines, medium_lines, long_lines] if count > 0)
            
            if length_categories_used >= 3:
                length_dist_score = 0.9
            elif length_categories_used == 2:
                length_dist_score = 0.6
            else:
                # All same category
                if long_lines == total_lines:
                    length_dist_score = 0.15  # all long = wall of text
                elif short_lines == total_lines:
                    length_dist_score = 0.4  # all short = choppy
                else:
                    length_dist_score = 0.4
        else:
            length_dist_score = 0.3
        
        # === 8. SEQUENTIAL COHERENCE ===
        # Check if numbered items are sequential (1, 2, 3, ...)
        
        numbers_found = []
        for line in non_empty_lines:
            m = re.match(r'^\s*(\d{1,3})[\.\)]\s', line.strip())
            if m:
                numbers_found.append(int(m.group(1)))
        
        coherence_bonus = 0.0
        if len(numbers_found) >= 2:
            # Check if sequential
            is_sequential = all(numbers_found[i] == numbers_found[i-1] + 1 
                              for i in range(1, len(numbers_found)))
            if is_sequential and numbers_found[0] == 1:
                coherence_bonus = 0.15 * min(len(numbers_found) / 3.0, 1.0)
            elif is_sequential:
                coherence_bonus = 0.08
        
        # === 9. RESPONSE LENGTH APPROPRIATENESS ===
        # Very short responses are likely poorly organized by default
        # Very long responses need MORE structure
        
        query_complexity = len(query.split())
        response_words = len(response.split())
        
        if response_words < 20:
            length_factor = 0.4  # too short to have much structure
        elif response_words < 50:
            length_factor = 0.7
        elif response_words < 200:
            length_factor = 1.0
        elif response_words < 400:
            # Long responses need structure even more
            if has_organizational:
                length_factor = 1.0
            else:
                length_factor = 0.5  # long without structure = bad
        else:
            if has_organizational:
                length_factor = 0.95
            else:
                length_factor = 0.35
        
        # === 10. OPENING AND CLOSING QUALITY ===
        # Good responses often start with a clear opening and end with a conclusion
        
        first_line = non_empty_lines[0].strip().lower() if non_empty_lines else ""
        last_line = non_empty_lines[-1].strip().lower() if non_empty_lines else ""
        
        opening_quality = 0.5
        # Good openings: direct address, context setting
        opening_patterns = [
            r'^(i |i\'m |it\'s |let|here|sure|great|absolutely|of course|certainly|to )',
            r'^(imagine|consider|think|first|the |this |when |in )',
        ]
        for pat in opening_patterns:
            if re.match(pat, first_line):
                opening_quality = 0.7
                break
        
        # Check if response has a concluding sentiment
        closing_patterns = [
            r'(remember|overall|in (summary|conclusion)|finally|hope|feel free|don\'t hesitate|good luck)',
            r'(let me know|happy to help|reach out|best wishes)',
        ]
        closing_quality = 0.4
        for pat in closing_patterns:
            if re.search(pat, last_line):
                closing_quality = 0.8
                break
        
        framing_score = (opening_quality + closing_quality) / 2.0
        
        # === COMPOSITE SCORE ===
        # Weight the components with emphasis on the most discriminative features
        
        weights = {
            'variety': 0.12,
            'ratio': 0.10,
            'rhythm': 0.12,
            'balance': 0.08,
            'para_quality': 0.15,
            'nesting': 0.05,
            'length_dist': 0.10,
            'length_factor': 0.10,
            'framing': 0.08,
            'coherence_bonus': 1.0,  # additive, not weighted same way
        }
        
        weighted_sum = (
            weights['variety'] * variety_score +
            weights['ratio'] * ratio_score +
            weights['rhythm'] * rhythm_score +
            weights['balance'] * balance_score +
            weights['para_quality'] * para_quality +
            weights['nesting'] * nesting_score +
            weights['length_dist'] * length_dist_score +
            weights['length_factor'] * length_factor +
            weights['framing'] * framing_score +
            coherence_bonus  # additive bonus
        )
        
        total_weight = sum(v for k, v in weights.items() if k != 'coherence_bonus')
        
        # Normalize to 0-1 range
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Clamp
        base_score = max(0.0, min(1.0, base_score))
        
        # === PENALTY: WALL OF TEXT DETECTION ===
        # If response is one giant paragraph with no structure
        if num_blanks == 0 and total_lines <= 2 and response_words > 60 and not has_organizational:
            base_score *= 0.5
        
        # === BONUS: RICH FORMATTING ===
        # If response uses multiple formatting features together
        formatting_features = sum([
            type_counts.get('header', 0) > 0,
            type_counts.get('numbered_item', 0) >= 2,
            type_counts.get('bullet_item', 0) >= 2,
            type_counts.get('colon_def', 0) >= 2,
            num_blanks >= 2,
            unique_indents >= 2,
        ])
        if formatting_features >= 3:
            base_score = min(base_score * 1.15, 1.0)
        
        # Scale to 1-5 range
        final_score = 1.0 + base_score * 4.0
        
        # Round to 1 decimal
        final_score = round(final_score, 1)
        
        return max(1.0, min(5.0, final_score))
    
    except Exception:
        return 2.5