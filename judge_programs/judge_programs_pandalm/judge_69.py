def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical structure analysis approach based on information density
    distribution, visual rhythm patterns, and structural entropy.
    
    This variant focuses on:
    - Information density distribution across segments
    - Visual rhythm (alternation patterns between structural elements)
    - Structural entropy (diversity of formatting elements used)
    - Depth of hierarchical nesting
    - Ratio-based proportionality of structural components
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        total_chars = len(response)
        total_words = len(response.split())
        
        if total_words == 0:
            return 0.0
        
        # Very short responses get a baseline score based on completeness
        if total_words < 5:
            return max(1.0, min(3.0, total_words * 0.6))
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # ===== FEATURE 1: Structural Element Classification =====
        # Classify each line into a structural type
        line_types = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                line_types.append('blank')
            elif re.match(r'^#{1,6}\s+', stripped):
                line_types.append('header_md')
            elif re.match(r'^[A-Z][A-Za-z\s]{2,50}:?\s*$', stripped) and len(stripped.split()) <= 8:
                line_types.append('header_implicit')
            elif re.match(r'^(\d+[\.\)]\s|[a-z][\.\)]\s)', stripped):
                line_types.append('numbered_item')
            elif re.match(r'^[\-\*\•\▪\►\→\‣\⁃]\s', stripped):
                line_types.append('bullet_item')
            elif re.match(r'^\*\*[^*]+\*\*', stripped) or re.match(r'^__[^_]+__', stripped):
                line_types.append('bold_start')
            elif len(stripped) > 0 and stripped[-1] == ':' and len(stripped.split()) <= 10:
                line_types.append('label_line')
            elif re.match(r'^\|.*\|', stripped):
                line_types.append('table_row')
            elif re.match(r'^[-=]{3,}$', stripped):
                line_types.append('separator')
            else:
                line_types.append('prose')
        
        # ===== FEATURE 2: Structural Entropy =====
        # Measure diversity of structural element types used
        type_counts = Counter(t for t in line_types if t != 'blank')
        if len(type_counts) == 0:
            structural_entropy = 0.0
        else:
            total_typed = sum(type_counts.values())
            probs = [c / total_typed for c in type_counts.values()]
            structural_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # Normalize entropy (max entropy for 8 types = log2(8) = 3)
        entropy_score = min(structural_entropy / 2.5, 1.0) * 15
        
        # ===== FEATURE 3: Visual Rhythm Analysis =====
        # Analyze the pattern of line types for regularity/rhythm
        # Good formatting creates recognizable patterns
        simplified_types = []
        for t in line_types:
            if t in ('header_md', 'header_implicit', 'label_line', 'bold_start'):
                simplified_types.append('H')
            elif t in ('numbered_item', 'bullet_item'):
                simplified_types.append('L')
            elif t == 'blank':
                simplified_types.append('B')
            elif t == 'prose':
                simplified_types.append('P')
            elif t == 'table_row':
                simplified_types.append('T')
            else:
                simplified_types.append('O')
        
        rhythm_string = ''.join(simplified_types)
        
        # Detect repeating patterns (e.g., "HLLLB" repeated = well-structured sections)
        rhythm_score = 0.0
        
        # Check for section patterns: Header followed by content followed by blank
        section_pattern = re.findall(r'H[LP]+B?', rhythm_string)
        if section_pattern:
            rhythm_score += len(section_pattern) * 3
        
        # Check for list consistency (consecutive list items)
        list_runs = re.findall(r'L{2,}', rhythm_string)
        if list_runs:
            avg_list_len = sum(len(r) for r in list_runs) / len(list_runs)
            rhythm_score += min(avg_list_len * 1.5, 8)
        
        # Check for prose-blank alternation (paragraph structure)
        para_pattern = re.findall(r'P+B', rhythm_string)
        if len(para_pattern) >= 2:
            rhythm_score += min(len(para_pattern) * 2, 8)
        
        rhythm_score = min(rhythm_score, 20)
        
        # ===== FEATURE 4: Information Density Distribution =====
        # Split response into quartiles and measure word density uniformity
        if len(non_empty_lines) >= 4:
            quarter = len(non_empty_lines) // 4
            quartiles = [
                non_empty_lines[:quarter],
                non_empty_lines[quarter:2*quarter],
                non_empty_lines[2*quarter:3*quarter],
                non_empty_lines[3*quarter:]
            ]
            quartile_densities = []
            for q in quartiles:
                q_text = ' '.join(q)
                q_words = len(q_text.split())
                q_chars = len(q_text)
                # Words per character as density proxy
                density = q_words / max(q_chars, 1)
                quartile_densities.append(density)
            
            # Coefficient of variation - lower is more uniform
            if quartile_densities:
                mean_d = sum(quartile_densities) / len(quartile_densities)
                if mean_d > 0:
                    variance = sum((d - mean_d)**2 for d in quartile_densities) / len(quartile_densities)
                    cv = math.sqrt(variance) / mean_d
                    # Lower CV = more uniform distribution = better
                    density_score = max(0, (1 - cv) * 10)
                else:
                    density_score = 0
            else:
                density_score = 0
        else:
            # For short responses, give moderate score
            density_score = 5.0
        
        # ===== FEATURE 5: Sentence Length Variance Within Paragraphs =====
        # Good writing has varied sentence lengths
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len)**2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Some variance is good (not monotonous), but not too much
                cv = std_dev / mean_len
                if 0.2 <= cv <= 0.8:
                    sentence_variety_score = 8.0
                elif cv < 0.2:
                    sentence_variety_score = 4.0  # Too uniform
                else:
                    sentence_variety_score = 5.0  # Too varied
            else:
                sentence_variety_score = 3.0
        else:
            sentence_variety_score = 4.0
        
        # ===== FEATURE 6: Hierarchical Depth =====
        # Measure nesting/indentation levels
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.add(leading_spaces)
        
        # Check for markdown heading levels
        heading_levels = set()
        for line in lines:
            hm = re.match(r'^(#{1,6})\s', line.strip())
            if hm:
                heading_levels.add(len(hm.group(1)))
        
        depth = max(len(indent_levels), len(heading_levels) + 1)
        depth_score = min(depth * 2.5, 10)
        
        # ===== FEATURE 7: Completeness Signals =====
        # Check for intro/conclusion structure
        completeness_score = 0.0
        
        # Check if response has an introductory statement
        first_sentence = sentences[0] if sentences else ""
        if len(first_sentence.split()) >= 5:
            completeness_score += 2.0
        
        # Check if last sentence feels conclusive
        if sentences:
            last_sentence = sentences[-1]
            conclusive_words = ['overall', 'conclusion', 'summary', 'therefore', 'thus',
                              'in short', 'finally', 'ultimately', 'important', 'key']
            if any(w in last_sentence.lower() for w in conclusive_words):
                completeness_score += 3.0
            elif len(last_sentence.split()) >= 5:
                completeness_score += 1.0
        
        completeness_score = min(completeness_score, 5.0)
        
        # ===== FEATURE 8: Formatting Element Proportionality =====
        # The ratio of structural elements to content should be balanced
        structural_lines = sum(1 for t in line_types if t in 
                              ('header_md', 'header_implicit', 'numbered_item', 
                               'bullet_item', 'bold_start', 'label_line', 'separator', 'blank'))
        content_lines = sum(1 for t in line_types if t == 'prose')
        total_meaningful = len(non_empty_lines)
        
        if total_meaningful > 0:
            struct_ratio = structural_lines / (total_meaningful + sum(1 for t in line_types if t == 'blank'))
            # Ideal ratio: 20-50% structural elements
            if 0.15 <= struct_ratio <= 0.55:
                proportion_score = 8.0
            elif 0.05 <= struct_ratio <= 0.7:
                proportion_score = 5.0
            elif struct_ratio == 0:
                # Pure prose - can be okay for short responses
                if total_words < 50:
                    proportion_score = 5.0
                else:
                    proportion_score = 2.0  # Wall of text for longer responses
            else:
                proportion_score = 3.0
        else:
            proportion_score = 1.0
        
        # ===== FEATURE 9: Whitespace Utilization =====
        # Measure effective use of blank lines for visual separation
        blank_count = sum(1 for t in line_types if t == 'blank')
        total_lines = len(lines)
        
        if total_lines > 3:
            blank_ratio = blank_count / total_lines
            if 0.1 <= blank_ratio <= 0.35:
                whitespace_score = 7.0
            elif 0.05 <= blank_ratio <= 0.45:
                whitespace_score = 4.0
            elif blank_ratio == 0 and total_words > 40:
                whitespace_score = 1.0  # No visual breaks in longer text
            elif blank_ratio == 0:
                whitespace_score = 4.0  # Short text doesn't need breaks
            else:
                whitespace_score = 2.0  # Too many blanks
        else:
            whitespace_score = 4.0
        
        # ===== FEATURE 10: Repetition Penalty =====
        # Detect excessive repetition which indicates poor organization
        words = response.lower().split()
        if len(words) >= 10:
            # Check for repeated bigrams
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
            repetition_ratio = max_bigram_freq / len(bigrams) if bigrams else 0
            
            if repetition_ratio > 0.15:
                repetition_penalty = -10.0
            elif repetition_ratio > 0.08:
                repetition_penalty = -5.0
            else:
                repetition_penalty = 0.0
        else:
            repetition_penalty = 0.0
        
        # ===== FEATURE 11: Response Length Appropriateness =====
        # Relative to query complexity
        query_words = len(query.split()) if query else 5
        response_to_query_ratio = total_words / max(query_words, 1)
        
        if response_to_query_ratio < 0.5:
            length_score = 1.0  # Way too short
        elif response_to_query_ratio < 1.5:
            length_score = 3.0  # Somewhat short
        elif response_to_query_ratio <= 15:
            length_score = 7.0  # Good range
        elif response_to_query_ratio <= 30:
            length_score = 5.0  # Getting long
        else:
            length_score = 3.0  # Possibly too verbose
        
        # ===== AGGREGATE SCORE =====
        score = (
            entropy_score * 1.0 +        # Max 15 - diversity of structure
            rhythm_score * 1.0 +          # Max 20 - pattern regularity
            density_score * 0.8 +          # Max 10 - information distribution
            sentence_variety_score * 0.6 + # Max 8 - sentence variety
            depth_score * 0.7 +            # Max 10 - hierarchical depth
            completeness_score * 0.8 +     # Max 5 - intro/conclusion
            proportion_score * 0.9 +       # Max 8 - structural proportionality
            whitespace_score * 0.7 +        # Max 7 - whitespace usage
            length_score * 0.5 +           # Max 7 - length appropriateness
            repetition_penalty             # Negative penalty
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~= 15 + 20 + 8 + 4.8 + 7 + 4 + 7.2 + 4.9 + 3.5 = ~74.4
        score = max(0, min(100, score * 1.35))
        
        return round(score, 2)
        
    except Exception:
        try:
            # Fallback: basic length-based score
            if response and len(response.strip()) > 0:
                return min(max(len(response.strip().split()) * 0.3, 1.0), 30.0)
            return 0.0
        except Exception:
            return 0.0