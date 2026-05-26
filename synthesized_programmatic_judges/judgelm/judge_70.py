def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses an information-theoretic and visual layout approach:
    - Analyzes the "visual density" distribution across the response (how content is spread)
    - Measures structural entropy (diversity of structural elements used)
    - Evaluates line-length variance as a proxy for formatting variety
    - Checks for hierarchical depth indicators
    - Analyzes the ratio of "structural characters" to content characters
    - Penalizes repetition patterns that indicate poor organization
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
        
        # Very short responses - score based on appropriateness
        if len(response) < 10:
            return 1.0
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Visual Density Distribution (0-15 points)
        # Split response into vertical "bands" and measure how evenly
        # content is distributed - good formatting spreads content
        # ============================================================
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(lines)
        
        if total_lines > 0:
            # Divide into quartiles and check density balance
            num_bands = min(4, total_lines)
            if num_bands >= 2:
                band_size = total_lines / num_bands
                band_char_counts = []
                for i in range(num_bands):
                    start = int(i * band_size)
                    end = int((i + 1) * band_size)
                    band_chars = sum(len(l.strip()) for l in lines[start:end])
                    band_char_counts.append(band_chars)
                
                total_chars_in_bands = sum(band_char_counts)
                if total_chars_in_bands > 0:
                    # Calculate how evenly distributed content is
                    expected = total_chars_in_bands / num_bands
                    deviations = [abs(c - expected) / max(expected, 1) for c in band_char_counts]
                    avg_deviation = sum(deviations) / len(deviations)
                    # Lower deviation = more even distribution = better
                    density_score = max(0, 15 * (1 - avg_deviation))
                else:
                    density_score = 0
            else:
                density_score = 5  # Single line, neutral
            score += density_score
        
        # ============================================================
        # FEATURE 2: Structural Entropy (0-15 points)
        # Classify each line by its "type" and measure Shannon entropy
        # of the type distribution. More diverse = better organized.
        # ============================================================
        line_types = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                line_types.append('blank')
            elif re.match(r'^#{1,6}\s', stripped):
                line_types.append('markdown_header')
            elif re.match(r'^<h[1-6]', stripped, re.IGNORECASE):
                line_types.append('html_header')
            elif re.match(r'^\d+[\.\)]\s', stripped):
                line_types.append('numbered_item')
            elif re.match(r'^[-*•●▪►→]\s', stripped):
                line_types.append('bullet_item')
            elif re.match(r'^[A-Z][^.!?]*[:]\s*$', stripped):
                line_types.append('label_line')
            elif stripped.startswith('```') or stripped.startswith('~~~'):
                line_types.append('code_fence')
            elif re.match(r'^[\-=_\*]{3,}$', stripped):
                line_types.append('separator')
            elif re.match(r'^\|.*\|$', stripped):
                line_types.append('table_row')
            elif len(stripped) > 80:
                line_types.append('long_text')
            elif len(stripped) > 20:
                line_types.append('medium_text')
            else:
                line_types.append('short_text')
        
        type_counts = Counter(line_types)
        num_types_used = len(type_counts)
        total_classified = len(line_types)
        
        if total_classified > 0:
            # Shannon entropy of line type distribution
            entropy = 0.0
            for count in type_counts.values():
                p = count / total_classified
                if p > 0:
                    entropy -= p * math.log2(p)
            
            # Normalize: max entropy for the number of types
            max_entropy = math.log2(max(num_types_used, 1))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
            
            # Bonus for having more distinct types
            type_bonus = min(num_types_used / 5.0, 1.0)
            
            structural_entropy_score = 15 * (0.5 * normalized_entropy + 0.5 * type_bonus)
        else:
            structural_entropy_score = 0
        
        score += structural_entropy_score
        
        # ============================================================
        # FEATURE 3: Line Length Variance Profile (0-12 points)
        # Good formatting produces varied line lengths (headers are short,
        # paragraphs are longer, bullets are medium). Wall-of-text has
        # uniform long lines.
        # ============================================================
        if non_empty_lines:
            line_lengths = [len(l.rstrip()) for l in non_empty_lines]
            mean_len = sum(line_lengths) / len(line_lengths)
            
            if len(line_lengths) > 1:
                variance = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
                std_dev = math.sqrt(variance)
                # Coefficient of variation
                cv = std_dev / max(mean_len, 1)
                # Moderate CV is good (0.3-0.8 range is ideal)
                if cv < 0.1:
                    length_var_score = 3  # Too uniform
                elif cv < 0.3:
                    length_var_score = 6
                elif cv <= 0.8:
                    length_var_score = 12
                elif cv <= 1.5:
                    length_var_score = 9
                else:
                    length_var_score = 5  # Too chaotic
            else:
                length_var_score = 3  # Single line
            score += length_var_score
        
        # ============================================================
        # FEATURE 4: Structural Character Ratio (0-10 points)
        # Ratio of "structural" characters (bullets, colons, pipes, 
        # hashes for headers) to total characters. Some structure is
        # good, too much is noise.
        # ============================================================
        structural_chars = sum(1 for c in response if c in '•●▪►→|#:;-*')
        # Also count newlines as structural
        newline_count = response.count('\n')
        structural_total = structural_chars + newline_count
        total_len = len(response)
        
        if total_len > 0:
            struct_ratio = structural_total / total_len
            # Sweet spot: 0.02 to 0.15
            if struct_ratio < 0.005:
                struct_score = 2  # Almost no structure
            elif struct_ratio < 0.02:
                struct_score = 5
            elif struct_ratio <= 0.15:
                struct_score = 10
            elif struct_ratio <= 0.3:
                struct_score = 6
            else:
                struct_score = 2  # Too much structural noise
            score += struct_score
        
        # ============================================================
        # FEATURE 5: Repetition Penalty (0 to -10 points)
        # Detect repeated phrases/sentences which indicate poor organization
        # ============================================================
        sentences = re.split(r'[.!?\n]+', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 15]
        
        if len(sentences) > 1:
            sentence_counts = Counter(sentences)
            repeated = sum(1 for s, c in sentence_counts.items() if c > 1)
            repeat_ratio = repeated / len(sentence_counts) if sentence_counts else 0
            repetition_penalty = -10 * min(repeat_ratio, 1.0)
            score += repetition_penalty
        
        # Also check for repeated line patterns
        if len(non_empty_lines) > 2:
            line_set = set(l.strip().lower() for l in non_empty_lines if len(l.strip()) > 10)
            if len(non_empty_lines) > 0:
                uniqueness = len(line_set) / len(non_empty_lines)
                if uniqueness < 0.5:
                    score -= 5  # Heavy line repetition
        
        # ============================================================
        # FEATURE 6: Paragraph Segmentation Quality (0-10 points)
        # Check if the response uses blank lines to separate paragraphs
        # and whether paragraph sizes are reasonable
        # ============================================================
        # Split by blank lines to find paragraphs
        paragraphs = re.split(r'\n\s*\n', response.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if num_paragraphs == 1:
            # Single block - check if it's short enough to not need paragraphs
            word_count = len(response.split())
            if word_count <= 40:
                para_score = 7  # Short enough, single para is fine
            elif word_count <= 80:
                para_score = 4  # Could use some breaking up
            else:
                para_score = 1  # Wall of text
        elif num_paragraphs >= 2:
            # Multiple paragraphs - check size balance
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            
            if 15 <= avg_para_len <= 80:
                para_score = 10  # Good paragraph size
            elif 5 <= avg_para_len <= 150:
                para_score = 7
            else:
                para_score = 4
            
            # Bonus for having 2-6 paragraphs (well-organized)
            if 2 <= num_paragraphs <= 6:
                para_score = min(para_score + 2, 10)
        else:
            para_score = 0
        
        score += para_score
        
        # ============================================================
        # FEATURE 7: Opening and Closing Quality (0-8 points)
        # Good responses start with a clear topic/intro and end properly
        # ============================================================
        first_line = non_empty_lines[0].strip() if non_empty_lines else ""
        last_line = non_empty_lines[-1].strip() if non_empty_lines else ""
        
        opening_score = 0
        # Starts with a capital letter or structural element
        if first_line and first_line[0].isupper():
            opening_score += 2
        # First line is a reasonable length (not too short, not too long)
        if 10 < len(first_line) < 200:
            opening_score += 2
        
        closing_score = 0
        # Ends with proper punctuation
        if last_line and last_line[-1] in '.!?:)>"\'':
            closing_score += 2
        # Last line is substantive
        if len(last_line) > 5:
            closing_score += 2
        
        score += opening_score + closing_score
        
        # ============================================================
        # FEATURE 8: Content-to-Query Proportionality (0-8 points)
        # Response length should be proportional to query complexity
        # ============================================================
        query_words = len(query.split()) if query else 1
        response_words = len(response.split())
        
        if response_words < 3:
            prop_score = 1
        elif response_words < 10:
            prop_score = 3
        else:
            ratio = response_words / max(query_words, 1)
            if 0.5 <= ratio <= 15:
                prop_score = 8
            elif 0.3 <= ratio <= 25:
                prop_score = 6
            else:
                prop_score = 3
        
        score += prop_score
        
        # ============================================================
        # FEATURE 9: Indentation and Nesting Detection (0-7 points)
        # Check for intentional indentation patterns suggesting hierarchy
        # ============================================================
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.add(leading_spaces)
        
        num_indent_levels = len(indent_levels)
        if num_indent_levels >= 3:
            indent_score = 7  # Multiple indentation levels = hierarchy
        elif num_indent_levels == 2:
            indent_score = 5
        else:
            indent_score = 2  # Flat structure
        
        score += indent_score
        
        # ============================================================
        # FEATURE 10: Garbage/Noise Detection Penalty (0 to -8 points)
        # Detect signs of broken/garbage output
        # ============================================================
        noise_penalty = 0
        
        # Excessive code in non-code query
        code_keywords = ['import ', 'def ', 'class ', 'return ', 'if __name__']
        query_lower = query.lower() if query else ""
        is_code_query = any(kw in query_lower for kw in ['code', 'program', 'function', 'script', 'html', 'python', 'java'])
        
        if not is_code_query:
            code_line_count = sum(1 for l in non_empty_lines if any(kw in l for kw in code_keywords))
            if code_line_count > 3:
                noise_penalty -= 5
        
        # Detect "Output:" repetition pattern (like in examples)
        output_prefix_count = sum(1 for l in non_empty_lines if l.strip().startswith('Output:'))
        if output_prefix_count > 3:
            noise_penalty -= 3
        
        # Detect "Question:"/"Answer:" spam
        qa_spam = sum(1 for l in non_empty_lines if re.match(r'^(Question|Answer|Input|Output):', l.strip()))
        if qa_spam > 4:
            noise_penalty -= 4
        
        # Truncation detection (ends mid-word or mid-sentence without punctuation)
        if response_words > 20 and last_line and not last_line[-1] in '.!?:)]\'"':
            noise_penalty -= 1
        
        score += noise_penalty
        
        # ============================================================
        # FEATURE 11: Semantic Separator Usage (0-5 points)
        # Check for transition/separator patterns between sections
        # ============================================================
        separator_patterns = [
            r'\n\s*[-=_]{3,}\s*\n',  # Horizontal rules
            r'\n\s*\*\*\*\s*\n',     # Asterisk separators
            r'\n#{1,3}\s+\S',        # Markdown headers
            r'\n\d+\.\s+[A-Z]',     # Numbered sections starting with caps
            r'\n[A-Z][a-z]+:',       # Label patterns like "Summary:"
        ]
        
        separator_count = 0
        for pattern in separator_patterns:
            separator_count += len(re.findall(pattern, response))
        
        if separator_count >= 3:
            sep_score = 5
        elif separator_count >= 1:
            sep_score = 3
        else:
            sep_score = 1
        
        score += sep_score
        
        # Normalize to 0-10 range
        # Max theoretical: 15+15+12+10+0+10+8+8+7+0+5 = 90
        # Min theoretical: 0+0+0+0-10+0+0+0+0-8+0 = -18
        # Practical range: ~5 to ~80
        
        normalized = max(0, min(10, score / 8.0))
        
        return round(normalized, 2)
    
    except Exception:
        return 3.0