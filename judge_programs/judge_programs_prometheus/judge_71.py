def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses a DIFFERENT approach: analyzing the visual/spatial layout
    through indentation patterns, line-length variance, structural rhythm analysis,
    and information density distribution across response segments.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Line-length variance and rhythm analysis
        # Good formatting creates a rhythm of varying line lengths
        # (headers short, content medium, lists medium-short)
        # Wall-of-text has very low variance in line lengths
        # ============================================================
        if len(non_empty_lines) >= 2:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            mean_len = sum(line_lengths) / len(line_lengths)
            
            if mean_len > 0:
                # Coefficient of variation of line lengths
                variance = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
                std_dev = variance ** 0.5
                cv = std_dev / mean_len if mean_len > 0 else 0
                
                # Moderate CV suggests mixed structure (headers + content + lists)
                # Very low CV = wall of text or uniform lines
                # Very high CV = erratic formatting
                if 0.2 <= cv <= 1.2:
                    score += min(cv * 8, 8.0)
                elif cv < 0.2:
                    score += cv * 10  # low reward for uniform text
                else:
                    score += max(8.0 - (cv - 1.2) * 3, 2.0)
        
        # ============================================================
        # FEATURE 2: Blank line segmentation analysis
        # Measures how the response is broken into visual blocks
        # Good responses have multiple well-sized segments
        # ============================================================
        segments = []
        current_segment = []
        for line in lines:
            if line.strip() == '':
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
            else:
                current_segment.append(line)
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        num_segments = len(segments)
        
        if num_segments == 1 and total_chars > 200:
            # Single block wall of text - penalize
            score += 0.0
        elif num_segments == 1:
            score += 2.0
        elif 2 <= num_segments <= 4:
            score += 6.0
        elif 5 <= num_segments <= 8:
            score += 10.0
        elif 9 <= num_segments <= 15:
            score += 8.0
        else:
            score += 5.0
        
        # ============================================================
        # FEATURE 3: Segment size balance (Gini-like coefficient)
        # Well-organized responses have reasonably balanced segments
        # Not all the same size, but no single segment dominates
        # ============================================================
        if len(segments) >= 2:
            seg_sizes = sorted([len(s.strip()) for s in segments])
            n = len(seg_sizes)
            total_size = sum(seg_sizes)
            if total_size > 0:
                # Compute Gini coefficient
                cumulative = 0
                weighted_sum = 0
                for i, size in enumerate(seg_sizes):
                    cumulative += size
                    weighted_sum += (i + 1) * size
                gini = (2 * weighted_sum) / (n * total_size) - (n + 1) / n
                
                # Moderate gini (0.1-0.5) is good - some variation but not extreme
                if 0.05 <= gini <= 0.55:
                    score += 6.0
                elif gini < 0.05:
                    score += 3.0  # too uniform
                else:
                    score += max(6.0 - (gini - 0.55) * 8, 1.0)
        else:
            # Single segment
            if total_chars > 300:
                score += 0.0  # wall of text
            else:
                score += 2.0
        
        # ============================================================
        # FEATURE 4: Leading character pattern diversity
        # Analyzes what characters/patterns start each line
        # Good formatting has diverse line starters:
        # numbers, bullets, uppercase (headers), indentation, regular text
        # ============================================================
        import re
        
        starter_categories = set()
        category_counts = {}
        
        for line in non_empty_lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            cat = 'plain'
            if re.match(r'^\d+[\.\)]\s', stripped):
                cat = 'numbered'
            elif re.match(r'^[-*•►▪◦]\s', stripped):
                cat = 'bullet'
            elif re.match(r'^#{1,6}\s', stripped):
                cat = 'md_header'
            elif stripped.isupper() and len(stripped) < 80:
                cat = 'upper_header'
            elif stripped.endswith(':') and len(stripped) < 80:
                cat = 'label_line'
            elif re.match(r'^[A-Z][a-z]', stripped) and len(stripped) < 60 and not stripped.endswith('.'):
                cat = 'title_case_header'
            elif re.match(r'^\s{2,}', line) and not line.startswith('\t'):
                cat = 'indented'
            elif re.match(r'^[a-z]', stripped):
                cat = 'lowercase_start'
            elif stripped.startswith('**') or stripped.startswith('__'):
                cat = 'bold_start'
            
            starter_categories.add(cat)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        diversity = len(starter_categories)
        if diversity >= 4:
            score += 10.0
        elif diversity == 3:
            score += 7.0
        elif diversity == 2:
            score += 4.0
        else:
            score += 1.0
        
        # ============================================================
        # FEATURE 5: Structural element density
        # Ratio of "structural" lines (lists, headers, labels) to total lines
        # ============================================================
        structural_count = sum(v for k, v in category_counts.items() 
                             if k in ('numbered', 'bullet', 'md_header', 'upper_header', 
                                     'label_line', 'title_case_header', 'bold_start'))
        
        if len(non_empty_lines) > 0:
            structural_ratio = structural_count / len(non_empty_lines)
            # Sweet spot: 15-50% structural elements
            if 0.10 <= structural_ratio <= 0.55:
                score += 8.0
            elif structural_ratio > 0.55:
                score += 5.0  # over-structured
            elif 0.01 < structural_ratio < 0.10:
                score += 3.0
            else:
                score += 0.0  # no structural elements
        
        # ============================================================
        # FEATURE 6: Sentence length distribution within segments
        # Good paragraphs mix sentence lengths for readability
        # ============================================================
        sentences = re.split(r'[.!?]+\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sent = sum(sent_lengths) / len(sent_lengths)
            
            if mean_sent > 0:
                # Check for sentence length variation
                sent_var = sum((l - mean_sent) ** 2 for l in sent_lengths) / len(sent_lengths)
                sent_cv = (sent_var ** 0.5) / mean_sent if mean_sent > 0 else 0
                
                # Some variation is good
                if 0.2 <= sent_cv <= 0.8:
                    score += 5.0
                elif sent_cv > 0.8:
                    score += 3.0
                else:
                    score += 1.5
                
                # Penalize very long average sentences (hard to read)
                if mean_sent > 35:
                    score -= 2.0
                elif mean_sent < 8:
                    score -= 1.0
        else:
            score += 1.0
        
        # ============================================================
        # FEATURE 7: Opening and closing structure
        # Good responses have a clear opening statement and conclusion
        # ============================================================
        if len(segments) >= 2:
            first_seg = segments[0].strip()
            last_seg = segments[-1].strip()
            
            # Opening: relatively concise introduction
            first_words = len(first_seg.split())
            if 5 <= first_words <= 60:
                score += 3.0
            elif first_words > 60:
                score += 1.0
            else:
                score += 0.5
            
            # Check if last segment has concluding feel
            last_lower = last_seg.lower()
            concluding_signals = ['remember', 'overall', 'in summary', 'finally', 
                                 'in conclusion', 'hope', 'good luck', 'feel free',
                                 'don\'t hesitate', 'keep in mind', 'most importantly',
                                 'at the end', 'to sum up', 'lastly', 'take care']
            has_conclusion = any(sig in last_lower for sig in concluding_signals)
            if has_conclusion:
                score += 3.0
            else:
                score += 1.0
        else:
            score += 1.0
        
        # ============================================================
        # FEATURE 8: Character-level whitespace ratio
        # Measures the "breathing room" in the text
        # ============================================================
        newline_count = response.count('\n')
        space_ratio = newline_count / max(total_chars, 1)
        
        # Good formatting has some whitespace but not excessive
        if 0.01 <= space_ratio <= 0.08:
            score += 4.0
        elif 0.005 < space_ratio < 0.01:
            score += 2.0
        elif space_ratio > 0.08:
            score += 2.5
        else:
            score += 0.5  # almost no line breaks
        
        # ============================================================
        # FEATURE 9: Consistency of list/enumeration patterns
        # If lists are used, are they consistent?
        # ============================================================
        numbered_lines = [l.strip() for l in non_empty_lines if re.match(r'^\d+[\.\)]\s', l.strip())]
        bullet_lines = [l.strip() for l in non_empty_lines if re.match(r'^[-*•]\s', l.strip())]
        
        if len(numbered_lines) >= 2:
            # Check sequential numbering
            numbers = []
            for nl in numbered_lines:
                m = re.match(r'^(\d+)', nl)
                if m:
                    numbers.append(int(m.group(1)))
            
            if numbers:
                is_sequential = all(numbers[i] <= numbers[i+1] for i in range(len(numbers)-1))
                if is_sequential:
                    score += 4.0
                else:
                    score += 1.5
        
        if len(bullet_lines) >= 2:
            score += 3.0
        
        # ============================================================
        # FEATURE 10: Response length appropriateness relative to query complexity
        # ============================================================
        query_words = len(query.split())
        response_words = len(response.split())
        
        # Complex queries deserve longer, well-structured responses
        if query_words > 30:
            if response_words > 80:
                score += 3.0
            elif response_words > 40:
                score += 2.0
            else:
                score += 0.5
        else:
            if 20 <= response_words <= 200:
                score += 2.5
            else:
                score += 1.0
        
        # ============================================================
        # Normalize to 1-5 scale
        # ============================================================
        # Theoretical max ~68, min ~0
        # Map to 1-5
        raw_max = 68.0
        raw_min = 0.0
        
        normalized = (score - raw_min) / (raw_max - raw_min)  # 0 to 1
        final_score = 1.0 + normalized * 4.0  # 1 to 5
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5