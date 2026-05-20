def judging_function(query, response):
    """
    Evaluate structural organization and formatting quality of an LLM response.
    
    This variant focuses on:
    - Hierarchical depth analysis (nested structures)
    - Visual separation and rhythm (alternating structure types)
    - Information density distribution across segments
    - Structural variety score (mix of different formatting elements)
    - Proportional balance of structured vs unstructured content
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        import re
        import math
        from collections import Counter
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # ---- Feature 1: Structural Element Variety ----
        # Count distinct types of structural elements present
        structure_types_found = set()
        
        # Numbered lists (1. 2. etc)
        numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+\S', re.MULTILINE)
        numbered_items = numbered_pattern.findall(response)
        if len(numbered_items) >= 2:
            structure_types_found.add('numbered_list')
        
        # Bullet points (-, *, •)
        bullet_pattern = re.compile(r'^\s*[-*•]\s+\S', re.MULTILINE)
        bullet_items = bullet_pattern.findall(response)
        if len(bullet_items) >= 2:
            structure_types_found.add('bullet_list')
        
        # Headers (markdown ## or ALL CAPS lines or lines ending with colon)
        header_md = re.compile(r'^\s*#{1,6}\s+\S', re.MULTILINE)
        header_caps = re.compile(r'^[A-Z][A-Z\s]{4,}$', re.MULTILINE)
        header_colon = re.compile(r'^[A-Z][^.!?]{3,50}:\s*$', re.MULTILINE)
        bold_header = re.compile(r'^\s*\*\*[^*]+\*\*\s*$', re.MULTILINE)
        
        headers_found = (header_md.findall(response) + header_caps.findall(response) + 
                        header_colon.findall(response) + bold_header.findall(response))
        if headers_found:
            structure_types_found.add('headers')
        
        # Code blocks
        code_block_pattern = re.compile(r'```[\s\S]*?```')
        code_blocks = code_block_pattern.findall(response)
        if code_blocks:
            structure_types_found.add('code_block')
        
        # Inline code
        inline_code = re.compile(r'`[^`]+`')
        if inline_code.findall(response):
            structure_types_found.add('inline_code')
        
        # Bold/italic emphasis
        bold_pattern = re.compile(r'\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_')
        if bold_pattern.findall(response):
            structure_types_found.add('emphasis')
        
        # Parenthetical references or citations
        paren_pattern = re.compile(r'\([^)]{5,}\)')
        if len(paren_pattern.findall(response)) >= 2:
            structure_types_found.add('parentheticals')
        
        # Multiple paragraphs (separated by blank lines)
        paragraph_splits = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraph_splits if p.strip()]
        if len(paragraphs) >= 2:
            structure_types_found.add('multi_paragraph')
        
        # Indented/nested content
        indented = re.compile(r'^\s{4,}\S', re.MULTILINE)
        if indented.findall(response):
            structure_types_found.add('indentation')
        
        variety_score = len(structure_types_found)  # 0-9 possible
        
        # ---- Feature 2: Segment Length Distribution (rhythm analysis) ----
        # Good formatting creates segments of varying but reasonable length
        # Analyze the "rhythm" of line lengths
        line_lengths = [len(l.strip()) for l in lines]
        non_zero_lengths = [ll for ll in line_lengths if ll > 0]
        
        rhythm_score = 0.0
        if len(non_zero_lengths) >= 3:
            # Calculate coefficient of variation of line lengths
            mean_len = sum(non_zero_lengths) / len(non_zero_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in non_zero_lengths) / len(non_zero_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len
                # Moderate variation is good (0.3-0.8), too uniform or too chaotic is bad
                if 0.2 <= cv <= 1.5:
                    rhythm_score = min(1.0, 1.0 - abs(cv - 0.7) / 0.8)
                else:
                    rhythm_score = 0.1
            
            # Check for alternation between short and long lines (structural rhythm)
            transitions = 0
            median_len = sorted(non_zero_lengths)[len(non_zero_lengths) // 2]
            for i in range(1, len(non_zero_lengths)):
                prev_short = non_zero_lengths[i-1] < median_len
                curr_short = non_zero_lengths[i] < median_len
                if prev_short != curr_short:
                    transitions += 1
            if len(non_zero_lengths) > 1:
                transition_rate = transitions / (len(non_zero_lengths) - 1)
                rhythm_score += transition_rate * 0.5
        
        # ---- Feature 3: Information Density Distribution ----
        # Check if information is evenly distributed across paragraphs
        # rather than front-loaded or back-loaded
        density_score = 0.0
        if len(paragraphs) >= 2:
            para_lengths = [len(p) for p in paragraphs]
            total_para_len = sum(para_lengths)
            if total_para_len > 0:
                # Calculate Gini-like coefficient for paragraph length distribution
                n = len(para_lengths)
                sorted_lens = sorted(para_lengths)
                cumulative = 0
                gini_sum = 0
                for i, length in enumerate(sorted_lens):
                    cumulative += length
                    gini_sum += cumulative
                expected = total_para_len * n / 2
                if expected > 0:
                    gini = 1 - gini_sum / (expected * n) if n > 0 else 0
                    # Lower gini = more equal distribution = better
                    # But some variation is natural, so moderate gini is fine
                    density_score = max(0, 1.0 - abs(gini) * 2)
            
            # Bonus for having a reasonable number of paragraphs relative to length
            ideal_para_count = max(2, total_chars / 400)
            para_ratio = len(paragraphs) / ideal_para_count
            if 0.5 <= para_ratio <= 2.0:
                density_score += 0.5
            elif 0.3 <= para_ratio <= 3.0:
                density_score += 0.2
        elif len(paragraphs) == 1 and total_chars > 300:
            # Wall of text penalty
            density_score = -0.5
        
        # ---- Feature 4: Structural Coherence Signals ----
        # Look for signals that the structure serves the content
        coherence_score = 0.0
        
        # Opening/framing statement before structured content
        if paragraphs and len(paragraphs) >= 2:
            first_para = paragraphs[0]
            has_list_later = bool(numbered_items) or bool(bullet_items)
            # If first paragraph is a short intro followed by structured content
            if has_list_later and len(first_para) < 300 and not re.match(r'^\s*[-*•\d]', first_para):
                coherence_score += 1.0
        
        # Closing/summary paragraph after structured content
        if paragraphs and len(paragraphs) >= 3:
            last_para = paragraphs[-1]
            if not re.match(r'^\s*[-*•\d]', last_para) and len(last_para) > 20:
                coherence_score += 0.5
        
        # Consistent list style (not mixing numbered and bullets randomly)
        if numbered_items and bullet_items:
            # Could be intentional (nested) or sloppy
            # Check if they're in different sections
            numbered_positions = [m.start() for m in numbered_pattern.finditer(response)]
            bullet_positions = [m.start() for m in bullet_pattern.finditer(response)]
            
            if numbered_positions and bullet_positions:
                # If they're clearly separated, it's intentional nesting
                min_num = min(numbered_positions)
                max_num = max(numbered_positions)
                min_bul = min(bullet_positions)
                max_bul = max(bullet_positions)
                
                if max_num < min_bul or max_bul < min_num:
                    coherence_score += 0.5  # Clearly separated
                # else could be interleaved which is less ideal
        
        # ---- Feature 5: Proportional Structure Score ----
        # What fraction of the response is in structured elements?
        structured_chars = 0
        
        # Count chars in list items
        list_line_pattern = re.compile(r'^\s*(?:[-*•]|\d+[\.\)])\s+.*$', re.MULTILINE)
        for match in list_line_pattern.finditer(response):
            structured_chars += len(match.group())
        
        # Count chars in headers
        for pattern in [header_md, header_caps, header_colon, bold_header]:
            for match in pattern.finditer(response):
                structured_chars += len(match.group())
        
        # Count chars in code blocks
        for match in code_block_pattern.finditer(response):
            structured_chars += len(match.group())
        
        proportion_structured = structured_chars / total_chars if total_chars > 0 else 0
        
        # Ideal proportion depends on response length
        proportion_score = 0.0
        if total_chars < 200:
            # Short responses don't need much structure
            proportion_score = 1.0 - proportion_structured * 0.5  # Light penalty for over-structuring short text
        elif total_chars < 500:
            # Medium responses benefit from some structure
            if 0.1 <= proportion_structured <= 0.6:
                proportion_score = 1.0
            elif proportion_structured < 0.1:
                proportion_score = 0.5
            else:
                proportion_score = 0.7
        else:
            # Long responses need structure
            if 0.15 <= proportion_structured <= 0.7:
                proportion_score = 1.0
            elif proportion_structured < 0.15:
                proportion_score = 0.3
            else:
                proportion_score = 0.6
        
        # ---- Feature 6: Blank Line / Whitespace Usage ----
        blank_lines = sum(1 for l in lines if l.strip() == '')
        total_lines = len(lines)
        whitespace_ratio = blank_lines / total_lines if total_lines > 0 else 0
        
        whitespace_score = 0.0
        if total_chars > 200:
            if 0.05 <= whitespace_ratio <= 0.4:
                whitespace_score = 1.0
            elif whitespace_ratio < 0.05 and total_lines > 5:
                whitespace_score = 0.2  # No visual breaks
            elif whitespace_ratio > 0.4:
                whitespace_score = 0.5  # Too many blank lines
            else:
                whitespace_score = 0.6
        else:
            whitespace_score = 0.7  # Short text doesn't need much whitespace
        
        # ---- Feature 7: Sentence Structure within Paragraphs ----
        # Check for reasonable sentence lengths and variety
        sentence_pattern = re.compile(r'[^.!?]+[.!?]+')
        sentences = sentence_pattern.findall(response)
        sentence_score = 0.0
        
        if sentences:
            sent_lengths = [len(s.strip()) for s in sentences if len(s.strip()) > 5]
            if len(sent_lengths) >= 2:
                mean_sent = sum(sent_lengths) / len(sent_lengths)
                # Ideal mean sentence length: 50-150 chars
                if 30 <= mean_sent <= 200:
                    sentence_score += 0.5
                
                # Sentence length variety
                sent_var = sum((l - mean_sent) ** 2 for l in sent_lengths) / len(sent_lengths)
                sent_cv = math.sqrt(sent_var) / mean_sent if mean_sent > 0 else 0
                if 0.2 <= sent_cv <= 1.0:
                    sentence_score += 0.5
            elif len(sent_lengths) == 1:
                sentence_score = 0.3
        
        # ---- Feature 8: Response Length Appropriateness ----
        # Longer queries with complex questions deserve longer, more structured responses
        query_len = len(query)
        query_complexity = query.count('?') + query.count('\n') + len(query.split()) / 20
        
        length_score = 0.0
        if total_chars > 50:
            # Base: longer responses tend to need and benefit from structure
            if total_chars > 200:
                length_score = min(1.0, total_chars / 500)
            else:
                length_score = 0.5
        
        # ---- Combine all features with weights ----
        # Normalize each component to roughly 0-1 range
        
        # Variety: 0-9 types -> normalize
        variety_norm = min(1.0, variety_score / 4.0)
        
        # Rhythm: already roughly 0-1.5
        rhythm_norm = min(1.0, rhythm_score)
        
        # Density: roughly -0.5 to 1.5
        density_norm = min(1.0, max(0, (density_score + 0.5) / 2.0))
        
        # Coherence: 0-2
        coherence_norm = min(1.0, coherence_score / 1.5)
        
        # Proportion: 0-1
        proportion_norm = proportion_score
        
        # Whitespace: 0-1
        whitespace_norm = whitespace_score
        
        # Sentence: 0-1
        sentence_norm = sentence_score
        
        # Length: 0-1
        length_norm = length_score
        
        # Weighted combination
        weights = {
            'variety': 2.5,
            'rhythm': 1.0,
            'density': 1.5,
            'coherence': 1.5,
            'proportion': 1.5,
            'whitespace': 1.0,
            'sentence': 0.8,
            'length': 1.2,
        }
        
        total_weight = sum(weights.values())
        
        raw_score = (
            weights['variety'] * variety_norm +
            weights['rhythm'] * rhythm_norm +
            weights['density'] * density_norm +
            weights['coherence'] * coherence_norm +
            weights['proportion'] * proportion_norm +
            weights['whitespace'] * whitespace_norm +
            weights['sentence'] * sentence_norm +
            weights['length'] * length_norm
        ) / total_weight
        
        # Scale to 0-10
        final_score = raw_score * 10.0
        
        # Apply mild length bonus for substantive responses
        if total_chars > 300 and variety_score >= 1:
            final_score += 0.5
        if total_chars > 600 and variety_score >= 2:
            final_score += 0.5
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            return max(0.0, min(5.0, len(str(response)) / 200.0))
        except Exception:
            return 2.0