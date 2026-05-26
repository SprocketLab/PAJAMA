def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses an information-theoretic and visual rhythm approach:
    - Analyzes the "visual rhythm" of line lengths (variance patterns)
    - Measures information density distribution across segments
    - Evaluates structural entropy (diversity of structural elements)
    - Checks for hierarchical depth and nesting patterns
    - Analyzes sentence length variation as a readability signal
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        score = 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # ============================================================
        # FEATURE 1: Visual Rhythm Analysis (line length distribution)
        # ============================================================
        # Good formatting creates a varied "visual rhythm" - short headers,
        # medium list items, longer paragraphs. Wall-of-text has uniform rhythm.
        
        visual_rhythm_score = 0.0
        if len(non_empty_lines) >= 2:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            
            # Categorize lines by length: short (<30), medium (30-80), long (>80)
            short_lines = sum(1 for l in line_lengths if l < 30)
            medium_lines = sum(1 for l in line_lengths if 30 <= l < 80)
            long_lines = sum(1 for l in line_lengths if l >= 80)
            
            total = len(line_lengths)
            categories_present = sum(1 for c in [short_lines, medium_lines, long_lines] if c > 0)
            
            # More categories = more visual variety = better structure
            if categories_present >= 3:
                visual_rhythm_score = 3.0
            elif categories_present == 2:
                visual_rhythm_score = 2.0
            else:
                visual_rhythm_score = 0.5
            
            # Coefficient of variation of line lengths (normalized variability)
            mean_len = sum(line_lengths) / total
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in line_lengths) / total
                cv = math.sqrt(variance) / mean_len
                # CV between 0.3 and 1.5 is ideal for structured text
                if 0.3 <= cv <= 1.5:
                    visual_rhythm_score += 2.0
                elif 0.15 <= cv < 0.3 or 1.5 < cv <= 2.0:
                    visual_rhythm_score += 1.0
        else:
            # Single block of text
            visual_rhythm_score = 0.0
        
        score += min(visual_rhythm_score, 5.0)
        
        # ============================================================
        # FEATURE 2: Structural Element Entropy
        # ============================================================
        # Measures diversity of structural elements used. Higher entropy = 
        # more diverse formatting = generally better organized.
        
        element_counts = Counter()
        
        for line in non_empty_lines:
            stripped = line.strip()
            
            # Numbered items (1. 2. etc or 1) 2) etc)
            if re.match(r'^\d+[\.\)]\s', stripped):
                element_counts['numbered'] += 1
            # Bullet points
            elif re.match(r'^[\-\*\•\◦\▪\►]\s', stripped):
                element_counts['bullet'] += 1
            # Lines ending with colon (label/header-like)
            elif stripped.endswith(':') and len(stripped) < 80:
                element_counts['label'] += 1
            # Short emphatic lines (potential headers/subheaders)
            elif len(stripped) < 50 and not stripped.endswith('.') and not stripped.endswith(','):
                # Check if it looks like a header (capitalized, no ending punctuation)
                words = stripped.split()
                if len(words) >= 1 and len(words) <= 8:
                    cap_words = sum(1 for w in words if w[0].isupper() or w.startswith('#'))
                    if cap_words >= len(words) * 0.5 or stripped.startswith('#'):
                        element_counts['header'] += 1
                    else:
                        element_counts['text'] += 1
                else:
                    element_counts['text'] += 1
            else:
                element_counts['text'] += 1
        
        # Also detect inline structural patterns
        # Bold/italic markers
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+(?<!\*)\*(?!\*)', response))
        if bold_count > 0:
            element_counts['bold'] = bold_count
        if italic_count > 0:
            element_counts['italic'] = italic_count
        
        # Parenthetical asides
        paren_count = len(re.findall(r'\([^)]{5,}\)', response))
        if paren_count > 0:
            element_counts['parenthetical'] = paren_count
        
        # Calculate Shannon entropy of structural elements
        struct_entropy = 0.0
        total_elements = sum(element_counts.values())
        if total_elements > 0:
            for count in element_counts.values():
                p = count / total_elements
                if p > 0:
                    struct_entropy -= p * math.log2(p)
        
        # Normalize: max entropy for 7 categories is log2(7) ≈ 2.81
        normalized_entropy = min(struct_entropy / 2.81, 1.0) if total_elements > 0 else 0.0
        
        # Bonus for having non-text structural elements
        non_text_types = sum(1 for k in element_counts if k != 'text' and element_counts[k] > 0)
        
        entropy_score = normalized_entropy * 5.0 + min(non_text_types * 1.0, 4.0)
        score += min(entropy_score, 8.0)
        
        # ============================================================
        # FEATURE 3: Information Density Distribution
        # ============================================================
        # Split response into quartiles and measure info density per segment.
        # Well-structured responses distribute information more evenly.
        
        density_score = 0.0
        words = response.split()
        total_words = len(words)
        
        if total_words >= 20:
            # Split into 4 segments
            seg_size = total_words // 4
            segments = []
            for i in range(4):
                start = i * seg_size
                end = start + seg_size if i < 3 else total_words
                seg_text = ' '.join(words[start:end])
                segments.append(seg_text)
            
            # Measure "information density" via unique word ratio per segment
            densities = []
            for seg in segments:
                seg_words = seg.lower().split()
                if len(seg_words) > 0:
                    unique_ratio = len(set(seg_words)) / len(seg_words)
                    densities.append(unique_ratio)
            
            if len(densities) >= 2:
                # More uniform density = better distributed information
                mean_density = sum(densities) / len(densities)
                if mean_density > 0:
                    density_variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
                    density_cv = math.sqrt(density_variance) / mean_density
                    
                    # Low CV = uniform distribution = good
                    if density_cv < 0.05:
                        density_score = 3.0
                    elif density_cv < 0.10:
                        density_score = 2.5
                    elif density_cv < 0.15:
                        density_score = 2.0
                    elif density_cv < 0.25:
                        density_score = 1.5
                    else:
                        density_score = 0.5
        
        score += density_score
        
        # ============================================================
        # FEATURE 4: Sentence Length Variation Pattern
        # ============================================================
        # Good writing mixes sentence lengths. Analyze the pattern of 
        # sentence lengths for intentional variation vs monotony.
        
        sentence_score = 0.0
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if len(sentences) >= 3:
            sent_word_counts = [len(s.split()) for s in sentences]
            
            # Calculate "direction changes" in sentence length
            # (short->long->short patterns indicate intentional rhythm)
            direction_changes = 0
            for i in range(2, len(sent_word_counts)):
                prev_diff = sent_word_counts[i-1] - sent_word_counts[i-2]
                curr_diff = sent_word_counts[i] - sent_word_counts[i-1]
                if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                    direction_changes += 1
            
            max_changes = len(sent_word_counts) - 2
            if max_changes > 0:
                change_ratio = direction_changes / max_changes
                # Moderate variation (0.3-0.7) is ideal
                if 0.3 <= change_ratio <= 0.7:
                    sentence_score = 3.0
                elif 0.2 <= change_ratio < 0.3 or 0.7 < change_ratio <= 0.8:
                    sentence_score = 2.0
                else:
                    sentence_score = 1.0
            
            # Also reward having a mix of short and long sentences
            short_sents = sum(1 for c in sent_word_counts if c <= 8)
            long_sents = sum(1 for c in sent_word_counts if c > 20)
            if short_sents > 0 and long_sents > 0:
                sentence_score += 1.0
        
        score += min(sentence_score, 4.0)
        
        # ============================================================
        # FEATURE 5: Paragraph Segmentation Quality
        # ============================================================
        # Analyze paragraph structure via blank line separations.
        # Good responses have multiple focused paragraphs.
        
        para_score = 0.0
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paras = len(paragraphs)
        
        if num_paras >= 3:
            para_score += 3.0
        elif num_paras == 2:
            para_score += 2.0
        elif num_paras == 1:
            # Single paragraph - check if it's a wall of text
            if total_words > 100:
                para_score += 0.0  # Wall of text penalty
            else:
                para_score += 1.0
        
        # Check paragraph size balance
        if num_paras >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            max_para = max(para_lengths)
            min_para = min(para_lengths)
            if max_para > 0:
                balance_ratio = min_para / max_para
                if balance_ratio > 0.3:
                    para_score += 1.5
                elif balance_ratio > 0.15:
                    para_score += 0.75
        
        score += min(para_score, 4.5)
        
        # ============================================================
        # FEATURE 6: Hierarchical Nesting Detection
        # ============================================================
        # Detect indentation levels and nested structures.
        
        nesting_score = 0.0
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_level = leading_spaces // 2  # Normalize to 2-space increments
                indent_levels.add(indent_level)
        
        # Multiple indent levels suggest hierarchical organization
        if len(indent_levels) >= 3:
            nesting_score = 2.0
        elif len(indent_levels) == 2:
            nesting_score = 1.0
        
        # Also check for colon-introduced lists (a structural pattern)
        colon_then_list = 0
        for i in range(len(non_empty_lines) - 1):
            if non_empty_lines[i].strip().endswith(':'):
                next_line = non_empty_lines[i + 1].strip()
                if re.match(r'^[\d\-\*\•]', next_line):
                    colon_then_list += 1
        
        if colon_then_list > 0:
            nesting_score += min(colon_then_list * 1.0, 2.0)
        
        score += min(nesting_score, 3.5)
        
        # ============================================================
        # FEATURE 7: Opening and Closing Structure
        # ============================================================
        # Well-organized responses often have a clear opening and closing.
        
        framing_score = 0.0
        
        if len(sentences) >= 3:
            first_sent = sentences[0].lower().strip()
            last_sent = sentences[-1].lower().strip() if sentences[-1].strip() else (
                sentences[-2].lower().strip() if len(sentences) > 1 else ''
            )
            
            # Opening signals
            opening_patterns = [
                r'^(i |let|here|sure|great|absolutely|of course|certainly|to |the |imagine|'
                r'hey|hi |hello|welcome|thank|ok|okay|alright|first|before)',
                r'^(i\'m |i can|i\'d|i understand|i hear|i see|it\'s|it sounds)',
            ]
            for pat in opening_patterns:
                if re.match(pat, first_sent):
                    framing_score += 0.75
                    break
            
            # Closing signals
            closing_patterns = [
                r'(remember|hope|good luck|feel free|don\'t hesitate|let me know|'
                r'in summary|in conclusion|overall|finally|lastly|to sum up|'
                r'reach out|happy to help|best wishes|take care)',
            ]
            for pat in closing_patterns:
                if re.search(pat, last_sent):
                    framing_score += 0.75
                    break
        
        score += min(framing_score, 1.5)
        
        # ============================================================
        # FEATURE 8: Response Length Appropriateness
        # ============================================================
        # Very short responses rarely have good structure; very long ones
        # need more structure to be readable.
        
        length_score = 0.0
        if total_words < 20:
            length_score = -1.0
        elif total_words < 50:
            # Short response - structure less critical but still matters
            length_score = 0.5
        elif 50 <= total_words <= 300:
            # Good range for structured responses
            length_score = 1.5
        elif total_words > 300:
            # Long response - needs structure even more
            # Check if structure scales with length
            structural_elements = sum(v for k, v in element_counts.items() if k != 'text')
            if structural_elements >= 3:
                length_score = 2.0
            elif structural_elements >= 1:
                length_score = 0.5
            else:
                length_score = -1.0  # Long unstructured = bad
        
        score += length_score
        
        # ============================================================
        # NORMALIZATION: Map to 1-5 scale
        # ============================================================
        # Theoretical range: roughly -1 to ~28
        # Practical range for good responses: 5-25
        
        # Clamp and normalize
        raw_score = score
        
        # Sigmoid-like mapping to 1-5 range
        # Center around ~12 as midpoint (score of 3)
        normalized = 1.0 + 4.0 / (1.0 + math.exp(-0.25 * (raw_score - 10)))
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, round(normalized, 2)))
        
        return final_score
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5