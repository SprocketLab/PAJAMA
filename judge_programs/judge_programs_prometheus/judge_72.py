def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical document structure analysis approach.
    
    This variant uses a fundamentally different algorithm: it models the response
    as a hierarchical document tree, analyzing nesting depth, structural variety,
    visual rhythm patterns, and information density distribution across segments.
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
        total_words = len(response.split())
        
        if total_words < 3:
            return 0.5
        
        # =====================================================
        # COMPONENT 1: Visual Rhythm Analysis
        # Analyze the pattern of line lengths to detect structured formatting.
        # Well-organized text has varied but purposeful line-length patterns,
        # not uniform wall-of-text blocks.
        # =====================================================
        
        line_lengths = [len(l.strip()) for l in lines]
        
        # Classify each line by its visual role
        LINE_EMPTY = 0
        LINE_SHORT = 1   # < 40 chars (headers, list items, short statements)
        LINE_MEDIUM = 2  # 40-120 chars
        LINE_LONG = 3    # > 120 chars
        
        line_types = []
        for length in line_lengths:
            if length == 0:
                line_types.append(LINE_EMPTY)
            elif length < 40:
                line_types.append(LINE_SHORT)
            elif length <= 120:
                line_types.append(LINE_MEDIUM)
            else:
                line_types.append(LINE_LONG)
        
        # Calculate rhythm diversity (bigram entropy of line types)
        rhythm_score = 0.0
        if len(line_types) >= 2:
            bigrams = [(line_types[i], line_types[i+1]) for i in range(len(line_types)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            bigram_entropy = 0.0
            for count in bigram_counts.values():
                p = count / total_bigrams
                if p > 0:
                    bigram_entropy -= p * math.log2(p)
            # Normalize: max entropy for 4 types = log2(16) = 4
            rhythm_score = min(bigram_entropy / 3.0, 1.0)
        
        # =====================================================
        # COMPONENT 2: Structural Element Taxonomy
        # Instead of just counting bullets/headers, classify every line
        # into a structural taxonomy and score based on variety and appropriateness
        # =====================================================
        
        taxonomy = {
            'header': 0,
            'numbered_item': 0,
            'bullet_item': 0,
            'label_value': 0,  # "Key: value" patterns
            'emphasis_line': 0,  # Lines with bold/italic markers
            'paragraph_start': 0,
            'continuation': 0,
            'separator': 0,
            'empty': 0,
            'code_block': 0,
        }
        
        prev_was_empty = True
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                taxonomy['empty'] += 1
                prev_was_empty = True
                continue
            
            classified = False
            
            # Headers: markdown-style or ALL CAPS short lines
            if re.match(r'^#{1,6}\s+\S', stripped):
                taxonomy['header'] += 1
                classified = True
            elif len(stripped) < 60 and stripped.endswith(':') and not re.search(r'[.!?]', stripped[:-1]):
                taxonomy['header'] += 1
                classified = True
            elif len(stripped) < 50 and stripped.isupper() and len(stripped.split()) >= 2:
                taxonomy['header'] += 1
                classified = True
            
            # Numbered items
            if not classified and re.match(r'^(\d+[\.\)]\s+|\([a-z\d]+\)\s+|[a-z][\.\)]\s+)', stripped, re.IGNORECASE):
                taxonomy['numbered_item'] += 1
                classified = True
            
            # Bullet items
            if not classified and re.match(r'^[\-\*\•\◦\▪\►\→\➤\·\‣]\s+', stripped):
                taxonomy['bullet_item'] += 1
                classified = True
            
            # Label-value pairs ("Something: explanation")
            if not classified and re.match(r'^[A-Z][^:]{2,30}:\s+\S', stripped):
                taxonomy['label_value'] += 1
                classified = True
            
            # Emphasis lines (bold/italic markdown)
            if not classified and (re.match(r'^\*\*.*\*\*', stripped) or re.match(r'^\*[^*]+\*', stripped)):
                taxonomy['emphasis_line'] += 1
                classified = True
            
            # Code blocks
            if not classified and (stripped.startswith('```') or stripped.startswith('    ') and re.search(r'[{}\[\]();=]', stripped)):
                taxonomy['code_block'] += 1
                classified = True
            
            # Separator lines
            if not classified and re.match(r'^[\-=_\*]{3,}$', stripped):
                taxonomy['separator'] += 1
                classified = True
            
            # Paragraph start vs continuation
            if not classified:
                if prev_was_empty:
                    taxonomy['paragraph_start'] += 1
                else:
                    taxonomy['continuation'] += 1
            
            prev_was_empty = False
        
        # Score structural variety: how many different non-trivial types are used?
        structural_types_used = sum(1 for k, v in taxonomy.items() 
                                     if v > 0 and k not in ('empty', 'continuation'))
        variety_score = min(structural_types_used / 4.0, 1.0)
        
        # Score structural density: ratio of structural elements to total non-empty lines
        structural_elements = (taxonomy['header'] + taxonomy['numbered_item'] + 
                              taxonomy['bullet_item'] + taxonomy['label_value'] + 
                              taxonomy['emphasis_line'] + taxonomy['code_block'] +
                              taxonomy['separator'])
        
        if len(non_empty_lines) > 0:
            structural_density = min(structural_elements / max(len(non_empty_lines), 1), 0.8)
        else:
            structural_density = 0.0
        
        # =====================================================
        # COMPONENT 3: Segmentation Quality
        # Analyze how well the response is segmented into digestible chunks.
        # Compute statistics about segment sizes (paragraphs/blocks).
        # =====================================================
        
        # Split into segments by empty lines
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
        
        segmentation_score = 0.0
        if num_segments == 0:
            segmentation_score = 0.0
        elif num_segments == 1:
            # Single block: penalize based on length
            words_in_block = len(segments[0].split())
            if words_in_block < 50:
                segmentation_score = 0.4  # Short single block is okay
            elif words_in_block < 100:
                segmentation_score = 0.2
            else:
                segmentation_score = 0.05  # Wall of text
        else:
            # Multiple segments: check balance
            segment_word_counts = [len(s.split()) for s in segments]
            avg_segment_words = sum(segment_word_counts) / len(segment_word_counts)
            
            # Ideal segment size: 20-80 words
            size_quality = 0.0
            for wc in segment_word_counts:
                if 15 <= wc <= 80:
                    size_quality += 1.0
                elif 5 <= wc <= 120:
                    size_quality += 0.6
                elif wc < 5:
                    size_quality += 0.3
                else:
                    size_quality += 0.2
            size_quality /= len(segment_word_counts)
            
            # Coefficient of variation of segment sizes (moderate variation is good)
            if avg_segment_words > 0:
                std_dev = math.sqrt(sum((wc - avg_segment_words)**2 for wc in segment_word_counts) / len(segment_word_counts))
                cv = std_dev / avg_segment_words
                # CV between 0.2 and 0.8 is ideal
                if 0.15 <= cv <= 0.9:
                    balance_score = 1.0
                elif cv < 0.15:
                    balance_score = 0.6  # Too uniform
                else:
                    balance_score = 0.4  # Too varied
            else:
                balance_score = 0.3
            
            # Number of segments relative to total words
            segments_per_100_words = (num_segments / max(total_words, 1)) * 100
            if 2 <= segments_per_100_words <= 8:
                density_bonus = 1.0
            elif 1 <= segments_per_100_words <= 12:
                density_bonus = 0.7
            else:
                density_bonus = 0.3
            
            segmentation_score = 0.4 * size_quality + 0.3 * balance_score + 0.3 * density_bonus
        
        # =====================================================
        # COMPONENT 4: Discourse Flow Markers
        # Instead of just counting transition words, analyze position-based
        # discourse markers that signal organizational intent.
        # =====================================================
        
        # Opening markers (first segment signals)
        opening_patterns = [
            r'^(imagine|picture|think of|consider|let\'s|here\'s|i can|i\'m)',
            r'^(to begin|first|the first|starting)',
            r'^(hey|hi|hello|alright|okay|sure)',
        ]
        
        # Progression markers (mid-response signals)
        progression_patterns = [
            r'\b(additionally|furthermore|moreover|also|next|then|another)\b',
            r'\b(however|but|on the other hand|conversely|although|yet)\b',
            r'\b(for example|for instance|such as|like|specifically)\b',
            r'\b(importantly|crucially|notably|significantly)\b',
            r'\b(first|second|third|finally|lastly|in addition)\b',
        ]
        
        # Closing markers (final segment signals)
        closing_patterns = [
            r'\b(in conclusion|to summarize|overall|in summary|remember)\b',
            r'\b(finally|lastly|in the end|ultimately|to wrap up)\b',
            r'\b(don\'t hesitate|feel free|hope this|good luck|take care)\b',
        ]
        
        response_lower = response.lower()
        
        # Check opening
        first_50_chars = response_lower[:80]
        has_opening = any(re.search(p, first_50_chars) for p in opening_patterns)
        
        # Count progression markers
        progression_count = 0
        for pattern in progression_patterns:
            progression_count += len(re.findall(pattern, response_lower))
        
        # Check closing
        last_portion = response_lower[-150:] if len(response_lower) > 150 else response_lower
        has_closing = any(re.search(p, last_portion) for p in closing_patterns)
        
        # Discourse flow score
        discourse_score = 0.0
        if has_opening:
            discourse_score += 0.2
        if has_closing:
            discourse_score += 0.2
        
        # Progression markers per 100 words
        prog_per_100 = (progression_count / max(total_words, 1)) * 100
        if 1.5 <= prog_per_100 <= 6:
            discourse_score += 0.6
        elif 0.5 <= prog_per_100 <= 8:
            discourse_score += 0.35
        elif prog_per_100 > 0:
            discourse_score += 0.15
        
        # =====================================================
        # COMPONENT 5: Indentation and Nesting Depth Analysis
        # Analyze leading whitespace patterns to detect hierarchical structure
        # =====================================================
        
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.add(leading_spaces)
        
        nesting_score = 0.0
        num_indent_levels = len(indent_levels)
        if num_indent_levels >= 3:
            nesting_score = 1.0
        elif num_indent_levels == 2:
            nesting_score = 0.6
        elif num_indent_levels == 1:
            nesting_score = 0.2
        
        # =====================================================
        # COMPONENT 6: Sentence Length Variation within Paragraphs
        # Good writing has varied sentence lengths for readability
        # =====================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        sentence_variation_score = 0.0
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            if avg_sent_len > 0:
                sent_std = math.sqrt(sum((sl - avg_sent_len)**2 for sl in sent_lengths) / len(sent_lengths))
                sent_cv = sent_std / avg_sent_len
                # Moderate variation (CV 0.3-0.7) is ideal
                if 0.25 <= sent_cv <= 0.8:
                    sentence_variation_score = 1.0
                elif 0.15 <= sent_cv <= 1.0:
                    sentence_variation_score = 0.6
                else:
                    sentence_variation_score = 0.25
            
            # Also check average sentence length (10-25 words ideal)
            if 8 <= avg_sent_len <= 28:
                sentence_variation_score *= 1.0
            elif 5 <= avg_sent_len <= 35:
                sentence_variation_score *= 0.8
            else:
                sentence_variation_score *= 0.5
        elif len(sentences) >= 1:
            sentence_variation_score = 0.3
        
        # =====================================================
        # COMPONENT 7: Response Length Appropriateness
        # Based on query complexity, is the response appropriately sized?
        # =====================================================
        
        query_words = len(query.split())
        query_questions = query.count('?')
        
        # Estimate expected complexity
        query_complexity = min(query_words / 30.0, 1.0)
        
        length_score = 0.0
        if total_words < 20:
            length_score = 0.2
        elif total_words < 50:
            length_score = 0.5 if query_complexity < 0.3 else 0.3
        elif total_words < 150:
            length_score = 0.8
        elif total_words < 300:
            length_score = 0.9
        else:
            length_score = 0.7  # Very long might be okay but slight penalty
        
        # Bonus for good length-to-structure ratio
        if total_words > 80 and num_segments >= 2:
            length_score = min(length_score + 0.1, 1.0)
        
        # =====================================================
        # COMPONENT 8: Wall-of-Text Penalty
        # Specifically detect and penalize wall-of-text patterns
        # =====================================================
        
        wall_penalty = 0.0
        
        # Check if there's a single long block with no breaks
        if num_segments == 1 and total_words > 80:
            wall_penalty += 0.3
        
        # Check for very long lines (>200 chars without breaks)
        long_lines = sum(1 for l in non_empty_lines if len(l.strip()) > 200)
        if long_lines > 0:
            wall_penalty += 0.15 * min(long_lines, 3)
        
        # Check ratio of empty lines to non-empty lines
        if len(non_empty_lines) > 3:
            whitespace_ratio = taxonomy['empty'] / len(non_empty_lines)
            if whitespace_ratio < 0.05:
                wall_penalty += 0.15
        
        wall_penalty = min(wall_penalty, 0.6)
        
        # =====================================================
        # FINAL SCORING: Weighted combination
        # =====================================================
        
        raw_score = (
            0.12 * rhythm_score +
            0.18 * variety_score +
            0.10 * structural_density * 2.5 +  # Scale up since max is 0.8
            0.22 * segmentation_score +
            0.13 * discourse_score +
            0.03 * nesting_score +
            0.10 * sentence_variation_score +
            0.12 * length_score
        )
        
        # Apply wall-of-text penalty
        raw_score = raw_score * (1.0 - wall_penalty)
        
        # Ensure raw_score is in [0, 1]
        raw_score = max(0.0, min(1.0, raw_score))
        
        # Map to 1-5 scale
        final_score = 1.0 + raw_score * 4.0
        
        # Round to 1 decimal
        final_score = round(final_score, 2)
        
        return final_score
        
    except Exception:
        return 2.5