def judging_function(query, response):
    """
    Evaluate structural organization and formatting quality using a 
    hierarchical document structure analysis approach.
    
    This variant uses a fundamentally different approach: it models the response
    as a tree-like document structure, analyzing nesting depth, structural variety,
    visual rhythm (alternation of element types), and the ratio of structural 
    "scaffolding" to content. It also measures "scanability" - how easy it is
    to extract key information by scanning rather than reading linearly.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        import re
        from collections import Counter
        
        lines = response.split('\n')
        
        # === PHASE 1: Classify each line into a structural type ===
        line_types = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                line_types.append('BLANK')
            elif re.match(r'^#{1,6}\s+\S', stripped):
                line_types.append('MD_HEADER')
            elif re.match(r'^[A-Z][^.!?]*[:]\s*$', stripped) and len(stripped) < 80:
                line_types.append('IMPLICIT_HEADER')
            elif re.match(r'^\*\*[^*]+\*\*\s*$', stripped) or re.match(r'^__[^_]+__\s*$', stripped):
                line_types.append('BOLD_HEADER')
            elif re.match(r'^(\d+[\.\)]\s+|\(?[a-zA-Z][\.\)]\s+)', stripped):
                line_types.append('NUMBERED')
            elif re.match(r'^[\-\*\•\▪\◦\→\➤\►]\s+', stripped):
                line_types.append('BULLET')
            elif re.match(r'^```', stripped):
                line_types.append('CODE_FENCE')
            elif re.match(r'^\|.*\|', stripped):
                line_types.append('TABLE_ROW')
            elif re.match(r'^>\s+', stripped):
                line_types.append('BLOCKQUOTE')
            elif re.match(r'^[-=]{3,}\s*$', stripped):
                line_types.append('SEPARATOR')
            elif len(stripped) > 0:
                line_types.append('TEXT')
            else:
                line_types.append('BLANK')
        
        type_counts = Counter(line_types)
        total_lines = len(line_types)
        non_blank_lines = total_lines - type_counts.get('BLANK', 0)
        
        if non_blank_lines == 0:
            return 0.5
        
        # === PHASE 2: Structural Variety Score ===
        # How many different structural element types are used?
        structural_types = {'MD_HEADER', 'IMPLICIT_HEADER', 'BOLD_HEADER', 
                           'NUMBERED', 'BULLET', 'CODE_FENCE', 'TABLE_ROW',
                           'BLOCKQUOTE', 'SEPARATOR'}
        used_structural = set(line_types) & structural_types
        
        # Variety score: reward using multiple formatting types
        variety_score = min(len(used_structural) / 3.0, 1.0)  # Cap at 3 types
        
        # === PHASE 3: Visual Rhythm Analysis ===
        # Measure the pattern of transitions between element types
        # Good formatting has rhythmic alternation (header->content->list->content)
        # Bad formatting is monotonous (text->text->text->text)
        transitions = 0
        meaningful_transitions = 0
        prev_type = None
        for lt in line_types:
            if lt == 'BLANK':
                continue
            if prev_type is not None and lt != prev_type:
                transitions += 1
                # Especially reward transitions involving structural elements
                if lt in structural_types or prev_type in structural_types:
                    meaningful_transitions += 1
            prev_type = lt
        
        rhythm_score = 0.0
        if non_blank_lines > 1:
            transition_rate = transitions / (non_blank_lines - 1)
            meaningful_rate = meaningful_transitions / (non_blank_lines - 1)
            rhythm_score = min(transition_rate * 1.5, 1.0) * 0.5 + min(meaningful_rate * 3.0, 1.0) * 0.5
        
        # === PHASE 4: Chunking / Scanability Score ===
        # Measure how well content is broken into scannable chunks
        # Identify "chunks" - contiguous blocks of same-type content separated by blanks or headers
        chunks = []
        current_chunk_size = 0
        current_chunk_type = None
        
        for lt in line_types:
            if lt == 'BLANK':
                if current_chunk_size > 0:
                    chunks.append((current_chunk_type, current_chunk_size))
                    current_chunk_size = 0
                    current_chunk_type = None
            elif lt in {'MD_HEADER', 'IMPLICIT_HEADER', 'BOLD_HEADER', 'SEPARATOR'}:
                if current_chunk_size > 0:
                    chunks.append((current_chunk_type, current_chunk_size))
                chunks.append((lt, 1))
                current_chunk_size = 0
                current_chunk_type = None
            else:
                if current_chunk_type is None:
                    current_chunk_type = lt
                current_chunk_size += 1
        
        if current_chunk_size > 0:
            chunks.append((current_chunk_type, current_chunk_size))
        
        scanability_score = 0.0
        if chunks:
            # Ideal chunk size is 2-6 lines; penalize very large chunks
            chunk_sizes = [size for _, size in chunks]
            avg_chunk = sum(chunk_sizes) / len(chunk_sizes)
            num_chunks = len(chunk_sizes)
            
            # Reward having multiple chunks (information is broken up)
            chunk_count_score = min(num_chunks / 4.0, 1.0)
            
            # Penalize very large chunks (wall of text)
            oversized_penalty = sum(max(0, s - 8) for s in chunk_sizes) / max(non_blank_lines, 1)
            size_score = max(0, 1.0 - oversized_penalty)
            
            # Ideal average chunk size
            if avg_chunk < 1.5:
                avg_score = 0.5  # Too fragmented
            elif avg_chunk <= 6:
                avg_score = 1.0  # Ideal
            else:
                avg_score = max(0, 1.0 - (avg_chunk - 6) / 15.0)
            
            scanability_score = chunk_count_score * 0.4 + size_score * 0.35 + avg_score * 0.25
        
        # === PHASE 5: Whitespace Architecture Score ===
        # Analyze the pattern of blank lines - are they used purposefully?
        blank_positions = [i for i, lt in enumerate(line_types) if lt == 'BLANK']
        blank_ratio = len(blank_positions) / max(total_lines, 1)
        
        # Good: 10-30% blank lines for separation
        if blank_ratio == 0:
            whitespace_score = 0.2  # No whitespace separation at all
        elif blank_ratio < 0.08:
            whitespace_score = 0.4
        elif blank_ratio <= 0.35:
            whitespace_score = 1.0
        elif blank_ratio <= 0.5:
            whitespace_score = 0.6
        else:
            whitespace_score = 0.2  # Too much whitespace
        
        # Check if blank lines appear between content blocks (purposeful) vs random
        purposeful_blanks = 0
        for pos in blank_positions:
            before = line_types[pos - 1] if pos > 0 else 'BLANK'
            after = line_types[pos + 1] if pos < total_lines - 1 else 'BLANK'
            # A blank between two non-blank lines is purposeful separation
            if before != 'BLANK' and after != 'BLANK':
                purposeful_blanks += 1
        
        if blank_positions:
            purposeful_ratio = purposeful_blanks / len(blank_positions)
            whitespace_score *= (0.5 + 0.5 * purposeful_ratio)
        
        # === PHASE 6: Inline Formatting Density ===
        # Check for inline structural elements: bold, italic, code spans, links
        bold_count = len(re.findall(r'\*\*[^*]+\*\*|__[^_]+__', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)|(?<!_)_(?!_)[^_]+_(?!_)', response))
        code_span_count = len(re.findall(r'`[^`]+`', response))
        link_count = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', response))
        
        inline_elements = bold_count + italic_count + code_span_count + link_count
        words = response.split()
        word_count = len(words)
        
        # Inline formatting density - reward moderate use
        if word_count > 0:
            inline_density = inline_elements / (word_count / 50.0) if word_count > 50 else inline_elements
            inline_score = min(inline_density / 3.0, 1.0)
        else:
            inline_score = 0.0
        
        # === PHASE 7: Structural Scaffolding Ratio ===
        # What fraction of lines serve as structural scaffolding vs pure content?
        scaffold_types = structural_types | {'BLANK', 'SEPARATOR'}
        scaffold_lines = sum(1 for lt in line_types if lt in scaffold_types)
        scaffold_ratio = scaffold_lines / max(total_lines, 1)
        
        # Ideal: 15-40% scaffolding
        if scaffold_ratio < 0.05:
            scaffold_score = 0.2
        elif scaffold_ratio < 0.15:
            scaffold_score = 0.5 + (scaffold_ratio - 0.05) * 5.0
        elif scaffold_ratio <= 0.45:
            scaffold_score = 1.0
        elif scaffold_ratio <= 0.6:
            scaffold_score = 1.0 - (scaffold_ratio - 0.45) * 3.33
        else:
            scaffold_score = 0.2
        
        # === PHASE 8: Opening and Closing Structure ===
        # Good responses often have a clear opening statement and closing
        non_blank_indices = [i for i, lt in enumerate(line_types) if lt != 'BLANK']
        
        opening_score = 0.5  # Default neutral
        if non_blank_indices:
            first_type = line_types[non_blank_indices[0]]
            # Starting with a header or clear text is good
            if first_type in {'MD_HEADER', 'BOLD_HEADER', 'IMPLICIT_HEADER'}:
                opening_score = 1.0
            elif first_type == 'TEXT':
                first_line = lines[non_blank_indices[0]].strip()
                # Short opening line (topic sentence) is good
                if len(first_line.split()) <= 30:
                    opening_score = 0.7
                else:
                    opening_score = 0.5
        
        # === PHASE 9: Consistent List Formatting ===
        # If lists are used, are they consistent?
        list_consistency_score = 0.5  # Default neutral
        bullet_lines = [i for i, lt in enumerate(line_types) if lt == 'BULLET']
        numbered_lines = [i for i, lt in enumerate(line_types) if lt == 'NUMBERED']
        
        if bullet_lines or numbered_lines:
            list_consistency_score = 0.8
            # Check if bullets are grouped together (not scattered randomly)
            all_list_lines = sorted(bullet_lines + numbered_lines)
            if len(all_list_lines) >= 2:
                gaps = [all_list_lines[i+1] - all_list_lines[i] for i in range(len(all_list_lines)-1)]
                # Most list items should be close together (gap of 1-3)
                close_gaps = sum(1 for g in gaps if g <= 3)
                grouping_ratio = close_gaps / len(gaps)
                list_consistency_score = 0.6 + 0.4 * grouping_ratio
        
        # === PHASE 10: Length-Adjusted Expectations ===
        # Longer responses should have MORE structure; short responses get a pass
        length_chars = len(response)
        
        if length_chars < 100:
            # Very short - structure matters less
            length_multiplier = 0.7
            # But give a base score for being concise
            base_bonus = 2.0
        elif length_chars < 300:
            length_multiplier = 0.85
            base_bonus = 1.5
        elif length_chars < 800:
            length_multiplier = 1.0
            base_bonus = 1.0
        elif length_chars < 2000:
            length_multiplier = 1.1
            base_bonus = 0.5
        else:
            length_multiplier = 1.2
            base_bonus = 0.0
        
        # === PHASE 11: Wall-of-Text Detection ===
        # Specifically detect and penalize wall-of-text
        wall_penalty = 0.0
        
        # Find longest run of consecutive TEXT lines
        max_text_run = 0
        current_run = 0
        for lt in line_types:
            if lt == 'TEXT':
                current_run += 1
                max_text_run = max(max_text_run, current_run)
            else:
                current_run = 0
        
        if max_text_run > 10 and length_chars > 500:
            wall_penalty = min((max_text_run - 10) * 0.1, 0.4)
        
        # Also check character-level: very long lines suggest no line breaks
        long_line_count = sum(1 for line in lines if len(line.strip()) > 200)
        if long_line_count > 0 and length_chars > 400:
            wall_penalty += min(long_line_count * 0.15, 0.3)
        
        # === COMBINE SCORES ===
        # Weight the components
        raw_score = (
            variety_score * 1.5 +
            rhythm_score * 1.5 +
            scanability_score * 2.0 +
            whitespace_score * 1.5 +
            inline_score * 0.8 +
            scaffold_score * 1.2 +
            opening_score * 0.5 +
            list_consistency_score * 0.5
        )
        
        max_possible = 1.5 + 1.5 + 2.0 + 1.5 + 0.8 + 1.2 + 0.5 + 0.5  # = 9.5
        
        # Normalize to 0-1
        normalized = raw_score / max_possible
        
        # Apply length adjustment
        adjusted = normalized * length_multiplier
        
        # Apply wall-of-text penalty
        adjusted = max(0, adjusted - wall_penalty)
        
        # Scale to 0-10 with base bonus
        final_score = base_bonus + adjusted * 8.0
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0