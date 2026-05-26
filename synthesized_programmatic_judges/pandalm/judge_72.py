def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality.
    
    This variant uses a VISUAL LAYOUT SIMULATION approach:
    - Simulates how the text would render visually
    - Measures "visual rhythm" (alternation between different element types)
    - Computes a "scanability index" based on entry points for the eye
    - Analyzes hierarchical depth of the document structure
    - Measures the "chunking ratio" (how well info is chunked vs monolithic)
    
    This is fundamentally different from previous variants that count bullets/headers/paragraphs.
    Instead, this classifies each line into a visual element type and analyzes the SEQUENCE
    and PATTERN of those types.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        # Very short responses get a baseline proportional to length
        if len(response) < 20:
            return 1.0
        
        lines = response.split('\n')
        
        # === PHASE 1: Classify each line into a visual element type ===
        # Types: 'blank', 'header', 'bullet', 'numbered', 'short_text', 'medium_text', 'long_text', 'code', 'separator'
        
        def classify_line(line):
            stripped = line.strip()
            if len(stripped) == 0:
                return 'blank'
            if re.match(r'^#{1,6}\s', stripped):
                return 'header'
            if re.match(r'^[\*\-\+•◦▪▸►]\s', stripped):
                return 'bullet'
            if re.match(r'^\d+[\.\)]\s', stripped):
                return 'numbered'
            if re.match(r'^[=\-\*_]{3,}\s*$', stripped):
                return 'separator'
            if re.match(r'^(```|~~~)', stripped):
                return 'code'
            if stripped.endswith(':') and len(stripped.split()) <= 8:
                return 'header'  # label-like lines
            # Bold/emphasized standalone lines as headers
            if re.match(r'^\*\*[^*]+\*\*\s*$', stripped) and len(stripped.split()) <= 10:
                return 'header'
            
            word_count = len(stripped.split())
            if word_count <= 6:
                return 'short_text'
            elif word_count <= 25:
                return 'medium_text'
            else:
                return 'long_text'
        
        line_types = [classify_line(l) for l in lines]
        non_blank_types = [t for t in line_types if t != 'blank']
        
        if len(non_blank_types) == 0:
            return 0.0
        
        # === PHASE 2: Visual Rhythm Score ===
        # Measures how much variety and alternation exists in line types
        # A well-structured doc alternates between headers, content, lists, etc.
        
        type_transitions = 0
        for i in range(1, len(non_blank_types)):
            if non_blank_types[i] != non_blank_types[i-1]:
                type_transitions += 1
        
        max_transitions = max(len(non_blank_types) - 1, 1)
        rhythm_ratio = type_transitions / max_transitions  # 0 to 1
        
        # Count unique non-blank types used
        unique_types = set(non_blank_types)
        type_diversity = len(unique_types)
        
        # Rhythm score: reward both transitions and diversity
        rhythm_score = min(rhythm_ratio * 3.0 + (type_diversity - 1) * 1.5, 10.0)
        
        # === PHASE 3: Scanability Index ===
        # "Entry points" are visual anchors that help a reader scan: headers, bullets, numbers, bold text, separators
        
        entry_point_types = {'header', 'bullet', 'numbered', 'separator'}
        entry_points = sum(1 for t in non_blank_types if t in entry_point_types)
        
        # Also count inline formatting as mini entry points
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        inline_code_count = len(re.findall(r'`[^`]+`', response))
        colon_definitions = len(re.findall(r'\w+\s*:\s+\w', response))
        
        total_entry_points = entry_points + bold_count * 0.5 + inline_code_count * 0.3 + colon_definitions * 0.3
        
        # Normalize by response length (words)
        total_words = len(response.split())
        # Ideal: roughly 1 entry point per 15-30 words
        entry_density = total_entry_points / max(total_words / 20.0, 1.0)
        # Sweet spot around 1.0
        scanability_score = min(entry_density * 5.0, 10.0)
        
        # === PHASE 4: Chunking Ratio ===
        # How well is the text broken into digestible chunks?
        # A "chunk" is a contiguous group of non-blank lines separated by blanks
        
        chunks = []
        current_chunk_lines = 0
        for t in line_types:
            if t == 'blank':
                if current_chunk_lines > 0:
                    chunks.append(current_chunk_lines)
                    current_chunk_lines = 0
            else:
                current_chunk_lines += 1
        if current_chunk_lines > 0:
            chunks.append(current_chunk_lines)
        
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            return 1.0
        
        # Ideal chunk size: 2-6 lines
        chunk_quality_scores = []
        for c in chunks:
            if 2 <= c <= 6:
                chunk_quality_scores.append(1.0)
            elif c == 1:
                chunk_quality_scores.append(0.7)
            elif 7 <= c <= 10:
                chunk_quality_scores.append(0.5)
            else:  # very long chunks (wall of text)
                chunk_quality_scores.append(max(0.1, 1.0 - (c - 10) * 0.05))
        
        avg_chunk_quality = sum(chunk_quality_scores) / len(chunk_quality_scores)
        
        # Reward having multiple chunks (but not too many tiny ones)
        if total_words > 50:
            ideal_chunks = max(2, total_words // 30)
            chunk_count_ratio = min(num_chunks / ideal_chunks, 2.0)
            if chunk_count_ratio > 1.0:
                chunk_count_bonus = 2.0 - (chunk_count_ratio - 1.0) * 0.5
            else:
                chunk_count_bonus = chunk_count_ratio * 2.0
        else:
            chunk_count_bonus = 1.0 if num_chunks >= 1 else 0.0
        
        chunking_score = avg_chunk_quality * 5.0 + chunk_count_bonus * 2.5
        chunking_score = min(chunking_score, 10.0)
        
        # === PHASE 5: Hierarchical Depth ===
        # Does the response have a clear hierarchy? (e.g., headers -> sub-content -> lists)
        
        hierarchy_depth = 0
        has_headers = 'header' in unique_types
        has_lists = 'bullet' in unique_types or 'numbered' in unique_types
        has_body = 'medium_text' in unique_types or 'long_text' in unique_types
        has_separators = 'separator' in unique_types
        
        if has_headers:
            hierarchy_depth += 1
        if has_lists:
            hierarchy_depth += 1
        if has_body:
            hierarchy_depth += 1
        if has_separators:
            hierarchy_depth += 0.5
        
        # Check for nested structures (indented bullets, sub-lists)
        indented_lines = sum(1 for l in lines if l.startswith('  ') and l.strip())
        if indented_lines > 0:
            hierarchy_depth += 0.5
        
        # Check for header levels
        header_levels = set()
        for l in lines:
            m = re.match(r'^(#{1,6})\s', l.strip())
            if m:
                header_levels.add(len(m.group(1)))
        if len(header_levels) > 1:
            hierarchy_depth += 0.5
        
        hierarchy_score = min(hierarchy_depth * 2.5, 10.0)
        
        # === PHASE 6: Sentence Length Variance within paragraphs ===
        # Good writing varies sentence length. Monotonous = bad structure.
        
        sentences = re.split(r'[.!?]+\s+', response)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Ideal CV around 0.3-0.6 (some variety but not chaotic)
            if 0.2 <= cv <= 0.7:
                variance_score = 3.0
            elif cv < 0.2:
                variance_score = cv * 10.0  # too uniform
            else:
                variance_score = max(0, 3.0 - (cv - 0.7) * 2.0)
        else:
            variance_score = 1.0
        
        # === PHASE 7: Wall-of-text penalty ===
        # Specifically detect and penalize monolithic text blocks
        
        wall_penalty = 0.0
        
        # Single chunk with many words
        if num_chunks == 1 and total_words > 80:
            wall_penalty += min((total_words - 80) * 0.03, 5.0)
        
        # Very long lines (in terms of characters)
        long_line_count = sum(1 for l in lines if len(l.strip()) > 200)
        if long_line_count > 0:
            wall_penalty += long_line_count * 0.5
        
        # No formatting at all in a long response
        if total_words > 60 and entry_points == 0 and bold_count == 0 and num_chunks <= 2:
            wall_penalty += 3.0
        
        wall_penalty = min(wall_penalty, 8.0)
        
        # === PHASE 8: Logical grouping signal ===
        # Check if related content appears together (via topic continuity within chunks)
        # Use a simple proxy: word repetition within chunks vs across chunks
        
        grouping_score = 0.0
        if num_chunks >= 2:
            chunk_texts = []
            current_text = []
            for i, t in enumerate(line_types):
                if t == 'blank':
                    if current_text:
                        chunk_texts.append(' '.join(current_text))
                        current_text = []
                else:
                    current_text.append(lines[i].strip().lower())
            if current_text:
                chunk_texts.append(' '.join(current_text))
            
            if len(chunk_texts) >= 2:
                # Compute within-chunk word overlap vs between-chunk
                chunk_word_sets = []
                for ct in chunk_texts:
                    words = set(re.findall(r'\b[a-z]{3,}\b', ct))
                    # Remove very common words
                    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                                'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some', 'them',
                                'than', 'its', 'over', 'also', 'that', 'this', 'with', 'from', 'they',
                                'will', 'would', 'there', 'their', 'what', 'about', 'which', 'when',
                                'make', 'like', 'more', 'other', 'into', 'could', 'your', 'each'}
                    words -= stopwords
                    chunk_word_sets.append(words)
                
                # Between-chunk distinctness: how different are chunks from each other?
                if len(chunk_word_sets) >= 2:
                    pairwise_overlaps = []
                    for i in range(len(chunk_word_sets)):
                        for j in range(i+1, len(chunk_word_sets)):
                            if chunk_word_sets[i] and chunk_word_sets[j]:
                                intersection = chunk_word_sets[i] & chunk_word_sets[j]
                                union = chunk_word_sets[i] | chunk_word_sets[j]
                                jaccard = len(intersection) / max(len(union), 1)
                                pairwise_overlaps.append(jaccard)
                    
                    if pairwise_overlaps:
                        avg_overlap = sum(pairwise_overlaps) / len(pairwise_overlaps)
                        # Some overlap is good (coherent), but too much means no real grouping
                        # Ideal: 0.1-0.4
                        if 0.05 <= avg_overlap <= 0.4:
                            grouping_score = 3.0
                        elif avg_overlap < 0.05:
                            grouping_score = 1.5  # chunks too disconnected
                        else:
                            grouping_score = max(0, 3.0 - (avg_overlap - 0.4) * 5.0)
        
        # === COMBINE SCORES ===
        # Weight each component
        
        # For short responses (< 40 words), structural formatting matters less
        length_factor = min(total_words / 60.0, 1.0)  # ramps up to 1.0 at 60 words
        
        # Base content score: reward having actual content
        content_score = min(math.log(total_words + 1) * 1.2, 5.0)
        
        # Weighted combination
        structural_component = (
            rhythm_score * 0.20 +
            scanability_score * 0.20 +
            chunking_score * 0.20 +
            hierarchy_score * 0.15 +
            variance_score * 0.10 +
            grouping_score * 0.15
        )
        
        # Blend: for short responses, lean more on content; for long ones, lean on structure
        final_score = content_score * (1.0 - length_factor * 0.5) + structural_component * length_factor - wall_penalty
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Ultra-safe fallback
            if response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except Exception:
            return 0.0