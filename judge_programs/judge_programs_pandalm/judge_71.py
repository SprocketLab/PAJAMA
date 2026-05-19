def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality.
    
    This variant uses a DIFFERENT approach: analyzing the response as a hierarchy
    of visual/structural "zones" and computing a composite score based on:
    1. Visual rhythm analysis (alternation between different structural elements)
    2. Information density distribution (evenness across segments)
    3. Structural diversity index (Shannon-like diversity of element types)
    4. Nesting/indentation depth analysis
    5. Sentence-to-structure ratio (how well content is scaffolded)
    6. Repetition penalty via compression ratio estimation
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
        
        # ---- Feature 1: Structural Element Classification ----
        # Classify each line into a structural "type"
        lines = response.split('\n')
        line_types = []
        
        for line in lines:
            stripped = line.strip()
            if len(stripped) == 0:
                line_types.append('blank')
            elif re.match(r'^#{1,6}\s+', stripped):
                line_types.append('md_header')
            elif re.match(r'^[A-Z][A-Za-z\s]{2,50}:?\s*$', stripped) and len(stripped) < 60:
                line_types.append('title_line')
            elif re.match(r'^(\d+[\.\)]\s|[a-z][\.\)]\s)', stripped):
                line_types.append('numbered_item')
            elif re.match(r'^[\-\*\•\►\▸\◦\‣\⁃]\s', stripped):
                line_types.append('bullet_item')
            elif re.match(r'^\*\*[^*]+\*\*', stripped) or re.match(r'^__[^_]+__', stripped):
                line_types.append('bold_start')
            elif len(stripped) < 15 and stripped.endswith(':'):
                line_types.append('label')
            elif re.match(r'^[\-=_\*]{3,}$', stripped):
                line_types.append('separator')
            elif re.match(r'^\|.*\|$', stripped):
                line_types.append('table_row')
            elif re.match(r'^>', stripped):
                line_types.append('blockquote')
            elif re.match(r'^```', stripped):
                line_types.append('code_fence')
            else:
                line_types.append('prose')
        
        # Remove trailing blanks for analysis
        while line_types and line_types[-1] == 'blank':
            line_types.pop()
        while line_types and line_types[0] == 'blank':
            line_types.pop(0)
        
        if not line_types:
            return 1.0
        
        # ---- Feature 2: Structural Diversity Index (Shannon entropy of types) ----
        non_blank_types = [t for t in line_types if t != 'blank']
        if not non_blank_types:
            return 1.0
        
        type_counts = Counter(non_blank_types)
        total_typed = len(non_blank_types)
        
        diversity_entropy = 0.0
        for count in type_counts.values():
            p = count / total_typed
            if p > 0:
                diversity_entropy -= p * math.log2(p)
        
        # Normalize: max entropy for the number of distinct types
        n_distinct = len(type_counts)
        max_entropy = math.log2(n_distinct) if n_distinct > 1 else 1.0
        diversity_score = diversity_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Bonus for having multiple structural types (not just prose)
        has_structural = any(t in type_counts for t in [
            'md_header', 'title_line', 'numbered_item', 'bullet_item',
            'bold_start', 'label', 'table_row', 'blockquote', 'code_fence', 'separator'
        ])
        structural_type_count = sum(1 for t in type_counts if t != 'prose' and t != 'blank')
        
        # ---- Feature 3: Visual Rhythm (transitions between different types) ----
        transitions = 0
        for i in range(1, len(line_types)):
            if line_types[i] != line_types[i-1]:
                transitions += 1
        
        max_transitions = len(line_types) - 1 if len(line_types) > 1 else 1
        rhythm_score = transitions / max_transitions if max_transitions > 0 else 0.0
        
        # ---- Feature 4: Information Density Distribution ----
        # Split into segments (by blank lines) and measure word count evenness
        segments = []
        current_seg = []
        for line in lines:
            if line.strip() == '':
                if current_seg:
                    segments.append(' '.join(current_seg))
                    current_seg = []
            else:
                current_seg.append(line.strip())
        if current_seg:
            segments.append(' '.join(current_seg))
        
        n_segments = len(segments)
        
        if n_segments > 1:
            seg_lengths = [len(s.split()) for s in segments]
            mean_len = sum(seg_lengths) / len(seg_lengths) if seg_lengths else 1
            if mean_len > 0:
                cv = (sum((l - mean_len)**2 for l in seg_lengths) / len(seg_lengths))**0.5 / mean_len
                # Lower CV = more even distribution = better
                evenness_score = max(0, 1.0 - cv * 0.5)
            else:
                evenness_score = 0.5
        else:
            # Single block - check if it's short enough to not need segmentation
            word_count = len(response.split())
            if word_count < 40:
                evenness_score = 0.7  # Short response, single block is fine
            else:
                evenness_score = max(0, 0.5 - (word_count - 40) * 0.005)
        
        # ---- Feature 5: Compression Ratio (repetition penalty) ----
        # Estimate compressibility by looking at repeated n-grams
        words = response.lower().split()
        total_words = len(words)
        
        if total_words > 5:
            # Trigram repetition
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / len(trigrams) if trigrams else 0
            compression_penalty = min(1.0, repetition_ratio * 3)
        else:
            compression_penalty = 0.0
        
        # ---- Feature 6: Sentence-to-Structure Ratio ----
        # Count sentences
        sentence_endings = len(re.findall(r'[.!?](?:\s|$|")', response))
        sentence_endings = max(sentence_endings, 1)
        
        # Count structural markers
        structural_markers = sum(1 for t in line_types if t in [
            'md_header', 'title_line', 'numbered_item', 'bullet_item',
            'bold_start', 'label', 'separator', 'table_row'
        ])
        
        # Good ratio: some structure per group of sentences
        if sentence_endings > 3:
            # Longer responses benefit from structure
            structure_ratio = min(1.0, structural_markers / (sentence_endings * 0.3))
        else:
            # Short responses don't need much structure
            structure_ratio = 0.5 + min(0.5, structural_markers * 0.25)
        
        # ---- Feature 7: Paragraph Quality ----
        # Analyze paragraph count and sizes relative to total length
        paragraphs = [s for s in re.split(r'\n\s*\n', response) if s.strip()]
        n_paragraphs = len(paragraphs)
        
        if total_words > 80:
            # Longer text should have multiple paragraphs
            para_score = min(1.0, n_paragraphs / max(1, total_words / 60))
        elif total_words > 30:
            para_score = min(1.0, 0.5 + n_paragraphs * 0.2)
        else:
            para_score = 0.7  # Short text, paragraphs less important
        
        # ---- Feature 8: Inline formatting detection ----
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        code_inline = len(re.findall(r'`[^`]+`', response))
        inline_formatting = min(1.0, (bold_count + italic_count + code_inline) * 0.15)
        
        # ---- Feature 9: Response length appropriateness ----
        query_words = len(query.split()) if query else 5
        # Check if response length is reasonable
        if total_words < 3:
            length_score = 0.1
        elif total_words < 10:
            length_score = 0.3
        elif total_words < 20:
            length_score = 0.5
        elif total_words > 500:
            # Very long responses need excellent structure
            length_score = 0.6
        else:
            length_score = 0.8
        
        # ---- Feature 10: Leading sentence / topic sentence quality ----
        # Check if first sentence is substantive and relates to query
        first_line = lines[0].strip() if lines else ""
        first_words = first_line.split()
        
        topic_sentence_score = 0.5
        if len(first_words) > 5:
            topic_sentence_score = 0.7
            # Check for query term overlap in first sentence
            query_terms = set(query.lower().split())
            first_terms = set(first_line.lower().split())
            overlap = len(query_terms & first_terms)
            if overlap > 0:
                topic_sentence_score = min(1.0, 0.7 + overlap * 0.05)
        
        # ---- Composite Scoring ----
        # Weight the features differently based on response length
        if total_words > 100:
            # Longer responses: structure matters more
            score = (
                diversity_score * 12 +
                rhythm_score * 8 +
                evenness_score * 10 +
                (1 - compression_penalty) * 15 +
                structure_ratio * 12 +
                para_score * 10 +
                inline_formatting * 5 +
                length_score * 8 +
                topic_sentence_score * 5 +
                (min(1.0, structural_type_count * 0.3) * 10) +
                (min(1.0, n_segments / 3) * 5)
            )
        elif total_words > 30:
            # Medium responses
            score = (
                diversity_score * 8 +
                rhythm_score * 5 +
                evenness_score * 10 +
                (1 - compression_penalty) * 18 +
                structure_ratio * 8 +
                para_score * 8 +
                inline_formatting * 4 +
                length_score * 12 +
                topic_sentence_score * 8 +
                (min(1.0, structural_type_count * 0.3) * 7) +
                (min(1.0, n_segments / 2) * 5) +
                min(5, total_words * 0.05)
            )
        else:
            # Short responses: content matters more than structure
            score = (
                (1 - compression_penalty) * 20 +
                length_score * 20 +
                topic_sentence_score * 15 +
                evenness_score * 8 +
                diversity_score * 5 +
                inline_formatting * 3 +
                min(10, total_words * 0.3) +
                (min(1.0, structural_type_count * 0.3) * 5)
            )
        
        # Penalty for wall-of-text: single paragraph with many words
        if n_paragraphs == 1 and total_words > 100:
            wall_penalty = min(15, (total_words - 100) * 0.1)
            score -= wall_penalty
        
        # Penalty for excessive blank lines (poor whitespace usage)
        blank_count = line_types.count('blank')
        non_blank_count = len(line_types) - blank_count
        if non_blank_count > 0 and blank_count > non_blank_count * 2:
            score -= 5
        
        # Bonus for having headers in longer content
        if total_words > 60 and ('md_header' in type_counts or 'title_line' in type_counts):
            score += 5
        
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            if response and len(response.strip()) > 0:
                words = len(response.split())
                if words < 3:
                    return 5.0
                return min(50.0, words * 0.5)
            return 0.0
        except Exception:
            return 0.0