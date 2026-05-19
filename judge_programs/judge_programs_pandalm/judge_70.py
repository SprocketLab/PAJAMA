def judging_function(query, response):
    """
    Evaluate structural organization and formatting quality using a 
    hierarchical depth and visual rhythm analysis approach.
    
    This variant focuses on:
    1. Visual rhythm (alternation between different structural elements)
    2. Hierarchical depth (nesting levels of organization)
    3. Information density distribution across sections
    4. Structural element diversity (Shannon diversity of element types)
    5. Reading flow score based on progressive disclosure patterns
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
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if len(non_empty_lines) == 0:
            return 0.0
        
        # === 1. Classify each line into a structural element type ===
        def classify_line(line):
            stripped = line.strip()
            if not stripped:
                return 'blank'
            # Header patterns
            if re.match(r'^#{1,6}\s+', stripped):
                return 'header'
            if re.match(r'^[A-Z][^.!?]*:$', stripped) and len(stripped) < 80:
                return 'header'
            if re.match(r'^[A-Z][A-Z\s]{3,}$', stripped):
                return 'header'
            if re.match(r'^\*\*[^*]+\*\*$', stripped):
                return 'header'
            # Numbered list
            if re.match(r'^\d+[\.\)]\s+', stripped):
                return 'numbered'
            # Bullet list
            if re.match(r'^[\-\*\•\▪\▸\►\‣\⁃]\s+', stripped):
                return 'bullet'
            # Indented/nested content
            if re.match(r'^(\s{4,}|\t+)', line) and len(stripped) > 0:
                return 'nested'
            # Code block markers
            if stripped.startswith('```'):
                return 'code_marker'
            # Short line (could be label or caption)
            if len(stripped) < 30 and stripped.endswith(':'):
                return 'label'
            # Regular paragraph text
            return 'paragraph'
        
        line_types = [classify_line(l) for l in lines]
        non_blank_types = [t for t in line_types if t != 'blank']
        
        # === 2. Structural Element Diversity (Shannon diversity index) ===
        type_counts = Counter(non_blank_types)
        total_elements = len(non_blank_types)
        
        diversity_score = 0.0
        if total_elements > 0:
            proportions = [c / total_elements for c in type_counts.values()]
            shannon = -sum(p * math.log2(p) for p in proportions if p > 0)
            # Max possible with ~7 types is log2(7) ≈ 2.81
            # Normalize to 0-1 range
            max_shannon = math.log2(min(len(type_counts), 7)) if len(type_counts) > 1 else 1
            diversity_score = min(shannon / max(max_shannon, 0.001), 1.0)
        
        # Bonus for having multiple distinct structural types
        distinct_types = len(set(non_blank_types) - {'paragraph'})
        type_variety_bonus = min(distinct_types * 0.15, 0.6)
        
        # === 3. Visual Rhythm Analysis ===
        # Measure how structural types alternate (not monotonous)
        transitions = 0
        rhythm_changes = []
        for i in range(1, len(line_types)):
            if line_types[i] != line_types[i-1]:
                transitions += 1
                rhythm_changes.append(i)
        
        max_transitions = max(len(line_types) - 1, 1)
        rhythm_score = min(transitions / max_transitions, 1.0) if max_transitions > 0 else 0
        
        # Penalize if ALL lines are the same type (monotonous)
        if len(set(non_blank_types)) == 1 and total_elements > 3:
            rhythm_score *= 0.3
        
        # === 4. Hierarchical Depth Analysis ===
        max_depth = 0
        current_depth = 0
        depth_changes = 0
        
        for lt in non_blank_types:
            if lt == 'header':
                current_depth = 1
            elif lt in ('numbered', 'bullet'):
                current_depth = 2
            elif lt == 'nested':
                current_depth = 3
            elif lt == 'paragraph':
                current_depth = max(current_depth, 1)
            
            if current_depth > max_depth:
                depth_changes += 1
            max_depth = max(max_depth, current_depth)
        
        hierarchy_score = min(max_depth / 3.0, 1.0)
        
        # === 5. Information Density Distribution ===
        # Split response into chunks and measure word count variance
        # Good organization = relatively even distribution, not front/back heavy
        words_per_line = [len(l.split()) for l in non_empty_lines]
        
        if len(words_per_line) > 1:
            mean_wpl = sum(words_per_line) / len(words_per_line)
            variance = sum((w - mean_wpl) ** 2 for w in words_per_line) / len(words_per_line)
            std_wpl = math.sqrt(variance)
            cv = std_wpl / max(mean_wpl, 0.001)  # coefficient of variation
            
            # Some variation is good (headers shorter than paragraphs)
            # But extreme variation is bad
            if 0.3 <= cv <= 1.5:
                density_score = 1.0
            elif cv < 0.3:
                density_score = 0.5 + cv  # too uniform
            else:
                density_score = max(0.2, 1.0 - (cv - 1.5) * 0.3)
        else:
            density_score = 0.3
        
        # === 6. Blank Line / Whitespace Usage ===
        blank_count = sum(1 for t in line_types if t == 'blank')
        total_lines = len(lines)
        blank_ratio = blank_count / max(total_lines, 1)
        
        # Good whitespace usage: some blank lines for separation but not excessive
        if total_lines > 3:
            if 0.1 <= blank_ratio <= 0.4:
                whitespace_score = 1.0
            elif blank_ratio < 0.1:
                whitespace_score = 0.4 + blank_ratio * 6
            else:
                whitespace_score = max(0.2, 1.0 - (blank_ratio - 0.4) * 2)
        else:
            whitespace_score = 0.3 if blank_count == 0 else 0.6
        
        # === 7. Progressive Disclosure Pattern ===
        # Check if response follows intro -> body -> conclusion pattern
        progressive_score = 0.0
        
        if total_elements >= 3:
            # First element should be paragraph (intro)
            if non_blank_types[0] == 'paragraph':
                progressive_score += 0.3
            
            # Middle should have structured elements
            middle_types = non_blank_types[1:-1] if len(non_blank_types) > 2 else []
            structured_middle = sum(1 for t in middle_types if t in ('numbered', 'bullet', 'header', 'nested', 'label'))
            if middle_types:
                progressive_score += 0.4 * min(structured_middle / max(len(middle_types), 1), 1.0)
            
            # Last element being paragraph (conclusion) is a bonus
            if non_blank_types[-1] == 'paragraph' and len(non_blank_types) > 3:
                progressive_score += 0.3
        elif total_elements >= 1:
            progressive_score = 0.2
        
        # === 8. Sentence-level organization within paragraphs ===
        # Check for topic sentences and logical flow
        paragraph_blocks = []
        current_block = []
        for line in lines:
            if line.strip():
                current_block.append(line.strip())
            else:
                if current_block:
                    paragraph_blocks.append(' '.join(current_block))
                    current_block = []
        if current_block:
            paragraph_blocks.append(' '.join(current_block))
        
        sentence_org_score = 0.0
        if paragraph_blocks:
            good_paragraphs = 0
            for block in paragraph_blocks:
                sentences = re.split(r'[.!?]+\s+', block)
                sentences = [s for s in sentences if len(s.strip()) > 5]
                
                if len(sentences) >= 2:
                    # First sentence should be shorter or comparable (topic sentence)
                    first_len = len(sentences[0].split())
                    avg_rest = sum(len(s.split()) for s in sentences[1:]) / max(len(sentences) - 1, 1)
                    if first_len <= avg_rest * 1.5:
                        good_paragraphs += 1
                    else:
                        good_paragraphs += 0.5
                elif len(sentences) == 1:
                    good_paragraphs += 0.6
            
            sentence_org_score = good_paragraphs / max(len(paragraph_blocks), 1)
        
        # === 9. Wall-of-text penalty ===
        wall_penalty = 0.0
        total_words = len(response.split())
        
        # Long response with no structure
        if total_words > 50 and len(set(non_blank_types)) == 1 and non_blank_types[0] == 'paragraph':
            if len(paragraph_blocks) <= 1:
                wall_penalty = min(0.3, total_words / 500.0)
        
        # Very long single paragraph
        for block in paragraph_blocks:
            block_words = len(block.split())
            if block_words > 100 and len(paragraph_blocks) <= 2:
                wall_penalty = max(wall_penalty, 0.2)
        
        # === 10. Repetition penalty ===
        words = response.lower().split()
        if len(words) > 5:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            most_common_freq = bigram_counts.most_common(1)[0][1] if bigram_counts else 0
            repetition_ratio = most_common_freq / max(len(bigrams), 1)
            if repetition_ratio > 0.1:
                wall_penalty += min(0.3, repetition_ratio * 2)
        
        # === 11. Response length appropriateness ===
        query_words = len(query.split()) if query else 5
        response_words = total_words
        
        length_score = 0.5
        if response_words >= 20:
            length_score = 0.7
        if response_words >= 40:
            length_score = 0.85
        if response_words >= 80:
            length_score = 1.0
        if response_words < 10:
            length_score = 0.2
        # Very short responses can still be fine if query is simple
        if response_words < 15 and query_words < 10:
            length_score = max(length_score, 0.4)
        
        # === Combine all scores ===
        # Weights emphasizing structural diversity and visual rhythm
        weights = {
            'diversity': 1.5,
            'type_variety': 1.2,
            'rhythm': 1.0,
            'hierarchy': 1.3,
            'density': 0.7,
            'whitespace': 0.8,
            'progressive': 0.9,
            'sentence_org': 0.8,
            'length': 1.0,
        }
        
        raw_score = (
            weights['diversity'] * diversity_score +
            weights['type_variety'] * type_variety_bonus +
            weights['rhythm'] * rhythm_score +
            weights['hierarchy'] * hierarchy_score +
            weights['density'] * density_score +
            weights['whitespace'] * whitespace_score +
            weights['progressive'] * progressive_score +
            weights['sentence_org'] * sentence_org_score +
            weights['length'] * length_score
        )
        
        total_weight = sum(weights.values())
        normalized = (raw_score / total_weight) * 10.0  # Scale to 0-10
        
        # Apply penalties
        final_score = max(0.0, normalized - wall_penalty * 10.0)
        
        # Ensure range 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        # Small boost for responses that are substantive but simply well-written paragraphs
        # (not everything needs heavy formatting)
        if 20 <= total_words <= 150 and len(paragraph_blocks) >= 1 and final_score < 4:
            sentences_total = len(re.split(r'[.!?]+', response))
            if sentences_total >= 2:
                final_score = max(final_score, 3.5)
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Fallback: basic length-based score
            if response and len(response.strip()) > 0:
                return min(3.0, len(response.strip().split()) / 20.0)
            return 0.0
        except Exception:
            return 0.0