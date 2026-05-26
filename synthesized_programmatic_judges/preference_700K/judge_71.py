def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses a DIFFERENT approach: analyzing the response as a hierarchical
    document structure, measuring visual rhythm patterns, computing a "scanability" 
    score based on how easy it is to extract information visually, and evaluating
    the diversity of structural elements used.
    
    Key differentiators from other variants:
    - Uses "visual rhythm" analysis (alternating patterns of short/long lines)
    - Computes "information density gradient" across the response
    - Measures structural element diversity (Shannon diversity of element types)
    - Analyzes sentence length variance as a readability proxy
    - Uses a "chunk coherence" model based on semantic grouping signals
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        score = 0.0
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # ============================================================
        # FEATURE 1: Structural Element Diversity (Shannon Diversity Index)
        # ============================================================
        # Classify each non-empty line into a structural type
        element_types = []
        for line in non_empty_lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Classify line type
            if re.match(r'^#{1,6}\s', stripped):
                element_types.append('markdown_header')
            elif re.match(r'^[A-Z][A-Za-z\s]{2,50}:?\s*$', stripped) and len(stripped) < 60:
                element_types.append('title_line')
            elif re.match(r'^(\d+[\.\)]\s|[a-z][\.\)]\s)', stripped):
                element_types.append('numbered_item')
            elif re.match(r'^[\-\*\•\▪\▸\►\→\➤\‣]\s', stripped):
                element_types.append('bullet_item')
            elif re.match(r'^```', stripped):
                element_types.append('code_fence')
            elif re.match(r'^\|.*\|', stripped):
                element_types.append('table_row')
            elif re.match(r'^>', stripped):
                element_types.append('blockquote')
            elif re.match(r'^\*\*[^*]+\*\*', stripped) or re.match(r'^__[^_]+__', stripped):
                element_types.append('bold_start')
            elif len(stripped) < 80 and stripped.endswith(':'):
                element_types.append('label_line')
            elif len(stripped) > 100:
                element_types.append('long_paragraph')
            elif len(stripped) > 40:
                element_types.append('medium_paragraph')
            else:
                element_types.append('short_text')
        
        # Shannon diversity of element types
        if element_types:
            type_counts = Counter(element_types)
            total_elements = len(element_types)
            num_distinct_types = len(type_counts)
            
            shannon_diversity = 0.0
            for count in type_counts.values():
                p = count / total_elements
                if p > 0:
                    shannon_diversity -= p * math.log2(p)
            
            # Normalize: max possible is log2(num_types)
            max_diversity = math.log2(max(num_distinct_types, 1)) if num_distinct_types > 1 else 0
            normalized_diversity = shannon_diversity / max(max_diversity, 0.001) if max_diversity > 0 else 0
            
            # Reward having multiple structural types
            type_bonus = min(num_distinct_types / 5.0, 1.0) * 8.0
            diversity_score = normalized_diversity * 7.0 + type_bonus
            score += diversity_score
        
        # ============================================================
        # FEATURE 2: Visual Rhythm Analysis
        # ============================================================
        # Measure the pattern of line lengths - good formatting creates rhythm
        if len(non_empty_lines) >= 3:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            
            # Compute "rhythm" as the autocorrelation-like measure of length changes
            # Good structure alternates between short (headers/bullets) and long (content)
            transitions = []
            for i in range(1, len(line_lengths)):
                if line_lengths[i-1] > 0:
                    ratio = line_lengths[i] / max(line_lengths[i-1], 1)
                    transitions.append(ratio)
            
            if transitions:
                # Count significant transitions (ratio > 2 or < 0.5)
                significant_transitions = sum(1 for t in transitions if t > 2.0 or t < 0.5)
                rhythm_ratio = significant_transitions / len(transitions)
                
                # Some rhythm is good (indicates structure), but not too chaotic
                if 0.15 <= rhythm_ratio <= 0.7:
                    rhythm_score = 6.0 * (1.0 - abs(rhythm_ratio - 0.4) / 0.4)
                elif rhythm_ratio > 0.7:
                    rhythm_score = 2.0
                else:
                    rhythm_score = rhythm_ratio * 10.0
                
                score += max(rhythm_score, 0)
        
        # ============================================================
        # FEATURE 3: Scanability Score
        # ============================================================
        # How easy is it to scan and find information?
        scanability = 0.0
        
        # a) Presence of visual anchors (bold, italic, inline code, caps words)
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response)) + len(re.findall(r'__[^_]+__', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        inline_code = len(re.findall(r'`[^`]+`', response))
        visual_anchors = bold_count + italic_count + inline_code
        
        # Reward visual anchors proportionally to response length
        expected_anchors = total_chars / 300.0  # roughly 1 per 300 chars
        if expected_anchors > 0:
            anchor_ratio = min(visual_anchors / max(expected_anchors, 0.5), 3.0)
            scanability += anchor_ratio * 3.0
        
        # b) Presence of enumerative structures
        numbered_items = len(re.findall(r'(?m)^\s*\d+[\.\)]\s', response))
        bullet_items = len(re.findall(r'(?m)^\s*[\-\*\•\▪]\s', response))
        list_items = numbered_items + bullet_items
        
        if list_items >= 2:
            scanability += min(list_items * 1.5, 8.0)
        
        # c) Headers or section markers
        headers = len(re.findall(r'(?m)^#{1,6}\s', response))
        pseudo_headers = len(re.findall(r'(?m)^[A-Z][A-Za-z\s]{2,40}:\s*$', response))
        bold_headers = len(re.findall(r'(?m)^\*\*[^*]{3,50}\*\*\s*$', response))
        total_headers = headers + pseudo_headers + bold_headers
        
        if total_headers >= 1:
            scanability += min(total_headers * 2.5, 8.0)
        
        score += min(scanability, 18.0)
        
        # ============================================================
        # FEATURE 4: Information Density Gradient
        # ============================================================
        # Split response into quartiles and measure if structure is consistent
        if total_chars > 200:
            quartile_size = total_chars // 4
            quartiles = [
                response[:quartile_size],
                response[quartile_size:2*quartile_size],
                response[2*quartile_size:3*quartile_size],
                response[3*quartile_size:]
            ]
            
            # Measure structural density per quartile
            quartile_densities = []
            for q in quartiles:
                if not q.strip():
                    quartile_densities.append(0)
                    continue
                q_lines = q.split('\n')
                q_non_empty = [l for l in q_lines if l.strip()]
                q_structural = sum(1 for l in q_non_empty if 
                    re.match(r'^\s*[\-\*\•\d+[\.\)]]\s', l.strip()) or
                    re.match(r'^#{1,6}\s', l.strip()) or
                    len(l.strip()) < 60 and l.strip().endswith(':'))
                density = q_structural / max(len(q_non_empty), 1)
                quartile_densities.append(density)
            
            # Reward consistent structural density across quartiles
            if any(d > 0 for d in quartile_densities):
                non_zero = [d for d in quartile_densities if d > 0]
                if len(non_zero) >= 2:
                    mean_d = sum(non_zero) / len(non_zero)
                    variance = sum((d - mean_d)**2 for d in non_zero) / len(non_zero)
                    consistency = 1.0 / (1.0 + variance * 10)
                    spread = len(non_zero) / 4.0  # how many quartiles have structure
                    score += consistency * spread * 6.0
        
        # ============================================================
        # FEATURE 5: Sentence Length Variance (readability proxy)
        # ============================================================
        # Extract sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len)**2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            
            # Moderate variance is good (indicates varied sentence structure)
            # Too low = monotonous, too high = disorganized
            cv = std_dev / max(mean_len, 1)  # coefficient of variation
            
            if 0.3 <= cv <= 1.0:
                var_score = 5.0 * (1.0 - abs(cv - 0.6) / 0.6)
            elif cv < 0.3:
                var_score = cv * 8.0
            else:
                var_score = max(5.0 - (cv - 1.0) * 3.0, 0)
            
            score += max(var_score, 0)
        
        # ============================================================
        # FEATURE 6: Chunk Coherence Model
        # ============================================================
        # Identify "chunks" separated by blank lines or structural markers
        chunks = re.split(r'\n\s*\n', response)
        chunks = [c.strip() for c in chunks if c.strip()]
        
        num_chunks = len(chunks)
        
        if num_chunks >= 2:
            # Reward appropriate chunking
            expected_chunks = max(total_chars / 400.0, 2.0)
            chunk_ratio = num_chunks / expected_chunks
            
            if 0.5 <= chunk_ratio <= 2.5:
                chunk_score = 6.0 * (1.0 - abs(chunk_ratio - 1.2) / 2.0)
            else:
                chunk_score = 1.0
            
            score += max(chunk_score, 0)
            
            # Check if chunks have consistent internal structure
            chunk_line_counts = [len(c.split('\n')) for c in chunks]
            if len(chunk_line_counts) >= 2:
                mean_lines = sum(chunk_line_counts) / len(chunk_line_counts)
                if mean_lines > 0:
                    line_cv = math.sqrt(sum((x - mean_lines)**2 for x in chunk_line_counts) / len(chunk_line_counts)) / mean_lines
                    # Moderate consistency is good
                    if line_cv < 1.5:
                        score += (1.5 - line_cv) * 2.0
        elif num_chunks == 1 and total_chars > 500:
            # Penalize wall of text
            score -= 5.0
        
        # ============================================================
        # FEATURE 7: Opening and Closing Structure
        # ============================================================
        # Check if response has a clear opening and structure signal
        first_line = non_empty_lines[0].strip() if non_empty_lines else ""
        
        # Opening with context/intro before diving into structure
        if len(non_empty_lines) >= 3:
            # Check if first line is shorter (intro) followed by longer content
            if len(first_line) < 150 and any(len(l.strip()) > len(first_line) for l in non_empty_lines[1:4]):
                score += 2.0
        
        # Check for concluding structure
        if len(non_empty_lines) >= 4:
            last_lines = non_empty_lines[-2:]
            for ll in last_lines:
                ll_lower = ll.strip().lower()
                if any(w in ll_lower for w in ['in conclusion', 'overall', 'in summary', 'to summarize', 'hope this', 'in short', 'ultimately', 'the key']):
                    score += 2.0
                    break
        
        # ============================================================
        # FEATURE 8: Code Block Formatting (for technical queries)
        # ============================================================
        code_blocks = re.findall(r'```[\s\S]*?```', response)
        if code_blocks:
            # Well-formatted code blocks
            score += min(len(code_blocks) * 3.0, 6.0)
            
            # Check if code blocks have language specifiers
            lang_specified = len(re.findall(r'```\w+', response))
            if lang_specified > 0:
                score += 2.0
        
        # ============================================================
        # FEATURE 9: Response Length Appropriateness
        # ============================================================
        # Very short responses rarely have good structure
        if total_chars < 100:
            score *= 0.5
        elif total_chars < 200:
            score *= 0.7
        elif total_chars > 300:
            # Longer responses have more opportunity for structure
            length_bonus = min((total_chars - 300) / 1000.0, 1.0) * 3.0
            score += length_bonus
        
        # ============================================================
        # FEATURE 10: Anti-wall-of-text penalty
        # ============================================================
        if total_chars > 300:
            # Calculate average paragraph length
            avg_chunk_len = total_chars / max(num_chunks, 1)
            if avg_chunk_len > 600:
                # Heavy penalty for very long unbroken blocks
                penalty = min((avg_chunk_len - 600) / 400.0, 1.0) * 8.0
                score -= penalty
            
            # Check line break frequency
            newline_ratio = response.count('\n') / (total_chars / 100.0)
            if newline_ratio < 0.5 and total_chars > 400:
                score -= 3.0
        
        # ============================================================
        # FEATURE 11: Inline Structure Signals
        # ============================================================
        # Colons used as label-value separators
        colon_labels = len(re.findall(r'(?m)^[A-Za-z][A-Za-z\s]{1,30}:\s\S', response))
        if colon_labels >= 2:
            score += min(colon_labels * 1.0, 4.0)
        
        # Parenthetical clarifications (shows organized thinking)
        parens = len(re.findall(r'\([^)]{5,80}\)', response))
        if parens >= 1:
            score += min(parens * 0.5, 2.0)
        
        # Em-dashes for asides
        em_dashes = response.count('—') + response.count(' -- ')
        if em_dashes >= 1:
            score += min(em_dashes * 0.5, 1.5)
        
        # Normalize final score to 0-10 range
        # Theoretical max is around 70-80, typical good response ~30-50
        final_score = max(0.0, min(score / 6.5, 10.0))
        
        return round(final_score, 3)
        
    except Exception:
        return 2.0