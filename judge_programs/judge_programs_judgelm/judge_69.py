def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical structure analysis approach based on information density,
    visual segmentation patterns, and content coherence zones.
    
    This variant focuses on:
    - Visual segmentation (how text is broken into visual chunks)
    - Information density distribution across the response
    - Structural hierarchy depth analysis
    - Repetition/degeneration detection
    - Proportionality of structure to content complexity
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        score = 0.0
        
        # === 1. VISUAL SEGMENTATION ANALYSIS ===
        # Analyze how the response is broken into visual segments
        # (different from paragraph counting - we look at segment variety and purpose)
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        all_lines = lines
        
        total_chars = len(response)
        total_words = len(response.split())
        
        # Very short responses get baseline treatment
        if total_words < 3:
            return 1.0
        
        # Count distinct "visual zones" - contiguous blocks of non-empty lines
        zones = []
        current_zone = []
        for line in all_lines:
            if line.strip():
                current_zone.append(line.strip())
            else:
                if current_zone:
                    zones.append(current_zone)
                    current_zone = []
        if current_zone:
            zones.append(current_zone)
        
        num_zones = len(zones)
        
        # === 2. LINE TYPE CLASSIFICATION ===
        # Classify each non-empty line by its structural role
        import re
        
        line_types = []
        for line in non_empty_lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Heading-like: short, possibly with markers
            is_heading = False
            if len(stripped.split()) <= 8 and not stripped.endswith(','):
                if stripped.endswith(':') or stripped.startswith('#') or stripped.isupper():
                    is_heading = True
                if re.match(r'^[A-Z][^.!?]*$', stripped) and len(stripped.split()) <= 5:
                    is_heading = True
            
            # List item
            is_list = bool(re.match(r'^(\d+[\.\)]\s|[-•*]\s|[a-zA-Z][\.\)]\s)', stripped))
            
            # Label-value pair (e.g., "String: Kisanji")
            is_label_value = bool(re.match(r'^[A-Za-z\s]{1,30}:\s*.+', stripped)) and len(stripped.split()) <= 12
            
            # Code/markup
            is_markup = bool(re.match(r'^<[^>]+>', stripped)) or bool(re.match(r'^```', stripped))
            
            # Regular prose
            is_prose = not (is_heading or is_list or is_label_value or is_markup)
            
            if is_heading:
                line_types.append('heading')
            elif is_list:
                line_types.append('list')
            elif is_label_value:
                line_types.append('label')
            elif is_markup:
                line_types.append('markup')
            else:
                line_types.append('prose')
        
        # === 3. STRUCTURAL DIVERSITY SCORE ===
        # More diverse line types = better structure (up to a point)
        from collections import Counter
        type_counts = Counter(line_types)
        unique_types = len(type_counts)
        
        structural_types_used = sum(1 for t in ['heading', 'list', 'label'] if t in type_counts)
        
        # Reward for using structural elements proportionally
        structural_diversity_score = 0.0
        if structural_types_used >= 1:
            structural_diversity_score += 1.5
        if structural_types_used >= 2:
            structural_diversity_score += 1.0
        
        # === 4. INFORMATION DENSITY DISTRIBUTION ===
        # Analyze how evenly information is distributed across zones
        # Wall-of-text = one zone with everything; well-structured = distributed
        
        density_score = 0.0
        if num_zones == 1 and total_words > 50:
            # Single block - check if it's a wall of text
            zone_words = total_words
            avg_line_len = total_words / max(len(non_empty_lines), 1)
            if avg_line_len > 30:
                density_score -= 1.0  # Wall of text penalty
            elif avg_line_len < 20:
                density_score += 0.5
        elif num_zones >= 2:
            zone_sizes = [sum(len(line.split()) for line in z) for z in zones]
            if zone_sizes:
                avg_zone = sum(zone_sizes) / len(zone_sizes)
                if avg_zone > 0:
                    # Coefficient of variation - lower is more even distribution
                    import math
                    variance = sum((s - avg_zone) ** 2 for s in zone_sizes) / len(zone_sizes)
                    cv = math.sqrt(variance) / avg_zone if avg_zone > 0 else 0
                    if cv < 0.5:
                        density_score += 1.5  # Very even distribution
                    elif cv < 1.0:
                        density_score += 1.0
                    else:
                        density_score += 0.5
                
                # Reward appropriate number of zones relative to content length
                ideal_zones = max(1, min(8, total_words // 30))
                zone_ratio = num_zones / ideal_zones if ideal_zones > 0 else 1
                if 0.5 <= zone_ratio <= 2.0:
                    density_score += 1.0
        
        # === 5. DEGENERATION / REPETITION DETECTION ===
        # Detect repetitive patterns that indicate poor quality
        
        repetition_penalty = 0.0
        
        # Check for repeated lines
        if len(non_empty_lines) > 1:
            line_set = set(l.strip().lower() for l in non_empty_lines)
            repetition_ratio = len(line_set) / len(non_empty_lines)
            if repetition_ratio < 0.5:
                repetition_penalty += 3.0
            elif repetition_ratio < 0.7:
                repetition_penalty += 1.5
        
        # Check for repeated phrases (n-gram repetition)
        words = response.lower().split()
        if len(words) >= 10:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                repeat_ratio = repeated_trigrams / max(len(trigram_counts), 1)
                if repeat_ratio > 0.15:
                    repetition_penalty += 2.0
                elif repeat_ratio > 0.08:
                    repetition_penalty += 1.0
        
        # Check for "Output:" spam or similar repeated prefixes
        prefix_pattern = Counter()
        for line in non_empty_lines:
            first_word = line.strip().split(':')[0] if ':' in line else ''
            if first_word:
                prefix_pattern[first_word.lower().strip()] += 1
        
        for prefix, count in prefix_pattern.items():
            if count > 3 and count / max(len(non_empty_lines), 1) > 0.4:
                repetition_penalty += 1.5
                break
        
        # === 6. CONTENT COHERENCE AND COMPLETENESS ===
        
        coherence_score = 0.0
        
        # Check if response appears truncated
        if response.rstrip()[-1:] not in '.!?"\')>}]' and total_words > 20:
            # Might be truncated - small penalty but not huge
            coherence_score -= 0.5
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', response)
        real_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        
        if len(real_sentences) >= 2:
            coherence_score += 1.0
        
        # Appropriate response length relative to query complexity
        query_words = len(query.split()) if query else 5
        length_ratio = total_words / max(query_words, 1)
        
        if 1.0 <= length_ratio <= 20.0:
            coherence_score += 1.0
        elif length_ratio > 20.0:
            coherence_score += 0.5  # Might be too verbose
        elif length_ratio < 0.3:
            coherence_score -= 1.0  # Too terse
        
        # === 7. SEMANTIC NOISE DETECTION ===
        # Check for irrelevant structural artifacts
        
        noise_penalty = 0.0
        
        # Random code when not asked for
        query_lower = query.lower() if query else ""
        code_indicators = ['import ', 'def ', 'class ', 'return ', 'if __name__']
        code_lines = sum(1 for l in non_empty_lines if any(ci in l for ci in code_indicators))
        
        is_code_query = any(kw in query_lower for kw in ['code', 'program', 'function', 'html', 'script', 'python', 'java'])
        
        if code_lines > 3 and not is_code_query:
            noise_penalty += 2.0
        
        # HTML tags when not appropriate
        html_tags = len(re.findall(r'<[^>]+>', response))
        is_html_query = any(kw in query_lower for kw in ['html', 'tag', 'web', 'markup'])
        
        if html_tags > 3 and not is_html_query:
            noise_penalty += 1.5
        
        # === 8. PROPORTIONALITY ANALYSIS ===
        # Is the structure proportional to the content needs?
        
        proportionality_score = 0.0
        
        # For short responses (< 30 words), heavy structure is unnecessary
        if total_words < 15:
            # Short response - structure matters less, but clarity matters
            if len(non_empty_lines) <= 3:
                proportionality_score += 1.0
            # Check it's at least a complete thought
            if any(c in response for c in '.!?'):
                proportionality_score += 0.5
        elif total_words < 50:
            # Medium response - some structure is nice
            if 1 <= num_zones <= 4:
                proportionality_score += 1.0
            if structural_types_used >= 1:
                proportionality_score += 0.5
        else:
            # Long response - structure is important
            if num_zones >= 2:
                proportionality_score += 1.5
            if structural_types_used >= 1:
                proportionality_score += 1.0
            # Penalize long wall of text
            if num_zones == 1 and total_words > 80:
                proportionality_score -= 1.5
        
        # === 9. SENTENCE LENGTH VARIANCE ===
        # Good writing has varied sentence lengths
        
        sentence_variance_score = 0.0
        if len(real_sentences) >= 3:
            sent_lengths = [len(s.split()) for s in real_sentences]
            avg_len = sum(sent_lengths) / len(sent_lengths)
            if avg_len > 0:
                import math
                variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / avg_len
                if 0.2 <= cv <= 0.8:
                    sentence_variance_score += 1.0  # Good variety
                elif cv > 0.8:
                    sentence_variance_score += 0.3  # Too varied
                else:
                    sentence_variance_score += 0.3  # Too uniform
        
        # === 10. WHITESPACE EFFECTIVENESS ===
        # Analyze use of whitespace for readability
        
        whitespace_score = 0.0
        if total_chars > 0:
            blank_lines = sum(1 for l in all_lines if not l.strip())
            total_line_count = len(all_lines)
            
            if total_line_count > 3:
                blank_ratio = blank_lines / total_line_count
                if 0.1 <= blank_ratio <= 0.4:
                    whitespace_score += 1.0  # Good whitespace usage
                elif blank_ratio > 0.5:
                    whitespace_score -= 0.5  # Too much whitespace
            
            # Check for indentation usage (structural hierarchy)
            indented_lines = sum(1 for l in non_empty_lines if l.startswith('  ') or l.startswith('\t'))
            if indented_lines > 0 and total_words > 30:
                whitespace_score += 0.5
        
        # === FINAL SCORE ASSEMBLY ===
        
        score = (
            2.5 +                          # Base score
            structural_diversity_score +    # 0 to 2.5
            density_score +                 # -1 to 3.5
            coherence_score +               # -1.5 to 2.0
            proportionality_score +         # -1.5 to 2.5
            sentence_variance_score +       # 0 to 1.0
            whitespace_score +              # -0.5 to 1.5
            - repetition_penalty +          # 0 to 6.5
            - noise_penalty                 # 0 to 3.5
        )
        
        # Clamp to [0, 10]
        score = max(0.0, min(10.0, score))
        
        # Final adjustment: very short meaningless responses
        if total_words <= 2:
            score = min(score, 2.0)
        elif total_words <= 5 and '.' not in response and '!' not in response:
            score = min(score, 3.0)
        
        return round(score, 2)
    
    except Exception:
        return 3.0