def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses an information-theoretic and spatial analysis approach:
    - Entropy of line lengths (well-structured text has varied but purposeful line lengths)
    - Visual rhythm analysis (alternation between different structural elements)
    - Hierarchical depth detection (nested structures indicate organization)
    - Semantic chunking quality (how well content is divided into digestible pieces)
    - Whitespace distribution patterns
    - Opening/closing structure quality
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 20:
            return 0.5
        
        score = 0.0
        
        # === 1. LINE-LEVEL STRUCTURAL DIVERSITY (0-15 points) ===
        # Classify each line into a type and measure the diversity/rhythm of types
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        def classify_line(line):
            stripped = line.strip()
            if not stripped:
                return 'empty'
            if re.match(r'^#{1,6}\s', stripped):
                return 'header_md'
            if re.match(r'^[A-Z][A-Za-z\s]{0,50}:$', stripped):
                return 'header_colon'
            if re.match(r'^[A-Z][A-Za-z\s]{0,50}$', stripped) and len(stripped) < 60:
                return 'header_plain'
            if re.match(r'^\d+[\.\)]\s', stripped):
                return 'numbered'
            if re.match(r'^[-*•–—]\s', stripped):
                return 'bullet'
            if re.match(r'^[a-z][\.\)]\s', stripped):
                return 'lettered'
            if len(stripped) > 80:
                return 'long_para'
            if len(stripped) > 30:
                return 'medium_text'
            return 'short_text'
        
        line_types = [classify_line(l) for l in lines]
        non_empty_types = [t for t in line_types if t != 'empty']
        
        # Diversity of structural elements used
        type_set = set(non_empty_types)
        structural_types = {'header_md', 'header_colon', 'header_plain', 'numbered', 'bullet', 'lettered'}
        used_structural = type_set & structural_types
        
        diversity_score = min(len(used_structural) * 3.0, 9.0)
        # Bonus for having both headers and list items
        has_headers = bool(used_structural & {'header_md', 'header_colon', 'header_plain'})
        has_lists = bool(used_structural & {'numbered', 'bullet', 'lettered'})
        if has_headers and has_lists:
            diversity_score += 4.0
        elif has_lists:
            diversity_score += 2.0
        elif has_headers:
            diversity_score += 1.5
        
        score += min(diversity_score, 15.0)
        
        # === 2. RHYTHM AND ALTERNATION ANALYSIS (0-12 points) ===
        # Good structure alternates between different element types (not monotonous)
        # Compute bigram transitions in line types
        rhythm_score = 0.0
        if len(non_empty_types) >= 3:
            transitions = 0
            for i in range(1, len(non_empty_types)):
                if non_empty_types[i] != non_empty_types[i-1]:
                    transitions += 1
            transition_rate = transitions / (len(non_empty_types) - 1) if len(non_empty_types) > 1 else 0
            # Moderate transition rate is best (0.3-0.7) - not too uniform, not too chaotic
            if 0.2 <= transition_rate <= 0.8:
                rhythm_score = 8.0
            elif 0.1 <= transition_rate <= 0.9:
                rhythm_score = 5.0
            else:
                rhythm_score = 2.0
            
            # Check for structured sequences (e.g., header followed by content, repeated patterns)
            pattern_str = ''.join(['H' if t.startswith('header') else 
                                   'N' if t == 'numbered' else
                                   'B' if t == 'bullet' else
                                   'T' for t in non_empty_types])
            # Header-content pattern
            if re.search(r'H[TNB]+', pattern_str):
                rhythm_score += 2.0
            # Consistent list pattern
            if re.search(r'N{2,}', pattern_str) or re.search(r'B{2,}', pattern_str):
                rhythm_score += 2.0
        
        score += min(rhythm_score, 12.0)
        
        # === 3. PARAGRAPH SEGMENTATION QUALITY (0-15 points) ===
        # Split by double newlines or significant whitespace gaps
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        para_score = 0.0
        response_len = len(response)
        
        if response_len > 200:
            # For longer responses, multiple paragraphs are expected
            if num_paragraphs >= 4:
                para_score += 7.0
            elif num_paragraphs >= 3:
                para_score += 5.5
            elif num_paragraphs >= 2:
                para_score += 3.5
            else:
                para_score += 0.5  # Wall of text penalty
            
            # Paragraph length variance - should be somewhat balanced
            if num_paragraphs >= 2:
                para_lengths = [len(p) for p in paragraphs]
                avg_len = sum(para_lengths) / len(para_lengths)
                if avg_len > 0:
                    cv = (sum((l - avg_len)**2 for l in para_lengths) / len(para_lengths))**0.5 / avg_len
                    # CV between 0.2 and 1.0 is good (some variation but not extreme)
                    if 0.1 <= cv <= 1.2:
                        para_score += 4.0
                    elif cv <= 2.0:
                        para_score += 2.0
                    else:
                        para_score += 0.5
                
                # Check that no single paragraph is overwhelmingly long
                max_para_ratio = max(para_lengths) / sum(para_lengths)
                if max_para_ratio < 0.6:
                    para_score += 4.0
                elif max_para_ratio < 0.75:
                    para_score += 2.0
        else:
            # Short responses - simpler expectations
            if num_paragraphs >= 2:
                para_score += 8.0
            elif num_paragraphs == 1 and len(non_empty_lines) > 1:
                para_score += 5.0
            else:
                para_score += 3.0
        
        score += min(para_score, 15.0)
        
        # === 4. INFORMATION DENSITY DISTRIBUTION (0-12 points) ===
        # Analyze how information is spatially distributed across the response
        # Split response into quartiles and check each has substantive content
        density_score = 0.0
        
        if response_len > 100:
            quarter = response_len // 4
            quarters = [
                response[:quarter],
                response[quarter:2*quarter],
                response[2*quarter:3*quarter],
                response[3*quarter:]
            ]
            
            # Count substantive words per quarter
            def word_density(text):
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                return len(words)
            
            densities = [word_density(q) for q in quarters]
            total_density = sum(densities)
            
            if total_density > 0:
                # Check distribution evenness
                ratios = [d / total_density for d in densities]
                # Ideal: each quarter has ~25% of content
                evenness = 1.0 - sum(abs(r - 0.25) for r in ratios) / 2.0
                density_score += evenness * 8.0
            
            # Check for strong opening and closing
            first_sentence = re.split(r'[.!?]', response)[0] if response else ''
            if len(first_sentence.strip()) > 15:
                density_score += 2.0
            
            # Check last paragraph has closure
            if paragraphs:
                last_para = paragraphs[-1]
                closing_patterns = [
                    r'\bremember\b', r'\bin summary\b', r'\boverall\b', r'\bfinally\b',
                    r'\bdon\'t hesitate\b', r'\bfeel free\b', r'\bhope\b', r'\bgood luck\b',
                    r'\bin conclusion\b', r'\bto sum up\b', r'\blast\w*\b', r'\bimportant\w*\b'
                ]
                if any(re.search(p, last_para, re.IGNORECASE) for p in closing_patterns):
                    density_score += 2.0
        
        score += min(density_score, 12.0)
        
        # === 5. SENTENCE LENGTH VARIETY (0-8 points) ===
        # Good writing mixes sentence lengths
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        sent_variety_score = 0.0
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sl = sum(sent_lengths) / len(sent_lengths)
            if avg_sl > 0:
                sl_std = (sum((l - avg_sl)**2 for l in sent_lengths) / len(sent_lengths))**0.5
                # Some variety is good
                cv_sent = sl_std / avg_sl if avg_sl > 0 else 0
                if 0.3 <= cv_sent <= 1.0:
                    sent_variety_score += 5.0
                elif 0.15 <= cv_sent <= 1.5:
                    sent_variety_score += 3.0
                else:
                    sent_variety_score += 1.0
            
            # Not too many very long sentences
            long_sents = sum(1 for l in sent_lengths if l > 35)
            if long_sents / len(sent_lengths) < 0.3:
                sent_variety_score += 3.0
            elif long_sents / len(sent_lengths) < 0.5:
                sent_variety_score += 1.5
        
        score += min(sent_variety_score, 8.0)
        
        # === 6. EXPLICIT FORMATTING MARKERS (0-10 points) ===
        # Count specific formatting elements
        fmt_score = 0.0
        
        # Numbered lists (1. 2. 3. etc.)
        numbered_items = re.findall(r'(?m)^\s*\d+[\.\)]\s+\S', response)
        if len(numbered_items) >= 3:
            fmt_score += 4.0
        elif len(numbered_items) >= 2:
            fmt_score += 2.5
        elif len(numbered_items) >= 1:
            fmt_score += 1.0
        
        # Bullet points
        bullet_items = re.findall(r'(?m)^\s*[-*•]\s+\S', response)
        if len(bullet_items) >= 3:
            fmt_score += 3.0
        elif len(bullet_items) >= 2:
            fmt_score += 2.0
        elif len(bullet_items) >= 1:
            fmt_score += 1.0
        
        # Bold/emphasis markers
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        if bold_count >= 2:
            fmt_score += 1.5
        elif bold_count >= 1:
            fmt_score += 0.75
        
        # Colons used as sub-headers or labels within text
        colon_labels = re.findall(r'(?m)^\s*\d+[\.\)]\s*[A-Z][^:]{2,30}:', response)
        if len(colon_labels) >= 2:
            fmt_score += 1.5
        
        score += min(fmt_score, 10.0)
        
        # === 7. COHERENCE SIGNALS (0-8 points) ===
        # Look for discourse markers that signal organization
        coherence_score = 0.0
        
        # Sequencing markers
        seq_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bafter that\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bfollowing\b'
        ]
        seq_count = sum(1 for p in seq_markers if re.search(p, response, re.IGNORECASE))
        coherence_score += min(seq_count * 1.0, 3.0)
        
        # Contrast/addition markers
        logic_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\bon the other hand\b', r'\bthat said\b',
            r'\bfor instance\b', r'\bfor example\b', r'\bspecifically\b'
        ]
        logic_count = sum(1 for p in logic_markers if re.search(p, response, re.IGNORECASE))
        coherence_score += min(logic_count * 1.0, 3.0)
        
        # Topic sentences at paragraph starts
        if num_paragraphs >= 2:
            good_starts = 0
            for para in paragraphs:
                first_sent = re.split(r'[.!?]', para)[0].strip()
                words = first_sent.split()
                if 4 <= len(words) <= 25:
                    good_starts += 1
            start_ratio = good_starts / num_paragraphs
            coherence_score += start_ratio * 2.0
        
        score += min(coherence_score, 8.0)
        
        # === 8. WALL-OF-TEXT PENALTY (0 to -8 points) ===
        wall_penalty = 0.0
        
        if response_len > 300:
            # If only 1 paragraph for long text
            if num_paragraphs == 1 and len(non_empty_lines) <= 2:
                wall_penalty -= 5.0
            
            # Very long lines without breaks
            if non_empty_lines:
                max_line_len = max(len(l) for l in non_empty_lines)
                avg_line_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
                if avg_line_len > 400:
                    wall_penalty -= 3.0
                elif avg_line_len > 250:
                    wall_penalty -= 1.5
        
        # No whitespace/breaks at all
        newline_ratio = response.count('\n') / max(response_len, 1) * 100
        if response_len > 200 and newline_ratio < 0.3:
            wall_penalty -= 2.0
        
        score += wall_penalty
        
        # === 9. RESPONSE LENGTH APPROPRIATENESS (0-5 points) ===
        # Very short responses for complex queries may lack structure
        query_len = len(query) if query else 0
        length_score = 0.0
        
        if response_len > 150:
            length_score += 2.0
        if response_len > 300:
            length_score += 1.5
        if response_len > 500:
            length_score += 1.5
        
        score += min(length_score, 5.0)
        
        # === 10. EMPATHY/ENGAGEMENT OPENING (0-5 points) ===
        # Well-structured responses often start with acknowledgment
        opening_score = 0.0
        first_50 = response[:min(80, len(response))].lower()
        
        engagement_starters = [
            r'^i can', r'^i\'m', r'^it\'s', r'^that\'s', r'^imagine',
            r'^hey', r'^hello', r'^sure', r'^absolutely', r'^great question',
            r'^i understand', r'^i hear', r'^it sounds', r'^let\'s',
            r'^here', r'^to '
        ]
        if any(re.search(p, first_50) for p in engagement_starters):
            opening_score += 2.5
        
        # Check if response addresses the query topic
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query.lower())) if query else set()
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response.lower()))
        overlap = len(query_words & response_words)
        if overlap >= 3:
            opening_score += 2.5
        elif overlap >= 1:
            opening_score += 1.0
        
        score += min(opening_score, 5.0)
        
        # Normalize to 1-5 range
        # Max possible raw score: ~90 points
        # Typical good: 50-70, typical bad: 15-35
        raw_max = 85.0
        normalized = (score / raw_max) * 4.0 + 1.0  # Maps to 1-5
        
        # Clamp
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
    
    except Exception:
        return 2.5