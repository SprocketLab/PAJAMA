def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses an information-theoretic and visual layout approach:
    - Analyzes the "visual density profile" of the text (line length variance)
    - Measures hierarchical depth through indentation and nesting patterns
    - Evaluates information chunking via sentence-per-paragraph ratios
    - Scores structural diversity using entropy of structural element types
    - Analyzes the rhythm/cadence of paragraph lengths for readability
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response = response.strip()
        if len(response) < 20:
            return 1.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Visual Density Profile Analysis
        # Analyze the distribution of line lengths to detect visual variety
        # Good formatting creates varied line lengths (lists vs paragraphs)
        # ============================================================
        if len(non_empty_lines) >= 2:
            line_lengths = [len(l.strip()) for l in non_empty_lines]
            mean_len = sum(line_lengths) / len(line_lengths)
            
            # Coefficient of variation of line lengths
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len
                
                # Moderate CV suggests mixed structure (headers, lists, paragraphs)
                # Very low CV = wall of text; very high = chaotic
                if 0.2 <= cv <= 1.5:
                    density_score = min(cv * 3.0, 4.0)
                elif cv > 1.5:
                    density_score = max(4.0 - (cv - 1.5), 1.5)
                else:
                    density_score = cv * 5.0  # low CV penalized
                score += density_score
            
            # Bonus for having short lines mixed with longer ones (structural elements)
            short_lines = sum(1 for l in line_lengths if l < 60)
            long_lines = sum(1 for l in line_lengths if l >= 60)
            if short_lines > 0 and long_lines > 0:
                mix_ratio = min(short_lines, long_lines) / max(short_lines, long_lines)
                score += mix_ratio * 2.0
        
        # ============================================================
        # FEATURE 2: Structural Element Type Entropy
        # Classify each line into a type and measure diversity via entropy
        # ============================================================
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
            elif re.match(r'^\d+[\.\)]\s', stripped):
                element_types.append('numbered_item')
            elif re.match(r'^[-*•●◦▪]\s', stripped):
                element_types.append('bullet_item')
            elif re.match(r'^[a-z][\.\)]\s', stripped):
                element_types.append('letter_item')
            elif stripped.endswith(':') and len(stripped) < 80:
                element_types.append('label_line')
            elif re.match(r'^\*\*.*\*\*', stripped) or re.match(r'^__.*__', stripped):
                element_types.append('bold_header')
            elif len(stripped) < 40 and stripped[0].isupper() and not stripped.endswith('.'):
                element_types.append('short_header')
            else:
                element_types.append('paragraph_text')
        
        if element_types:
            type_counts = Counter(element_types)
            num_types = len(type_counts)
            total_elements = len(element_types)
            
            # Shannon entropy of element types
            entropy = 0.0
            for count in type_counts.values():
                p = count / total_elements
                if p > 0:
                    entropy -= p * math.log2(p)
            
            # More diverse structure = better (up to a point)
            # Max possible entropy depends on number of types
            max_entropy = math.log2(num_types) if num_types > 1 else 0
            
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
                score += normalized_entropy * 3.0
            
            # Bonus for having non-paragraph types
            non_para_ratio = 1.0 - (type_counts.get('paragraph_text', 0) / total_elements)
            if non_para_ratio > 0.1:
                score += min(non_para_ratio * 4.0, 3.0)
        
        # ============================================================
        # FEATURE 3: Paragraph Rhythm Analysis
        # Good writing has varied but not chaotic paragraph sizes
        # Analyze the "rhythm" of paragraph lengths
        # ============================================================
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if len(paragraphs) >= 2:
            para_lengths = [len(p) for p in paragraphs]
            
            # Score for having multiple paragraphs (not a wall of text)
            para_count_score = min(len(paragraphs) / 3.0, 3.0)
            score += para_count_score
            
            # Analyze rhythm: consecutive paragraph length ratios
            ratios = []
            for i in range(1, len(para_lengths)):
                if para_lengths[i-1] > 0:
                    ratio = para_lengths[i] / para_lengths[i-1]
                    ratios.append(ratio)
            
            if ratios:
                # Varied ratios suggest intentional structure
                ratio_variance = sum((r - 1.0) ** 2 for r in ratios) / len(ratios)
                # Moderate variance is good
                if 0.1 <= ratio_variance <= 4.0:
                    score += 2.0
                elif ratio_variance > 0.01:
                    score += 1.0
        elif len(paragraphs) == 1:
            # Single paragraph - check if it's very long (wall of text)
            if len(paragraphs[0]) > 300:
                score -= 2.0  # Penalize wall of text
        
        # ============================================================
        # FEATURE 4: Information Chunking Quality
        # Measure sentences per structural unit and consistency
        # ============================================================
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        num_sentences = max(len(sentences), 1)
        
        if len(paragraphs) > 0:
            sentences_per_para = num_sentences / len(paragraphs)
            
            # Ideal: 2-5 sentences per paragraph
            if 1.5 <= sentences_per_para <= 5.0:
                score += 3.0
            elif 1.0 <= sentences_per_para <= 7.0:
                score += 1.5
            else:
                score += 0.5
        
        # ============================================================
        # FEATURE 5: Whitespace Utilization Score
        # Measure how effectively whitespace is used for visual separation
        # ============================================================
        total_lines = len(lines)
        empty_lines = sum(1 for l in lines if not l.strip())
        
        if total_lines > 0:
            whitespace_ratio = empty_lines / total_lines
            
            # Some whitespace is good (0.1 to 0.4 range)
            if 0.05 <= whitespace_ratio <= 0.45:
                ws_score = 2.5
            elif whitespace_ratio > 0.45:
                ws_score = 1.0  # Too much whitespace
            else:
                ws_score = 0.5  # No whitespace separation
            score += ws_score
        
        # ============================================================
        # FEATURE 6: Semantic Signposting Detection
        # Look for discourse markers that signal structure
        # Different from transition words - these are organizational cues
        # ============================================================
        signpost_patterns = [
            r'\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly)\b',
            r'\b(?:step\s+\d|phase\s+\d|part\s+\d)\b',
            r'\b(?:in\s+summary|to\s+summarize|in\s+conclusion|to\s+conclude)\b',
            r'\b(?:on\s+one\s+hand|on\s+the\s+other\s+hand|alternatively)\b',
            r'\b(?:for\s+example|for\s+instance|such\s+as|specifically)\b',
            r'\b(?:here\s+are|the\s+following|as\s+follows|listed\s+below)\b',
            r'\b(?:key\s+points?|important(?:ly)?|note\s+that|keep\s+in\s+mind)\b',
            r'\b(?:let\'s\s+start|let\'s\s+begin|moving\s+on|next\s+up)\b',
        ]
        
        response_lower = response.lower()
        signpost_count = 0
        for pattern in signpost_patterns:
            signpost_count += len(re.findall(pattern, response_lower))
        
        signpost_score = min(signpost_count * 0.8, 3.0)
        score += signpost_score
        
        # ============================================================
        # FEATURE 7: Formatting Markup Detection
        # Detect bold, italic, code blocks, and other rich formatting
        # ============================================================
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        code_count = len(re.findall(r'`[^`]+`', response))
        colon_headers = len(re.findall(r'^[A-Z][^.!?\n]{2,40}:', response, re.MULTILINE))
        
        markup_elements = bold_count + italic_count + code_count + colon_headers
        if markup_elements > 0:
            score += min(markup_elements * 0.6, 3.0)
        
        # ============================================================
        # FEATURE 8: Response Length Appropriateness
        # Very short responses unlikely to have good structure
        # But length alone isn't enough
        # ============================================================
        word_count = len(response.split())
        if word_count < 30:
            score *= 0.6
        elif word_count < 50:
            score *= 0.8
        elif word_count > 100:
            # Longer responses that maintain structure get a small bonus
            if len(paragraphs) >= 2 or len(element_types) > 3:
                score += 1.0
        
        # ============================================================
        # PENALTY: Wall of Text Detection
        # Strong penalty for long unbroken blocks
        # ============================================================
        max_para_len = max((len(p) for p in paragraphs), default=0)
        if max_para_len > 500 and len(paragraphs) <= 2:
            score -= 3.0
        elif max_para_len > 400 and len(paragraphs) <= 1:
            score -= 4.0
        
        # ============================================================
        # PENALTY: Repetitive structure (all same type)
        # ============================================================
        if element_types:
            most_common_ratio = max(Counter(element_types).values()) / len(element_types)
            if most_common_ratio > 0.95 and len(element_types) > 3:
                score -= 1.0
        
        # Normalize to 1-5 range
        # Theoretical max around 25-30, theoretical min around -5
        # Map to 1-5 range
        normalized = 1.0 + (score / 7.0)  # Roughly maps typical range to 1-5
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5