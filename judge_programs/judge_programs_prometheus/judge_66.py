def judging_function(query, response):
    """
    Evaluates structural organization and formatting of a response.
    
    This variant focuses on:
    - Information density distribution (entropy of content across segments)
    - Sentence length variance as a proxy for structural variety
    - Indentation/nesting patterns
    - Opening/closing structure quality
    - Ratio of structural markers to content
    - Readability rhythm (alternating short/long segments)
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
        
        # === 1. LINE-LEVEL STRUCTURE ANALYSIS ===
        # Split into lines and analyze the distribution of line lengths
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        num_lines = len(non_empty_lines)
        
        # Reward having multiple lines (not a single block)
        if num_lines >= 3:
            score += 1.0
        elif num_lines >= 2:
            score += 0.5
        
        # === 2. BLANK LINE SEPARATION (paragraph breaks) ===
        # Count groups of consecutive blank lines as separators
        blank_line_groups = 0
        prev_blank = False
        for line in lines:
            if line.strip() == '':
                if not prev_blank:
                    blank_line_groups += 1
                prev_blank = True
            else:
                prev_blank = False
        
        # Reward appropriate number of paragraph breaks relative to response length
        words = response.split()
        word_count = len(words)
        
        if word_count > 0:
            ideal_breaks = max(1, word_count // 60)  # roughly one break per 60 words
            break_score = 1.0 - min(1.0, abs(blank_line_groups - ideal_breaks) / max(ideal_breaks, 1))
            score += break_score * 1.5
        
        # === 3. SENTENCE LENGTH VARIANCE (rhythm) ===
        # Good writing alternates between shorter and longer sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            
            if mean_len > 0:
                # Coefficient of variation of sentence lengths
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len
                
                # Moderate variance is good (0.3-0.7 CV is ideal)
                if 0.25 <= cv <= 0.8:
                    score += 1.2
                elif 0.15 <= cv <= 1.0:
                    score += 0.6
                else:
                    score += 0.2
        
        # === 4. ENUMERATION AND SEQUENTIAL STRUCTURE ===
        # Detect numbered items, lettered items, or consistent bullet patterns
        numbered_pattern = re.findall(r'(?:^|\n)\s*(\d+)[.)]\s', response)
        lettered_pattern = re.findall(r'(?:^|\n)\s*[a-zA-Z][.)]\s', response)
        dash_bullets = re.findall(r'(?:^|\n)\s*[-–—]\s', response)
        star_bullets = re.findall(r'(?:^|\n)\s*[•*]\s', response)
        colon_headers = re.findall(r'(?:^|\n)\s*[A-Z][^.!?\n]{2,30}:\s', response)
        
        list_items = len(numbered_pattern) + len(lettered_pattern) + len(dash_bullets) + len(star_bullets)
        
        # Check if numbered items are sequential
        if len(numbered_pattern) >= 2:
            nums = [int(n) for n in numbered_pattern]
            is_sequential = all(nums[i] + 1 == nums[i+1] for i in range(len(nums)-1))
            if is_sequential:
                score += 1.5
            else:
                score += 0.7
        elif list_items >= 2:
            score += 1.2
        elif list_items == 1:
            score += 0.3
        
        # Colon-based headers/labels
        if len(colon_headers) >= 2:
            score += 1.0
        elif len(colon_headers) >= 1:
            score += 0.4
        
        # === 5. INFORMATION DENSITY DISTRIBUTION ===
        # Split response into quarters and check if content is evenly distributed
        if word_count >= 20:
            quarter = word_count // 4
            quarters = [
                words[:quarter],
                words[quarter:2*quarter],
                words[2*quarter:3*quarter],
                words[3*quarter:]
            ]
            
            # Measure unique word ratio in each quarter
            quarter_densities = []
            for q in quarters:
                if len(q) > 0:
                    unique_ratio = len(set(w.lower() for w in q)) / len(q)
                    quarter_densities.append(unique_ratio)
            
            if len(quarter_densities) >= 3:
                density_range = max(quarter_densities) - min(quarter_densities)
                # Low range means even distribution of information
                if density_range < 0.15:
                    score += 0.8
                elif density_range < 0.3:
                    score += 0.4
        
        # === 6. OPENING QUALITY ===
        # Good responses often start with acknowledgment, context-setting, or clear topic statement
        first_sentence = sentences[0] if sentences else ""
        first_words = first_sentence.split()[:10]
        first_text = ' '.join(first_words).lower()
        
        # Empathetic/engaging openings
        engaging_openers = [
            r"^i('m| am|can)", r"^(it's|it is)", r"^(imagine|picture|think)",
            r"^(hey|hello|hi)\b", r"^(let'?s|we)", r"^(here|below)",
            r"^(great|good|excellent)", r"^(to |in order)",
            r"^(i understand|i hear|i see|i can see)",
            r"^(sorry|apolog)", r"^(absolutely|certainly|of course)",
        ]
        
        for pattern in engaging_openers:
            if re.search(pattern, first_text):
                score += 0.5
                break
        
        # === 7. WALL-OF-TEXT PENALTY ===
        # Penalize responses that are one giant paragraph with many words
        if word_count > 50 and blank_line_groups == 0 and list_items == 0:
            # Strong wall-of-text penalty
            penalty = min(2.0, (word_count - 50) / 100)
            score -= penalty
        
        # Also penalize very long lines without breaks
        if non_empty_lines:
            max_line_len = max(len(l) for l in non_empty_lines)
            avg_line_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
            
            if avg_line_len > 300 and num_lines <= 2:
                score -= 1.0
        
        # === 8. CONNECTIVE TISSUE / DISCOURSE MARKERS ===
        # Check for logical flow markers (different from transition words - focus on discourse structure)
        discourse_markers = [
            r'\b(first(ly)?|second(ly)?|third(ly)?|finally|lastly|next|then)\b',
            r'\b(however|nevertheless|nonetheless|on the other hand)\b',
            r'\b(for (example|instance)|such as|specifically|in particular)\b',
            r'\b(in (summary|conclusion|addition)|to (summarize|sum up)|overall)\b',
            r'\b(moreover|furthermore|additionally|also|besides)\b',
            r'\b(therefore|thus|consequently|as a result|hence)\b',
            r'\b(remember|note|keep in mind|importantly)\b',
        ]
        
        response_lower = response.lower()
        marker_categories_found = 0
        for pattern in discourse_markers:
            if re.search(pattern, response_lower):
                marker_categories_found += 1
        
        # Reward diversity of discourse marker types
        score += min(1.5, marker_categories_found * 0.35)
        
        # === 9. SEGMENT COHERENCE ===
        # Check if paragraphs/segments start with topic-sentence-like patterns
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if len(paragraphs) >= 2:
            topic_sentence_count = 0
            for para in paragraphs:
                first_sent = re.split(r'[.!?]', para)[0].strip()
                words_in_first = len(first_sent.split())
                # Topic sentences are typically moderate length (5-25 words)
                if 4 <= words_in_first <= 30:
                    topic_sentence_count += 1
            
            topic_ratio = topic_sentence_count / len(paragraphs)
            score += topic_ratio * 0.8
        
        # === 10. STRUCTURAL VARIETY SCORE ===
        # Reward responses that use MULTIPLE formatting techniques
        techniques_used = 0
        if blank_line_groups >= 1:
            techniques_used += 1
        if list_items >= 2:
            techniques_used += 1
        if len(colon_headers) >= 1:
            techniques_used += 1
        if marker_categories_found >= 2:
            techniques_used += 1
        if len(paragraphs) >= 2:
            techniques_used += 1
        
        # Bonus for combining multiple techniques
        if techniques_used >= 3:
            score += 1.0
        elif techniques_used >= 2:
            score += 0.5
        
        # === 11. RESPONSE COMPLETENESS INDICATOR ===
        # Responses that seem to have structure throughout (not just at start)
        if word_count > 30:
            second_half = ' '.join(words[word_count//2:])
            has_structure_in_second_half = bool(
                re.search(r'\d+[.)]\s|[-•*]\s|[A-Z][^.!?\n]{2,25}:', second_half)
                or re.search(r'\b(also|additionally|finally|remember|moreover)\b', second_half.lower())
            )
            if has_structure_in_second_half:
                score += 0.5
        
        # === NORMALIZE SCORE ===
        # Map to 1-5 range
        # Raw score typically ranges from about -1 to 10
        # Clamp and normalize
        raw_min = -1.0
        raw_max = 9.0
        
        normalized = (score - raw_min) / (raw_max - raw_min)  # 0 to 1
        normalized = max(0.0, min(1.0, normalized))
        
        final_score = 1.0 + normalized * 4.0  # 1 to 5
        
        # Round to one decimal
        final_score = round(final_score, 1)
        
        return final_score
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5