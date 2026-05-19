def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a
    hierarchical structure analysis approach based on information-theoretic
    principles and visual layout simulation.
    
    This variant focuses on:
    1. Visual density analysis (simulating how text would appear rendered)
    2. Information chunking quality (entropy of segment lengths)
    3. Structural hierarchy depth detection
    4. Repetition/redundancy penalties
    5. Signal-to-noise ratio in formatting elements
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
        
        # Very short responses - score based on whether they're appropriate
        if len(response) < 10:
            return 1.0
        
        score = 0.0
        
        # ============================================================
        # 1. VISUAL DENSITY ANALYSIS
        # Simulate rendering: compute the "visual rhythm" of the text
        # by analyzing line-by-line character density patterns
        # ============================================================
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        empty_lines = [l for l in lines if not l.strip()]
        
        if len(lines) > 0:
            # Compute character density per line
            line_lengths = [len(l.rstrip()) for l in lines]
            
            # Visual rhythm score: variance in line lengths suggests structure
            if len(line_lengths) > 1:
                mean_len = sum(line_lengths) / len(line_lengths)
                if mean_len > 0:
                    # Coefficient of variation of line lengths
                    variance = sum((x - mean_len) ** 2 for x in line_lengths) / len(line_lengths)
                    cv = math.sqrt(variance) / mean_len if mean_len > 0 else 0
                    # Moderate variation is good (suggests mixed structure)
                    # Too low = wall of text, too high = chaotic
                    if 0.3 <= cv <= 1.5:
                        score += 6.0 * min(cv, 1.0)
                    elif cv > 1.5:
                        score += 4.0
                    else:
                        score += cv * 8.0  # reward some variation
            
            # Empty line ratio (breathing room)
            empty_ratio = len(empty_lines) / max(len(lines), 1)
            if 0.1 <= empty_ratio <= 0.45:
                score += 5.0
            elif 0.05 <= empty_ratio < 0.1:
                score += 2.5
            elif 0.45 < empty_ratio <= 0.6:
                score += 2.0
            elif empty_ratio > 0.6:
                score -= 2.0  # too sparse
        
        # ============================================================
        # 2. INFORMATION CHUNKING QUALITY
        # Analyze how information is segmented using entropy of chunk sizes
        # ============================================================
        # Split by double newlines to find "chunks"
        chunks = re.split(r'\n\s*\n', response)
        chunks = [c.strip() for c in chunks if c.strip()]
        num_chunks = len(chunks)
        
        if num_chunks > 1:
            chunk_sizes = [len(c) for c in chunks]
            total_size = sum(chunk_sizes)
            
            if total_size > 0:
                # Entropy of chunk size distribution
                probs = [s / total_size for s in chunk_sizes]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                max_entropy = math.log2(num_chunks) if num_chunks > 1 else 1
                
                # Normalized entropy - higher means more even distribution
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Reward balanced chunking (not one huge chunk + tiny ones)
                score += 5.0 * norm_entropy
                
                # Bonus for having a reasonable number of chunks
                if 2 <= num_chunks <= 8:
                    score += 3.0
                elif 8 < num_chunks <= 15:
                    score += 2.0
                elif num_chunks > 15:
                    score += 0.5  # too fragmented
        else:
            # Single chunk - penalize if it's long (wall of text)
            if len(response) > 300:
                score -= 3.0
            elif len(response) > 150:
                score -= 1.0
        
        # ============================================================
        # 3. STRUCTURAL HIERARCHY DEPTH
        # Detect nested/layered organization patterns
        # ============================================================
        hierarchy_depth = 0
        hierarchy_elements = set()
        
        # Detect markdown-style headers at different levels
        h1_pattern = re.findall(r'^#{1}\s+\S', response, re.MULTILINE)
        h2_pattern = re.findall(r'^#{2}\s+\S', response, re.MULTILINE)
        h3_pattern = re.findall(r'^#{3,}\s+\S', response, re.MULTILINE)
        
        if h1_pattern:
            hierarchy_elements.add('h1')
        if h2_pattern:
            hierarchy_elements.add('h2')
        if h3_pattern:
            hierarchy_elements.add('h3')
        
        # Detect numbered lists (various formats)
        numbered_items = re.findall(r'^\s*\d+[\.\)]\s+\S', response, re.MULTILINE)
        if numbered_items:
            hierarchy_elements.add('numbered')
        
        # Detect lettered sub-items
        lettered_items = re.findall(r'^\s{2,}[a-z][\.\)]\s+\S', response, re.MULTILINE)
        if lettered_items:
            hierarchy_elements.add('lettered_sub')
        
        # Detect bullet points (various styles)
        bullet_items = re.findall(r'^\s*[-*•▪▸►]\s+\S', response, re.MULTILINE)
        if bullet_items:
            hierarchy_elements.add('bullets')
        
        # Detect indented sub-bullets
        sub_bullets = re.findall(r'^\s{2,}[-*•]\s+\S', response, re.MULTILINE)
        if sub_bullets:
            hierarchy_elements.add('sub_bullets')
        
        # Detect bold/emphasized text used as section markers
        bold_markers = re.findall(r'\*\*[^*]+\*\*', response)
        if bold_markers and len(bold_markers) >= 2:
            hierarchy_elements.add('bold_sections')
        
        # Detect colon-delimited labels (e.g., "Category: value")
        colon_labels = re.findall(r'^[A-Z][^:\n]{2,30}:\s+\S', response, re.MULTILINE)
        if colon_labels and len(colon_labels) >= 2:
            hierarchy_elements.add('colon_labels')
        
        hierarchy_depth = len(hierarchy_elements)
        
        # Score hierarchy: more diverse structural elements = better organization
        if hierarchy_depth >= 4:
            score += 10.0
        elif hierarchy_depth == 3:
            score += 8.0
        elif hierarchy_depth == 2:
            score += 6.0
        elif hierarchy_depth == 1:
            score += 3.5
        
        # Bonus for list items being consistent in quantity
        total_list_items = len(numbered_items) + len(bullet_items)
        if 3 <= total_list_items <= 12:
            score += 3.0
        elif total_list_items > 12:
            score += 1.5
        elif total_list_items > 0:
            score += 1.0
        
        # ============================================================
        # 4. REPETITION / REDUNDANCY PENALTY
        # Detect repeated phrases, lines, or patterns that indicate
        # poor quality (copy-paste, loops, degenerate output)
        # ============================================================
        
        # Check for repeated lines
        line_texts = [l.strip() for l in non_empty_lines if len(l.strip()) > 5]
        if line_texts:
            line_counter = Counter(line_texts)
            most_common_count = line_counter.most_common(1)[0][1] if line_counter else 1
            unique_ratio = len(set(line_texts)) / len(line_texts) if line_texts else 1
            
            if unique_ratio < 0.5:
                score -= 8.0  # heavy repetition
            elif unique_ratio < 0.75:
                score -= 4.0
            elif most_common_count > 3:
                score -= 3.0
        
        # Check for repeated n-grams (3-grams) indicating loops
        words = response.lower().split()
        if len(words) > 10:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counter = Counter(trigrams)
            if trigram_counter:
                max_trigram_freq = trigram_counter.most_common(1)[0][1]
                expected_freq = len(trigrams) / max(len(set(trigrams)), 1)
                if max_trigram_freq > max(5, expected_freq * 3):
                    score -= 5.0
        
        # ============================================================
        # 5. SIGNAL-TO-NOISE RATIO IN FORMATTING
        # Detect garbage formatting, broken HTML, code dumps that
        # aren't asked for, etc.
        # ============================================================
        
        # Detect excessive HTML tags (noise)
        html_tags = re.findall(r'<[^>]+>', response)
        if html_tags:
            # If query doesn't ask for HTML, HTML tags are noise
            query_lower = query.lower() if query else ""
            if 'html' not in query_lower and 'tag' not in query_lower and 'code' not in query_lower:
                if len(html_tags) > 5:
                    score -= 4.0
                elif len(html_tags) > 2:
                    score -= 1.5
        
        # Detect code blocks when not asked for
        code_blocks = re.findall(r'```[\s\S]*?```', response)
        query_lower = query.lower() if query else ""
        code_keywords = {'code', 'program', 'function', 'script', 'implement', 'python', 'java', 'write a'}
        query_asks_code = any(kw in query_lower for kw in code_keywords)
        
        if code_blocks and not query_asks_code:
            score -= 2.0
        
        # Detect "Output:" or "Input:" spam (degenerate pattern)
        output_spam = re.findall(r'(?:Output|Input)\s*:', response)
        if len(output_spam) > 4:
            score -= 5.0
        elif len(output_spam) > 2:
            score -= 2.0
        
        # Detect "Question:" / "Answer:" spam
        qa_spam = re.findall(r'(?:Question|Answer)\s*:', response)
        if len(qa_spam) > 4:
            score -= 5.0
        
        # ============================================================
        # 6. SENTENCE STRUCTURE QUALITY
        # Analyze sentence length distribution within paragraphs
        # ============================================================
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if len(sentences) >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Good sentence length variety
            if len(sent_lengths) > 2:
                sent_var = sum((x - mean_sent_len) ** 2 for x in sent_lengths) / len(sent_lengths)
                sent_cv = math.sqrt(sent_var) / mean_sent_len if mean_sent_len > 0 else 0
                
                if 0.2 <= sent_cv <= 0.8:
                    score += 3.0  # good variety
                elif sent_cv < 0.2:
                    score += 1.0  # monotonous
                else:
                    score += 1.5  # too variable
            
            # Reasonable average sentence length (10-25 words is ideal)
            if 8 <= mean_sent_len <= 25:
                score += 2.0
            elif 5 <= mean_sent_len < 8 or 25 < mean_sent_len <= 35:
                score += 1.0
        elif len(sentences) == 1:
            # Single sentence - fine for short responses
            if len(response) < 100:
                score += 1.0
        
        # ============================================================
        # 7. OPENING / CLOSING STRUCTURE
        # Check if response has clear beginning and doesn't trail off
        # ============================================================
        
        # Penalize responses that start with lowercase or punctuation (broken start)
        first_char = response[0] if response else ''
        if first_char.islower() or first_char in '.,:;)]}':
            score -= 2.0
        
        # Penalize responses that end abruptly mid-word/sentence
        if response and not response[-1] in '.!?"\')]}:0123456789>*`':
            # Check if it looks truncated
            last_word = response.split()[-1] if response.split() else ''
            if len(last_word) > 1 and not last_word[-1].isalnum():
                pass  # ends with some punctuation-like char
            elif len(response) > 200:
                score -= 2.0  # likely truncated long response
        
        # ============================================================
        # 8. PROPORTIONALITY CHECK
        # Response length should be proportional to query complexity
        # ============================================================
        query_len = len(query.split()) if query else 0
        resp_len = len(response.split())
        
        # Very short response to complex query
        if query_len > 15 and resp_len < 5:
            score -= 3.0
        
        # Response is substantive
        if resp_len >= 20:
            score += 2.0
        elif resp_len >= 10:
            score += 1.0
        
        # ============================================================
        # 9. INDENTATION PATTERN ANALYSIS
        # Detect meaningful indentation as a structural signal
        # ============================================================
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.add(leading_spaces)
        
        if len(indent_levels) >= 3:
            score += 3.0  # multiple indentation levels = hierarchy
        elif len(indent_levels) == 2:
            score += 1.5
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Clamp to 0-10 range
        score = max(0.0, min(10.0, score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This maps the raw score through a curve that emphasizes differences
        midpoint = 5.0
        steepness = 0.5
        normalized = 10.0 / (1.0 + math.exp(-steepness * (score - midpoint)))
        
        return round(normalized, 2)
        
    except Exception:
        return 3.0