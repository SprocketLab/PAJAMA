def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant focuses on a HIERARCHICAL STRUCTURE ANALYSIS approach:
    - Analyzes the document as a tree structure (headers -> sections -> paragraphs -> sentences)
    - Measures structural depth and balance
    - Evaluates formatting diversity and appropriateness
    - Scores visual rhythm (alternation between different element types)
    - Considers response length appropriateness relative to query complexity
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # ============================================================
        # COMPONENT 1: Structural Element Classification
        # Classify each line into a type to analyze document "rhythm"
        # ============================================================
        
        LINE_HEADER = 'header'
        LINE_BULLET = 'bullet'
        LINE_NUMBERED = 'numbered'
        LINE_BOLD_PHRASE = 'bold_phrase'
        LINE_CODE = 'code'
        LINE_SEPARATOR = 'separator'
        LINE_TEXT = 'text'
        LINE_EMPTY = 'empty'
        
        def classify_line(line):
            stripped = line.strip()
            if not stripped:
                return LINE_EMPTY
            if re.match(r'^#{1,6}\s+', stripped):
                return LINE_HEADER
            if re.match(r'^\*{3,}$|^-{3,}$|^={3,}$', stripped):
                return LINE_SEPARATOR
            if re.match(r'^```', stripped):
                return LINE_CODE
            if re.match(r'^(\d+[\.\)]\s+|\([a-z]\)\s+|[a-z][\.\)]\s+)', stripped, re.IGNORECASE):
                return LINE_NUMBERED
            if re.match(r'^[\-\*\+•◦▪▸►]\s+', stripped):
                return LINE_BULLET
            if re.match(r'^\*\*[^*]+\*\*', stripped) and len(stripped) < 80:
                return LINE_BOLD_PHRASE
            return LINE_TEXT
        
        line_types = [classify_line(l) for l in lines]
        non_empty_types = [t for t in line_types if t != LINE_EMPTY]
        
        type_counts = Counter(non_empty_types)
        
        # ============================================================
        # COMPONENT 2: Structural Diversity Score
        # Measures how many different structural elements are used
        # ============================================================
        
        formatting_elements = {LINE_HEADER, LINE_BULLET, LINE_NUMBERED, LINE_BOLD_PHRASE, LINE_CODE, LINE_SEPARATOR}
        used_formatting = sum(1 for t in formatting_elements if type_counts.get(t, 0) > 0)
        
        # Score: 0-15 points
        diversity_score = min(used_formatting * 3.5, 15.0)
        
        # ============================================================
        # COMPONENT 3: Visual Rhythm Analysis
        # Good documents alternate between element types creating "rhythm"
        # Bad documents are monotonous (all same type)
        # ============================================================
        
        def compute_rhythm_score(types_seq):
            if len(types_seq) < 3:
                return 0.0
            transitions = 0
            for i in range(1, len(types_seq)):
                if types_seq[i] != types_seq[i-1]:
                    transitions += 1
            transition_rate = transitions / (len(types_seq) - 1)
            # Ideal rhythm has moderate transitions (0.3-0.7)
            if transition_rate < 0.1:
                return 1.0
            elif transition_rate < 0.3:
                return 4.0
            elif transition_rate <= 0.7:
                return 8.0
            else:
                return 5.0  # Too chaotic
        
        rhythm_score = compute_rhythm_score(non_empty_types)  # 0-8 points
        
        # ============================================================
        # COMPONENT 4: Hierarchical Depth Analysis
        # Measures if the document has proper nesting/hierarchy
        # ============================================================
        
        header_levels = []
        for line in lines:
            stripped = line.strip()
            m = re.match(r'^(#{1,6})\s+', stripped)
            if m:
                header_levels.append(len(m.group(1)))
        
        # Check for sub-items (indented bullets/numbers under items)
        indented_items = 0
        for line in lines:
            if re.match(r'^(\s{2,}|\t+)[\-\*\+•]\s+', line) or re.match(r'^(\s{2,}|\t+)\d+[\.\)]\s+', line):
                indented_items += 1
        
        hierarchy_depth = 0
        if header_levels:
            unique_levels = len(set(header_levels))
            hierarchy_depth += min(unique_levels * 2, 6)
        if indented_items > 0:
            hierarchy_depth += min(indented_items, 4)
        
        hierarchy_score = min(hierarchy_depth, 10.0)  # 0-10 points
        
        # ============================================================
        # COMPONENT 5: Paragraph Structure Quality
        # Analyzes paragraph lengths, consistency, and spacing
        # ============================================================
        
        # Split into paragraphs by empty lines
        paragraphs = []
        current_para = []
        for line in lines:
            if line.strip() == '':
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        num_paragraphs = len(paragraphs)
        
        para_score = 0.0
        if num_paragraphs == 1:
            # Single block - check if it's long (wall of text penalty)
            word_count = len(response.split())
            if word_count > 100:
                para_score = -3.0  # Wall of text penalty
            elif word_count > 50:
                para_score = 0.0
            else:
                para_score = 3.0  # Short response, single para is fine
        elif num_paragraphs <= 3:
            para_score = 5.0
        elif num_paragraphs <= 8:
            para_score = 8.0
        else:
            para_score = 6.0  # Too fragmented
        
        # Check paragraph length variance (consistent lengths are better)
        if num_paragraphs >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_len = sum(para_lengths) / len(para_lengths)
            if avg_len > 0:
                cv = (sum((l - avg_len)**2 for l in para_lengths) / len(para_lengths))**0.5 / avg_len
                # Moderate variation is good (not all identical, not wildly different)
                if cv < 0.3:
                    para_score += 2.0
                elif cv < 0.8:
                    para_score += 3.0
                elif cv < 1.5:
                    para_score += 1.0
                # else: too variable, no bonus
        
        para_score = max(min(para_score, 12.0), -3.0)  # -3 to 12 points
        
        # ============================================================
        # COMPONENT 6: Inline Formatting Richness
        # Bold, italic, code spans, links within text
        # ============================================================
        
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        code_span_count = len(re.findall(r'`[^`]+`', response))
        
        inline_elements = bold_count + italic_count + code_span_count
        
        if inline_elements == 0:
            inline_score = 0.0
        elif inline_elements <= 3:
            inline_score = 4.0
        elif inline_elements <= 10:
            inline_score = 7.0
        elif inline_elements <= 25:
            inline_score = 10.0
        else:
            inline_score = 8.0  # Overformatted slightly
        
        # 0-10 points
        
        # ============================================================
        # COMPONENT 7: Opening and Closing Structure
        # Good responses have an intro and conclusion
        # ============================================================
        
        structure_bookend_score = 0.0
        
        if num_paragraphs >= 2:
            first_para = paragraphs[0].strip()
            # Check if first paragraph is introductory (not a list item or header)
            first_type = classify_line(first_para.split('\n')[0])
            if first_type == LINE_TEXT:
                first_words = len(first_para.split())
                if 10 <= first_words <= 80:
                    structure_bookend_score += 4.0
                elif first_words < 10:
                    structure_bookend_score += 1.5
            
            # Check if there's a concluding paragraph
            last_para = paragraphs[-1].strip()
            last_type = classify_line(last_para.split('\n')[0])
            if last_type == LINE_TEXT:
                last_words = len(last_para.split())
                if 8 <= last_words <= 60:
                    structure_bookend_score += 3.0
        
        structure_bookend_score = min(structure_bookend_score, 7.0)  # 0-7 points
        
        # ============================================================
        # COMPONENT 8: List Quality Analysis
        # If lists are present, check consistency and completeness
        # ============================================================
        
        list_items = type_counts.get(LINE_NUMBERED, 0) + type_counts.get(LINE_BULLET, 0)
        
        list_quality_score = 0.0
        if list_items >= 2:
            # Lists exist - good
            list_quality_score += 3.0
            
            # Check if numbered items are sequential
            numbered_values = []
            for line in lines:
                m = re.match(r'^(\d+)[\.\)]\s+', line.strip())
                if m:
                    numbered_values.append(int(m.group(1)))
            
            if numbered_values:
                # Check if sequential
                is_sequential = all(numbered_values[i] <= numbered_values[i+1] 
                                   for i in range(len(numbered_values)-1))
                if is_sequential:
                    list_quality_score += 2.0
            
            # Check consistency: are list items similar in length?
            list_line_lengths = []
            for i, line in enumerate(lines):
                if line_types[i] in (LINE_BULLET, LINE_NUMBERED):
                    list_line_lengths.append(len(line.strip()))
            
            if len(list_line_lengths) >= 3:
                avg_ll = sum(list_line_lengths) / len(list_line_lengths)
                if avg_ll > 0:
                    ll_cv = (sum((l - avg_ll)**2 for l in list_line_lengths) / len(list_line_lengths))**0.5 / avg_ll
                    if ll_cv < 0.5:
                        list_quality_score += 3.0  # Consistent list items
                    elif ll_cv < 1.0:
                        list_quality_score += 1.5
        
        list_quality_score = min(list_quality_score, 8.0)  # 0-8 points
        
        # ============================================================
        # COMPONENT 9: Whitespace Effectiveness
        # Proper use of blank lines to separate sections
        # ============================================================
        
        empty_line_count = sum(1 for t in line_types if t == LINE_EMPTY)
        total_lines = len(lines)
        
        if total_lines > 0:
            empty_ratio = empty_line_count / total_lines
        else:
            empty_ratio = 0
        
        whitespace_score = 0.0
        if total_lines <= 3:
            whitespace_score = 3.0  # Short response, whitespace not critical
        elif empty_ratio == 0:
            whitespace_score = 0.0  # No spacing at all
        elif empty_ratio < 0.05:
            whitespace_score = 2.0
        elif empty_ratio < 0.15:
            whitespace_score = 5.0
        elif empty_ratio < 0.3:
            whitespace_score = 7.0
        elif empty_ratio < 0.5:
            whitespace_score = 5.0
        else:
            whitespace_score = 2.0  # Too much whitespace
        
        # 0-7 points
        
        # ============================================================
        # COMPONENT 10: Response Length Appropriateness
        # Relative to query complexity
        # ============================================================
        
        query_words = len(query.split()) if query else 5
        response_words = len(response.split())
        
        # Estimate query complexity
        question_marks = query.count('?')
        query_complexity = min(query_words / 10.0, 3.0) + min(question_marks, 2)
        
        # For complex queries, longer structured responses are better
        length_score = 0.0
        if response_words < 20:
            length_score = 1.0  # Very short
        elif response_words < 50:
            length_score = 3.0
        elif response_words < 150:
            length_score = 5.0
        else:
            length_score = 4.0  # Long is ok but not automatically better
        
        # Bonus for longer responses that ARE well-structured
        if response_words > 80 and used_formatting >= 2:
            length_score += 2.0
        
        length_score = min(length_score, 7.0)  # 0-7 points
        
        # ============================================================
        # COMPONENT 11: Sentence-Level Quality within Paragraphs
        # Check that text paragraphs have reasonable sentence structure
        # ============================================================
        
        sentences = re.split(r'[.!?]+\s+', response)
        sentence_count = len(sentences)
        
        sentence_score = 0.0
        if sentence_count >= 3:
            sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
            if sent_lengths:
                avg_sent = sum(sent_lengths) / len(sent_lengths)
                if 8 <= avg_sent <= 25:
                    sentence_score = 4.0  # Good sentence length
                elif 5 <= avg_sent <= 35:
                    sentence_score = 2.5
                else:
                    sentence_score = 1.0
        else:
            sentence_score = 1.5
        
        # 0-4 points
        
        # ============================================================
        # COMPONENT 12: Logical Grouping Signal Detection
        # Look for topic sentences, transition patterns between sections
        # ============================================================
        
        grouping_score = 0.0
        
        # Check for colon-terminated introductions (e.g., "Here are the steps:")
        colon_intros = len(re.findall(r'[a-zA-Z]\:\s*$', response, re.MULTILINE))
        if colon_intros > 0:
            grouping_score += min(colon_intros * 1.5, 4.0)
        
        # Check for labeled sections (bold labels followed by content)
        labeled_sections = len(re.findall(r'\*\*[^*]+\*\*[\s:]+\S', response))
        if labeled_sections >= 2:
            grouping_score += min(labeled_sections * 0.8, 4.0)
        
        grouping_score = min(grouping_score, 6.0)  # 0-6 points
        
        # ============================================================
        # AGGREGATE SCORE
        # ============================================================
        
        # Max theoretical: 15 + 8 + 10 + 12 + 10 + 7 + 8 + 7 + 7 + 4 + 6 = 94
        raw_score = (
            diversity_score +       # 0-15
            rhythm_score +          # 0-8
            hierarchy_score +       # 0-10
            para_score +            # -3 to 12
            inline_score +          # 0-10
            structure_bookend_score + # 0-7
            list_quality_score +    # 0-8
            whitespace_score +      # 0-7
            length_score +          # 0-7
            sentence_score +        # 0-4
            grouping_score          # 0-6
        )
        
        # Normalize to 0-10 scale
        # Practical range is roughly 0-80
        normalized = max(0.0, min(raw_score / 8.0, 10.0))
        
        return round(normalized, 3)
    
    except Exception:
        return 2.0