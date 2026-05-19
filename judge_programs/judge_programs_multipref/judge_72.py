def judging_function(query, response):
    """
    Evaluates structural organization and formatting using a hierarchical 
    document structure analysis approach. Instead of simple pattern counting,
    this variant builds a structural "skeleton" of the response and evaluates
    the quality of that skeleton based on depth, balance, rhythm, and 
    visual hierarchy principles.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.0
        
        import re
        from collections import Counter
        from math import log2, sqrt
        
        lines = response.split('\n')
        total_chars = len(response)
        total_lines = len(lines)
        
        # === PHASE 1: Build structural skeleton ===
        # Classify each line into a structural role
        ROLE_EMPTY = 'empty'
        ROLE_HEADER = 'header'
        ROLE_LIST_NUMBERED = 'num_list'
        ROLE_LIST_BULLET = 'bullet_list'
        ROLE_SUB_LIST = 'sub_list'
        ROLE_CODE = 'code'
        ROLE_SEPARATOR = 'separator'
        ROLE_TEXT = 'text'
        
        skeleton = []
        
        for line in lines:
            stripped = line.strip()
            leading_spaces = len(line) - len(line.lstrip())
            
            if not stripped:
                skeleton.append((ROLE_EMPTY, 0, ''))
            elif re.match(r'^#{1,6}\s+', stripped):
                level = len(re.match(r'^(#+)', stripped).group(1))
                skeleton.append((ROLE_HEADER, level, stripped))
            elif re.match(r'^\*{3,}$|^-{3,}$|^={3,}$|^_{3,}$', stripped):
                skeleton.append((ROLE_SEPARATOR, 0, stripped))
            elif re.match(r'^```', stripped):
                skeleton.append((ROLE_CODE, 0, stripped))
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                if leading_spaces >= 3:
                    skeleton.append((ROLE_SUB_LIST, 2, stripped))
                else:
                    skeleton.append((ROLE_LIST_NUMBERED, 1, stripped))
            elif re.match(r'^[\-\*\+•◦▪]\s+', stripped):
                if leading_spaces >= 3:
                    skeleton.append((ROLE_SUB_LIST, 2, stripped))
                else:
                    skeleton.append((ROLE_LIST_BULLET, 1, stripped))
            elif stripped.endswith(':') and len(stripped) < 80 and not stripped.endswith('::'):
                # Label-like lines (e.g., "Ingredients:")
                skeleton.append((ROLE_HEADER, 3, stripped))
            elif re.match(r'^\*\*[^*]+\*\*', stripped) and len(stripped) < 100:
                # Bold-start lines acting as pseudo-headers
                skeleton.append((ROLE_HEADER, 4, stripped))
            else:
                skeleton.append((ROLE_TEXT, 0, stripped))
        
        roles = [s[0] for s in skeleton]
        non_empty_roles = [r for r in roles if r != ROLE_EMPTY]
        
        if not non_empty_roles:
            return 0.0
        
        # === PHASE 2: Structural diversity score ===
        # Measure how many different structural elements are used
        role_counts = Counter(non_empty_roles)
        unique_roles = len(role_counts)
        
        # Reward diversity of structural elements (max around 5-6 types)
        diversity_score = min(unique_roles / 4.0, 1.5) * 10  # 0-15
        
        # === PHASE 3: Hierarchical depth analysis ===
        # Measure the depth of the structural hierarchy
        levels_used = set()
        for role, level, text in skeleton:
            if role in (ROLE_HEADER, ROLE_LIST_NUMBERED, ROLE_LIST_BULLET, ROLE_SUB_LIST):
                levels_used.add(level)
        
        hierarchy_depth = len(levels_used)
        hierarchy_score = min(hierarchy_depth / 3.0, 1.0) * 12  # 0-12
        
        # === PHASE 4: Rhythm and visual flow analysis ===
        # Analyze the pattern of structural elements - good documents have rhythm
        # (e.g., header -> text -> list -> text -> header -> text -> list)
        # Bad documents are monotonous (text text text text)
        
        # Create a simplified sequence for rhythm analysis
        simplified = []
        for r in roles:
            if r == ROLE_EMPTY:
                simplified.append('S')  # space/separator
            elif r in (ROLE_HEADER,):
                simplified.append('H')
            elif r in (ROLE_LIST_NUMBERED, ROLE_LIST_BULLET, ROLE_SUB_LIST):
                simplified.append('L')
            elif r == ROLE_CODE:
                simplified.append('C')
            else:
                simplified.append('T')
        
        # Measure bigram diversity in the structural sequence
        if len(simplified) >= 2:
            bigrams = [''.join(simplified[i:i+2]) for i in range(len(simplified)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            unique_bigrams = len(bigram_counts)
            
            # Bigram entropy as a measure of structural rhythm
            bigram_entropy = 0.0
            for count in bigram_counts.values():
                p = count / total_bigrams
                if p > 0:
                    bigram_entropy -= p * log2(p)
            
            # Normalize: max entropy for uniform distribution
            max_entropy = log2(max(unique_bigrams, 1)) if unique_bigrams > 0 else 0
            rhythm_ratio = bigram_entropy / max(max_entropy, 0.001)
            rhythm_score = rhythm_ratio * min(unique_bigrams / 3.0, 1.5) * 8  # 0-12
        else:
            rhythm_score = 0.0
        
        # === PHASE 5: Visual whitespace and breathing room ===
        # Good formatting uses empty lines to create visual separation
        empty_count = roles.count(ROLE_EMPTY)
        non_empty_count = len(non_empty_roles)
        
        if non_empty_count > 0:
            whitespace_ratio = empty_count / non_empty_count
            # Optimal ratio is around 0.2-0.5 (some breathing room)
            if whitespace_ratio < 0.05:
                whitespace_score = 2.0  # Wall of text penalty
            elif whitespace_ratio < 0.1:
                whitespace_score = 5.0
            elif whitespace_ratio <= 0.6:
                whitespace_score = 10.0
            elif whitespace_ratio <= 1.0:
                whitespace_score = 7.0
            else:
                whitespace_score = 4.0  # Too much whitespace
        else:
            whitespace_score = 0.0
        
        # === PHASE 6: Block coherence analysis ===
        # Identify contiguous blocks of same-type content and evaluate balance
        blocks = []
        if skeleton:
            current_type = skeleton[0][0]
            current_len = 1
            for i in range(1, len(skeleton)):
                if skeleton[i][0] == current_type:
                    current_len += 1
                else:
                    blocks.append((current_type, current_len))
                    current_type = skeleton[i][0]
                    current_len = 1
            blocks.append((current_type, current_len))
        
        non_empty_blocks = [(t, l) for t, l in blocks if t != ROLE_EMPTY]
        
        if len(non_empty_blocks) > 1:
            block_lengths = [l for _, l in non_empty_blocks]
            avg_block = sum(block_lengths) / len(block_lengths)
            
            # Penalize very long monotonous blocks
            max_block = max(block_lengths)
            if max_block > 15:
                block_penalty = -5.0
            elif max_block > 10:
                block_penalty = -2.0
            else:
                block_penalty = 0.0
            
            # Reward having multiple well-sized blocks
            num_blocks = len(non_empty_blocks)
            block_variety_score = min(num_blocks / 5.0, 1.5) * 6 + block_penalty  # up to 9
        else:
            block_variety_score = 0.0
        
        # === PHASE 7: Inline formatting richness ===
        # Check for bold, italic, inline code, etc.
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        inline_code_count = len(re.findall(r'`[^`]+`', response))
        colon_structure = len(re.findall(r'\w+\s*:', response))
        
        inline_elements = bold_count + italic_count + inline_code_count
        # Reward moderate use of inline formatting
        if inline_elements == 0:
            inline_score = 0.0
        elif inline_elements <= 3:
            inline_score = 4.0
        elif inline_elements <= 10:
            inline_score = 8.0
        elif inline_elements <= 20:
            inline_score = 6.0
        else:
            inline_score = 4.0  # Overformatted
        
        # === PHASE 8: Opening and closing structure ===
        # Good responses often have an intro paragraph and a conclusion
        intro_score = 0.0
        
        # Check if response starts with a text paragraph before structured content
        first_content_roles = [r for r in roles if r != ROLE_EMPTY][:3]
        if first_content_roles and first_content_roles[0] == ROLE_TEXT:
            intro_score += 3.0
        
        # Check if there's structural content in the middle
        has_structure = any(r in (ROLE_HEADER, ROLE_LIST_NUMBERED, ROLE_LIST_BULLET) 
                          for r in non_empty_roles)
        if has_structure:
            intro_score += 2.0
        
        # === PHASE 9: Paragraph length analysis ===
        # Analyze text blocks for appropriate paragraph sizing
        text_blocks = []
        current_text = []
        for role, level, text in skeleton:
            if role == ROLE_TEXT:
                current_text.append(text)
            else:
                if current_text:
                    text_blocks.append(' '.join(current_text))
                    current_text = []
        if current_text:
            text_blocks.append(' '.join(current_text))
        
        para_score = 0.0
        if text_blocks:
            para_lengths = [len(tb) for tb in text_blocks]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            
            # Ideal paragraph length: 100-400 characters
            if 80 <= avg_para_len <= 400:
                para_score = 8.0
            elif 40 <= avg_para_len <= 600:
                para_score = 5.0
            elif avg_para_len > 600:
                para_score = 2.0  # Too long paragraphs
            else:
                para_score = 4.0  # Very short
            
            # Multiple well-sized paragraphs is better
            if len(text_blocks) >= 2:
                para_score += 2.0
        else:
            # All structured content, no text blocks - still okay if lists/headers
            if has_structure:
                para_score = 5.0
        
        # === PHASE 10: Proportionality check ===
        # For the response length, is the amount of structure proportional?
        if total_chars > 0:
            structured_lines = sum(1 for r in non_empty_roles 
                                  if r in (ROLE_HEADER, ROLE_LIST_NUMBERED, 
                                          ROLE_LIST_BULLET, ROLE_SUB_LIST))
            structure_density = structured_lines / max(len(non_empty_roles), 1)
            
            # For longer responses, we expect more structure
            length_factor = min(total_chars / 200.0, 3.0)
            
            if total_chars < 150:
                # Short responses don't need much structure
                proportionality_score = 5.0
            elif structure_density > 0.1:
                proportionality_score = min(structure_density * length_factor * 5, 10.0)
            else:
                # Long response with no structure = wall of text
                proportionality_score = max(0, 5.0 - length_factor)
        else:
            proportionality_score = 0.0
        
        # === COMBINE SCORES ===
        total = (
            diversity_score +       # 0-15: variety of structural elements
            hierarchy_score +       # 0-12: depth of hierarchy
            rhythm_score +          # 0-12: structural rhythm
            whitespace_score +      # 0-10: visual breathing room
            block_variety_score +   # 0-9: block balance
            inline_score +          # 0-8: inline formatting
            intro_score +           # 0-5: opening structure
            para_score +            # 0-10: paragraph sizing
            proportionality_score   # 0-10: structure proportional to length
        )
        
        # Normalize to 0-100 range (max theoretical ~91)
        normalized = max(0.0, min(100.0, total * 1.1))
        
        # Apply a slight sigmoid-like compression to spread scores
        # This helps discriminate in the middle range
        midpoint = 45.0
        steepness = 0.06
        compressed = 100.0 / (1.0 + 2.718 ** (-steepness * (normalized - midpoint)))
        
        return round(compressed, 2)
        
    except Exception:
        return 25.0