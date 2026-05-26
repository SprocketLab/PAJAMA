def judging_function(query, response):
    """
    Evaluate structural organization and formatting of an LLM response.
    
    Focuses on:
    - Use of formatting elements (headers, lists, bullet points, paragraphs)
    - Logical paragraph structure and separation
    - Avoidance of wall-of-text
    - Clear topic sentences and transitions
    - Effective use of whitespace
    - Logical grouping of related ideas
    
    Returns a score where HIGHER = BETTER structural organization.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        import re
        import math
        from collections import Counter
        
        score = 0.0
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # ============================================================
        # 1. PARAGRAPH STRUCTURE (0-15 points)
        # ============================================================
        paragraphs = re.split(r'\n\s*\n', response.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Reward multiple paragraphs (shows organization)
        if num_paragraphs >= 4:
            para_score = 15.0
        elif num_paragraphs == 3:
            para_score = 13.0
        elif num_paragraphs == 2:
            para_score = 10.0
        elif num_paragraphs == 1:
            # Single paragraph - check if it's a wall of text
            if len(response) > 500:
                para_score = 2.0  # Wall of text penalty
            elif len(response) > 300:
                para_score = 4.0
            else:
                para_score = 6.0  # Short single paragraph is okay
        else:
            para_score = 0.0
        
        score += para_score
        
        # ============================================================
        # 2. LIST/ENUMERATION DETECTION (0-20 points)
        # ============================================================
        # Numbered lists: "1.", "2.", etc.
        numbered_items = re.findall(r'(?:^|\n)\s*(\d+)[.)]\s+\S', response)
        # Bullet points: "- ", "* ", "• "
        bullet_items = re.findall(r'(?:^|\n)\s*[-*•]\s+\S', response)
        # Letter lists: "a)", "b)", etc.
        letter_items = re.findall(r'(?:^|\n)\s*[a-zA-Z][.)]\s+\S', response)
        
        total_list_items = len(numbered_items) + len(bullet_items) + len(letter_items)
        
        if total_list_items >= 5:
            list_score = 20.0
        elif total_list_items >= 3:
            list_score = 16.0
        elif total_list_items >= 2:
            list_score = 12.0
        elif total_list_items == 1:
            list_score = 6.0
        else:
            list_score = 0.0
        
        score += list_score
        
        # ============================================================
        # 3. HEADERS / SECTION MARKERS (0-10 points)
        # ============================================================
        # Markdown headers
        md_headers = re.findall(r'(?:^|\n)#{1,6}\s+\S', response)
        # Bold text as headers (standalone bold lines)
        bold_headers = re.findall(r'(?:^|\n)\s*\*\*[^*]+\*\*\s*(?:\n|$)', response)
        # Colon-terminated labels (e.g., "Step 1:", "Note:", "Important:")
        colon_headers = re.findall(r'(?:^|\n)\s*[A-Z][A-Za-z\s]{1,30}:\s', response)
        
        total_headers = len(md_headers) + len(bold_headers) + len(colon_headers)
        
        if total_headers >= 3:
            header_score = 10.0
        elif total_headers >= 2:
            header_score = 7.0
        elif total_headers == 1:
            header_score = 4.0
        else:
            header_score = 0.0
        
        score += header_score
        
        # ============================================================
        # 4. SENTENCE STRUCTURE QUALITY (0-15 points)
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences > 0:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Variance in sentence length (indicates varied structure)
            if len(sent_lengths) > 1:
                mean_len = avg_sent_len
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0
            
            # Good average sentence length: 10-25 words
            if 10 <= avg_sent_len <= 25:
                sent_score = 8.0
            elif 7 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
                sent_score = 5.0
            else:
                sent_score = 2.0
            
            # Reward some variation in sentence length (not monotonous)
            if std_dev > 3:
                sent_score += 4.0
            elif std_dev > 1.5:
                sent_score += 2.0
            
            # Bonus for having enough sentences (shows developed content)
            if num_sentences >= 5:
                sent_score += 3.0
            elif num_sentences >= 3:
                sent_score += 1.5
            
            sent_score = min(sent_score, 15.0)
        else:
            sent_score = 0.0
        
        score += sent_score
        
        # ============================================================
        # 5. TRANSITION / CONNECTIVE WORDS (0-10 points)
        # ============================================================
        transition_words = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bfinally\b', r'\bin addition\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bas a result\b', r'\btherefore\b', r'\bconsequently\b',
            r'\bmeanwhile\b', r'\bnevertheless\b', r'\bin summary\b',
            r'\bto begin\b', r'\bin conclusion\b', r'\bspecifically\b',
            r'\bimportantly\b', r'\bnotably\b', r'\balternatively\b',
            r'\bhere\b', r'\bnow\b', r'\bremember\b', r'\blet\'s\b',
            r'\balso\b', r'\bthat said\b', r'\bwhile\b'
        ]
        
        response_lower = response.lower()
        transition_count = 0
        for tw in transition_words:
            matches = re.findall(tw, response_lower)
            transition_count += len(matches)
        
        if transition_count >= 5:
            transition_score = 10.0
        elif transition_count >= 3:
            transition_score = 7.0
        elif transition_count >= 2:
            transition_score = 5.0
        elif transition_count >= 1:
            transition_score = 3.0
        else:
            transition_score = 0.0
        
        score += transition_score
        
        # ============================================================
        # 6. WALL-OF-TEXT PENALTY (0 to -10 points)
        # ============================================================
        # Check for very long unbroken text blocks
        wall_penalty = 0.0
        
        for para in paragraphs:
            words_in_para = len(para.split())
            if words_in_para > 150:
                wall_penalty -= 5.0
            elif words_in_para > 100:
                wall_penalty -= 3.0
            elif words_in_para > 80:
                wall_penalty -= 1.0
        
        # If entire response is one long block with no line breaks
        if num_paragraphs == 1 and len(response.split()) > 100:
            wall_penalty -= 3.0
        
        wall_penalty = max(wall_penalty, -10.0)
        score += wall_penalty
        
        # ============================================================
        # 7. OPENING / CLOSING STRUCTURE (0-10 points)
        # ============================================================
        opening_score = 0.0
        
        # Check for a clear opening that acknowledges the query
        first_para = paragraphs[0] if paragraphs else ""
        first_para_lower = first_para.lower()
        
        # Empathetic / acknowledging openings
        opening_patterns = [
            r'^i\s+(can\s+)?(see|hear|understand|sense)',
            r'^i\'m\s+(genuinely\s+|really\s+|truly\s+)?sorry',
            r'^it\'s\s+(completely\s+|totally\s+|perfectly\s+)?(understandable|okay|fine|natural)',
            r'^(that\'s|this\s+is)\s+',
            r'^(imagine|think\s+of|consider|let\'s)',
            r'^(hey|hello|hi)\s+there',
            r'^(great|good|excellent)\s+(question|point)',
            r'^(here|below)\s+(are|is)',
            r'^(to|in\s+order\s+to)\s+'
        ]
        
        for pat in opening_patterns:
            if re.search(pat, first_para_lower):
                opening_score += 3.0
                break
        
        # Check if opening paragraph is reasonably sized (not too long)
        first_para_words = len(first_para.split())
        if 10 <= first_para_words <= 60:
            opening_score += 3.0
        elif first_para_words < 10:
            opening_score += 1.0
        
        # Check for concluding signals
        last_para = paragraphs[-1] if paragraphs else ""
        last_para_lower = last_para.lower()
        closing_patterns = [
            r'remember', r'in (summary|conclusion)', r'overall', r'finally',
            r'don\'t\s+(hesitate|forget|be\s+afraid)', r'feel\s+free',
            r'hope\s+this', r'good\s+luck', r'take\s+care',
            r'keep\s+in\s+mind', r'most\s+importantly'
        ]
        
        for pat in closing_patterns:
            if re.search(pat, last_para_lower):
                opening_score += 4.0
                break
        
        opening_score = min(opening_score, 10.0)
        score += opening_score
        
        # ============================================================
        # 8. FORMATTING DIVERSITY BONUS (0-10 points)
        # ============================================================
        formatting_elements = 0
        
        if num_paragraphs >= 2:
            formatting_elements += 1
        if total_list_items >= 2:
            formatting_elements += 1
        if total_headers >= 1:
            formatting_elements += 1
        if transition_count >= 2:
            formatting_elements += 1
        # Check for emphasis markers (bold, italic)
        if re.search(r'\*\*\S', response) or re.search(r'\*\S', response):
            formatting_elements += 1
        # Check for code blocks or special formatting
        if re.search(r'```', response) or re.search(r'`\S', response):
            formatting_elements += 1
        
        diversity_score = min(formatting_elements * 2.0, 10.0)
        score += diversity_score
        
        # ============================================================
        # 9. RESPONSE LENGTH APPROPRIATENESS (0-5 points)
        # ============================================================
        response_words = len(response.split())
        query_words = len(query.split())
        
        # Longer queries generally warrant longer, well-structured responses
        if response_words >= 50 and response_words <= 500:
            length_score = 5.0
        elif response_words >= 30:
            length_score = 3.0
        elif response_words >= 20:
            length_score = 2.0
        else:
            length_score = 1.0
        
        score += length_score
        
        # ============================================================
        # 10. COHERENT LINE BREAKS (0-5 points)
        # ============================================================
        # Check that line breaks occur at logical points
        newline_count = response.count('\n')
        
        if newline_count >= 3 and newline_count <= 20:
            linebreak_score = 5.0
        elif newline_count >= 1:
            linebreak_score = 3.0
        elif response_words < 60:
            linebreak_score = 2.0  # Short responses don't need many breaks
        else:
            linebreak_score = 0.0
        
        score += linebreak_score
        
        # ============================================================
        # NORMALIZE to 1-5 scale
        # ============================================================
        # Max theoretical score: 15+20+10+15+10+0+10+10+5+5 = 100
        # Min theoretical score: 0 + (-10) = -10
        # Practical range: ~5 to ~80
        
        # Normalize to 1-5
        raw_max = 85.0
        raw_min = 0.0
        
        # Clamp
        score = max(score, raw_min)
        score = min(score, raw_max)
        
        # Map to 1-5
        normalized = 1.0 + (score - raw_min) / (raw_max - raw_min) * 4.0
        
        # Round to 1 decimal
        normalized = round(normalized, 2)
        
        # Final clamp
        normalized = max(1.0, min(5.0, normalized))
        
        return normalized
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5