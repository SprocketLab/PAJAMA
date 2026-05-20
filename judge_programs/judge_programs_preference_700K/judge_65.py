def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    Returns a score where HIGHER = BETTER quality.
    
    This variant focuses on:
    - Detection of formatting elements (headers, lists, code blocks, etc.)
    - Paragraph structure and whitespace usage
    - Logical segmentation and topic transitions
    - Readability metrics based on sentence/paragraph length variance
    - Penalization of wall-of-text and poorly organized content
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
        
        score = 0.0
        resp_len = len(response)
        
        # ============================================================
        # 1. BASIC LENGTH SCORING (0-8 points)
        # Responses that are too short lack structure opportunity
        # ============================================================
        word_count = len(response.split())
        if word_count < 5:
            return 1.0
        
        # Length reward with diminishing returns
        if word_count >= 20:
            length_score = min(8.0, 2.0 + math.log2(word_count / 20) * 2.5)
        elif word_count >= 10:
            length_score = 1.5
        else:
            length_score = 0.5
        score += length_score
        
        # ============================================================
        # 2. PARAGRAPH STRUCTURE (0-15 points)
        # Good responses break content into digestible paragraphs
        # ============================================================
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response) if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Single block of text penalty
        lines = [l for l in response.split('\n') if l.strip()]
        num_lines = len(lines)
        
        if num_paragraphs >= 3:
            para_score = min(8.0, 3.0 + num_paragraphs * 0.8)
        elif num_paragraphs == 2:
            para_score = 3.0
        elif num_lines > 1:
            para_score = 1.5
        else:
            para_score = 0.0
        
        # Reward balanced paragraph lengths (not one huge + tiny ones)
        if num_paragraphs >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            if avg_para_len > 0:
                variance = sum((l - avg_para_len)**2 for l in para_lengths) / len(para_lengths)
                cv = math.sqrt(variance) / avg_para_len if avg_para_len > 0 else 0
                # Lower CV = more balanced = better (up to 4 points)
                balance_bonus = max(0, 4.0 - cv * 2.5)
                para_score += balance_bonus
            
            # Ideal paragraph size bonus (30-80 words per paragraph)
            if 20 <= avg_para_len <= 100:
                para_score += 2.0
            elif 10 <= avg_para_len <= 150:
                para_score += 1.0
        
        score += min(15.0, para_score)
        
        # ============================================================
        # 3. LIST DETECTION (0-12 points)
        # Numbered lists, bullet points, dashes
        # ============================================================
        bullet_patterns = [
            r'^\s*[-•·]\s+\S',          # bullet points
            r'^\s*\*\s+\S',             # asterisk bullets
            r'^\s*\d+[\.\)]\s+\S',      # numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+\S', # lettered lists
        ]
        
        list_items = 0
        for line in lines:
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    list_items += 1
                    break
        
        if list_items >= 5:
            list_score = 10.0
        elif list_items >= 3:
            list_score = 7.0
        elif list_items >= 2:
            list_score = 5.0
        elif list_items == 1:
            list_score = 2.0
        else:
            list_score = 0.0
        
        # Bonus for consistent list formatting
        if list_items >= 2:
            numbered = sum(1 for l in lines if re.match(r'^\s*\d+[\.\)]\s', l))
            bulleted = sum(1 for l in lines if re.match(r'^\s*[-•·\*]\s', l))
            if numbered >= 2 or bulleted >= 2:
                list_score += 2.0  # consistency bonus
        
        score += min(12.0, list_score)
        
        # ============================================================
        # 4. HEADERS AND SECTION MARKERS (0-10 points)
        # Markdown headers, bold text as headers, ALL CAPS headers
        # ============================================================
        header_patterns = [
            r'^#{1,6}\s+\S',                    # Markdown headers
            r'^\*\*[^*]+\*\*\s*$',              # Bold-only lines (headers)
            r'^__[^_]+__\s*$',                   # Underline-bold headers
            r'^[A-Z][A-Z\s]{3,}:?\s*$',         # ALL CAPS headers
            r'^\*\*[^*]+\*\*:',                  # Bold followed by colon
        ]
        
        header_count = 0
        for line in lines:
            for pattern in header_patterns:
                if re.match(pattern, line.strip()):
                    header_count += 1
                    break
        
        if header_count >= 4:
            header_score = 10.0
        elif header_count >= 2:
            header_score = 6.0 + header_count * 0.5
        elif header_count == 1:
            header_score = 3.0
        else:
            header_score = 0.0
        
        score += min(10.0, header_score)
        
        # ============================================================
        # 5. CODE BLOCK FORMATTING (0-6 points)
        # Proper use of code blocks when code is present
        # ============================================================
        code_block_count = len(re.findall(r'```', response))
        inline_code_count = len(re.findall(r'`[^`]+`', response))
        
        code_score = 0.0
        if code_block_count >= 2:  # at least one complete code block
            code_score += 4.0
            # Language specification bonus
            if re.search(r'```\w+', response):
                code_score += 2.0
        elif inline_code_count >= 1:
            code_score += 2.0
        
        score += min(6.0, code_score)
        
        # ============================================================
        # 6. BOLD/ITALIC EMPHASIS (0-6 points)
        # Appropriate use of emphasis for key terms
        # ============================================================
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        
        emphasis_items = bold_count + italic_count
        if emphasis_items >= 4:
            emphasis_score = 5.0
        elif emphasis_items >= 2:
            emphasis_score = 3.0
        elif emphasis_items >= 1:
            emphasis_score = 1.5
        else:
            emphasis_score = 0.0
        
        # Penalize over-emphasis (everything bold is nothing bold)
        if bold_count > 15:
            emphasis_score -= 2.0
        
        score += max(0.0, min(6.0, emphasis_score))
        
        # ============================================================
        # 7. SENTENCE STRUCTURE QUALITY (0-10 points)
        # Varied sentence lengths, proper punctuation, topic sentences
        # ============================================================
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        sent_score = 0.0
        if num_sentences >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Reward moderate sentence length (10-25 words average)
            if 8 <= avg_sent_len <= 30:
                sent_score += 3.0
            elif 5 <= avg_sent_len <= 40:
                sent_score += 1.5
            
            # Reward sentence length variety
            if len(sent_lengths) >= 3:
                variance = sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                if 3 <= std_dev <= 15:
                    sent_score += 3.0  # good variety
                elif 1 <= std_dev < 3:
                    sent_score += 1.0  # some variety
            
            # Multiple sentences bonus
            sent_score += min(4.0, num_sentences * 0.4)
        elif num_sentences >= 1:
            sent_score += 1.0
        
        score += min(10.0, sent_score)
        
        # ============================================================
        # 8. TRANSITION WORDS AND LOGICAL CONNECTORS (0-8 points)
        # Words that indicate logical flow and organization
        # ============================================================
        transition_words = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bfinally\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bhowever\b', r'\bin addition\b', r'\bon the other hand\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\btherefore\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\balternatively\b',
            r'\badditionally\b', r'\bimportantly\b', r'\bnotably\b',
            r'\bin contrast\b', r'\bsimilarly\b', r'\blikewise\b',
            r'\bmeanwhile\b', r'\bsubsequently\b', r'\bultimately\b',
        ]
        
        resp_lower = response.lower()
        transition_count = 0
        for tw in transition_words:
            transition_count += len(re.findall(tw, resp_lower))
        
        if transition_count >= 5:
            transition_score = 7.0
        elif transition_count >= 3:
            transition_score = 5.0
        elif transition_count >= 2:
            transition_score = 3.0
        elif transition_count >= 1:
            transition_score = 1.5
        else:
            transition_score = 0.0
        
        score += min(8.0, transition_score)
        
        # ============================================================
        # 9. WALL-OF-TEXT PENALTY (0 to -12 points)
        # Penalize responses that are long but have no structure
        # ============================================================
        wall_penalty = 0.0
        
        # Long single paragraph
        if num_paragraphs == 1 and word_count > 80:
            wall_penalty -= min(8.0, (word_count - 80) * 0.04)
        
        # Very long lines without breaks
        max_line_len = max((len(l) for l in lines), default=0)
        if max_line_len > 500 and num_lines <= 2:
            wall_penalty -= 4.0
        
        # No formatting at all in a long response
        has_any_formatting = (list_items > 0 or header_count > 0 or 
                             code_block_count > 0 or bold_count > 0 or
                             num_paragraphs > 1)
        if word_count > 100 and not has_any_formatting:
            wall_penalty -= 3.0
        
        score += wall_penalty
        
        # ============================================================
        # 10. OPENING/CLOSING STRUCTURE (0-5 points)
        # Good responses often have an intro and conclusion
        # ============================================================
        structure_score = 0.0
        
        # Check for introductory/framing sentence
        first_sentence = sentences[0] if sentences else ""
        intro_patterns = [
            r'\b(here|let me|i\'ll|this|there are|the answer|to answer|great question)\b',
            r'\b(overview|summary|explanation|guide|steps|following)\b',
        ]
        for ip in intro_patterns:
            if re.search(ip, first_sentence.lower()):
                structure_score += 2.0
                break
        
        # Check for concluding element
        if sentences:
            last_part = response[-200:].lower()
            conclusion_patterns = [
                r'\b(in summary|in conclusion|overall|to summarize|hope this helps|let me know)\b',
                r'\b(in short|bottom line|key takeaway|final)\b',
            ]
            for cp in conclusion_patterns:
                if re.search(cp, last_part):
                    structure_score += 2.0
                    break
        
        # Colon usage for introducing lists/explanations
        colon_lines = sum(1 for l in lines if ':' in l and len(l.split(':')[0].split()) <= 8)
        if colon_lines >= 2:
            structure_score += 1.0
        
        score += min(5.0, structure_score)
        
        # ============================================================
        # 11. RESPONSE COMPLETENESS SIGNAL (0-5 points)
        # Check if response seems complete vs truncated
        # ============================================================
        completeness_score = 0.0
        
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        if last_char in '.!?)"\':':
            completeness_score += 2.0
        elif last_char == '…' or response.rstrip().endswith('...'):
            completeness_score += 0.5
        
        # Complete sentences ratio
        if num_sentences >= 2:
            completeness_score += 2.0
        
        # Balanced code blocks (opened and closed)
        if code_block_count % 2 == 0 and code_block_count > 0:
            completeness_score += 1.0
        
        score += min(5.0, completeness_score)
        
        # ============================================================
        # 12. INFORMATION DENSITY AND GROUPING (0-5 points)
        # Reward responses where related info is grouped together
        # ============================================================
        grouping_score = 0.0
        
        # Multiple distinct sections (indicated by headers or numbered items)
        distinct_sections = max(num_paragraphs, header_count + 1, 
                               1 if list_items == 0 else list_items // 2)
        if distinct_sections >= 4:
            grouping_score += 4.0
        elif distinct_sections >= 3:
            grouping_score += 3.0
        elif distinct_sections >= 2:
            grouping_score += 2.0
        
        # Tables (rare but excellent formatting)
        if '|' in response and re.search(r'\|.*\|.*\|', response):
            table_rows = len(re.findall(r'\|.*\|.*\|', response))
            if table_rows >= 3:
                grouping_score += 3.0
        
        score += min(5.0, grouping_score)
        
        # ============================================================
        # 13. CONTEXTUAL APPROPRIATENESS BONUS (0-5 points)
        # Formatting should be appropriate for the query type
        # ============================================================
        context_score = 0.0
        query_lower = query.lower()
        
        # Questions asking for lists/steps should have lists
        list_query = any(kw in query_lower for kw in [
            'steps', 'how to', 'list', 'what are', 'ways to', 'tips',
            'methods', 'techniques', 'examples', 'differences between'
        ])
        if list_query and list_items >= 2:
            context_score += 3.0
        elif list_query and list_items == 0 and word_count > 50:
            context_score -= 2.0  # penalty for not using lists when expected
        
        # Code-related queries should have code blocks
        code_query = any(kw in query_lower for kw in [
            'code', 'function', 'sql', 'python', 'javascript', 'program',
            'script', 'table', 'select', 'create', 'query', 'api'
        ])
        if code_query and code_block_count >= 2:
            context_score += 2.0
        
        # Short conversational queries may not need heavy formatting
        if len(query.split()) < 15 and word_count < 60:
            # Don't penalize short responses to short queries
            context_score += 1.0
        
        score += max(-2.0, min(5.0, context_score))
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Clamp to 0-100 range
        final_score = max(0.0, min(100.0, score))
        
        # Scale to 1-5 range for consistency with examples
        # Map 0-100 -> 1-5
        final_score_scaled = 1.0 + (final_score / 100.0) * 4.0
        
        return round(final_score_scaled, 2)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            # Basic fallback based on length
            if response and len(response.split()) > 20:
                return 2.5
            return 1.5
        except:
            return 1.0