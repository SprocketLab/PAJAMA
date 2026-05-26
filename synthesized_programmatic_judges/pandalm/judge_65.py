def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    Uses a feature-based scoring approach focusing on formatting elements,
    paragraph structure, logical flow, and readability organization.
    
    Returns a score from 0-100 where higher = better structural organization.
    """
    try:
        import re
        import math
        import string
        
        # Handle edge cases
        if not response or not isinstance(response, str):
            return 0
        
        response = response.strip()
        if len(response) == 0:
            return 0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Response length adequacy (0-15 points)
        # Penalize extremely short or extremely long responses relative to query
        # ============================================================
        resp_len = len(response)
        word_count = len(response.split())
        query_words = len(query.split()) if query else 5
        
        # Very short responses are poorly organized by default
        if word_count <= 3:
            return max(2, word_count)
        
        # Length ratio relative to query
        length_ratio = word_count / max(query_words, 1)
        
        if word_count < 10:
            length_score = 3
        elif word_count < 20:
            length_score = 7
        elif word_count < 50:
            length_score = 12
        elif word_count < 150:
            length_score = 15
        elif word_count < 300:
            length_score = 13
        elif word_count < 500:
            length_score = 10
        else:
            length_score = 8
        
        score += length_score
        
        # ============================================================
        # FEATURE 2: Sentence structure and variety (0-20 points)
        # Good organization uses clear, well-formed sentences
        # ============================================================
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 5
        
        # Sentence count score - more sentences generally means better structure
        if num_sentences == 1:
            sent_count_score = 3
        elif num_sentences == 2:
            sent_count_score = 8
        elif 3 <= num_sentences <= 6:
            sent_count_score = 12
        elif 7 <= num_sentences <= 12:
            sent_count_score = 10
        else:
            sent_count_score = 8
        
        # Sentence length variety - good writing varies sentence length
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Some variety is good, too much is chaotic
            if 2 <= std_dev <= 8:
                variety_bonus = 5
            elif 1 <= std_dev < 2:
                variety_bonus = 3
            elif std_dev > 8:
                variety_bonus = 2
            else:
                variety_bonus = 1
        else:
            variety_bonus = 0
        
        # Check for very long run-on sentences (bad)
        runon_penalty = 0
        for sl in sent_lengths:
            if sl > 50:
                runon_penalty += 2
        
        sentence_score = min(20, sent_count_score + variety_bonus - runon_penalty)
        score += max(0, sentence_score)
        
        # ============================================================
        # FEATURE 3: Formatting elements detection (0-20 points)
        # Detect lists, headers, bullet points, numbered items
        # ============================================================
        formatting_score = 0
        
        lines = response.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        
        # Detect numbered lists (1. 2. 3. or 1) 2) 3))
        numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+')
        numbered_items = sum(1 for l in lines if numbered_pattern.match(l))
        if numbered_items >= 2:
            formatting_score += min(8, numbered_items * 2)
        
        # Detect bullet points (-, *, •)
        bullet_pattern = re.compile(r'^\s*[-*•]\s+')
        bullet_items = sum(1 for l in lines if bullet_pattern.match(l))
        if bullet_items >= 2:
            formatting_score += min(8, bullet_items * 2)
        
        # Detect headers (markdown style # or ALL CAPS lines or lines ending with :)
        header_pattern = re.compile(r'^\s*#{1,6}\s+')
        colon_header_pattern = re.compile(r'^[A-Z][^.!?]*:\s*$')
        caps_header_pattern = re.compile(r'^[A-Z\s]{5,}$')
        
        headers_found = 0
        for l in non_empty_lines:
            if header_pattern.match(l):
                headers_found += 1
            elif colon_header_pattern.match(l):
                headers_found += 1
            elif caps_header_pattern.match(l.strip()) and len(l.strip().split()) <= 6:
                headers_found += 1
        
        if headers_found >= 1:
            formatting_score += min(6, headers_found * 3)
        
        # Detect bold/italic markers
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        if bold_count > 0 or italic_count > 0:
            formatting_score += min(3, bold_count + italic_count)
        
        formatting_score = min(20, formatting_score)
        score += formatting_score
        
        # ============================================================
        # FEATURE 4: Paragraph structure (0-15 points)
        # Good responses use paragraphs to separate ideas
        # ============================================================
        paragraph_score = 0
        
        # Count paragraphs (separated by blank lines or double newlines)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if num_paragraphs == 1:
            # Single paragraph - check if it's a wall of text
            if word_count > 100:
                paragraph_score = 2  # Wall of text penalty
            elif word_count > 50:
                paragraph_score = 5
            else:
                paragraph_score = 8  # Short enough that single paragraph is fine
        elif num_paragraphs == 2:
            paragraph_score = 10
        elif 3 <= num_paragraphs <= 5:
            paragraph_score = 15
        elif num_paragraphs > 5:
            paragraph_score = 12
        
        # Check if using line breaks for list-like structure (even without blank lines)
        if num_paragraphs == 1 and len(non_empty_lines) > 2:
            # Multiple lines without paragraph breaks but still structured
            paragraph_score = max(paragraph_score, min(10, len(non_empty_lines) * 2))
        
        score += paragraph_score
        
        # ============================================================
        # FEATURE 5: Logical flow and transition words (0-15 points)
        # Check for discourse markers and logical connectors
        # ============================================================
        flow_score = 0
        
        transition_words = [
            'first', 'second', 'third', 'finally', 'additionally', 'moreover',
            'furthermore', 'however', 'nevertheless', 'in contrast', 'on the other hand',
            'for example', 'for instance', 'specifically', 'in particular',
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'therefore', 'consequently', 'as a result', 'thus',
            'meanwhile', 'subsequently', 'next', 'then',
            'also', 'in addition', 'besides', 'likewise',
            'although', 'while', 'whereas', 'despite',
            'importantly', 'notably', 'significantly'
        ]
        
        response_lower = response.lower()
        transition_count = 0
        for tw in transition_words:
            occurrences = len(re.findall(r'\b' + re.escape(tw) + r'\b', response_lower))
            transition_count += occurrences
        
        if transition_count == 0:
            flow_score = 2
        elif transition_count == 1:
            flow_score = 5
        elif 2 <= transition_count <= 4:
            flow_score = 10
        elif 5 <= transition_count <= 8:
            flow_score = 15
        else:
            flow_score = 12  # Too many might be forced
        
        # Check for topic sentences (sentences starting with key phrases)
        topic_starters = [
            'the ', 'this ', 'these ', 'in ', 'when ', 'a ', 'an ',
            'one ', 'another ', 'it '
        ]
        topic_sentence_count = 0
        for s in sentences:
            s_lower = s.strip().lower()
            for starter in topic_starters:
                if s_lower.startswith(starter):
                    topic_sentence_count += 1
                    break
        
        if topic_sentence_count >= 2:
            flow_score = min(15, flow_score + 2)
        
        score += flow_score
        
        # ============================================================
        # FEATURE 6: Repetition penalty (0 to -15 points)
        # Penalize excessive repetition which indicates poor organization
        # ============================================================
        repetition_penalty = 0
        
        words = re.findall(r'\b\w+\b', response_lower)
        if len(words) > 5:
            # Check for repeated consecutive words/phrases
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            from collections import Counter
            bigram_counts = Counter(bigrams)
            
            # Exclude common bigrams
            common_bigrams = {'of the', 'in the', 'to the', 'and the', 'is a', 'it is',
                            'to be', 'on the', 'for the', 'with the', 'that the', 'is the'}
            
            for bg, count in bigram_counts.items():
                if bg not in common_bigrams and count > 3:
                    repetition_penalty += (count - 3) * 2
            
            # Check for repeated sentences
            sent_texts = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
            sent_counter = Counter(sent_texts)
            for s_text, count in sent_counter.items():
                if count > 1:
                    repetition_penalty += (count - 1) * 5
        
        repetition_penalty = min(15, repetition_penalty)
        score -= repetition_penalty
        
        # ============================================================
        # FEATURE 7: Completeness indicators (0-10 points)
        # Check if the response feels complete and well-concluded
        # ============================================================
        completeness_score = 0
        
        # Ends with proper punctuation
        if response.rstrip()[-1] in '.!?"\'':
            completeness_score += 3
        
        # Doesn't end mid-sentence (truncation check)
        last_line = non_empty_lines[-1] if non_empty_lines else ""
        if last_line and last_line[-1] not in '.!?"\'):;':
            completeness_score -= 3  # Likely truncated
        
        # Has some kind of concluding element
        conclusion_markers = ['in conclusion', 'overall', 'in summary', 'to summarize',
                            'in short', 'ultimately', 'finally']
        has_conclusion = any(cm in response_lower for cm in conclusion_markers)
        if has_conclusion:
            completeness_score += 4
        
        # Response addresses the query (basic check)
        query_words_set = set(re.findall(r'\b\w{4,}\b', query.lower())) if query else set()
        response_words_set = set(re.findall(r'\b\w{4,}\b', response_lower))
        if query_words_set:
            overlap = len(query_words_set & response_words_set)
            relevance = overlap / len(query_words_set)
            completeness_score += min(3, int(relevance * 5))
        else:
            completeness_score += 2
        
        completeness_score = max(0, min(10, completeness_score))
        score += completeness_score
        
        # ============================================================
        # FEATURE 8: Wall-of-text penalty (0 to -10 points)
        # Specifically penalize dense, unbroken text blocks
        # ============================================================
        wall_penalty = 0
        
        # Check average line length (characters)
        if non_empty_lines:
            avg_line_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
            max_line_len = max(len(l) for l in non_empty_lines)
            
            # Very long lines without breaks
            if max_line_len > 500 and len(non_empty_lines) <= 2:
                wall_penalty += 5
            if max_line_len > 800:
                wall_penalty += 3
            
            # Single block of text with many words
            if len(non_empty_lines) == 1 and word_count > 80:
                wall_penalty += 5
        
        wall_penalty = min(10, wall_penalty)
        score -= wall_penalty
        
        # ============================================================
        # FEATURE 9: Information density and specificity bonus (0-5 points)
        # Reward responses with concrete details organized well
        # ============================================================
        specificity_score = 0
        
        # Check for specific examples or elaboration
        example_markers = ['for example', 'such as', 'e.g.', 'including', 'like ',
                          'for instance', 'specifically']
        example_count = sum(1 for em in example_markers if em in response_lower)
        specificity_score += min(3, example_count * 1.5)
        
        # Check for parenthetical clarifications
        paren_count = len(re.findall(r'\([^)]+\)', response))
        if paren_count > 0:
            specificity_score += min(2, paren_count)
        
        score += min(5, specificity_score)
        
        # ============================================================
        # Final score normalization
        # ============================================================
        # Ensure score is in 0-100 range
        score = max(0, min(100, score))
        
        # Apply a mild scaling to spread scores more
        # Most raw scores will be 15-80 range, scale to be more discriminative
        score = round(score, 2)
        
        return score
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 25
        except:
            pass
        return 0