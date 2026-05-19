def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique approach focused on:
    - Punctuation correctness and variety
    - Sentence structure diversity (length variance)
    - Transition word usage (cohesion)
    - Paragraph structure
    - Lexical sophistication (longer words ratio, not just TTR)
    - Avoidance of repetitive sentence starts
    - Comma usage patterns
    - Question/exclamation variety
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.5
        
        # ---- Feature 1: Sentence structure diversity (variance in sentence lengths) ----
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        if len(sentences) < 1:
            return 1.0
        
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        # Variance of sentence lengths - higher variance means more variety
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance_sl)
            # Coefficient of variation
            cv_sl = std_sl / mean_sl if mean_sl > 0 else 0
        else:
            cv_sl = 0
        
        # Score: moderate CV is best (0.3-0.7 range), too low = monotonous, too high = chaotic
        if cv_sl < 0.1:
            sent_variety_score = 2.0
        elif cv_sl < 0.3:
            sent_variety_score = 5.0
        elif cv_sl <= 0.7:
            sent_variety_score = 8.0
        elif cv_sl <= 1.0:
            sent_variety_score = 6.0
        else:
            sent_variety_score = 4.0
        
        # ---- Feature 2: Transition/cohesion words ----
        transition_words = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'specifically', 'particularly',
            'alternatively', 'similarly', 'likewise', 'conversely', 'nonetheless',
            'accordingly', 'hence', 'thus', 'indeed', 'notably',
            'in addition', 'on the other hand', 'for example', 'for instance',
            'in contrast', 'as a result', 'in particular', 'in fact',
            'that said', 'to summarize', 'in summary', 'overall',
            'first', 'second', 'third', 'finally', 'also', 'besides',
            'although', 'while', 'whereas', 'since', 'because',
            'certainly', 'essentially', 'importantly'
        }
        
        text_lower = text.lower()
        words = re.findall(r"[a-z']+", text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return 1.0
        
        transition_count = 0
        for tw in transition_words:
            if ' ' in tw:
                transition_count += text_lower.count(tw)
            else:
                transition_count += words.count(tw)
        
        transition_density = transition_count / max(word_count, 1) * 100
        # Ideal: 2-6% transition word density
        if transition_density < 0.5:
            transition_score = 3.0
        elif transition_density < 2.0:
            transition_score = 6.0
        elif transition_density <= 6.0:
            transition_score = 9.0
        elif transition_density <= 10.0:
            transition_score = 6.0
        else:
            transition_score = 4.0
        
        # ---- Feature 3: Punctuation variety and correctness ----
        # Count different punctuation types used
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-()"\'"':
                punct_types.add(ch)
        
        # Check for comma usage (commas per sentence)
        comma_count = text.count(',')
        commas_per_sent = comma_count / max(len(sentences), 1)
        
        # Check for colon/semicolon usage (sign of complex writing)
        colon_count = text.count(':')
        semicolon_count = text.count(';')
        
        punct_variety_score = min(len(punct_types) * 1.2, 8.0)
        
        # Penalize no commas or excessive commas
        if commas_per_sent < 0.3:
            punct_variety_score -= 1.0
        elif commas_per_sent > 4.0:
            punct_variety_score -= 0.5
        
        # Bonus for semicolons/colons (sophisticated punctuation)
        if semicolon_count > 0:
            punct_variety_score += 0.5
        if colon_count > 0:
            punct_variety_score += 0.3
        
        punct_variety_score = max(0, min(10, punct_variety_score))
        
        # ---- Feature 4: Sentence start diversity ----
        # Check if sentences start with varied words
        if len(sentences) >= 2:
            first_words = []
            for s in sentences:
                w = s.strip().split()
                if w:
                    first_words.append(w[0].lower())
            
            if first_words:
                unique_starts = len(set(first_words))
                total_starts = len(first_words)
                start_diversity = unique_starts / total_starts
            else:
                start_diversity = 0.5
        else:
            start_diversity = 0.5
        
        # Score: higher diversity is better
        start_diversity_score = start_diversity * 10.0
        
        # ---- Feature 5: Lexical sophistication ----
        # Ratio of "sophisticated" words (7+ characters, not common)
        common_short = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                       'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                       'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see',
                       'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too', 'use'}
        
        sophisticated_count = sum(1 for w in words if len(w) >= 7)
        sophistication_ratio = sophisticated_count / max(word_count, 1)
        
        # Ideal range: 15-35% sophisticated words
        if sophistication_ratio < 0.05:
            lexical_score = 3.0
        elif sophistication_ratio < 0.15:
            lexical_score = 6.0
        elif sophistication_ratio <= 0.35:
            lexical_score = 9.0
        elif sophistication_ratio <= 0.50:
            lexical_score = 7.0
        else:
            lexical_score = 5.0
        
        # ---- Feature 6: Paragraph structure ----
        # Well-structured responses have multiple paragraphs or clear sections
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Check for structural elements (numbered lists, headers, etc.)
        has_structure = bool(re.search(r'(\d+\.\s|#{1,3}\s|\*\*[^*]+\*\*|•|\-\s)', text))
        
        if num_paragraphs == 1 and word_count > 80:
            # Long single paragraph - not great structure
            structure_score = 4.0
        elif num_paragraphs >= 2 and num_paragraphs <= 15:
            structure_score = 7.0
        elif num_paragraphs > 15:
            structure_score = 5.5
        else:
            structure_score = 5.0
        
        if has_structure:
            structure_score += 1.5
        
        structure_score = min(10, structure_score)
        
        # ---- Feature 7: Engagement and tone ----
        # Check for direct address, questions, exclamations (engaging writing)
        has_question = '?' in text
        has_exclamation = '!' in text
        has_direct_address = any(w in words for w in ['you', 'your', "you're", "you'll"])
        
        engagement_score = 4.0
        if has_question:
            engagement_score += 1.5
        if has_exclamation:
            engagement_score += 1.0
        if has_direct_address:
            engagement_score += 1.5
        
        # Check for opening that's engaging (not just "I" or bare start)
        first_sentence = sentences[0] if sentences else ""
        first_words_check = first_sentence.lower().split()[:3]
        engaging_openers = {'certainly', 'great', 'awesome', 'absolutely', 'excellent',
                           'wonderful', 'interesting', "that's", 'the', 'organizing',
                           'there'}
        if first_words_check and first_words_check[0] in engaging_openers:
            engagement_score += 1.0
        
        engagement_score = min(10, engagement_score)
        
        # ---- Feature 8: Grammar heuristics ----
        grammar_score = 8.0
        
        # Check for double spaces
        double_spaces = text.count('  ')
        if double_spaces > 3:
            grammar_score -= 1.0
        
        # Check for missing space after punctuation
        missing_space = len(re.findall(r'[.!?,;:][a-zA-Z]', text))
        if missing_space > 2:
            grammar_score -= min(2.0, missing_space * 0.5)
        
        # Check for repeated words (e.g., "the the")
        repeated = len(re.findall(r'\b(\w+)\s+\1\b', text_lower))
        if repeated > 0:
            grammar_score -= min(2.0, repeated * 1.0)
        
        # Check capitalization after sentence-ending punctuation
        cap_after_period = re.findall(r'[.!?]\s+[a-z]', text)
        if len(cap_after_period) > 1:
            grammar_score -= min(1.5, len(cap_after_period) * 0.5)
        
        grammar_score = max(0, min(10, grammar_score))
        
        # ---- Feature 9: Response length appropriateness ----
        # Not too short, not excessively padded
        length_score = 5.0
        if word_count < 15:
            length_score = 2.0
        elif word_count < 30:
            length_score = 4.0
        elif word_count < 50:
            length_score = 6.0
        elif word_count <= 200:
            length_score = 8.0
        elif word_count <= 400:
            length_score = 7.0
        else:
            length_score = 6.0
        
        # ---- Combine all features with weights ----
        weights = {
            'sent_variety': 0.10,
            'transition': 0.15,
            'punct_variety': 0.10,
            'start_diversity': 0.10,
            'lexical': 0.12,
            'structure': 0.13,
            'engagement': 0.12,
            'grammar': 0.10,
            'length': 0.08,
        }
        
        final_score = (
            weights['sent_variety'] * sent_variety_score +
            weights['transition'] * transition_score +
            weights['punct_variety'] * punct_variety_score +
            weights['start_diversity'] * start_diversity_score +
            weights['lexical'] * lexical_score +
            weights['structure'] * structure_score +
            weights['engagement'] * engagement_score +
            weights['grammar'] * grammar_score +
            weights['length'] * length_score
        )
        
        # Scale to 0-100 range
        final_score = final_score * 10.0
        
        return round(final_score, 2)
    
    except Exception:
        return 25.0