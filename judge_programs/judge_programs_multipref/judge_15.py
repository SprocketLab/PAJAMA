def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique approach focused on:
    - Punctuation variety and correctness
    - Discourse connectives and cohesion markers
    - Sentence structure diversity (measured by sentence length variance)
    - Lexical sophistication (longer words ratio, not just type-token ratio)
    - Paragraph/structural coherence
    - Avoidance of repetitive sentence starts
    - Comma usage patterns (proxy for syntactic complexity)
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
        
        # ---- 1. Sentence extraction ----
        # Split on sentence-ending punctuation but keep abbreviations intact
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # ---- 2. Word extraction ----
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        num_words = max(len(words), 1)
        words_lower = [w.lower() for w in words]
        
        # ---- 3. Punctuation variety score ----
        # Good writing uses diverse punctuation: commas, colons, semicolons, dashes, parentheses
        punct_types = {
            'comma': len(re.findall(r',', text)),
            'colon': len(re.findall(r':', text)),
            'semicolon': len(re.findall(r';', text)),
            'dash': len(re.findall(r'[—–-]{1,2}', text)),
            'parenthesis': len(re.findall(r'[()]', text)),
            'question_mark': len(re.findall(r'\?', text)),
            'exclamation': len(re.findall(r'!', text)),
        }
        punct_types_used = sum(1 for v in punct_types.values() if v > 0)
        # Normalize: using 4+ types is excellent
        punct_variety_score = min(punct_types_used / 5.0, 1.0) * 10
        
        # Comma density (proxy for syntactic complexity) - ideal range 0.03-0.08 per word
        comma_density = punct_types['comma'] / num_words
        if 0.03 <= comma_density <= 0.10:
            comma_score = 8.0
        elif 0.01 <= comma_density < 0.03:
            comma_score = 5.0
        elif comma_density > 0.10:
            comma_score = 4.0
        else:
            comma_score = 2.0
        
        # ---- 4. Discourse connectives and cohesion ----
        cohesion_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'specifically',
            'alternatively', 'similarly', 'conversely', 'notably',
            'importantly', 'essentially', 'particularly', 'accordingly',
            'subsequently', 'ultimately', 'overall', 'indeed',
            'certainly', 'clearly', 'evidently', 'typically',
            'generally', 'primarily', 'essentially', 'fundamentally',
            'although', 'whereas', 'while', 'since', 'because',
            'thus', 'hence', 'also', 'besides', 'instead',
        ]
        # Connecting phrases
        connecting_phrases = [
            'in addition', 'on the other hand', 'for example', 'for instance',
            'in contrast', 'as a result', 'in other words', 'that said',
            'to summarize', 'in conclusion', 'more specifically', 'in particular',
            'at the same time', 'in fact', 'as such', 'to be specific',
            'with that in mind', 'having said that', 'it is important',
            'this means', 'this is because', 'keep in mind',
        ]
        
        text_lower = text.lower()
        cohesion_count = sum(1 for w in cohesion_words if w in words_lower)
        phrase_count = sum(1 for p in connecting_phrases if p in text_lower)
        
        total_cohesion = cohesion_count + phrase_count * 1.5
        # Normalize per 100 words
        cohesion_per_100 = (total_cohesion / num_words) * 100
        # Ideal: 2-6 per 100 words
        if 1.5 <= cohesion_per_100 <= 8:
            cohesion_score = 10.0
        elif 0.5 <= cohesion_per_100 < 1.5:
            cohesion_score = 6.0
        elif cohesion_per_100 > 8:
            cohesion_score = 7.0
        else:
            cohesion_score = 3.0
        
        # ---- 5. Sentence length variance (structural diversity) ----
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 2:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Ideal CV: 0.3-0.7 (good variety without being chaotic)
            if 0.25 <= cv <= 0.8:
                sent_variety_score = 10.0
            elif 0.15 <= cv < 0.25:
                sent_variety_score = 7.0
            elif cv > 0.8:
                sent_variety_score = 6.0
            else:
                sent_variety_score = 4.0
            
            # Penalize if average sentence length is too extreme
            if mean_len < 5:
                sent_variety_score *= 0.6
            elif mean_len > 40:
                sent_variety_score *= 0.7
        else:
            sent_variety_score = 4.0
        
        # ---- 6. Sentence start diversity ----
        # Penalize repetitive sentence beginnings
        if len(sentences) >= 3:
            first_words = []
            for s in sentences:
                s_stripped = re.sub(r'^[\d.\-*#\s]+', '', s.strip())
                fw_match = re.match(r'[A-Za-z]+', s_stripped)
                if fw_match:
                    first_words.append(fw_match.group().lower())
            
            if first_words:
                fw_counter = Counter(first_words)
                most_common_freq = fw_counter.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(first_words)
                # Lower repetition is better
                if repetition_ratio <= 0.25:
                    start_diversity_score = 10.0
                elif repetition_ratio <= 0.4:
                    start_diversity_score = 7.0
                elif repetition_ratio <= 0.6:
                    start_diversity_score = 5.0
                else:
                    start_diversity_score = 3.0
            else:
                start_diversity_score = 5.0
        else:
            start_diversity_score = 5.0
        
        # ---- 7. Lexical sophistication ----
        # Ratio of "sophisticated" words (7+ chars) that aren't super common
        common_long_words = {
            'because', 'through', 'between', 'another', 'however',
            'something', 'important', 'different', 'together', 'anything',
            'everything', 'everyone', 'someone', 'without', 'against',
            'example', 'following', 'including', 'provide', 'available',
        }
        
        sophisticated_count = 0
        for w in words_lower:
            if len(w) >= 7 and w not in common_long_words:
                sophisticated_count += 1
        
        sophistication_ratio = sophisticated_count / num_words
        # Ideal: 0.15-0.35
        if 0.12 <= sophistication_ratio <= 0.40:
            lexical_score = 10.0
        elif 0.08 <= sophistication_ratio < 0.12:
            lexical_score = 7.0
        elif sophistication_ratio > 0.40:
            lexical_score = 6.0
        else:
            lexical_score = 4.0
        
        # ---- 8. Character-level n-gram repetition (unique approach) ----
        # Excessive character trigram repetition signals repetitive/formulaic text
        if len(text) >= 50:
            char_trigrams = [text_lower[i:i+3] for i in range(len(text_lower) - 2)]
            trigram_counter = Counter(char_trigrams)
            total_trigrams = len(char_trigrams)
            unique_trigrams = len(trigram_counter)
            trigram_diversity = unique_trigrams / max(total_trigrams, 1)
            # Higher diversity is better (less repetitive)
            char_rep_score = min(trigram_diversity * 30, 10.0)
        else:
            char_rep_score = 5.0
        
        # ---- 9. Engagement / tone quality ----
        # Presence of direct address, questions, and engaging language
        engagement_markers = [
            "you ", "your ", "you'll", "you're", "let's", "here's",
            "here are", "let me", "we can", "we'll",
        ]
        engagement_count = sum(1 for m in engagement_markers if m in text_lower)
        # Has a question?
        has_question = '?' in text
        
        engagement_score = min(engagement_count * 1.5 + (2.0 if has_question else 0), 10.0)
        engagement_score = max(engagement_score, 2.0)  # baseline
        
        # ---- 10. Structural formatting bonus ----
        # Bold text, numbered lists, etc. indicate well-organized response
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', text))
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
        has_bullet = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
        
        structure_bonus = 0
        if has_bold:
            structure_bonus += 2.0
        if has_numbered_list:
            structure_bonus += 1.5
        if has_bullet:
            structure_bonus += 1.0
        structure_bonus = min(structure_bonus, 4.0)
        
        # ---- 11. Response length adequacy relative to query ----
        query_words = len(re.findall(r'[a-zA-Z]+', query)) if query else 5
        length_ratio = num_words / max(query_words, 1)
        
        if length_ratio >= 3:
            length_score = 8.0
        elif length_ratio >= 1.5:
            length_score = 6.0
        else:
            length_score = 3.0
        
        # Bonus for substantial responses
        if num_words >= 80:
            length_score = min(length_score + 2.0, 10.0)
        
        # ---- FINAL WEIGHTED COMBINATION ----
        weights = {
            'punct_variety': 0.08,
            'comma': 0.06,
            'cohesion': 0.15,
            'sent_variety': 0.12,
            'start_diversity': 0.10,
            'lexical': 0.12,
            'char_rep': 0.07,
            'engagement': 0.10,
            'structure': 0.10,  # applied to structure_bonus scaled to 10
            'length': 0.10,
        }
        
        structure_scaled = min(structure_bonus * 2.5, 10.0)
        
        final_score = (
            weights['punct_variety'] * punct_variety_score +
            weights['comma'] * comma_score +
            weights['cohesion'] * cohesion_score +
            weights['sent_variety'] * sent_variety_score +
            weights['start_diversity'] * start_diversity_score +
            weights['lexical'] * lexical_score +
            weights['char_rep'] * char_rep_score +
            weights['engagement'] * engagement_score +
            weights['structure'] * structure_scaled +
            weights['length'] * length_score
        )
        
        # Scale to 0-100
        final_score = final_score * 10
        
        # Clamp
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 25.0