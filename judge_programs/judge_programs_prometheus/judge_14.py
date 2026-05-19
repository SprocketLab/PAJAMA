def judging_function(query, response):
    """
    Evaluates language quality and readability using:
    - Punctuation diversity and correctness
    - Sentence structure variation (length variance)
    - Lexical sophistication (longer unique words ratio)
    - Discourse markers and cohesion signals
    - Spelling-like heuristics (unusual character patterns)
    - Sentence opening diversity
    - Comma usage density (proxy for complex sentence structures)
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
        
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", text)
        if len(words) < 3:
            return 0.5
        
        lower_words = [w.lower() for w in words]
        num_words = len(words)
        
        # 1. Sentence structure variation score
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        num_sentences = max(len(sentences), 1)
        
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 2:
            mean_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sent_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sent_len = math.sqrt(variance)
            # Good writing has moderate variation (not all same length, not wildly different)
            # Coefficient of variation
            cv = std_sent_len / max(mean_sent_len, 1)
            # Ideal CV around 0.3-0.6
            if cv < 0.1:
                variation_score = 3.0
            elif cv < 0.3:
                variation_score = 5.0 + (cv - 0.1) * 15
            elif cv <= 0.6:
                variation_score = 8.0
            elif cv <= 1.0:
                variation_score = 8.0 - (cv - 0.6) * 7.5
            else:
                variation_score = max(2.0, 5.0 - cv * 2)
            
            # Penalize very short or very long average sentence length
            if mean_sent_len < 5:
                variation_score *= 0.6
            elif mean_sent_len < 8:
                variation_score *= 0.8
            elif mean_sent_len > 40:
                variation_score *= 0.7
            elif mean_sent_len > 30:
                variation_score *= 0.85
        else:
            variation_score = 4.0
        
        # 2. Sentence opening diversity
        sentence_openers = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sentence_openers.append(sw[0].lower())
        
        if len(sentence_openers) >= 2:
            opener_counts = Counter(sentence_openers)
            unique_openers = len(opener_counts)
            opener_diversity = unique_openers / len(sentence_openers)
            # Penalize repetitive starts (e.g., always starting with "it" or "you")
            max_opener_freq = max(opener_counts.values()) / len(sentence_openers)
            opener_score = opener_diversity * 6.0 + (1.0 - max_opener_freq) * 4.0
        else:
            opener_score = 5.0
        
        # 3. Punctuation richness and diversity
        punct_types = set()
        punct_counts = Counter()
        for ch in text:
            if ch in '.,;:!?-—()[]"\'…':
                punct_types.add(ch)
                punct_counts[ch] += 1
        
        # Diverse punctuation suggests sophisticated writing
        punct_diversity = min(len(punct_types) / 6.0, 1.0)  # normalize to max ~6 types
        
        # Comma density (commas per sentence) - proxy for complex structures
        comma_count = punct_counts.get(',', 0)
        comma_per_sent = comma_count / num_sentences
        # Ideal: 1-3 commas per sentence
        if comma_per_sent < 0.3:
            comma_score = 4.0
        elif comma_per_sent <= 3.0:
            comma_score = 7.0 + min((comma_per_sent - 0.3) * 2, 3.0)
        else:
            comma_score = max(4.0, 10.0 - (comma_per_sent - 3.0) * 2)
        
        punct_score = punct_diversity * 5.0 + comma_score * 0.5
        
        # 4. Discourse markers and cohesion (different set from transition words)
        cohesion_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bnevertheless\b',
            r'\bin addition\b', r'\bon the other hand\b', r'\bconsequently\b',
            r'\btherefore\b', r'\bthus\b', r'\bmeanwhile\b', r'\bnonetheless\b',
            r'\bfor instance\b', r'\bfor example\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bas a result\b', r'\bin contrast\b',
            r'\bsimilarly\b', r'\blikewise\b', r'\binstead\b',
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\bdespite\b',
            r'\bregardless\b', r'\baccordingly\b', r'\bhence\b',
            r'\bin fact\b', r'\bindeed\b', r'\bcertainly\b',
            r'\bultimately\b', r'\boverall\b', r'\bin summary\b',
        ]
        
        marker_count = 0
        marker_types_found = 0
        text_lower = text.lower()
        for pattern in cohesion_markers:
            matches = re.findall(pattern, text_lower)
            if matches:
                marker_types_found += 1
                marker_count += len(matches)
        
        # Normalize by text length
        markers_per_100_words = (marker_count / num_words) * 100
        marker_diversity = min(marker_types_found / 5.0, 1.0)
        
        cohesion_score = min(markers_per_100_words * 2.5, 5.0) + marker_diversity * 5.0
        
        # 5. Lexical sophistication - proportion of "sophisticated" words
        # (words with 3+ syllables, excluding common ones)
        def estimate_syllables(word):
            word = word.lower()
            if len(word) <= 2:
                return 1
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e') and count > 1:
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            return max(count, 1)
        
        common_simple = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                         'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                         'and', 'but', 'or', 'not', 'no', 'so', 'if', 'as', 'just',
                         'about', 'also', 'very', 'really', 'already', 'another',
                         'everything', 'everyone', 'something', 'anything', 'nothing',
                         'however', 'because', 'although', 'important', 'remember',
                         'understand', 'different', 'together', 'beautiful', 'possible',
                         'probably', 'certainly', 'absolutely', 'completely', 'actually'}
        
        sophisticated_count = 0
        content_words = [w for w in lower_words if w not in common_simple and len(w) > 3]
        for w in content_words:
            syl = estimate_syllables(w)
            if syl >= 3 and len(w) >= 6:
                sophisticated_count += 1
        
        sophistication_ratio = sophisticated_count / max(num_words, 1)
        # Ideal range: 0.05-0.20
        if sophistication_ratio < 0.02:
            lexical_score = 3.0
        elif sophistication_ratio < 0.05:
            lexical_score = 3.0 + (sophistication_ratio - 0.02) * 100
        elif sophistication_ratio <= 0.20:
            lexical_score = 6.0 + (sophistication_ratio - 0.05) * 26.67
        elif sophistication_ratio <= 0.30:
            lexical_score = 10.0 - (sophistication_ratio - 0.20) * 20
        else:
            lexical_score = max(3.0, 8.0 - sophistication_ratio * 10)
        
        # 6. Spelling heuristic: detect unusual consonant clusters or patterns
        # that suggest errors or poor writing
        unusual_patterns = [
            r'[bcdfghjklmnpqrstvwxyz]{5,}',  # 5+ consecutive consonants
            r'(.)\1{3,}',  # 4+ repeated characters
            r'\b[A-Z]{2,}\b',  # ALL CAPS words (shouting)
        ]
        
        anomaly_count = 0
        for pattern in unusual_patterns:
            anomaly_count += len(re.findall(pattern, text))
        
        anomaly_penalty = min(anomaly_count * 0.5, 3.0)
        
        # 7. Contraction and informal language detection
        # Not penalizing per se, but tracking register consistency
        contractions = re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text_lower)
        contraction_rate = len(contractions) / max(num_words, 1) * 100
        
        # Formal markers
        formal_words = ['furthermore', 'consequently', 'nevertheless', 'therefore',
                        'additionally', 'moreover', 'henceforth', 'subsequently',
                        'notwithstanding', 'accordingly']
        formal_count = sum(1 for w in lower_words if w in formal_words)
        
        # Register consistency: mixing very formal and very informal is bad
        if contraction_rate > 3 and formal_count > 2:
            register_penalty = 1.5
        else:
            register_penalty = 0.0
        
        # 8. Response length adequacy relative to query
        query_words = len(re.findall(r"[a-zA-Z']+", query)) if query else 10
        response_query_ratio = num_words / max(query_words, 1)
        
        if response_query_ratio < 0.5:
            length_score = 3.0
        elif response_query_ratio < 1.0:
            length_score = 5.0
        elif response_query_ratio <= 4.0:
            length_score = 8.0
        elif response_query_ratio <= 6.0:
            length_score = 7.0
        else:
            length_score = 6.0
        
        # 9. Numbered list / structured content bonus
        has_numbered_list = bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
        has_structure = bool(re.search(r'\n\s*\n', text))  # paragraph breaks
        structure_bonus = 0.0
        if has_numbered_list:
            structure_bonus += 0.5
        if has_structure:
            structure_bonus += 0.3
        
        # 10. Empathy/engagement signals (for conversational responses)
        empathy_phrases = [
            r"\bi understand\b", r"\bi can see\b", r"\bthat's understandable\b",
            r"\bit's okay\b", r"\bit's completely\b", r"\bit's perfectly\b",
            r"\bi'm sorry\b", r"\bi hear\b", r"\byou're feeling\b",
            r"\bdon't worry\b", r"\bfeel free\b", r"\btake your time\b",
            r"\babsolutely\b", r"\bgenuinely\b", r"\bsincerely\b"
        ]
        empathy_count = sum(1 for p in empathy_phrases if re.search(p, text_lower))
        empathy_bonus = min(empathy_count * 0.4, 1.5)
        
        # Combine all scores with weights
        # variation_score: 0-10
        # opener_score: 0-10
        # punct_score: 0-10
        # cohesion_score: 0-10
        # lexical_score: 0-10
        # length_score: 0-8
        
        final_score = (
            variation_score * 0.15 +
            opener_score * 0.12 +
            punct_score * 0.12 +
            cohesion_score * 0.18 +
            lexical_score * 0.18 +
            length_score * 0.10 +
            comma_score * 0.08 +
            structure_bonus +
            empathy_bonus -
            anomaly_penalty -
            register_penalty
        )
        
        # Bonus for sufficient sentence count (well-developed responses)
        if num_sentences >= 4:
            final_score += 0.5
        if num_sentences >= 6:
            final_score += 0.3
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
    
    except Exception:
        return 3.0