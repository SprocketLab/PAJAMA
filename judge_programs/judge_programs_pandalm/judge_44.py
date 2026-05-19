def judging_function(query, response):
    """
    Evaluate clarity and conciseness using a structural coherence approach.
    
    This variant focuses on:
    1. Paragraph/structural organization quality
    2. Functional word ratio (detecting bloat via filler/connector overuse)
    3. Unique information density (content words per total words)
    4. Repetition detection via compression ratio simulation
    5. Sentence-level coherence (topic continuity between sentences)
    6. Response completeness relative to query demands
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        query = query.strip() if query else ""
        
        if len(response) == 0:
            return 0.0
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response.lower())
        if len(words) == 0:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        if len(sentences) == 0:
            sentences = [response]
        
        # === Feature 1: Compression Ratio (repetition detection) ===
        # Simulate compression by looking at how much unique content exists
        # relative to total content using character-level trigrams
        char_text = response.lower()
        if len(char_text) >= 3:
            char_trigrams = [char_text[i:i+3] for i in range(len(char_text) - 2)]
            unique_trigrams = len(set(char_trigrams))
            total_trigrams = len(char_trigrams)
            compression_ratio = unique_trigrams / total_trigrams if total_trigrams > 0 else 1.0
        else:
            compression_ratio = 1.0
        
        # Heavily penalize very repetitive text (compression_ratio < 0.3 is very repetitive)
        if compression_ratio < 0.15:
            compression_score = 0.0
        elif compression_ratio < 0.3:
            compression_score = compression_ratio * 2.0
        else:
            compression_score = min(1.0, 0.4 + compression_ratio * 0.7)
        
        # === Feature 2: Functional/filler word ratio ===
        filler_words = {
            'very', 'really', 'quite', 'rather', 'somewhat', 'basically',
            'actually', 'literally', 'essentially', 'practically', 'virtually',
            'just', 'simply', 'merely', 'kind', 'sort', 'like', 'stuff',
            'things', 'thing', 'etc', 'overall', 'general', 'generally',
            'obviously', 'clearly', 'definitely', 'certainly', 'absolutely',
            'totally', 'completely', 'entirely', 'utterly', 'highly'
        }
        
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / len(words) if len(words) > 0 else 0
        filler_score = max(0, 1.0 - filler_ratio * 8)  # Penalize heavily
        
        # === Feature 3: Content word density ===
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'when', 'where', 'why', 'how', 'if', 'while', 'also', 'about'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_density = len(content_words) / len(words) if len(words) > 0 else 0
        # Ideal content density is around 0.4-0.6
        if content_density < 0.2:
            density_score = content_density * 3
        elif content_density > 0.8:
            density_score = max(0.3, 1.0 - (content_density - 0.8) * 3)
        else:
            density_score = 0.6 + (content_density - 0.2) * 0.67
        
        # === Feature 4: Phrase-level repetition detection ===
        # Check for repeated phrases (3-word windows)
        if len(words) >= 3:
            phrases_3 = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            phrase_counts = Counter(phrases_3)
            total_phrases = len(phrases_3)
            repeated_phrases = sum(c - 1 for c in phrase_counts.values() if c > 1)
            phrase_repetition_ratio = repeated_phrases / total_phrases if total_phrases > 0 else 0
        else:
            phrase_repetition_ratio = 0
        
        phrase_rep_score = max(0, 1.0 - phrase_repetition_ratio * 5)
        
        # === Feature 5: Sentence-level topic coherence ===
        # Measure how well sentences connect (shared content words between adjacent sentences)
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                words_a = set(re.findall(r'[a-zA-Z]+', sentences[i].lower())) - stop_words
                words_b = set(re.findall(r'[a-zA-Z]+', sentences[i+1].lower())) - stop_words
                if len(words_a) > 0 and len(words_b) > 0:
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    # We want some overlap (coherence) but not too much (redundancy)
                    sim = overlap / union if union > 0 else 0
                    # Ideal similarity between sentences: 0.1-0.4
                    if sim > 0.7:
                        # Too similar = redundant sentences
                        coherence_scores.append(0.3)
                    elif sim > 0.4:
                        coherence_scores.append(0.7)
                    elif sim >= 0.05:
                        coherence_scores.append(1.0)
                    else:
                        coherence_scores.append(0.5)  # No connection at all
                else:
                    coherence_scores.append(0.5)
            coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.7
        else:
            coherence_score = 0.6  # Single sentence - neutral
        
        # === Feature 6: Sentence length variance (consistency) ===
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 2:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Some variation is good (not monotonous), but too much is chaotic
            if cv < 0.1:
                variance_score = 0.7  # Too uniform
            elif cv < 0.5:
                variance_score = 1.0  # Good variation
            elif cv < 1.0:
                variance_score = 0.6
            else:
                variance_score = 0.3  # Too chaotic
        else:
            variance_score = 0.6
        
        # === Feature 7: Response substantiveness ===
        # Check if response actually addresses the query
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - stop_words
        response_content = set(content_words)
        
        if len(query_words) > 0:
            query_coverage = len(query_words & response_content) / len(query_words)
        else:
            query_coverage = 0.5
        
        # Don't want too much query echo (just repeating the question)
        if len(response_content) > 0:
            query_echo = len(query_words & response_content) / len(response_content)
        else:
            query_echo = 0
        
        if query_echo > 0.8:
            relevance_score = 0.4  # Mostly just echoing the query
        elif query_coverage > 0.3:
            relevance_score = min(1.0, 0.6 + query_coverage * 0.5)
        else:
            relevance_score = 0.4 + query_coverage
        
        # === Feature 8: Word-level unique ratio ===
        word_counts = Counter(words)
        unique_word_ratio = len(word_counts) / len(words) if len(words) > 0 else 0
        
        # For short texts, high unique ratio is expected
        # For longer texts, lower is expected but too low means repetitive
        expected_unique = 1.0 / (1.0 + math.log(max(1, len(words)) / 10))
        if unique_word_ratio < expected_unique * 0.5:
            vocab_score = unique_word_ratio / (expected_unique * 0.5) * 0.5
        else:
            vocab_score = min(1.0, 0.5 + unique_word_ratio * 0.6)
        
        # === Feature 9: Appropriate length ===
        # Not too short (empty/trivial) and not excessively long
        # Relative to query complexity
        query_word_count = len(re.findall(r'[a-zA-Z]+', query))
        
        if len(words) < 3:
            length_score = 0.2
        elif len(words) < 8:
            length_score = 0.4
        elif len(words) < 15:
            length_score = 0.7
        elif len(words) <= 150:
            length_score = 1.0
        elif len(words) <= 300:
            length_score = 0.8
        else:
            length_score = max(0.3, 0.8 - (len(words) - 300) / 1000)
        
        # === Feature 10: Structural markers (lists, organization) ===
        has_structure = 0
        if re.search(r'^\s*[-•*]\s', response, re.MULTILINE):
            has_structure += 0.1
        if re.search(r'^\s*\d+[.)]\s', response, re.MULTILINE):
            has_structure += 0.1
        if re.search(r'\b(first|second|third|finally|additionally|moreover|furthermore)\b', response.lower()):
            has_structure += 0.05
        structure_score = min(1.0, 0.7 + has_structure)
        
        # === Combine all features with weights ===
        final_score = (
            compression_score * 2.5 +      # Repetition is a major clarity killer
            filler_score * 1.0 +            # Filler words reduce conciseness
            density_score * 1.5 +           # Content density matters
            phrase_rep_score * 2.0 +        # Phrase repetition is very bad
            coherence_score * 1.5 +         # Sentence coherence
            variance_score * 0.5 +          # Sentence variety
            relevance_score * 1.5 +         # Actually addressing the query
            vocab_score * 1.0 +             # Vocabulary richness
            length_score * 1.5 +            # Appropriate length
            structure_score * 0.5           # Organization
        )
        
        max_possible = 2.5 + 1.0 + 1.5 + 2.0 + 1.5 + 0.5 + 1.5 + 1.0 + 1.5 + 0.5  # = 13.5
        
        # Normalize to 0-100
        normalized = (final_score / max_possible) * 100
        
        return round(max(0.0, min(100.0, normalized)), 2)
        
    except Exception:
        return 25.0