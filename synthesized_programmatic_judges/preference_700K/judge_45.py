def judging_function(query, response):
    """
    Evaluates clarity and conciseness using:
    - Sentence structure analysis (avg sentence length, variance)
    - Information density (unique content words / total words)
    - Filler/weasel word detection
    - Redundancy detection via sentence-level semantic overlap (word set similarity between sentences)
    - Structural signals (code blocks, formatting)
    - Compression ratio as proxy for information density
    - Passive voice estimation
    - Parenthetical/aside density
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            return 0.5
        
        # ---- Basic tokenization ----
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*|\d+(?:\.\d+)?", response)
        word_count = len(words)
        if word_count < 2:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        # ---- Sentence splitting ----
        # Split on sentence-ending punctuation but not on abbreviations or decimals
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$|\n{2,}', response_stripped)
        sentences = [s.strip() for s in sentences if s and len(s.strip()) > 3]
        if not sentences:
            sentences = [response_stripped]
        num_sentences = len(sentences)
        
        # ---- 1. Sentence length analysis ----
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*|\d+(?:\.\d+)?", s)
            sent_word_counts.append(len(s_words))
        
        avg_sent_len = sum(sent_word_counts) / max(len(sent_word_counts), 1)
        
        # Ideal sentence length: 10-20 words. Penalize very long or very short.
        if 10 <= avg_sent_len <= 20:
            sent_len_score = 1.0
        elif avg_sent_len < 5:
            sent_len_score = 0.5
        elif avg_sent_len < 10:
            sent_len_score = 0.7 + 0.03 * avg_sent_len
        elif avg_sent_len <= 30:
            sent_len_score = 1.0 - 0.03 * (avg_sent_len - 20)
        else:
            sent_len_score = max(0.2, 0.7 - 0.01 * (avg_sent_len - 30))
        
        # Sentence length variance — moderate variance is good (shows varied structure)
        if len(sent_word_counts) > 1:
            mean_swc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_swc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            cv = math.sqrt(variance) / max(mean_swc, 1)  # coefficient of variation
            # Ideal CV: 0.3-0.7
            if 0.2 <= cv <= 0.8:
                variance_score = 1.0
            elif cv < 0.2:
                variance_score = 0.7  # too uniform
            else:
                variance_score = max(0.5, 1.0 - 0.3 * (cv - 0.8))
        else:
            variance_score = 0.7
        
        # ---- 2. Filler / weasel / hedge words ----
        filler_phrases = [
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b', r'\bliterally\b',
            r'\bjust\b', r'\breally\b', r'\bvery\b', r'\bquite\b', r'\brather\b',
            r'\bsomewhat\b', r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bin my opinion\b', r'\bi think that\b', r'\bit seems like\b',
            r'\bit is worth noting that\b', r'\bit should be noted that\b',
            r'\bneedless to say\b', r'\bas a matter of fact\b',
            r'\bat the end of the day\b', r'\ball things considered\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bfor what it\'s worth\b', r'\bin terms of\b',
            r'\bthe thing is\b', r'\bto be honest\b', r'\bto be fair\b',
            r'\bif you will\b', r'\bso to speak\b', r'\bas such\b',
            r'\bpretty much\b', r'\ba little bit\b', r'\bat this point in time\b',
            r'\bthe fact that\b', r'\bin a sense\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / max(word_count, 1)
        filler_score = max(0.0, 1.0 - filler_ratio * 15)  # penalize heavily
        
        # ---- 3. Redundancy: pairwise sentence overlap ----
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
                'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
                'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
                'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
                'who', 'whom', 'this', 'that', 'these', 'those', 'it', 'its',
                'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
                'his', 'she', 'her', 'they', 'them', 'their', 'about', 'up',
                'out', 'then', 'there', 'here', 'also', 'over',
            }
            w = re.findall(r'[a-z]+', text.lower())
            return set(w2 for w2 in w if w2 not in stop_words and len(w2) > 2)
        
        if num_sentences >= 2:
            sent_content_sets = [get_content_words(s) for s in sentences]
            overlaps = []
            for i in range(len(sent_content_sets)):
                for j in range(i + 1, len(sent_content_sets)):
                    s1, s2 = sent_content_sets[i], sent_content_sets[j]
                    if s1 and s2:
                        intersection = len(s1 & s2)
                        smaller = min(len(s1), len(s2))
                        if smaller > 0:
                            overlaps.append(intersection / smaller)
            
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                max_overlap = max(overlaps)
                # High overlap between sentences = redundancy
                redundancy_score = max(0.0, 1.0 - avg_overlap * 1.5 - max(0, max_overlap - 0.6) * 0.5)
            else:
                redundancy_score = 0.8
        else:
            redundancy_score = 0.8
        
        # ---- 4. Information density: unique content words / total words ----
        all_content = get_content_words(response)
        content_ratio = len(all_content) / max(word_count, 1)
        # Higher ratio = more diverse vocabulary = more information dense
        info_density_score = min(1.0, content_ratio * 3.0)  # scale up, cap at 1.0
        
        # ---- 5. Compression ratio proxy: character count vs word count ----
        char_count = len(response_stripped)
        avg_word_len = char_count / max(word_count, 1)
        # Very long average word length might indicate jargon/complexity
        # Very short might indicate shallow content
        if 4.0 <= avg_word_len <= 6.5:
            word_len_score = 1.0
        elif avg_word_len < 4.0:
            word_len_score = 0.7
        else:
            word_len_score = max(0.5, 1.0 - 0.1 * (avg_word_len - 6.5))
        
        # ---- 6. Passive voice estimation ----
        passive_patterns = [
            r'\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|written|spoken|taken|given|made|done|seen|known|found|shown|told|thought|felt|left|kept|held|brought|set|put|run|come|become|gone|got|gotten)\b',
        ]
        passive_count = 0
        for pat in passive_patterns:
            passive_count += len(re.findall(pat, response_lower))
        
        passive_ratio = passive_count / max(num_sentences, 1)
        passive_score = max(0.5, 1.0 - passive_ratio * 0.4)
        
        # ---- 7. Parenthetical/aside density ----
        parens = len(re.findall(r'\([^)]{5,}\)', response))
        dashes = len(re.findall(r'--[^-]+--|—[^—]+—', response))
        aside_count = parens + dashes
        aside_ratio = aside_count / max(num_sentences, 1)
        aside_score = max(0.5, 1.0 - aside_ratio * 0.3)
        
        # ---- 8. Directness: Does the response start addressing the query quickly? ----
        # Check if first sentence contains relevant query terms
        query_content = get_content_words(query)
        first_sent_content = get_content_words(sentences[0]) if sentences else set()
        if query_content and first_sent_content:
            first_sent_relevance = len(query_content & first_sent_content) / max(len(query_content), 1)
            directness_score = min(1.0, 0.5 + first_sent_relevance)
        else:
            directness_score = 0.6
        
        # ---- 9. Excessive preamble detection ----
        preamble_patterns = [
            r'^(sure|okay|great question|that\'s a great question|good question|well|so|alright)',
            r'^(i\'d be happy to|i can help|let me|i\'ll try)',
            r'^(thank you for|thanks for)',
        ]
        has_preamble = False
        first_50 = response_lower[:80]
        for pat in preamble_patterns:
            if re.search(pat, first_50):
                has_preamble = True
                break
        preamble_penalty = 0.95 if has_preamble else 1.0
        
        # ---- 10. Response length appropriateness ----
        # Not too short (might be uninformative), not too bloated
        query_words = re.findall(r"[a-zA-Z']+", query)
        query_len = len(query_words)
        
        # Reasonable response length relative to query complexity
        if word_count < 10:
            length_score = 0.4  # probably too terse
        elif word_count < 20:
            length_score = 0.6
        elif word_count < 300:
            length_score = 1.0
        elif word_count < 500:
            length_score = 0.9
        else:
            length_score = max(0.6, 0.9 - 0.001 * (word_count - 500))
        
        # ---- 11. Repetition of specific words (excluding stop words) ----
        content_words_list = [w for w in words_lower if w in all_content or len(w) > 3]
        if content_words_list:
            word_freq = Counter(content_words_list)
            total_cw = len(content_words_list)
            # Calculate how much of the text is dominated by repeated words
            max_freq = max(word_freq.values())
            top_word_dominance = max_freq / total_cw
            # Also check number of words appearing 3+ times
            repeated_words = sum(1 for w, c in word_freq.items() if c >= 3)
            repeat_ratio = repeated_words / max(len(word_freq), 1)
            
            repetition_score = max(0.3, 1.0 - top_word_dominance * 1.5 - repeat_ratio * 0.5)
        else:
            repetition_score = 0.7
        
        # ---- 12. Code block bonus (for technical queries) ----
        has_code = bool(re.search(r'```', response))
        query_is_technical = bool(re.search(r'(?:code|sql|function|class|table|create|select|program|script|api|html|css|python|java)', query.lower()))
        code_bonus = 0.3 if (has_code and query_is_technical) else 0.0
        
        # ---- Combine scores ----
        # Weighted combination
        score = (
            sent_len_score * 1.2 +
            variance_score * 0.6 +
            filler_score * 1.5 +
            redundancy_score * 1.8 +
            info_density_score * 1.5 +
            word_len_score * 0.5 +
            passive_score * 0.5 +
            aside_score * 0.3 +
            directness_score * 1.0 +
            length_score * 1.2 +
            repetition_score * 1.0 +
            code_bonus
        )
        
        total_weight = 1.2 + 0.6 + 1.5 + 1.8 + 1.5 + 0.5 + 0.5 + 0.3 + 1.0 + 1.2 + 1.0  # = 11.1
        
        normalized = score / total_weight  # 0 to ~1.1
        normalized *= preamble_penalty
        
        # Scale to 0-10
        final_score = round(normalized * 10, 2)
        final_score = max(0.0, min(10.0, final_score))
        
        return final_score
        
    except Exception:
        # Fallback: return a middling score
        try:
            if response and len(response.strip()) > 20:
                return 5.0
            return 2.0
        except Exception:
            return 3.0