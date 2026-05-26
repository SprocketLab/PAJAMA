def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, signal-to-noise ratio,
    and structural efficiency metrics. Uses compression ratio estimation, 
    function word ratio, and directness scoring.
    
    Different from other variants by focusing on:
    - Information density (content word ratio, unique information per sentence)
    - Compression ratio proxy (how much could be compressed)
    - Directness/confidence of language
    - Structural formatting efficiency
    - Sentence-level information gain (new info per sentence)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # METRIC 1: Information Density (content vs function words)
        # ============================================================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'also', 'about', 'up', 'any', 'much', 'many'
        }
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        # Ideal content ratio is around 0.45-0.60
        info_density_score = min(content_ratio / 0.55, 1.0) * 10
        
        # ============================================================
        # METRIC 2: Compression Ratio Proxy (redundancy detection)
        # Measures how much unique information vs repeated information
        # ============================================================
        # Use character-level trigrams to estimate compressibility
        response_lower = response.lower()
        char_trigrams = [response_lower[i:i+3] for i in range(len(response_lower) - 2)]
        if char_trigrams:
            trigram_counts = Counter(char_trigrams)
            total_trigrams = len(char_trigrams)
            unique_trigrams = len(trigram_counts)
            # Higher ratio = less repetitive = better
            compression_ratio = unique_trigrams / total_trigrams if total_trigrams > 0 else 1
            # Also compute entropy of trigram distribution
            entropy = 0
            for count in trigram_counts.values():
                p = count / total_trigrams
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(unique_trigrams) if unique_trigrams > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            compression_score = (compression_ratio * 0.4 + normalized_entropy * 0.6) * 10
        else:
            compression_score = 5.0
        
        # ============================================================
        # METRIC 3: Sentence-level Information Gain
        # Each sentence should introduce new concepts, not repeat
        # ============================================================
        sentence_word_sets = []
        for s in sentences:
            s_words = set(re.findall(r'[a-zA-Z]+', s.lower()))
            s_content = s_words - function_words
            sentence_word_sets.append(s_content)
        
        if len(sentence_word_sets) > 1:
            cumulative = set()
            new_info_ratios = []
            for s_set in sentence_word_sets:
                if len(s_set) > 0:
                    new_words = s_set - cumulative
                    new_ratio = len(new_words) / len(s_set) if len(s_set) > 0 else 0
                    new_info_ratios.append(new_ratio)
                    cumulative.update(s_set)
            
            if new_info_ratios:
                avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
            else:
                avg_new_info = 0.5
            info_gain_score = avg_new_info * 10
        else:
            info_gain_score = 6.0
        
        # ============================================================
        # METRIC 4: Directness and Confidence
        # Penalize wishy-washy, indirect, or overly cautious language
        # ============================================================
        weak_phrases = [
            r'\bi think\b', r'\bi believe\b', r'\bit seems\b', r'\bperhaps\b',
            r'\bmaybe\b', r'\bpossibly\b', r'\bmight be\b', r'\bcould be\b',
            r'\bsort of\b', r'\bkind of\b', r'\bmore or less\b',
            r'\bin my opinion\b', r'\bit is important to note\b',
            r'\bit should be noted\b', r'\bit is worth mentioning\b',
            r'\bneedless to say\b', r'\bgoes without saying\b',
            r'\bas you know\b', r'\bas we all know\b',
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b',
            r'\bliterally\b', r'\bhonestly\b', r'\bfrankly\b',
            r'\bin other words\b', r'\bthat is to say\b',
            r'\bwhat i mean is\b', r'\bto put it another way\b',
        ]
        
        weak_count = 0
        for pattern in weak_phrases:
            weak_count += len(re.findall(pattern, response_lower))
        
        weak_ratio = weak_count / num_sentences
        directness_score = max(0, 10 - weak_ratio * 5)
        
        # ============================================================
        # METRIC 5: Filler and Bloat Detection
        # Detect empty calories in text
        # ============================================================
        filler_phrases = [
            r'\bin order to\b',  # just use "to"
            r'\bdue to the fact that\b',  # just use "because"
            r'\bat this point in time\b',  # just use "now"
            r'\bfor the purpose of\b',  # just use "for" or "to"
            r'\bin the event that\b',  # just use "if"
            r'\bwith regard to\b',  # just use "about"
            r'\bin terms of\b',
            r'\bon the other hand\b',
            r'\bit is important to\b',
            r'\bit is essential to\b',
            r'\bit is crucial to\b',
            r'\bit is necessary to\b',
            r'\bthere are several\b',
            r'\bthere are many\b',
            r'\bthere are a number of\b',
            r'\ba wide range of\b',
            r'\ba variety of\b',
            r'\bin addition to this\b',
            r'\bfurthermore\b',
            r'\bmoreover\b',
            r'\bnevertheless\b',
            r'\bnonetheless\b',
            r'\bhowever it is\b',
            r'\bthat being said\b',
            r'\bhaving said that\b',
        ]
        
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / num_sentences
        bloat_score = max(0, 10 - filler_ratio * 4)
        
        # ============================================================
        # METRIC 6: Structural Clarity (formatting signals)
        # Reward clear structure but not excessively
        # ============================================================
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bullets = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
        has_headers = bool(re.search(r'^\s*#{1,4}\s|^\s*\*{2}.+\*{2}\s*$', response, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        
        # Count structural elements
        num_list_items = len(re.findall(r'^\s*(?:\d+[\.\)]|[-*•])\s', response, re.MULTILINE))
        
        structure_score = 5.0  # baseline
        if has_numbered_list or has_bullets:
            structure_score += 1.5
        if has_headers:
            structure_score += 1.0
        if has_bold:
            structure_score += 0.5
        
        # Penalize if ALL structure and no prose (feels like just a skeleton)
        prose_lines = [l for l in response.split('\n') if l.strip() and not re.match(r'^\s*(?:\d+[\.\)]|[-*•]|#{1,4})\s', l)]
        if num_list_items > 0:
            prose_to_list_ratio = len(prose_lines) / (num_list_items + len(prose_lines))
            if prose_to_list_ratio < 0.15:
                structure_score -= 1.0  # Too listy, no explanation
        
        structure_score = min(structure_score, 10.0)
        
        # ============================================================
        # METRIC 7: Average Sentence Length Penalty
        # Very long sentences hurt clarity; very short ones may lack substance
        # ============================================================
        sent_lengths = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            if s_words:
                sent_lengths.append(len(s_words))
        
        if sent_lengths:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            # Ideal range: 10-20 words per sentence
            if avg_sent_len < 5:
                length_score = 4.0
            elif avg_sent_len <= 10:
                length_score = 6.0 + (avg_sent_len - 5) * 0.6
            elif avg_sent_len <= 20:
                length_score = 9.0
            elif avg_sent_len <= 30:
                length_score = 9.0 - (avg_sent_len - 20) * 0.2
            else:
                length_score = max(3.0, 5.0 - (avg_sent_len - 30) * 0.1)
            
            # Also penalize high variance in sentence length
            if len(sent_lengths) > 1:
                mean_len = sum(sent_lengths) / len(sent_lengths)
                variance = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len if mean_len > 0 else 0
                # Moderate variation is good (0.3-0.6), extreme is bad
                if cv > 1.0:
                    length_score -= 1.0
        else:
            length_score = 5.0
        
        # ============================================================
        # METRIC 8: Opening Directness
        # Does the response get to the point quickly?
        # ============================================================
        first_100_chars = response_lower[:150]
        
        # Penalize generic openings
        generic_openings = [
            r'^(that\'s a (?:great|good|excellent|wonderful) (?:question|idea|thought))',
            r'^(great question)',
            r'^(what a (?:great|good|excellent|wonderful))',
            r'^(absolutely)',
            r'^(of course)',
            r'^(sure thing)',
            r'^(certainly!?\s+(?:i\'d be happy|i would be happy|let me))',
        ]
        
        opening_score = 8.0
        for pattern in generic_openings:
            if re.search(pattern, first_100_chars):
                opening_score -= 1.5
                break
        
        # Reward responses that address the query topic early
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - function_words if query else set()
        first_sentence_words = set(re.findall(r'[a-zA-Z]+', sentences[0].lower())) if sentences else set()
        
        if query_words and first_sentence_words:
            query_overlap = len(query_words & first_sentence_words) / len(query_words) if query_words else 0
            opening_score += min(query_overlap * 3, 2.0)
        
        opening_score = min(opening_score, 10.0)
        
        # ============================================================
        # METRIC 9: Repetition at phrase level (2-4 word phrases)
        # ============================================================
        def get_ngrams(word_list, n):
            return [' '.join(word_list[i:i+n]) for i in range(len(word_list) - n + 1)]
        
        # Filter out structural words for phrase repetition
        content_word_sequence = [w for w in words if len(w) > 2]
        
        phrase_repetition_penalty = 0
        for n in [2, 3, 4]:
            ngrams = get_ngrams(content_word_sequence, n)
            if ngrams:
                ngram_counts = Counter(ngrams)
                repeated = sum(1 for c in ngram_counts.values() if c > 2)
                phrase_repetition_penalty += repeated * (n - 1) * 0.3
        
        repetition_score = max(0, 10 - phrase_repetition_penalty)
        
        # ============================================================
        # METRIC 10: Response-to-Query Relevance Efficiency
        # How efficiently does the response address the query?
        # ============================================================
        if query_words and content_words:
            content_word_set = set(content_words)
            relevance_overlap = len(query_words & content_word_set) / len(query_words) if query_words else 0
            # Efficiency: relevant content per total words
            efficiency = relevance_overlap / (total_words / 100) if total_words > 0 else 0
            relevance_efficiency_score = min(efficiency * 15, 10.0)
        else:
            relevance_efficiency_score = 5.0
        
        # ============================================================
        # COMBINE SCORES with weights
        # ============================================================
        weights = {
            'info_density': 0.10,
            'compression': 0.08,
            'info_gain': 0.15,
            'directness': 0.12,
            'bloat': 0.12,
            'structure': 0.10,
            'sent_length': 0.08,
            'opening': 0.10,
            'repetition': 0.10,
            'relevance_efficiency': 0.05,
        }
        
        scores = {
            'info_density': info_density_score,
            'compression': compression_score,
            'info_gain': info_gain_score,
            'directness': directness_score,
            'bloat': bloat_score,
            'structure': structure_score,
            'sent_length': length_score,
            'opening': opening_score,
            'repetition': repetition_score,
            'relevance_efficiency': relevance_efficiency_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
    
    except Exception:
        return 5.0