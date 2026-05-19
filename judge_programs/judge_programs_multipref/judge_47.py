def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, signal-to-noise ratio,
    and structural efficiency metrics. Uses compression ratio estimation, function word
    density, clause complexity analysis, and directness of engagement with the query.
    
    This variant focuses on:
    - Information density (content words vs function words ratio)
    - Compression ratio proxy (unique information per unit length)
    - Direct query engagement (how quickly the response addresses the query)
    - Clause complexity (subordinate clause indicators)
    - Filler/padding phrase detection
    - Sentence-level entropy/variation
    - Redundancy detection via shingling similarity between segments
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        words = re.findall(r'[a-zA-Z]+', response.lower())
        if len(words) < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # === 1. FUNCTION WORD DENSITY (lower is more information-dense) ===
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'although',
            'though', 'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'also',
            'about', 'up', 'really', 'actually', 'basically', 'essentially',
        }
        
        func_count = sum(1 for w in words if w in function_words)
        func_density = func_count / len(words)
        # Ideal function word density is around 0.40-0.50; too high means bloated
        # Score: penalize if above 0.55
        density_score = max(0, 1.0 - max(0, func_density - 0.42) * 4.0)
        
        # === 2. FILLER/PADDING PHRASE DETECTION ===
        filler_phrases = [
            r'\bof course\b', r'\bit is important to note that\b',
            r'\bit is worth noting that\b', r'\bit should be noted that\b',
            r'\bneedless to say\b', r'\bas a matter of fact\b',
            r'\bin other words\b', r'\bthat being said\b',
            r'\bat the end of the day\b', r'\ball in all\b',
            r'\bto be honest\b', r'\bto tell you the truth\b',
            r'\bin my opinion\b', r'\bi think that\b',
            r'\bas you may know\b', r'\bas we all know\b',
            r'\bit goes without saying\b', r'\bfor what it\'s worth\b',
            r'\bwith that being said\b', r'\bhaving said that\b',
            r'\bin terms of\b', r'\bwhen it comes to\b',
            r'\bthe fact that\b', r'\bdue to the fact that\b',
            r'\bin order to\b', r'\bfor the purpose of\b',
            r'\bthere are several\b', r'\bthere are many\b',
            r'\bthere are a number of\b', r'\bit can be\b',
            r'\bthis is because\b', r'\bthe reason is\b',
            r'\blet me\b', r'\blet\'s\b',
            r'\bhere are some\b', r'\bhere\'s\b',
            r'\bgreat question\b', r'\bthat\'s a great\b',
            r'\bthat\'s an excellent\b', r'\ba classic\b',
            r'\babsolutely\b', r'\bdefinitely\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_per_sentence = filler_count / num_sentences
        filler_score = max(0, 1.0 - filler_per_sentence * 0.5)
        
        # === 3. REDUNDANCY via SHINGLING between sentence halves ===
        # Split response into two halves and measure content overlap
        def get_shingles(text, k=3):
            text_words = re.findall(r'[a-z]+', text.lower())
            if len(text_words) < k:
                return set()
            return set(tuple(text_words[i:i+k]) for i in range(len(text_words) - k + 1))
        
        mid = len(response) // 2
        first_half = response[:mid]
        second_half = response[mid:]
        
        shingles1 = get_shingles(first_half, 4)
        shingles2 = get_shingles(second_half, 4)
        
        if shingles1 and shingles2:
            jaccard = len(shingles1 & shingles2) / len(shingles1 | shingles2)
        else:
            jaccard = 0.0
        
        # Some overlap is expected (topic consistency), but high overlap = redundancy
        redundancy_score = max(0, 1.0 - max(0, jaccard - 0.05) * 5.0)
        
        # === 4. SENTENCE-LEVEL INFORMATION ENTROPY ===
        # Measure variation in sentence lengths (good writing has varied rhythm)
        sent_lengths = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            cv = math.sqrt(variance) / max(mean_len, 1)  # coefficient of variation
            # Moderate variation is good (0.3-0.7), too uniform or too chaotic is bad
            if cv < 0.15:
                rhythm_score = 0.6  # too uniform/monotonous
            elif cv > 1.2:
                rhythm_score = 0.5  # too chaotic
            else:
                rhythm_score = min(1.0, 0.6 + cv * 0.6)
        else:
            rhythm_score = 0.5
        
        # === 5. DIRECTNESS OF ENGAGEMENT ===
        # How quickly does the response get to the point?
        # Check first sentence for actual content vs preamble
        first_sent = sentences[0] if sentences else ""
        first_words = re.findall(r'[a-z]+', first_sent.lower())
        
        preamble_patterns = [
            r'^(that\'s a |what a |great |excellent |good |awesome |wonderful )',
            r'^(sure|certainly|absolutely|of course|definitely)',
            r'^(i\'d be happy to|i\'m glad you|thanks for)',
        ]
        
        has_preamble = any(re.search(p, first_sent.lower().strip()) for p in preamble_patterns)
        
        # Extract key content words from query
        query_words = set(re.findall(r'[a-z]+', query.lower())) - function_words
        first_sent_words = set(first_words) - function_words
        
        if query_words:
            query_engagement = len(query_words & first_sent_words) / len(query_words)
        else:
            query_engagement = 0.5
        
        # Penalize preamble but reward if it still contains query-relevant content
        directness_score = 0.5
        if has_preamble:
            directness_score = 0.4 + query_engagement * 0.3
        else:
            directness_score = 0.6 + query_engagement * 0.4
        
        # === 6. STRUCTURAL FORMATTING EFFICIENCY ===
        # Check for use of formatting that aids clarity
        has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_colon_structure = bool(re.search(r':\s*\n', response))
        
        structure_score = 0.5
        total_words = len(words)
        
        # Formatting is beneficial for longer responses
        if total_words > 60:
            if has_numbered:
                structure_score += 0.2
            if has_bold:
                structure_score += 0.15
            if has_colon_structure:
                structure_score += 0.1
        else:
            # Short responses don't need heavy formatting
            structure_score = 0.7
        
        structure_score = min(1.0, structure_score)
        
        # === 7. COMPRESSION RATIO PROXY ===
        # Unique words / total words — higher means more diverse vocabulary per word
        unique_words = set(words)
        compression = len(unique_words) / len(words)
        # Short texts naturally have higher compression; normalize
        expected_compression = 1.0 / (1.0 + math.log(max(len(words), 1)) * 0.15)
        compression_ratio = compression / max(expected_compression, 0.01)
        compression_score = min(1.0, compression_ratio * 0.7)
        
        # === 8. SUBORDINATE CLAUSE COMPLEXITY ===
        # Too many subordinate clauses make text hard to follow
        subordinators = [
            r'\bwhich\b', r'\bwhereas\b', r'\balthough\b', r'\bwhereby\b',
            r'\bwherein\b', r'\bnotwithstanding\b', r'\binasmuch\b',
            r'\bprovided that\b', r'\bgiven that\b',
        ]
        
        subord_count = sum(len(re.findall(p, response_lower)) for p in subordinators)
        subord_per_sent = subord_count / num_sentences
        complexity_score = max(0, 1.0 - subord_per_sent * 0.3)
        
        # === 9. AVERAGE SENTENCE LENGTH PENALTY ===
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        # Ideal: 10-20 words per sentence
        if avg_sent_len < 5:
            sent_len_score = 0.5
        elif avg_sent_len <= 20:
            sent_len_score = 1.0
        elif avg_sent_len <= 30:
            sent_len_score = 1.0 - (avg_sent_len - 20) * 0.03
        else:
            sent_len_score = max(0.3, 1.0 - (avg_sent_len - 20) * 0.025)
        
        # === 10. CONCISENESS — response length relative to query complexity ===
        query_content_words = len(query_words)
        response_content_words = len([w for w in words if w not in function_words])
        
        # A rough heuristic: very long responses for simple queries lose points
        if query_content_words > 0:
            expansion_ratio = response_content_words / max(query_content_words, 1)
            if expansion_ratio > 30:
                length_penalty = max(0.5, 1.0 - (expansion_ratio - 30) * 0.005)
            else:
                length_penalty = 1.0
        else:
            length_penalty = 0.9
        
        # === COMBINE SCORES ===
        # Weighted combination
        weights = {
            'density': 1.5,
            'filler': 1.8,
            'redundancy': 1.5,
            'rhythm': 0.8,
            'directness': 1.2,
            'structure': 1.0,
            'compression': 1.0,
            'complexity': 0.8,
            'sent_len': 1.0,
            'length_penalty': 0.8,
        }
        
        scores = {
            'density': density_score,
            'filler': filler_score,
            'redundancy': redundancy_score,
            'rhythm': rhythm_score,
            'directness': directness_score,
            'structure': structure_score,
            'compression': compression_score,
            'complexity': complexity_score,
            'sent_len': sent_len_score,
            'length_penalty': length_penalty,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
        final_score = (weighted_sum / total_weight) * 10.0
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 5.0