def judging_function(query, response):
    """
    Evaluates clarity and conciseness using:
    - Information density (content words vs function words ratio)
    - Sentence structure variance (penalize monotonous or overly complex structures)
    - Redundancy detection via sliding window semantic overlap
    - Precision of language (specific/concrete words vs vague/abstract)
    - Signal-to-noise ratio (meaningful content vs filler phrases)
    - Compression ratio heuristic
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 3]
        
        if not words:
            return 0.0
        
        word_count = len(words)
        sent_count = max(len(sentences), 1)
        
        # =====================
        # 1. FUNCTION WORD RATIO (information density)
        # =====================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'must', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'it',
            'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'if', 'than', 'too', 'very', 'just',
            'about', 'up', 'there', 'here', 'when', 'where', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'also', 'any', 'because',
            'while', 'although', 'though', 'since', 'until', 'unless', 'whether'
        }
        
        func_count = sum(1 for w in words if w in function_words)
        content_ratio = 1.0 - (func_count / max(word_count, 1))
        # Ideal content ratio is around 0.45-0.60
        info_density_score = 1.0 - abs(content_ratio - 0.52) * 2.5
        info_density_score = max(0, min(1, info_density_score))
        
        # =====================
        # 2. FILLER / BLOAT PHRASES
        # =====================
        filler_phrases = [
            r'\bit is important to note that\b', r'\bit should be noted that\b',
            r'\bin order to\b', r'\bdue to the fact that\b', r'\bat the end of the day\b',
            r'\bfor all intents and purposes\b', r'\bneedless to say\b',
            r'\bit goes without saying\b', r'\bas a matter of fact\b',
            r'\bin terms of\b', r'\bwith regard to\b', r'\bwith respect to\b',
            r'\bin the event that\b', r'\bfor the purpose of\b',
            r'\bin the process of\b', r'\bon the other hand\b',
            r'\bthe fact that\b', r'\bit is worth mentioning\b',
            r'\bas you can see\b', r'\bas mentioned earlier\b',
            r'\bas previously stated\b', r'\bin my opinion\b',
            r'\bi think that\b', r'\bi believe that\b',
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b',
            r'\bliterally\b', r'\bobviously\b', r'\bclearly\b',
            r'\bcertainly\b', r'\bdefinitely\b', r'\babsolutely\b',
            r'\bsimply put\b', r'\bto be honest\b', r'\bquite frankly\b',
            r'\ball things considered\b', r'\bby and large\b',
            r'\bmore or less\b', r'\bso to speak\b',
            r'\bkind of\b', r'\bsort of\b', r'\ba lot of\b',
            r'\bplease note that\b', r'\bkeep in mind that\b',
            r'\bit is also worth noting\b', r'\bwhat i mean is\b',
        ]
        
        resp_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, resp_lower))
        
        filler_penalty = min(filler_count * 0.08, 0.5)
        filler_score = 1.0 - filler_penalty
        
        # =====================
        # 3. SENTENCE LENGTH ANALYSIS (clarity via structure)
        # =====================
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if sent_word_counts:
            avg_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            # Ideal average sentence length: 12-22 words
            if avg_sent_len < 5:
                sent_len_score = 0.4
            elif avg_sent_len <= 12:
                sent_len_score = 0.6 + 0.4 * ((avg_sent_len - 5) / 7)
            elif avg_sent_len <= 22:
                sent_len_score = 1.0
            elif avg_sent_len <= 35:
                sent_len_score = 1.0 - 0.5 * ((avg_sent_len - 22) / 13)
            else:
                sent_len_score = 0.3
            
            # Sentence length VARIANCE — some variety is good, too much is bad
            if len(sent_word_counts) > 1:
                mean_sl = avg_sent_len
                variance = sum((x - mean_sl)**2 for x in sent_word_counts) / len(sent_word_counts)
                std_sl = math.sqrt(variance)
                cv = std_sl / max(mean_sl, 1)
                # Coefficient of variation between 0.3-0.6 is ideal
                if cv < 0.15:
                    variety_score = 0.6  # too monotonous
                elif cv <= 0.6:
                    variety_score = 0.6 + 0.4 * min((cv - 0.15) / 0.45, 1.0)
                elif cv <= 1.0:
                    variety_score = 1.0 - 0.3 * ((cv - 0.6) / 0.4)
                else:
                    variety_score = 0.5
            else:
                variety_score = 0.6
        else:
            sent_len_score = 0.5
            variety_score = 0.5
        
        # =====================
        # 4. SLIDING WINDOW REDUNDANCY DETECTION
        # =====================
        # Check if similar word sets appear in nearby sentences
        redundancy_score = 1.0
        if len(sentences) >= 2:
            sent_word_sets = []
            for s in sentences:
                s_words = set(re.findall(r'[a-zA-Z]{3,}', s.lower()))
                s_words -= function_words
                sent_word_sets.append(s_words)
            
            overlap_penalties = []
            window_size = 3
            for i in range(len(sent_word_sets)):
                for j in range(i + 1, min(i + window_size, len(sent_word_sets))):
                    s1, s2 = sent_word_sets[i], sent_word_sets[j]
                    if s1 and s2:
                        intersection = len(s1 & s2)
                        smaller = min(len(s1), len(s2))
                        if smaller > 0:
                            overlap = intersection / smaller
                            if overlap > 0.6:
                                overlap_penalties.append(overlap - 0.6)
            
            if overlap_penalties:
                avg_penalty = sum(overlap_penalties) / max(len(sentences) - 1, 1)
                redundancy_score = max(0.3, 1.0 - avg_penalty * 3.0)
        
        # =====================
        # 5. SPECIFICITY SCORE (concrete vs vague language)
        # =====================
        vague_words = {
            'thing', 'things', 'stuff', 'something', 'somehow', 'somewhat',
            'somewhere', 'someone', 'anything', 'everything', 'nothing',
            'whatever', 'whichever', 'whoever', 'various', 'several',
            'many', 'much', 'often', 'sometimes', 'usually', 'generally',
            'typically', 'probably', 'possibly', 'perhaps', 'maybe',
            'likely', 'unlikely', 'certain', 'uncertain', 'etc',
            'really', 'very', 'quite', 'rather', 'pretty', 'fairly',
            'somewhat', 'slightly', 'extremely', 'incredibly', 'amazingly',
            'great', 'good', 'bad', 'nice', 'fine', 'okay', 'interesting',
            'important', 'significant', 'relevant', 'appropriate'
        }
        
        vague_count = sum(1 for w in words if w in vague_words)
        vague_ratio = vague_count / max(word_count, 1)
        specificity_score = max(0.2, 1.0 - vague_ratio * 8.0)
        
        # Bonus for numbers, proper nouns, technical terms (specificity indicators)
        number_matches = len(re.findall(r'\b\d+\.?\d*\b', response))
        # Words with capitals (potential proper nouns/technical terms) - skip sentence starts
        capital_words = re.findall(r'(?<!\. )(?<!\.\n)(?<!^)[A-Z][a-z]+', response)
        specificity_bonus = min(0.15, (number_matches + len(capital_words)) * 0.01)
        specificity_score = min(1.0, specificity_score + specificity_bonus)
        
        # =====================
        # 6. RESPONSE COMPLETENESS / SUBSTANTIVENESS
        # =====================
        # Very short responses often lack substance
        if word_count < 10:
            substance_score = 0.2
        elif word_count < 25:
            substance_score = 0.4 + 0.03 * (word_count - 10)
        elif word_count < 50:
            substance_score = 0.85
        elif word_count <= 300:
            substance_score = 1.0
        elif word_count <= 500:
            substance_score = 1.0 - 0.002 * (word_count - 300)
        else:
            substance_score = max(0.4, 0.6 - 0.0005 * (word_count - 500))
        
        # =====================
        # 7. STRUCTURAL CLARITY INDICATORS
        # =====================
        structural_score = 0.5
        
        # Code blocks (good for technical queries)
        has_code = bool(re.search(r'```', response))
        has_technical_query = bool(re.search(r'(?:code|sql|function|create|select|table|program|script)', query.lower())) if query else False
        if has_code and has_technical_query:
            structural_score += 0.2
        
        # Lists/enumeration (structured = clear)
        has_list = bool(re.search(r'(?:^|\n)\s*[\-\*•]\s', response)) or bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        if has_list:
            structural_score += 0.15
        
        # Formatting markers: bold, italics (emphasis = clarity)
        has_emphasis = bool(re.search(r'\*[^*]+\*', response)) or bool(re.search(r'\*\*[^*]+\*\*', response))
        if has_emphasis:
            structural_score += 0.1
        
        # Multiple paragraphs for longer responses (good structure)
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        if word_count > 80 and len(paragraphs) >= 2:
            structural_score += 0.1
        elif word_count > 80 and len(paragraphs) == 1:
            structural_score -= 0.1  # wall of text penalty
        
        structural_score = max(0, min(1, structural_score))
        
        # =====================
        # 8. QUERY RELEVANCE (topical alignment without word overlap)
        # =====================
        relevance_score = 0.5  # default
        if query:
            query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
            query_content = query_words - function_words
            resp_words = set(re.findall(r'[a-zA-Z]{3,}', resp_lower))
            resp_content = resp_words - function_words
            
            if query_content:
                # What fraction of query content words appear in response
                coverage = len(query_content & resp_content) / len(query_content)
                # We want some coverage but not parrot-like repetition
                if coverage < 0.05:
                    relevance_score = 0.2
                elif coverage <= 0.5:
                    relevance_score = 0.4 + coverage * 1.0
                else:
                    relevance_score = 0.9
        
        # =====================
        # 9. UNIQUE INFORMATION RATE (compression heuristic)
        # =====================
        # Ratio of unique character trigrams to total trigrams
        if len(response) > 10:
            trigrams = [response[i:i+3] for i in range(len(response) - 2)]
            unique_trigrams = len(set(trigrams))
            total_trigrams = len(trigrams)
            compression_ratio = unique_trigrams / max(total_trigrams, 1)
            # Higher ratio = more unique info, less repetition
            # Short responses naturally have high ratio, long ones lower
            # Normalize by expected ratio for length
            expected_ratio = min(1.0, 50.0 / max(math.sqrt(total_trigrams), 1))
            normalized_compression = min(1.0, compression_ratio / max(expected_ratio, 0.01))
            compression_score = 0.3 + 0.7 * normalized_compression
        else:
            compression_score = 0.5
        
        # =====================
        # 10. DIRECTNESS (does it get to the point quickly?)
        # =====================
        directness_score = 1.0
        
        # Penalize responses that start with excessive preamble
        preamble_patterns = [
            r'^(?:well,?\s+)?(?:that\'s\s+)?(?:a\s+)?(?:great|good|excellent|interesting|wonderful)\s+question',
            r'^(?:sure|certainly|absolutely|of course)[,!]\s+(?:i\'d be happy to|let me|i can)',
            r'^thank you for (?:your|the|asking)',
            r'^i\'m glad you asked',
            r'^so,?\s+',
            r'^well,?\s+',
        ]
        for pattern in preamble_patterns:
            if re.search(pattern, resp_lower[:100]):
                directness_score -= 0.1
        
        directness_score = max(0.3, directness_score)
        
        # =====================
        # COMBINE SCORES
        # =====================
        weights = {
            'info_density': 0.10,
            'filler': 0.12,
            'sent_len': 0.10,
            'variety': 0.06,
            'redundancy': 0.14,
            'specificity': 0.10,
            'substance': 0.14,
            'structural': 0.06,
            'relevance': 0.08,
            'compression': 0.05,
            'directness': 0.05,
        }
        
        scores = {
            'info_density': info_density_score,
            'filler': filler_score,
            'sent_len': sent_len_score,
            'variety': variety_score,
            'redundancy': redundancy_score,
            'specificity': specificity_score,
            'substance': substance_score,
            'structural': structural_score,
            'relevance': relevance_score,
            'compression': compression_score,
            'directness': directness_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = final_score * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0