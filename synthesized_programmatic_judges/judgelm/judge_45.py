def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, structural coherence,
    and signal-to-noise ratio analysis.
    
    This variant focuses on:
    1. Information density (unique content words per total words ratio)
    2. Structural coherence (sentence-to-sentence logical flow via shared topic words)
    3. Signal-to-noise ratio (meaningful content vs filler/boilerplate)
    4. Response-query alignment (does the response address the query?)
    5. Compression ratio (how efficiently information is conveyed)
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
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses (under 2 chars) are almost always bad
        if len(response_stripped) < 3:
            return 0.5
        
        # ---- Tokenization helpers ----
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+', text.lower())
        
        def get_sentences(text):
            # Split on sentence-ending punctuation or newlines
            sents = re.split(r'[.!?]+|\n+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 1]
        
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'this', 'that', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'well', 'like', 'even', 'back', 'much', 'get', 'got',
            'make', 'made', 'know', 'known', 'think', 'see', 'come', 'go', 'take',
            'say', 'said'
        }
        
        FILLER_PHRASES = [
            'it is important to note that', 'it should be noted that',
            'it is worth mentioning that', 'as a matter of fact',
            'in order to', 'at the end of the day', 'for what it is worth',
            'it goes without saying', 'needless to say', 'as you can see',
            'basically', 'essentially', 'actually', 'literally',
            'in terms of', 'with respect to', 'in regard to',
            'as previously mentioned', 'as stated above', 'as noted earlier',
            'it is clear that', 'it is evident that', 'it is obvious that',
            'the fact that', 'due to the fact that', 'in light of the fact that',
            'there is no doubt that', 'without a doubt',
        ]
        
        response_lower = response.lower()
        words = tokenize(response)
        query_words = tokenize(query)
        
        if len(words) == 0:
            return 0.5
        
        content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        query_content = set(w for w in query_words if w not in STOP_WORDS and len(w) > 2)
        
        sentences = get_sentences(response)
        
        # ---- FEATURE 1: Information Density ----
        # Ratio of unique content words to total words
        if len(words) > 0:
            unique_content = set(content_words)
            info_density = len(unique_content) / len(words)
        else:
            info_density = 0.0
        
        # ---- FEATURE 2: Repetition Detection via Content Word Frequency Entropy ----
        # High entropy = diverse vocabulary; low entropy = repetitive
        content_freq = Counter(content_words)
        if len(content_freq) > 1:
            total_cw = sum(content_freq.values())
            probs = [c / total_cw for c in content_freq.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(content_freq))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        elif len(content_freq) == 1:
            normalized_entropy = 0.3  # Single content word repeated
        else:
            normalized_entropy = 0.0
        
        # ---- FEATURE 3: Sentence-level Redundancy ----
        # Measure how much consecutive sentences repeat each other's content words
        sent_redundancy = 0.0
        if len(sentences) > 1:
            sent_content_sets = []
            for s in sentences:
                s_words = set(w for w in tokenize(s) if w not in STOP_WORDS and len(w) > 2)
                sent_content_sets.append(s_words)
            
            overlap_scores = []
            for i in range(len(sent_content_sets) - 1):
                s1 = sent_content_sets[i]
                s2 = sent_content_sets[i + 1]
                if len(s1) > 0 and len(s2) > 0:
                    union = s1 | s2
                    if len(union) > 0:
                        overlap = len(s1 & s2) / len(union)
                        overlap_scores.append(overlap)
            
            if overlap_scores:
                sent_redundancy = sum(overlap_scores) / len(overlap_scores)
        
        # ---- FEATURE 4: Filler Phrase Density ----
        filler_count = 0
        for phrase in FILLER_PHRASES:
            filler_count += response_lower.count(phrase)
        
        filler_density = filler_count / max(len(sentences), 1)
        
        # ---- FEATURE 5: Query-Response Alignment ----
        # How well does the response address the query content?
        if len(query_content) > 0:
            response_content_set = set(content_words)
            alignment = len(query_content & response_content_set) / len(query_content)
        else:
            alignment = 0.5  # Neutral if no query content
        
        # ---- FEATURE 6: Structural Noise Detection ----
        # Detect HTML tags, code artifacts, repeated patterns, gibberish
        noise_score = 0.0
        
        # HTML/code tags
        html_tags = len(re.findall(r'<[^>]+>', response))
        noise_score += min(html_tags * 0.05, 0.4)
        
        # Repeated exact lines
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counter = Counter(lines)
            repeated_lines = sum(c - 1 for c in line_counter.values() if c > 1)
            noise_score += min(repeated_lines / len(lines), 0.5)
        
        # Long runs of repeated characters or patterns
        repeated_chars = len(re.findall(r'(.)\1{4,}', response))
        noise_score += min(repeated_chars * 0.1, 0.3)
        
        # Detect "Input:" "Output:" patterns that suggest template artifacts
        template_artifacts = len(re.findall(r'(?:Input|Output|Question|Answer)\s*:', response))
        if template_artifacts > 2:
            noise_score += min((template_artifacts - 2) * 0.05, 0.3)
        
        noise_score = min(noise_score, 1.0)
        
        # ---- FEATURE 7: Compression Efficiency ----
        # Penalize very long responses relative to query length; reward concise ones
        # But also penalize extremely short responses that don't convey info
        query_len = max(len(query_words), 1)
        response_len = len(words)
        
        # Optimal response length is roughly 1-5x query length for most tasks
        length_ratio = response_len / query_len
        
        if response_len < 3:
            length_score = 0.2
        elif length_ratio < 0.3:
            length_score = 0.3  # Too short
        elif length_ratio <= 8:
            length_score = 1.0  # Good range
        elif length_ratio <= 15:
            length_score = 0.8  # Getting verbose
        else:
            length_score = max(0.3, 1.0 - (length_ratio - 15) * 0.02)
        
        # ---- FEATURE 8: Sentence Length Variance ----
        # Good writing has varied but controlled sentence lengths
        if len(sentences) > 1:
            sent_lengths = [len(tokenize(s)) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Penalize very long average sentence length (hard to read)
            if avg_sent_len > 35:
                sent_len_penalty = min((avg_sent_len - 35) * 0.02, 0.3)
            else:
                sent_len_penalty = 0.0
            
            # Some variance is good, extreme variance is bad
            if len(sent_lengths) > 1:
                variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / avg_sent_len if avg_sent_len > 0 else 0
                # CV between 0.2-0.6 is ideal
                if cv < 0.1:
                    variety_score = 0.7  # Too uniform
                elif cv <= 0.7:
                    variety_score = 1.0
                else:
                    variety_score = max(0.5, 1.0 - (cv - 0.7) * 0.5)
            else:
                variety_score = 0.8
        else:
            sent_len_penalty = 0.0
            variety_score = 0.8
        
        # ---- FEATURE 9: Substantiveness ----
        # Content word ratio - higher means more substance per word
        content_ratio = len(content_words) / len(words) if len(words) > 0 else 0
        
        # ---- FEATURE 10: Exact Phrase Repetition ----
        # Check for repeated 4-grams (stronger repetition signal)
        phrase_repetition_penalty = 0.0
        if len(words) >= 8:
            four_grams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
            fg_counter = Counter(four_grams)
            repeated_4grams = sum(c - 1 for c in fg_counter.values() if c > 1)
            phrase_repetition_penalty = min(repeated_4grams / max(len(four_grams), 1) * 3, 1.0)
        
        # ---- Combine Features into Final Score ----
        # Base score starts at 5 (neutral)
        score = 5.0
        
        # Information density: +0 to +1.5
        score += info_density * 1.5
        
        # Vocabulary diversity (entropy): +0 to +1.5
        score += normalized_entropy * 1.5
        
        # Low sentence redundancy: +0 to +1.0
        score += (1.0 - sent_redundancy) * 1.0
        
        # Low filler density: -0 to -1.0
        score -= filler_density * 1.0
        
        # Query alignment: +0 to +1.0
        score += alignment * 1.0
        
        # Low noise: -0 to -2.5
        score -= noise_score * 2.5
        
        # Length appropriateness: +0 to +0.8
        score += (length_score - 0.5) * 1.6
        
        # Sentence variety: +0 to +0.5
        score += (variety_score - 0.5) * 1.0
        
        # Sentence length penalty: -0 to -0.5
        score -= sent_len_penalty
        
        # Content ratio: +0 to +0.8
        score += content_ratio * 0.8
        
        # Phrase repetition: -0 to -2.0
        score -= phrase_repetition_penalty * 2.0
        
        # ---- Edge case adjustments ----
        
        # Single word/very terse responses to complex queries
        if len(words) <= 2 and len(query_words) > 5:
            score = min(score, 2.0)
        
        # Responses that are just punctuation or whitespace-heavy
        alpha_ratio = sum(1 for c in response if c.isalpha()) / max(len(response), 1)
        if alpha_ratio < 0.3:
            score -= 2.0
        
        # Bonus for responses that are clean and direct (no artifacts)
        if noise_score == 0 and filler_density == 0 and phrase_repetition_penalty == 0:
            if 0.3 < info_density < 0.8 and normalized_entropy > 0.7:
                score += 0.5
        
        # Clamp to [0, 10]
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 3.0