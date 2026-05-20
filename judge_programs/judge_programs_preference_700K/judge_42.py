def judging_function(query, response):
    """
    Evaluate clarity and conciseness of an LLM response.
    
    This variant focuses on:
    - Information density (ratio of meaningful content words to total words)
    - Sentence structure quality (average sentence length, variance)
    - Filler/hedge word penalization
    - Specificity signals (concrete details, examples, proper nouns)
    - Structural coherence (logical connectors, paragraph organization)
    - Redundancy detection via n-gram repetition analysis
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        if len(response_clean) < 5:
            return 0.5
        
        # Tokenize into words and sentences
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response_clean.lower())
        sentences = re.split(r'[.!?]+(?:\s|$)', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        if not words:
            return 0.5
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. FILLER / HEDGE WORD PENALTY ----
        filler_words = {
            'basically', 'essentially', 'actually', 'literally', 'really', 'very',
            'quite', 'somewhat', 'rather', 'perhaps', 'maybe', 'possibly',
            'just', 'simply', 'obviously', 'clearly', 'definitely', 'certainly',
            'honestly', 'frankly', 'personally', 'arguably', 'supposedly',
            'kind', 'sort', 'stuff', 'things', 'thing', 'like',
            'anyway', 'anyways', 'well', 'so', 'um', 'uh',
        }
        
        hedge_phrases = [
            'i think', 'i believe', 'i feel like', 'in my opinion',
            'it seems like', 'it appears that', 'more or less',
            'to be honest', 'to be fair', 'at the end of the day',
            'in a nutshell', 'as a matter of fact', 'the fact that',
            'it is important to note', 'it should be noted',
            'it is worth mentioning', 'needless to say',
            'as you know', 'as we all know', 'it goes without saying',
        ]
        
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / max(num_words, 1)
        
        response_lower = response_clean.lower()
        hedge_count = sum(response_lower.count(phrase) for phrase in hedge_phrases)
        hedge_penalty = min(hedge_count * 0.15, 1.5)
        
        filler_score = max(0, 1.0 - filler_ratio * 4.0 - hedge_penalty)
        
        # ---- 2. SENTENCE LENGTH QUALITY ----
        # Optimal average sentence length: 12-22 words
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if sent_word_counts:
            avg_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            # Score peaks around 15-18 words per sentence
            if avg_sent_len < 5:
                sent_len_score = 0.4
            elif avg_sent_len < 12:
                sent_len_score = 0.4 + 0.6 * (avg_sent_len - 5) / 7
            elif avg_sent_len <= 22:
                sent_len_score = 1.0
            elif avg_sent_len <= 35:
                sent_len_score = 1.0 - 0.5 * (avg_sent_len - 22) / 13
            else:
                sent_len_score = 0.3
            
            # Sentence length variance — moderate variety is good
            if len(sent_word_counts) > 1:
                mean_sl = sum(sent_word_counts) / len(sent_word_counts)
                variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
                std_dev = math.sqrt(variance)
                cv = std_dev / max(mean_sl, 1)
                # Coefficient of variation: 0.3-0.6 is ideal
                if cv < 0.1:
                    variety_score = 0.5  # Too monotonous
                elif cv < 0.3:
                    variety_score = 0.5 + 0.5 * (cv - 0.1) / 0.2
                elif cv <= 0.7:
                    variety_score = 1.0
                else:
                    variety_score = max(0.3, 1.0 - (cv - 0.7) * 0.5)
            else:
                variety_score = 0.6
        else:
            sent_len_score = 0.5
            variety_score = 0.5
        
        # ---- 3. SPECIFICITY / INFORMATION DENSITY ----
        # Detect concrete details: numbers, proper nouns, technical terms, examples
        
        # Numbers and quantitative info
        numbers = re.findall(r'\b\d+[\d,.]*\b', response_clean)
        number_density = min(len(numbers) / max(num_words, 1) * 30, 1.0)
        
        # Capitalized words (potential proper nouns, excluding sentence starts)
        cap_words = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]{2,}', response_clean)
        proper_noun_density = min(len(cap_words) / max(num_words, 1) * 15, 1.0)
        
        # Stop words ratio (lower = more information dense)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'than', 'too',
            'and', 'but', 'or', 'if', 'while', 'because', 'until', 'about',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / max(num_words, 1)
        # Ideal content ratio: 0.4-0.6
        info_density = min(content_ratio / 0.5, 1.2)
        
        specificity_score = 0.3 * number_density + 0.2 * proper_noun_density + 0.5 * info_density
        specificity_score = min(specificity_score, 1.0)
        
        # ---- 4. REDUNDANCY DETECTION ----
        # Check for repeated n-grams (trigrams and 4-grams)
        def get_ngrams(word_list, n):
            return [tuple(word_list[i:i+n]) for i in range(len(word_list) - n + 1)]
        
        redundancy_penalty = 0.0
        
        if num_words > 15:
            for n in [3, 4, 5]:
                ngrams = get_ngrams(words, n)
                if ngrams:
                    ngram_counts = Counter(ngrams)
                    repeated = sum(c - 1 for c in ngram_counts.values() if c > 1)
                    redundancy_ratio = repeated / max(len(ngrams), 1)
                    redundancy_penalty += redundancy_ratio * (n * 0.8)
        
        # Also check for repeated sentences or near-duplicate sentences
        if len(sentences) > 1:
            sent_set = set()
            dup_sents = 0
            for s in sentences:
                s_normalized = re.sub(r'\s+', ' ', s.lower().strip())
                if s_normalized in sent_set:
                    dup_sents += 1
                sent_set.add(s_normalized)
            redundancy_penalty += dup_sents * 0.3
        
        redundancy_score = max(0, 1.0 - redundancy_penalty)
        
        # ---- 5. COHERENCE / LOGICAL FLOW ----
        # Presence of logical connectors and transition words
        connectors = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'specifically',
            'for example', 'for instance', 'in contrast', 'on the other hand',
            'in addition', 'as a result', 'in particular', 'that said',
            'first', 'second', 'third', 'finally', 'also', 'thus',
            'because', 'since', 'although', 'while', 'whereas',
        ]
        
        connector_count = 0
        for conn in connectors:
            connector_count += response_lower.count(conn)
        
        # Ideal: roughly 1 connector per 2-3 sentences
        if num_sentences > 0:
            connector_ratio = connector_count / num_sentences
            if connector_ratio < 0.1:
                coherence_score = 0.4
            elif connector_ratio < 0.3:
                coherence_score = 0.4 + 0.6 * (connector_ratio - 0.1) / 0.2
            elif connector_ratio <= 1.0:
                coherence_score = 1.0
            else:
                coherence_score = max(0.5, 1.0 - (connector_ratio - 1.0) * 0.2)
        else:
            coherence_score = 0.4
        
        # ---- 6. RESPONSE LENGTH ADEQUACY ----
        # Responses should be substantial enough to be useful but not bloated
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_len = len(query_words)
        
        # Longer/more complex queries deserve longer responses
        # Base expected length scales with query complexity
        if query_len < 10:
            ideal_min, ideal_max = 30, 200
        elif query_len < 30:
            ideal_min, ideal_max = 50, 350
        elif query_len < 80:
            ideal_min, ideal_max = 60, 500
        else:
            ideal_min, ideal_max = 80, 600
        
        if num_words < ideal_min * 0.3:
            length_score = 0.2
        elif num_words < ideal_min:
            length_score = 0.2 + 0.8 * (num_words - ideal_min * 0.3) / (ideal_min * 0.7)
        elif num_words <= ideal_max:
            length_score = 1.0
        elif num_words <= ideal_max * 1.5:
            length_score = 1.0 - 0.3 * (num_words - ideal_max) / (ideal_max * 0.5)
        else:
            length_score = 0.5
        
        # ---- 7. DIRECTNESS ----
        # Penalize responses that start with unnecessary preamble
        directness_penalty = 0.0
        
        preamble_patterns = [
            r'^(sure|okay|ok|alright|great question|good question|interesting question)',
            r'^(i can help|i\'d be happy to|let me|i\'ll try)',
            r'^(that\'s a great|that\'s an interesting|that\'s a good)',
            r'^(thank you for|thanks for)',
            r'^(well,?\s)',
            r'^(so,?\s)',
        ]
        
        first_100 = response_lower[:100]
        for pattern in preamble_patterns:
            if re.search(pattern, first_100):
                directness_penalty += 0.1
        
        directness_score = max(0, 1.0 - directness_penalty)
        
        # ---- 8. ACTIVE VOICE / CLARITY INDICATORS ----
        # Passive voice indicators
        passive_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+(?:\w+ed|known|seen|done|made|given|taken|found)\b',
        ]
        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, response_lower))
        
        passive_ratio = passive_count / max(num_sentences, 1)
        voice_score = max(0.3, 1.0 - passive_ratio * 0.3)
        
        # ---- COMBINE SCORES ----
        # Weighted combination
        final_score = (
            filler_score * 1.5 +       # Penalize filler heavily
            sent_len_score * 1.2 +      # Sentence structure
            variety_score * 0.8 +       # Sentence variety
            specificity_score * 1.5 +   # Information density
            redundancy_score * 1.5 +    # Penalize redundancy
            coherence_score * 1.0 +     # Logical flow
            length_score * 1.2 +        # Appropriate length
            directness_score * 0.8 +    # Direct communication
            voice_score * 0.5          # Active voice preference
        )
        
        total_weight = 1.5 + 1.2 + 0.8 + 1.5 + 1.5 + 1.0 + 1.2 + 0.8 + 0.5  # = 10.0
        
        # Normalize to 0-10 scale
        normalized = (final_score / total_weight) * 10.0
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This makes the scoring more discriminative
        centered = normalized - 5.0
        spread = 5.0 / (1.0 + math.exp(-centered * 0.6))
        final = 2.5 + spread
        
        return round(max(0.0, min(10.0, final)), 3)
        
    except Exception:
        return 3.0