def judging_function(query, response):
    """
    Evaluates clarity and conciseness using information density, signal-to-noise ratio,
    and structural coherence metrics. Uses a fundamentally different approach based on:
    - Information density (unique content words per total words ratio)
    - Filler/weasel word detection with weighted penalties
    - Sentence-level coherence (topic consistency between adjacent sentences)
    - Precision of language (specific vs vague word usage)
    - Response-query alignment (relevance signal)
    - Compression ratio (how efficiently information is packed)
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
            return 0.5
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        if len(words) < 3:
            return 0.5
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 2]
        if not sentences:
            sentences = [response]
        
        # ---- METRIC 1: Information Density via Hapax Legomena Ratio ----
        # Words appearing exactly once carry more information
        word_counts = Counter(words)
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = hapax / len(word_counts) if word_counts else 0
        # Optimal hapax ratio is moderate (0.5-0.7) - too high means incoherent, too low means repetitive
        hapax_score = 1.0 - 2.0 * abs(hapax_ratio - 0.6)
        hapax_score = max(0, min(1, hapax_score))
        
        # ---- METRIC 2: Filler and Weasel Words (weighted by severity) ----
        severe_fillers = [
            'basically', 'actually', 'literally', 'honestly', 'really', 'very',
            'quite', 'rather', 'somewhat', 'perhaps', 'maybe', 'probably',
            'stuff', 'things', 'whatever', 'somehow', 'anyway', 'anyways'
        ]
        moderate_fillers = [
            'just', 'like', 'kind of', 'sort of', 'you know', 'i mean',
            'well', 'so', 'right', 'okay', 'um', 'uh', 'hmm'
        ]
        weasel_words = [
            'might', 'could', 'may', 'seems', 'appears', 'possibly',
            'it is said', 'some people say', 'it is thought', 'allegedly'
        ]
        dismissive_phrases = [
            'not a big deal', 'no big deal', 'get over it', 'move on',
            'just deal with it', 'toughen up', 'suck it up'
        ]
        
        response_lower = response.lower()
        
        severe_count = sum(response_lower.count(f) for f in severe_fillers)
        moderate_count = sum(response_lower.count(f) for f in moderate_fillers)
        weasel_count = sum(response_lower.count(f) for f in weasel_words)
        dismissive_count = sum(response_lower.count(f) for f in dismissive_phrases)
        
        filler_penalty = (severe_count * 1.5 + moderate_count * 0.8 + weasel_count * 1.0 + dismissive_count * 2.5) / max(len(words), 1)
        filler_score = max(0, 1.0 - filler_penalty * 8)
        
        # ---- METRIC 3: Sentence-level Coherence (Jaccard between adjacent sentences) ----
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
                'or', 'if', 'while', 'although', 'this', 'that', 'these', 'those',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
                'who', 'whom', 'about', 'up', 'also', 'much', 'even', 'still'
            }
            w = re.findall(r'[a-zA-Z]+', text.lower())
            return set(w) - stop_words
        
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                cw1 = get_content_words(sentences[i])
                cw2 = get_content_words(sentences[i + 1])
                if cw1 or cw2:
                    jaccard = len(cw1 & cw2) / len(cw1 | cw2) if (cw1 | cw2) else 0
                    coherence_scores.append(jaccard)
            avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
            # Optimal coherence is moderate (0.05-0.3) - too high means repetitive, too low means disjointed
            if avg_coherence < 0.05:
                coherence_metric = 0.3
            elif avg_coherence > 0.4:
                coherence_metric = max(0.3, 1.0 - (avg_coherence - 0.4) * 3)
            else:
                coherence_metric = 0.5 + avg_coherence * 1.5
        else:
            coherence_metric = 0.5
        
        # ---- METRIC 4: Specificity Score ----
        # Specific/concrete words vs vague/abstract words
        vague_words = {
            'thing', 'things', 'stuff', 'something', 'anything', 'everything',
            'someone', 'anyone', 'everyone', 'good', 'bad', 'nice', 'great',
            'fine', 'okay', 'interesting', 'important', 'significant', 'various',
            'different', 'certain', 'particular', 'general', 'specific', 'overall',
            'basically', 'essentially', 'fundamental', 'key', 'main', 'major'
        }
        
        content_words = get_content_words(response)
        vague_count = len(content_words & vague_words)
        specificity_score = max(0, 1.0 - (vague_count / max(len(content_words), 1)) * 5)
        
        # ---- METRIC 5: Query-Response Alignment (Relevance) ----
        query_content = get_content_words(query)
        response_content = get_content_words(response)
        
        if query_content and response_content:
            overlap = len(query_content & response_content)
            alignment = overlap / max(len(query_content), 1)
            alignment_score = min(1.0, alignment * 2.5)
        else:
            alignment_score = 0.3
        
        # ---- METRIC 6: Compression / Efficiency ----
        # Unique content words per sentence - measures how much new info each sentence adds
        seen_content = set()
        new_info_per_sentence = []
        for sent in sentences:
            cw = get_content_words(sent)
            new_words = cw - seen_content
            if cw:
                new_info_ratio = len(new_words) / len(cw)
                new_info_per_sentence.append(new_info_ratio)
            seen_content |= cw
        
        if new_info_per_sentence:
            avg_new_info = sum(new_info_per_sentence) / len(new_info_per_sentence)
            # High new info ratio means less repetition
            efficiency_score = min(1.0, avg_new_info * 1.2)
        else:
            efficiency_score = 0.5
        
        # ---- METRIC 7: Structural Clarity ----
        # Presence of organizational elements: numbered lists, colons, clear structure
        has_numbered_list = bool(re.search(r'\d+[\.\)]\s', response))
        has_colon_structure = response.count(':') >= 1
        has_paragraph_breaks = response.count('\n\n') >= 1
        
        structural_bonus = 0.0
        if has_numbered_list:
            structural_bonus += 0.15
        if has_colon_structure:
            structural_bonus += 0.05
        if has_paragraph_breaks:
            structural_bonus += 0.05
        structural_bonus = min(0.2, structural_bonus)
        
        # ---- METRIC 8: Sentence Variance (not just avg length) ----
        # Good writing has varied sentence lengths
        sent_word_counts = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
        if len(sent_word_counts) >= 2:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Moderate CV (0.2-0.6) is ideal
            if cv < 0.1:
                variance_score = 0.4  # Too uniform
            elif cv > 0.8:
                variance_score = 0.5  # Too chaotic
            else:
                variance_score = 0.6 + cv * 0.5
            variance_score = min(1.0, variance_score)
            
            # Penalize very long average sentences
            if mean_len > 30:
                variance_score *= 0.8
            elif mean_len > 40:
                variance_score *= 0.6
        else:
            variance_score = 0.5
        
        # ---- METRIC 9: Empathy/Engagement Detection ----
        # For queries that seem emotional, check if response acknowledges feelings
        emotional_query_words = {'feeling', 'feel', 'frustrated', 'sad', 'angry', 'stressed',
                                 'worried', 'anxious', 'upset', 'disappointed', 'heartbroken',
                                 'lonely', 'loneliness', 'despair', 'devastated', 'exhausted'}
        query_words_set = set(re.findall(r'[a-zA-Z]+', query.lower()))
        is_emotional_query = len(query_words_set & emotional_query_words) >= 1
        
        empathy_words = {'understand', 'sorry', 'hear', 'feel', 'acknowledge', 'valid',
                         'natural', 'okay', 'normal', 'understandable', 'genuinely',
                         'completely', 'absolutely', 'care', 'support', 'comfort'}
        response_words_set = set(words)
        
        if is_emotional_query:
            empathy_match = len(response_words_set & empathy_words)
            empathy_score = min(1.0, empathy_match * 0.25)
            # Also check for dismissive tone
            dismissive_found = sum(1 for dp in dismissive_phrases if dp in response_lower)
            if dismissive_found > 0:
                empathy_score = max(0, empathy_score - 0.5)
        else:
            empathy_score = 0.5  # Neutral for non-emotional queries
        
        # ---- METRIC 10: Directness ----
        # Penalize responses that start with unnecessary preamble
        first_sentence = sentences[0] if sentences else ""
        preamble_patterns = [
            r'^(well|so|okay|ok|hmm|hm|ah|oh)\b',
            r'^(that\'s a (good|great|interesting) question)',
            r'^(let me (think|see))',
        ]
        preamble_penalty = 0
        for pat in preamble_patterns:
            if re.search(pat, first_sentence.lower()):
                preamble_penalty += 0.1
        
        directness_score = max(0, 1.0 - preamble_penalty)
        
        # ---- COMBINE SCORES ----
        # Weighted combination
        weights = {
            'hapax': 0.08,
            'filler': 0.15,
            'coherence': 0.10,
            'specificity': 0.10,
            'alignment': 0.12,
            'efficiency': 0.12,
            'variance': 0.08,
            'empathy': 0.10,
            'directness': 0.08,
        }
        
        raw_score = (
            weights['hapax'] * hapax_score +
            weights['filler'] * filler_score +
            weights['coherence'] * coherence_metric +
            weights['specificity'] * specificity_score +
            weights['alignment'] * alignment_score +
            weights['efficiency'] * efficiency_score +
            weights['variance'] * variance_score +
            weights['empathy'] * empathy_score +
            weights['directness'] * directness_score
        )
        
        # Add structural bonus
        raw_score += structural_bonus
        
        # Remaining weight normalization
        total_weight = sum(weights.values())
        raw_score = raw_score / total_weight  # Normalize to ~0-1 range
        
        # Add structural bonus back (it's already added above, let's handle properly)
        # Re-do: compute weighted sum properly
        base_score = (
            weights['hapax'] * hapax_score +
            weights['filler'] * filler_score +
            weights['coherence'] * coherence_metric +
            weights['specificity'] * specificity_score +
            weights['alignment'] * alignment_score +
            weights['efficiency'] * efficiency_score +
            weights['variance'] * variance_score +
            weights['empathy'] * empathy_score +
            weights['directness'] * directness_score
        ) / total_weight
        
        final_score = base_score + structural_bonus
        
        # Scale to 1-5 range
        final_score = 1.0 + final_score * 4.0
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 2.5