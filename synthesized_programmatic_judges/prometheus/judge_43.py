def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, filler detection,
    hedging language analysis, sentence structure variance, and signal-to-noise ratio.
    
    This variant focuses on:
    1. Filler/hedge word density (penalize)
    2. Information density via unique content words per sentence
    3. Sentence structure consistency (low variance in length = monotonous)
    4. Redundancy detection via n-gram repetition across sentences
    5. Directness score (ratio of assertive vs passive/hedging constructions)
    6. Transition word usage (indicates logical flow)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        # Tokenize words (lowercase)
        def get_words(text):
            return re.findall(r"[a-z']+", text.lower())
        
        all_words = get_words(response)
        if not all_words:
            return 0.5
        
        total_words = len(all_words)
        
        # ---- Feature 1: Filler and hedge word density ----
        filler_words = {
            'just', 'really', 'very', 'quite', 'pretty', 'basically', 'actually',
            'literally', 'honestly', 'simply', 'stuff', 'things', 'kind', 'kinda',
            'sort', 'sorta', 'like', 'well', 'anyway', 'anyways', 'right',
            'obviously', 'clearly', 'definitely', 'certainly', 'perhaps',
            'probably', 'maybe', 'somewhat', 'somehow', 'whatever', 'wherever',
            'whenever', 'however', 'moreover', 'furthermore', 'nevertheless',
            'hmm', 'huh', 'oh', 'ah', 'um', 'uh', 'nifty', 'cool', 'wow'
        }
        
        hedge_phrases = [
            'i think', 'i guess', 'i suppose', 'it seems', 'it might',
            'it could', 'it may', 'kind of', 'sort of', 'more or less',
            'in a way', 'to some extent', 'if you will', 'so to speak',
            'you know', 'i mean', 'or something', 'or whatever',
            'might not', 'may not', 'could be', 'would be',
            'not really', "don't really", 'not sure', 'not certain'
        ]
        
        filler_count = sum(1 for w in all_words if w in filler_words)
        filler_density = filler_count / max(total_words, 1)
        
        response_lower = response.lower()
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in response_lower)
        hedge_density = hedge_count / max(len(sentences), 1)
        
        # Score: lower filler/hedge = better (0-10 scale component)
        filler_score = max(0, 10 - filler_density * 60 - hedge_density * 8)
        
        # ---- Feature 2: Information density (unique content words per sentence) ----
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
            'and', 'but', 'or', 'if', 'it', 'its', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'up', 'about', 'also', 'while', 'until'
        }
        
        content_words_per_sentence = []
        for sent in sentences:
            words = get_words(sent)
            content = [w for w in words if w not in stop_words and len(w) > 2]
            if words:
                content_words_per_sentence.append(len(set(content)) / max(len(words), 1))
        
        avg_info_density = sum(content_words_per_sentence) / max(len(content_words_per_sentence), 1)
        # Higher info density = more content-rich sentences
        info_score = min(10, avg_info_density * 20)
        
        # ---- Feature 3: Redundancy via cross-sentence content overlap (bigrams) ----
        def get_content_bigrams(text):
            words = get_words(text)
            content = [w for w in words if w not in stop_words and len(w) > 2]
            if len(content) < 2:
                return set()
            return set(zip(content[:-1], content[1:]))
        
        if len(sentences) >= 2:
            sentence_bigrams = [get_content_bigrams(s) for s in sentences]
            overlap_scores = []
            for i in range(len(sentence_bigrams)):
                for j in range(i + 1, len(sentence_bigrams)):
                    if sentence_bigrams[i] and sentence_bigrams[j]:
                        overlap = len(sentence_bigrams[i] & sentence_bigrams[j])
                        union = len(sentence_bigrams[i] | sentence_bigrams[j])
                        if union > 0:
                            overlap_scores.append(overlap / union)
            
            avg_overlap = sum(overlap_scores) / max(len(overlap_scores), 1) if overlap_scores else 0
        else:
            avg_overlap = 0
        
        # Lower overlap = less redundancy = better
        redundancy_score = max(0, 10 - avg_overlap * 40)
        
        # ---- Feature 4: Sentence length variance (moderate variance is good) ----
        sent_lengths = [len(get_words(s)) for s in sentences]
        if len(sent_lengths) >= 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(mean_len, 1)  # coefficient of variation
            
            # Moderate CV (0.2-0.5) is ideal - indicates varied but not chaotic structure
            if cv < 0.1:
                variety_score = 4.0  # too monotonous
            elif cv < 0.3:
                variety_score = 8.0  # good variety
            elif cv < 0.6:
                variety_score = 7.0  # decent variety
            else:
                variety_score = 5.0  # too chaotic
        else:
            variety_score = 5.0
        
        # ---- Feature 5: Directness / assertiveness ----
        # Count passive/weak constructions
        weak_patterns = [
            r'\bmight be\b', r'\bcould be\b', r'\bmay be\b',
            r'\bit is possible\b', r'\bthere is\b', r'\bthere are\b',
            r'\bit seems\b', r'\bit appears\b', r'\bprobably\b',
            r'\bperhaps\b', r'\bmaybe\b', r'\bmight not\b',
            r'\bcould not\b', r'\bmay not\b', r'\bwon\'t be able\b',
            r'\bnot be able\b', r'\bit might\b', r'\bit may\b',
            r'\bit could\b', r'\bit probably\b'
        ]
        
        weak_count = sum(len(re.findall(p, response_lower)) for p in weak_patterns)
        weak_density = weak_count / max(len(sentences), 1)
        
        # Strong/direct patterns
        strong_patterns = [
            r'\bhere(?:\'s| is| are)\b', r'\blet(?:\'s| us)\b',
            r'\bremember\b', r'\bfirst\b', r'\bnext\b', r'\bthen\b',
            r'\bstart\b', r'\bensure\b', r'\bmake sure\b',
            r'\bimportant\b', r'\bkey\b', r'\bessential\b',
            r'\bcritical\b', r'\bfocus\b', r'\bstep\b'
        ]
        
        strong_count = sum(len(re.findall(p, response_lower)) for p in strong_patterns)
        strong_density = strong_count / max(len(sentences), 1)
        
        directness_score = min(10, max(0, 5 + strong_density * 4 - weak_density * 6))
        
        # ---- Feature 6: Structural clarity (numbered lists, colons, clear organization) ----
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[.)]\s', response))
        has_colon_structure = len(re.findall(r':\s', response)) >= 2
        has_paragraph_breaks = response.count('\n\n') >= 1
        
        structure_score = 5.0
        if has_numbered_list:
            structure_score += 2.0
        if has_colon_structure:
            structure_score += 1.0
        if has_paragraph_breaks:
            structure_score += 1.0
        structure_score = min(10, structure_score)
        
        # ---- Feature 7: Empathy/engagement markers (relevant for support queries) ----
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bit\'s understandable\b', r'\bthat\'s understandable\b',
            r'\bcompletely understandable\b', r'\babsolutely\b',
            r'\bi\'m sorry\b', r'\bwe value\b', r'\bwe appreciate\b',
            r'\byour feelings\b', r'\byour experience\b',
            r'\bit\'s okay\b', r'\bit\'s perfectly\b', r'\bnatural to\b'
        ]
        
        empathy_count = sum(1 for p in empathy_patterns if re.search(p, response_lower))
        
        # Check if query seems emotional/support-seeking
        emotional_query_words = ['feeling', 'frustrated', 'stress', 'sad', 'lonely',
                                  'heartbroken', 'devastated', 'struggling', 'difficult',
                                  'regret', 'sorry', 'help', 'comfort', 'support']
        query_lower = query.lower()
        is_emotional_query = sum(1 for w in emotional_query_words if w in query_lower) >= 2
        
        if is_emotional_query:
            empathy_score = min(10, 4 + empathy_count * 1.5)
        else:
            empathy_score = 6.0  # neutral baseline
        
        # ---- Feature 8: Average sentence length penalty ----
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        if avg_sent_len < 5:
            length_score = 4.0  # too terse
        elif avg_sent_len < 10:
            length_score = 7.0
        elif avg_sent_len < 20:
            length_score = 9.0  # sweet spot
        elif avg_sent_len < 30:
            length_score = 7.0
        else:
            length_score = 4.0  # too long
        
        # ---- Feature 9: Negation/inability density (penalize responses saying what can't be done) ----
        inability_patterns = [
            r'\bcan\'t\b', r'\bcannot\b', r'\bunable to\b',
            r'\bnot able\b', r'\bwon\'t\b', r'\bmight not\b',
            r'\bmay not\b', r'\bdon\'t have\b', r'\bdoesn\'t have\b'
        ]
        inability_count = sum(len(re.findall(p, response_lower)) for p in inability_patterns)
        inability_density = inability_count / max(len(sentences), 1)
        inability_score = max(0, 10 - inability_density * 10)
        
        # ---- Feature 10: Query-response relevance via keyword overlap ----
        query_content = set(w for w in get_words(query) if w not in stop_words and len(w) > 2)
        response_content = set(w for w in all_words if w not in stop_words and len(w) > 2)
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        relevance_score = min(10, relevance * 15 + 3)
        
        # ---- Combine all features with weights ----
        weights = {
            'filler': 0.15,
            'info_density': 0.12,
            'redundancy': 0.12,
            'variety': 0.08,
            'directness': 0.13,
            'structure': 0.08,
            'empathy': 0.10,
            'sent_length': 0.07,
            'inability': 0.08,
            'relevance': 0.07
        }
        
        scores = {
            'filler': filler_score,
            'info_density': info_score,
            'redundancy': redundancy_score,
            'variety': variety_score,
            'directness': directness_score,
            'structure': structure_score,
            'empathy': empathy_score,
            'sent_length': length_score,
            'inability': inability_score,
            'relevance': relevance_score
        }
        
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        total_weight = sum(weights.values())
        
        # Normalize to 0-10 scale
        raw_score = weighted_sum / total_weight
        
        # Map to 1-5 scale to match expected output range
        final_score = max(1.0, min(5.0, raw_score / 2.0))
        
        # Apply small bonus for substantial responses (not too short)
        if total_words < 20:
            final_score *= 0.7
        elif total_words < 40:
            final_score *= 0.85
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5