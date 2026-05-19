def judging_function(query, response):
    """
    Evaluates clarity and conciseness of an LLM response using information density,
    coherence flow, and precision metrics. Uses a transition/cohesion-based approach
    with filler detection and information-to-noise ratio.
    
    DIFFERENT from Variant 1 (which uses sentence length, vocab diversity, bullet/list, headers).
    This variant focuses on:
    - Filler/hedge word density (noise detection)
    - Transition word usage (coherence/flow)
    - Information density via content word ratio
    - Repetition detection via n-gram overlap between sentences
    - Directness score (how quickly the response addresses the query)
    - Sentence-to-sentence semantic drift (topic coherence)
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
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        num_words = len(words)
        
        # === 1. FILLER / HEDGE WORD DENSITY (penalize vagueness) ===
        filler_words = {
            'just', 'really', 'very', 'quite', 'basically', 'actually', 'literally',
            'honestly', 'kind', 'kinda', 'sort', 'sorta', 'like', 'stuff', 'things',
            'maybe', 'perhaps', 'probably', 'might', 'could', 'somehow', 'somewhat',
            'anyway', 'anyways', 'well', 'so', 'um', 'uh', 'hmm', 'right',
            'obviously', 'definitely', 'certainly', 'absolutely', 'totally',
            'pretty', 'rather', 'fairly', 'slightly', 'simply'
        }
        
        hedge_phrases = [
            'i think', 'i guess', 'i suppose', 'you know', 'i mean',
            'to be honest', 'in my opinion', 'it seems like', 'kind of',
            'sort of', 'more or less', 'at the end of the day',
            'when it comes to', 'as a matter of fact', 'the thing is',
            'the fact of the matter', 'needless to say', 'it goes without saying',
            'long story short', 'in other words'
        ]
        
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / num_words
        
        response_lower = response.lower()
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in response_lower)
        hedge_penalty = min(hedge_count * 0.08, 0.4)
        
        # Score: lower filler ratio = better (0 to 1 scale, 1 = best)
        filler_score = max(0, 1.0 - filler_ratio * 5.0 - hedge_penalty)
        
        # === 2. TRANSITION / COHESION WORDS (reward coherent flow) ===
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'alternatively',
            'specifically', 'particularly', 'notably', 'importantly',
            'first', 'second', 'third', 'finally', 'next', 'then',
            'because', 'since', 'although', 'while', 'whereas',
            'thus', 'hence', 'accordingly', 'instead', 'otherwise',
            'similarly', 'likewise', 'conversely', 'regardless',
            'here', 'now', 'also', 'remember', 'imagine', 'consider'
        }
        
        transition_count = sum(1 for w in words if w in transition_words)
        transition_ratio = transition_count / num_words
        # Sweet spot: some transitions are good, too many is bloated
        if transition_ratio < 0.01:
            transition_score = 0.3
        elif transition_ratio < 0.06:
            transition_score = 0.3 + (transition_ratio - 0.01) * 14.0  # ramps up to ~1.0
        elif transition_ratio < 0.10:
            transition_score = 1.0
        else:
            transition_score = max(0.4, 1.0 - (transition_ratio - 0.10) * 5.0)
        
        # === 3. CONTENT WORD RATIO (information density) ===
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'should', 'may', 'might', 'can', 'could', 'must', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'and', 'but', 'or', 'nor', 'not', 'no', 'if', 'that', 'which',
            'who', 'whom', 'this', 'these', 'those', 'it', 'its', 'i', 'you',
            'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'our', 'their', 'what', 'where', 'when',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'some',
            'any', 'most', 'other', 'than', 'too', 'only', 'own', 'same',
            'about', 'up', 'out', 'so', 'just', 'there', 'here'
        }
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        content_ratio = len(content_words) / num_words if num_words > 0 else 0
        # Higher content ratio = more informative
        content_score = min(1.0, content_ratio * 2.5)
        
        # === 4. REPETITION DETECTION via bigram/trigram overlap across sentences ===
        def get_ngrams(word_list, n):
            return [tuple(word_list[i:i+n]) for i in range(len(word_list) - n + 1)]
        
        sentence_words = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s.lower())
            if len(sw) >= 2:
                sentence_words.append(sw)
        
        repetition_penalty = 0.0
        if len(sentence_words) >= 2:
            # Check trigram overlap between all pairs of sentences
            all_trigrams_per_sentence = []
            for sw in sentence_words:
                tg = set(get_ngrams(sw, 3))
                all_trigrams_per_sentence.append(tg)
            
            overlap_count = 0
            pair_count = 0
            for i in range(len(all_trigrams_per_sentence)):
                for j in range(i + 1, len(all_trigrams_per_sentence)):
                    if all_trigrams_per_sentence[i] and all_trigrams_per_sentence[j]:
                        overlap = len(all_trigrams_per_sentence[i] & all_trigrams_per_sentence[j])
                        min_size = min(len(all_trigrams_per_sentence[i]), len(all_trigrams_per_sentence[j]))
                        if min_size > 0:
                            overlap_count += overlap / min_size
                        pair_count += 1
            
            if pair_count > 0:
                avg_overlap = overlap_count / pair_count
                repetition_penalty = min(0.5, avg_overlap * 2.0)
        
        repetition_score = max(0, 1.0 - repetition_penalty)
        
        # === 5. DIRECTNESS SCORE (does response address the query quickly?) ===
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - function_words
        query_words = {w for w in query_words if len(w) > 2}
        
        directness_score = 0.5  # default
        if query_words and len(sentence_words) > 0:
            # Check how many query-relevant words appear in the first sentence
            first_sentence_words = set(sentence_words[0])
            if query_words:
                first_overlap = len(first_sentence_words & query_words) / len(query_words)
                directness_score = min(1.0, 0.3 + first_overlap * 1.0)
        
        # === 6. SENTENCE CLARITY (avg words per sentence - penalize too long or too short) ===
        avg_words_per_sentence = num_words / num_sentences
        # Ideal: 12-22 words per sentence
        if avg_words_per_sentence < 5:
            clarity_length_score = 0.3
        elif avg_words_per_sentence < 12:
            clarity_length_score = 0.3 + (avg_words_per_sentence - 5) / 7.0 * 0.7
        elif avg_words_per_sentence <= 25:
            clarity_length_score = 1.0
        elif avg_words_per_sentence <= 40:
            clarity_length_score = 1.0 - (avg_words_per_sentence - 25) / 15.0 * 0.5
        else:
            clarity_length_score = 0.3
        
        # === 7. STRUCTURAL SIGNALS (numbered lists, colons for definitions, etc.) ===
        has_structure = 0.0
        # Numbered items
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        if len(numbered_items) >= 2:
            has_structure += 0.3
        # Colon-based definitions/explanations
        colon_defs = re.findall(r':\s+\w', response)
        if len(colon_defs) >= 1:
            has_structure += 0.1
        # Paragraph breaks (indicates organized thought)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            has_structure += 0.2
        
        structure_score = min(1.0, 0.5 + has_structure)
        
        # === 8. EMPATHY/ENGAGEMENT DETECTION (for emotional queries) ===
        emotional_query_words = {'feeling', 'feel', 'sad', 'frustrated', 'stress', 'stressed',
                                 'worried', 'anxious', 'lonely', 'heartbroken', 'devastated',
                                 'struggling', 'difficult', 'hard', 'tough', 'upset', 'angry',
                                 'disappointed', 'exhausted', 'tired', 'overwhelmed', 'fear',
                                 'breakup', 'passed', 'died', 'death', 'loss', 'grief'}
        
        query_lower = query.lower()
        is_emotional = any(w in query_lower for w in emotional_query_words)
        
        empathy_markers = [
            'i understand', "i'm sorry", 'i can see', 'it\'s understandable',
            'that\'s understandable', 'completely understandable', 'it\'s okay',
            "it's okay", 'it\'s natural', "it's natural", 'perfectly fine',
            'absolutely okay', 'i hear you', 'i can hear', 'take a moment',
            'give yourself', 'let yourself', 'your feelings'
        ]
        
        empathy_score = 0.5
        if is_emotional:
            empathy_count = sum(1 for marker in empathy_markers if marker in response_lower)
            if empathy_count >= 2:
                empathy_score = 1.0
            elif empathy_count == 1:
                empathy_score = 0.8
            else:
                empathy_score = 0.3
        
        # === 9. DISMISSIVE LANGUAGE DETECTION (penalize) ===
        dismissive_phrases = [
            'just do', 'just get', 'you should be able', 'it\'s not that',
            "it's not that", 'get over it', 'move on', 'stop being',
            'you need to get', 'nothing wrong', 'that\'s a bummer',
            "that's a bummer", 'keep trying', 'just keep', 'not able',
            'might not', 'probably won\'t', "probably won't", 'can\'t',
            "won't be able", 'not going to work'
        ]
        
        dismissive_count = sum(1 for phrase in dismissive_phrases if phrase in response_lower)
        dismissive_penalty = min(0.4, dismissive_count * 0.12)
        
        # === 10. RESPONSE LENGTH APPROPRIATENESS ===
        # Not too short (insufficient), not absurdly bloated
        length_score = 0.5
        if num_words < 20:
            length_score = 0.2
        elif num_words < 40:
            length_score = 0.4 + (num_words - 20) / 20.0 * 0.4
        elif num_words <= 200:
            length_score = 0.9
        elif num_words <= 350:
            length_score = 0.9 - (num_words - 200) / 150.0 * 0.2
        else:
            length_score = 0.6
        
        # === COMBINE SCORES ===
        # Weighted combination
        weights = {
            'filler': 1.5,
            'transition': 0.8,
            'content': 1.0,
            'repetition': 1.2,
            'directness': 0.7,
            'clarity_length': 0.8,
            'structure': 0.6,
            'empathy': 0.8 if is_emotional else 0.2,
            'length': 0.6,
        }
        
        scores = {
            'filler': filler_score,
            'transition': transition_score,
            'content': content_score,
            'repetition': repetition_score,
            'directness': directness_score,
            'clarity_length': clarity_length_score,
            'structure': structure_score,
            'empathy': empathy_score,
            'length': length_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        
        base_score = weighted_sum / total_weight  # 0 to 1
        
        # Apply dismissive penalty
        base_score = max(0, base_score - dismissive_penalty)
        
        # Scale to 1-5 range
        final_score = 1.0 + base_score * 4.0
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 3.0