def judging_function(query, response):
    """
    Evaluates clarity and conciseness of an LLM response.
    
    This variant focuses on:
    - Information density (ratio of meaningful content words to total words)
    - Sentence structure variety and complexity balance
    - Filler/weasel word detection with weighted penalties
    - Signal-to-noise ratio (concrete vs vague language)
    - Readability via syllable-based metrics
    - Structural coherence (logical flow indicators)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Tokenize
        words = re.findall(r"[a-zA-Z']+", response_clean.lower())
        if len(words) < 5:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        if not sentences:
            return 1.0
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Syllable-based readability (Flesch-like approach)
        # ============================================================
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 2:
                return 1
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / max(num_words, 1)
        avg_words_per_sentence = num_words / max(num_sentences, 1)
        
        # Optimal readability: avg 1.4-1.8 syllables/word, 12-22 words/sentence
        syllable_score = 1.0 - min(abs(avg_syllables_per_word - 1.6) / 1.0, 1.0)
        
        # Sentence length score - penalize too short or too long
        if avg_words_per_sentence < 5:
            sent_len_score = 0.3
        elif avg_words_per_sentence <= 25:
            sent_len_score = 1.0 - abs(avg_words_per_sentence - 16) / 20.0
        else:
            sent_len_score = max(0.1, 1.0 - (avg_words_per_sentence - 25) / 30.0)
        
        readability_score = 0.5 * syllable_score + 0.5 * sent_len_score
        
        # ============================================================
        # FEATURE 2: Information density - concrete vs filler words
        # ============================================================
        # Filler/empty words that add no information
        filler_words = {
            'just', 'really', 'very', 'quite', 'pretty', 'basically',
            'actually', 'literally', 'honestly', 'simply', 'totally',
            'absolutely', 'definitely', 'certainly', 'obviously',
            'clearly', 'surely', 'stuff', 'things', 'thing', 'like',
            'kind', 'sort', 'kinda', 'sorta', 'gonna', 'wanna',
            'well', 'anyway', 'anyways', 'right', 'okay', 'ok',
            'hmm', 'huh', 'umm', 'uh', 'um', 'nifty', 'cool',
            'neat', 'wild', 'crazy'
        }
        
        # Weasel/hedge words that reduce clarity
        weasel_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could',
            'probably', 'somewhat', 'somehow', 'something',
            'sometimes', 'somewhere', 'seems', 'appear',
            'appears', 'apparently', 'arguably', 'presumably',
            'roughly', 'approximately'
        }
        
        # Dismissive/unhelpful phrases
        dismissive_patterns = [
            r"it'?s? just", r"you should just", r"just do",
            r"get over it", r"move on", r"not a big deal",
            r"no big deal", r"that'?s? life", r"part of life",
            r"get yourself together", r"you need to get"
        ]
        
        filler_count = sum(1 for w in words if w in filler_words)
        weasel_count = sum(1 for w in words if w in weasel_words)
        
        dismissive_count = 0
        response_lower = response_clean.lower()
        for pat in dismissive_patterns:
            dismissive_count += len(re.findall(pat, response_lower))
        
        filler_ratio = filler_count / max(num_words, 1)
        weasel_ratio = weasel_count / max(num_words, 1)
        
        # Information density score
        density_score = max(0.0, 1.0 - (filler_ratio * 4.0) - (weasel_ratio * 3.0) - (dismissive_count * 0.15))
        
        # ============================================================
        # FEATURE 3: Sentence structure variety
        # ============================================================
        sentence_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
        sentence_lengths = [sl for sl in sentence_lengths if sl > 0]
        
        if len(sentence_lengths) > 2:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((sl - mean_len) ** 2 for sl in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation - some variety is good
            cv = std_dev / max(mean_len, 1)
            # Optimal CV is around 0.3-0.5
            if cv < 0.1:
                variety_score = 0.5  # too uniform
            elif cv < 0.6:
                variety_score = 1.0
            else:
                variety_score = max(0.3, 1.0 - (cv - 0.6) * 1.5)
        else:
            variety_score = 0.6
        
        # ============================================================
        # FEATURE 4: Concrete action/advice indicators
        # ============================================================
        # Look for actionable language
        action_patterns = [
            r'\b(first|second|third|next|then|finally|start|begin)\b',
            r'\b(try|consider|remember|ensure|make sure|focus)\b',
            r'\b(step \d|tip \d|\d[\.\)]\s)',
            r'\b(for example|for instance|such as|specifically)\b',
            r'\b(here are|here\'s|following|below)\b',
        ]
        
        action_count = 0
        for pat in action_patterns:
            action_count += len(re.findall(pat, response_lower))
        
        # Normalize - diminishing returns
        action_score = min(1.0, action_count / 5.0)
        
        # ============================================================
        # FEATURE 5: Empathy and engagement quality
        # ============================================================
        # Check if query seems emotional/personal
        emotional_query_words = {'feeling', 'feel', 'sad', 'frustrated', 'stressed',
                                  'lonely', 'heartbroken', 'devastated', 'struggling',
                                  'difficult', 'hard', 'worried', 'anxious', 'upset',
                                  'down', 'breakup', 'passed away', 'lost'}
        query_lower = query.lower()
        query_words = set(re.findall(r"[a-zA-Z']+", query_lower))
        is_emotional = len(query_words & emotional_query_words) >= 1
        
        empathy_phrases = [
            r"i understand", r"i can see", r"i can hear",
            r"that'?s? (completely |totally |absolutely )?(understandable|okay|normal|natural|fine)",
            r"it'?s? (perfectly |completely |totally )?(okay|fine|normal|natural|understandable)",
            r"i'?m? (sorry|genuinely sorry|truly sorry)",
            r"your feelings? (are|is) valid",
            r"it'?s? (okay|fine|natural|normal) to (feel|be|grieve)",
        ]
        
        empathy_count = 0
        for pat in empathy_phrases:
            empathy_count += len(re.findall(pat, response_lower))
        
        if is_emotional:
            empathy_score = min(1.0, empathy_count / 2.0)
        else:
            empathy_score = 0.5  # neutral when not needed
        
        # ============================================================
        # FEATURE 6: Repetition detection (semantic redundancy proxy)
        # ============================================================
        # Check for repeated n-grams (trigrams)
        if len(words) >= 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            repetition_penalty = min(1.0, repetition_ratio * 10.0)
        else:
            repetition_penalty = 0.0
        
        repetition_score = 1.0 - repetition_penalty
        
        # ============================================================
        # FEATURE 7: Query relevance (content word overlap)
        # ============================================================
        # Use content words (longer words) to check relevance
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
            'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
            'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'that',
            'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'how', 'when', 'where', 'why', 'if', 'then', 'there', 'here',
            'about', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
            'further', 'once', 'also', 'much', 'many', 'well', 'back',
            'even', 'still', 'now', 'get', 'got', 'make', 'made',
        }
        
        query_content = set(w for w in re.findall(r"[a-zA-Z']+", query_lower) 
                          if w not in stop_words and len(w) > 2)
        response_content = set(w for w in words if w not in stop_words and len(w) > 2)
        
        if query_content:
            overlap = len(query_content & response_content) / max(len(query_content), 1)
            relevance_score = min(1.0, overlap * 2.5)  # scaled
        else:
            relevance_score = 0.5
        
        # ============================================================
        # FEATURE 8: Structural organization signals
        # ============================================================
        # Numbered lists, colons for explanations, paragraph breaks
        has_numbered_list = bool(re.search(r'(\d+[\.\)]\s|\b(first|second|third|fourth|fifth)\b)', response_lower))
        has_colon_explanation = response_clean.count(':') >= 1
        has_paragraphs = response_clean.count('\n\n') >= 1
        has_dash_list = bool(re.search(r'^\s*[-•]', response_clean, re.MULTILINE))
        
        structure_signals = sum([has_numbered_list, has_colon_explanation, 
                                has_paragraphs, has_dash_list])
        structure_score = min(1.0, structure_signals * 0.3)
        
        # ============================================================
        # FEATURE 9: Tone appropriateness
        # ============================================================
        # Detect if response is dismissive or condescending
        condescending_patterns = [
            r"you should be able to",
            r"it'?s? not that (hard|difficult|bad)",
            r"you('re| are) (just )?not (using|doing)",
            r"read the manual",
            r"you('re| are) (probably|just)",
            r"that'?s? a bummer",
            r"get some rest and you'?ll feel better",
        ]
        
        condescending_count = 0
        for pat in condescending_patterns:
            condescending_count += len(re.findall(pat, response_lower))
        
        tone_score = max(0.0, 1.0 - condescending_count * 0.25)
        
        # ============================================================
        # FEATURE 10: Precision of language (unique content ratio)
        # ============================================================
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        if content_words:
            unique_content = set(content_words)
            precision_ratio = len(unique_content) / max(len(content_words), 1)
            # Higher unique ratio = more precise, less repetitive
            precision_score = min(1.0, precision_ratio * 1.2)
        else:
            precision_score = 0.3
        
        # ============================================================
        # FEATURE 11: Response completeness proxy
        # ============================================================
        # Longer responses that maintain quality are better (up to a point)
        if num_words < 20:
            completeness_score = 0.3
        elif num_words < 50:
            completeness_score = 0.6
        elif num_words <= 200:
            completeness_score = 1.0
        else:
            completeness_score = max(0.6, 1.0 - (num_words - 200) / 500.0)
        
        # ============================================================
        # COMBINE SCORES with weights
        # ============================================================
        weights = {
            'readability': 1.2,
            'density': 1.8,
            'variety': 0.6,
            'action': 1.0,
            'empathy': 1.0 if is_emotional else 0.3,
            'repetition': 1.0,
            'relevance': 1.5,
            'structure': 0.7,
            'tone': 1.5,
            'precision': 0.8,
            'completeness': 0.8,
        }
        
        scores = {
            'readability': readability_score,
            'density': density_score,
            'variety': variety_score,
            'action': action_score,
            'empathy': empathy_score,
            'repetition': repetition_score,
            'relevance': relevance_score,
            'structure': structure_score,
            'tone': tone_score,
            'precision': precision_score,
            'completeness': completeness_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
        raw_score = weighted_sum / total_weight  # 0 to 1
        
        # Map to 1-5 scale
        final_score = 1.0 + raw_score * 4.0
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        return 2.5