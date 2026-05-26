def judging_function(query, response):
    """
    Evaluates clarity and conciseness of an LLM response.
    Higher scores = better clarity and conciseness.
    Returns a score from 0 to 10.
    
    Strategy: Focus on sentence-level clarity metrics, information density,
    filler/hedge word penalization, structural organization, and 
    readability indicators.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # === FEATURE 1: Filler and hedge word density (penalize) ===
        filler_words = [
            'basically', 'actually', 'literally', 'just', 'really', 'very',
            'quite', 'rather', 'somewhat', 'perhaps', 'maybe', 'possibly',
            'kind of', 'sort of', 'you know', 'i mean', 'like', 'well',
            'anyway', 'anyways', 'honestly', 'frankly', 'obviously',
            'clearly', 'definitely', 'certainly', 'probably', 'apparently',
            'hmm', 'huh', 'uh', 'um', 'oh', 'ah', 'so yeah', 'right',
            'nifty', 'stuff', 'things', 'thingy', 'whatever'
        ]
        
        response_lower = response_clean.lower()
        words = re.findall(r'\b[a-z\']+\b', response_lower)
        word_count = len(words)
        
        if word_count == 0:
            return 0.5
        
        filler_count = 0
        for filler in filler_words:
            if ' ' in filler:
                filler_count += response_lower.count(filler)
            else:
                filler_count += sum(1 for w in words if w == filler)
        
        filler_density = filler_count / max(word_count, 1)
        # Score: lower filler density is better (0 to 1 scale)
        filler_score = max(0, 1.0 - filler_density * 8)
        
        # === FEATURE 2: Sentence structure quality ===
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length (words per sentence)
        sentence_lengths = []
        for sent in sentences:
            sent_words = re.findall(r'\b\w+\b', sent)
            if sent_words:
                sentence_lengths.append(len(sent_words))
        
        avg_sent_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)
        
        # Ideal sentence length: 12-22 words
        if 12 <= avg_sent_len <= 22:
            sent_len_score = 1.0
        elif avg_sent_len < 12:
            sent_len_score = max(0.3, avg_sent_len / 12)
        else:
            sent_len_score = max(0.3, 1.0 - (avg_sent_len - 22) / 30)
        
        # Sentence length variance (some variety is good, too much is bad)
        if len(sentence_lengths) > 1:
            mean_sl = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(mean_sl, 1)
            # Moderate variation is ideal (cv around 0.3-0.5)
            if 0.2 <= cv <= 0.6:
                variety_score = 1.0
            elif cv < 0.2:
                variety_score = 0.6 + cv * 2
            else:
                variety_score = max(0.4, 1.0 - (cv - 0.6) * 0.8)
        else:
            variety_score = 0.5
        
        # === FEATURE 3: Redundancy / repetition detection ===
        # Check for repeated n-grams (trigrams)
        if word_count > 10:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition_rate = repeated_trigrams / max(len(trigrams), 1)
            redundancy_score = max(0, 1.0 - trigram_repetition_rate * 10)
        else:
            redundancy_score = 0.7
        
        # Check for repeated bigrams more aggressively
        if word_count > 6:
            bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            # Filter out common functional bigrams
            functional_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'and', 'or', 'it', 'you', 'your', 'i', 'we', 'they'}
            content_repeated_bigrams = sum(
                1 for bg, c in bigram_counts.items() 
                if c > 2 and not all(w in functional_words for w in bg)
            )
            bigram_penalty = min(content_repeated_bigrams * 0.05, 0.3)
            redundancy_score = max(0, redundancy_score - bigram_penalty)
        
        # === FEATURE 4: Vocabulary richness (type-token ratio) ===
        unique_words = set(words)
        # Use root TTR to handle length bias
        if word_count > 0:
            ttr = len(unique_words) / math.sqrt(word_count)
            # Normalize: typical range is 3-8 for root TTR
            vocab_score = min(1.0, max(0, (ttr - 2) / 6))
        else:
            vocab_score = 0.3
        
        # === FEATURE 5: Structure indicators (lists, paragraphs, organization) ===
        has_numbered_list = bool(re.search(r'\d+[.)]\s', response_clean))
        has_bullet_list = bool(re.search(r'[-•*]\s', response_clean))
        has_paragraphs = response_clean.count('\n\n') >= 1
        has_colon_headers = bool(re.search(r'\w+:\s', response_clean))
        
        structure_score = 0.5  # baseline
        if has_numbered_list:
            structure_score += 0.2
        if has_bullet_list:
            structure_score += 0.15
        if has_paragraphs:
            structure_score += 0.1
        if has_colon_headers:
            structure_score += 0.05
        structure_score = min(1.0, structure_score)
        
        # === FEATURE 6: Vagueness penalty ===
        vague_phrases = [
            'and stuff', 'or something', 'things like that', 'you know what i mean',
            'it depends', 'it\'s hard to say', 'there are many', 'various things',
            'a lot of things', 'many things', 'some things', 'different things',
            'and so on', 'etc etc', 'blah blah', 'yada yada',
            'it might not', 'it probably won\'t', 'might not be able',
            'may not be able', 'it may not'
        ]
        
        vague_count = sum(1 for phrase in vague_phrases if phrase in response_lower)
        vagueness_score = max(0, 1.0 - vague_count * 0.15)
        
        # === FEATURE 7: Dismissive / unhelpful language detection ===
        dismissive_phrases = [
            'just do', 'just try', 'just keep', 'just remember',
            'you should be able', 'it\'s not that hard',
            'get yourself together', 'move on', 'get over it',
            'that\'s a bummer', 'no big deal', 'not a big deal',
            'you\'re just not', 'maybe you\'re just'
        ]
        
        dismissive_count = sum(1 for phrase in dismissive_phrases if phrase in response_lower)
        dismissive_score = max(0, 1.0 - dismissive_count * 0.2)
        
        # === FEATURE 8: Directness and engagement ===
        # Starting with acknowledgment or direct address is good
        starts_well = 0.5
        first_50 = response_lower[:80]
        
        good_starts = [
            'i can', 'i\'m', 'i understand', 'it\'s', 'that\'s', 'here',
            'to ', 'the ', 'imagine', 'hey', 'let\'s', 'sure',
            'absolutely', 'great', 'thank', 'welcome'
        ]
        weak_starts = [
            'well,', 'so,', 'hmm', 'uh', 'ok so', 'okay so'
        ]
        
        for gs in good_starts:
            if first_50.startswith(gs):
                starts_well = 0.8
                break
        for ws in weak_starts:
            if first_50.startswith(ws):
                starts_well = 0.3
                break
        
        # === FEATURE 9: Proportion of content words vs function words ===
        function_words_set = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'should', 'may', 'might', 'can', 'could', 'must', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'and',
            'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither',
            'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom'
        }
        
        content_words = [w for w in words if w not in function_words_set and len(w) > 2]
        content_ratio = len(content_words) / max(word_count, 1)
        # Ideal content ratio: 0.4-0.6
        if 0.35 <= content_ratio <= 0.65:
            content_score = 1.0
        elif content_ratio < 0.35:
            content_score = max(0.3, content_ratio / 0.35)
        else:
            content_score = max(0.5, 1.0 - (content_ratio - 0.65) * 2)
        
        # === FEATURE 10: Query relevance (simple keyword overlap) ===
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        # Remove very common words
        query_content = query_words - function_words_set
        response_word_set = set(words)
        
        if query_content:
            overlap = len(query_content & response_word_set) / len(query_content)
            relevance_score = min(1.0, overlap * 1.5)
        else:
            relevance_score = 0.5
        
        # === FEATURE 11: Empathy/engagement markers (for emotional queries) ===
        emotional_query_words = {'feeling', 'feel', 'sad', 'frustrated', 'stress', 'stressed',
                                 'heartbroken', 'lonely', 'loneliness', 'despair', 'exhausted',
                                 'struggling', 'difficult', 'worried', 'anxious', 'upset',
                                 'devastated', 'regret', 'angry', 'disappointed'}
        
        query_is_emotional = bool(emotional_query_words & query_content)
        
        empathy_score = 0.5  # neutral baseline
        if query_is_emotional:
            empathy_markers = [
                'i understand', 'i can see', 'i can hear', 'i\'m sorry',
                'it\'s okay', 'it\'s completely', 'it\'s perfectly',
                'that\'s understandable', 'completely understandable',
                'totally understandable', 'it\'s natural', 'it\'s normal',
                'give yourself', 'take a moment', 'let yourself'
            ]
            empathy_count = sum(1 for em in empathy_markers if em in response_lower)
            empathy_score = min(1.0, 0.4 + empathy_count * 0.2)
        
        # === FEATURE 12: Negative pattern detection ===
        negative_penalty = 0.0
        
        # Overly casual/dismissive for serious topics
        if query_is_emotional:
            casual_dismissive = ['bummer', 'no big deal', 'get over', 'move on',
                                'just a', 'not that bad', 'cheer up']
            for cd in casual_dismissive:
                if cd in response_lower:
                    negative_penalty += 0.1
        
        # Contradictory or unhelpful (saying "can't" or "won't" without alternative)
        inability_phrases = ['can\'t provide', 'won\'t be able', 'not able to',
                            'might not be able', 'probably won\'t']
        for ip in inability_phrases:
            if ip in response_lower:
                negative_penalty += 0.05
        
        negative_penalty = min(negative_penalty, 0.4)
        
        # === COMBINE SCORES ===
        # Weights tuned for clarity and conciseness focus
        weights = {
            'filler': 1.5,
            'sent_len': 1.0,
            'variety': 0.5,
            'redundancy': 1.2,
            'vocab': 0.8,
            'structure': 0.8,
            'vagueness': 1.3,
            'dismissive': 1.0,
            'starts_well': 0.4,
            'content': 0.6,
            'relevance': 0.9,
            'empathy': 0.7,
        }
        
        scores = {
            'filler': filler_score,
            'sent_len': sent_len_score,
            'variety': variety_score,
            'redundancy': redundancy_score,
            'vocab': vocab_score,
            'structure': structure_score,
            'vagueness': vagueness_score,
            'dismissive': dismissive_score,
            'starts_well': starts_well,
            'content': content_score,
            'relevance': relevance_score,
            'empathy': empathy_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
        base_score = weighted_sum / total_weight  # 0 to 1
        
        # Apply negative penalty
        base_score = max(0, base_score - negative_penalty)
        
        # Scale to 0-10
        final_score = base_score * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except:
            return 3.0