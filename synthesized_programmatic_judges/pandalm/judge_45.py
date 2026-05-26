def judging_function(query, response):
    """
    Evaluates clarity and conciseness using a dependency/structure-based approach:
    - Compression ratio (response length relative to query)
    - Functional word ratio (proportion of "glue" words indicating bloat)
    - Repetition detection via sliding window character n-grams
    - Sentence structure variety (measuring syntactic diversity via sentence openings)
    - Information density (unique content words per total words)
    - Transition/connector quality
    - Penalize degenerate patterns (extreme repetition, empty, etc.)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Tokenize
        def tokenize(text):
            return re.findall(r"[a-zA-Z']+", text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 2]
        
        resp_words = tokenize(response)
        query_words = tokenize(query)
        resp_sentences = get_sentences(response)
        
        if len(resp_words) == 0:
            return 1.0
        
        num_words = len(resp_words)
        num_sentences = max(len(resp_sentences), 1)
        
        # === Feature 1: Character-level repetition via 6-gram sliding window ===
        def char_ngram_repetition(text, n=6):
            text_lower = text.lower()
            if len(text_lower) < n:
                return 0.0
            ngrams = [text_lower[i:i+n] for i in range(len(text_lower) - n + 1)]
            counter = Counter(ngrams)
            total = len(ngrams)
            if total == 0:
                return 0.0
            # Fraction of ngrams that appear more than expected
            repeated = sum(c - 1 for c in counter.values() if c > 1)
            return repeated / total
        
        char_rep = char_ngram_repetition(response_stripped)
        # Score: 0 repetition = 1.0, high repetition = 0.0
        repetition_score = max(0.0, 1.0 - char_rep * 3.0)
        
        # === Feature 2: Word-level repetition (consecutive duplicate phrases) ===
        def consecutive_repeat_ratio(words, window=3):
            if len(words) < window * 2:
                return 0.0
            repeats = 0
            total = 0
            for i in range(len(words) - window):
                phrase = tuple(words[i:i+window])
                for j in range(i+1, min(i+20, len(words) - window + 1)):
                    if tuple(words[j:j+window]) == phrase:
                        repeats += 1
                        break
                total += 1
            return repeats / max(total, 1)
        
        consec_rep = consecutive_repeat_ratio(resp_words)
        word_rep_score = max(0.0, 1.0 - consec_rep * 4.0)
        
        # === Feature 3: Functional/filler word ratio ===
        filler_words = {
            'very', 'really', 'quite', 'rather', 'somewhat', 'basically',
            'actually', 'literally', 'just', 'simply', 'merely', 'essentially',
            'practically', 'virtually', 'definitely', 'certainly', 'absolutely',
            'obviously', 'clearly', 'naturally', 'of', 'course', 'needless',
            'to', 'say', 'in', 'fact', 'as', 'matter'
        }
        
        hedging_words = {
            'might', 'maybe', 'perhaps', 'possibly', 'could', 'seem',
            'seems', 'appear', 'appears', 'likely', 'unlikely', 'tend',
            'tends', 'somewhat', 'arguably', 'potentially'
        }
        
        bloat_phrases = [
            'it is important to note that', 'it should be noted that',
            'it is worth mentioning that', 'in order to', 'due to the fact that',
            'for the purpose of', 'in the event that', 'at the end of the day',
            'when it comes to', 'in terms of', 'on the other hand',
            'as a matter of fact', 'the fact that', 'it goes without saying',
            'needless to say', 'at this point in time', 'each and every',
            'first and foremost', 'in my opinion i think'
        ]
        
        resp_lower = response.lower()
        bloat_count = sum(resp_lower.count(phrase) for phrase in bloat_phrases)
        
        filler_count = sum(1 for w in resp_words if w in filler_words)
        hedge_count = sum(1 for w in resp_words if w in hedging_words)
        
        filler_ratio = (filler_count + hedge_count) / num_words
        # Ideal filler ratio is low but not zero (some function words are needed)
        filler_score = max(0.0, 1.0 - max(0, filler_ratio - 0.05) * 5.0)
        bloat_score = max(0.0, 1.0 - bloat_count * 0.3)
        
        # === Feature 4: Sentence opening diversity ===
        def sentence_opening_diversity(sentences):
            if len(sentences) <= 1:
                return 0.8  # neutral for single sentence
            openings = []
            for s in sentences:
                words = tokenize(s)
                if len(words) >= 2:
                    openings.append(tuple(words[:2]))
                elif len(words) == 1:
                    openings.append((words[0],))
            if not openings:
                return 0.5
            unique = len(set(openings))
            total = len(openings)
            return unique / total
        
        opening_diversity = sentence_opening_diversity(resp_sentences)
        
        # === Feature 5: Information density ===
        # Unique content words (excluding very common words) per total words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'if', 'then', 'also', 'about', 'up'
        }
        
        content_words = [w for w in resp_words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        if len(content_words) > 0:
            content_uniqueness = len(unique_content) / len(content_words)
        else:
            content_uniqueness = 0.0
        
        info_density = len(content_words) / num_words if num_words > 0 else 0.0
        
        # Combined information score
        info_score = 0.5 * content_uniqueness + 0.5 * min(info_density / 0.5, 1.0)
        
        # === Feature 6: Average sentence length (prefer moderate) ===
        avg_sent_len = num_words / num_sentences
        # Ideal range: 10-25 words per sentence
        if avg_sent_len < 5:
            sent_len_score = 0.4
        elif avg_sent_len < 10:
            sent_len_score = 0.6 + 0.4 * (avg_sent_len - 5) / 5
        elif avg_sent_len <= 25:
            sent_len_score = 1.0
        elif avg_sent_len <= 40:
            sent_len_score = 1.0 - 0.4 * (avg_sent_len - 25) / 15
        else:
            sent_len_score = 0.4
        
        # === Feature 7: Response length appropriateness ===
        # Not too short (loses info), not too long (loses conciseness)
        query_len = max(len(query_words), 1)
        
        if num_words < 3:
            length_score = 0.15
        elif num_words < 8:
            length_score = 0.35
        elif num_words < 15:
            length_score = 0.6
        elif num_words <= 150:
            length_score = 1.0
        elif num_words <= 300:
            length_score = 1.0 - 0.3 * (num_words - 150) / 150
        else:
            length_score = 0.5
        
        # === Feature 8: Transition word quality ===
        good_transitions = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'meanwhile', 'specifically', 'alternatively',
            'conversely', 'similarly', 'likewise', 'instead', 'nonetheless',
            'thus', 'hence', 'accordingly', 'whereas', 'while'
        }
        
        transition_count = sum(1 for w in resp_words if w in good_transitions)
        # Some transitions are good for clarity, but too many is bloat
        if num_sentences <= 1:
            transition_score = 0.7
        else:
            trans_per_sent = transition_count / num_sentences
            if trans_per_sent < 0.1:
                transition_score = 0.6  # Could use more connectors
            elif trans_per_sent <= 0.5:
                transition_score = 1.0
            else:
                transition_score = max(0.3, 1.0 - (trans_per_sent - 0.5) * 1.5)
        
        # === Feature 9: Degenerate pattern detection ===
        degenerate_penalty = 0.0
        
        # Check for extreme single-word repetition
        word_counts = Counter(resp_words)
        if resp_words:
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            if most_common_word not in stop_words:
                dominance = most_common_count / num_words
                if dominance > 0.3:
                    degenerate_penalty += (dominance - 0.3) * 3.0
        
        # Check if response is just echoing the query
        if query_words and resp_words:
            query_set = set(query_words)
            resp_set = set(resp_words)
            content_query = query_set - stop_words
            content_resp = resp_set - stop_words
            if content_query and content_resp:
                overlap = len(content_resp & content_query) / max(len(content_resp), 1)
                # Very high overlap with query + short response = just parroting
                if overlap > 0.8 and num_words < len(query_words) * 1.3:
                    degenerate_penalty += 0.3
        
        # Check for truncated response
        if response_stripped and response_stripped[-1] not in '.!?"\')]}':
            # Might be truncated
            if len(response_stripped) > 100:
                degenerate_penalty += 0.15
        
        # === Feature 10: Clause-level complexity (commas per sentence) ===
        comma_count = response.count(',')
        commas_per_sent = comma_count / num_sentences
        if commas_per_sent <= 3:
            clause_score = 1.0
        elif commas_per_sent <= 6:
            clause_score = 0.8
        else:
            clause_score = max(0.3, 1.0 - (commas_per_sent - 3) * 0.1)
        
        # === Combine all features ===
        score = (
            repetition_score * 2.0 +      # Heavy weight on no-repetition
            word_rep_score * 1.8 +          # Word-level repetition
            filler_score * 1.0 +            # Filler words
            bloat_score * 0.8 +             # Bloat phrases
            opening_diversity * 1.2 +       # Sentence variety
            info_score * 1.5 +              # Information density
            sent_len_score * 1.0 +          # Sentence length
            length_score * 1.5 +            # Response length appropriateness
            transition_score * 0.5 +        # Transition quality
            clause_score * 0.7             # Clause complexity
        )
        
        # Apply degenerate penalty
        score = score * max(0.1, 1.0 - degenerate_penalty)
        
        # Normalize to 0-100 range
        max_possible = (2.0 + 1.8 + 1.0 + 0.8 + 1.2 + 1.5 + 1.0 + 1.5 + 0.5 + 0.7)  # = 12.0
        normalized = (score / max_possible) * 100.0
        
        return round(max(0.0, min(100.0, normalized)), 2)
        
    except Exception:
        return 25.0