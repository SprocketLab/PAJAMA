def judging_function(query, response):
    """
    Evaluate clarity and conciseness of an LLM response.
    Higher scores = better quality.
    Uses a feature-based approach focusing on:
    1. Appropriate length (not too short, not too bloated)
    2. Repetition detection (word-level and phrase-level)
    3. Sentence structure quality
    4. Information density
    5. Vocabulary diversity
    """
    try:
        import re
        import math
        import collections
        import string
        
        # Edge cases
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        response_text = response.strip()
        query_text = query.strip()
        
        # Tokenize
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        response_words = tokenize(response_text)
        query_words = tokenize(query_text)
        sentences = get_sentences(response_text)
        
        num_words = len(response_words)
        num_sentences = max(len(sentences), 1)
        
        if num_words == 0:
            return 1.0
        
        # ============ FEATURE 1: Length appropriateness (0-15) ============
        query_len = len(query_words)
        
        # Ideal response length is roughly 3-8x the query length, capped
        ideal_min = max(10, query_len * 2)
        ideal_max = max(80, query_len * 10)
        
        length_score = 15.0
        if num_words < 3:
            length_score = 2.0
        elif num_words < ideal_min:
            # Too short - penalize but not as harshly
            ratio = num_words / ideal_min
            length_score = 5.0 + 10.0 * ratio
        elif num_words > ideal_max * 2:
            # Way too long
            length_score = 5.0
        elif num_words > ideal_max:
            overshoot = (num_words - ideal_max) / ideal_max
            length_score = max(5.0, 15.0 - 10.0 * overshoot)
        
        # ============ FEATURE 2: Word-level repetition (0-20) ============
        word_counts = collections.Counter(response_words)
        
        # Filter out common stop words for repetition analysis
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
            'than', 'too', 'very', 'just', 'because', 'if', 'when', 'where',
            'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my',
            'also', 'while'
        }
        
        content_words = [w for w in response_words if w not in stop_words]
        content_counts = collections.Counter(content_words)
        num_content = max(len(content_words), 1)
        
        # Calculate repetition ratio for content words
        if content_counts:
            max_content_freq = max(content_counts.values())
            # What fraction of content words are the most repeated word?
            max_repeat_ratio = max_content_freq / num_content
            
            # Count words that appear more than expected
            over_repeated = sum(1 for w, c in content_counts.items() if c > max(2, num_content * 0.08))
            over_repeat_ratio = over_repeated / max(len(content_counts), 1)
        else:
            max_repeat_ratio = 0
            over_repeat_ratio = 0
        
        # Vocabulary diversity (type-token ratio for content words)
        unique_content = len(content_counts)
        ttr = unique_content / num_content if num_content > 0 else 0
        
        word_rep_score = 20.0
        # Penalize high max repetition
        if max_repeat_ratio > 0.3:
            word_rep_score -= 12.0
        elif max_repeat_ratio > 0.15:
            word_rep_score -= 6.0 * ((max_repeat_ratio - 0.15) / 0.15)
        
        # Penalize low diversity
        if ttr < 0.3 and num_content > 5:
            word_rep_score -= 8.0
        elif ttr < 0.5 and num_content > 5:
            word_rep_score -= 4.0 * ((0.5 - ttr) / 0.2)
        
        word_rep_score = max(0.0, word_rep_score)
        
        # ============ FEATURE 3: Phrase-level repetition (0-20) ============
        def get_ngrams(words, n):
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        phrase_rep_score = 20.0
        
        for n in [3, 4, 5]:
            ngrams = get_ngrams(response_words, n)
            if len(ngrams) < 2:
                continue
            ngram_counts = collections.Counter(ngrams)
            total_ngrams = len(ngrams)
            repeated = sum(c - 1 for c in ngram_counts.values() if c > 1)
            repeat_fraction = repeated / total_ngrams
            
            if repeat_fraction > 0.3:
                phrase_rep_score -= 8.0
            elif repeat_fraction > 0.1:
                phrase_rep_score -= 4.0 * ((repeat_fraction - 0.1) / 0.2)
        
        # Check for repeated sentences
        if num_sentences > 1:
            sent_texts = [s.lower().strip() for s in sentences]
            sent_counter = collections.Counter(sent_texts)
            dup_sents = sum(c - 1 for c in sent_counter.values() if c > 1)
            if dup_sents > 0:
                phrase_rep_score -= min(10.0, dup_sents * 5.0)
        
        # Check for near-duplicate sentences (high word overlap)
        if num_sentences > 1:
            sent_word_sets = [set(tokenize(s)) - stop_words for s in sentences]
            near_dup_count = 0
            for i in range(len(sent_word_sets)):
                for j in range(i + 1, len(sent_word_sets)):
                    if sent_word_sets[i] and sent_word_sets[j]:
                        intersection = sent_word_sets[i] & sent_word_sets[j]
                        union = sent_word_sets[i] | sent_word_sets[j]
                        if union:
                            jaccard = len(intersection) / len(union)
                            if jaccard > 0.7:
                                near_dup_count += 1
            if near_dup_count > 0:
                phrase_rep_score -= min(8.0, near_dup_count * 3.0)
        
        phrase_rep_score = max(0.0, phrase_rep_score)
        
        # ============ FEATURE 4: Sentence clarity (0-15) ============
        avg_sentence_len = num_words / num_sentences
        
        sentence_score = 15.0
        # Ideal average sentence length: 10-25 words
        if avg_sentence_len > 50:
            sentence_score -= 8.0
        elif avg_sentence_len > 30:
            sentence_score -= 4.0 * ((avg_sentence_len - 30) / 20)
        
        if avg_sentence_len < 4 and num_words > 5:
            sentence_score -= 5.0
        
        # Check for very long run-on sentences
        for sent in sentences:
            sent_words = tokenize(sent)
            if len(sent_words) > 60:
                sentence_score -= 3.0
                break
        
        sentence_score = max(0.0, sentence_score)
        
        # ============ FEATURE 5: Substantiveness & relevance (0-15) ============
        # Check if response actually addresses the query
        query_content = set(query_words) - stop_words
        response_content = set(content_words)
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
        else:
            overlap = 0.5
        
        substance_score = 15.0
        
        # Penalize if response is just echoing the query with no added info
        if num_words < 5:
            substance_score = 3.0
        elif response_content and query_content:
            # Check if response adds information beyond query
            new_content = response_content - query_content
            added_ratio = len(new_content) / max(len(response_content), 1)
            if added_ratio < 0.1 and num_words < 15:
                substance_score -= 8.0
        
        # Penalize empty/placeholder responses
        placeholder_patterns = [r'^<noinput>$', r'^\[.*\]$', r'^n/a$', r'^none$']
        for pat in placeholder_patterns:
            if re.match(pat, response_text.strip(), re.IGNORECASE):
                return 1.0
        
        substance_score = max(0.0, substance_score)
        
        # ============ FEATURE 6: Filler/hedge word density (0-15) ============
        filler_words = {
            'basically', 'essentially', 'actually', 'literally', 'really',
            'honestly', 'frankly', 'obviously', 'clearly', 'simply',
            'definitely', 'certainly', 'absolutely', 'totally', 'completely',
            'extremely', 'incredibly', 'remarkably', 'significantly',
            'furthermore', 'moreover', 'additionally', 'nevertheless',
            'nonetheless', 'consequently', 'subsequently', 'accordingly'
        }
        
        filler_count = sum(1 for w in response_words if w in filler_words)
        filler_ratio = filler_count / num_words
        
        filler_score = 15.0
        if filler_ratio > 0.1:
            filler_score -= 10.0
        elif filler_ratio > 0.05:
            filler_score -= 5.0 * ((filler_ratio - 0.05) / 0.05)
        
        filler_score = max(0.0, filler_score)
        
        # ============ BONUS/PENALTY: Truncation detection ============
        truncation_penalty = 0.0
        # Check if response appears truncated (ends mid-word or mid-sentence)
        if response_text and response_text[-1] not in '.!?"\')':
            # Might be truncated
            last_char = response_text[-1]
            if last_char.isalpha() and num_words > 20:
                truncation_penalty = 5.0
        
        # ============ BONUS: Structural markers ============
        structure_bonus = 0.0
        # Reward use of clear structural elements when appropriate
        if num_words > 30:
            if re.search(r'\b(first|second|third|finally|additionally|however|moreover|in contrast|on the other hand)\b', response_text.lower()):
                structure_bonus += 2.0
            # Lists or bullet points
            if re.search(r'(\n\s*[-•*]\s|\n\s*\d+[.)]\s)', response_text):
                structure_bonus += 1.0
        
        # ============ COMBINE SCORES ============
        total = (
            length_score +       # 0-15
            word_rep_score +     # 0-20
            phrase_rep_score +   # 0-20
            sentence_score +     # 0-15
            substance_score +    # 0-15
            filler_score +       # 0-15
            structure_bonus -    # 0-3
            truncation_penalty   # 0-5
        )
        
        # Normalize to 0-100 range
        max_possible = 15 + 20 + 20 + 15 + 15 + 15 + 3  # 103
        normalized = (total / max_possible) * 100.0
        
        # Clamp
        normalized = max(0.0, min(100.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        # Fallback: return a middle-of-the-road score
        try:
            if not response or not response.strip():
                return 0.0
            return 40.0
        except Exception:
            return 40.0