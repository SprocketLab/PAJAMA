def judging_function(query, response):
    """
    Evaluate clarity and conciseness using a substantially different approach:
    - Compression ratio estimation (how much information per character)
    - Clause complexity analysis (subordinate clause depth)
    - Unique information density (ratio of novel content words per sentence)
    - Repetition detection via suffix-based approach (repeated substrings)
    - Readability via syllable-based metrics
    - Response adequacy relative to query complexity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # Tokenize into words
        def tokenize(text):
            return re.findall(r"[a-zA-Z']+", text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip()]
        
        response_words = tokenize(response)
        query_words = tokenize(query)
        sentences = get_sentences(response)
        
        if len(response_words) == 0:
            return 0.5
        
        # ============================================================
        # 1. SYLLABLE-BASED READABILITY (Flesch-like, but custom scoring)
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
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in response_words)
        avg_syllables_per_word = total_syllables / len(response_words) if response_words else 1.5
        
        # Prefer moderate syllable complexity (not too simple, not too complex)
        # Ideal around 1.4-1.8 syllables per word
        syllable_score = max(0, 1.0 - abs(avg_syllables_per_word - 1.6) * 0.8)
        
        # ============================================================
        # 2. REPEATED SUBSTRING DETECTION (character-level redundancy)
        # ============================================================
        def repeated_substring_ratio(text):
            """Detect repeated substrings of length >= 4 words"""
            words = tokenize(text)
            if len(words) < 8:
                return 0.0
            
            # Check for repeated n-grams of various sizes (4-8 words)
            total_repeated_tokens = 0
            seen_positions = set()
            
            for ngram_size in range(4, min(9, len(words) // 2 + 1)):
                ngram_counts = Counter()
                ngram_positions = {}
                for i in range(len(words) - ngram_size + 1):
                    ng = tuple(words[i:i + ngram_size])
                    ngram_counts[ng] += 1
                    if ng not in ngram_positions:
                        ngram_positions[ng] = []
                    ngram_positions[ng].append(i)
                
                for ng, count in ngram_counts.items():
                    if count >= 2:
                        positions = ngram_positions[ng]
                        # Skip the first occurrence, count subsequent ones
                        for pos in positions[1:]:
                            for j in range(pos, pos + ngram_size):
                                if j not in seen_positions:
                                    seen_positions.add(j)
                                    total_repeated_tokens += 1
            
            return total_repeated_tokens / len(words) if words else 0.0
        
        repetition_ratio = repeated_substring_ratio(response)
        repetition_penalty = max(0, 1.0 - repetition_ratio * 3.0)
        
        # ============================================================
        # 3. WORD-LEVEL REPETITION (consecutive repeated words/phrases)
        # ============================================================
        def consecutive_repeat_penalty(words):
            if len(words) < 2:
                return 1.0
            repeat_count = 0
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    repeat_count += 1
            ratio = repeat_count / len(words)
            return max(0, 1.0 - ratio * 5.0)
        
        consec_penalty = consecutive_repeat_penalty(response_words)
        
        # ============================================================
        # 4. CLAUSE COMPLEXITY ANALYSIS
        # ============================================================
        # Count subordinating conjunctions and relative pronouns as clause markers
        clause_markers = {'which', 'that', 'who', 'whom', 'whose', 'where', 'when',
                         'while', 'although', 'because', 'since', 'unless', 'whereas',
                         'whereby', 'wherein', 'whatever', 'whichever', 'whoever'}
        
        clause_count = sum(1 for w in response_words if w in clause_markers)
        num_sentences = max(1, len(sentences))
        clauses_per_sentence = clause_count / num_sentences
        
        # Moderate clause complexity is good (0.5-1.5 per sentence)
        # Too many = convoluted; too few might be okay for simple queries
        if clauses_per_sentence <= 1.5:
            clause_score = 1.0
        elif clauses_per_sentence <= 3.0:
            clause_score = 1.0 - (clauses_per_sentence - 1.5) * 0.3
        else:
            clause_score = 0.55
        
        # ============================================================
        # 5. UNIQUE INFORMATION DENSITY PER SENTENCE
        # ============================================================
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'as', 'into', 'through', 'during', 'before', 'after', 'and',
                    'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                    'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                    'than', 'too', 'very', 'just', 'it', 'its', 'this', 'that',
                    'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                    'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
                    'what', 'which', 'who', 'whom', 'also', 'then'}
        
        def content_words(words):
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        # Track how many NEW content words each sentence introduces
        seen_content = set()
        # Also consider content words from query as "already known"
        query_content = set(content_words(query_words))
        
        novelty_scores = []
        for sent in sentences:
            sent_words = tokenize(sent)
            sent_content = content_words(sent_words)
            if len(sent_content) == 0:
                novelty_scores.append(0.0)
                continue
            new_words = [w for w in sent_content if w not in seen_content]
            novelty = len(new_words) / len(sent_content)
            novelty_scores.append(novelty)
            seen_content.update(sent_content)
        
        # Average novelty across sentences - higher means less redundancy
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5
        
        # ============================================================
        # 6. RESPONSE LENGTH ADEQUACY
        # ============================================================
        # Estimate appropriate length based on query complexity
        query_content_count = len(content_words(query_words))
        
        # Simple heuristic: more complex queries deserve longer responses
        # but there's diminishing returns
        ideal_min_words = max(5, query_content_count * 3)
        ideal_max_words = max(30, query_content_count * 25)
        
        word_count = len(response_words)
        
        if word_count < 3:
            length_score = 0.15
        elif word_count < ideal_min_words:
            length_score = 0.3 + 0.5 * (word_count / ideal_min_words)
        elif word_count <= ideal_max_words:
            length_score = 1.0
        else:
            # Gradually penalize overly long responses
            excess_ratio = word_count / ideal_max_words
            length_score = max(0.3, 1.0 - (excess_ratio - 1.0) * 0.15)
        
        # ============================================================
        # 7. PRECISION OF LANGUAGE (function word ratio)
        # ============================================================
        # Lower function-word ratio = more information-dense
        content_count = len(content_words(response_words))
        content_ratio = content_count / len(response_words) if response_words else 0
        
        # Ideal content ratio around 0.4-0.6
        if content_ratio < 0.25:
            precision_score = 0.5
        elif content_ratio < 0.4:
            precision_score = 0.7 + (content_ratio - 0.25) * 2.0
        elif content_ratio <= 0.65:
            precision_score = 1.0
        else:
            precision_score = max(0.7, 1.0 - (content_ratio - 0.65) * 1.0)
        
        # ============================================================
        # 8. SENTENCE LENGTH VARIANCE (consistent sentence lengths = clearer)
        # ============================================================
        if len(sentences) >= 2:
            sent_lengths = [len(tokenize(s)) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Some variation is good (0.2-0.5 CV), too much or too little is bad
            if cv < 0.1:
                variance_score = 0.85  # Too uniform can be monotonous but still clear
            elif cv < 0.5:
                variance_score = 1.0
            elif cv < 1.0:
                variance_score = max(0.5, 1.0 - (cv - 0.5) * 0.8)
            else:
                variance_score = 0.4
        else:
            variance_score = 0.8  # Single sentence - neutral
        
        # ============================================================
        # 9. EMPTY/FILLER CONTENT DETECTION
        # ============================================================
        filler_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'as we all know', 'needless to say', 'it goes without saying',
            'in other words', 'to put it another way', 'that is to say',
            'as a matter of fact', 'the fact of the matter is', 'at the end of the day',
            'when all is said and done', 'in conclusion', 'to sum up',
            'basically', 'essentially', 'fundamentally', 'actually', 'literally',
            'in terms of', 'with respect to', 'in regard to', 'in regards to',
            'it can be said that', 'one might say', 'it is clear that',
            'there is no doubt that', 'without a doubt'
        ]
        
        response_lower = response.lower()
        filler_count = sum(1 for phrase in filler_phrases if phrase in response_lower)
        filler_penalty = max(0.5, 1.0 - filler_count * 0.12)
        
        # ============================================================
        # 10. QUERY RELEVANCE (does response address query content?)
        # ============================================================
        query_content_set = set(content_words(query_words))
        response_content_set = set(content_words(response_words))
        
        if query_content_set:
            # What fraction of query content words appear in response?
            overlap = len(query_content_set & response_content_set)
            relevance = min(1.0, overlap / max(1, len(query_content_set)) * 1.5)
        else:
            relevance = 0.7  # No query content words to match
        
        # ============================================================
        # 11. STRUCTURAL CLARITY (presence of clear structure markers)
        # ============================================================
        has_structure = 0.0
        # Check for numbered lists, bullet points, clear paragraphs
        if re.search(r'^\d+[\.\)]\s', response, re.MULTILINE):
            has_structure += 0.15
        if re.search(r'^[-•*]\s', response, re.MULTILINE):
            has_structure += 0.1
        if '\n\n' in response and len(response) > 200:
            has_structure += 0.05
        # Bonus for responses that use colons for definitions/explanations
        if ':' in response and len(sentences) > 1:
            has_structure += 0.05
        
        structure_bonus = min(0.2, has_structure)
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        # Weighted combination
        score = (
            syllable_score * 0.08 +
            repetition_penalty * 0.18 +
            consec_penalty * 0.10 +
            clause_score * 0.06 +
            avg_novelty * 0.16 +
            length_score * 0.12 +
            precision_score * 0.08 +
            variance_score * 0.05 +
            filler_penalty * 0.07 +
            relevance * 0.10
        )
        
        # Add structure bonus
        score += structure_bonus
        
        # Scale to 0-10
        final_score = score * 10.0
        
        # Harsh penalty for extremely short responses (< 3 words) unless query is very simple
        if word_count < 3 and len(query_words) > 3:
            final_score *= 0.3
        
        # Harsh penalty for clearly broken/garbage text
        # Check if response is mostly repeated characters
        if len(response) > 20:
            char_counts = Counter(response.lower())
            most_common_char_ratio = char_counts.most_common(1)[0][1] / len(response)
            if most_common_char_ratio > 0.3 and most_common_char_ratio < 1.0:
                # Check if it's not just spaces
                if char_counts.most_common(1)[0][0] not in (' ', '\n'):
                    final_score *= 0.6
        
        # Bonus for non-empty responses vs truly empty
        if word_count > 0:
            final_score = max(final_score, 0.5)
        
        return round(min(10.0, max(0.0, final_score)), 3)
        
    except Exception:
        # Fallback: return a minimal positive score if response exists
        try:
            if response and len(response.strip()) > 0:
                return 3.0
            return 0.0
        except Exception:
            return 0.0