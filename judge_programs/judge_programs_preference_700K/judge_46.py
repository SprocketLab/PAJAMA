def judging_function(query, response):
    """
    Evaluates clarity and conciseness using:
    - Information density (content words vs function words ratio)
    - Sentence structure variance (penalize monotonous or overly complex structures)
    - Precision scoring (specific/concrete words vs vague/abstract ones)
    - Redundancy detection via sentence-level semantic overlap using word sets
    - Signal-to-noise ratio (meaningful content vs filler phrases)
    - Directness scoring (how quickly the response gets to the point)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0

        import re
        import math
        from collections import Counter

        response_text = response.strip()
        query_text = query.strip()

        if len(response_text) < 5:
            return 0.5

        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response_text.lower())
        if len(words) < 3:
            return 1.0

        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        if not sentences:
            sentences = [response_text]

        # ============ 1. FUNCTION WORD / CONTENT WORD ANALYSIS ============
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'just', 'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that',
            'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom', 'also', 'about', 'up',
        }

        content_words = [w for w in words if w not in function_words and len(w) > 2]
        func_words = [w for w in words if w in function_words]

        # Information density: ratio of content words
        info_density = len(content_words) / max(len(words), 1)
        # Ideal density is around 0.5-0.65; too high can be telegraphic, too low is bloated
        density_score = 1.0 - abs(info_density - 0.57) * 2.5
        density_score = max(0.0, min(1.0, density_score))

        # ============ 2. FILLER / WEASEL PHRASE DETECTION ============
        filler_phrases = [
            r'\bit is worth noting that\b', r'\bit should be noted that\b',
            r'\bit is important to note\b', r'\bit goes without saying\b',
            r'\bneedless to say\b', r'\bas a matter of fact\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bfor what it\'s worth\b', r'\bat the end of the day\b',
            r'\bin my opinion\b', r'\bi think that\b', r'\bi believe that\b',
            r'\bi would say\b', r'\bi would argue\b',
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b',
            r'\bliterally\b', r'\bobviously\b', r'\bclearly\b',
            r'\bof course\b', r'\bin terms of\b', r'\bwith respect to\b',
            r'\bwith regard to\b', r'\bin this regard\b',
            r'\bthat being said\b', r'\bhaving said that\b',
            r'\ball things considered\b', r'\bas previously mentioned\b',
            r'\bas i mentioned\b', r'\bas stated\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bto be honest\b', r'\bto be fair\b',
            r'\bquite\b', r'\brather\b', r'\bsomewhat\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\bpotentially\b',
        ]

        response_lower = response_text.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))

        filler_rate = filler_count / max(len(sentences), 1)
        filler_score = max(0.0, 1.0 - filler_rate * 0.3)

        # ============ 3. SENTENCE LENGTH VARIANCE & QUALITY ============
        sent_lengths = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
        sent_lengths = [sl for sl in sent_lengths if sl > 0]

        if len(sent_lengths) >= 2:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            # Ideal average sentence length: 12-22 words
            if avg_sent_len < 5:
                len_score = 0.5
            elif avg_sent_len <= 22:
                len_score = min(1.0, 0.5 + (avg_sent_len - 5) / 34.0)
            elif avg_sent_len <= 35:
                len_score = max(0.3, 1.0 - (avg_sent_len - 22) / 26.0)
            else:
                len_score = 0.2

            # Sentence length variance — some variance is good (rhythm), too much is bad
            variance = sum((sl - avg_sent_len) ** 2 for sl in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(avg_sent_len, 1)  # coefficient of variation
            # Ideal CV around 0.3-0.6
            if cv < 0.1:
                var_score = 0.6  # too monotonous
            elif cv <= 0.6:
                var_score = 0.8 + 0.2 * (cv - 0.1) / 0.5
            elif cv <= 1.0:
                var_score = max(0.5, 1.0 - (cv - 0.6) * 0.75)
            else:
                var_score = 0.4
        else:
            avg_sent_len = len(words)
            len_score = 0.6 if 8 <= avg_sent_len <= 30 else 0.4
            var_score = 0.5

        # ============ 4. REDUNDANCY DETECTION (sentence-level overlap) ============
        def get_content_word_set(text):
            w = re.findall(r'[a-zA-Z]+', text.lower())
            return set(w2 for w2 in w if w2 not in function_words and len(w2) > 2)

        redundancy_penalties = 0
        if len(sentences) >= 2:
            sent_word_sets = [get_content_word_set(s) for s in sentences]
            pair_count = 0
            total_overlap = 0.0
            for i in range(len(sent_word_sets)):
                for j in range(i + 1, min(i + 4, len(sent_word_sets))):
                    # Only compare nearby sentences
                    s1, s2 = sent_word_sets[i], sent_word_sets[j]
                    if len(s1) > 0 and len(s2) > 0:
                        union = s1 | s2
                        if len(union) > 0:
                            overlap = len(s1 & s2) / len(union)
                            total_overlap += overlap
                            pair_count += 1
                            if overlap > 0.6:
                                redundancy_penalties += 1

            avg_overlap = total_overlap / max(pair_count, 1)
            redundancy_score = max(0.0, 1.0 - avg_overlap * 1.2 - redundancy_penalties * 0.15)
        else:
            redundancy_score = 0.8  # single sentence, neutral

        # ============ 5. DIRECTNESS / GETTING TO THE POINT ============
        # Check if first sentence contains content relevant to query
        query_words = set(re.findall(r'[a-zA-Z]+', query_text.lower()))
        query_content = query_words - function_words
        query_content = {w for w in query_content if len(w) > 2}

        first_sent_words = get_content_word_set(sentences[0]) if sentences else set()
        if len(query_content) > 0 and len(first_sent_words) > 0:
            first_sent_relevance = len(first_sent_words & query_content) / max(len(query_content), 1)
            directness_score = min(1.0, 0.4 + first_sent_relevance * 2.0)
        else:
            directness_score = 0.5

        # Penalize responses that start with meta-commentary
        meta_starts = [
            r'^(sure|okay|great|well|so|alright)',
            r'^(that\'s a great question|good question|interesting question)',
            r'^(i\'d be happy to|i can help|let me)',
            r'^(welcome to|please read)',
            r'^(thank you for|thanks for)',
        ]
        first_words = response_lower[:100]
        meta_penalty = 0
        for pattern in meta_starts:
            if re.search(pattern, first_words):
                meta_penalty = 0.05
                break

        directness_score = max(0.0, directness_score - meta_penalty)

        # ============ 6. SPECIFICITY / CONCRETENESS ============
        # Detect specific details: numbers, proper nouns, technical terms, examples
        specific_markers = 0
        # Numbers and percentages
        specific_markers += len(re.findall(r'\d+%?', response_text))
        # Quoted terms or titles (in asterisks, quotes, backticks)
        specific_markers += len(re.findall(r'[*"`][\w\s]+[*"`]', response_text))
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response_text)
        specific_markers += len(proper_nouns)
        # Parenthetical clarifications
        specific_markers += len(re.findall(r'\([^)]+\)', response_text))

        specificity_rate = specific_markers / max(len(words) / 20, 1)
        specificity_score = min(1.0, 0.3 + specificity_rate * 0.25)

        # ============ 7. RESPONSE LENGTH APPROPRIATENESS ============
        word_count = len(words)
        # Moderate length responses tend to be better for clarity
        if word_count < 15:
            length_appropriateness = 0.3
        elif word_count < 30:
            length_appropriateness = 0.5 + (word_count - 15) / 30.0
        elif word_count <= 200:
            length_appropriateness = 1.0
        elif word_count <= 400:
            length_appropriateness = max(0.6, 1.0 - (word_count - 200) / 500.0)
        else:
            length_appropriateness = max(0.4, 0.6 - (word_count - 400) / 2000.0)

        # ============ 8. UNIQUE VOCABULARY RICHNESS ============
        if len(content_words) > 0:
            unique_content = set(content_words)
            vocab_richness = len(unique_content) / max(len(content_words), 1)
            # Penalize very low richness (lots of repetition)
            if vocab_richness > 0.8:
                vocab_score = 1.0
            elif vocab_richness > 0.5:
                vocab_score = 0.6 + (vocab_richness - 0.5) * 1.33
            else:
                vocab_score = max(0.2, vocab_richness * 1.2)
        else:
            vocab_score = 0.3

        # ============ 9. STRUCTURAL CLARITY (formatting signals) ============
        has_code_block = 1 if '```' in response_text else 0
        has_list = 1 if re.search(r'(?m)^[\s]*[-*•]\s', response_text) or re.search(r'(?m)^[\s]*\d+[.)]\s', response_text) else 0
        has_paragraph_breaks = 1 if '\n\n' in response_text else 0

        # Structural elements help clarity for longer responses
        if word_count > 80:
            structure_score = 0.5 + 0.15 * has_list + 0.15 * has_paragraph_breaks + 0.1 * has_code_block
        else:
            structure_score = 0.7 + 0.1 * has_list + 0.1 * has_paragraph_breaks

        # ============ 10. CLAUSE COMPLEXITY (subordinate clause density) ============
        subordinators = [
            r'\balthough\b', r'\bwhereas\b', r'\bnevertheless\b', r'\bnotwithstanding\b',
            r'\binasmuch\b', r'\bwherein\b', r'\bwhereby\b', r'\btherefore\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\bhowever\b', r'\bconsequently\b',
        ]
        subordinate_count = 0
        for pat in subordinators:
            subordinate_count += len(re.findall(pat, response_lower))

        # Some subordination is fine, too much is convoluted
        sub_rate = subordinate_count / max(len(sentences), 1)
        if sub_rate <= 0.3:
            complexity_score = 1.0
        elif sub_rate <= 0.8:
            complexity_score = 0.8
        else:
            complexity_score = max(0.4, 1.0 - sub_rate * 0.5)

        # ============ COMBINE SCORES ============
        weights = {
            'density': 0.10,
            'filler': 0.12,
            'sent_len': 0.10,
            'variance': 0.07,
            'redundancy': 0.15,
            'directness': 0.12,
            'specificity': 0.10,
            'length_approp': 0.06,
            'vocab': 0.08,
            'structure': 0.05,
            'complexity': 0.05,
        }

        composite = (
            weights['density'] * density_score +
            weights['filler'] * filler_score +
            weights['sent_len'] * len_score +
            weights['variance'] * var_score +
            weights['redundancy'] * redundancy_score +
            weights['directness'] * directness_score +
            weights['specificity'] * specificity_score +
            weights['length_approp'] * length_appropriateness +
            weights['vocab'] * vocab_score +
            weights['structure'] * structure_score +
            weights['complexity'] * complexity_score
        )

        # Scale to 0-10 range
        final_score = composite * 10.0

        # Bonus for substantive responses that are well-structured
        if word_count >= 40 and redundancy_score > 0.7 and specificity_score > 0.5:
            final_score += 0.5

        # Penalty for very short, potentially unhelpful responses
        if word_count < 20:
            final_score *= 0.7

        final_score = max(0.0, min(10.0, final_score))

        return round(final_score, 3)

    except Exception:
        try:
            # Fallback: simple word count based score
            words = response.split() if response else []
            if len(words) < 5:
                return 1.0
            elif len(words) < 20:
                return 3.0
            elif len(words) <= 200:
                return 5.5
            else:
                return 4.5
        except Exception:
            return 3.0