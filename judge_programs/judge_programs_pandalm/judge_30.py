def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using sentence-level analysis,
    structural coherence patterns, and information density metrics.
    
    This variant focuses on:
    - Sentence-level structure quality (clause complexity)
    - Information density via unique content words per sentence
    - Repetition detection at multiple granularities
    - Factual language patterns (specificity markers, attribution patterns)
    - Hallucination red-flags (absolute claims, sensationalism)
    - Response completeness relative to query complexity
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
        
        # === 1. Sentence-level analysis ===
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # === 2. Truncation / incompleteness detection ===
        truncation_penalty = 0.0
        if response_stripped[-1] not in '.!?"\')':
            truncation_penalty = -5.0
        # Check if response ends mid-word or mid-sentence
        last_sentence = sentences[-1] if sentences else response_stripped
        words_last = last_sentence.split()
        if len(words_last) < 3 and len(sentences) > 1:
            truncation_penalty -= 2.0
        
        # === 3. Repetition analysis at multiple levels ===
        words = re.findall(r'[a-z]+', response_stripped.lower())
        total_words = max(len(words), 1)
        
        # Trigram repetition
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i:i+3]))
        trigram_counter = Counter(trigrams)
        if trigrams:
            repeated_trigrams = sum(c - 1 for c in trigram_counter.values() if c > 1)
            trigram_repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
        else:
            trigram_repetition_ratio = 0.0
        
        # Sentence-level repetition (near-duplicate sentences)
        sentence_texts = [re.sub(r'[^a-z\s]', '', s.lower()).strip() for s in sentences]
        sentence_counter = Counter(sentence_texts)
        duplicate_sentences = sum(c - 1 for c in sentence_counter.values() if c > 1)
        sentence_repetition_ratio = duplicate_sentences / num_sentences
        
        # Word-level repetition (beyond stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'only', 'same', 'than',
            'too', 'very', 'just', 'also', 'that', 'this', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'he', 'she', 'his', 'her',
            'we', 'our', 'you', 'your', 'i', 'me', 'my', 'which', 'who', 'whom',
            'what', 'where', 'when', 'how', 'while', 'if', 'then', 'about', 'up',
            'out', 'over', 'between', 'under', 'again', 'there', 'here', 'once',
        }
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        content_word_count = max(len(content_words), 1)
        unique_content = len(set(content_words))
        content_diversity = unique_content / content_word_count if content_word_count > 0 else 0
        
        repetition_penalty = -(trigram_repetition_ratio * 15 + sentence_repetition_ratio * 20)
        
        # === 4. Information density per sentence ===
        info_per_sentence_scores = []
        for s in sentences:
            s_words = re.findall(r'[a-z]+', s.lower())
            s_content = [w for w in s_words if w not in stopwords and len(w) > 2]
            s_unique_content = len(set(s_content))
            info_per_sentence_scores.append(s_unique_content)
        
        avg_info_per_sentence = sum(info_per_sentence_scores) / num_sentences if info_per_sentence_scores else 0
        # Reward moderate info density (not too sparse, not incoherent)
        info_density_score = min(avg_info_per_sentence / 5.0, 1.0) * 10  # caps at 10
        
        # === 5. Clause complexity (subordinate clauses, connectives) ===
        subordinating_conjunctions = [
            'because', 'although', 'though', 'since', 'unless', 'whereas',
            'while', 'when', 'where', 'if', 'whether', 'after', 'before',
            'until', 'once', 'provided', 'assuming', 'given'
        ]
        relative_pronouns = ['which', 'who', 'whom', 'whose', 'that', 'where', 'when']
        
        clause_markers = 0
        words_lower = response_stripped.lower()
        for conj in subordinating_conjunctions:
            clause_markers += len(re.findall(r'\b' + conj + r'\b', words_lower))
        for rp in relative_pronouns:
            # Only count when used mid-sentence (not at start)
            clause_markers += len(re.findall(r',\s*' + rp + r'\b', words_lower))
        
        clause_complexity_score = min(clause_markers / max(num_sentences * 0.5, 1), 1.0) * 5
        
        # === 6. Factual specificity markers ===
        # Numbers, dates, proper nouns (capitalized words not at sentence start)
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', response_stripped)
        number_score = min(len(numbers) * 0.5, 3.0)
        
        # Proper attribution / citation patterns
        attribution_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin particular\b', r'\bspecifically\b', r'\bnamely\b',
            r'\bevidence\b', r'\bdata\b', r'\bfound that\b',
            r'\bsuggests?\b', r'\bindicates?\b', r'\bdemonstrates?\b'
        ]
        attribution_count = 0
        for pat in attribution_patterns:
            attribution_count += len(re.findall(pat, words_lower))
        attribution_score = min(attribution_count * 1.0, 5.0)
        
        # === 7. Appropriate hedging (epistemic markers) ===
        hedging_words = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends? to\b', r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bpossibly\b', r'\bperhaps\b', r'\blikely\b', r'\bprobably\b',
            r'\bapproximately\b', r'\babout\b', r'\baround\b',
            r'\bin some cases\b', r'\bit depends\b', r'\bcan vary\b',
            r'\bnot necessarily\b', r'\bto some extent\b'
        ]
        hedge_count = 0
        for pat in hedging_words:
            hedge_count += len(re.findall(pat, words_lower))
        # Moderate hedging is good; too much is wishy-washy
        hedge_ratio = hedge_count / num_sentences
        if hedge_ratio <= 0.5:
            hedging_score = hedge_count * 1.0
        else:
            hedging_score = 0.5 * num_sentences - (hedge_count - 0.5 * num_sentences) * 0.5
        hedging_score = min(max(hedging_score, 0), 4.0)
        
        # === 8. Hallucination / sensationalism red flags ===
        red_flag_patterns = [
            r'\balways\b', r'\bnever\b', r'\babsolutely\b', r'\bdefinitely\b',
            r'\bwithout a doubt\b', r'\bundeniably\b', r'\bincontrovertibly\b',
            r'\beveryone knows\b', r'\bit is obvious\b', r'\bclearly\b',
            r'\bshocking\b', r'\bincredible\b', r'\bunbelievable\b',
            r'\bamazing\b', r'\bterrible\b', r'\bhorrible\b',
            r'\bconspiracy\b', r'\bthey don\'t want you to know\b',
            r'\bsecret(?:ly)?\b', r'\bhidden truth\b', r'\bwake up\b',
            r'\bsheeple\b', r'\bmainstream media\b',
            r'\b100%\b', r'\bguaranteed\b', r'\bproven fact\b'
        ]
        red_flag_count = 0
        for pat in red_flag_patterns:
            red_flag_count += len(re.findall(pat, words_lower))
        red_flag_penalty = -min(red_flag_count * 2.0, 10.0)
        
        # === 9. Structural coherence: paragraph/sentence flow ===
        # Check for discourse markers that indicate logical flow
        discourse_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bconsequently\b',
            r'\btherefore\b', r'\bthus\b', r'\bas a result\b', r'\bmeanwhile\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bfirst(?:ly)?\b',
            r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b', r'\bfinally\b',
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\badditionally\b'
        ]
        discourse_count = 0
        for pat in discourse_markers:
            discourse_count += len(re.findall(pat, words_lower))
        discourse_score = min(discourse_count * 1.5, 5.0)
        
        # === 10. Response length adequacy relative to query ===
        query_words = re.findall(r'[a-z]+', query.lower())
        query_word_count = len(query_words)
        
        # Estimate expected response length based on query complexity
        # Complex queries (longer, with question words) deserve longer answers
        question_indicators = ['explain', 'describe', 'compare', 'contrast', 'analyze',
                               'discuss', 'elaborate', 'detail', 'how', 'why', 'what']
        query_complexity = sum(1 for w in query_words if w in question_indicators)
        
        expected_min_words = 15 + query_complexity * 10
        length_ratio = total_words / max(expected_min_words, 1)
        
        if length_ratio < 0.3:
            length_score = -5.0
        elif length_ratio < 0.6:
            length_score = -2.0
        elif length_ratio < 1.0:
            length_score = length_ratio * 5
        elif length_ratio < 3.0:
            length_score = 5.0
        else:
            length_score = 5.0 - min((length_ratio - 3.0) * 0.5, 3.0)
        
        # === 11. Query-response relevance via keyword coverage ===
        query_content = set(w for w in query_words if w not in stopwords and len(w) > 2)
        response_content_set = set(content_words)
        if query_content:
            coverage = len(query_content & response_content_set) / len(query_content)
        else:
            coverage = 0.5  # neutral if query has no content words
        relevance_score = coverage * 8
        
        # === 12. Explanation depth: definitions, examples, elaboration ===
        explanation_patterns = [
            r'\bthis means\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bwhich means\b',
            r'\brefers to\b', r'\bis defined as\b', r'\bis known as\b',
            r'\bcan be described as\b', r'\bthe reason\b', r'\bthis is because\b'
        ]
        explanation_count = 0
        for pat in explanation_patterns:
            explanation_count += len(re.findall(pat, words_lower))
        explanation_score = min(explanation_count * 2.0, 5.0)
        
        # === 13. Content diversity bonus ===
        # Reward responses that cover multiple aspects
        content_diversity_score = min(content_diversity * 8, 6.0)
        
        # === 14. Sentence length variance (natural writing has varied lengths) ===
        if num_sentences >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / num_sentences
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / num_sentences
            std_dev = math.sqrt(variance)
            # Some variance is good, but not too extreme
            if std_dev < 1:
                variance_score = 0.0  # too uniform
            elif std_dev < 8:
                variance_score = 2.0  # good variety
            else:
                variance_score = 1.0  # too wild
        else:
            variance_score = 0.0
        
        # === 15. Empty/garbage detection ===
        garbage_penalty = 0.0
        # Check for excessive non-alphabetic content
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        alpha_ratio = alpha_chars / max(len(response_stripped), 1)
        if alpha_ratio < 0.5:
            garbage_penalty = -10.0
        
        # Check for noinput or empty-like responses
        if re.search(r'<noinput>|<no\s*input>', words_lower):
            garbage_penalty = -15.0
        
        # === Combine all scores ===
        total_score = (
            info_density_score +          # 0 to 10
            clause_complexity_score +     # 0 to 5
            number_score +                # 0 to 3
            attribution_score +           # 0 to 5
            hedging_score +               # 0 to 4
            red_flag_penalty +            # -10 to 0
            discourse_score +             # 0 to 5
            length_score +                # -5 to 5
            relevance_score +             # 0 to 8
            explanation_score +           # 0 to 5
            content_diversity_score +     # 0 to 6
            variance_score +              # 0 to 2
            repetition_penalty +          # -35 to 0
            truncation_penalty +          # -7 to 0
            garbage_penalty               # -15 to 0
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~58, theoretical min ~-72
        # Practical good response: ~30-45, practical bad: ~0-15
        normalized = max(0.0, min(100.0, (total_score + 10) * 100 / 68))
        
        return round(normalized, 2)
    
    except Exception:
        return 25.0  # Safe fallback mid-range score