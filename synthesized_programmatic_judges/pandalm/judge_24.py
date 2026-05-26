def judging_function(query, response):
    """
    Evaluates completeness and coverage using a structural decomposition approach:
    - Decomposes the query into sub-questions/aspects and checks coverage
    - Measures information density via unique clause/sentence analysis
    - Detects repetition/padding vs genuine content
    - Evaluates explanation depth through causal/reasoning markers
    - Checks for examples, specifics, and elaboration patterns
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not response.strip():
            return 0.0

        if not query or not query.strip():
            return 5.0

        query = query.strip()
        response = response.strip()

        response_words = response.lower().split()
        query_words = query.lower().split()

        if len(response_words) == 0:
            return 0.0

        # ---- 1. Query Aspect Extraction & Coverage ----
        # Extract key content words from query (nouns, verbs, adjectives approximation)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'that', 'this', 'these', 'those', 'what', 'which',
            'who', 'whom', 'its', 'it', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'up',
            'about', 'give', 'given', 'following', 'input', 'provide', 'write',
            'describe', 'explain', 'generate', 'create', 'make', 'come'
        }

        query_content_words = [w for w in re.findall(r'[a-z]+', query.lower())
                               if w not in stop_words and len(w) > 2]
        response_lower = response.lower()

        if query_content_words:
            covered = sum(1 for w in set(query_content_words) if w in response_lower)
            query_coverage = covered / len(set(query_content_words))
        else:
            query_coverage = 0.5

        # ---- 2. Sentence-level Analysis ----
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)

        # Unique sentence content - detect repetition at sentence level
        sentence_fingerprints = set()
        for s in sentences:
            words = tuple(sorted(set(re.findall(r'[a-z]+', s.lower()))))
            sentence_fingerprints.add(words)

        sentence_uniqueness = len(sentence_fingerprints) / num_sentences if num_sentences > 0 else 0

        # ---- 3. Repetition Penalty (word-level) ----
        word_counts = Counter(w for w in response_words if w not in stop_words and len(w) > 2)
        if word_counts:
            total_content = sum(word_counts.values())
            unique_content = len(word_counts)
            # High repetition ratio = bad
            repetition_ratio = 1.0 - (unique_content / total_content) if total_content > 0 else 0
            # Check for extreme repetition (same word appears way too much)
            max_freq = max(word_counts.values())
            extreme_repetition = max_freq / total_content if total_content > 0 else 0
        else:
            repetition_ratio = 0.5
            extreme_repetition = 0

        repetition_penalty = 0
        if extreme_repetition > 0.3:
            repetition_penalty = extreme_repetition * 15
        if repetition_ratio > 0.7:
            repetition_penalty += (repetition_ratio - 0.7) * 10

        # ---- 4. Explanation Depth Markers ----
        # Causal/reasoning connectors indicate deeper explanation
        depth_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bthis means\b',
            r'\bin order to\b', r'\bso that\b', r'\bdue to\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bincluding\b', r'\billustrat\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bnevertheless\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bnot only\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bimportantly\b', r'\bsignificantly\b',
        ]

        depth_count = 0
        for pattern in depth_markers:
            depth_count += len(re.findall(pattern, response_lower))

        # Normalize by response length
        depth_density = depth_count / num_sentences if num_sentences > 0 else 0
        depth_score = min(depth_density * 2.5, 3.0)

        # ---- 5. Specificity Detection ----
        # Numbers, proper nouns, quoted terms, technical terms
        numbers = len(re.findall(r'\b\d+\.?\d*\b', response))
        quoted = len(re.findall(r'"[^"]+"|\'[^\']+\'|"[^"]+"|"[^"]+"', response))
        # Words with mixed case or capitalized mid-sentence (proxy for proper nouns/terms)
        capitalized_mid = len(re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]{2,}', response))

        specificity_score = min((numbers * 0.3 + quoted * 0.5 + capitalized_mid * 0.1), 2.0)

        # ---- 6. Multi-aspect Detection ----
        # Check if query asks for multiple things (and, or, also, both, compare, contrast, list)
        multi_aspect_query = bool(re.search(
            r'\band\b|\bor\b|\balso\b|\bboth\b|\bcompar|\bcontrast|\blist\b|\bdifference|\bsimilar',
            query.lower()
        ))

        # Count distinct "topic shifts" in response via paragraph/sentence diversity
        if multi_aspect_query and len(sentences) >= 2:
            # Check if different sentences cover different vocabulary
            sentence_word_sets = []
            for s in sentences:
                words_set = set(re.findall(r'[a-z]+', s.lower())) - stop_words
                if words_set:
                    sentence_word_sets.append(words_set)

            if len(sentence_word_sets) >= 2:
                pairwise_novelty = []
                for i in range(1, len(sentence_word_sets)):
                    prev_union = set()
                    for j in range(i):
                        prev_union |= sentence_word_sets[j]
                    new_words = sentence_word_sets[i] - prev_union
                    novelty = len(new_words) / max(len(sentence_word_sets[i]), 1)
                    pairwise_novelty.append(novelty)
                avg_novelty = sum(pairwise_novelty) / len(pairwise_novelty)
                multi_aspect_score = avg_novelty * 3.0
            else:
                multi_aspect_score = 0.0
        else:
            multi_aspect_score = 0.0

        # ---- 7. Length Adequacy (non-linear) ----
        # Not just "longer is better" but adequate length relative to query complexity
        query_question_marks = query.count('?')
        query_complexity = len(query_content_words) + query_question_marks * 3

        # Expected minimum word count based on query complexity
        expected_min = max(15, query_complexity * 3)
        length_ratio = len(response_words) / expected_min

        # Sigmoid-like adequacy score
        if length_ratio < 0.3:
            length_score = length_ratio * 3  # very short = bad
        elif length_ratio < 1.0:
            length_score = 0.9 + (length_ratio - 0.3) * 1.5
        elif length_ratio < 3.0:
            length_score = 1.95 + (length_ratio - 1.0) * 0.3  # diminishing returns
        else:
            length_score = 2.55  # cap

        # ---- 8. Structural Completeness ----
        # Does response have introduction + body + conclusion patterns?
        has_intro = bool(re.search(
            r'^.{10,}', response  # non-trivial opening
        ))
        has_conclusion = bool(re.search(
            r'\b(in conclusion|overall|in summary|to summarize|ultimately|therefore)\b',
            response_lower
        ))
        has_elaboration = num_sentences >= 3

        structural_score = (0.3 * has_intro + 0.3 * has_elaboration + 0.2 * has_conclusion +
                           0.2 * (num_sentences >= 5))

        # ---- 9. Empty/Garbage Detection ----
        # Check for <noinput>, placeholder text, or clearly broken responses
        garbage_patterns = [r'<noinput>', r'^\s*$', r'^N/A$', r'^\[.*\]$']
        is_garbage = any(re.search(p, response.strip(), re.IGNORECASE) for p in garbage_patterns)
        if is_garbage:
            return 0.5

        # Check for truncation (ends mid-word or mid-sentence without punctuation)
        truncation_penalty = 0
        if len(response) > 50 and not re.search(r'[.!?"\']$', response.strip()):
            truncation_penalty = 0.5

        # ---- 10. Combine Scores ----
        # Weighted combination
        total = (
            query_coverage * 25 +          # 0-25: covers query aspects
            depth_score * 6 +              # 0-18: explanation depth
            specificity_score * 4 +         # 0-8: concrete details
            multi_aspect_score * 4 +        # 0-12: covers multiple aspects
            length_score * 8 +             # 0-20.4: adequate length
            structural_score * 8 +          # 0-8: structural completeness
            sentence_uniqueness * 5 -       # 0-5: unique sentences
            repetition_penalty -            # penalty for repetition
            truncation_penalty * 3          # penalty for truncation
        )

        # Clamp to 0-100
        total = max(0.0, min(100.0, total))

        return round(total, 2)

    except Exception:
        # Fallback: simple length-based score
        try:
            words = len(response.split()) if response else 0
            return min(words * 0.5, 50.0)
        except Exception:
            return 0.0