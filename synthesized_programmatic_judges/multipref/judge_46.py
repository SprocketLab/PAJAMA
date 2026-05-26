def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, sentence structure variance,
    directness of opening, and lexical precision metrics.
    
    This variant focuses on:
    - Information density (content words vs total words ratio)
    - Sentence structure variance (penalize monotonous same-length sentences)
    - Opening directness (how quickly the response addresses the query)
    - Lexical precision (specific words vs vague/generic words)
    - Parenthetical/qualifier burden
    - Response-to-query relevance via keyword coverage efficiency
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
        query = query.strip()
        
        if len(response) < 10:
            return 1.0
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Information Density
        # Ratio of content words (non-function words) to total words
        # ============================================================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'also', 'about', 'up', 'there', 'here', 'while',
            'although', 'though', 'since', 'until', 'unless', 'whether', 'however',
            'therefore', 'thus', 'hence', 'accordingly', 'moreover', 'furthermore',
            'additionally', 'besides', 'still', 'nonetheless', 'nevertheless',
            'regardless', 'instead', 'meanwhile', 'otherwise', 'rather', 'indeed',
        }
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        info_density = len(content_words) / total_words if total_words > 0 else 0.5
        # Ideal density is around 0.45-0.60
        info_density_score = 1.0 - abs(info_density - 0.52) * 2.5
        info_density_score = max(0.0, min(1.0, info_density_score))
        
        # ============================================================
        # FEATURE 2: Sentence Length Variance (Coefficient of Variation)
        # Good writing has varied sentence lengths; monotonous = bad
        # ============================================================
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 2:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance_sl)
            cv = std_sl / mean_sl if mean_sl > 0 else 0
            # CV around 0.4-0.7 is good variety
            if cv < 0.15:
                variety_score = 0.3  # too monotonous
            elif cv < 0.3:
                variety_score = 0.6
            elif cv < 0.8:
                variety_score = 1.0
            else:
                variety_score = 0.7  # too chaotic
        else:
            variety_score = 0.5
        
        # ============================================================
        # FEATURE 3: Opening Directness
        # How quickly does the response get to the point?
        # Penalize filler openings like "That's a great question!"
        # ============================================================
        filler_openings = [
            r'^(that\'?s?\s+a\s+(great|good|excellent|wonderful|fantastic|interesting)\s+(question|idea|thought))',
            r'^(great\s+question)',
            r'^(oh\s+wow)',
            r'^(well\s*,?\s*that\'?s?\s+)',
            r'^(i\'?m\s+glad\s+you\s+asked)',
            r'^(thank\s+you\s+for\s+(asking|your|the))',
            r'^(what\s+a\s+(great|good|excellent)\s+question)',
            r'^(absolutely\s*[!,.])',
            r'^(of\s+course\s*[!,.])',
        ]
        
        opening_text = response[:200].lower().strip()
        has_filler_opening = False
        for pattern in filler_openings:
            if re.search(pattern, opening_text):
                has_filler_opening = True
                break
        
        # Check if first sentence is substantive (addresses query content)
        first_sent = sentences[0] if sentences else ""
        first_sent_words = re.findall(r'[a-zA-Z]+', first_sent.lower())
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - function_words
        query_words = {w for w in query_words if len(w) > 2}
        
        first_sent_content = set(first_sent_words) - function_words
        if query_words:
            opening_relevance = len(first_sent_content & query_words) / max(len(query_words), 1)
        else:
            opening_relevance = 0.5
        
        directness_score = 0.7
        if has_filler_opening:
            directness_score -= 0.25
        directness_score += opening_relevance * 0.3
        directness_score = max(0.0, min(1.0, directness_score))
        
        # ============================================================
        # FEATURE 4: Lexical Precision
        # Penalize vague/weasel words; reward specific terminology
        # ============================================================
        vague_words = {
            'things', 'stuff', 'something', 'somehow', 'somewhat', 'somewhere',
            'basically', 'essentially', 'generally', 'typically', 'usually',
            'probably', 'possibly', 'perhaps', 'maybe', 'kind', 'sort',
            'really', 'actually', 'literally', 'virtually', 'practically',
            'quite', 'fairly', 'rather', 'pretty', 'extremely', 'incredibly',
            'absolutely', 'totally', 'completely', 'utterly', 'definitely',
            'certainly', 'obviously', 'clearly', 'apparently', 'seemingly',
            'various', 'several', 'numerous', 'many', 'lots', 'plenty',
            'certain', 'particular', 'specific', 'overall', 'entire',
        }
        
        vague_count = sum(1 for w in words if w in vague_words)
        vague_ratio = vague_count / total_words
        precision_score = 1.0 - (vague_ratio * 15)  # penalize heavily
        precision_score = max(0.2, min(1.0, precision_score))
        
        # ============================================================
        # FEATURE 5: Parenthetical / Qualifier Burden
        # Count parenthetical asides, excessive commas, qualifiers
        # ============================================================
        parenthetical_count = response.count('(') + response.count(')')
        # Excessive parentheticals per sentence
        paren_per_sent = parenthetical_count / (2 * num_sentences)
        paren_score = 1.0 - (paren_per_sent * 0.3)
        paren_score = max(0.3, min(1.0, paren_score))
        
        # Comma density (too many commas = convoluted)
        comma_count = response.count(',')
        comma_per_word = comma_count / total_words if total_words > 0 else 0
        # Normal is about 0.04-0.08 commas per word
        if comma_per_word > 0.12:
            comma_penalty = (comma_per_word - 0.12) * 8
        else:
            comma_penalty = 0
        paren_score -= comma_penalty
        paren_score = max(0.2, min(1.0, paren_score))
        
        # ============================================================
        # FEATURE 6: Query Keyword Coverage Efficiency
        # How efficiently does the response cover query topics?
        # (Cover more query words in fewer response words = better)
        # ============================================================
        response_word_set = set(words)
        if query_words:
            coverage = len(response_word_set & query_words) / len(query_words)
        else:
            coverage = 0.5
        
        # Efficiency: coverage per 100 words
        efficiency = coverage / (total_words / 100) if total_words > 0 else 0
        # Normalize: good efficiency is around 0.3-1.0
        efficiency_score = min(1.0, efficiency * 1.5)
        efficiency_score = max(0.1, efficiency_score)
        
        # ============================================================
        # FEATURE 7: Structural Formatting Bonus
        # Reward use of bold, numbered lists, headers for organization
        # But penalize excessive formatting without content
        # ============================================================
        has_bold = len(re.findall(r'\*\*[^*]+\*\*', response)) > 0
        has_numbered = len(re.findall(r'^\s*\d+[\.\)]\s', response, re.MULTILINE)) > 0
        has_headers = len(re.findall(r'^#{1,4}\s', response, re.MULTILINE)) > 0
        
        format_elements = sum([has_bold, has_numbered, has_headers])
        
        # Formatting is good for longer responses, less important for short
        if total_words > 80:
            format_score = 0.5 + format_elements * 0.17
        elif total_words > 40:
            format_score = 0.6 + format_elements * 0.1
        else:
            # Short responses don't need formatting
            format_score = 0.7
        format_score = min(1.0, format_score)
        
        # ============================================================
        # FEATURE 8: Repetition Detection (semantic-level)
        # Detect repeated phrases (3-grams) across the response
        # ============================================================
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i+1], words[i+2]))
        
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        total_trigrams = max(len(trigrams), 1)
        repetition_ratio = repeated_trigrams / total_trigrams
        
        # Also check for repeated sentence starters
        starters = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s.lower())
            if len(sw) >= 2:
                starters.append((sw[0], sw[1]))
        
        starter_counts = Counter(starters)
        repeated_starters = sum(1 for count in starter_counts.values() if count > 1)
        starter_repetition = repeated_starters / max(len(starters), 1)
        
        repetition_score = 1.0 - (repetition_ratio * 5) - (starter_repetition * 2)
        repetition_score = max(0.1, min(1.0, repetition_score))
        
        # ============================================================
        # FEATURE 9: Average Word Length (proxy for sophistication)
        # ============================================================
        avg_word_len = sum(len(w) for w in words) / total_words if total_words > 0 else 4
        # Ideal average word length is around 4.5-6.0
        if avg_word_len < 3.5:
            word_len_score = 0.4
        elif avg_word_len < 4.5:
            word_len_score = 0.7
        elif avg_word_len <= 6.5:
            word_len_score = 1.0
        elif avg_word_len <= 7.5:
            word_len_score = 0.8
        else:
            word_len_score = 0.5
        
        # ============================================================
        # FEATURE 10: Conciseness - penalize unnecessarily long responses
        # relative to query complexity
        # ============================================================
        query_word_count = len(re.findall(r'[a-zA-Z]+', query))
        # Rough expected response length based on query complexity
        if query_word_count <= 10:
            ideal_ratio = 12  # short questions need moderate answers
        elif query_word_count <= 25:
            ideal_ratio = 8
        else:
            ideal_ratio = 6
        
        actual_ratio = total_words / max(query_word_count, 1)
        if actual_ratio > ideal_ratio * 2:
            conciseness_score = 0.5
        elif actual_ratio > ideal_ratio * 1.5:
            conciseness_score = 0.7
        elif actual_ratio < 2:
            conciseness_score = 0.5  # too short
        else:
            conciseness_score = 1.0
        
        # ============================================================
        # FEATURE 11: Discourse Marker Appropriateness
        # Some discourse markers add clarity, too many add bloat
        # ============================================================
        discourse_markers = [
            'first', 'second', 'third', 'finally', 'lastly',
            'next', 'then', 'step', 'here', 'now', 'let',
        ]
        marker_count = sum(1 for w in words if w in discourse_markers)
        markers_per_100 = (marker_count / total_words) * 100 if total_words > 0 else 0
        
        if markers_per_100 < 0.5:
            discourse_score = 0.6  # could use more structure
        elif markers_per_100 <= 3.0:
            discourse_score = 1.0  # good use
        else:
            discourse_score = 0.7  # overuse
        
        # ============================================================
        # COMBINE SCORES with weights
        # ============================================================
        weights = {
            'info_density': 1.2,
            'variety': 0.8,
            'directness': 1.5,
            'precision': 1.3,
            'paren': 0.6,
            'efficiency': 0.9,
            'format': 1.0,
            'repetition': 1.1,
            'word_len': 0.5,
            'conciseness': 1.0,
            'discourse': 0.6,
        }
        
        scores = {
            'info_density': info_density_score,
            'variety': variety_score,
            'directness': directness_score,
            'precision': precision_score,
            'paren': paren_score,
            'efficiency': efficiency_score,
            'format': format_score,
            'repetition': repetition_score,
            'word_len': word_len_score,
            'conciseness': conciseness_score,
            'discourse': discourse_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        final_score = (weighted_sum / total_weight) * 10  # Scale to 0-10
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a neutral score
        return 5.0