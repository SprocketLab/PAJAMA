def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Sentence-level dependency chain analysis (does each sentence build on previous?)
    - Causal/logical connector density and proper usage
    - Contradiction detection via negation pattern analysis
    - Argument depth (premise -> elaboration -> conclusion pattern)
    - Repetition/circular reasoning detection via sentence similarity
    - Progressive information flow (new information introduced per sentence)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 3:
            return 0.5
        
        # Split into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation
            sents = re.split(r'(?<=[.!?])\s+', text)
            sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 2]
            return sents
        
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
        sentences = split_sentences(response)
        all_tokens = tokenize(response)
        query_tokens = set(tokenize(query))
        
        if not all_tokens:
            return 0.5
        
        num_sentences = len(sentences)
        
        # ============================================================
        # 1. CAUSAL/LOGICAL CONNECTOR ANALYSIS (not just transition words)
        # ============================================================
        # Categorize connectors by logical function
        causal_connectors = ['because', 'since', 'therefore', 'thus', 'hence', 
                            'consequently', 'as a result', 'due to', 'owing to',
                            'leads to', 'causes', 'results in', 'so that']
        contrastive_connectors = ['however', 'although', 'despite', 'nevertheless',
                                  'on the other hand', 'in contrast', 'whereas',
                                  'while', 'but', 'yet', 'conversely', 'unlike']
        additive_connectors = ['furthermore', 'moreover', 'additionally', 'also',
                              'in addition', 'besides', 'as well as']
        temporal_connectors = ['then', 'next', 'subsequently', 'afterwards',
                              'finally', 'first', 'second', 'third', 'meanwhile',
                              'before', 'after', 'when', 'once']
        exemplification = ['for example', 'for instance', 'such as', 'including',
                          'specifically', 'in particular', 'namely']
        conclusive = ['in conclusion', 'to summarize', 'overall', 'in summary',
                     'ultimately', 'in short']
        
        response_lower = response.lower()
        
        connector_counts = {}
        for category, connectors in [('causal', causal_connectors), 
                                      ('contrastive', contrastive_connectors),
                                      ('additive', additive_connectors),
                                      ('temporal', temporal_connectors),
                                      ('exemplification', exemplification),
                                      ('conclusive', conclusive)]:
            count = 0
            for conn in connectors:
                count += len(re.findall(r'\b' + re.escape(conn) + r'\b', response_lower))
            connector_counts[category] = count
        
        total_connectors = sum(connector_counts.values())
        # Variety of connector types used
        connector_type_variety = sum(1 for v in connector_counts.values() if v > 0)
        
        # Score: reward variety of connector types (max 6 types)
        connector_variety_score = min(connector_type_variety / 3.0, 1.0) * 10
        
        # Connector density (connectors per sentence, sweet spot around 0.3-0.8)
        if num_sentences > 0:
            connector_density = total_connectors / num_sentences
            if connector_density < 0.1:
                density_score = connector_density * 30  # low density penalized
            elif connector_density <= 1.0:
                density_score = 3.0 + connector_density * 4
            else:
                density_score = max(0, 7.0 - (connector_density - 1.0) * 2)  # over-use penalized
        else:
            density_score = 0
        
        # ============================================================
        # 2. PROGRESSIVE INFORMATION FLOW (each sentence adds new info)
        # ============================================================
        sentence_token_sets = []
        for s in sentences:
            tokens = set(tokenize(s)) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                                          'it', 'this', 'that', 'of', 'to', 'in', 'and',
                                          'for', 'on', 'with', 'as', 'at', 'by', 'from'}
            sentence_token_sets.append(tokens)
        
        new_info_ratios = []
        accumulated = set()
        for i, token_set in enumerate(sentence_token_sets):
            if i == 0:
                # First sentence: measure info relative to query
                new_tokens = token_set - query_tokens
                ratio = len(new_tokens) / max(len(token_set), 1)
                accumulated = token_set.copy()
            else:
                new_tokens = token_set - accumulated
                ratio = len(new_tokens) / max(len(token_set), 1)
                accumulated |= token_set
            new_info_ratios.append(ratio)
        
        if new_info_ratios:
            avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
            # Penalize if later sentences add nothing new (circular reasoning)
            if len(new_info_ratios) > 2:
                late_info = sum(new_info_ratios[len(new_info_ratios)//2:]) / max(len(new_info_ratios[len(new_info_ratios)//2:]), 1)
            else:
                late_info = avg_new_info
        else:
            avg_new_info = 0
            late_info = 0
        
        info_flow_score = (avg_new_info * 6 + late_info * 4)  # max ~10
        info_flow_score = min(info_flow_score, 10)
        
        # ============================================================
        # 3. CIRCULAR REASONING / REPETITION DETECTION
        # ============================================================
        # Check pairwise sentence similarity (high similarity = repetitive/circular)
        max_similarity = 0
        avg_similarity = 0
        pair_count = 0
        
        if num_sentences >= 2:
            similarities = []
            for i in range(len(sentence_token_sets)):
                for j in range(i + 1, len(sentence_token_sets)):
                    s1, s2 = sentence_token_sets[i], sentence_token_sets[j]
                    if len(s1) == 0 and len(s2) == 0:
                        sim = 1.0
                    elif len(s1 | s2) == 0:
                        sim = 0
                    else:
                        # Cosine-like overlap
                        intersection = len(s1 & s2)
                        sim = intersection / math.sqrt(max(len(s1), 1) * max(len(s2), 1))
                    similarities.append(sim)
            
            if similarities:
                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
        
        # High similarity is bad (circular reasoning)
        circular_penalty = max_similarity * 5 + avg_similarity * 5  # 0-10 penalty
        non_circular_score = max(0, 10 - circular_penalty)
        
        # ============================================================
        # 4. ARGUMENT DEPTH: Premise-Elaboration-Conclusion pattern
        # ============================================================
        # Check if response has a claim/thesis, supporting details, and conclusion
        has_opening_claim = False
        has_elaboration = False
        has_conclusion = False
        
        if num_sentences >= 1:
            first_sent_lower = sentences[0].lower()
            # Opening claims often define or state something
            claim_patterns = [r'\bis\b', r'\bare\b', r'\bmeans\b', r'\brefers to\b',
                            r'\bcan be\b', r'\binvolves\b', r'\bdefine', r'\brepresent']
            for p in claim_patterns:
                if re.search(p, first_sent_lower):
                    has_opening_claim = True
                    break
        
        if num_sentences >= 3:
            # Middle sentences should elaborate
            middle_sents = ' '.join(sentences[1:-1]).lower()
            elaboration_signals = ['for example', 'such as', 'this means', 'specifically',
                                  'in other words', 'because', 'which', 'where', 'when',
                                  'including', 'allows', 'enables', 'provides', 'helps']
            for sig in elaboration_signals:
                if sig in middle_sents:
                    has_elaboration = True
                    break
            
            # Last sentence might conclude
            last_sent_lower = sentences[-1].lower()
            conclusion_signals = ['therefore', 'thus', 'overall', 'in conclusion',
                                 'in summary', 'important', 'ultimately', 'this is why',
                                 'as a result', 'essential']
            for sig in conclusion_signals:
                if sig in last_sent_lower:
                    has_conclusion = True
                    break
        
        argument_structure_score = 0
        if has_opening_claim:
            argument_structure_score += 4
        if has_elaboration:
            argument_structure_score += 3
        if has_conclusion:
            argument_structure_score += 3
        
        # ============================================================
        # 5. INTERNAL CONTRADICTION DETECTION
        # ============================================================
        # Look for sentences that negate what a previous sentence stated
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', "n't", 'cannot', 'without'}
        
        contradiction_score = 0
        sent_polarities = []
        for s in sentences:
            tokens = tokenize(s)
            neg_count = sum(1 for t in tokens if t in negation_words)
            # Also check for "n't" pattern
            neg_count += len(re.findall(r"n't", s.lower()))
            sent_polarities.append(neg_count % 2)  # 0 = positive, 1 = negative
        
        # Check if sentences about similar topics have conflicting polarities
        contradiction_found = False
        for i in range(len(sentence_token_sets)):
            for j in range(i + 1, len(sentence_token_sets)):
                if sentence_token_sets[i] and sentence_token_sets[j]:
                    overlap = len(sentence_token_sets[i] & sentence_token_sets[j])
                    overlap_ratio = overlap / min(len(sentence_token_sets[i]), len(sentence_token_sets[j]))
                    if overlap_ratio > 0.5 and sent_polarities[i] != sent_polarities[j]:
                        contradiction_found = True
                        break
            if contradiction_found:
                break
        
        contradiction_penalty = 5 if contradiction_found else 0
        
        # ============================================================
        # 6. RESPONSE COMPLETENESS & STRUCTURAL QUALITY
        # ============================================================
        # Sentence count reward (more sentences = more developed argument, up to a point)
        if num_sentences == 0:
            completeness_score = 0
        elif num_sentences == 1:
            completeness_score = 2
        elif num_sentences == 2:
            completeness_score = 4
        elif num_sentences <= 5:
            completeness_score = 6 + (num_sentences - 3) * 1
        elif num_sentences <= 10:
            completeness_score = 8
        else:
            completeness_score = 8 + min(2, (num_sentences - 10) * 0.1)
        completeness_score = min(completeness_score, 10)
        
        # Average sentence length (too short = underdeveloped, too long = rambling)
        if num_sentences > 0:
            avg_sent_len = len(all_tokens) / num_sentences
            if avg_sent_len < 5:
                sent_len_score = avg_sent_len / 5 * 5
            elif avg_sent_len <= 25:
                sent_len_score = 5 + (avg_sent_len - 5) / 20 * 5
            else:
                sent_len_score = max(3, 10 - (avg_sent_len - 25) / 10 * 3)
        else:
            sent_len_score = 0
        
        # ============================================================
        # 7. QUERY RELEVANCE (logical coherence includes staying on topic)
        # ============================================================
        response_token_set = set(all_tokens)
        content_query_tokens = query_tokens - {'the', 'a', 'an', 'is', 'are', 'of', 'to', 
                                                'in', 'and', 'for', 'on', 'with', 'what',
                                                'how', 'why', 'when', 'where', 'which', 'do',
                                                'does', 'did', 'can', 'could', 'would', 'should'}
        if content_query_tokens:
            relevance = len(content_query_tokens & response_token_set) / len(content_query_tokens)
        else:
            relevance = 0.5
        
        relevance_score = relevance * 10
        
        # ============================================================
        # 8. DEGENERATE RESPONSE DETECTION
        # ============================================================
        # Check for excessive repetition of words
        if all_tokens:
            token_freq = Counter(all_tokens)
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'this', 
                         'that', 'of', 'to', 'in', 'and', 'for', 'on', 'with', 'as',
                         'at', 'by', 'from', 'or', 'be', 'their', 'they', 'them'}
            content_tokens = [t for t in all_tokens if t not in stop_words]
            if content_tokens:
                content_freq = Counter(content_tokens)
                most_common_freq = content_freq.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(content_tokens)
                if repetition_ratio > 0.3:
                    degenerate_penalty = min(15, (repetition_ratio - 0.3) * 50)
                else:
                    degenerate_penalty = 0
            else:
                degenerate_penalty = 5
        else:
            degenerate_penalty = 10
        
        # Check for near-identical consecutive substrings (stuttering/looping)
        if len(response) > 20:
            # Check if large chunks repeat
            half = len(response) // 2
            first_half = response[:half]
            second_half = response[half:half + len(first_half)]
            if first_half == second_half:
                degenerate_penalty += 15
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        # Weights emphasize logical structure over surface features
        raw_score = (
            connector_variety_score * 1.2 +    # max ~12
            density_score * 1.0 +               # max ~7
            info_flow_score * 1.5 +             # max ~15
            non_circular_score * 1.2 +          # max ~12
            argument_structure_score * 1.0 +    # max ~10
            completeness_score * 1.0 +          # max ~10
            sent_len_score * 0.5 +              # max ~5
            relevance_score * 1.0 -             # max ~10
            contradiction_penalty * 1.0 -       # max penalty ~5
            degenerate_penalty * 1.5            # max penalty ~22.5
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~81, practical max ~70
        final_score = max(0, min(100, raw_score * 100 / 75))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            if response and len(response.strip()) > 10:
                return 25.0
            return 5.0
        except Exception:
            return 5.0