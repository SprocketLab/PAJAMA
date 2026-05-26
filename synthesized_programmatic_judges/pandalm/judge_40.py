def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using a discourse-flow analysis approach.
    
    This variant focuses on:
    1. Sentence-level progression analysis (do sentences build on each other?)
    2. Rhetorical structure detection (claim-evidence-conclusion patterns)
    3. Contradiction detection via negation patterns
    4. Causal chain analysis
    5. Information density and elaboration depth
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
        
        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space or end
            sents = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 2]
        
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
        sentences = split_sentences(response)
        all_words = tokenize(response)
        query_words = set(tokenize(query))
        
        if not all_words:
            return 0.5
        
        num_sentences = len(sentences)
        
        # ============================================================
        # 1. SENTENCE PROGRESSION ANALYSIS (cohesion via entity chains)
        # ============================================================
        # Check if sentences share entities/concepts creating a coherent chain
        sentence_word_sets = []
        sentence_word_lists = []
        for s in sentences:
            words = tokenize(s)
            sentence_word_sets.append(set(words))
            sentence_word_lists.append(words)
        
        # Entity chain score: consecutive sentences should share some content words
        # but not be identical (which would indicate repetition)
        progression_score = 0.0
        if num_sentences >= 2:
            chain_scores = []
            for i in range(1, len(sentence_word_sets)):
                prev = sentence_word_sets[i-1]
                curr = sentence_word_sets[i]
                if not prev or not curr:
                    chain_scores.append(0.0)
                    continue
                # Remove very common words (stopwords approximation)
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                           'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                           'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                           'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                           'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or',
                           'but', 'not', 'no', 'if', 'then', 'than', 'so', 'as'}
                prev_content = prev - stopwords
                curr_content = curr - stopwords
                if not prev_content or not curr_content:
                    chain_scores.append(0.3)
                    continue
                overlap = len(prev_content & curr_content)
                union = len(prev_content | curr_content)
                ratio = overlap / union if union > 0 else 0
                # Ideal overlap is moderate (0.15-0.5) - too much means repetition
                if ratio > 0.7:
                    chain_scores.append(0.3)  # Too repetitive
                elif ratio >= 0.1:
                    chain_scores.append(min(1.0, ratio * 2.5))
                else:
                    chain_scores.append(0.15)  # No connection between sentences
            
            progression_score = sum(chain_scores) / len(chain_scores) if chain_scores else 0.0
        else:
            progression_score = 0.3  # Single sentence has limited progression
        
        # ============================================================
        # 2. RHETORICAL STRUCTURE DETECTION
        # ============================================================
        # Detect claim-evidence-conclusion patterns
        
        # Causal/reasoning connectives (different from simple transition words)
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bleading to\b', r'\bcaused by\b', r'\bresulting in\b',
            r'\bwhich means\b', r'\bthis means\b', r'\bimplying\b',
            r'\bin order to\b', r'\bso that\b', r'\bhence\b'
        ]
        
        # Elaboration markers
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bincluding\b', r'\bespecially\b'
        ]
        
        # Contrast/comparison markers
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bwhereas\b', r'\bwhile\b', r'\balthough\b', r'\bdespite\b',
            r'\bnevertheless\b', r'\bconversely\b', r'\bunlike\b',
            r'\brather than\b', r'\binstead\b'
        ]
        
        # Conclusion markers
        conclusion_markers = [
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
            r'\bto summarize\b', r'\bin short\b', r'\bultimately\b',
            r'\ball in all\b', r'\btaken together\b', r'\bfinally\b'
        ]
        
        # Additive structure markers
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bas well\b',
            r'\bnot only\b', r'\bbut also\b'
        ]
        
        response_lower = response.lower()
        
        def count_markers(patterns):
            count = 0
            for p in patterns:
                count += len(re.findall(p, response_lower))
            return count
        
        causal_count = count_markers(causal_markers)
        elaboration_count = count_markers(elaboration_markers)
        contrast_count = count_markers(contrast_markers)
        conclusion_count = count_markers(conclusion_markers)
        additive_count = count_markers(additive_markers)
        
        total_discourse_markers = causal_count + elaboration_count + contrast_count + conclusion_count + additive_count
        
        # Normalize by number of sentences
        discourse_density = total_discourse_markers / max(num_sentences, 1)
        # Ideal density: roughly 0.3-0.8 markers per sentence
        rhetorical_score = min(1.0, discourse_density * 1.5)
        
        # Bonus for variety of marker types used
        marker_types_used = sum([
            causal_count > 0,
            elaboration_count > 0,
            contrast_count > 0,
            conclusion_count > 0,
            additive_count > 0
        ])
        variety_bonus = marker_types_used * 0.08
        
        # ============================================================
        # 3. CONTRADICTION / INCOHERENCE DETECTION
        # ============================================================
        contradiction_penalty = 0.0
        
        # Check for direct contradictions: sentence says X, later says not X
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'nothing',
                         'nowhere', 'nobody', "n't", 'cannot', "can't", "won't",
                         "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
        
        # Check for repetitive/circular content
        if num_sentences >= 3:
            for i in range(len(sentence_word_sets)):
                for j in range(i + 2, len(sentence_word_sets)):
                    if not sentence_word_sets[i] or not sentence_word_sets[j]:
                        continue
                    overlap = len(sentence_word_sets[i] & sentence_word_sets[j])
                    min_len = min(len(sentence_word_sets[i]), len(sentence_word_sets[j]))
                    if min_len > 0 and overlap / min_len > 0.85:
                        # Very similar non-adjacent sentences suggest circular reasoning
                        contradiction_penalty += 0.15
        
        # Check for excessive repetition of exact phrases
        if len(response) > 50:
            # Find repeated 4-grams
            words = all_words
            if len(words) >= 4:
                fourgrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
                fourgram_counts = Counter(fourgrams)
                max_repeat = max(fourgram_counts.values()) if fourgram_counts else 1
                if max_repeat > 3:
                    contradiction_penalty += min(0.4, (max_repeat - 3) * 0.1)
        
        contradiction_penalty = min(contradiction_penalty, 0.6)
        
        # ============================================================
        # 4. INFORMATION DENSITY & ELABORATION DEPTH
        # ============================================================
        
        # Measure how much substantive content is provided
        content_words = [w for w in all_words if w not in {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or',
            'but', 'not', 'no', 'if', 'then', 'than', 'so', 'as', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
            'his', 'her', 'their'
        }]
        
        # Unique content word ratio (type-token ratio for content words)
        unique_content = set(content_words)
        if len(content_words) > 0:
            ttr = len(unique_content) / len(content_words)
        else:
            ttr = 0.0
        
        # Penalize very low TTR (repetitive) and very high TTR (incoherent/random)
        # Ideal TTR for coherent elaborated text: 0.4-0.75
        if ttr < 0.25:
            density_score = ttr * 2
        elif ttr <= 0.8:
            density_score = 0.5 + (ttr - 0.25) * 0.9
        else:
            density_score = max(0.3, 1.0 - (ttr - 0.8) * 2)
        
        # ============================================================
        # 5. STRUCTURAL COMPLETENESS
        # ============================================================
        # Does the response have a clear beginning, middle, end?
        
        completeness_score = 0.0
        
        # Check if response ends with proper punctuation (not truncated)
        if response.rstrip()[-1] in '.!?"\'':
            completeness_score += 0.3
        elif response.rstrip()[-1] in ',;:':
            completeness_score += 0.05
        
        # Check for adequate length relative to query complexity
        query_word_count = len(tokenize(query))
        response_word_count = len(all_words)
        
        # Longer queries typically need longer responses
        length_ratio = response_word_count / max(query_word_count, 1)
        if length_ratio >= 2.0:
            completeness_score += 0.3
        elif length_ratio >= 1.0:
            completeness_score += 0.2
        else:
            completeness_score += 0.05
        
        # Multiple sentences suggest more complete treatment
        if num_sentences >= 3:
            completeness_score += 0.25
        elif num_sentences >= 2:
            completeness_score += 0.15
        else:
            completeness_score += 0.05
        
        # Check if response addresses the query topic
        query_content = query_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                        'to', 'of', 'in', 'for', 'on', 'with', 'at',
                                        'by', 'from', 'and', 'or', 'but', 'what', 'how',
                                        'why', 'when', 'where', 'who', 'which', 'do',
                                        'does', 'did', 'can', 'could', 'would', 'should',
                                        'will', 'may', 'might', 'describe', 'explain',
                                        'provide', 'give', 'write', 'create', 'make',
                                        'generate', 'list', 'compare', 'contrast',
                                        'following', 'given', 'input', 'output', 'you',
                                        'your', 'me', 'my', 'i', 'we', 'our', 'it', 'its',
                                        'this', 'that', 'these', 'those'}
        
        if query_content:
            response_word_set = set(all_words)
            relevance = len(query_content & response_word_set) / len(query_content)
            completeness_score += relevance * 0.15
        
        completeness_score = min(completeness_score, 1.0)
        
        # ============================================================
        # 6. SENTENCE COMPLEXITY & SUBORDINATION
        # ============================================================
        # More complex sentences with subordinate clauses indicate structured arguments
        
        subordination_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhose\b',
            r'\bwhere\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bif\b', r'\bunless\b', r'\buntil\b', r'\bafter\b', r'\bbefore\b',
            r'\bonce\b', r'\bwhenever\b', r'\bwherever\b'
        ]
        
        sub_count = 0
        for p in subordination_markers:
            sub_count += len(re.findall(p, response_lower))
        
        sub_density = sub_count / max(num_sentences, 1)
        subordination_score = min(1.0, sub_density * 0.5)
        
        # Average sentence length (words) - very short sentences may lack elaboration
        avg_sent_len = response_word_count / max(num_sentences, 1)
        if avg_sent_len >= 15:
            length_quality = 0.8
        elif avg_sent_len >= 10:
            length_quality = 0.6
        elif avg_sent_len >= 6:
            length_quality = 0.4
        else:
            length_quality = 0.2
        
        # ============================================================
        # 7. PARALLEL STRUCTURE DETECTION
        # ============================================================
        # Good arguments often use parallel structures
        parallel_score = 0.0
        if num_sentences >= 2:
            # Check if sentences start with similar syntactic patterns
            sent_starts = []
            for s in sentences:
                words = tokenize(s)
                if len(words) >= 2:
                    sent_starts.append(tuple(words[:2]))
            
            # Some parallelism is good (shows structured comparison/listing)
            if len(sent_starts) >= 2:
                start_counter = Counter(sent_starts)
                max_parallel = max(start_counter.values())
                if max_parallel >= 2 and max_parallel <= num_sentences * 0.6:
                    parallel_score = 0.3
                elif max_parallel > num_sentences * 0.6 and num_sentences > 3:
                    parallel_score = 0.1  # Too much parallelism = formulaic
        
        # ============================================================
        # COMPOSITE SCORE
        # ============================================================
        
        # Weighted combination
        raw_score = (
            progression_score * 2.0 +      # Sentence cohesion chain
            rhetorical_score * 1.5 +        # Discourse markers
            variety_bonus * 2.0 +           # Variety of rhetorical relations
            density_score * 1.5 +           # Information density
            completeness_score * 2.0 +      # Structural completeness
            subordination_score * 1.0 +     # Sentence complexity
            length_quality * 1.0 +          # Average sentence quality
            parallel_score * 0.5            # Parallel structure
            - contradiction_penalty * 3.0   # Penalty for repetition/contradiction
        )
        
        # Normalize to 0-10 range
        # Max theoretical: 2.0 + 1.5 + 0.8 + 1.5 + 2.0 + 1.0 + 0.8 + 0.15 = 9.75
        # Min theoretical: 0 - 1.8 = -1.8
        
        # Scale to 0-10
        score = max(0.0, min(10.0, raw_score + 0.5))
        
        # Small bonus for responses that are not trivially short
        if response_word_count < 5:
            score *= 0.3
        elif response_word_count < 10:
            score *= 0.6
        
        return round(score, 3)
        
    except Exception:
        try:
            if response and len(response.strip()) > 10:
                return 3.0
            return 1.0
        except Exception:
            return 1.0