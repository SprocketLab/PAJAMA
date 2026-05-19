def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    dependency analysis, discourse marker patterns, and coherence flow modeling.
    
    This variant focuses on:
    1. Sentence-level logical connective analysis (causal, conditional, contrastive chains)
    2. Argument depth estimation via clause nesting
    3. Topic consistency using sentence-to-sentence entity threading
    4. Contradiction detection via negation pattern analysis
    5. Structural completeness (intro/body/conclusion patterns)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses get low base scores
        if len(response_stripped) < 5:
            return 0.5
        
        # === Split into sentences ===
        def split_sentences(text):
            # Split on sentence-ending punctuation, but be careful with abbreviations
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            # Also split on newlines that seem to separate ideas
            expanded = []
            for s in sents:
                parts = re.split(r'\n\s*\n|\n(?=[A-Z])', s)
                expanded.extend(parts)
            return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 2]
        
        sentences = split_sentences(response_stripped)
        num_sentences = len(sentences)
        
        # === 1. LOGICAL CONNECTIVE CHAIN ANALYSIS ===
        # Instead of just counting transition words, analyze the TYPE and SEQUENCE of logical connectives
        causal_markers = r'\b(because|therefore|thus|hence|consequently|as a result|since|due to|owing to|leads to|causes|so that|for this reason)\b'
        conditional_markers = r'\b(if|unless|provided that|assuming|given that|in case|whether|when|whenever)\b'
        contrastive_markers = r'\b(however|but|although|nevertheless|on the other hand|despite|yet|whereas|conversely|in contrast|nonetheless|still|even though)\b'
        additive_markers = r'\b(furthermore|moreover|additionally|also|in addition|besides|likewise|similarly|equally|too)\b'
        elaboration_markers = r'\b(specifically|in particular|for example|for instance|such as|namely|that is|in other words|to illustrate)\b'
        conclusion_markers = r'\b(in conclusion|to summarize|overall|in summary|finally|ultimately|in short|to conclude|all in all)\b'
        
        connective_types_per_sentence = []
        for sent in sentences:
            sent_lower = sent.lower()
            types_found = set()
            if re.search(causal_markers, sent_lower):
                types_found.add('causal')
            if re.search(conditional_markers, sent_lower):
                types_found.add('conditional')
            if re.search(contrastive_markers, sent_lower):
                types_found.add('contrastive')
            if re.search(additive_markers, sent_lower):
                types_found.add('additive')
            if re.search(elaboration_markers, sent_lower):
                types_found.add('elaboration')
            if re.search(conclusion_markers, sent_lower):
                types_found.add('conclusion')
            connective_types_per_sentence.append(types_found)
        
        # Score: variety of connective types used (richer argumentation)
        all_types_used = set()
        for types in connective_types_per_sentence:
            all_types_used.update(types)
        connective_variety_score = min(len(all_types_used) / 4.0, 1.0)  # max at 4 types
        
        # Score: proportion of sentences with connectives (logical flow)
        sentences_with_connectives = sum(1 for t in connective_types_per_sentence if t)
        if num_sentences > 1:
            connective_density = sentences_with_connectives / num_sentences
            # Sweet spot: not too few, not every single sentence
            connective_density_score = min(connective_density * 2.0, 1.0) if connective_density < 0.7 else 0.9
        else:
            connective_density_score = 0.2
        
        # === 2. ARGUMENT DEPTH: Clause nesting and complexity ===
        def estimate_clause_depth(text):
            # Count subordinating conjunctions and relative pronouns as depth indicators
            subordinators = r'\b(that|which|who|whom|whose|where|when|while|although|because|since|if|unless|until|before|after|as|once|whereas)\b'
            matches = re.findall(subordinators, text.lower())
            return len(matches)
        
        total_depth = sum(estimate_clause_depth(s) for s in sentences)
        avg_depth = total_depth / max(num_sentences, 1)
        # Optimal depth: 1-3 subordinate clauses per sentence on average
        depth_score = min(avg_depth / 2.0, 1.0) if avg_depth <= 3 else max(0.5, 1.0 - (avg_depth - 3) * 0.15)
        
        # === 3. ENTITY THREADING: Topic consistency via noun phrase overlap between consecutive sentences ===
        def extract_content_words(text):
            # Extract meaningful words (nouns, verbs roughly approximated by length and not being stopwords)
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
                        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                        'just', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
                        'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
                        'them', 'their', 'what', 'which', 'who', 'whom', 'and', 'but', 'or',
                        'if', 'while', 'because', 'about', 'up', 'down', 'also', 'how'}
            words = re.findall(r'\b[a-z]+\b', text.lower())
            return set(w for w in words if w not in stopwords and len(w) > 2)
        
        if num_sentences >= 2:
            thread_scores = []
            for i in range(1, num_sentences):
                words_prev = extract_content_words(sentences[i-1])
                words_curr = extract_content_words(sentences[i])
                if words_prev and words_curr:
                    overlap = len(words_prev & words_curr)
                    union_size = len(words_prev | words_curr)
                    # Modified: use overlap ratio relative to smaller set (entity threading)
                    min_size = min(len(words_prev), len(words_curr))
                    thread_ratio = overlap / max(min_size, 1)
                    thread_scores.append(min(thread_ratio, 1.0))
                else:
                    thread_scores.append(0.0)
            
            avg_thread = sum(thread_scores) / len(thread_scores) if thread_scores else 0
            # Penalize very low threading (disconnected ideas) and very high (repetitive)
            if avg_thread < 0.05:
                threading_score = 0.1
            elif avg_thread > 0.8:
                threading_score = 0.5  # likely repetitive
            else:
                threading_score = min(avg_thread * 2.5, 1.0)
        else:
            threading_score = 0.3  # Single sentence: neutral
        
        # === 4. CONTRADICTION / INCOHERENCE DETECTION ===
        contradiction_penalty = 0.0
        
        # Check for negation flips between consecutive sentences on same topic
        negation_pattern = r'\b(not|no|never|none|nothing|neither|nor|cannot|can\'t|won\'t|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b'
        for i in range(1, num_sentences):
            words_prev = extract_content_words(sentences[i-1])
            words_curr = extract_content_words(sentences[i])
            overlap = words_prev & words_curr
            if len(overlap) >= 2:  # Same topic
                prev_neg = len(re.findall(negation_pattern, sentences[i-1].lower()))
                curr_neg = len(re.findall(negation_pattern, sentences[i].lower()))
                # If one has negation and other doesn't on same topic, potential contradiction
                if (prev_neg > 0) != (curr_neg > 0):
                    # Only penalize if the overlap is substantial (likely same claim)
                    if len(overlap) >= 3:
                        contradiction_penalty += 0.15
        
        contradiction_penalty = min(contradiction_penalty, 0.5)
        
        # Check for exact repetition of sentences (sign of incoherence/generation error)
        sentence_texts = [s.lower().strip() for s in sentences]
        unique_sentences = set(sentence_texts)
        if num_sentences > 1:
            repetition_ratio = len(unique_sentences) / num_sentences
            repetition_penalty = max(0, (1.0 - repetition_ratio) * 1.5)
        else:
            repetition_penalty = 0.0
        
        # === 5. STRUCTURAL COMPLETENESS ===
        # Does the response have a recognizable structure?
        
        # Check for opening/framing statement
        has_framing = False
        if sentences:
            first_sent_lower = sentences[0].lower()
            framing_patterns = [
                r'^(yes|no|sure|certainly|of course|absolutely)',
                r'\b(question|answer|here|let me|i\'ll|we can|to understand)\b',
                r'\b(the|a|an)\s+\w+\s+(is|are|was|were|can|has|have)\b',
            ]
            for pat in framing_patterns:
                if re.search(pat, first_sent_lower):
                    has_framing = True
                    break
        
        # Check for concluding/summarizing statement
        has_conclusion = False
        if len(sentences) >= 2:
            last_sent_lower = sentences[-1].lower()
            if re.search(conclusion_markers, last_sent_lower):
                has_conclusion = True
            # Also check for summary-like patterns
            if re.search(r'\b(overall|in short|this (means|shows|indicates|suggests))\b', last_sent_lower):
                has_conclusion = True
        
        structure_score = 0.3
        if has_framing:
            structure_score += 0.35
        if has_conclusion:
            structure_score += 0.35
        
        # === 6. RESPONSE COMPLETENESS AND RELEVANCE ===
        # Check if response seems truncated
        truncation_penalty = 0.0
        if response_stripped[-1] not in '.!?"\')]}':
            # Might be truncated
            if len(response_stripped) > 100:
                truncation_penalty = 0.1  # Minor penalty, could still be coherent
        
        # Check for garbage/code when not expected
        query_lower = query.lower()
        expects_code = any(kw in query_lower for kw in ['code', 'program', 'function', 'html', 'python', 'javascript', 'script', 'css'])
        
        code_pattern_count = len(re.findall(r'[{}<>=/;]', response_stripped))
        code_ratio = code_pattern_count / max(len(response_stripped), 1)
        
        garbage_penalty = 0.0
        if not expects_code and code_ratio > 0.1:
            garbage_penalty = min(code_ratio * 3, 0.5)
        
        # Check for off-topic rambling (response generates new questions/prompts)
        off_topic_patterns = re.findall(r'(Question:|Input:|Output:|Answer:)', response_stripped)
        if len(off_topic_patterns) > 2:
            garbage_penalty += min(len(off_topic_patterns) * 0.1, 0.4)
        
        # === 7. LENGTH APPROPRIATENESS ===
        response_words = len(response_stripped.split())
        query_words = len(query.strip().split()) if query.strip() else 5
        
        # Very short responses to substantive queries
        if response_words < 3 and query_words > 5:
            length_score = 0.1
        elif response_words < 10 and query_words > 10:
            length_score = 0.3
        elif response_words > 5:
            length_score = min(math.log(response_words + 1) / math.log(200), 1.0)
        else:
            length_score = 0.4
        
        # === 8. SENTENCE QUALITY ===
        # Average sentence length (too short = fragmented, too long = run-on)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        
        if avg_sent_len < 3:
            sent_quality = 0.2
        elif avg_sent_len < 5:
            sent_quality = 0.5
        elif avg_sent_len <= 25:
            sent_quality = 1.0
        elif avg_sent_len <= 40:
            sent_quality = 0.7
        else:
            sent_quality = 0.4
        
        # Variance in sentence length (some variety is good)
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(mean_len, 1)
            # Moderate variation is ideal
            if 0.2 <= cv <= 0.8:
                variety_bonus = 0.15
            elif cv < 0.2:
                variety_bonus = 0.0  # Too uniform
            else:
                variety_bonus = 0.05  # Too variable
        else:
            variety_bonus = 0.0
        
        # === COMBINE SCORES ===
        # Weighted combination of all components
        raw_score = (
            connective_variety_score * 1.2 +      # Logical connective variety
            connective_density_score * 1.0 +       # How well-connected sentences are
            depth_score * 1.0 +                     # Argument complexity
            threading_score * 1.5 +                 # Topic consistency
            structure_score * 0.8 +                 # Structural completeness
            length_score * 0.8 +                    # Appropriate length
            sent_quality * 0.7 +                    # Sentence quality
            variety_bonus * 1.0                     # Sentence variety
        )
        
        max_possible = 1.2 + 1.0 + 1.0 + 1.5 + 0.8 + 0.8 + 0.7 + 0.15  # = 7.15
        
        # Apply penalties
        total_penalty = contradiction_penalty + repetition_penalty + truncation_penalty + garbage_penalty
        
        # Normalize to 0-10 scale
        normalized = (raw_score / max_possible) * 10.0
        final_score = max(0.0, normalized - total_penalty * 10.0)
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, final_score))
        
        # Apply floor for responses that at least attempt to answer
        if response_words >= 10 and garbage_penalty < 0.2:
            final_score = max(final_score, 1.5)
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: simple length-based score
        try:
            words = len(response.strip().split())
            return min(max(words * 0.1, 0.5), 5.0)
        except Exception:
            return 1.0