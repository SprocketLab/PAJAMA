def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    analysis: conditional/causal reasoning chains, discourse markers,
    contradiction detection, and information density progression.
    
    This variant focuses on:
    1. Sentence-level causal/logical connective chains
    2. Argument depth via nested reasoning detection
    3. Contradiction detection through negation patterns
    4. Information progression (new info per sentence)
    5. Structural completeness signals
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        query = query.strip() if query else ""
        
        # === Split into sentences ===
        def split_sentences(text):
            # Split on sentence-ending punctuation, keeping non-empty results
            parts = re.split(r'(?<=[.!?])\s+', text)
            # Also split on newlines for responses that use line breaks
            expanded = []
            for p in parts:
                subparts = re.split(r'\n+', p)
                expanded.extend(subparts)
            return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 2]
        
        sentences = split_sentences(response_stripped)
        num_sentences = len(sentences)
        
        # === 1. Causal/Logical Reasoning Chain Score ===
        # Detect causal connectives that indicate reasoning chains
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b',
            r'\bcaused by\b', r'\bresulting in\b', r'\bimplies\b',
            r'\bif\b.*\bthen\b', r'\bgiven that\b', r'\bit follows\b',
            r'\bfor this reason\b', r'\baccordingly\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bwhen\b', r'\bassuming\b', r'\bprovided\b',
            r'\bunless\b', r'\bin case\b', r'\bwhenever\b',
            r'\bsuppose\b', r'\bgiven\b'
        ]
        
        # Elaboration/explanation markers (different from simple transitions)
        elaboration_markers = [
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bincluding\b',
            r'\bto illustrate\b', r'\bto clarify\b', r'\bmeaning\b'
        ]
        
        # Concession/contrast markers (show nuanced reasoning)
        concession_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bdespite\b', r'\bnevertheless\b',
            r'\bnonetheless\b', r'\bon the other hand\b', r'\bwhile\b',
            r'\byet\b', r'\bbut\b', r'\beven though\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bregardless\b'
        ]
        
        # Conclusion/summary markers
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\bfinally\b', r'\bultimately\b',
            r'\bin short\b', r'\bto sum up\b', r'\ball in all\b'
        ]
        
        response_lower = response_stripped.lower()
        
        def count_markers(markers, text):
            count = 0
            for pattern in markers:
                count += len(re.findall(pattern, text))
            return count
        
        causal_count = count_markers(causal_markers, response_lower)
        conditional_count = count_markers(conditional_markers, response_lower)
        elaboration_count = count_markers(elaboration_markers, response_lower)
        concession_count = count_markers(concession_markers, response_lower)
        conclusion_count = count_markers(conclusion_markers, response_lower)
        
        # Weighted reasoning density
        total_reasoning_markers = (
            causal_count * 2.0 +
            conditional_count * 1.5 +
            elaboration_count * 1.5 +
            concession_count * 1.8 +
            conclusion_count * 1.2
        )
        
        # Normalize by number of sentences
        if num_sentences > 0:
            reasoning_density = total_reasoning_markers / max(num_sentences, 1)
        else:
            reasoning_density = 0
        
        # Score: 0-2.0 points, diminishing returns
        reasoning_score = min(2.0, reasoning_density * 1.5)
        
        # === 2. Argument Depth: Multi-sentence reasoning chains ===
        # Look for sequences of sentences where one builds on the previous
        chain_score = 0.0
        if num_sentences >= 2:
            chain_links = 0
            for i in range(1, num_sentences):
                sent_lower = sentences[i].lower()
                # Check if this sentence references or builds on previous
                has_causal = any(re.search(p, sent_lower) for p in causal_markers)
                has_elaboration = any(re.search(p, sent_lower) for p in elaboration_markers)
                has_concession = any(re.search(p, sent_lower) for p in concession_markers)
                
                # Check for pronoun references (anaphora = building on previous)
                has_reference = bool(re.search(
                    r'\b(this|these|that|those|it|its|they|their|such|the above|the following)\b',
                    sent_lower
                ))
                
                if has_causal or has_elaboration or has_concession or has_reference:
                    chain_links += 1
            
            chain_ratio = chain_links / (num_sentences - 1)
            chain_score = min(1.5, chain_ratio * 2.0)
        
        # === 3. Contradiction Detection ===
        # Look for internal contradictions (sentences that negate each other)
        contradiction_penalty = 0.0
        
        # Check for exact repetition of content (circular reasoning)
        if num_sentences >= 2:
            sentence_words = []
            for s in sentences:
                words = set(re.findall(r'\b\w+\b', s.lower()))
                words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                          'to', 'of', 'in', 'for', 'and', 'or', 'but', 'with', 'at', 'by'}
                sentence_words.append(words)
            
            # Check for near-duplicate sentences (circular reasoning)
            duplicate_count = 0
            for i in range(len(sentence_words)):
                for j in range(i + 1, len(sentence_words)):
                    if len(sentence_words[i]) > 3 and len(sentence_words[j]) > 3:
                        intersection = sentence_words[i] & sentence_words[j]
                        union = sentence_words[i] | sentence_words[j]
                        if union:
                            sim = len(intersection) / len(union)
                            if sim > 0.8:
                                duplicate_count += 1
            
            if num_sentences > 0:
                duplicate_ratio = duplicate_count / max(num_sentences, 1)
                contradiction_penalty = min(2.0, duplicate_ratio * 3.0)
        
        # === 4. Information Progression ===
        # Measure how much new information each sentence adds
        progression_score = 0.0
        if num_sentences >= 2:
            cumulative_words = set()
            new_info_ratios = []
            for s in sentences:
                words = set(re.findall(r'\b\w+\b', s.lower()))
                content_words = words - {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                         'be', 'been', 'to', 'of', 'in', 'for', 'and',
                                         'or', 'but', 'with', 'at', 'by', 'on', 'it',
                                         'that', 'this', 'i', 'you', 'he', 'she', 'we',
                                         'they', 'my', 'your', 'his', 'her', 'our', 'their'}
                if cumulative_words and content_words:
                    new_words = content_words - cumulative_words
                    ratio = len(new_words) / max(len(content_words), 1)
                    new_info_ratios.append(ratio)
                cumulative_words |= content_words
            
            if new_info_ratios:
                avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
                # Good progression: each sentence adds ~40-80% new info
                # Too low = repetitive, too high might be incoherent
                if avg_new_info > 0.3:
                    progression_score = min(1.5, avg_new_info * 2.0)
                else:
                    progression_score = avg_new_info * 1.0
        
        # === 5. Structural Completeness ===
        # Does the response have a beginning, middle, end structure?
        completeness_score = 0.0
        
        # Check if response ends with proper punctuation (not truncated)
        if response_stripped[-1] in '.!?)':
            completeness_score += 0.3
        
        # Check for complete sentences (subject-verb patterns)
        complete_sentence_count = 0
        for s in sentences:
            # Very rough check: has a verb-like word
            if re.search(r'\b(is|are|was|were|has|have|had|can|could|will|would|shall|should|may|might|do|does|did|being|been)\b', s.lower()):
                complete_sentence_count += 1
            elif re.search(r'\b\w+(?:s|ed|ing|es)\b', s.lower()):
                complete_sentence_count += 1
        
        if num_sentences > 0:
            complete_ratio = complete_sentence_count / num_sentences
            completeness_score += complete_ratio * 0.7
        
        # === 6. Response Substance ===
        # Penalize extremely short or empty responses
        word_count = len(re.findall(r'\b\w+\b', response_stripped))
        
        substance_score = 0.0
        if word_count <= 3:
            substance_score = 0.0
        elif word_count <= 10:
            substance_score = 0.5
        elif word_count <= 30:
            substance_score = 1.0
        elif word_count <= 100:
            substance_score = 1.5
        else:
            substance_score = 1.8
        
        # === 7. Relevance to Query ===
        # Check if response addresses the query
        query_words = set(re.findall(r'\b\w+\b', query.lower())) if query else set()
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'to', 'of', 'in', 'for', 'and', 'or', 'but', 'with', 'at', 'by',
                      'on', 'it', 'that', 'this', 'i', 'you', 'he', 'she', 'we', 'they',
                      'my', 'your', 'his', 'her', 'our', 'their', 'what', 'how', 'can',
                      'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        relevance_score = 0.0
        if query_content and response_content:
            overlap = query_content & response_content
            relevance_ratio = len(overlap) / max(len(query_content), 1)
            relevance_score = min(1.0, relevance_ratio * 1.2)
        
        # === 8. Noise/Garbage Detection ===
        noise_penalty = 0.0
        
        # Check for HTML tags (usually noise)
        html_tags = len(re.findall(r'<[^>]+>', response_stripped))
        if html_tags > 2:
            noise_penalty += min(1.5, html_tags * 0.3)
        
        # Check for code-like content when not expected
        code_indicators = len(re.findall(r'(?:import |def |class |print\(|return |\{|\}|;$)', response_stripped))
        if code_indicators > 3 and 'code' not in query.lower() and 'program' not in query.lower():
            noise_penalty += min(1.0, code_indicators * 0.2)
        
        # Check for excessive repetition of phrases
        words_list = re.findall(r'\b\w+\b', response_lower)
        if len(words_list) > 10:
            # Check trigram repetition
            trigrams = [' '.join(words_list[i:i+3]) for i in range(len(words_list) - 2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                most_common_count = trigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / max(len(trigrams), 1)
                if repetition_ratio > 0.1:
                    noise_penalty += min(1.5, repetition_ratio * 10)
        
        # Check for "Output:" repetition pattern (like in examples)
        output_pattern_count = len(re.findall(r'(?:Output:|Input:|Question:|Answer:)', response_stripped))
        if output_pattern_count > 3:
            noise_penalty += min(1.0, (output_pattern_count - 2) * 0.3)
        
        # === 9. Sentence Coherence via Topic Continuity ===
        # Check if adjacent sentences share content words (topic continuity)
        coherence_score = 0.0
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, min(num_sentences, 20)):  # Limit for performance
                prev_words = set(re.findall(r'\b\w{3,}\b', sentences[i-1].lower())) - stop_words
                curr_words = set(re.findall(r'\b\w{3,}\b', sentences[i].lower())) - stop_words
                if prev_words and curr_words:
                    shared = prev_words & curr_words
                    continuity = len(shared) / min(len(prev_words), len(curr_words))
                    continuity_scores.append(continuity)
            
            if continuity_scores:
                avg_continuity = sum(continuity_scores) / len(continuity_scores)
                # Moderate continuity is good (0.1-0.5), too high = repetitive, too low = incoherent
                if 0.05 <= avg_continuity <= 0.6:
                    coherence_score = min(1.0, avg_continuity * 2.5)
                elif avg_continuity > 0.6:
                    coherence_score = max(0.2, 1.0 - (avg_continuity - 0.6) * 2)
                else:
                    coherence_score = avg_continuity * 2.0
        
        # === Combine all scores ===
        raw_score = (
            reasoning_score * 1.0 +       # 0-2.0
            chain_score * 1.0 +            # 0-1.5
            progression_score * 1.0 +      # 0-1.5
            completeness_score * 1.0 +     # 0-1.0
            substance_score * 1.0 +        # 0-1.8
            relevance_score * 1.0 +        # 0-1.0
            coherence_score * 1.0 -        # 0-1.0
            contradiction_penalty * 1.0 -  # 0-2.0
            noise_penalty * 1.0            # 0-4.0
        )
        
        # Scale to 0-10 range
        # Max possible ≈ 2+1.5+1.5+1+1.8+1+1 = 9.8
        # Min possible ≈ 0 - 2 - 4 = -6
        
        # Normalize
        final_score = max(0.0, min(10.0, raw_score + 1.0))  # shift up slightly
        
        # Round to 1 decimal
        final_score = round(final_score, 1)
        
        return final_score
        
    except Exception as e:
        # Fallback: return a middle score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except:
            return 0.0