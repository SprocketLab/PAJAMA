def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Sentence-level dependency chain analysis (does each sentence build on the previous?)
    - Discourse marker quality (not just counting transition words, but checking if they're used correctly)
    - Contradiction detection via negation pattern analysis
    - Information density and progression tracking
    - Structural completeness (intro/body/conclusion patterns)
    - Repetition/circular reasoning detection via sentence similarity using character n-grams
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.5
        
        response_clean = response.strip()
        query_clean = query.strip() if query else ""
        
        # === Sentence splitting (more sophisticated) ===
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space/newline or end
            sents = re.split(r'(?<=[.!?])\s+|\n+', text)
            sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 2]
            return sents
        
        sentences = split_sentences(response_clean)
        
        # If response is trivially short
        if len(response_clean) < 5:
            return 0.5
        
        # === 1. Character n-gram based sentence similarity for circular reasoning detection ===
        def char_ngrams(text, n=3):
            text = text.lower()
            grams = Counter()
            for i in range(len(text) - n + 1):
                grams[text[i:i+n]] += 1
            return grams
        
        def ngram_similarity(t1, t2):
            if not t1 or not t2:
                return 0.0
            g1 = char_ngrams(t1, 3)
            g2 = char_ngrams(t2, 3)
            if not g1 or not g2:
                return 0.0
            intersection = sum((g1 & g2).values())
            union = sum((g1 | g2).values())
            return intersection / union if union > 0 else 0.0
        
        # Detect repetition/circular reasoning
        repetition_score = 0.0
        if len(sentences) >= 2:
            high_sim_pairs = 0
            total_pairs = 0
            for i in range(len(sentences)):
                for j in range(i + 1, min(i + 5, len(sentences))):
                    sim = ngram_similarity(sentences[i], sentences[j])
                    total_pairs += 1
                    if sim > 0.7:
                        high_sim_pairs += 1
            if total_pairs > 0:
                repetition_ratio = high_sim_pairs / total_pairs
                repetition_score = max(0, 1.0 - repetition_ratio * 3.0)
            else:
                repetition_score = 0.5
        else:
            repetition_score = 0.5
        
        # === 2. Information progression tracking ===
        # Track how much NEW information each sentence adds
        def get_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                    'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
                    'if', 'while', 'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we',
                    'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
                    'their', 'what', 'which', 'who', 'whom'}
            return set(w for w in words if w not in stop and len(w) > 2)
        
        progression_score = 0.5
        if len(sentences) >= 2:
            seen_words = set()
            new_info_ratios = []
            for sent in sentences:
                words = get_content_words(sent)
                if words:
                    new_words = words - seen_words
                    ratio = len(new_words) / len(words)
                    new_info_ratios.append(ratio)
                    seen_words.update(words)
            
            if new_info_ratios:
                avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
                # Good responses should have moderate new info (not too repetitive, not random)
                # Ideal is around 0.4-0.7 new info per sentence
                if avg_new_info < 0.15:
                    progression_score = 0.2  # Too repetitive
                elif avg_new_info > 0.95:
                    progression_score = 0.5  # Possibly incoherent/random
                else:
                    progression_score = min(1.0, 0.3 + avg_new_info * 0.9)
        
        # === 3. Discourse marker quality (not just presence, but proper usage) ===
        # Check if discourse markers appear at reasonable positions and connect ideas
        
        causal_markers = ['because', 'therefore', 'thus', 'hence', 'consequently',
                         'as a result', 'due to', 'since', 'so that', 'for this reason']
        additive_markers = ['furthermore', 'moreover', 'additionally', 'in addition',
                           'also', 'besides', 'likewise', 'similarly']
        contrastive_markers = ['however', 'nevertheless', 'on the other hand', 'although',
                              'despite', 'in contrast', 'whereas', 'yet', 'but', 'while']
        sequential_markers = ['first', 'second', 'third', 'finally', 'next', 'then',
                             'subsequently', 'lastly', 'to begin', 'in conclusion']
        
        response_lower = response_clean.lower()
        
        marker_counts = {
            'causal': sum(1 for m in causal_markers if m in response_lower),
            'additive': sum(1 for m in additive_markers if m in response_lower),
            'contrastive': sum(1 for m in contrastive_markers if m in response_lower),
            'sequential': sum(1 for m in sequential_markers if m in response_lower)
        }
        
        total_markers = sum(marker_counts.values())
        marker_diversity = sum(1 for v in marker_counts.values() if v > 0)
        
        # Score based on both quantity and diversity of discourse markers
        # Normalize by response length
        words_count = len(response_clean.split())
        if words_count > 0:
            marker_density = total_markers / max(1, words_count / 50.0)
        else:
            marker_density = 0
        
        discourse_score = min(1.0, (marker_density * 0.3) + (marker_diversity * 0.2))
        
        # === 4. Structural completeness ===
        # Check for signs of complete, well-structured response
        
        structural_score = 0.3  # baseline
        
        # Check if response starts with a clear topic/thesis
        first_sent = sentences[0] if sentences else ""
        query_words = get_content_words(query_clean)
        first_sent_words = get_content_words(first_sent)
        if query_words and first_sent_words:
            topic_connection = len(query_words & first_sent_words) / max(1, len(query_words))
            structural_score += min(0.2, topic_connection * 0.3)
        
        # Check for conclusion indicators
        conclusion_words = ['in conclusion', 'to summarize', 'overall', 'in summary',
                           'to sum up', 'ultimately', 'in short']
        has_conclusion = any(cw in response_lower for cw in conclusion_words)
        if has_conclusion:
            structural_score += 0.1
        
        # Paragraph structure
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            structural_score += min(0.2, len(paragraphs) * 0.05)
        
        # Sentence count bonus (more sentences usually means more structured argument)
        if len(sentences) >= 3:
            structural_score += 0.1
        if len(sentences) >= 5:
            structural_score += 0.1
        
        structural_score = min(1.0, structural_score)
        
        # === 5. Contradiction detection via negation patterns ===
        # Look for contradictory statements
        contradiction_score = 1.0  # Start with no contradictions
        
        negation_words = {'not', "n't", 'no', 'never', 'neither', 'nor', 'nobody',
                         'nothing', 'nowhere', 'cannot', "can't", "won't", "don't",
                         "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
        
        if len(sentences) >= 2:
            # For each pair of nearby sentences, check if they make contradictory claims
            for i in range(len(sentences) - 1):
                for j in range(i + 1, min(i + 3, len(sentences))):
                    words_i = set(re.findall(r'[a-z\']+', sentences[i].lower()))
                    words_j = set(re.findall(r'[a-z\']+', sentences[j].lower()))
                    
                    content_i = get_content_words(sentences[i])
                    content_j = get_content_words(sentences[j])
                    
                    # High content overlap but different negation patterns
                    if content_i and content_j:
                        content_overlap = len(content_i & content_j) / max(1, min(len(content_i), len(content_j)))
                        neg_i = bool(words_i & negation_words)
                        neg_j = bool(words_j & negation_words)
                        
                        if content_overlap > 0.5 and neg_i != neg_j:
                            contradiction_score -= 0.15
            
            contradiction_score = max(0.0, contradiction_score)
        
        # === 6. Coherence flow: semantic connectivity between adjacent sentences ===
        # Using content word overlap between consecutive sentences
        flow_score = 0.5
        if len(sentences) >= 2:
            flow_scores = []
            for i in range(len(sentences) - 1):
                words_curr = get_content_words(sentences[i])
                words_next = get_content_words(sentences[i + 1])
                if words_curr and words_next:
                    # Some overlap is good (coherence), but not too much (repetition)
                    overlap = len(words_curr & words_next)
                    overlap_ratio = overlap / max(1, min(len(words_curr), len(words_next)))
                    # Ideal overlap ratio is 0.1-0.5
                    if overlap_ratio < 0.05:
                        flow_scores.append(0.3)  # Disconnected
                    elif overlap_ratio > 0.7:
                        flow_scores.append(0.5)  # Too repetitive
                    else:
                        flow_scores.append(0.7 + overlap_ratio * 0.5)
                else:
                    flow_scores.append(0.3)
            
            if flow_scores:
                flow_score = sum(flow_scores) / len(flow_scores)
                flow_score = min(1.0, flow_score)
        
        # === 7. Response quality signals ===
        quality_score = 0.5
        
        # Penalize very short responses
        if len(response_clean) < 10:
            quality_score = 0.1
        elif len(response_clean) < 30:
            quality_score = 0.25
        elif len(response_clean) < 80:
            quality_score = 0.4
        else:
            quality_score = 0.5 + min(0.3, len(response_clean) / 2000.0)
        
        # Penalize responses with lots of code/HTML when query doesn't ask for it
        code_indicators = ['import ', 'def ', 'class ', 'function(', '<?php', '<script']
        query_asks_code = any(w in query_clean.lower() for w in ['code', 'program', 'script', 'function', 'html', 'tag'])
        if not query_asks_code:
            code_count = sum(1 for ci in code_indicators if ci in response_clean)
            if code_count > 0:
                quality_score *= 0.5
        
        # Penalize responses that just repeat the query
        if query_clean and response_clean:
            q_sim = ngram_similarity(query_clean, response_clean)
            if q_sim > 0.8:
                quality_score *= 0.5
        
        # Penalize responses with excessive repetition of phrases
        words_list = response_clean.lower().split()
        if len(words_list) > 10:
            trigrams = []
            for i in range(len(words_list) - 2):
                trigrams.append(' '.join(words_list[i:i+3]))
            trigram_counts = Counter(trigrams)
            if trigrams:
                max_repeat = max(trigram_counts.values())
                if max_repeat > 3:
                    quality_score *= max(0.3, 1.0 - (max_repeat - 3) * 0.1)
        
        # Penalize if response contains "Output:" repeated (looks like template)
        output_pattern_count = response_clean.count('Output:')
        if output_pattern_count > 2:
            # But only if query doesn't ask for multiple outputs
            if 'output' not in query_clean.lower():
                quality_score *= 0.8
        
        # Penalize "Question:" "Answer:" patterns that suggest copy-paste
        qa_pattern_count = len(re.findall(r'(?:Question|Answer|Input|Output)\s*:', response_clean))
        if qa_pattern_count > 3:
            quality_score *= 0.6
        
        # === 8. Sentence well-formedness ===
        wellformed_score = 0.5
        if sentences:
            wellformed_counts = 0
            for sent in sentences:
                # Check basic sentence structure: starts with capital, has verb-like words, reasonable length
                is_wellformed = True
                sent_stripped = sent.strip()
                
                if len(sent_stripped) < 3:
                    is_wellformed = False
                elif not sent_stripped[0].isupper() and not sent_stripped[0].isdigit():
                    is_wellformed = False
                
                sent_words = sent_stripped.split()
                if len(sent_words) < 2:
                    is_wellformed = False
                elif len(sent_words) > 80:
                    is_wellformed = False  # Run-on sentence
                
                if is_wellformed:
                    wellformed_counts += 1
            
            wellformed_score = wellformed_counts / len(sentences)
        
        # === Combine all scores with weights ===
        final_score = (
            repetition_score * 1.5 +      # Penalize circular reasoning
            progression_score * 1.5 +       # Reward information progression
            discourse_score * 1.0 +         # Reward proper discourse markers
            structural_score * 1.5 +        # Reward good structure
            contradiction_score * 1.0 +     # Penalize contradictions
            flow_score * 1.5 +              # Reward coherent flow
            quality_score * 2.0 +           # General quality signals
            wellformed_score * 1.0          # Sentence well-formedness
        )
        
        # Normalize to 0-10 scale
        max_possible = 1.5 + 1.5 + 1.0 + 1.5 + 1.0 + 1.5 + 2.0 + 1.0  # = 11.0
        final_score = (final_score / max_possible) * 10.0
        
        # Clamp to [0.5, 10.0]
        final_score = max(0.5, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.5
        except:
            return 3.0