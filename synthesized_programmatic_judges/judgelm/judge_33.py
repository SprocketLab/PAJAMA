def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    Returns a score where HIGHER = BETTER quality.
    
    This variant focuses on:
    - Sentence-level logical flow and connectivity
    - Argument structure indicators (premises, conclusions, transitions)
    - Internal consistency (no contradictions or circular reasoning)
    - Coherent paragraph structure
    - Absence of non-sequiturs and irrelevant content
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        if len(response_stripped) == 0:
            return 0.0
        
        # ==========================================
        # FEATURE 1: Response substantiveness
        # ==========================================
        words = response_stripped.split()
        word_count = len(words)
        
        if word_count <= 1:
            return 0.5
        
        # Sentences
        sentence_endings = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 2]
        sentence_count = max(len(sentences), 1)
        
        # Base length score: reward moderate length, diminishing returns
        if word_count < 3:
            length_score = 0.5
        elif word_count < 10:
            length_score = 2.0
        elif word_count < 30:
            length_score = 4.0
        elif word_count < 80:
            length_score = 5.5
        elif word_count < 200:
            length_score = 6.0
        else:
            length_score = 5.5  # slight penalty for very long
        
        # ==========================================
        # FEATURE 2: Logical connectors and transitions
        # ==========================================
        response_lower = response_stripped.lower()
        
        # Causal/logical connectors
        causal_connectors = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bleading to\b', r'\bresulting in\b', r'\bcaused by\b'
        ]
        
        # Transitional phrases
        transition_connectors = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bon the other hand\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bmeanwhile\b', r'\bsimilarly\b', r'\bin contrast\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bthat is\b', r'\bin other words\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\badditionally\b', r'\balso\b', r'\bwhile\b', r'\balthough\b',
            r'\bdespite\b', r'\byet\b'
        ]
        
        # Conclusion indicators
        conclusion_markers = [
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
            r'\bto summarize\b', r'\bin short\b', r'\bultimately\b',
            r'\btherefore\b', r'\bthus\b', r'\ball in all\b'
        ]
        
        causal_count = sum(1 for p in causal_connectors if re.search(p, response_lower))
        transition_count = sum(1 for p in transition_connectors if re.search(p, response_lower))
        conclusion_count = sum(1 for p in conclusion_markers if re.search(p, response_lower))
        
        # Normalize by word count to avoid pure length bias
        connector_density = (causal_count + transition_count) / max(word_count, 1) * 100
        
        # Score: reward having connectors but not excessively
        connector_raw = causal_count * 1.5 + transition_count * 1.0 + conclusion_count * 1.0
        connector_score = min(connector_raw, 8.0)  # cap
        
        # ==========================================
        # FEATURE 3: Sentence structure quality
        # ==========================================
        
        # Average sentence length (too short = fragmented, too long = run-on)
        if sentence_count > 0:
            avg_sent_len = word_count / sentence_count
        else:
            avg_sent_len = word_count
        
        # Ideal average sentence length is 10-25 words
        if 10 <= avg_sent_len <= 25:
            sent_len_score = 3.0
        elif 6 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
            sent_len_score = 2.0
        elif 3 <= avg_sent_len < 6 or 35 < avg_sent_len <= 50:
            sent_len_score = 1.0
        else:
            sent_len_score = 0.5
        
        # Sentence length variance (some variation is good for flow)
        if len(sentences) >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variation is good
            cv = std_dev / max(mean_sl, 1)
            if 0.2 <= cv <= 0.8:
                variety_score = 2.0
            elif 0.1 <= cv < 0.2 or 0.8 < cv <= 1.2:
                variety_score = 1.0
            else:
                variety_score = 0.5
        else:
            variety_score = 0.5
        
        # ==========================================
        # FEATURE 4: Vocabulary richness & coherence
        # ==========================================
        words_lower = [w.lower().strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        words_clean = [w for w in words_lower if w]
        
        if len(words_clean) > 3:
            unique_ratio = len(set(words_clean)) / len(words_clean)
        else:
            unique_ratio = 0.5
        
        # Very low unique ratio suggests repetition (bad coherence)
        if unique_ratio >= 0.6:
            vocab_score = 3.0
        elif unique_ratio >= 0.4:
            vocab_score = 2.0
        elif unique_ratio >= 0.25:
            vocab_score = 1.0
        else:
            vocab_score = 0.0
        
        # ==========================================
        # FEATURE 5: Repetition detection (penalize)
        # ==========================================
        # Check for repeated sentences or phrases
        repetition_penalty = 0.0
        
        if len(sentences) >= 2:
            sentence_set = set()
            duplicate_count = 0
            for s in sentences:
                s_normalized = ' '.join(s.lower().split())
                if s_normalized in sentence_set:
                    duplicate_count += 1
                sentence_set.add(s_normalized)
            
            if duplicate_count > 0:
                repetition_penalty += min(duplicate_count * 2.0, 6.0)
        
        # Check for repeated n-grams (3-grams)
        if len(words_clean) >= 6:
            trigrams = [' '.join(words_clean[i:i+3]) for i in range(len(words_clean) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            repetition_penalty += min(repeated_trigrams * 0.5, 3.0)
        
        # ==========================================
        # FEATURE 6: Relevance to query
        # ==========================================
        query_words = set(w.lower().strip(string.punctuation) for w in query_stripped.split() 
                         if len(w.strip(string.punctuation)) > 2)
        # Remove very common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                      'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'would', 'could',
                      'will', 'with', 'this', 'that', 'from', 'they', 'were', 'what', 'when',
                      'where', 'which', 'who', 'how', 'why', 'does', 'did', 'about', 'into',
                      'more', 'some', 'than', 'them', 'then', 'there', 'these', 'those', 'your',
                      'also', 'just', 'like', 'make', 'many', 'much', 'over', 'such', 'take',
                      'very', 'most', 'only', 'come', 'its'}
        
        query_content_words = query_words - stop_words
        response_word_set = set(words_clean)
        
        if query_content_words:
            overlap = len(query_content_words & response_word_set) / len(query_content_words)
            relevance_score = overlap * 4.0
        else:
            relevance_score = 2.0  # neutral
        
        # ==========================================
        # FEATURE 7: Structural coherence indicators
        # ==========================================
        structural_score = 0.0
        
        # Proper capitalization at sentence starts
        proper_caps = 0
        for s in sentences:
            if s and s[0].isupper():
                proper_caps += 1
        if sentence_count > 0:
            cap_ratio = proper_caps / sentence_count
            structural_score += cap_ratio * 1.5
        
        # Ends with proper punctuation
        if response_stripped[-1] in '.!?':
            structural_score += 1.0
        elif response_stripped[-1] in ',:;':
            structural_score += 0.3
        
        # Has paragraph breaks for longer responses (indicates organization)
        if word_count > 50:
            paragraphs = [p.strip() for p in response_stripped.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                structural_score += 1.0
        
        # ==========================================
        # FEATURE 8: Absence of incoherence signals
        # ==========================================
        incoherence_penalty = 0.0
        
        # Random code in non-code context
        query_asks_code = bool(re.search(r'\b(code|program|function|script|html|python|java|implement)\b', 
                                          query_stripped.lower()))
        
        if not query_asks_code:
            code_indicators = [
                r'import\s+\w+', r'def\s+\w+\(', r'class\s+\w+:', 
                r'if\s+\w+\s*==', r'for\s+\w+\s+in\s+', r'print\(',
                r'\{\{', r'\}\}', r'function\s*\('
            ]
            code_count = sum(1 for p in code_indicators if re.search(p, response_stripped))
            if code_count >= 2:
                incoherence_penalty += 3.0
        
        # HTML in non-HTML context
        query_asks_html = bool(re.search(r'\b(html|tag|webpage|web page|website)\b', query_stripped.lower()))
        if not query_asks_html:
            html_tags = re.findall(r'<[^>]+>', response_stripped)
            if len(html_tags) > 2:
                incoherence_penalty += 2.0
        
        # "Input:" "Output:" repetition patterns (bot confusion)
        io_pattern_count = len(re.findall(r'(?:Input|Output)\s*:', response_stripped))
        if io_pattern_count > 2:
            incoherence_penalty += min(io_pattern_count * 0.8, 4.0)
        
        # "Question:" "Answer:" repetition (going off-topic)
        qa_pattern_count = len(re.findall(r'(?:Question|Answer)\s*:', response_stripped))
        if qa_pattern_count > 2:
            incoherence_penalty += min((qa_pattern_count - 2) * 1.0, 4.0)
        
        # Random hash symbols or markdown artifacts
        hash_count = response_stripped.count('#')
        if hash_count > 5 and not query_asks_code:
            incoherence_penalty += 1.0
        
        # ==========================================
        # FEATURE 9: Argument completeness
        # ==========================================
        completeness_score = 0.0
        
        # Check if response seems truncated
        if response_stripped[-1] not in '.!?"\')]}' and word_count > 20:
            # Possibly truncated
            completeness_score -= 0.5
        
        # Check for explanatory depth
        explanatory_markers = [
            r'\bthis\s+(?:is|means|indicates|suggests|shows)\b',
            r'\bit\s+(?:is|means|indicates|suggests|shows|can|will|may)\b',
            r'\bwhich\s+(?:is|means|indicates|suggests|shows)\b',
            r'\bcan\s+be\b', r'\bis\s+(?:a|an|the)\b',
            r'\bare\s+(?:used|known|considered|called)\b'
        ]
        explanation_count = sum(1 for p in explanatory_markers if re.search(p, response_lower))
        completeness_score += min(explanation_count * 0.7, 3.0)
        
        # ==========================================
        # FEATURE 10: Topic consistency (using word overlap between sentences)
        # ==========================================
        topic_consistency_score = 0.0
        
        if len(sentences) >= 2:
            # Check consecutive sentence overlap (content words)
            consecutive_overlaps = []
            for i in range(len(sentences) - 1):
                words_s1 = set(w.lower().strip(string.punctuation) for w in sentences[i].split() 
                              if len(w.strip(string.punctuation)) > 3) - stop_words
                words_s2 = set(w.lower().strip(string.punctuation) for w in sentences[i+1].split() 
                              if len(w.strip(string.punctuation)) > 3) - stop_words
                
                if words_s1 and words_s2:
                    overlap = len(words_s1 & words_s2) / min(len(words_s1), len(words_s2))
                    consecutive_overlaps.append(overlap)
            
            if consecutive_overlaps:
                avg_overlap = sum(consecutive_overlaps) / len(consecutive_overlaps)
                # Some overlap is good (topical consistency), but not too much (repetition)
                if 0.1 <= avg_overlap <= 0.5:
                    topic_consistency_score = 3.0
                elif 0.05 <= avg_overlap < 0.1 or 0.5 < avg_overlap <= 0.7:
                    topic_consistency_score = 2.0
                elif avg_overlap > 0.7:
                    topic_consistency_score = 1.0  # too repetitive
                else:
                    topic_consistency_score = 1.0  # low coherence
        else:
            topic_consistency_score = 1.5  # neutral for single sentence
        
        # ==========================================
        # COMBINE ALL FEATURES
        # ==========================================
        
        # Weighted combination
        raw_score = (
            length_score * 0.15 +          # 0-6 * 0.15 = 0-0.9
            connector_score * 0.12 +        # 0-8 * 0.12 = 0-0.96
            sent_len_score * 0.10 +         # 0-3 * 0.10 = 0-0.3
            variety_score * 0.08 +           # 0-2 * 0.08 = 0-0.16
            vocab_score * 0.10 +             # 0-3 * 0.10 = 0-0.3
            relevance_score * 0.15 +         # 0-4 * 0.15 = 0-0.6
            structural_score * 0.10 +        # 0-3.5 * 0.10 = 0-0.35
            completeness_score * 0.10 +      # -0.5-3 * 0.10 = -0.05-0.3
            topic_consistency_score * 0.10   # 0-3 * 0.10 = 0-0.3
        )
        # Theoretical max ~4.17
        
        # Apply penalties
        total_penalty = repetition_penalty + incoherence_penalty
        
        # Scale to 0-10
        scaled_score = (raw_score / 4.0) * 10.0
        scaled_score -= total_penalty
        
        # Clamp
        final_score = max(0.5, min(10.0, scaled_score))
        
        # Very short responses that aren't code/tags get extra penalty
        if word_count <= 3 and not query_asks_html:
            final_score = min(final_score, 2.0)
        
        if word_count <= 5 and sentence_count <= 1:
            final_score = min(final_score, 3.5)
        
        return round(final_score, 2)
    
    except Exception as e:
        # Fallback: return a middling score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.5
        except:
            return 2.0