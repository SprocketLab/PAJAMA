def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level structural analysis, logical connective density,
    and reasoning chain detection.
    
    This variant focuses on:
    1. Sentence-level reasoning chain analysis (connectives between sentences)
    2. Causal/logical marker density within sentences
    3. Enumeration and structured breakdown detection
    4. Question-answer alignment with explanation depth
    5. Information density vs. noise ratio
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses almost never show reasoning
        if len(response_stripped) < 5:
            return 0.5
        
        # === Feature 1: Sentence-level analysis ===
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+(?:\s|$)', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # === Feature 2: Logical/causal connectives between and within sentences ===
        # These indicate reasoning flow
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bleads to\b', r'\bleading to\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b',
            r'\bwhich means\b', r'\bthis means\b', r'\bimplying\b',
            r'\baccordingly\b', r'\bit follows\b'
        ]
        
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bin other words\b', r'\bto illustrate\b', r'\bto clarify\b'
        ]
        
        sequential_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bafter that\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bfollowing this\b',
            r'\bsubsequently\b', r'\bstep\s*\d+\b', r'\b\d+\)\s', r'\b\d+\.\s'
        ]
        
        contrast_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\bbut\b', r'\byet\b',
            r'\bnonetheless\b'
        ]
        
        additive_connectives = [
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\blikewise\b', r'\bsimilarly\b'
        ]
        
        resp_lower = response_stripped.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectives, resp_lower)
        elaboration_count = count_patterns(elaboration_connectives, resp_lower)
        sequential_count = count_patterns(sequential_connectives, resp_lower)
        contrast_count = count_patterns(contrast_connectives, resp_lower)
        additive_count = count_patterns(additive_connectives, resp_lower)
        
        total_connectives = (causal_count * 2.0 + elaboration_count * 1.8 + 
                            sequential_count * 1.5 + contrast_count * 1.3 + 
                            additive_count * 1.0)
        
        # Normalize by number of sentences
        connective_density = total_connectives / num_sentences if num_sentences > 0 else 0
        # Cap and scale: 0-3.0 range is typical
        connective_score = min(connective_density, 3.0) / 3.0 * 10.0
        
        # === Feature 3: Sentence length progression and complexity ===
        # Good reasoning often has sentences of moderate-to-long length
        # with some variation (not all same length)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        # Ideal average sentence length for reasoning: 10-25 words
        if avg_sent_length < 3:
            length_score = 1.0
        elif avg_sent_length < 8:
            length_score = 3.0
        elif avg_sent_length <= 25:
            length_score = 6.0 + 4.0 * min((avg_sent_length - 8) / 17.0, 1.0)
        else:
            length_score = max(7.0, 10.0 - (avg_sent_length - 25) * 0.1)
        
        # Sentence length variance (some variation is good)
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate std_dev is good (3-8)
            if std_dev < 1:
                var_bonus = 0
            elif std_dev <= 8:
                var_bonus = min(std_dev / 8.0, 1.0) * 2.0
            else:
                var_bonus = max(0, 2.0 - (std_dev - 8) * 0.1)
        else:
            var_bonus = 0
        
        length_score = min(10.0, length_score + var_bonus)
        
        # === Feature 4: Explanation depth markers ===
        # Phrases that indicate the response is explaining WHY or HOW
        explanation_markers = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis works because\b',
            r'\bthis happens because\b', r'\bthis is due to\b',
            r'\bthe key\b', r'\bthe main\b', r'\bimportant to note\b',
            r'\bit\'s worth noting\b', r'\bkeep in mind\b',
            r'\bto understand\b', r'\blet\'s\b', r'\bconsider\b',
            r'\bwe can see\b', r'\bwe know\b', r'\bthis suggests\b',
            r'\bthis indicates\b', r'\bthis shows\b', r'\bwhich shows\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\boverall\b', r'\bputting it together\b'
        ]
        
        explanation_count = count_patterns(explanation_markers, resp_lower)
        explanation_score = min(explanation_count * 1.5, 10.0)
        
        # === Feature 5: Structural formatting (bullets, numbers, headers) ===
        bullet_patterns = [
            r'^\s*[-•*]\s+', r'^\s*\d+[\.\)]\s+', r'^\s*[a-zA-Z][\.\)]\s+',
            r'^\s*step\s*\d', r'^\s*#{1,3}\s+'
        ]
        
        lines = response_stripped.split('\n')
        structured_lines = 0
        for line in lines:
            for bp in bullet_patterns:
                if re.match(bp, line, re.IGNORECASE):
                    structured_lines += 1
                    break
        
        structure_ratio = structured_lines / max(len(lines), 1)
        # Having some structure is good, but not required
        structure_score = min(structure_ratio * 15.0, 8.0) if structured_lines >= 2 else structured_lines * 2.0
        
        # === Feature 6: Inter-sentence coherence ===
        # Check if sentences reference previous content (pronouns, demonstratives)
        reference_words = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bit\b', r'\bsuch\b', r'\bthe above\b', r'\bas mentioned\b',
            r'\bas noted\b', r'\bas discussed\b'
        ]
        
        reference_count = 0
        for i, sent in enumerate(sentences):
            if i == 0:
                continue  # First sentence doesn't need back-references
            sent_lower = sent.lower()
            for rw in reference_words:
                if re.search(rw, sent_lower):
                    reference_count += 1
                    break  # Count once per sentence
        
        if num_sentences > 1:
            coherence_ratio = reference_count / (num_sentences - 1)
        else:
            coherence_ratio = 0
        
        coherence_score = min(coherence_ratio * 10.0, 8.0)
        
        # === Feature 7: Content substance vs. noise ===
        # Detect repetition, garbage, HTML, code artifacts
        words = resp_lower.split()
        num_words = len(words)
        
        if num_words == 0:
            return 0.5
        
        # Repetition detection: bigram repetition rate
        if num_words > 2:
            bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
            repetition_rate = repeated_bigrams / max(len(bigrams), 1)
        else:
            repetition_rate = 0
        
        # High repetition is bad
        repetition_penalty = min(repetition_rate * 15.0, 8.0)
        
        # HTML/code noise detection
        html_matches = len(re.findall(r'<[^>]+>', response_stripped))
        code_matches = len(re.findall(r'(import |def |class |function |var |const |let )', response_stripped))
        noise_penalty = min((html_matches + code_matches) * 0.5, 4.0)
        
        # === Feature 8: Response length adequacy ===
        # Very short responses rarely show reasoning
        if num_words < 5:
            length_adequacy = 0.5
        elif num_words < 15:
            length_adequacy = 2.0
        elif num_words < 30:
            length_adequacy = 4.0
        elif num_words < 60:
            length_adequacy = 6.0
        elif num_words < 150:
            length_adequacy = 8.0
        elif num_words < 300:
            length_adequacy = 9.0
        else:
            # Very long might be rambling
            length_adequacy = max(6.0, 9.0 - (num_words - 300) * 0.005)
        
        # === Feature 9: Query-response alignment ===
        # Check if response addresses the query topic
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        response_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resp_lower))
        
        # Remove very common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'would', 'could', 'should', 'will', 'with',
                     'this', 'that', 'from', 'they', 'were', 'what', 'when',
                     'how', 'where', 'which', 'who', 'whom', 'why', 'does',
                     'did', 'about', 'into', 'than', 'then', 'them', 'these',
                     'those', 'some', 'such', 'more', 'also', 'just', 'any'}
        
        query_content = query_words - stopwords
        response_content = response_words - stopwords
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
        else:
            overlap = 0.5
        
        alignment_score = overlap * 6.0  # Max 6.0
        
        # === Feature 10: Intermediate conclusion markers ===
        # Sentences that state intermediate results before final conclusion
        intermediate_markers = [
            r'\bso\b', r'\bthus\b', r'\bmeaning\b', r'\bwhich means\b',
            r'\bfrom this\b', r'\bgiven this\b', r'\bbased on\b',
            r'\bwith this in mind\b', r'\btaking into account\b',
            r'\bwe can conclude\b', r'\bwe can see\b', r'\bwe find\b',
            r'\bthat means\b', r'\bthat suggests\b'
        ]
        
        intermediate_count = count_patterns(intermediate_markers, resp_lower)
        intermediate_score = min(intermediate_count * 2.0, 8.0)
        
        # === Combine all features with weights ===
        # Weighted combination
        raw_score = (
            connective_score * 0.18 +        # Logical connectives
            length_score * 0.10 +             # Sentence complexity
            explanation_score * 0.15 +        # Explanation depth
            structure_score * 0.08 +          # Formatting structure
            coherence_score * 0.10 +          # Inter-sentence coherence
            length_adequacy * 0.15 +          # Response length
            alignment_score * 0.08 +          # Query alignment
            intermediate_score * 0.12 +       # Intermediate conclusions
            # Penalties
            -repetition_penalty * 0.12 +
            -noise_penalty * 0.08
        )
        
        # Bonus for multi-sentence responses with connectives
        if num_sentences >= 3 and total_connectives >= 2:
            raw_score += 1.0
        
        # Bonus for having both causal AND sequential reasoning
        if causal_count >= 1 and sequential_count >= 1:
            raw_score += 0.8
        
        # Penalty for single-word or near-empty responses
        if num_words <= 3:
            raw_score = min(raw_score, 2.0)
        elif num_words <= 8:
            raw_score = min(raw_score, 4.0)
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            words = len(response.split()) if response else 0
            return min(max(words * 0.1, 0.5), 5.0)
        except Exception:
            return 2.0