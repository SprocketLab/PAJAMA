def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level analysis of logical flow, causal connectives, and
    progressive reasoning structure.
    
    Algorithm: Analyzes the response at the sentence level for:
    1. Causal/logical connective density between sentences
    2. Progressive information buildup (each sentence adds new content)
    3. Explanation depth via subordinate clause patterns
    4. Ratio of assertive claims that are followed by supporting elaboration
    5. Sentence-to-sentence coherence via shared reference terms
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query else ""
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if len(sentences) == 0:
            return 0.5
        
        # --- Feature 1: Causal/Logical Connectives ---
        # These indicate reasoning steps and explanations
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bso that\b', r'\bin order to\b',
            r'\bfor this reason\b', r'\bit follows\b', r'\baccordingly\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bcaused by\b',
            r'\bleads to\b', r'\bresults in\b', r'\bimplies\b',
        ]
        
        sequential_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bfollowing this\b', r'\bin the first place\b',
            r'\bto begin\b', r'\bto start\b', r'\bstep\s*\d+\b',
            r'\blastly\b', r'\binitially\b', r'\bafter that\b',
            r'\b\d+\)\s', r'\b\d+\.\s',
        ]
        
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bmore specifically\b',
            r'\bto clarify\b', r'\bto explain\b', r'\bthis includes\b',
        ]
        
        contrast_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bdespite\b',
            r'\bwhile\b', r'\byet\b', r'\bbut\b', r'\bin contrast\b',
            r'\bnonetheless\b', r'\beven though\b',
        ]
        
        response_lower = response.lower()
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_connectives)
        sequential_count = sum(len(re.findall(p, response_lower)) for p in sequential_connectives)
        elaboration_count = sum(len(re.findall(p, response_lower)) for p in elaboration_connectives)
        contrast_count = sum(len(re.findall(p, response_lower)) for p in contrast_connectives)
        
        total_connectives = causal_count * 2.0 + sequential_count * 1.5 + elaboration_count * 1.8 + contrast_count * 1.2
        
        # Normalize by number of sentences
        connective_density = total_connectives / max(len(sentences), 1)
        connective_score = min(connective_density * 1.5, 3.0)  # max 3 points
        
        # --- Feature 2: Progressive Information Buildup ---
        # Measure how much new content each successive sentence introduces
        def get_content_words(text):
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'as', 'into', 'through', 'during', 'before', 'after', 'and',
                         'or', 'not', 'no', 'but', 'if', 'than', 'that', 'this',
                         'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'them',
                         'their', 'my', 'your', 'his', 'her', 'our', 'what', 'which',
                         'who', 'when', 'where', 'how', 'all', 'each', 'every', 'both',
                         'few', 'more', 'most', 'other', 'some', 'such', 'only', 'also',
                         'very', 'just', 'about', 'up', 'out', 'so', 'there', 'here'}
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        if len(sentences) >= 2:
            cumulative_words = set()
            new_info_ratios = []
            for sent in sentences:
                sent_words = get_content_words(sent)
                if len(sent_words) > 0:
                    new_words = sent_words - cumulative_words
                    ratio = len(new_words) / len(sent_words)
                    new_info_ratios.append(ratio)
                    cumulative_words.update(sent_words)
            
            if new_info_ratios:
                avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
                # We want moderate new info (not too repetitive, not random)
                # Best is around 0.5-0.8
                if avg_new_info > 0.95:
                    # Might be incoherent/random
                    progressive_score = 0.5
                elif avg_new_info < 0.15:
                    # Very repetitive
                    progressive_score = 0.3
                else:
                    progressive_score = min(avg_new_info * 2.0, 1.5)
            else:
                progressive_score = 0.3
        else:
            progressive_score = 0.3
        
        # --- Feature 3: Sentence-to-Sentence Coherence ---
        # Adjacent sentences should share some reference terms (anaphora, topic continuity)
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if len(words_a) > 0 and len(words_b) > 0:
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    coherence_scores.append(overlap / union if union > 0 else 0)
            
            if coherence_scores:
                avg_coherence = sum(coherence_scores) / len(coherence_scores)
                # Moderate coherence is best (0.1-0.4 range)
                coherence_score = min(avg_coherence * 4.0, 1.5)
            else:
                coherence_score = 0.3
        else:
            coherence_score = 0.2
        
        # --- Feature 4: Subordinate Clause Depth ---
        # Count subordinate clauses that indicate reasoning depth
        subordinate_patterns = [
            r'\bwhich\b', r'\bwhere\b', r'\bwhen\b', r'\bwho\b',
            r'\bthat\s+(?:is|are|was|were|has|have|had|will|would|could|should|can|may|might)\b',
            r',\s*\w+ing\b',  # participial phrases
            r'\bif\s+\w+', r'\bunless\b', r'\bprovided that\b',
            r'\bgiven that\b', r'\bassuming\b',
        ]
        
        subordinate_count = sum(len(re.findall(p, response_lower)) for p in subordinate_patterns)
        subordinate_density = subordinate_count / max(len(sentences), 1)
        subordinate_score = min(subordinate_density * 0.8, 1.5)
        
        # --- Feature 5: Response Substantiveness ---
        # Meaningful length relative to query (not too short, not garbage)
        response_words = re.findall(r'\b\w+\b', response)
        query_words = re.findall(r'\b\w+\b', query) if query else []
        
        word_count = len(response_words)
        
        if word_count <= 2:
            length_score = 0.0
        elif word_count <= 5:
            length_score = 0.3
        elif word_count <= 15:
            length_score = 0.7
        elif word_count <= 50:
            length_score = 1.2
        elif word_count <= 150:
            length_score = 1.5
        elif word_count <= 300:
            length_score = 1.3
        else:
            length_score = 1.0
        
        # --- Feature 6: Explanation Markers (why/how explanations) ---
        explanation_patterns = [
            r'\bthis is\b', r'\bthis means\b', r'\bthe reason\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\boverall\b', r'\bput simply\b', r'\bin short\b',
            r'\bwe can see\b', r'\bwe know\b', r'\bit is important\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bconsider\b',
            r'\blet\'s\b', r'\blet us\b', r'\brecall\b',
        ]
        
        explanation_count = sum(len(re.findall(p, response_lower)) for p in explanation_patterns)
        explanation_score = min(explanation_count * 0.4, 1.0)
        
        # --- Feature 7: Absence of Garbage/Repetition ---
        # Detect repetitive content or garbage patterns
        # Check for repeated sentences
        sentence_set = set()
        duplicate_count = 0
        for s in sentences:
            s_normalized = re.sub(r'\s+', ' ', s.lower().strip())
            if s_normalized in sentence_set:
                duplicate_count += 1
            sentence_set.add(s_normalized)
        
        repetition_penalty = min(duplicate_count * 0.8, 3.0)
        
        # Check for HTML/code garbage
        html_code_ratio = len(re.findall(r'<[^>]+>|```|import\s+\w+|def\s+\w+|class\s+\w+', response)) / max(word_count, 1)
        garbage_penalty = min(html_code_ratio * 5.0, 2.0)
        
        # Check for excessive question marks or prompt-like patterns (response asking questions back)
        question_echo_patterns = re.findall(r'(?:Question:|Input:|Output:)', response)
        echo_penalty = min(len(question_echo_patterns) * 0.5, 2.0)
        
        # --- Feature 8: Multi-sentence reasoning chains ---
        # Reward responses where sentences build on each other
        # Look for pronoun references to previous content
        reference_pronouns = [r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
                             r'\bit\b', r'\bsuch\b', r'\bthe above\b', r'\bas mentioned\b']
        
        if len(sentences) >= 2:
            # Count references in non-first sentences
            ref_count = 0
            for sent in sentences[1:]:
                sent_lower = sent.lower()
                for pron in reference_pronouns:
                    ref_count += len(re.findall(pron, sent_lower))
            
            reference_density = ref_count / max(len(sentences) - 1, 1)
            reference_score = min(reference_density * 0.3, 1.0)
        else:
            reference_score = 0.0
        
        # --- Combine all features ---
        raw_score = (
            connective_score +      # max 3.0
            progressive_score +     # max 1.5
            coherence_score +       # max 1.5
            subordinate_score +     # max 1.5
            length_score +          # max 1.5
            explanation_score +     # max 1.0
            reference_score -       # max 1.0
            repetition_penalty -    # max 3.0
            garbage_penalty -       # max 2.0
            echo_penalty            # max 2.0
        )
        
        # Scale to 0-10 range
        # Theoretical max ~11.0, theoretical min ~ -7.0
        # Shift and scale
        final_score = (raw_score + 2.0) * (10.0 / 12.0)
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, final_score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Center around 5, stretch differences
        centered = final_score - 5.0
        transformed = 5.0 + 5.0 * (2.0 / (1.0 + math.exp(-0.5 * centered)) - 1.0)
        
        return round(max(0.0, min(10.0, transformed)), 2)
        
    except Exception:
        return 3.0