def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    a discourse coherence and causal chain analysis approach.
    
    This variant focuses on:
    1. Causal/logical connective density (tracking discourse markers that signal reasoning)
    2. Sentence-to-sentence coherence flow (do sentences build on each other?)
    3. Explanation depth via clause complexity
    4. Progressive information development (does new info appear incrementally?)
    5. Absence of noise/irrelevant content
    
    Different from other variants by using discourse marker taxonomy, 
    inter-sentence semantic bridging, and clause-level complexity analysis.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses likely lack reasoning
        if len(response_stripped) < 5:
            return 0.5
        
        # ---- Feature 1: Discourse Marker Taxonomy ----
        # Categorize connectives by their reasoning function
        
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bleads to\b', r'\bresulting in\b',
            r'\bfor this reason\b', r'\bowing to\b', r'\bthat\'s why\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bso that\b',
            r'\bin order to\b', r'\bthe reason\b'
        ]
        
        sequential_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bafter that\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bfollowing\b', r'\bsubsequently\b',
            r'\bstep\s*\d', r'\b\d+\)\s', r'\b\d+\.\s',
            r'\binitially\b', r'\bafterward\b', r'\bpreviously\b'
        ]
        
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bmore specifically\b',
            r'\bto clarify\b', r'\bto elaborate\b', r'\bin detail\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bdespite\b', r'\bin contrast\b', r'\byet\b', r'\bstill\b',
            r'\bnonetheless\b', r'\beven though\b', r'\bconversely\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b', r'\bwhenever\b', r'\bgiven that\b',
            r'\bsuppose\b', r'\bwould\b.*\bif\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bto sum up\b',
            r'\bultimately\b', r'\ball in all\b', r'\btaken together\b'
        ]
        
        resp_lower = response_stripped.lower()
        
        def count_markers(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_markers(causal_markers, resp_lower)
        sequential_count = count_markers(sequential_markers, resp_lower)
        elaboration_count = count_markers(elaboration_markers, resp_lower)
        contrastive_count = count_markers(contrastive_markers, resp_lower)
        conditional_count = count_markers(conditional_markers, resp_lower)
        conclusion_count = count_markers(conclusion_markers, resp_lower)
        
        total_discourse = (causal_count + sequential_count + elaboration_count + 
                          contrastive_count + conditional_count + conclusion_count)
        
        # Weight causal and sequential markers more heavily for reasoning
        weighted_discourse = (causal_count * 2.0 + sequential_count * 1.8 + 
                             elaboration_count * 1.5 + contrastive_count * 1.3 +
                             conditional_count * 1.0 + conclusion_count * 1.2)
        
        # ---- Feature 2: Sentence-level analysis ----
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # ---- Feature 3: Inter-sentence bridging (content word overlap between consecutive sentences) ----
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'once', 'and', 'or', 'nor', 'not', 'no',
                'so', 'very', 'just', 'also', 'than', 'too', 'this', 'that', 'these',
                'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                'he', 'she', 'him', 'her', 'his', 'they', 'them', 'their', 'what',
                'which', 'who', 'whom', 'where', 'when', 'how', 'all', 'each',
                'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                'only', 'own', 'same', 'here', 'there', 'up', 'down', 'about'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        bridging_scores = []
        if num_sentences >= 2:
            for i in range(1, len(sentences)):
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    bridging_scores.append(overlap / union if union > 0 else 0)
        
        avg_bridging = sum(bridging_scores) / len(bridging_scores) if bridging_scores else 0
        
        # ---- Feature 4: Clause complexity (subordinate clauses indicate deeper reasoning) ----
        subordinate_patterns = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhere\b',
            r'\bwhen\b', r'\bwhile\b', r'\balthough\b', r'\bbecause\b',
            r'\bsince\b', r'\bunless\b', r'\bif\b', r'\beven though\b',
            r'\bso that\b', r'\bin order that\b', r'\bwherever\b', r'\bwhenever\b'
        ]
        
        clause_count = 0
        for p in subordinate_patterns:
            clause_count += len(re.findall(p, resp_lower))
        
        clauses_per_sentence = clause_count / num_sentences
        
        # ---- Feature 5: Progressive information development ----
        # Check if new content words are introduced progressively (not all at once)
        if num_sentences >= 3:
            cumulative_words = set()
            new_word_counts = []
            for s in sentences:
                cw = get_content_words(s)
                new_words = cw - cumulative_words
                new_word_counts.append(len(new_words))
                cumulative_words.update(cw)
            
            # Good reasoning introduces info gradually; check variance isn't too extreme
            if len(new_word_counts) > 1:
                mean_new = sum(new_word_counts) / len(new_word_counts)
                if mean_new > 0:
                    # Coefficient of variation - lower means more even distribution
                    variance = sum((x - mean_new)**2 for x in new_word_counts) / len(new_word_counts)
                    cv = math.sqrt(variance) / mean_new
                    progressive_score = max(0, 1 - cv * 0.3)  # Penalize high variance
                else:
                    progressive_score = 0.2
            else:
                progressive_score = 0.5
        else:
            progressive_score = 0.3
        
        # ---- Feature 6: Noise/garbage detection ----
        # Detect repetitive content, HTML tags, code blocks, off-topic rambling
        
        # Repetition detection: check for repeated sentences
        sentence_set = set()
        repeated = 0
        for s in sentences:
            s_normalized = re.sub(r'\s+', ' ', s.lower().strip())
            if s_normalized in sentence_set:
                repeated += 1
            sentence_set.add(s_normalized)
        repetition_ratio = repeated / num_sentences if num_sentences > 0 else 0
        
        # HTML/code noise
        html_tags = len(re.findall(r'<[^>]+>', response_stripped))
        code_blocks = len(re.findall(r'```|import |def |class |function\(', response_stripped))
        
        noise_penalty = min(1.0, (html_tags * 0.1 + code_blocks * 0.15 + repetition_ratio * 0.5))
        
        # ---- Feature 7: Response substantiveness relative to query ----
        query_words = get_content_words(query)
        response_words = get_content_words(response_stripped)
        
        # Does the response address the query?
        if query_words and response_words:
            query_coverage = len(query_words & response_words) / len(query_words) if query_words else 0
        else:
            query_coverage = 0.3
        
        # ---- Feature 8: Explanation depth indicators ----
        explanation_phrases = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis means\b',
            r'\bin other words\b', r'\bto explain\b', r'\bto understand\b',
            r'\blet me explain\b', r'\bhere\'s why\b', r'\bthe key\b',
            r'\bimportant to note\b', r'\bnote that\b', r'\bkeep in mind\b',
            r'\bit\'s worth\b', r'\bconsider\b', r'\bwe can see\b',
            r'\bthis suggests\b', r'\bthis indicates\b', r'\bthis implies\b',
            r'\bas we can see\b', r'\blooking at\b', r'\banalyzing\b',
            r'\bbreaking down\b', r'\blet\'s\b', r'\bwe need to\b'
        ]
        
        explanation_count = count_markers(explanation_phrases, resp_lower)
        
        # ---- Feature 9: Appropriate length ----
        word_count = len(re.findall(r'\S+', response_stripped))
        
        # Length scoring: too short = bad, moderate = good, very long can be okay if structured
        if word_count < 5:
            length_score = 0.1
        elif word_count < 15:
            length_score = 0.3
        elif word_count < 30:
            length_score = 0.5
        elif word_count < 60:
            length_score = 0.7
        elif word_count < 150:
            length_score = 0.85
        elif word_count < 300:
            length_score = 0.95
        else:
            length_score = 0.9  # Slightly less for very long (might be rambling)
        
        # ---- Feature 10: Question-response alignment ----
        # Check if response starts with relevant content (not garbage)
        first_30_chars = response_stripped[:30].lower()
        
        # Penalize responses that start with noise
        bad_starts = ['.', '#', 'input:', 'output:', '```', 'import ', 'def ']
        start_penalty = 0
        for bs in bad_starts:
            if first_30_chars.startswith(bs):
                start_penalty = 0.2
                break
        
        # ---- Combine all features into final score ----
        
        # Discourse marker density (normalized by sentence count)
        discourse_density = min(weighted_discourse / num_sentences, 3.0) / 3.0 if num_sentences > 0 else 0
        
        # Normalize individual scores to [0, 1] range
        bridging_norm = min(avg_bridging * 3, 1.0)  # Scale up since overlap is usually small
        clause_norm = min(clauses_per_sentence / 2.5, 1.0)
        explanation_norm = min(explanation_count / 3.0, 1.0)
        
        # Weighted combination
        score = (
            discourse_density * 1.8 +        # Reasoning connectives
            bridging_norm * 1.2 +             # Sentence coherence
            clause_norm * 1.0 +               # Clause complexity
            progressive_score * 0.8 +         # Info development
            explanation_norm * 1.5 +           # Explicit explanations
            length_score * 1.5 +              # Appropriate length
            query_coverage * 0.8 +            # Relevance
            (1 - noise_penalty) * 1.0         # Clean content
        )
        
        # Apply start penalty
        score -= start_penalty * 2
        
        # Normalize to 0-10 scale
        max_possible = 1.8 + 1.2 + 1.0 + 0.8 + 1.5 + 1.5 + 0.8 + 1.0  # = 9.6
        normalized_score = (score / max_possible) * 10
        
        # Clamp to [0.5, 10]
        final_score = max(0.5, min(10.0, normalized_score))
        
        # Special case: extremely short responses with no reasoning
        if word_count <= 3 and total_discourse == 0:
            final_score = min(final_score, 1.5)
        
        if word_count <= 8 and total_discourse == 0 and num_sentences <= 1:
            final_score = min(final_score, 2.5)
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except:
            return 1.0