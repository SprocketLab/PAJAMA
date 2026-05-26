def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level discourse analysis and causal/logical connective tracking.
    
    Algorithm: Analyzes the logical flow between sentences by detecting
    causal connectives, sequential markers, explanatory phrases, contrast
    markers, and conditional reasoning. Also measures sentence-to-sentence
    coherence via shared vocabulary progression.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5
        
        # ---- Feature 1: Causal/Logical Connective Density ----
        # These indicate reasoning transparency
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bleading to\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bthis suggests\b',
            r'\bin order to\b', r'\bso that\b', r'\bwhich means\b',
            r'\bwhich leads\b', r'\bwhich causes\b', r'\bfor this reason\b',
        ]
        
        explanatory_phrases = [
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bspecifically\b', r'\bnamely\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bto illustrate\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bto explain\b',
            r'\bin particular\b', r'\bmore specifically\b',
        ]
        
        sequential_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bafter that\b', r'\bfinally\b',
            r'\bsubsequently\b', r'\bfollowing this\b', r'\bat this point\b',
            r'\bonce\b', r'\bstep\b', r'\bto begin\b', r'\binitially\b',
            r'\bafterward\b', r'\blast(?:ly)?\b', r'\bin the first\b',
            r'\bin the second\b', r'\bin the third\b',
        ]
        
        contrast_markers = [
            r'\bhowever\b', r'\bbut\b', r'\bon the other hand\b',
            r'\bnevertheless\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bin contrast\b', r'\balthough\b', r'\bdespite\b',
            r'\byet\b', r'\bconversely\b', r'\bunlike\b',
            r'\bon the contrary\b', r'\bdiffers?\b',
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bwhen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bprovided that\b', r'\bin case\b', r'\bsuppose\b',
            r'\bwould\b', r'\bcould\b', r'\bdepending on\b',
        ]
        
        response_lower = response.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text, re.IGNORECASE))
            return total
        
        causal_count = count_patterns(causal_connectives, response_lower)
        explanatory_count = count_patterns(explanatory_phrases, response_lower)
        sequential_count = count_patterns(sequential_markers, response_lower)
        contrast_count = count_patterns(contrast_markers, response_lower)
        conditional_count = count_patterns(conditional_markers, response_lower)
        
        # Weighted connective score (normalized by sentence count)
        connective_raw = (
            causal_count * 3.0 +
            explanatory_count * 2.5 +
            sequential_count * 2.0 +
            contrast_count * 1.5 +
            conditional_count * 1.0
        )
        
        # Normalize: connectives per sentence, with diminishing returns
        connective_per_sent = connective_raw / max(num_sentences, 1)
        connective_score = min(connective_per_sent * 2.5, 10.0)
        
        # ---- Feature 2: Sentence-to-Sentence Vocabulary Progression ----
        # Measures how each sentence builds on previous ones (shared + new concepts)
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'and',
                'or', 'not', 'no', 'but', 'if', 'this', 'that', 'these',
                'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our',
                'you', 'your', 'he', 'she', 'his', 'her', 'i', 'me', 'my',
                'also', 'more', 'most', 'very', 'much', 'so', 'too', 'than',
                'which', 'who', 'whom', 'what', 'where', 'when', 'how',
                'all', 'each', 'every', 'both', 'some', 'any', 'such',
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        progression_score = 0.0
        if num_sentences >= 2:
            cumulative_words = set()
            coherence_values = []
            novelty_values = []
            
            for i, sent in enumerate(sentences):
                current_words = get_content_words(sent)
                if i == 0:
                    cumulative_words = current_words.copy()
                    continue
                
                if len(current_words) > 0 and len(cumulative_words) > 0:
                    overlap = len(current_words & cumulative_words)
                    new_words = len(current_words - cumulative_words)
                    
                    # Coherence: some overlap shows building on previous ideas
                    coherence = overlap / max(len(current_words), 1)
                    coherence_values.append(coherence)
                    
                    # Novelty: new words show progression of ideas
                    novelty = new_words / max(len(current_words), 1)
                    novelty_values.append(novelty)
                
                cumulative_words |= current_words
            
            if coherence_values and novelty_values:
                avg_coherence = sum(coherence_values) / len(coherence_values)
                avg_novelty = sum(novelty_values) / len(novelty_values)
                # Best: balanced coherence and novelty (both moderate)
                # This indicates building arguments step by step
                progression_score = (min(avg_coherence, 0.6) / 0.6) * 5.0 + \
                                    (min(avg_novelty, 0.7) / 0.7) * 5.0
            else:
                progression_score = 2.0
        else:
            progression_score = 1.0
        
        # ---- Feature 3: Sentence Complexity Distribution ----
        # Reasoning often involves varied sentence lengths (short conclusions, longer explanations)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)
        
        if len(sent_lengths) >= 2:
            variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Some variation is good (shows different types of statements)
            length_variation_score = min(std_dev / 5.0, 1.0) * 4.0
        else:
            length_variation_score = 0.5
        
        # ---- Feature 4: Depth/Elaboration Score ----
        word_count = len(response.split())
        
        # Penalize very short responses (unlikely to show reasoning)
        if word_count < 10:
            depth_score = 1.0
        elif word_count < 25:
            depth_score = 3.0
        elif word_count < 50:
            depth_score = 5.5
        elif word_count < 100:
            depth_score = 7.5
        elif word_count < 200:
            depth_score = 8.5
        else:
            depth_score = 8.0  # Very long may be verbose, slight penalty
        
        # ---- Feature 5: Clause density (commas, semicolons per sentence) ----
        # Complex sentences with multiple clauses often indicate more detailed reasoning
        comma_count = response.count(',')
        semicolon_count = response.count(';')
        colon_count = response.count(':')
        clause_markers = comma_count + semicolon_count * 2 + colon_count * 1.5
        clause_density = clause_markers / max(num_sentences, 1)
        clause_score = min(clause_density / 3.0, 1.0) * 6.0
        
        # ---- Feature 6: Repetition Penalty ----
        # Repetitive responses indicate low quality / no real reasoning
        words_list = re.findall(r'[a-z]+', response_lower)
        if len(words_list) > 5:
            word_freq = Counter(words_list)
            # Check for excessive repetition of non-stop words
            content_words_list = [w for w in words_list if len(w) > 3]
            if content_words_list:
                content_freq = Counter(content_words_list)
                most_common_count = content_freq.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(content_words_list)
                if repetition_ratio > 0.3:
                    repetition_penalty = (repetition_ratio - 0.3) * 20.0
                else:
                    repetition_penalty = 0.0
            else:
                repetition_penalty = 0.0
            
            # Also check for repeated phrases (bigrams)
            bigrams = [' '.join(words_list[i:i+2]) for i in range(len(words_list)-1)]
            if bigrams:
                bigram_freq = Counter(bigrams)
                most_common_bigram = bigram_freq.most_common(1)[0][1]
                bigram_ratio = most_common_bigram / len(bigrams)
                if bigram_ratio > 0.15:
                    repetition_penalty += (bigram_ratio - 0.15) * 15.0
        else:
            repetition_penalty = 0.0
        
        # ---- Feature 7: Query Engagement ----
        # Does the response address the query terms?
        query_words = get_content_words(query) if query else set()
        response_words = get_content_words(response)
        if query_words:
            query_coverage = len(query_words & response_words) / max(len(query_words), 1)
            engagement_score = query_coverage * 4.0
        else:
            engagement_score = 2.0
        
        # ---- Feature 8: Explicit Reasoning Verbs ----
        reasoning_verbs = [
            r'\bmeans\b', r'\bimplies\b', r'\bsuggests\b', r'\bindicates\b',
            r'\bdemonstrates\b', r'\bshows\b', r'\breveals\b', r'\bexplains\b',
            r'\billustrates\b', r'\brepresents\b', r'\bsignifies\b',
            r'\binvolves\b', r'\brequires\b', r'\ballows\b', r'\benables\b',
            r'\bcauses\b', r'\bresults\b', r'\baffects\b', r'\binfluences\b',
            r'\bdetermines\b', r'\bdepends\b',
        ]
        reasoning_verb_count = count_patterns(reasoning_verbs, response_lower)
        reasoning_verb_score = min(reasoning_verb_count / max(num_sentences, 1) * 3.0, 6.0)
        
        # ---- Combine all features ----
        total_score = (
            connective_score * 1.5 +       # max ~15
            progression_score * 1.2 +       # max ~12
            length_variation_score * 0.8 +  # max ~3.2
            depth_score * 1.5 +             # max ~12.75
            clause_score * 0.8 +            # max ~4.8
            engagement_score * 1.0 +        # max ~4
            reasoning_verb_score * 1.0      # max ~6
            - repetition_penalty * 2.0      # penalty
        )
        
        # Normalize to 0-100 range
        # Theoretical max is roughly 57.75, practical max around 40-45
        normalized = max(0.0, min(total_score * 100.0 / 50.0, 100.0))
        
        # Apply a slight sigmoid-like transformation for better discrimination
        # in the middle range
        if normalized > 0:
            normalized = 100.0 * (1.0 / (1.0 + math.exp(-0.08 * (normalized - 40))))
        
        return round(normalized, 2)
        
    except Exception:
        try:
            # Absolute fallback: simple length-based score
            if response and isinstance(response, str):
                return min(len(response.split()) / 5.0, 50.0)
            return 0.0
        except Exception:
            return 0.0