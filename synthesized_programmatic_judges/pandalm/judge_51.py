def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level analysis of logical flow, causal/explanatory language,
    and progressive elaboration patterns.
    
    This variant focuses on:
    1. Causal/explanatory connective density
    2. Sentence-to-sentence information progression
    3. Claim-support pairing detection
    4. Depth of elaboration chains
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        import re
        from collections import Counter
        
        # === 1. CAUSAL/EXPLANATORY CONNECTIVE ANALYSIS ===
        # Detect phrases that signal reasoning/explanation
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bwhich implies\b', r'\bso that\b',
            r'\bin order to\b', r'\bfor this reason\b', r'\bit follows\b',
            r'\baccordingly\b', r'\bas such\b',
        ]
        
        elaboration_markers = [
            r'\bspecifically\b', r'\bin particular\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bto illustrate\b', r'\bto clarify\b',
            r'\bmore specifically\b', r'\bincluding\b',
        ]
        
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bwhereas\b',
            r'\bwhile\b', r'\balthough\b', r'\bin contrast\b',
            r'\bnevertheless\b', r'\bconversely\b', r'\bbut\b',
            r'\bdespite\b', r'\bunlike\b', r'\bdiffers?\b',
        ]
        
        sequencing_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterwards?\b', r'\binitially\b', r'\bto begin\b',
            r'\bfollowing this\b', r'\bafter that\b', r'\blast(?:ly)?\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\badditionally\b', r'\balso\b',
        ]
        
        response_lower = response.lower()
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_connectives)
        elaboration_count = sum(len(re.findall(p, response_lower)) for p in elaboration_markers)
        contrast_count = sum(len(re.findall(p, response_lower)) for p in contrast_markers)
        sequence_count = sum(len(re.findall(p, response_lower)) for p in sequencing_markers)
        
        # === 2. SENTENCE-LEVEL ANALYSIS ===
        # Split into sentences
        sentence_endings = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # === 3. INFORMATION PROGRESSION (new vocab introduced per sentence) ===
        # Measures whether each sentence adds new information
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'and', 'or', 'not',
                'no', 'but', 'if', 'than', 'too', 'very', 'just', 'about', 'up',
                'out', 'so', 'it', 'its', 'this', 'that', 'these', 'those', 'they',
                'them', 'their', 'he', 'she', 'his', 'her', 'we', 'our', 'you',
                'your', 'i', 'my', 'me', 'which', 'who', 'whom', 'what', 'when',
                'where', 'how', 'all', 'each', 'every', 'both', 'more', 'most',
                'other', 'some', 'such', 'only', 'own', 'same', 'also',
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        if num_sentences >= 2:
            cumulative_words = set()
            new_word_ratios = []
            for sent in sentences:
                sent_words = get_content_words(sent)
                if len(sent_words) > 0:
                    new_words = sent_words - cumulative_words
                    ratio = len(new_words) / len(sent_words)
                    new_word_ratios.append(ratio)
                    cumulative_words.update(sent_words)
            
            # Average new information ratio (skip first sentence)
            if len(new_word_ratios) > 1:
                progression_score = sum(new_word_ratios[1:]) / len(new_word_ratios[1:])
            else:
                progression_score = 0.3
        else:
            progression_score = 0.1  # Single sentence = low progression
        
        # === 4. CLAIM-SUPPORT PATTERN DETECTION ===
        # Look for patterns where a claim is followed by supporting detail
        # Heuristic: sentence pairs where second sentence has explanatory markers
        # or provides specifics about the first
        claim_support_pairs = 0
        if num_sentences >= 2:
            for i in range(len(sentences) - 1):
                s2_lower = sentences[i + 1].lower()
                # Check if second sentence explains/supports first
                has_support_signal = any(
                    re.search(p, s2_lower) 
                    for p in causal_connectives + elaboration_markers
                )
                # Also check if second sentence shares topic words with first
                words1 = get_content_words(sentences[i])
                words2 = get_content_words(sentences[i + 1])
                if words1 and words2:
                    topic_overlap = len(words1 & words2) / max(len(words1), 1)
                    if has_support_signal or (topic_overlap > 0.2 and len(words2 - words1) > 1):
                        claim_support_pairs += 1
            
            claim_support_ratio = claim_support_pairs / max(num_sentences - 1, 1)
        else:
            claim_support_ratio = 0.0
        
        # === 5. DEPTH OF ELABORATION CHAINS ===
        # Detect chains of reasoning (A -> B -> C)
        # Approximated by consecutive sentences with connectives
        chain_length = 0
        max_chain = 0
        current_chain = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_connective = any(
                re.search(p, sent_lower) 
                for p in causal_connectives + sequencing_markers + elaboration_markers
            )
            if has_connective:
                current_chain += 1
                max_chain = max(max_chain, current_chain)
            else:
                current_chain = 0
        
        # === 6. SENTENCE COMPLEXITY AND VARIETY ===
        # Longer, more varied sentences suggest more detailed reasoning
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sent_length = sum(sentence_lengths) / num_sentences if sentence_lengths else 0
        
        # Variance in sentence length (variety suggests different roles: claim vs support)
        if len(sentence_lengths) > 1:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            length_variety = min(variance ** 0.5 / max(mean_len, 1), 1.0)
        else:
            length_variety = 0.0
        
        # === 7. REPETITION PENALTY ===
        # Detect repetitive content (bad sign for reasoning)
        words_list = re.findall(r'[a-z]+', response_lower)
        if len(words_list) > 5:
            # Bigram repetition
            bigrams = [f"{words_list[i]}_{words_list[i+1]}" for i in range(len(words_list)-1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                max_bigram_freq = max(bigram_counts.values())
                total_bigrams = len(bigrams)
                repetition_ratio = max_bigram_freq / total_bigrams
                # High repetition is bad
                repetition_penalty = max(0, (repetition_ratio - 0.05) * 10)
            else:
                repetition_penalty = 0
        else:
            repetition_penalty = 0
        
        # === 8. RESPONSE SUBSTANTIVENESS ===
        total_words = len(words_list)
        # Minimum viable response length
        if total_words < 3:
            return 0.5
        
        # === 9. QUERY-RESPONSE ALIGNMENT ===
        # Check that response addresses the query topic
        query_content = get_content_words(query)
        response_content = get_content_words(response)
        if query_content and response_content:
            relevance = len(query_content & response_content) / max(len(query_content), 1)
        else:
            relevance = 0.5
        
        # === COMPOSITE SCORING ===
        # Normalize connective counts by number of sentences
        causal_density = min(causal_count / num_sentences, 2.0)
        elaboration_density = min(elaboration_count / num_sentences, 2.0)
        contrast_density = min(contrast_count / num_sentences, 1.5)
        sequence_density = min(sequence_count / num_sentences, 2.0)
        
        total_connective_density = (
            causal_density * 1.5 +
            elaboration_density * 1.2 +
            contrast_density * 1.0 +
            sequence_density * 0.8
        )
        
        # Sentence count score (more sentences = more steps, up to a point)
        sentence_score = min(num_sentences / 6.0, 1.0)
        
        # Word count score (adequate length for reasoning)
        length_score = min(total_words / 80.0, 1.0)
        
        # Chain depth score
        chain_score = min(max_chain / 3.0, 1.0)
        
        # Avg sentence length score (moderate length is ideal)
        sent_length_score = min(avg_sent_length / 20.0, 1.0)
        
        # Final composite
        score = (
            total_connective_density * 1.8 +      # Causal/explanatory language (max ~8.1)
            progression_score * 3.0 +               # New info per sentence (max 3.0)
            claim_support_ratio * 3.5 +             # Claim-support pairs (max 3.5)
            chain_score * 2.5 +                     # Reasoning chains (max 2.5)
            sentence_score * 2.0 +                  # Number of steps (max 2.0)
            length_score * 1.5 +                    # Adequate length (max 1.5)
            length_variety * 1.0 +                  # Sentence variety (max 1.0)
            sent_length_score * 1.0 +               # Sentence complexity (max 1.0)
            relevance * 1.5 +                       # Query relevance (max 1.5)
            - repetition_penalty * 2.0              # Repetition penalty
        )
        
        # Clamp to 0-10 range
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
    
    except Exception:
        return 2.0