def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    discourse marker analysis, causal/logical connective density, 
    explanation depth signals, and progressive elaboration patterns.
    
    This variant focuses on:
    1. Causal/logical connective detection (because, therefore, since, thus, etc.)
    2. Discourse progression markers (first, then, next, finally, etc.)
    3. Explanation signals (this means, in other words, for example, etc.)
    4. Clause complexity as proxy for reasoning depth
    5. Progressive information buildup (later sentences reference earlier concepts)
    6. Hedging/qualification markers showing nuanced reasoning
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 5:
            return 0.5
        
        import re
        from collections import Counter
        
        response_lower = response_clean.lower()
        
        # Tokenize into sentences using regex
        sentences = [s.strip() for s in re.split(r'[.!?]+', response_clean) if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-z]+\b', response_lower)
        num_words = max(len(words), 1)
        
        # === 1. Causal/Logical Connectives (reasoning markers) ===
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bwhich leads to\b',
            r'\bleading to\b', r'\bresulting in\b', r'\bcaused by\b',
            r'\bif\b.*\bthen\b', r'\bgiven that\b', r'\bassuming\b',
            r'\bit follows\b', r'\baccordingly\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, response_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 8.0, 15.0)
        
        # === 2. Discourse Progression / Sequencing Markers ===
        sequence_markers = [
            r'\bfirst\b', r'\bfirstly\b', r'\bsecond\b', r'\bsecondly\b',
            r'\bthird\b', r'\bthirdly\b', r'\bnext\b', r'\bthen\b',
            r'\bfinally\b', r'\blastly\b', r'\bafterward[s]?\b',
            r'\bsubsequently\b', r'\bin addition\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\balso\b', r'\badditionally\b',
            r'\bstep\b', r'\bphase\b', r'\bstage\b',
            r'\bto begin\b', r'\bto start\b', r'\bin the end\b',
        ]
        seq_count = 0
        for pattern in sequence_markers:
            seq_count += len(re.findall(pattern, response_lower))
        
        seq_density = seq_count / num_sentences
        seq_score = min(seq_density * 6.0, 12.0)
        
        # === 3. Explanation/Clarification Signals ===
        explanation_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bspecifically\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bto clarify\b', r'\bput simply\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bto explain\b',
            r'\bin particular\b', r'\bto be specific\b',
            r'\billustrat', r'\bdemonstrat',
        ]
        expl_count = 0
        for pattern in explanation_markers:
            expl_count += len(re.findall(pattern, response_lower))
        
        expl_score = min(expl_count * 3.0, 12.0)
        
        # === 4. Contrast/Comparison Markers (showing multi-perspective reasoning) ===
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bwhereas\b', r'\bwhile\b', r'\balthough\b', r'\bthough\b',
            r'\bdespite\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bconversely\b', r'\bbut\b', r'\byet\b',
            r'\bunlike\b', r'\bdiffers?\b', r'\bsimilar\b',
        ]
        contrast_count = 0
        for pattern in contrast_markers:
            contrast_count += len(re.findall(pattern, response_lower))
        
        contrast_density = contrast_count / num_sentences
        contrast_score = min(contrast_density * 5.0, 10.0)
        
        # === 5. Clause Complexity (commas per sentence as proxy for multi-clause reasoning) ===
        comma_count = response_clean.count(',')
        avg_commas_per_sentence = comma_count / num_sentences
        # Multi-clause sentences suggest more complex reasoning
        clause_score = min(avg_commas_per_sentence * 2.5, 8.0)
        
        # === 6. Progressive Elaboration Detection ===
        # Check if later sentences build on concepts from earlier sentences
        # by measuring word overlap between consecutive sentences
        progressive_score = 0.0
        if num_sentences >= 2:
            sentence_words = []
            for s in sentences:
                sw = set(re.findall(r'\b[a-z]+\b', s.lower()))
                # Remove very common words
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                            'it', 'its', 'this', 'that', 'and', 'or', 'not', 'as'}
                sw = sw - stopwords
                sentence_words.append(sw)
            
            # Check coherence: consecutive sentences sharing content words
            coherence_links = 0
            new_info_count = 0
            accumulated = set()
            for i in range(len(sentence_words)):
                if i == 0:
                    accumulated = sentence_words[i].copy()
                    continue
                current = sentence_words[i]
                if accumulated and current:
                    overlap = len(current & accumulated)
                    new_words = len(current - accumulated)
                    if overlap > 0 and new_words > 0:
                        coherence_links += 1
                    if new_words > 0:
                        new_info_count += 1
                accumulated |= current
            
            if num_sentences > 1:
                coherence_ratio = coherence_links / (num_sentences - 1)
                progressive_score = coherence_ratio * 8.0
        
        # === 7. Hedging/Qualification (nuanced reasoning) ===
        hedge_markers = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\boften\b', r'\btends? to\b', r'\bin most cases\b',
            r'\bit depends\b', r'\brelatively\b', r'\bsomewhat\b',
            r'\bperhaps\b', r'\blikely\b', r'\bpossibly\b',
            r'\bin some cases\b', r'\bnot always\b', r'\bnot necessarily\b',
            r'\bcan be\b', r'\bmay be\b',
        ]
        hedge_count = 0
        for pattern in hedge_markers:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_score = min(hedge_count * 2.0, 6.0)
        
        # === 8. Structural Depth (sentence count bonus) ===
        # More sentences generally means more step-wise explanation
        # But diminishing returns
        import math
        depth_score = min(math.log(num_sentences + 1) * 3.0, 10.0)
        
        # === 9. Repetition Penalty ===
        # Detect repetitive content (sign of low-quality generation)
        word_freq = Counter(words)
        if num_words > 10:
            # Calculate ratio of unique words
            unique_ratio = len(word_freq) / num_words
            # Very low unique ratio = repetitive
            if unique_ratio < 0.3:
                repetition_penalty = 15.0
            elif unique_ratio < 0.4:
                repetition_penalty = 8.0
            elif unique_ratio < 0.5:
                repetition_penalty = 3.0
            else:
                repetition_penalty = 0.0
        else:
            repetition_penalty = 0.0
        
        # Check for repeated phrases (3-grams)
        if len(words) >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_freq = Counter(trigrams)
            repeated_trigrams = sum(1 for v in trigram_freq.values() if v > 2)
            if repeated_trigrams > 3:
                repetition_penalty += 10.0
        
        # === 10. Emptiness / Minimal Response Penalty ===
        brevity_penalty = 0.0
        if num_words < 10:
            brevity_penalty = 10.0
        elif num_words < 20:
            brevity_penalty = 5.0
        elif num_words < 30:
            brevity_penalty = 2.0
        
        # === 11. Query-Response Relevance via Key Concept Coverage ===
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]+\b', query_lower))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'it', 'its', 'this', 'that', 'and', 'or', 'not', 'as',
                    'what', 'how', 'why', 'when', 'where', 'who', 'which',
                    'your', 'you', 'me', 'my', 'i', 'we', 'our', 'they', 'them',
                    'he', 'she', 'his', 'her', 'up', 'down', 'about'}
        query_content = query_words - stopwords
        response_word_set = set(words)
        
        if query_content:
            coverage = len(query_content & response_word_set) / len(query_content)
            relevance_score = coverage * 5.0
        else:
            relevance_score = 2.5
        
        # === 12. "Why" explanation detection ===
        # Sentences that contain explanatory structure
        why_patterns = [
            r'\bthis is\b.*\bbecause\b',
            r'\bthe reason\b',
            r'\bwhich allows\b',
            r'\bwhich enables\b',
            r'\bwhich helps\b',
            r'\bwhich makes\b',
            r'\bmeaning that\b',
            r'\bso that\b',
            r'\bin order\b',
            r'\bthe purpose\b',
            r'\bthe goal\b',
            r'\bimplies\b',
            r'\bsuggests that\b',
            r'\bindicates\b',
        ]
        why_count = 0
        for pattern in why_patterns:
            why_count += len(re.findall(pattern, response_lower))
        
        why_score = min(why_count * 3.0, 10.0)
        
        # === Aggregate Score ===
        total = (
            causal_score +          # max 15
            seq_score +             # max 12
            expl_score +            # max 12
            contrast_score +        # max 10
            clause_score +          # max 8
            progressive_score +     # max 8
            hedge_score +           # max 6
            depth_score +           # max 10
            relevance_score +       # max 5
            why_score -             # max 10
            repetition_penalty -    # max ~25
            brevity_penalty         # max 10
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~ 96, typical good response ~40-60
        score = max(0.0, min(100.0, total))
        
        return round(score, 2)
        
    except Exception:
        return 5.0