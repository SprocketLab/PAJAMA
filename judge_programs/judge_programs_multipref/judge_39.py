def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis,
    causal/logical connector tracking, sentence-level coherence via TF vectors,
    and contradiction detection.
    
    This variant focuses on:
    1. Discourse marker analysis (causal, contrastive, additive, temporal connectors)
    2. Sentence-to-sentence semantic drift (cosine similarity of TF vectors between consecutive sentences)
    3. Argument depth via nested reasoning detection
    4. Contradiction signals
    5. Topic consistency via entropy of word distributions across segments
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation, keeping it simple but effective
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
            # Also split on newlines that seem to separate ideas
            expanded = []
            for s in sents:
                parts = re.split(r'\n\s*\n', s)
                expanded.extend(parts)
            # Filter empty
            return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 5]
        
        sentences = split_sentences(response_clean)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 1.0
        
        # ---- Feature 1: Discourse Marker Density and Variety ----
        # Categorize discourse markers by function
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bfor this reason\b', r'\bit follows that\b', r'\baccordingly\b'
        ]
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bdespite\b', r'\bwhile\b', r'\byet\b', r'\binstead\b',
            r'\bnonetheless\b', r'\beven though\b', r'\bwhereas\b'
        ]
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\blikewise\b', r'\bsimilarly\b',
            r'\bnot only\b', r'\bbesides\b', r'\bwhat\'s more\b'
        ]
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b', r'\bafterward\b',
            r'\bbefore\b', r'\bafter\b', r'\bmeanwhile\b', r'\binitially\b',
            r'\blastly\b', r'\bto begin\b', r'\bto start\b'
        ]
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b', r'\bgiven that\b'
        ]
        concluding_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bto sum up\b', r'\ball in all\b',
            r'\bin essence\b', r'\bultimately\b'
        ]
        
        response_lower = response_clean.lower()
        
        def count_markers(patterns, text):
            total = 0
            unique = 0
            for p in patterns:
                matches = re.findall(p, text)
                if matches:
                    unique += 1
                    total += len(matches)
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers, response_lower)
        contrast_count, contrast_unique = count_markers(contrastive_markers, response_lower)
        additive_count, additive_unique = count_markers(additive_markers, response_lower)
        temporal_count, temporal_unique = count_markers(temporal_markers, response_lower)
        conditional_count, conditional_unique = count_markers(conditional_markers, response_lower)
        concluding_count, concluding_unique = count_markers(concluding_markers, response_lower)
        
        total_markers = causal_count + contrast_count + additive_count + temporal_count + conditional_count + concluding_count
        total_unique_categories = sum(1 for x in [causal_unique, contrast_unique, additive_unique, temporal_unique, conditional_unique, concluding_unique] if x > 0)
        total_unique_markers = causal_unique + contrast_unique + additive_unique + temporal_unique + conditional_unique + concluding_unique
        
        # Marker density per sentence
        marker_density = total_markers / max(num_sentences, 1)
        # Optimal density is around 0.3-0.8 markers per sentence
        if marker_density <= 0.8:
            density_score = min(marker_density / 0.5, 1.0)
        else:
            density_score = max(0.5, 1.0 - (marker_density - 0.8) * 0.3)
        
        # Category variety bonus (using multiple types of connectors shows structured reasoning)
        variety_score = min(total_unique_categories / 4.0, 1.0)
        
        # Unique marker diversity
        marker_diversity_score = min(total_unique_markers / 8.0, 1.0)
        
        discourse_score = 0.4 * density_score + 0.35 * variety_score + 0.25 * marker_diversity_score
        
        # ---- Feature 2: Sentence-to-Sentence Coherence via TF cosine similarity ----
        def tokenize_words(text):
            return re.findall(r'[a-z]+', text.lower())
        
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                     'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                     'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                     'our', 'their', 'and', 'or', 'but', 'not', 'no', 'so', 'if', 'than',
                     'too', 'very', 'just', 'about', 'up', 'out', 'all', 'also', 'more',
                     'some', 'any', 'each', 'which', 'who', 'whom', 'what', 'when', 'where',
                     'how', 'there', 'here', 'then', 'now', 'only', 'own', 'same', 'other',
                     'such', 'both', 'over', 'after', 'before', 'between', 'under', 'again'}
        
        def get_tf_vector(text):
            words = [w for w in tokenize_words(text) if w not in stopwords and len(w) > 2]
            return Counter(words)
        
        def cosine_sim(vec1, vec2):
            if not vec1 or not vec2:
                return 0.0
            all_keys = set(vec1.keys()) | set(vec2.keys())
            dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
            mag1 = math.sqrt(sum(v*v for v in vec1.values()))
            mag2 = math.sqrt(sum(v*v for v in vec2.values()))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)
        
        # Compute pairwise consecutive sentence similarity
        coherence_scores = []
        if num_sentences >= 2:
            sent_vectors = [get_tf_vector(s) for s in sentences]
            for i in range(len(sent_vectors) - 1):
                sim = cosine_sim(sent_vectors[i], sent_vectors[i+1])
                coherence_scores.append(sim)
            
            # Also compute similarity with a sliding window of 3
            if len(sent_vectors) >= 3:
                for i in range(len(sent_vectors) - 2):
                    sim = cosine_sim(sent_vectors[i], sent_vectors[i+2])
                    coherence_scores.append(sim * 0.5)  # weighted less
        
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            # Check for coherence drops (sudden topic shifts)
            consecutive_only = []
            if num_sentences >= 2:
                sent_vectors = [get_tf_vector(s) for s in sentences]
                for i in range(len(sent_vectors) - 1):
                    consecutive_only.append(cosine_sim(sent_vectors[i], sent_vectors[i+1]))
            
            if consecutive_only:
                min_coherence = min(consecutive_only)
                # Variance in coherence - lower variance = more consistent flow
                mean_c = sum(consecutive_only) / len(consecutive_only)
                variance_c = sum((c - mean_c)**2 for c in consecutive_only) / len(consecutive_only)
                stability = 1.0 / (1.0 + variance_c * 10)
            else:
                min_coherence = 0.0
                stability = 0.5
            
            # Combine: average coherence, minimum coherence, stability
            coherence_feature = 0.5 * min(avg_coherence * 3, 1.0) + 0.25 * min(min_coherence * 4, 1.0) + 0.25 * stability
        else:
            coherence_feature = 0.3  # single sentence, neutral
        
        # ---- Feature 3: Argument Depth and Nested Reasoning ----
        # Look for reasoning chains: premise -> intermediate -> conclusion patterns
        reasoning_patterns = [
            r'\bif\b.{5,80}\bthen\b',
            r'\bnot only\b.{5,80}\bbut also\b',
            r'\bwhile\b.{5,80}\b(however|nevertheless|still)\b',
            r'\b(because|since)\b.{5,100}\b(therefore|thus|so|hence)\b',
            r'\b(first|to begin)\b.{5,200}\b(second|next|then)\b',
            r'\b(on one hand|on the one hand)\b.{5,200}\b(on the other|conversely)\b',
        ]
        
        reasoning_chain_count = 0
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response_lower, re.DOTALL)
            reasoning_chain_count += len(matches)
        
        # Nested clauses (subordination depth indicator)
        subordinators = re.findall(r'\b(which|that|who|whom|whose|where|when|because|although|while|if|unless|since|after|before)\b', response_lower)
        subordination_density = len(subordinators) / max(num_sentences, 1)
        
        # Optimal subordination: 1-3 per sentence shows complex but clear reasoning
        if subordination_density <= 2.5:
            sub_score = min(subordination_density / 1.5, 1.0)
        else:
            sub_score = max(0.3, 1.0 - (subordination_density - 2.5) * 0.2)
        
        chain_score = min(reasoning_chain_count / 3.0, 1.0)
        
        argument_depth = 0.5 * chain_score + 0.5 * sub_score
        
        # ---- Feature 4: Contradiction Detection ----
        # Simple contradiction signals
        contradiction_patterns = [
            r'\b(is|are|was|were)\b.{1,30}\b(is not|are not|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\balways\b.{1,50}\bnever\b',
            r'\bnever\b.{1,50}\balways\b',
            r'\bimpossible\b.{1,50}\bpossible\b',
            r'\bcannot\b.{1,50}\bcan\b',
        ]
        
        contradiction_count = 0
        for pattern in contradiction_patterns:
            # Only flag if within same sentence or adjacent sentences
            for sent in sentences:
                matches = re.findall(pattern, sent.lower())
                contradiction_count += len(matches)
        
        # Penalize contradictions
        contradiction_penalty = min(contradiction_count * 0.15, 0.5)
        
        # ---- Feature 5: Structural Organization ----
        # Check for enumeration/numbering (indicates organized thinking)
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*|-|•)\s', response_clean)
        has_enumeration = len(numbered_items) >= 2
        
        # Check for structural markers (headers, bold text, sections)
        structural_markers = re.findall(r'(?:^|\n)\s*(?:#{1,4}\s|(?:\*\*|__).+(?:\*\*|__))', response_clean)
        has_structure = len(structural_markers) >= 1
        
        # Check for introduction and conclusion signals
        has_intro = bool(re.search(r'^.{0,200}(here are|let me|i\'ll|let\'s|there are several|the following)', response_lower))
        has_conclusion = bool(re.search(r'(in conclusion|overall|to summarize|in summary|remember|keep in mind|the key|most importantly).{0,100}$', response_lower))
        
        # Paragraph structure
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response_clean) if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Score paragraph balance (are paragraphs roughly similar in length?)
        if num_paragraphs >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            if avg_para_len > 0:
                para_cv = math.sqrt(sum((l - avg_para_len)**2 for l in para_lengths) / len(para_lengths)) / avg_para_len
                para_balance = 1.0 / (1.0 + para_cv)
            else:
                para_balance = 0.3
        else:
            para_balance = 0.4
        
        structure_score = (
            0.2 * (1.0 if has_enumeration else 0.3) +
            0.15 * (1.0 if has_structure else 0.3) +
            0.15 * (1.0 if has_intro else 0.3) +
            0.15 * (1.0 if has_conclusion else 0.3) +
            0.35 * para_balance
        )
        
        # ---- Feature 6: Topic Consistency via Segment Entropy ----
        # Split response into thirds and check topic consistency
        words_all = tokenize_words(response_lower)
        content_words = [w for w in words_all if w not in stopwords and len(w) > 2]
        
        if len(content_words) >= 9:
            third = len(content_words) // 3
            segments = [
                Counter(content_words[:third]),
                Counter(content_words[third:2*third]),
                Counter(content_words[2*third:])
            ]
            
            # Cross-segment similarity
            seg_sims = []
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    seg_sims.append(cosine_sim(segments[i], segments[j]))
            
            topic_consistency = sum(seg_sims) / max(len(seg_sims), 1)
            topic_score = min(topic_consistency * 2.5, 1.0)
        else:
            topic_score = 0.5  # too short to meaningfully assess
        
        # ---- Feature 7: Query Relevance ----
        query_words = set(tokenize_words(query.lower())) - stopwords
        response_content = set(content_words)
        
        if query_words:
            relevance = len(query_words & response_content) / len(query_words)
        else:
            relevance = 0.5
        
        relevance_score = min(relevance * 1.5, 1.0)
        
        # ---- Feature 8: Sentence Length Variation (natural writing) ----
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) >= 2:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            if avg_sent_len > 0:
                sent_cv = math.sqrt(sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths)) / avg_sent_len
                # Moderate variation (CV 0.3-0.7) is ideal for natural writing
                if sent_cv < 0.2:
                    variation_score = 0.4  # too uniform, robotic
                elif sent_cv <= 0.7:
                    variation_score = 0.7 + 0.3 * ((sent_cv - 0.2) / 0.5)
                else:
                    variation_score = max(0.3, 1.0 - (sent_cv - 0.7) * 0.5)
            else:
                variation_score = 0.3
        else:
            variation_score = 0.4
        
        # ---- Feature 9: Completeness Signal ----
        # Check if response seems truncated
        last_chars = response_clean[-5:] if len(response_clean) >= 5 else response_clean
        seems_truncated = not bool(re.search(r'[.!?"\)}\]]$', last_chars.strip()))
        truncation_penalty = 0.15 if seems_truncated else 0.0
        
        # Check if response has sufficient length relative to query complexity
        query_word_count = len(query.split())
        response_word_count = len(response_clean.split())
        
        # Longer queries typically need longer responses
        if query_word_count > 20:
            length_adequacy = min(response_word_count / 80, 1.0)
        else:
            length_adequacy = min(response_word_count / 50, 1.0)
        
        # ---- Combine all features ----
        # Weighted combination emphasizing logical coherence features
        raw_score = (
            2.0 * discourse_score +        # Discourse markers (logical connectors)
            2.0 * coherence_feature +       # Sentence-to-sentence coherence
            1.5 * argument_depth +          # Reasoning depth
            1.5 * structure_score +         # Organizational structure
            1.0 * topic_score +             # Topic consistency
            0.8 * relevance_score +         # Query relevance
            0.5 * variation_score +         # Natural writing variation
            0.7 * length_adequacy           # Adequate response length
        )
        
        # Apply penalties
        raw_score -= contradiction_penalty * 10
        raw_score -= truncation_penalty * 10
        
        # Normalize to 0-10 range
        max_possible = 2.0 + 2.0 + 1.5 + 1.5 + 1.0 + 0.8 + 0.5 + 0.7  # = 10.0
        
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 20:
                return 3.0
            return 1.0
        except:
            return 1.0