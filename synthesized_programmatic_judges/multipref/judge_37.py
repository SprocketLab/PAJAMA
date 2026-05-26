def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectives)
    - Sentence-level coherence via topic continuity (entity/noun tracking across sentences)
    - Argument density and depth scoring
    - Structural balance and proportion analysis
    - Redundancy/repetition penalty via sentence-level similarity using character n-grams
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.0
        
        # ============================================================
        # 1. DISCOURSE MARKER ANALYSIS
        # Score based on presence and variety of logical connectives
        # ============================================================
        
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\bthis means\b', r'\bthis leads to\b',
            r'\bcaused by\b', r'\bresulting in\b', r'\bwhich means\b',
            r'\baccordingly\b', r'\bowing to\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\byet\b', r'\beven though\b',
            r'\bnonetheless\b', r'\bconversely\b', r'\bon the contrary\b',
            r'\binstead\b', r'\brather than\b'
        ]
        
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bnot only\b', r'\bas well as\b',
            r'\blikewise\b', r'\bsimilarly\b', r'\bequally\b'
        ]
        
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bthen\b',
            r'\bnext\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bbefore\b', r'\bafter\b',
            r'\binitially\b', r'\beventually\b', r'\bultimately\b',
            r'\bto begin\b', r'\bto start\b', r'\blast\b', r'\blastly\b'
        ]
        
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bin other words\b', r'\bto clarify\b'
        ]
        
        summary_markers = [
            r'\bin summary\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto summarize\b', r'\bin short\b', r'\ball in all\b',
            r'\bto sum up\b', r'\bin brief\b', r'\btaken together\b'
        ]
        
        resp_lower = response_stripped.lower()
        
        def count_marker_types(markers):
            types_found = set()
            total_count = 0
            for pattern in markers:
                matches = re.findall(pattern, resp_lower)
                if matches:
                    types_found.add(pattern)
                    total_count += len(matches)
            return len(types_found), total_count
        
        causal_types, causal_count = count_marker_types(causal_markers)
        contrast_types, contrast_count = count_marker_types(contrastive_markers)
        additive_types, additive_count = count_marker_types(additive_markers)
        temporal_types, temporal_count = count_marker_types(temporal_markers)
        elaboration_types, elaboration_count = count_marker_types(elaboration_markers)
        summary_types, summary_count = count_marker_types(summary_markers)
        
        # Category diversity (how many different categories are used)
        categories_used = sum(1 for t in [causal_types, contrast_types, additive_types,
                                           temporal_types, elaboration_types, summary_types] if t > 0)
        
        # Total unique marker types
        total_marker_types = (causal_types + contrast_types + additive_types +
                              temporal_types + elaboration_types + summary_types)
        
        total_marker_count = (causal_count + contrast_count + additive_count +
                              temporal_count + elaboration_count + summary_count)
        
        # Discourse marker score: reward variety and density
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        marker_density = total_marker_count / max(word_count, 1) * 100  # per 100 words
        
        # Score: variety of categories (0-6) * 2 + unique types (capped) + density bonus
        discourse_score = (
            categories_used * 2.5 +
            min(total_marker_types, 15) * 0.8 +
            min(marker_density, 8) * 1.5
        )
        # Bonus for causal reasoning markers specifically (key for logical coherence)
        discourse_score += min(causal_types, 5) * 1.5
        
        # ============================================================
        # 2. SENTENCE-LEVEL TOPIC CONTINUITY
        # Track content words across consecutive sentences
        # ============================================================
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response_stripped)
        # Also split on newlines that seem to separate ideas
        expanded_sentences = []
        for s in sentences:
            parts = re.split(r'\n\s*\n', s)
            for p in parts:
                p = p.strip()
                if len(p) > 5:
                    expanded_sentences.append(p)
        
        if not expanded_sentences:
            expanded_sentences = [response_stripped]
        
        num_sentences = len(expanded_sentences)
        
        # Extract content words (nouns, verbs, adjectives - approximated by filtering stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'just', 'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that',
            'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'she', 'him', 'her', 'his', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'these', 'those', 'am', 'about', 'up', 'down',
            'also', 'like', 'well', 'much', 'many', 'any', 'get', 'got', 'make',
            'made', 'go', 'going', 'come', 'take', 'know', 'see', 'think', 'say',
            'said', 'one', 'two', 'three', 'new', 'good', 'great', 'right', 'still',
            're', 've', 'll', 'let', 'us', 'way', 'even', 'back', 'thing', 'things',
            'really', 'want', 'give', 'look', 'use', 'find', 'tell', 'ask', 'work',
            'seem', 'feel', 'try', 'leave', 'call', 'keep', 'put', 'show', 'turn',
            'start', 'help', 'become', 'begin', 'run', 'move', 'live', 'play',
            'able', 'may', 'might', 'must', 'need', 'want', 'able', 'sure'
        }
        
        def get_content_words(text):
            text_lower = text.lower()
            tokens = re.findall(r'[a-z]{3,}', text_lower)
            return set(t for t in tokens if t not in stopwords)
        
        # Compute topic continuity: overlap between consecutive sentences
        continuity_scores = []
        sentence_content = [get_content_words(s) for s in expanded_sentences]
        
        for i in range(1, len(sentence_content)):
            prev_words = sentence_content[i - 1]
            curr_words = sentence_content[i]
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words)
                union = len(prev_words | curr_words)
                if union > 0:
                    continuity_scores.append(overlap / union)
                else:
                    continuity_scores.append(0)
            else:
                continuity_scores.append(0)
        
        # Also check 2-sentence window for broader coherence
        broad_continuity = []
        for i in range(2, len(sentence_content)):
            combined_prev = sentence_content[i - 1] | sentence_content[i - 2]
            curr_words = sentence_content[i]
            if combined_prev and curr_words:
                overlap = len(combined_prev & curr_words)
                union = len(combined_prev | curr_words)
                if union > 0:
                    broad_continuity.append(overlap / union)
        
        avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.3
        avg_broad_continuity = sum(broad_continuity) / len(broad_continuity) if broad_continuity else 0.3
        
        # Score: higher continuity = better coherence (but not too high = repetitive)
        # Sweet spot around 0.15-0.40
        continuity_combined = (avg_continuity * 0.6 + avg_broad_continuity * 0.4)
        if continuity_combined > 0.5:
            # Penalize excessive repetition
            coherence_score = 12 - (continuity_combined - 0.5) * 10
        else:
            coherence_score = min(continuity_combined / 0.35, 1.0) * 12
        coherence_score = max(0, coherence_score)
        
        # ============================================================
        # 3. ARGUMENT DEPTH AND STRUCTURE
        # ============================================================
        
        # Check for numbered/lettered lists (indicates structured argument)
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*\*\d+)', response_stripped)
        step_patterns = re.findall(r'(?:step|phase|stage)\s*\d+', resp_lower)
        
        # Check for bold/emphasized headers (markdown structure)
        bold_headers = re.findall(r'\*\*[^*]+\*\*', response_stripped)
        hash_headers = re.findall(r'^#{1,4}\s+.+', response_stripped, re.MULTILINE)
        
        structural_elements = len(numbered_items) + len(step_patterns) + len(bold_headers) * 0.5 + len(hash_headers) * 0.7
        
        # Argument depth: look for multi-level reasoning
        conditional_patterns = re.findall(r'\bif\b.*?\bthen\b|\bwhen\b.*?\b(?:will|would|should)\b', resp_lower)
        comparison_patterns = re.findall(r'\bcompared to\b|\bmore\s+\w+\s+than\b|\bless\s+\w+\s+than\b|\bbetter\b|\bworse\b', resp_lower)
        qualification_patterns = re.findall(r'\balthough\b|\bwhile\b.*?\b(?:also|still)\b|\bdepending on\b|\bin some cases\b|\bnot always\b', resp_lower)
        
        depth_indicators = len(conditional_patterns) + len(comparison_patterns) + len(qualification_patterns)
        
        structure_score = (
            min(structural_elements, 12) * 1.0 +
            min(depth_indicators, 8) * 1.2
        )
        
        # ============================================================
        # 4. REDUNDANCY PENALTY via character trigram sentence similarity
        # ============================================================
        
        def char_trigrams(text):
            text = text.lower().strip()
            if len(text) < 3:
                return Counter()
            return Counter(text[i:i+3] for i in range(len(text) - 2))
        
        def trigram_similarity(t1, t2):
            c1 = char_trigrams(t1)
            c2 = char_trigrams(t2)
            if not c1 or not c2:
                return 0
            intersection = sum((c1 & c2).values())
            total = sum(c1.values()) + sum(c2.values())
            if total == 0:
                return 0
            return 2 * intersection / total  # Dice coefficient
        
        # Check for highly similar (near-duplicate) sentences
        redundancy_count = 0
        if num_sentences > 2:
            for i in range(len(expanded_sentences)):
                for j in range(i + 1, len(expanded_sentences)):
                    sim = trigram_similarity(expanded_sentences[i], expanded_sentences[j])
                    if sim > 0.7:
                        redundancy_count += 1
        
        redundancy_penalty = min(redundancy_count * 3, 15)
        
        # ============================================================
        # 5. INTERNAL CONTRADICTION DETECTION
        # ============================================================
        
        contradiction_indicators = 0
        
        # Look for direct contradictions: "X is Y" followed by "X is not Y" patterns
        affirmative_claims = re.findall(r'(\w+)\s+(?:is|are|was|were)\s+(\w+)', resp_lower)
        negative_claims = re.findall(r'(\w+)\s+(?:is|are|was|were)\s+not\s+(\w+)', resp_lower)
        
        for neg_subj, neg_pred in negative_claims:
            for aff_subj, aff_pred in affirmative_claims:
                if neg_subj == aff_subj and neg_pred == aff_pred:
                    contradiction_indicators += 1
        
        contradiction_penalty = contradiction_indicators * 4
        
        # ============================================================
        # 6. RESPONSE COMPLETENESS AND PROPORTION
        # ============================================================
        
        # Check if response appears truncated
        truncation_penalty = 0
        if response_stripped[-1] not in '.!?"\')':
            # Likely truncated
            truncation_penalty = 3
        
        # Check for balanced paragraph lengths (not one giant block)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response_stripped) if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            if avg_para_len > 0:
                para_variance = sum((l - avg_para_len) ** 2 for l in para_lengths) / len(para_lengths)
                para_cv = math.sqrt(para_variance) / avg_para_len  # coefficient of variation
                # Moderate variation is good (not all same length, not wildly different)
                if para_cv < 0.3:
                    balance_score = 4
                elif para_cv < 0.7:
                    balance_score = 5
                elif para_cv < 1.2:
                    balance_score = 3
                else:
                    balance_score = 1
            else:
                balance_score = 2
        else:
            # Single paragraph: less structured
            if word_count > 80:
                balance_score = 2  # Long single block is less organized
            else:
                balance_score = 3  # Short response, single para is fine
        
        # ============================================================
        # 7. QUERY RELEVANCE via content word alignment
        # ============================================================
        
        query_content = get_content_words(query)
        response_content = set()
        for sc in sentence_content:
            response_content |= sc
        
        if query_content:
            relevance_ratio = len(query_content & response_content) / len(query_content)
        else:
            relevance_ratio = 0.5
        
        relevance_score = min(relevance_ratio, 1.0) * 8
        
        # ============================================================
        # 8. OPENING QUALITY - Does it establish context/thesis?
        # ============================================================
        
        opening_score = 0
        if expanded_sentences:
            first_sent = expanded_sentences[0].lower()
            # Check if opening addresses the query
            if query_content:
                first_content = get_content_words(expanded_sentences[0])
                opening_overlap = len(query_content & first_content) / max(len(query_content), 1)
                opening_score += min(opening_overlap, 1.0) * 3
            
            # Check for confident opening (not hedging excessively)
            strong_openers = [r'\bcertainly\b', r'\babsolutely\b', r'\bhere\b', r'\blet\'s\b',
                              r'\bthe\b.*\bis\b', r'\bthere are\b', r'\byou can\b']
            for pattern in strong_openers:
                if re.search(pattern, first_sent):
                    opening_score += 0.5
                    break
        
        opening_score = min(opening_score, 5)
        
        # ============================================================
        # 9. SENTENCE LENGTH VARIETY (indicates rhetorical skill)
        # ============================================================
        
        sent_lengths = []
        for s in expanded_sentences:
            wc = len(s.split())
            if wc > 0:
                sent_lengths.append(wc)
        
        variety_score = 0
        if len(sent_lengths) > 2:
            avg_sl = sum(sent_lengths) / len(sent_lengths)
            if avg_sl > 0:
                sl_variance = sum((l - avg_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
                sl_cv = math.sqrt(sl_variance) / avg_sl
                # Some variety is good (0.3-0.8 CV)
                if 0.2 <= sl_cv <= 1.0:
                    variety_score = 4
                elif sl_cv < 0.2:
                    variety_score = 2  # Too uniform
                else:
                    variety_score = 2  # Too chaotic
        else:
            variety_score = 2
        
        # ============================================================
        # 10. LENGTH BONUS (longer, substantive responses tend to be better)
        # ============================================================
        
        # Logarithmic length bonus
        length_bonus = min(math.log(max(word_count, 1)) * 1.5, 10)
        
        # ============================================================
        # FINAL SCORE COMPUTATION
        # ============================================================
        
        raw_score = (
            discourse_score * 1.0 +      # up to ~30
            coherence_score * 1.0 +       # up to 12
            structure_score * 1.0 +        # up to ~22
            balance_score * 1.0 +          # up to 5
            relevance_score * 1.0 +        # up to 8
            opening_score * 1.0 +          # up to 5
            variety_score * 1.0 +          # up to 4
            length_bonus * 1.0 -           # up to 10
            redundancy_penalty -           # up to 15
            contradiction_penalty -        # variable
            truncation_penalty             # 0 or 3
        )
        
        # Normalize to 0-100 range
        # Theoretical max around 96, typical good response ~50-70
        final_score = max(0.0, min(100.0, raw_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Never crash - return neutral score
        try:
            if response and len(response.strip()) > 20:
                return 25.0
            return 0.0
        except:
            return 0.0