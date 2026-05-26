def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis:
    - Causal/logical connective density and proper usage
    - Argument chain detection (claim → evidence → conclusion patterns)
    - Sentence-to-sentence semantic progression (topic continuity vs drift)
    - Contradiction detection via negation pattern analysis
    - Structural completeness (intro/body/conclusion signals)
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not query:
            return 1.0

        response = response.strip()
        query = query.strip()

        if len(response) < 20:
            return 1.0

        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(sentences) == 0:
            return 1.0

        # ---- Feature 1: Causal/Logical Connective Analysis ----
        # Categorize connectives by their logical function
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bleading to\b', r'\bcaused by\b', r'\bso that\b',
            r'\bhence\b', r'\baccordingly\b', r'\bfor this reason\b'
        ]
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bwhile\b', r'\byet\b',
            r'\bdespite\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bnonetheless\b', r'\beven though\b', r'\brather\b'
        ]
        additive_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bwhat\'s more\b', r'\bbesides\b'
        ]
        conditional_connectives = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b',
            r'\bin case\b', r'\bassuming\b', r'\bwhen\b',
            r'\bwhenever\b', r'\bas long as\b'
        ]
        concluding_connectives = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\bultimately\b', r'\bin short\b',
            r'\ball in all\b', r'\bto sum up\b', r'\bfinally\b'
        ]

        resp_lower = response.lower()

        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total

        causal_count = count_patterns(causal_connectives, resp_lower)
        contrastive_count = count_patterns(contrastive_connectives, resp_lower)
        additive_count = count_patterns(additive_connectives, resp_lower)
        conditional_count = count_patterns(conditional_connectives, resp_lower)
        concluding_count = count_patterns(concluding_connectives, resp_lower)

        total_connectives = causal_count + contrastive_count + additive_count + conditional_count + concluding_count

        # Connective diversity: how many different categories are used
        categories_used = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                           conditional_count, concluding_count] if c > 0)

        # Normalize by sentence count
        connective_density = total_connectives / max(len(sentences), 1)
        # Ideal density is around 0.3-0.6 per sentence
        connective_score = min(connective_density / 0.5, 1.0) * 0.6 + (categories_used / 5.0) * 0.4

        # ---- Feature 2: Argument Chain Detection ----
        # Look for claim-evidence-conclusion patterns
        claim_indicators = [
            r'\bi believe\b', r'\bit is\b', r'\bthis is\b', r'\bthe key\b',
            r'\bimportant(ly)?\b', r'\bessential(ly)?\b', r'\bcrucial(ly)?\b',
            r'\bshould\b', r'\bmust\b', r'\bneed to\b', r'\bcan\b'
        ]
        evidence_indicators = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\blike\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bconsider\b', r'\bimagine\b', r'\bsuppose\b',
            r'\bresearch\b', r'\bstudies\b', r'\bdata\b', r'\bevidence\b'
        ]
        elaboration_indicators = [
            r'\bthis means\b', r'\bin other words\b', r'\bthat is\b',
            r'\bput (simply|differently)\b', r'\bessentially\b',
            r'\bto clarify\b', r'\bto explain\b', r'\bwhat this means\b'
        ]

        claim_count = count_patterns(claim_indicators, resp_lower)
        evidence_count = count_patterns(evidence_indicators, resp_lower)
        elaboration_count = count_patterns(elaboration_indicators, resp_lower)

        # Good arguments have a mix of claims and evidence
        argument_elements = min(claim_count, 5) + min(evidence_count, 3) * 1.5 + min(elaboration_count, 3) * 1.2
        argument_score = min(argument_elements / 8.0, 1.0)

        # ---- Feature 3: Sentence-to-Sentence Topic Continuity ----
        # Measure how well consecutive sentences connect via shared content words
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
                'or', 'if', 'while', 'that', 'this', 'it', 'its', 'i', 'you', 'your',
                'we', 'they', 'them', 'their', 'he', 'she', 'his', 'her', 'my', 'our',
                'me', 'him', 'us', 'what', 'which', 'who', 'whom'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)

        if len(sentences) >= 2:
            continuity_scores = []
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if words_a and words_b:
                    # Use overlap coefficient (not Jaccard - that was used before)
                    overlap = len(words_a & words_b)
                    min_size = min(len(words_a), len(words_b))
                    overlap_coeff = overlap / max(min_size, 1)
                    continuity_scores.append(overlap_coeff)
                else:
                    continuity_scores.append(0.0)

            avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0
            # Also check for smooth progression (low variance = consistent flow)
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                variance_c = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                std_c = math.sqrt(variance_c)
                # Penalize high variance (indicates topic jumping)
                consistency_bonus = max(0, 1.0 - std_c * 2)
            else:
                consistency_bonus = 0.5

            continuity_score = avg_continuity * 0.6 + consistency_bonus * 0.4
        else:
            continuity_score = 0.3

        # ---- Feature 4: Contradiction/Negation Pattern Analysis ----
        # Detect potential internal contradictions
        negation_patterns = [
            r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bnone\b',
            r'\bdon\'t\b', r'\bdoesn\'t\b', r'\bwon\'t\b', r'\bcan\'t\b',
            r'\bcannot\b', r'\bshouldn\'t\b', r'\bwouldn\'t\b', r'\bisn\'t\b',
            r'\baren\'t\b', r'\bwasn\'t\b', r'\bweren\'t\b', r'\bhaven\'t\b',
            r'\bhasn\'t\b', r'\bnothing\b', r'\bnowhere\b'
        ]

        # Check for "X is Y" followed by "X is not Y" type contradictions
        contradiction_penalty = 0.0

        # Look for sentences that negate content from previous sentences
        for i in range(len(sentences)):
            for j in range(i + 1, min(i + 4, len(sentences))):
                sent_i_words = get_content_words(sentences[i])
                sent_j_words = get_content_words(sentences[j])
                shared = sent_i_words & sent_j_words

                if len(shared) >= 2:
                    # Check if one has negation and the other doesn't for the same topic
                    neg_i = count_patterns(negation_patterns, sentences[i].lower())
                    neg_j = count_patterns(negation_patterns, sentences[j].lower())
                    if (neg_i > 0) != (neg_j > 0) and len(shared) >= 3:
                        contradiction_penalty += 0.1

        contradiction_penalty = min(contradiction_penalty, 0.5)

        # ---- Feature 5: Structural Completeness ----
        # Check for introduction, body, conclusion structure

        # Opening signals
        opening_patterns = [
            r'^(i |let|to |first|imagine|picture|think|consider|welcome|hello|hi |hey)',
            r'^(it\'s|it is|this is|here|we )',
            r'^(i\'m |i can |i understand|i hear|i see)'
        ]
        has_opening = any(re.match(p, resp_lower.strip()) for p in opening_patterns)

        # Closing signals
        closing_patterns = [
            r'\bremember\b.*[.!]$', r'\bgood luck\b', r'\bfeel free\b',
            r'\bdon\'t hesitate\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto summarize\b', r'\bhope this\b', r'\bwish you\b',
            r'\btake care\b', r'\ball the best\b', r'\bkeep\b.*\b[.!]$'
        ]
        has_closing = any(re.search(p, resp_lower) for p in closing_patterns)

        structure_score = 0.3  # base
        if has_opening:
            structure_score += 0.35
        if has_closing:
            structure_score += 0.35

        # ---- Feature 6: Progressive Development ----
        # Check if the response develops ideas progressively (new content in each sentence)
        if len(sentences) >= 3:
            seen_content = set()
            new_content_ratios = []
            for sent in sentences:
                words = get_content_words(sent)
                if words:
                    new_words = words - seen_content
                    ratio = len(new_words) / len(words)
                    new_content_ratios.append(ratio)
                    seen_content.update(words)

            if new_content_ratios:
                avg_new = sum(new_content_ratios) / len(new_content_ratios)
                # Good responses introduce new content while maintaining some overlap
                # Too much new content = topic drift, too little = repetition
                # Ideal is around 0.5-0.8
                if avg_new > 0.3:
                    development_score = min(avg_new / 0.7, 1.0)
                else:
                    development_score = avg_new / 0.3 * 0.5  # penalize heavy repetition
            else:
                development_score = 0.3
        else:
            development_score = 0.4

        # ---- Feature 7: Query Relevance via Topic Alignment ----
        query_content = get_content_words(query)
        response_content = get_content_words(response)

        if query_content and response_content:
            query_coverage = len(query_content & response_content) / max(len(query_content), 1)
            relevance_score = min(query_coverage * 2, 1.0)
        else:
            relevance_score = 0.3

        # ---- Feature 8: Sentence Complexity and Sophistication ----
        # Longer, well-formed sentences suggest more developed arguments
        avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        # Ideal sentence length: 12-25 words
        if 12 <= avg_sent_len <= 25:
            complexity_score = 1.0
        elif avg_sent_len < 12:
            complexity_score = avg_sent_len / 12.0
        else:
            complexity_score = max(0.5, 1.0 - (avg_sent_len - 25) / 30.0)

        # ---- Feature 9: Enumeration/Step structure (ordered reasoning) ----
        has_numbered = bool(re.search(r'\b[1-9]\.\s', response))
        has_ordered_words = bool(re.search(
            r'\b(first|second|third|next|then|finally|lastly|step)\b', resp_lower
        ))
        ordering_score = 0.0
        if has_numbered:
            ordering_score += 0.6
        if has_ordered_words:
            ordering_score += 0.4

        # ---- Feature 10: Empathy/Acknowledgment before advice (logical social flow) ----
        # In responses to emotional queries, good logical flow starts with acknowledgment
        emotional_query_words = ['feeling', 'frustrated', 'stressed', 'sad', 'upset',
                                  'worried', 'anxious', 'heartbroken', 'lonely', 'down',
                                  'struggling', 'difficult', 'hard time']
        is_emotional_query = any(w in query.lower() for w in emotional_query_words)

        acknowledgment_patterns = [
            r'\bi (can )?(see|hear|understand|imagine)\b',
            r'\bit\'s (completely |totally |perfectly )?(understandable|okay|ok|normal|natural)\b',
            r'\bi\'m (sorry|genuinely sorry)\b',
            r'\bthat\'s (tough|hard|difficult|understandable)\b',
            r'\byour feelings\b', r'\byou\'re feeling\b'
        ]

        if is_emotional_query:
            # Check if acknowledgment comes BEFORE advice
            first_ack_pos = len(resp_lower)
            for p in acknowledgment_patterns:
                m = re.search(p, resp_lower)
                if m:
                    first_ack_pos = min(first_ack_pos, m.start())

            advice_patterns = [r'\btry\b', r'\byou (should|could|might)\b', r'\bstart\b',
                               r'\bmake sure\b', r'\bremember to\b']
            first_advice_pos = len(resp_lower)
            for p in advice_patterns:
                m = re.search(p, resp_lower)
                if m:
                    first_advice_pos = min(first_advice_pos, m.start())

            if first_ack_pos < first_advice_pos and first_ack_pos < len(resp_lower) * 0.3:
                empathy_flow_score = 1.0
            elif first_ack_pos < len(resp_lower):
                empathy_flow_score = 0.5
            else:
                empathy_flow_score = 0.2
        else:
            empathy_flow_score = 0.6  # neutral for non-emotional queries

        # ---- Feature 11: Response Length Adequacy ----
        word_count = len(response.split())
        if word_count < 30:
            length_score = 0.3
        elif word_count < 60:
            length_score = 0.6
        elif word_count < 200:
            length_score = 1.0
        else:
            length_score = 0.8

        # ---- Weighted Combination ----
        weights = {
            'connective': 0.14,
            'argument': 0.12,
            'continuity': 0.12,
            'contradiction': 0.08,
            'structure': 0.10,
            'development': 0.08,
            'relevance': 0.10,
            'complexity': 0.06,
            'ordering': 0.05,
            'empathy_flow': 0.08,
            'length': 0.07,
        }

        raw_score = (
            weights['connective'] * connective_score +
            weights['argument'] * argument_score +
            weights['continuity'] * continuity_score -
            weights['contradiction'] * contradiction_penalty +
            weights['structure'] * structure_score +
            weights['development'] * development_score +
            weights['relevance'] * relevance_score +
            weights['complexity'] * complexity_score +
            weights['ordering'] * ordering_score +
            weights['empathy_flow'] * empathy_flow_score +
            weights['length'] * length_score
        )

        # Scale to 1-5 range
        # raw_score theoretical range: roughly -0.04 to 1.0
        # Practical range: 0.1 to 0.85
        final_score = 1.0 + raw_score * 5.0

        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, final_score))

        return round(final_score, 2)

    except Exception:
        return 3.0