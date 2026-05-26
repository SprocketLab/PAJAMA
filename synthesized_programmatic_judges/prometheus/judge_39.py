def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using a discourse-graph approach.
    
    This variant builds a lightweight "discourse graph" by:
    1. Identifying discourse relations (causal, contrastive, elaborative, conditional)
    2. Measuring argument depth via nested reasoning markers
    3. Detecting logical fallacies and contradictions via negation patterns
    4. Scoring structural completeness (intro/body/conclusion pattern)
    5. Measuring referential coherence via pronoun/demonstrative chains
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 1.0
        
        if not isinstance(query, str) or not isinstance(response, str):
            return 1.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 20:
            return 1.0
        
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if len(sentences) == 0:
            return 1.0
        
        words = re.findall(r'\b\w+\b', response.lower())
        num_words = len(words)
        
        if num_words < 5:
            return 1.0
        
        # ============================================================
        # 1. DISCOURSE RELATION DENSITY (causal, contrastive, elaborative, conditional, temporal)
        # ============================================================
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bfor this reason\b',
            r'\bthat\'s why\b', r'\baccordingly\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhereas\b',
            r'\bdespite\b', r'\byet\b', r'\binstead\b', r'\bwhile\b',
            r'\bconversely\b', r'\beven though\b', r'\brather than\b'
        ]
        
        elaborative_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bmore specifically\b', r'\bto clarify\b', r'\bincluding\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bin case\b',
            r'\bassuming\b', r'\bwhen\b.*\bthen\b', r'\bshould you\b',
            r'\bwould\b.*\bif\b', r'\bgiven that\b', r'\bsuppose\b'
        ]
        
        temporal_markers = [
            r'\bfirst\b', r'\bthen\b', r'\bnext\b', r'\bafter\b',
            r'\bbefore\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bmeanwhile\b', r'\bonce\b', r'\bpreviously\b',
            r'\binitially\b', r'\beventually\b', r'\blastly\b'
        ]
        
        resp_lower = response.lower()
        
        def count_markers(patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, resp_lower))
            return total
        
        causal_count = count_markers(causal_markers)
        contrastive_count = count_markers(contrastive_markers)
        elaborative_count = count_markers(elaborative_markers)
        conditional_count = count_markers(conditional_markers)
        temporal_count = count_markers(temporal_markers)
        
        total_discourse = causal_count + contrastive_count + elaborative_count + conditional_count + temporal_count
        
        # Discourse density normalized by sentence count
        discourse_density = total_discourse / max(len(sentences), 1)
        # Reward variety of discourse types used
        discourse_types_used = sum([
            1 for c in [causal_count, contrastive_count, elaborative_count, conditional_count, temporal_count] if c > 0
        ])
        
        # Score: density (capped) + variety bonus
        discourse_score = min(discourse_density, 1.5) * 4.0 + discourse_types_used * 1.5
        discourse_score = min(discourse_score, 15.0)
        
        # ============================================================
        # 2. ARGUMENT DEPTH — nested reasoning chains
        # ============================================================
        # Look for chains like "because X, therefore Y" or multi-step reasoning
        depth_patterns = [
            r'\b(because|since).*\b(therefore|thus|so|hence)\b',
            r'\b(if|when).*\b(then|consequently)\b',
            r'\b(not only).*\b(but also)\b',
            r'\b(first|initially).*\b(then|next).*\b(finally|lastly)\b',
            r'\b(on one hand).*\b(on the other)\b',
            r'\b(while|although).*\b(still|nevertheless)\b',
        ]
        
        depth_count = 0
        for pat in depth_patterns:
            if re.search(pat, resp_lower, re.DOTALL):
                depth_count += 1
        
        # Also check for multi-sentence reasoning chains
        # (sentence i contains a claim, sentence i+1 provides support)
        support_starters = re.compile(r'^(this is because|this means|as a result|therefore|thus|hence|consequently|in fact|indeed|moreover|furthermore)', re.IGNORECASE)
        chain_count = 0
        for i in range(1, len(sentences)):
            if support_starters.match(sentences[i].strip()):
                chain_count += 1
        
        depth_score = min(depth_count * 2.0 + chain_count * 1.5, 10.0)
        
        # ============================================================
        # 3. CONTRADICTION / INCOHERENCE DETECTION
        # ============================================================
        # Detect potential contradictions via opposing sentiment patterns in close proximity
        contradiction_penalty = 0.0
        
        positive_phrases = [
            r'\bis good\b', r'\bis great\b', r'\bshould\b', r'\brecommend\b',
            r'\bimportant\b', r'\bhelpful\b', r'\bbeneficial\b', r'\beffective\b'
        ]
        negative_phrases = [
            r'\bis bad\b', r'\bis terrible\b', r'\bshould not\b', r'\bdon\'t recommend\b',
            r'\bunimportant\b', r'\bunhelpful\b', r'\bharmful\b', r'\bineffective\b',
            r'\bwon\'t work\b', r'\buseless\b'
        ]
        
        # Check for hedging that undermines previous assertions
        undermining_patterns = [
            r'\bbut (i\'m not sure|i don\'t know|who knows|it might not)\b',
            r'\bprobably won\'t (work|help|matter)\b',
            r'\bmight not be\b.*\bactually\b',
        ]
        
        for pat in undermining_patterns:
            if re.search(pat, resp_lower):
                contradiction_penalty += 2.0
        
        # Detect "can't"/"won't"/"might not" heavy responses (indicating uncertainty/inability)
        inability_markers = re.findall(r'\b(can\'t|cannot|won\'t|unable|might not|may not|probably not)\b', resp_lower)
        inability_ratio = len(inability_markers) / max(num_words, 1)
        if inability_ratio > 0.03:
            contradiction_penalty += min(inability_ratio * 50, 5.0)
        
        contradiction_penalty = min(contradiction_penalty, 8.0)
        
        # ============================================================
        # 4. STRUCTURAL COMPLETENESS (intro-body-conclusion pattern)
        # ============================================================
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 10]
        if len(paragraphs) == 0:
            paragraphs = [p.strip() for p in response.split('\n') if len(p.strip()) > 10]
        
        structural_score = 0.0
        
        # Check for opening acknowledgment/framing
        opening_patterns = [
            r'^(i understand|i can see|it\'s|that\'s|imagine|let\'s|here|to)',
            r'^(i\'m sorry|i hear|it sounds|absolutely|certainly|of course)',
            r'^(great question|good point|hey|alright|ok so)',
        ]
        has_opening = any(re.match(p, resp_lower) for p in opening_patterns)
        if has_opening:
            structural_score += 2.0
        
        # Check for concluding/summarizing markers
        concluding_patterns = [
            r'\b(in summary|to summarize|in conclusion|overall|to wrap up)\b',
            r'\b(remember|keep in mind|the key|most importantly)\b',
            r'\b(don\'t hesitate|feel free|good luck|hope this helps)\b',
            r'\b(all in all|at the end of the day|the bottom line)\b',
        ]
        has_conclusion = any(re.search(p, resp_lower) for p in concluding_patterns)
        if has_conclusion:
            structural_score += 2.0
        
        # Multi-paragraph bonus (shows organized thinking)
        if len(paragraphs) >= 2:
            structural_score += 1.5
        if len(paragraphs) >= 3:
            structural_score += 1.0
        
        # Numbered/lettered list detection (shows structured argumentation)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        lettered_items = re.findall(r'(?:^|\n)\s*[a-z][\.\)]\s', response)
        if len(numbered_items) >= 2 or len(lettered_items) >= 2:
            structural_score += 2.0
        
        structural_score = min(structural_score, 8.0)
        
        # ============================================================
        # 5. REFERENTIAL COHERENCE — pronoun/demonstrative chains
        # ============================================================
        # Good coherent text uses pronouns and demonstratives that refer back
        # to previously mentioned concepts, creating a cohesive chain
        
        referential_words = re.findall(r'\b(this|that|these|those|it|they|them|their|its|such|the same|aforementioned)\b', resp_lower)
        referential_density = len(referential_words) / max(num_words, 1)
        
        # Optimal range: 0.03-0.10 (too few = disconnected, too many = vague)
        if 0.03 <= referential_density <= 0.10:
            referential_score = 4.0
        elif 0.02 <= referential_density < 0.03 or 0.10 < referential_density <= 0.15:
            referential_score = 2.5
        elif referential_density < 0.02:
            referential_score = 1.0
        else:
            referential_score = 1.5
        
        # ============================================================
        # 6. SENTENCE-LEVEL COHERENCE — topic continuity between adjacent sentences
        # ============================================================
        # Measure content word overlap between consecutive sentences
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'if', 'that', 'this',
                'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what',
                'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'each',
                'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                'than', 'too', 'very', 'just', 'about', 'above', 'after', 'again',
                'also', 'am', 'any', 'because', 'before', 'between', 'down', 'here',
                'there', 'up', 'out', 'over', 'own', 'same', 'then', 'these', 'those',
            }
            w = re.findall(r'\b[a-z]+\b', text.lower())
            return set(w2 for w2 in w if w2 not in stop_words and len(w2) > 2)
        
        if len(sentences) >= 2:
            continuity_scores = []
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if len(words_a) == 0 or len(words_b) == 0:
                    continuity_scores.append(0.0)
                else:
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    # Use a modified overlap coefficient (not Jaccard)
                    # overlap / min(len_a, len_b) — Szymkiewicz–Simpson coefficient
                    min_size = min(len(words_a), len(words_b))
                    if min_size > 0:
                        continuity_scores.append(overlap / min_size)
                    else:
                        continuity_scores.append(0.0)
            
            avg_continuity = sum(continuity_scores) / max(len(continuity_scores), 1)
            # Also check variance — low variance means consistent coherence
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                var_c = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                consistency_bonus = max(0, 1.0 - math.sqrt(var_c) * 3)
            else:
                consistency_bonus = 0.5
            
            continuity_score = avg_continuity * 6.0 + consistency_bonus * 2.0
        else:
            continuity_score = 2.0
        
        continuity_score = min(continuity_score, 8.0)
        
        # ============================================================
        # 7. QUERY RELEVANCE — does the response actually address the query?
        # ============================================================
        query_content = get_content_words(query)
        response_content = get_content_words(response)
        
        if len(query_content) > 0 and len(response_content) > 0:
            query_coverage = len(query_content & response_content) / max(len(query_content), 1)
        else:
            query_coverage = 0.0
        
        relevance_score = min(query_coverage * 10.0, 6.0)
        
        # ============================================================
        # 8. RESPONSE LENGTH AND SUBSTANCE
        # ============================================================
        # Longer, more substantive responses tend to have better argumentation
        # but penalize extremely short responses
        if num_words < 30:
            length_score = 0.5
        elif num_words < 60:
            length_score = 2.0
        elif num_words < 120:
            length_score = 3.5
        elif num_words < 250:
            length_score = 4.0
        else:
            length_score = 4.5
        
        length_score = min(length_score, 5.0)
        
        # ============================================================
        # 9. EMPATHETIC / APPROPRIATE TONE ALIGNMENT
        # ============================================================
        # For emotional queries, check if response acknowledges emotions
        emotional_query_words = re.findall(r'\b(feel|feeling|emotion|stress|frustrat|sad|angry|upset|worried|anxious|heartbroken|devastat|loneli|despair|exhaust|down|struggle)\b', query.lower())
        
        tone_score = 0.0
        if len(emotional_query_words) > 0:
            empathy_markers = re.findall(r'\b(understand|sorry|hear|feel|natural|okay|ok|valid|normal|completely|absolutely|tough|difficult|hard|support|care|comfort)\b', resp_lower)
            if len(empathy_markers) >= 3:
                tone_score = 3.0
            elif len(empathy_markers) >= 1:
                tone_score = 1.5
            else:
                tone_score = -1.0  # Penalty for ignoring emotional context
            
            # Check for dismissive language
            dismissive = re.findall(r'\b(just get over|move on|stop|don\'t worry about it|it\'s nothing|no big deal|get yourself together)\b', resp_lower)
            if dismissive:
                tone_score -= 2.0
        else:
            tone_score = 1.0  # Neutral baseline for non-emotional queries
        
        tone_score = max(tone_score, -3.0)
        
        # ============================================================
        # 10. ACTIONABILITY — concrete advice/steps
        # ============================================================
        action_patterns = [
            r'\b(try|start|begin|consider|make sure|ensure|remember|keep|take|ask|seek|reach out|practice|focus)\b',
            r'\b(step \d|first|second|third|here\'s how|here are|you can|you could|you might)\b',
        ]
        
        action_count = 0
        for pat in action_patterns:
            action_count += len(re.findall(pat, resp_lower))
        
        actionability_score = min(action_count * 0.5, 5.0)
        
        # ============================================================
        # FINAL COMPOSITE SCORE
        # ============================================================
        raw_score = (
            discourse_score * 0.18 +       # max ~2.7
            depth_score * 0.10 +            # max ~1.0
            structural_score * 0.14 +       # max ~1.12
            referential_score * 0.08 +      # max ~0.32
            continuity_score * 0.12 +       # max ~0.96
            relevance_score * 0.12 +        # max ~0.72
            length_score * 0.08 +           # max ~0.40
            tone_score * 0.10 +             # max ~0.30
            actionability_score * 0.08 -    # max ~0.40
            contradiction_penalty * 0.10    # max penalty ~0.80
        )
        
        # Normalize to 1-5 scale
        # Theoretical max around 7.5, typical range 1-5
        final_score = 1.0 + (raw_score / 7.5) * 4.0
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        # Round to one decimal
        final_score = round(final_score, 1)
        
        return final_score
        
    except Exception:
        return 2.5