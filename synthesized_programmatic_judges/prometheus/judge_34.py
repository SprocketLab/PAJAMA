def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    analysis of causal/logical connective chains, contradiction detection,
    and hierarchical structure assessment.
    
    This variant focuses on:
    1. Sentence-to-sentence semantic progression (topic continuity via noun tracking)
    2. Causal/logical connective chain analysis (depth of reasoning)
    3. Contradiction/inconsistency detection via negation patterns
    4. Discourse structure scoring (intro-body-conclusion pattern)
    5. Sentence complexity and subordination depth
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
        
        # Split response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) == 0:
            return 1.0
        
        # ---- Helper functions ----
        
        def extract_content_words(text):
            """Extract meaningful content words (nouns, verbs, adjectives approximation)."""
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some', 'them',
                'than', 'its', 'over', 'such', 'that', 'this', 'with', 'will', 'each',
                'from', 'they', 'were', 'which', 'their', 'said', 'what', 'when', 'who',
                'how', 'may', 'also', 'about', 'would', 'make', 'like', 'could', 'into',
                'just', 'very', 'your', 'more', 'other', 'should', 'being', 'there',
                'where', 'after', 'most', 'these', 'then', 'here', 'does', 'did', 'get',
                'got', 'might', 'still', 'much', 'any', 'way', 'too', 'well'
            }
            return [w for w in words if w not in stop_words]
        
        def jaccard_similarity(set1, set2):
            if not set1 or not set2:
                return 0.0
            intersection = set1 & set2
            union = set1 | set2
            return len(intersection) / len(union) if union else 0.0
        
        # ---- 1. Sentence-to-sentence topic continuity (noun/content word overlap chain) ----
        
        sentence_words = [set(extract_content_words(s)) for s in sentences]
        
        continuity_scores = []
        for i in range(1, len(sentence_words)):
            sim = jaccard_similarity(sentence_words[i-1], sentence_words[i])
            continuity_scores.append(sim)
        
        if continuity_scores:
            avg_continuity = sum(continuity_scores) / len(continuity_scores)
            # We want moderate continuity (too low = disjointed, too high = repetitive)
            # Optimal around 0.15-0.35
            if avg_continuity < 0.05:
                continuity_score = 2.0
            elif avg_continuity < 0.10:
                continuity_score = 4.0
            elif avg_continuity < 0.15:
                continuity_score = 6.0
            elif avg_continuity <= 0.40:
                continuity_score = 8.0
            elif avg_continuity <= 0.55:
                continuity_score = 6.0
            else:
                continuity_score = 3.0  # Too repetitive
            
            # Check for abrupt topic shifts (low outliers)
            if len(continuity_scores) >= 3:
                mean_c = sum(continuity_scores) / len(continuity_scores)
                abrupt_shifts = sum(1 for c in continuity_scores if c < mean_c * 0.2)
                continuity_score -= abrupt_shifts * 0.5
        else:
            continuity_score = 4.0
        
        continuity_score = max(1.0, min(10.0, continuity_score))
        
        # ---- 2. Causal/logical connective chain analysis ----
        
        # Categorize logical connectives by type
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b', r'\bcaused by\b',
            r'\bleading to\b', r'\bhence\b', r'\baccordingly\b', r'\bso that\b',
            r'\bfor this reason\b', r'\bthat\'s why\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bthis is because\b', r'\bthis is due to\b'
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bdespite\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bwhile\b', r'\byet\b', r'\bbut\b',
            r'\bin contrast\b', r'\bconversely\b', r'\bnonetheless\b',
            r'\beven though\b', r'\bregardless\b', r'\binstead\b'
        ]
        
        additive_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b', r'\badditionally\b',
            r'\balso\b', r'\bbesides\b', r'\bwhat\'s more\b', r'\bnot only\b',
            r'\bequally\b', r'\blikewise\b', r'\bsimilarly\b'
        ]
        
        conditional_connectives = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bin case\b',
            r'\bassuming\b', r'\bwhen\b', r'\bwhenever\b', r'\bsuppose\b'
        ]
        
        concluding_connectives = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bultimately\b', r'\bin short\b', r'\bfinally\b',
            r'\bto sum up\b', r'\ball in all\b', r'\bin the end\b'
        ]
        
        response_lower = response.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectives, response_lower)
        contrastive_count = count_patterns(contrastive_connectives, response_lower)
        additive_count = count_patterns(additive_connectives, response_lower)
        conditional_count = count_patterns(conditional_connectives, response_lower)
        concluding_count = count_patterns(concluding_connectives, response_lower)
        
        total_connectives = causal_count + contrastive_count + additive_count + conditional_count + concluding_count
        
        # Diversity of connective types used
        types_used = sum(1 for c in [causal_count, contrastive_count, additive_count, 
                                      conditional_count, concluding_count] if c > 0)
        
        # Normalize by number of sentences
        connective_density = total_connectives / max(len(sentences), 1)
        
        # Score: reward density and diversity
        connective_score = min(types_used * 1.5, 5.0) + min(connective_density * 4.0, 5.0)
        connective_score = max(1.0, min(10.0, connective_score))
        
        # ---- 3. Contradiction/inconsistency detection ----
        
        # Look for patterns that suggest internal contradictions
        contradiction_patterns = [
            r'\bbut (earlier|previously|before)\b.*\b(said|mentioned|stated)\b',
            r'\bcontradicts?\b',
            r'\binconsisten\w*\b',
        ]
        
        # Check for negation flip-flops within close sentences
        negation_words = {'not', "n't", 'never', 'no', 'none', 'neither', 'nor', 'nothing', 'nobody'}
        
        contradiction_penalty = 0.0
        
        # Check for sentences that assert then negate the same concept
        for i in range(len(sentences) - 1):
            s1_words = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            s2_words = set(re.findall(r'\b\w+\b', sentences[i+1].lower()))
            
            s1_has_neg = bool(s1_words & negation_words)
            s2_has_neg = bool(s2_words & negation_words)
            
            # If consecutive sentences have high overlap but different negation polarity
            content1 = set(extract_content_words(sentences[i]))
            content2 = set(extract_content_words(sentences[i+1]))
            overlap = jaccard_similarity(content1, content2)
            
            if overlap > 0.4 and s1_has_neg != s2_has_neg:
                contradiction_penalty += 1.5
        
        # Check for explicit contradiction patterns
        for pat in contradiction_patterns:
            if re.search(pat, response_lower):
                contradiction_penalty += 0.5
        
        contradiction_score = max(1.0, 10.0 - contradiction_penalty * 2)
        
        # ---- 4. Discourse structure (intro-body-conclusion) ----
        
        # Check if response has a recognizable structure
        has_greeting_or_acknowledgment = bool(re.search(
            r'^(i\s+(can|understand|hear|see|\'m|am)|it\'s|that\'s|hey|hello|imagine|let\'s|welcome)',
            response_lower.strip()
        ))
        
        has_conclusion = bool(re.search(
            r'(remember|in conclusion|finally|overall|ultimately|to sum|don\'t forget|'
            r'keep in mind|most importantly|the key|in short|last but)',
            response_lower
        ))
        
        # Check for enumeration / structured points
        numbered_points = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_structure = numbered_points >= 2
        
        # Check for paragraph breaks (indicates organized thought)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        has_paragraphs = len(paragraphs) >= 2
        
        structure_score = 3.0
        if has_greeting_or_acknowledgment:
            structure_score += 1.5
        if has_conclusion:
            structure_score += 1.5
        if has_structure:
            structure_score += 2.0
        if has_paragraphs:
            structure_score += 1.0
        if len(sentences) >= 4:
            structure_score += 1.0
        
        structure_score = max(1.0, min(10.0, structure_score))
        
        # ---- 5. Sentence complexity and subordination ----
        
        # Measure syntactic complexity via subordinate clause indicators
        subordination_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhose\b',
            r'\bwhere\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bbecause\b', r'\bsince\b', r'\bunless\b', r'\bif\b',
            r'\beven though\b', r'\bso that\b', r'\bin order to\b'
        ]
        
        total_subordination = 0
        for marker_pat in subordination_markers:
            total_subordination += len(re.findall(marker_pat, response_lower))
        
        sub_density = total_subordination / max(len(sentences), 1)
        
        # Average sentence length (words)
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        avg_sent_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)
        
        # Variance in sentence length (some variation is good)
        if len(sentence_lengths) > 1:
            mean_len = avg_sent_len
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            length_variation = std_dev / max(mean_len, 1)
        else:
            length_variation = 0
        
        complexity_score = 3.0
        # Reward moderate subordination
        if sub_density >= 0.5:
            complexity_score += 1.5
        if sub_density >= 1.0:
            complexity_score += 1.0
        if sub_density >= 2.0:
            complexity_score += 0.5
        
        # Reward appropriate sentence length (not too short, not too long)
        if 10 <= avg_sent_len <= 25:
            complexity_score += 2.0
        elif 8 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
            complexity_score += 1.0
        
        # Reward some variation in sentence length
        if 0.15 <= length_variation <= 0.6:
            complexity_score += 1.5
        elif 0.1 <= length_variation < 0.15 or 0.6 < length_variation <= 0.8:
            complexity_score += 0.5
        
        complexity_score = max(1.0, min(10.0, complexity_score))
        
        # ---- 6. Query-response alignment (logical relevance) ----
        
        query_content = set(extract_content_words(query))
        response_content = set(extract_content_words(response))
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        relevance_score = max(1.0, min(10.0, 2.0 + relevance * 8.0))
        
        # ---- 7. Progressive development detection ----
        # Check if the response builds ideas progressively (later sentences reference earlier concepts)
        
        progressive_score = 5.0
        if len(sentence_words) >= 3:
            cumulative_words = set()
            forward_refs = 0
            for i, sw in enumerate(sentence_words):
                if i > 0 and sw:
                    # How many words in this sentence were introduced in earlier sentences
                    backward_ref = len(sw & cumulative_words) / max(len(sw), 1)
                    if backward_ref > 0.1:
                        forward_refs += 1
                cumulative_words |= sw
            
            ref_ratio = forward_refs / max(len(sentence_words) - 1, 1)
            progressive_score = 3.0 + ref_ratio * 7.0
        
        progressive_score = max(1.0, min(10.0, progressive_score))
        
        # ---- 8. Empathy/acknowledgment scoring (for emotional queries) ----
        
        emotional_query_indicators = [
            r'\bfeel\w*\b', r'\bfrustrat\w*\b', r'\bstress\w*\b', r'\bsad\w*\b',
            r'\bangr\w*\b', r'\bworr\w*\b', r'\banxi\w*\b', r'\bupset\b',
            r'\bheartbrok\w*\b', r'\blone\w*\b', r'\bdespair\b', r'\bdevasta\w*\b',
            r'\bexhaust\w*\b', r'\btired\b', r'\bdown\b', r'\bstruggl\w*\b'
        ]
        
        query_emotional = sum(1 for p in emotional_query_indicators if re.search(p, query.lower()))
        is_emotional_query = query_emotional >= 2
        
        empathy_score = 5.0
        if is_emotional_query:
            empathy_patterns = [
                r'\bi (understand|can see|can hear|hear|see|know|realize)\b',
                r'\bit\'s (okay|ok|understandable|natural|normal|perfectly|completely)\b',
                r'\bthat\'s (understandable|okay|ok|natural|normal|tough|hard)\b',
                r'\bi\'m (sorry|here)\b',
                r'\byou\'re (not alone|right|valid)\b',
                r'\bfeel free\b', r'\bdon\'t hesitate\b',
                r'\bcompletely understandable\b', r'\btotally understandable\b',
                r'\bperfectly (fine|okay|ok|normal|natural|understandable)\b',
                r'\bgive yourself\b', r'\ballow yourself\b', r'\bpermission\b'
            ]
            
            empathy_hits = sum(1 for p in empathy_patterns if re.search(p, response_lower))
            
            # Dismissive patterns
            dismissive_patterns = [
                r'\bjust (get over|move on|deal with|forget)\b',
                r'\byou should be\b', r'\bstop (feeling|being|worrying)\b',
                r'\bget yourself together\b', r'\bget rid of\b',
                r'\bit\'s (just|only) a\b', r'\bmaybe you\'re just\b'
            ]
            
            dismissive_hits = sum(1 for p in dismissive_patterns if re.search(p, response_lower))
            
            empathy_score = 3.0 + empathy_hits * 1.2 - dismissive_hits * 2.0
            empathy_score = max(1.0, min(10.0, empathy_score))
        
        # ---- 9. Specificity and actionability ----
        
        # Check for specific, actionable advice vs vague platitudes
        action_patterns = [
            r'\btry\b', r'\bconsider\b', r'\bstart\b', r'\bbegin\b',
            r'\bfirst\b', r'\bnext\b', r'\bthen\b', r'\bstep\b',
            r'\bfor (example|instance)\b', r'\bsuch as\b', r'\blike\b',
            r'\bspecifically\b', r'\bin particular\b'
        ]
        
        action_count = sum(1 for p in action_patterns if re.search(p, response_lower))
        specificity_score = min(10.0, 3.0 + action_count * 0.8)
        
        # ---- Combine all scores with weights ----
        
        weights = {
            'continuity': 0.15,
            'connective': 0.15,
            'contradiction': 0.10,
            'structure': 0.15,
            'complexity': 0.10,
            'relevance': 0.10,
            'progressive': 0.10,
            'empathy': 0.08,
            'specificity': 0.07
        }
        
        scores = {
            'continuity': continuity_score,
            'connective': connective_score,
            'contradiction': contradiction_score,
            'structure': structure_score,
            'complexity': complexity_score,
            'relevance': relevance_score,
            'progressive': progressive_score,
            'empathy': empathy_score,
            'specificity': specificity_score
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Length bonus/penalty: very short responses tend to be less coherent arguments
        word_count = len(re.findall(r'\b\w+\b', response))
        if word_count < 30:
            final_score *= 0.7
        elif word_count < 50:
            final_score *= 0.85
        elif word_count > 80:
            final_score *= 1.05
        
        # Normalize to 1-5 scale to match the examples
        # Current range is roughly 1-10, map to 1-5
        final_score = 1.0 + (final_score - 1.0) * (4.0 / 9.0)
        
        return max(1.0, min(5.0, round(final_score, 2)))
        
    except Exception:
        return 3.0