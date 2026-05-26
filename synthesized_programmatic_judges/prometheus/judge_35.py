def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectors)
    - Sentence-level coherence via topic threading (shared noun/entity chains)
    - Argument depth detection (premise-conclusion patterns)
    - Contradiction/inconsistency detection
    - Structural progression analysis (intro-body-conclusion pattern)
    """
    try:
        import re
        import math
        from collections import Counter, defaultdict
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_clean = response.strip()
        if len(response_clean) < 20:
            return 0.5
        
        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space or end
            sents = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 3]
        
        sentences = split_sentences(response_clean)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5
        
        # Extract words (lowercased, alpha only)
        def get_words(text):
            return [w.lower() for w in re.findall(r'[a-zA-Z]+', text) if len(w) > 1]
        
        def get_content_words(text):
            stop = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                    'do','does','did','will','would','could','should','may','might','shall','can',
                    'to','of','in','for','on','with','at','by','from','as','into','through','during',
                    'before','after','above','below','between','out','off','over','under','again',
                    'further','then','once','here','there','when','where','why','how','all','both',
                    'each','few','more','most','other','some','such','no','nor','not','only','own',
                    'same','so','than','too','very','just','because','but','and','or','if','while',
                    'about','up','it','its','this','that','these','those','i','me','my','we','our',
                    'you','your','he','him','his','she','her','they','them','their','what','which',
                    'who','whom','am','also','like','get','got','much','many','even','still','well',
                    'make','made','take','go','going','come','see','know','think','want','need',
                    'use','try','keep','let','say','said','tell','told'}
            words = get_words(text)
            return [w for w in words if w not in stop]
        
        # ============================================================
        # 1. DISCOURSE MARKER RICHNESS AND APPROPRIATENESS (0-20)
        # ============================================================
        
        # Categorized discourse markers
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
            r'\bas a result\b', r'\bdue to\b', r'\bsince\b', r'\bso that\b', r'\bfor this reason\b',
            r'\bthis means\b', r'\bthis leads to\b', r'\bowing to\b', r'\bthat\'s why\b',
            r'\bin order to\b', r'\baccordingly\b'
        ]
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b', r'\bon the other hand\b',
            r'\bin contrast\b', r'\bwhile\b', r'\bwhereas\b', r'\bdespite\b', r'\byet\b',
            r'\binstead\b', r'\brather\b', r'\bstill\b', r'\beven though\b', r'\bnonetheless\b'
        ]
        additive_markers = [
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b', r'\bin addition\b',
            r'\balso\b', r'\bbesides\b', r'\blikewise\b', r'\bsimilarly\b', r'\bwhat\'s more\b',
            r'\bnot only\b', r'\bas well\b'
        ]
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b', r'\bthen\b', r'\bfinally\b',
            r'\bafterward\b', r'\bsubsequently\b', r'\bmeanwhile\b', r'\bpreviously\b',
            r'\bbefore\b', r'\bafter\b', r'\bin the end\b', r'\bto begin\b', r'\blast\b',
            r'\binitially\b', r'\beventually\b'
        ]
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bsuch as\b', r'\bthat is\b', r'\bin other words\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bto clarify\b', r'\bimagine\b', r'\bconsider\b'
        ]
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b', r'\bin summary\b',
            r'\bto sum up\b', r'\ball in all\b', r'\bin short\b', r'\bultimately\b',
            r'\bremember\b', r'\bthe key\b', r'\bthe point\b'
        ]
        
        response_lower = response_clean.lower()
        
        def count_markers(patterns, text):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text))
            return count
        
        causal_count = count_markers(causal_markers, response_lower)
        contrastive_count = count_markers(contrastive_markers, response_lower)
        additive_count = count_markers(additive_markers, response_lower)
        temporal_count = count_markers(temporal_markers, response_lower)
        elaboration_count = count_markers(elaboration_markers, response_lower)
        conclusion_count = count_markers(conclusion_markers, response_lower)
        
        total_markers = causal_count + contrastive_count + additive_count + temporal_count + elaboration_count + conclusion_count
        
        # Count distinct categories used
        categories_used = sum([
            causal_count > 0,
            contrastive_count > 0,
            additive_count > 0,
            temporal_count > 0,
            elaboration_count > 0,
            conclusion_count > 0
        ])
        
        # Marker density (markers per sentence) - sweet spot around 0.3-0.8
        marker_density = total_markers / max(num_sentences, 1)
        density_score = 0
        if marker_density < 0.05:
            density_score = 2
        elif marker_density < 0.15:
            density_score = 5
        elif marker_density < 0.3:
            density_score = 8
        elif marker_density < 0.6:
            density_score = 10
        elif marker_density < 1.0:
            density_score = 8
        else:
            density_score = 5  # over-connected
        
        # Category diversity bonus
        diversity_score = min(categories_used * 1.8, 10)
        
        discourse_score = (density_score + diversity_score) / 2.0  # 0-10 -> normalize to 0-20
        discourse_score = discourse_score * 2.0
        
        # ============================================================
        # 2. TOPIC THREADING / ENTITY COHERENCE (0-20)
        # ============================================================
        # Measure how well consecutive sentences share content words (entity chains)
        
        sent_content_words = [set(get_content_words(s)) for s in sentences]
        
        if num_sentences >= 2:
            # Adjacent sentence coherence
            adjacent_overlaps = []
            for i in range(len(sent_content_words) - 1):
                s1 = sent_content_words[i]
                s2 = sent_content_words[i + 1]
                if len(s1) > 0 and len(s2) > 0:
                    # Dice coefficient
                    overlap = 2 * len(s1 & s2) / (len(s1) + len(s2))
                    adjacent_overlaps.append(overlap)
            
            if adjacent_overlaps:
                avg_adjacent = sum(adjacent_overlaps) / len(adjacent_overlaps)
                # Check consistency of coherence (low variance = more consistent threading)
                if len(adjacent_overlaps) > 1:
                    mean_o = avg_adjacent
                    var_o = sum((x - mean_o)**2 for x in adjacent_overlaps) / len(adjacent_overlaps)
                    std_o = math.sqrt(var_o)
                    consistency = max(0, 1 - std_o * 2)  # penalize high variance
                else:
                    consistency = 0.5
                
                # Sweet spot: some overlap but not too repetitive
                if avg_adjacent < 0.02:
                    thread_quality = 2  # disconnected
                elif avg_adjacent < 0.08:
                    thread_quality = 6
                elif avg_adjacent < 0.2:
                    thread_quality = 10  # good coherence
                elif avg_adjacent < 0.4:
                    thread_quality = 8  # slightly repetitive
                else:
                    thread_quality = 5  # too repetitive
                
                threading_score = thread_quality * 0.7 + consistency * 10 * 0.3
            else:
                threading_score = 3
        else:
            threading_score = 5  # single sentence, neutral
        
        threading_score = threading_score * 2.0  # scale to 0-20
        
        # ============================================================
        # 3. ARGUMENT DEPTH & PREMISE-CONCLUSION STRUCTURE (0-20)
        # ============================================================
        
        # Detect if-then patterns, because-therefore chains, conditional reasoning
        conditional_patterns = [
            r'\bif\b.{5,60}\bthen\b',
            r'\bwhen\b.{5,60}\b(?:you|it|this|they|we)\b.{3,40}\b(?:will|can|should|would)\b',
            r'\bby\b.{3,40}\b(?:you|it|this|they|we)\b.{3,40}\b(?:can|will|would)\b',
        ]
        
        conditional_count = 0
        for p in conditional_patterns:
            conditional_count += len(re.findall(p, response_lower))
        
        # Detect reasoning chains: premise indicators followed by conclusion indicators
        premise_indicators = [r'\bgiven that\b', r'\bsince\b', r'\bbecause\b', r'\bconsidering\b',
                             r'\bthe fact that\b', r'\bit\'s\s+\w+\s+that\b', r'\bnote that\b']
        conclusion_indicators = [r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bso\b',
                                r'\bthis means\b', r'\bconsequently\b', r'\bas a result\b',
                                r'\bit follows\b', r'\bwe can\b', r'\byou can\b', r'\byou should\b']
        
        premise_count = sum(len(re.findall(p, response_lower)) for p in premise_indicators)
        conclusion_ind_count = sum(len(re.findall(p, response_lower)) for p in conclusion_indicators)
        
        # Detect explanatory depth: "this is because", "the reason is", etc.
        explanation_patterns = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis happens\b',
            r'\bwhat this means\b', r'\bin essence\b', r'\bfundamentally\b',
            r'\bthe idea is\b', r'\bthe concept\b', r'\bthink of it\b',
            r'\banalog\w*\b', r'\bmetaphor\b', r'\bjust like\b', r'\bsimilar to\b'
        ]
        explanation_count = sum(len(re.findall(p, response_lower)) for p in explanation_patterns)
        
        # Numbered/structured arguments
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean))
        lettered_items = len(re.findall(r'(?:^|\n)\s*[a-e][\.\)]\s', response_clean))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response_clean))
        structured_items = numbered_items + lettered_items + bullet_items
        
        # Compute argument depth score
        reasoning_elements = (
            min(conditional_count, 3) * 2 +
            min(premise_count, 3) * 1.5 +
            min(conclusion_ind_count, 3) * 1.5 +
            min(explanation_count, 4) * 1.5 +
            min(structured_items, 6) * 1.0
        )
        
        argument_score = min(reasoning_elements, 20)
        
        # ============================================================
        # 4. CONTRADICTION / INCONSISTENCY DETECTION (0-15, penalty-based)
        # ============================================================
        
        contradiction_penalty = 0
        
        # Check for negation contradictions
        negation_pairs = [
            (r'\byou should\b', r'\byou should not\b'),
            (r'\byou can\b', r'\byou cannot\b'),
            (r'\bit is\b', r'\bit is not\b'),
            (r'\bimportant\b', r'\bnot important\b'),
            (r'\bpossible\b', r'\bnot possible\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
        ]
        
        for pos, neg in negation_pairs:
            if re.search(pos, response_lower) and re.search(neg, response_lower):
                # Check if they're in different sentences (potential contradiction)
                contradiction_penalty += 2
        
        # Check for hedging overuse (might indicate uncertainty/incoherence)
        hedge_words = [r'\bmaybe\b', r'\bperhaps\b', r'\bprobably\b', r'\bmight\b',
                       r'\bi guess\b', r'\bi think\b', r'\bnot sure\b', r'\bpossibly\b']
        hedge_count = sum(len(re.findall(p, response_lower)) for p in hedge_words)
        hedge_ratio = hedge_count / max(num_sentences, 1)
        if hedge_ratio > 0.5:
            contradiction_penalty += 3
        elif hedge_ratio > 0.3:
            contradiction_penalty += 1
        
        # Check for "but" immediately contradicting previous claim without resolution
        # (simplified: excessive "but" usage)
        but_count = len(re.findall(r'\bbut\b', response_lower))
        if but_count > num_sentences * 0.4 and num_sentences > 3:
            contradiction_penalty += 2
        
        consistency_score = max(0, 15 - contradiction_penalty)
        
        # ============================================================
        # 5. STRUCTURAL PROGRESSION (intro-body-conclusion) (0-15)
        # ============================================================
        
        progression_score = 0
        
        if num_sentences >= 3:
            # Check for opening engagement (addresses the query)
            query_words = set(get_content_words(query))
            first_sent_words = set(get_content_words(sentences[0]))
            
            # Opening relevance
            if len(query_words) > 0 and len(first_sent_words) > 0:
                opening_relevance = len(query_words & first_sent_words) / max(len(query_words), 1)
                progression_score += min(opening_relevance * 8, 4)
            
            # Check if response addresses the user (engagement markers)
            engagement_patterns = [r'\byou\b', r'\byour\b', r'\blet\'s\b', r'\bimagine\b',
                                  r'\bconsider\b', r'\bthink\b', r'\bwe\b']
            engagement_count = sum(len(re.findall(p, response_lower)) for p in engagement_patterns)
            if engagement_count > 0:
                progression_score += min(engagement_count * 0.5, 3)
            
            # Check for concluding/summarizing elements in last portion
            last_third = ' '.join(sentences[-(max(num_sentences // 3, 1)):]).lower()
            concluding_patterns = [r'\bremember\b', r'\bin summary\b', r'\boverall\b',
                                  r'\bdon\'t\s+(?:forget|hesitate)\b', r'\bfeel free\b',
                                  r'\bgood luck\b', r'\bhope\b', r'\bwish\b',
                                  r'\bthe key\b', r'\bmost importantly\b', r'\bin the end\b',
                                  r'\bkeep in mind\b', r'\btake care\b']
            has_conclusion = any(re.search(p, last_third) for p in concluding_patterns)
            if has_conclusion:
                progression_score += 4
            
            # Progressive development: middle section should have more content words
            if num_sentences >= 4:
                first_q = sent_content_words[0]
                middle_words = set()
                for i in range(1, num_sentences - 1):
                    middle_words |= sent_content_words[i]
                # Middle should introduce new concepts
                new_in_middle = middle_words - first_q
                if len(new_in_middle) > 3:
                    progression_score += min(len(new_in_middle) * 0.3, 4)
        else:
            # Very short response - limited structure possible
            progression_score = 3
        
        progression_score = min(progression_score, 15)
        
        # ============================================================
        # 6. RESPONSE COMPLETENESS & APPROPRIATE LENGTH (0-10)
        # ============================================================
        
        word_count = len(get_words(response_clean))
        
        # Reasonable length scoring
        if word_count < 20:
            length_score = 2
        elif word_count < 50:
            length_score = 5
        elif word_count < 100:
            length_score = 7
        elif word_count < 200:
            length_score = 9
        elif word_count < 350:
            length_score = 10
        else:
            length_score = 8  # might be verbose
        
        # Check if response actually addresses the query topic
        query_content = set(get_content_words(query))
        response_content = set(get_content_words(response_clean))
        if len(query_content) > 0:
            topic_coverage = len(query_content & response_content) / len(query_content)
            length_score = length_score * (0.5 + 0.5 * min(topic_coverage * 2, 1.0))
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        # Weights reflect importance for logical coherence:
        # discourse_score: 0-20 (weight: 1.0)
        # threading_score: 0-20 (weight: 1.0) 
        # argument_score: 0-20 (weight: 1.0)
        # consistency_score: 0-15 (weight: 1.0)
        # progression_score: 0-15 (weight: 1.0)
        # length_score: 0-10 (weight: 1.0)
        
        raw_total = (
            discourse_score +      # 0-20
            threading_score +      # 0-20
            argument_score +       # 0-20
            consistency_score +    # 0-15
            progression_score +    # 0-15
            length_score           # 0-10
        )
        
        # Max possible: 20+20+20+15+15+10 = 100
        # Normalize to 1-5 scale
        normalized = (raw_total / 100.0) * 4.0 + 1.0
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        # Apply slight non-linear scaling to increase discrimination
        # Map [1,5] through a sigmoid-like curve centered at 3
        centered = (final_score - 3.0) / 2.0  # [-1, 1]
        stretched = centered * 1.2  # slightly expand
        final_score = 3.0 + stretched * 2.0
        final_score = max(1.0, min(5.0, round(final_score, 2)))
        
        return final_score
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 3.0
            return 1.5
        except:
            return 2.0