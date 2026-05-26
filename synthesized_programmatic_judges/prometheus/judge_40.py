def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using a dependency-chain
    analysis approach: tracks how well ideas connect through causal/logical
    connectives, measures argument depth via clause nesting, detects contradictions
    through negation patterns, and evaluates progressive development of ideas.
    
    This variant focuses on:
    - Causal/logical dependency chains (if→then, because→therefore patterns)
    - Argument depth via subordinate clause detection
    - Negation consistency / contradiction detection
    - Progressive idea development (new concept introduction rate)
    - Discourse coherence via entity continuity (pronoun/noun reference chains)
    """
    try:
        if not query or not response:
            return 1.0
        
        if len(response.strip()) < 20:
            return 1.0
        
        import re
        import math
        from collections import Counter, defaultdict
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) == 0:
            return 1.0
        
        # ============================================================
        # FEATURE 1: Causal/Logical Dependency Chains
        # Measure how well the response builds causal reasoning chains
        # ============================================================
        
        causal_starters = [
            r'\bbecause\b', r'\bsince\b', r'\bas a result\b', r'\btherefore\b',
            r'\bthus\b', r'\bhence\b', r'\bconsequently\b', r'\bso that\b',
            r'\bdue to\b', r'\bowing to\b', r'\bleading to\b', r'\bresulting in\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bthis leads to\b',
            r'\bin order to\b', r'\bfor this reason\b'
        ]
        
        conditional_patterns = [
            r'\bif\b.*?\bthen\b', r'\bwhen\b.*?\b(will|can|should|would)\b',
            r'\bprovided that\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bin case\b', r'\bunless\b'
        ]
        
        causal_count = 0
        for pattern in causal_starters:
            causal_count += len(re.findall(pattern, response_lower))
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        # Normalize by number of sentences
        causal_density = (causal_count + conditional_count * 1.5) / max(len(sentences), 1)
        causal_score = min(causal_density * 3.0, 1.0)  # cap at 1.0
        
        # ============================================================
        # FEATURE 2: Argument Depth via Subordinate Clause Detection
        # Deeper nesting = more sophisticated argumentation
        # ============================================================
        
        subordinators = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhere\b',
            r'\bwhile\b', r'\balthough\b', r'\beven though\b', r'\bwhereas\b',
            r'\bwhereby\b', r'\bwherein\b'
        ]
        
        clause_depth_scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            depth = 0
            for sub in subordinators:
                depth += len(re.findall(sub, sent_lower))
            # Count commas as potential clause boundaries
            comma_count = sent.count(',')
            # Semicolons indicate complex compound sentences
            semi_count = sent.count(';')
            clause_complexity = depth + comma_count * 0.3 + semi_count * 0.5
            clause_depth_scores.append(clause_complexity)
        
        avg_clause_depth = sum(clause_depth_scores) / max(len(clause_depth_scores), 1)
        depth_score = min(avg_clause_depth / 3.0, 1.0)
        
        # ============================================================
        # FEATURE 3: Negation Consistency / Contradiction Detection
        # Look for contradictory statements within the response
        # ============================================================
        
        negation_words = {'not', "n't", 'no', 'never', 'neither', 'nor', 'nothing', 
                          'nowhere', 'nobody', 'none', "don't", "doesn't", "didn't",
                          "won't", "wouldn't", "shouldn't", "can't", "cannot", "couldn't"}
        
        # Extract key verb phrases from each sentence
        sentence_polarities = []
        sentence_key_concepts = []
        
        for sent in sentences:
            words = re.findall(r'\b\w+\b', sent.lower())
            has_negation = any(w in negation_words or "n't" in w for w in words)
            # Extract content words (nouns, verbs - approximated by longer words)
            content_words = set(w for w in words if len(w) > 3 and w not in {
                'this', 'that', 'these', 'those', 'with', 'from', 'have', 'been',
                'will', 'would', 'could', 'should', 'might', 'about', 'their',
                'there', 'they', 'them', 'then', 'than', 'what', 'when', 'where',
                'which', 'while', 'your', 'yours', 'also', 'just', 'more', 'some',
                'into', 'over', 'such', 'very', 'each', 'much', 'most', 'other'
            })
            sentence_polarities.append(has_negation)
            sentence_key_concepts.append(content_words)
        
        # Check for potential contradictions: same concepts with different polarities
        contradiction_count = 0
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                overlap = sentence_key_concepts[i] & sentence_key_concepts[j]
                if len(overlap) >= 2 and sentence_polarities[i] != sentence_polarities[j]:
                    # Potential contradiction - weighted by concept overlap
                    overlap_ratio = len(overlap) / max(
                        min(len(sentence_key_concepts[i]), len(sentence_key_concepts[j])), 1)
                    if overlap_ratio > 0.4:
                        contradiction_count += 1
        
        contradiction_penalty = min(contradiction_count * 0.15, 0.5)
        consistency_score = 1.0 - contradiction_penalty
        
        # ============================================================
        # FEATURE 4: Progressive Idea Development
        # Measure how new concepts are introduced progressively
        # ============================================================
        
        cumulative_concepts = set()
        new_concept_rates = []
        
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sent.lower())
            content_words = set(w for w in words if len(w) > 4)
            new_concepts = content_words - cumulative_concepts
            if len(content_words) > 0:
                novelty_rate = len(new_concepts) / len(content_words)
            else:
                novelty_rate = 0
            new_concept_rates.append(novelty_rate)
            cumulative_concepts.update(content_words)
        
        # Ideal: starts with high novelty, gradually decreases (building on established ideas)
        # Bad: all novelty at start then nothing, or random spikes
        if len(new_concept_rates) >= 3:
            # Check for gradual decrease pattern (good sign of building on ideas)
            first_third = new_concept_rates[:len(new_concept_rates)//3 + 1]
            last_third = new_concept_rates[-(len(new_concept_rates)//3 + 1):]
            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)
            
            # Good: first third has higher novelty than last third
            if avg_first > avg_last:
                progression_score = 0.7 + 0.3 * min((avg_first - avg_last) / 0.5, 1.0)
            else:
                progression_score = 0.4
            
            # Also check for smooth transitions (low variance in novelty rates)
            if len(new_concept_rates) > 1:
                diffs = [abs(new_concept_rates[i+1] - new_concept_rates[i]) 
                         for i in range(len(new_concept_rates)-1)]
                avg_diff = sum(diffs) / len(diffs)
                smoothness = max(0, 1.0 - avg_diff * 2)
                progression_score = progression_score * 0.6 + smoothness * 0.4
            
        elif len(new_concept_rates) >= 1:
            progression_score = 0.4
        else:
            progression_score = 0.3
        
        # ============================================================
        # FEATURE 5: Entity Continuity / Reference Chains
        # Good coherence: entities introduced early are referenced later
        # ============================================================
        
        pronouns = {'it', 'its', 'they', 'them', 'their', 'theirs', 'this', 'that',
                     'these', 'those', 'he', 'she', 'his', 'her', 'him'}
        
        pronoun_usage = 0
        total_words_count = 0
        for sent in sentences:
            words = re.findall(r'\b\w+\b', sent.lower())
            total_words_count += len(words)
            for w in words:
                if w in pronouns:
                    pronoun_usage += 1
        
        # Moderate pronoun usage indicates good referencing
        if total_words_count > 0:
            pronoun_ratio = pronoun_usage / total_words_count
            # Sweet spot: 3-10% pronouns
            if 0.03 <= pronoun_ratio <= 0.10:
                reference_score = 1.0
            elif pronoun_ratio < 0.03:
                reference_score = 0.5 + pronoun_ratio * 16.7  # linear up to 0.03
            else:
                reference_score = max(0.4, 1.0 - (pronoun_ratio - 0.10) * 5)
        else:
            reference_score = 0.3
        
        # ============================================================
        # FEATURE 6: Discourse Marker Variety
        # Using diverse connectives shows sophisticated argument structure
        # ============================================================
        
        discourse_categories = {
            'additive': [r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b', 
                        r'\badditionally\b', r'\balso\b', r'\bbesides\b'],
            'adversative': [r'\bhowever\b', r'\bnevertheless\b', r'\bon the other hand\b',
                           r'\byet\b', r'\bbut\b', r'\bdespite\b', r'\binstead\b',
                           r'\bnonetheless\b', r'\bstill\b'],
            'causal': [r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
                      r'\bas a result\b', r'\baccordingly\b'],
            'temporal': [r'\bfirst\b', r'\bthen\b', r'\bnext\b', r'\bfinally\b',
                        r'\bsubsequently\b', r'\bmeanwhile\b', r'\bafterwards\b',
                        r'\binitially\b', r'\beventually\b'],
            'elaborative': [r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
                           r'\bin particular\b', r'\bnamely\b', r'\bthat is\b',
                           r'\bin other words\b'],
            'conclusive': [r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
                          r'\bin summary\b', r'\bto sum up\b', r'\ball in all\b']
        }
        
        categories_used = 0
        total_markers = 0
        for category, patterns in discourse_categories.items():
            cat_count = 0
            for pattern in patterns:
                cat_count += len(re.findall(pattern, response_lower))
            if cat_count > 0:
                categories_used += 1
            total_markers += cat_count
        
        # Variety score: how many categories are represented
        variety_score = min(categories_used / 4.0, 1.0)
        # Density score: markers per sentence
        marker_density = total_markers / max(len(sentences), 1)
        density_score = min(marker_density / 1.0, 1.0)
        
        discourse_score = variety_score * 0.6 + density_score * 0.4
        
        # ============================================================
        # FEATURE 7: Structural Completeness
        # Does the response have intro, body, conclusion-like structure?
        # ============================================================
        
        # Check for opening acknowledgment/framing
        opening_patterns = [
            r'^(i |it\'s |this |the |to |imagine |let|here|sure|absolutely|great|hey|hi|hello|ok|well)',
            r'^(i can|i understand|i hear|i\'m|it sounds|it seems)',
        ]
        has_opening = any(re.match(p, response_lower.strip()) for p in opening_patterns)
        
        # Check for concluding signals
        closing_patterns = [
            r'\bremember\b', r'\bin (short|summary|conclusion)\b', r'\boverall\b',
            r'\bkeep in mind\b', r'\bdon\'t forget\b', r'\blast(ly)?\b',
            r'\bmost importantly\b', r'\bfinal(ly)?\b', r'\bto wrap\b',
            r'\bgood luck\b', r'\bhope this\b', r'\bfeel free\b'
        ]
        
        # Check last 2 sentences for closing signals
        closing_text = ' '.join(sentences[-2:]).lower() if len(sentences) >= 2 else response_lower
        has_closing = any(re.search(p, closing_text) for p in closing_patterns)
        
        structural_score = 0.4
        if has_opening:
            structural_score += 0.3
        if has_closing:
            structural_score += 0.3
        
        # ============================================================
        # FEATURE 8: Query Relevance via Semantic Alignment
        # Check if the response addresses the core topic of the query
        # ============================================================
        
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        
        # Remove very common words
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
                     'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
                     'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
                     'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
                     'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
                     'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                     'it', 'its', 'they', 'them', 'their'}
        
        query_content = query_words - stopwords
        response_content = response_words - stopwords
        
        if len(query_content) > 0:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        relevance_score = min(relevance * 1.5, 1.0)
        
        # ============================================================
        # FEATURE 9: Response Substantiveness
        # Adequate length and information density
        # ============================================================
        
        word_count = len(re.findall(r'\b\w+\b', response))
        
        # Ideal range: 80-300 words for most responses
        if word_count < 30:
            length_score = 0.2
        elif word_count < 60:
            length_score = 0.4
        elif word_count < 100:
            length_score = 0.7
        elif word_count <= 350:
            length_score = 1.0
        else:
            length_score = max(0.6, 1.0 - (word_count - 350) / 500)
        
        # Unique word ratio (vocabulary richness)
        all_words = re.findall(r'\b\w+\b', response_lower)
        if len(all_words) > 0:
            unique_ratio = len(set(all_words)) / len(all_words)
            vocab_score = min(unique_ratio * 1.5, 1.0)
        else:
            vocab_score = 0.3
        
        substantive_score = length_score * 0.5 + vocab_score * 0.5
        
        # ============================================================
        # FEATURE 10: Empathy/Engagement Appropriateness
        # For queries that involve emotional content, check for empathetic language
        # ============================================================
        
        emotional_query_signals = ['feel', 'stress', 'frustrat', 'sad', 'lonely', 'depress',
                                   'anxious', 'worry', 'concern', 'afraid', 'fear', 'upset',
                                   'heartbroken', 'devastat', 'exhaust', 'struggle', 'difficult',
                                   'tough', 'hard time', 'down', 'regret', 'pain']
        
        is_emotional_query = any(sig in query_lower for sig in emotional_query_signals)
        
        empathy_markers = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b', r'\bi\'m sorry\b',
            r'\bthat\'s (completely |totally |absolutely )?(understandable|okay|fine|normal|natural|valid)\b',
            r'\bit\'s (completely |totally |absolutely )?(okay|fine|normal|natural|understandable|valid)\b',
            r'\bperfectly (fine|okay|normal|natural|understandable)\b',
            r'\byour feelings\b', r'\bhow you feel\b', r'\bfeel this way\b',
            r'\bgive yourself\b', r'\bbe kind to yourself\b', r'\btake (your |a )?time\b',
            r'\bhere for you\b', r'\blisten\b'
        ]
        
        empathy_count = 0
        for pattern in empathy_markers:
            empathy_count += len(re.findall(pattern, response_lower))
        
        if is_emotional_query:
            empathy_score = min(empathy_count / 2.0, 1.0)
        else:
            empathy_score = 0.5  # neutral for non-emotional queries
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        # Weighted combination
        weights = {
            'causal': 0.12,
            'depth': 0.08,
            'consistency': 0.10,
            'progression': 0.10,
            'reference': 0.06,
            'discourse': 0.14,
            'structural': 0.10,
            'relevance': 0.10,
            'substantive': 0.10,
            'empathy': 0.10,
        }
        
        raw_score = (
            weights['causal'] * causal_score +
            weights['depth'] * depth_score +
            weights['consistency'] * consistency_score +
            weights['progression'] * progression_score +
            weights['reference'] * reference_score +
            weights['discourse'] * discourse_score +
            weights['structural'] * structural_score +
            weights['relevance'] * relevance_score +
            weights['substantive'] * substantive_score +
            weights['empathy'] * empathy_score
        )
        
        # Scale to 1-5 range
        final_score = 1.0 + raw_score * 4.0
        
        # Clamp
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0