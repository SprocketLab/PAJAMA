def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis:
    - Causal/logical connective density and proper usage
    - Sentence-level progression (topic continuity vs. topic drift)
    - Argument depth (claim → evidence → conclusion pattern)
    - Contradiction detection via negation patterns
    - Repetition penalty (circular reasoning indicator)
    - Structural completeness (intro/body/conclusion signals)
    """
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5

        query = query.strip() if query else ""

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5

        words = re.findall(r'\b[a-zA-Z]+\b', response.lower())
        num_words = len(words)
        if num_words == 0:
            return 0.5

        score = 0.0

        # ============================================================
        # 1. CAUSAL/LOGICAL CONNECTIVE ANALYSIS (0-20 points)
        # Categorize connectives by their discourse function
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bleading to\b', r'\bcaused by\b', r'\bhence\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b'
        ]
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bdespite\b', r'\bin contrast\b', r'\byet\b',
            r'\bbut\b', r'\bconversely\b', r'\binstead\b'
        ]
        additive_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bnot only\b', r'\bas well as\b'
        ]
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bto illustrate\b'
        ]
        conclusion_connectives = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\bultimately\b', r'\bin short\b',
            r'\ball in all\b', r'\bto conclude\b'
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
        elaboration_count = count_patterns(elaboration_connectives, resp_lower)
        conclusion_count = count_patterns(conclusion_connectives, resp_lower)

        # Weighted connective score - causal and elaboration connectives are
        # stronger indicators of logical structure
        connective_total = (causal_count * 3.0 + contrastive_count * 2.5 +
                           additive_count * 1.5 + elaboration_count * 2.5 +
                           conclusion_count * 2.0)
        
        # Normalize by number of sentences - expect ~0.3-0.5 connectives per sentence
        connective_density = connective_total / max(num_sentences, 1)
        # Diminishing returns - optimal around density of 2-4
        connective_score = min(20, connective_density * 5)
        score += connective_score

        # Bonus for diversity of connective types used
        types_used = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                      elaboration_count, conclusion_count] if c > 0)
        diversity_bonus = min(5, types_used * 1.2)
        score += diversity_bonus

        # ============================================================
        # 2. SENTENCE-LEVEL TOPIC PROGRESSION (0-20 points)
        # Check if consecutive sentences share topical content words
        # (theme-rheme progression) without being repetitive
        # ============================================================
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'out', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'because',
                'but', 'and', 'or', 'if', 'while', 'that', 'which', 'who',
                'whom', 'this', 'these', 'those', 'it', 'its', 'they',
                'their', 'them', 'he', 'she', 'his', 'her', 'we', 'our',
                'you', 'your', 'i', 'me', 'my'
            }
            w = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(w) - stop_words

        if num_sentences >= 2:
            coherence_scores = []
            for i in range(1, num_sentences):
                prev_content = get_content_words(sentences[i-1])
                curr_content = get_content_words(sentences[i])
                if prev_content and curr_content:
                    overlap = len(prev_content & curr_content)
                    union = len(prev_content | curr_content)
                    # We want moderate overlap (0.1-0.5) - not too much (repetitive)
                    # and not too little (incoherent)
                    ratio = overlap / max(union, 1)
                    # Optimal coherence: some overlap but also new info
                    if 0.05 <= ratio <= 0.6:
                        coherence_scores.append(1.0)
                    elif ratio > 0.6:
                        # Too much overlap - potentially repetitive
                        coherence_scores.append(0.4)
                    else:
                        # No overlap - potential topic jump
                        coherence_scores.append(0.2)
                else:
                    coherence_scores.append(0.3)
            
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            progression_score = avg_coherence * 20
        else:
            progression_score = 8  # Single sentence - neutral
        
        score += progression_score

        # ============================================================
        # 3. ARGUMENT DEPTH / CLAIM-EVIDENCE PATTERN (0-15 points)
        # Detect claim → support → elaboration patterns
        # ============================================================
        
        # Claim indicators (assertions)
        claim_patterns = [
            r'\bis\b', r'\bare\b', r'\bmeans\b', r'\bsuggests\b',
            r'\bimplies\b', r'\bindicates\b', r'\bshows\b',
            r'\bdemonstrates\b', r'\brefers to\b'
        ]
        # Support/evidence indicators
        support_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\bevidence\b', r'\bdata\b',
            r'\bresearch\b', r'\bstudies\b', r'\baccording to\b',
            r'\bthis means\b', r'\bthis suggests\b', r'\bin fact\b'
        ]
        # Qualification/nuance indicators
        qualification_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bit depends\b', r'\bin some cases\b', r'\bnot always\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\btends to\b', r'\bmay\b', r'\bmight\b'
        ]

        support_count = count_patterns(support_patterns, resp_lower)
        qualification_count = count_patterns(qualification_patterns, resp_lower)
        
        # Argument depth: having both claims + support + qualification
        depth_indicators = min(support_count, 3) * 2.5 + min(qualification_count, 3) * 1.5
        argument_depth_score = min(15, depth_indicators)
        score += argument_depth_score

        # ============================================================
        # 4. REPETITION / CIRCULAR REASONING PENALTY (0 to -15 points)
        # ============================================================
        
        # Check for repeated phrases (3+ word sequences)
        if num_words >= 6:
            trigrams = []
            for i in range(len(words) - 2):
                trigrams.append(tuple(words[i:i+3]))
            
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for _, c in trigram_counts.items() if c > 1)
            total_trigrams = max(len(trigrams), 1)
            repetition_ratio = repeated_trigrams / total_trigrams
            
            # Also check for repeated sentences (near-duplicate)
            sentence_contents = [get_content_words(s) for s in sentences]
            near_duplicate_pairs = 0
            for i in range(len(sentence_contents)):
                for j in range(i+1, len(sentence_contents)):
                    if sentence_contents[i] and sentence_contents[j]:
                        si, sj = sentence_contents[i], sentence_contents[j]
                        overlap = len(si & sj)
                        union_size = len(si | sj)
                        if union_size > 0 and overlap / union_size > 0.75:
                            near_duplicate_pairs += 1
            
            repetition_penalty = -(repetition_ratio * 10 + near_duplicate_pairs * 3)
            repetition_penalty = max(-15, repetition_penalty)
        else:
            repetition_penalty = 0
        
        score += repetition_penalty

        # ============================================================
        # 5. STRUCTURAL COMPLETENESS (0-15 points)
        # Does the response have an identifiable structure?
        # ============================================================
        
        # Check for opening statement (defines/introduces topic)
        has_intro = False
        if num_sentences >= 1:
            first_sent_lower = sentences[0].lower()
            intro_patterns = [
                r'\bis a\b', r'\bare\b', r'\brefers to\b', r'\bmeans\b',
                r'\bcan be defined\b', r'\binvolves\b', r'\bwhen\b',
                r'\bthe\b.*\bis\b'
            ]
            for p in intro_patterns:
                if re.search(p, first_sent_lower):
                    has_intro = True
                    break
        
        # Check for concluding/wrapping statement
        has_conclusion = False
        if num_sentences >= 2:
            last_sent_lower = sentences[-1].lower()
            conclusion_patterns = [
                r'\boverall\b', r'\bin conclusion\b', r'\btherefore\b',
                r'\bthus\b', r'\bimportant\b', r'\bultimately\b',
                r'\bin summary\b', r'\bshould\b', r'\bmust\b',
                r'\bcan help\b', r'\bto ensure\b'
            ]
            for p in conclusion_patterns:
                if re.search(p, last_sent_lower):
                    has_conclusion = True
                    break
        
        # Check for body development (middle sentences that elaborate)
        has_body = num_sentences >= 3
        
        structural_score = 0
        if has_intro:
            structural_score += 5
        if has_body:
            structural_score += 5
        if has_conclusion:
            structural_score += 5
        
        score += structural_score

        # ============================================================
        # 6. INFORMATION DENSITY & DEVELOPMENT (0-15 points)
        # Unique content words ratio and progressive information addition
        # ============================================================
        
        content_words = list(get_content_words(response))
        unique_content = set(content_words)
        
        if num_words > 0:
            # Vocabulary richness
            vocab_richness = len(unique_content) / max(num_words, 1)
            # Optimal range: 0.3-0.6 for well-developed text
            if 0.25 <= vocab_richness <= 0.65:
                richness_score = 8
            elif vocab_richness > 0.65:
                richness_score = 5  # Might be too terse / list-like
            else:
                richness_score = vocab_richness * 20  # Too repetitive
        else:
            richness_score = 0

        # Progressive information: each sentence should add new content
        if num_sentences >= 2:
            seen_content = set()
            new_info_ratios = []
            for s in sentences:
                s_content = get_content_words(s)
                if s_content:
                    new_words = s_content - seen_content
                    new_ratio = len(new_words) / len(s_content) if s_content else 0
                    new_info_ratios.append(new_ratio)
                    seen_content |= s_content
            
            if new_info_ratios:
                avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
                # We want each sentence to bring ~40-80% new information
                info_development = min(7, avg_new_info * 10)
            else:
                info_development = 3
        else:
            info_development = 3

        score += richness_score + info_development

        # ============================================================
        # 7. RESPONSE LENGTH APPROPRIATENESS (0-10 points)
        # Very short responses are unlikely to have good argument structure
        # ============================================================
        
        # Length scoring with diminishing returns
        if num_words < 10:
            length_score = num_words * 0.3
        elif num_words < 20:
            length_score = 3 + (num_words - 10) * 0.2
        elif num_words < 50:
            length_score = 5 + (num_words - 20) * 0.1
        elif num_words < 150:
            length_score = 8 + min(2, (num_words - 50) * 0.02)
        else:
            length_score = 9  # Very long - might be rambling
        
        length_score = min(10, length_score)
        score += length_score

        # ============================================================
        # 8. INTERNAL CONTRADICTION CHECK (-10 to 0 points)
        # Look for contradictory statements
        # ============================================================
        
        contradiction_penalty = 0
        negation_words = {'not', 'no', "n't", 'never', 'neither', 'nor', 'nothing', 'nowhere', 'none'}
        
        if num_sentences >= 2:
            for i in range(len(sentences)):
                for j in range(i+1, min(i+3, len(sentences))):
                    words_i = set(re.findall(r'\b[a-zA-Z]+\b', sentences[i].lower()))
                    words_j = set(re.findall(r'\b[a-zA-Z]+\b', sentences[j].lower()))
                    
                    content_i = words_i - negation_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'they', 'this', 'that'}
                    content_j = words_j - negation_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'they', 'this', 'that'}
                    
                    has_neg_i = bool(words_i & negation_words)
                    has_neg_j = bool(words_j & negation_words)
                    
                    # If sentences share significant content but differ in negation
                    if content_i and content_j:
                        shared = len(content_i & content_j)
                        if shared >= 3 and has_neg_i != has_neg_j:
                            contradiction_penalty -= 3
        
        contradiction_penalty = max(-10, contradiction_penalty)
        score += contradiction_penalty

        # ============================================================
        # 9. QUERY RELEVANCE ALIGNMENT (0-5 bonus)
        # Response should logically address the query
        # ============================================================
        
        if query:
            query_content = get_content_words(query)
            response_content = get_content_words(response)
            if query_content and response_content:
                relevance = len(query_content & response_content) / max(len(query_content), 1)
                relevance_score = min(5, relevance * 8)
            else:
                relevance_score = 1
        else:
            relevance_score = 2.5
        
        score += relevance_score

        # Clamp final score to 0-100 range
        score = max(0, min(100, score))
        
        return round(score, 2)

    except Exception:
        # Fallback: return a minimal score based on length
        try:
            return min(30, len(response.split()) * 0.5) if response else 0.0
        except Exception:
            return 0.0