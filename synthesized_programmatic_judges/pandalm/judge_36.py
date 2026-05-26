def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis:
    - Causal/logical connective density and proper usage
    - Sentence-level progression (topic continuity vs. topic drift)
    - Argument depth (claim → evidence → conclusion pattern)
    - Contradiction detection via negation pattern analysis
    - Repetition penalty (circular reasoning detection)
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

        # =====================================================
        # 1. DISCOURSE CONNECTIVE ANALYSIS (0-20 points)
        # Categorize connectives by their logical function
        # =====================================================
        causal_connectives = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'leads to',
            'causing', 'resulting in', 'for this reason', 'accordingly'
        ]
        contrastive_connectives = [
            'however', 'but', 'although', 'nevertheless', 'on the other hand',
            'in contrast', 'whereas', 'yet', 'despite', 'while',
            'conversely', 'nonetheless', 'even though', 'rather'
        ]
        elaboration_connectives = [
            'for example', 'for instance', 'specifically', 'in particular',
            'such as', 'namely', 'to illustrate', 'in other words',
            'that is', 'indeed', 'in fact'
        ]
        sequential_connectives = [
            'first', 'second', 'third', 'finally', 'next', 'then',
            'subsequently', 'afterward', 'initially', 'lastly',
            'to begin with', 'in addition', 'furthermore', 'moreover'
        ]
        conclusion_connectives = [
            'in conclusion', 'to summarize', 'overall', 'in summary',
            'ultimately', 'in the end', 'to conclude', 'all in all'
        ]

        resp_lower = response.lower()

        def count_connectives(connective_list):
            count = 0
            for c in connective_list:
                count += len(re.findall(r'\b' + re.escape(c) + r'\b', resp_lower))
            return count

        causal_count = count_connectives(causal_connectives)
        contrastive_count = count_connectives(contrastive_connectives)
        elaboration_count = count_connectives(elaboration_connectives)
        sequential_count = count_connectives(sequential_connectives)
        conclusion_count = count_connectives(conclusion_connectives)

        total_connectives = causal_count + contrastive_count + elaboration_count + sequential_count + conclusion_count
        
        # Diversity of connective types used
        types_used = sum(1 for c in [causal_count, contrastive_count, elaboration_count, 
                                      sequential_count, conclusion_count] if c > 0)
        
        # Connective density normalized by sentence count
        if num_sentences > 1:
            connective_density = total_connectives / (num_sentences - 1)
        else:
            connective_density = total_connectives
        
        # Ideal density is around 0.3-0.7 per inter-sentence gap
        density_score = min(connective_density * 8, 12)
        type_diversity_score = types_used * 1.6  # up to 8 points
        
        discourse_score = min(density_score + type_diversity_score, 20)
        score += discourse_score

        # =====================================================
        # 2. SENTENCE-LEVEL TOPIC CONTINUITY (0-15 points)
        # Measure how well each sentence connects to the previous
        # using content word overlap between adjacent sentences
        # =====================================================
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'or', 'not',
            'no', 'but', 'if', 'than', 'that', 'this', 'these', 'those', 'it',
            'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'him', 'her', 'his', 'they', 'them', 'their', 'which', 'who', 'whom',
            'what', 'where', 'when', 'how', 'why', 'all', 'each', 'every', 'both',
            'more', 'most', 'other', 'some', 'such', 'also', 'so', 'very', 'just',
            'about', 'up', 'out', 'then', 'there', 'here', 'only', 'over'
        }

        def get_content_words(text):
            w = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(w) - stopwords

        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, num_sentences):
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    # We want moderate overlap (0.1-0.5), not too high (repetitive) 
                    # and not too low (disconnected)
                    raw_ratio = overlap / union if union > 0 else 0
                    # Bell curve centered around 0.25
                    continuity = math.exp(-((raw_ratio - 0.25) ** 2) / (2 * 0.15 ** 2))
                    continuity_scores.append(continuity)
                else:
                    continuity_scores.append(0.3)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores)
            score += avg_continuity * 15
        else:
            score += 5  # Single sentence gets partial credit

        # =====================================================
        # 3. ARGUMENT DEPTH via claim-evidence-conclusion (0-20 points)
        # Detect patterns: assertion → support → synthesis
        # =====================================================
        
        # Claim indicators
        claim_patterns = [
            r'\b(is|are|was|were)\b.*\b(important|essential|crucial|necessary|significant|key)\b',
            r'\b(means|suggests|implies|indicates|shows|demonstrates)\b',
            r'\b(can be|could be|should be|must be|would be)\b',
            r'\bthe\s+\w+\s+(is|are|was|were)\b',
        ]
        
        # Evidence/support indicators  
        evidence_patterns = [
            r'\bfor (example|instance)\b',
            r'\bsuch as\b',
            r'\bincluding\b',
            r'\bspecifically\b',
            r'\b(research|studies|data|evidence)\s+(shows?|suggests?|indicates?)\b',
            r'\baccording to\b',
            r'\bthis (means|shows|demonstrates|illustrates)\b',
        ]
        
        # Synthesis/conclusion indicators
        synthesis_patterns = [
            r'\b(therefore|thus|hence|consequently|accordingly)\b',
            r'\b(in conclusion|overall|to summarize|in summary)\b',
            r'\b(this (means|shows|demonstrates) that)\b',
            r'\b(as a result|for this reason)\b',
        ]

        def count_patterns(patterns, text):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text.lower()))
            return count

        claim_count = count_patterns(claim_patterns, response)
        evidence_count_val = count_patterns(evidence_patterns, response)
        synthesis_count = count_patterns(synthesis_patterns, response)

        # Reward having all three components
        has_claim = min(claim_count, 3)
        has_evidence = min(evidence_count_val, 3)
        has_synthesis = min(synthesis_count, 2)
        
        argument_components = has_claim + has_evidence + has_synthesis
        
        # Bonus for having all three types present
        all_present_bonus = 0
        if claim_count > 0 and evidence_count_val > 0 and synthesis_count > 0:
            all_present_bonus = 5
        elif claim_count > 0 and (evidence_count_val > 0 or synthesis_count > 0):
            all_present_bonus = 2
        
        argument_depth_score = min(argument_components * 2 + all_present_bonus, 20)
        score += argument_depth_score

        # =====================================================
        # 4. REPETITION / CIRCULAR REASONING PENALTY (0 to -15)
        # =====================================================
        
        # Check for repeated phrases (3+ word sequences)
        if num_words >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            unique_trigram_ratio = 1 - (repeated_trigrams / max(len(trigrams), 1))
            
            # Check for repeated sentences
            sentence_texts = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in sentences]
            sentence_counts = Counter(sentence_texts)
            repeated_sentences = sum(c - 1 for c in sentence_counts.values() if c > 1)
            
            # Check for repeated words (excessive)
            word_counts = Counter(words)
            # Exclude stopwords from repetition check
            content_word_counts = {w: c for w, c in word_counts.items() if w not in stopwords}
            if content_word_counts:
                max_word_freq = max(content_word_counts.values())
                avg_word_freq = sum(content_word_counts.values()) / len(content_word_counts)
                word_rep_ratio = max_word_freq / max(num_words, 1)
            else:
                word_rep_ratio = 0
                avg_word_freq = 0

            repetition_penalty = 0
            repetition_penalty += min(repeated_trigrams * 0.5, 5)
            repetition_penalty += min(repeated_sentences * 3, 6)
            if word_rep_ratio > 0.1:
                repetition_penalty += min((word_rep_ratio - 0.1) * 30, 4)
            
            score -= repetition_penalty
        
        # =====================================================
        # 5. STRUCTURAL COMPLETENESS (0-15 points)
        # Does the response have a clear beginning, middle, end?
        # =====================================================
        
        if num_sentences >= 3:
            first_sent = sentences[0].lower()
            last_sent = sentences[-1].lower()
            
            # Opening: Does first sentence introduce the topic?
            # Check if it references key query terms
            query_content = get_content_words(query) if query else set()
            first_content = get_content_words(first_sent)
            
            intro_relevance = len(query_content & first_content) / max(len(query_content), 1) if query_content else 0.5
            intro_score = min(intro_relevance * 5, 5)
            
            # Middle: Multiple sentences developing ideas
            middle_score = min((num_sentences - 2) * 1.5, 5)
            
            # Ending: Does last sentence provide closure?
            closure_indicators = [
                r'\b(overall|therefore|thus|in conclusion|ultimately|in summary)\b',
                r'\b(important|essential|crucial|significant)\b',
                r'\b(can|could|should|will|would)\b.*\b(help|improve|make|create|lead)\b',
            ]
            has_closure = any(re.search(p, last_sent) for p in closure_indicators)
            end_score = 3 if has_closure else 1
            
            structure_score = intro_score + middle_score + end_score
        elif num_sentences == 2:
            structure_score = 6
        else:
            structure_score = 3
        
        score += min(structure_score, 15)

        # =====================================================
        # 6. INTERNAL CONTRADICTION DETECTION (0 to -10)
        # =====================================================
        
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', "n't", 'cannot', "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't", "shouldn't"}
        
        contradiction_penalty = 0
        if num_sentences >= 2:
            for i in range(num_sentences):
                for j in range(i+1, min(i+3, num_sentences)):
                    words_i = set(re.findall(r'\b[a-zA-Z]+\b', sentences[i].lower()))
                    words_j = set(re.findall(r'\b[a-zA-Z]+\b', sentences[j].lower()))
                    
                    content_i = words_i - stopwords
                    content_j = words_j - stopwords
                    
                    neg_i = bool(words_i & negation_words)
                    neg_j = bool(words_j & negation_words)
                    
                    # If sentences share many content words but differ in negation
                    if content_i and content_j:
                        shared = len(content_i & content_j)
                        overlap_ratio = shared / min(len(content_i), len(content_j))
                        
                        if overlap_ratio > 0.5 and neg_i != neg_j:
                            # Possible contradiction - but only penalize if not using contrastive connective
                            sent_j_text = sentences[j].lower()
                            has_contrast = any(c in sent_j_text for c in ['however', 'but', 'although', 'while', 'whereas', 'on the other hand'])
                            if not has_contrast:
                                contradiction_penalty += 3 * overlap_ratio
        
        score -= min(contradiction_penalty, 10)

        # =====================================================
        # 7. RESPONSE SUBSTANTIVENESS (0-15 points)
        # Longer, more detailed responses with varied vocabulary
        # =====================================================
        
        # Length reward (diminishing returns)
        length_score = min(math.log(num_words + 1) * 2, 8)
        
        # Vocabulary richness (type-token ratio adjusted for length)
        unique_words = len(set(words))
        if num_words > 0:
            # Use root TTR to normalize for length
            ttr = unique_words / math.sqrt(num_words)
            vocab_score = min(ttr * 1.2, 7)
        else:
            vocab_score = 0
        
        substantive_score = length_score + vocab_score
        score += min(substantive_score, 15)

        # =====================================================
        # 8. QUERY RELEVANCE BONUS (0-5 points)
        # =====================================================
        if query:
            query_words = get_content_words(query)
            response_words = get_content_words(response)
            if query_words:
                relevance = len(query_words & response_words) / len(query_words)
                score += min(relevance * 5, 5)
            else:
                score += 2.5

        # Normalize to 0-100 range
        # Max theoretical: 20 + 15 + 20 + 15 + 15 + 5 = 90, plus no penalties
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)

    except Exception:
        return 5.0