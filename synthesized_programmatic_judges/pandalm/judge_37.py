def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis:
    - Causal/logical connective density and proper usage
    - Sentence-level progression (topic continuity via subject tracking)
    - Argument depth (claim -> evidence -> conclusion pattern)
    - Contradiction detection via negation pattern analysis
    - Information flow (given-new contract)
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
        # 1. DISCOURSE CONNECTIVE ANALYSIS (weighted by type)
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bfor this reason\b', r'\bit follows\b'
        ]
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b', r'\bwhereas\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bdespite\b',
            r'\byet\b', r'\bconversely\b', r'\bwhile\b', r'\bbut\b',
            r'\binstead\b', r'\brather than\b'
        ]
        elaborative_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bnamely\b', r'\bthat is\b'
        ]
        sequential_connectives = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bin conclusion\b', r'\bto summarize\b',
            r'\blastly\b', r'\bin summary\b'
        ]

        resp_lower = response.lower()

        def count_patterns(patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, resp_lower))
            return total

        causal_count = count_patterns(causal_connectives)
        contrastive_count = count_patterns(contrastive_connectives)
        elaborative_count = count_patterns(elaborative_connectives)
        sequential_count = count_patterns(sequential_connectives)

        # Weighted connective score - causal and contrastive indicate stronger logic
        connective_raw = (causal_count * 3.0 + contrastive_count * 2.5 +
                          elaborative_count * 1.5 + sequential_count * 2.0)
        
        # Normalize by number of sentences
        connective_density = connective_raw / max(num_sentences, 1)
        # Score: diminishing returns, max around 15 points
        connective_score = min(15.0, connective_density * 5.0)
        score += connective_score

        # ============================================================
        # 2. SENTENCE-TO-SENTENCE TOPIC CONTINUITY (Centering Theory inspired)
        # ============================================================
        def get_content_words(text):
            stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'as', 'into', 'through', 'during', 'before', 'after', 'it',
                    'its', 'this', 'that', 'these', 'those', 'and', 'or', 'but',
                    'not', 'no', 'if', 'than', 'so', 'very', 'just', 'about',
                    'up', 'out', 'them', 'they', 'their', 'he', 'she', 'his',
                    'her', 'we', 'our', 'you', 'your', 'i', 'my', 'me'}
            w = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            return set(w) - stop

        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, num_sentences):
                prev_content = get_content_words(sentences[i - 1])
                curr_content = get_content_words(sentences[i])
                if prev_content and curr_content:
                    overlap = len(prev_content & curr_content)
                    union = len(prev_content | curr_content)
                    # Modified: weight by overlap count not just ratio
                    continuity = (overlap / max(union, 1)) * 0.5 + min(overlap / 3.0, 1.0) * 0.5
                    continuity_scores.append(continuity)
                else:
                    continuity_scores.append(0.0)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores)
            # Check for continuity variance (too uniform might be repetitive)
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                var_c = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                # Some variance is good (not monotonous)
                variance_bonus = min(math.sqrt(var_c) * 2.0, 1.0)
            else:
                variance_bonus = 0.0
            
            continuity_score = avg_continuity * 12.0 + variance_bonus * 2.0
            score += min(15.0, continuity_score)
        else:
            # Single sentence gets partial credit
            score += 3.0

        # ============================================================
        # 3. ARGUMENT DEPTH: Claim-Evidence-Conclusion pattern detection
        # ============================================================
        # Detect claim markers
        claim_patterns = [
            r'\b(?:is|are|was|were)\s+(?:a|an|the)\s+',
            r'\b(?:means|suggests|implies|indicates|shows|demonstrates)\b',
            r'\b(?:important|essential|crucial|necessary|significant)\b',
            r'\b(?:can be|could be|should be|must be)\b'
        ]
        evidence_patterns = [
            r'\b(?:for example|for instance|such as|including|like)\b',
            r'\b(?:research|studies|data|evidence|according to)\b',
            r'\b(?:specifically|in particular|notably)\b',
            r'\b\d+(?:\.\d+)?%?\b',  # numbers as evidence
        ]
        conclusion_patterns = [
            r'\b(?:therefore|thus|hence|consequently|in conclusion)\b',
            r'\b(?:overall|in summary|to summarize|ultimately)\b',
            r'\b(?:this (?:means|shows|demonstrates|suggests|indicates))\b'
        ]

        claim_count = count_patterns(claim_patterns)
        evidence_count = count_patterns(evidence_patterns)
        conclusion_count = count_patterns(conclusion_patterns)

        # Check if there's a progression: claims early, evidence middle, conclusions late
        has_structure = 0
        if num_sentences >= 3:
            first_third = ' '.join(sentences[:max(1, num_sentences // 3)]).lower()
            last_third = ' '.join(sentences[max(1, num_sentences - num_sentences // 3):]).lower()
            
            first_claims = sum(len(re.findall(p, first_third)) for p in claim_patterns)
            last_conclusions = sum(len(re.findall(p, last_third)) for p in conclusion_patterns)
            
            if first_claims > 0 and last_conclusions > 0:
                has_structure = 3.0
            elif first_claims > 0 or last_conclusions > 0:
                has_structure = 1.5

        depth_raw = min(claim_count, 5) * 0.8 + min(evidence_count, 4) * 1.2 + min(conclusion_count, 3) * 1.5
        depth_score = min(15.0, depth_raw + has_structure)
        score += depth_score

        # ============================================================
        # 4. CONTRADICTION / REPETITION DETECTION
        # ============================================================
        # Check for near-duplicate sentences (sign of poor structure)
        repetition_penalty = 0.0
        if num_sentences >= 2:
            for i in range(num_sentences):
                for j in range(i + 1, num_sentences):
                    w_i = set(re.findall(r'\b[a-zA-Z]+\b', sentences[i].lower()))
                    w_j = set(re.findall(r'\b[a-zA-Z]+\b', sentences[j].lower()))
                    if w_i and w_j:
                        sim = len(w_i & w_j) / max(len(w_i | w_j), 1)
                        if sim > 0.85:
                            repetition_penalty += 4.0
                        elif sim > 0.7:
                            repetition_penalty += 2.0
        
        # Check for word-level repetition (same word repeated excessively)
        word_counts = Counter(words)
        if num_words > 10:
            max_word_freq = max(word_counts.values())
            content_words_list = [w for w in words if w not in {'the', 'a', 'an', 'is', 'are', 'was',
                                                                  'were', 'to', 'of', 'in', 'for', 'and',
                                                                  'or', 'it', 'that', 'this', 'with', 'on',
                                                                  'at', 'by', 'from', 'as', 'be'}]
            if content_words_list:
                content_counts = Counter(content_words_list)
                max_content_freq = max(content_counts.values())
                if max_content_freq > num_words * 0.15:
                    repetition_penalty += min(5.0, (max_content_freq / num_words) * 20.0)

        # Negation contradiction check: look for "X is Y" followed by "X is not Y"
        negation_contradictions = 0
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                s_i = sentences[i].lower()
                s_j = sentences[j].lower()
                # Simple: check if one sentence is roughly the negation of another
                if ' not ' in s_j or "n't" in s_j:
                    s_j_clean = s_j.replace(' not ', ' ').replace("n't", '')
                    w_i = set(re.findall(r'\b[a-zA-Z]+\b', s_i))
                    w_j = set(re.findall(r'\b[a-zA-Z]+\b', s_j_clean))
                    if w_i and w_j:
                        sim = len(w_i & w_j) / max(len(w_i | w_j), 1)
                        if sim > 0.75:
                            negation_contradictions += 1

        contradiction_penalty = negation_contradictions * 3.0
        total_penalty = min(15.0, repetition_penalty + contradiction_penalty)
        score -= total_penalty

        # ============================================================
        # 5. INFORMATION DENSITY AND PROGRESSION
        # ============================================================
        # Track new information introduced per sentence
        if num_sentences >= 2:
            seen_content = set()
            new_info_per_sentence = []
            for s in sentences:
                cw = get_content_words(s)
                new_words = cw - seen_content
                if cw:
                    new_ratio = len(new_words) / len(cw)
                else:
                    new_ratio = 0
                new_info_per_sentence.append(new_ratio)
                seen_content.update(cw)
            
            avg_new_info = sum(new_info_per_sentence) / len(new_info_per_sentence)
            # Good responses introduce new info but not completely disconnected
            # Ideal: moderate new info (0.3-0.7 per sentence)
            info_scores = []
            for ratio in new_info_per_sentence:
                if 0.25 <= ratio <= 0.75:
                    info_scores.append(1.0)
                elif ratio < 0.25:
                    info_scores.append(ratio / 0.25 * 0.5)
                else:
                    info_scores.append(max(0.3, 1.0 - (ratio - 0.75) * 2.0))
            
            info_flow_score = (sum(info_scores) / len(info_scores)) * 10.0
            score += min(12.0, info_flow_score)
        else:
            score += 4.0

        # ============================================================
        # 6. RESPONSE COMPLETENESS AND STRUCTURAL ADEQUACY
        # ============================================================
        # Longer, well-structured responses generally score better for coherence
        # But penalize if too short for the query complexity
        query_words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        query_complexity = len(query_words)
        
        # Check if response seems truncated
        truncation_penalty = 0.0
        if response.rstrip()[-1:] not in '.!?")\']':
            truncation_penalty = 3.0
        if len(response) > 50 and response.rstrip()[-1:] == ',':
            truncation_penalty = 4.0
        
        # Sentence count adequacy
        if num_sentences >= 3:
            structure_bonus = min(5.0, num_sentences * 0.8)
        elif num_sentences == 2:
            structure_bonus = 2.5
        else:
            structure_bonus = 1.0
        
        score += structure_bonus - truncation_penalty

        # ============================================================
        # 7. QUERY RELEVANCE (logical coherence includes staying on topic)
        # ============================================================
        if query_words:
            query_content = get_content_words(query)
            response_content = get_content_words(response)
            if query_content and response_content:
                relevance = len(query_content & response_content) / max(len(query_content), 1)
                relevance_score = min(8.0, relevance * 10.0)
                score += relevance_score
            else:
                score += 2.0
        else:
            score += 2.0

        # ============================================================
        # 8. SENTENCE COMPLEXITY BALANCE
        # ============================================================
        # Good logical writing has varied but not wildly different sentence lengths
        if num_sentences >= 2:
            sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
            avg_len = sum(sent_lengths) / len(sent_lengths)
            
            if avg_len > 0:
                # Coefficient of variation
                std_len = math.sqrt(sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths))
                cv = std_len / avg_len
                
                # Moderate variation is best (0.2-0.5)
                if 0.15 <= cv <= 0.6:
                    complexity_score = 5.0
                elif cv < 0.15:
                    complexity_score = 2.0  # Too uniform
                else:
                    complexity_score = max(1.0, 5.0 - (cv - 0.6) * 5.0)
                
                score += complexity_score
            else:
                score += 1.0

        # Normalize to 0-100 range
        # Max theoretical: ~15 + 15 + 15 + 12 + 5 + 8 + 5 = ~75 (without penalties)
        score = max(0.0, score)
        normalized = min(100.0, score * 1.4)
        
        return round(normalized, 2)

    except Exception:
        return 25.0