def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using a
    sentence-level analysis approach. This variant analyzes each sentence
    individually for claim strength, hedging, and epistemic markers, then
    builds a composite score based on the distribution of sentence-level
    calibration scores.
    
    Different from other variants: uses sentence-level claim classification,
    ratio-based scoring of claim types, query complexity detection, and
    a coherence model based on epistemic flow between sentences.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 1.0
        
        # Split into sentences using multiple delimiters
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 1.0
        
        # ---- Query complexity analysis ----
        # Detect if query is about subjective, debatable, or uncertain topics
        query_lower = query.lower()
        
        subjective_query_markers = [
            'opinion', 'think', 'ethical', 'moral', 'should', 'best', 'worst',
            'argument', 'debate', 'controversial', 'perspective', 'view',
            'philosophy', 'believe', 'wrong', 'right', 'impact', 'effect',
            'how has', 'what does', 'imagine', 'likely', 'what went wrong',
            'is there', 'contemplating', 'trying to gauge', 'recommend'
        ]
        
        factual_query_markers = [
            'sql', 'code', 'function', 'create table', 'select', 'algorithm',
            'calculate', 'formula', 'syntax', 'compile', 'error', 'bug',
            'implement', 'database', 'api'
        ]
        
        query_subjectivity = sum(1 for m in subjective_query_markers if m in query_lower)
        query_factuality = sum(1 for m in factual_query_markers if m in query_lower)
        
        is_subjective_query = query_subjectivity > query_factuality
        is_factual_query = query_factuality > query_subjectivity + 1
        
        # ---- Sentence-level classification ----
        # Classify each sentence into categories based on epistemic strength
        
        # Strong hedging / uncertainty markers (per sentence)
        hedge_patterns = [
            r'\b(might|may|could|perhaps|possibly|potentially)\b',
            r'\b(likely|unlikely|probably|presumably)\b',
            r'\b(suggests?|indicates?|implies?|appears?)\b',
            r'\b(seems?\s+to|tends?\s+to|appears?\s+to)\b',
            r'\b(research\s+suggests?|studies?\s+(show|suggest|indicate))\b',
            r'\b(in\s+my\s+(experience|opinion|view))\b',
            r'\b(generally|typically|often|usually|sometimes|occasionally)\b',
            r'\b(to\s+some\s+(extent|degree))\b',
            r'\b(it\'?s?\s+(possible|plausible|conceivable))\b',
            r'\b(one\s+(could|might|may)\s+argue)\b',
            r'\b(arguabl[ey]|debatabl[ey])\b',
            r'\b(not\s+necessarily|not\s+always|not\s+entirely)\b',
        ]
        
        # Overconfident / absolutist markers
        overconfident_patterns = [
            r'\b(always|never|certainly|definitely|absolutely|undoubtedly)\b',
            r'\b(obviously|clearly|without\s+doubt|no\s+question)\b',
            r'\b(everyone\s+knows|it\'?s?\s+a\s+fact|the\s+truth\s+is)\b',
            r'\b(guaranteed|proven\s+fact|indisputable)\b',
            r'\b(there\s+is\s+no\s+way|impossible\s+that|cannot\s+be)\b',
            r'\b(must\s+be|has\s+to\s+be|is\s+definitely)\b',
        ]
        
        # Evidence/source reference patterns
        evidence_patterns = [
            r'\b(according\s+to|based\s+on|as\s+(noted|mentioned|described))\b',
            r'\b(for\s+example|for\s+instance|such\s+as|e\.g\.)\b',
            r'\b(research|study|studies|data|evidence|findings)\b',
            r'\b(historically|traditionally|in\s+practice)\b',
            r'\b(source|reference|cited|documented)\b',
            r'\b(as\s+you\s+(possibly|probably|may)\s+know)\b',
        ]
        
        # Acknowledgment of complexity/nuance
        nuance_patterns = [
            r'\b(however|on\s+the\s+other\s+hand|that\s+said|although)\b',
            r'\b(it\s+depends|depends\s+on|varies|variable)\b',
            r'\b(both|multiple|several|various)\s+(perspectives?|views?|factors?|aspects?)\b',
            r'\b(nuanc|complex|complicated|multifaceted)\b',
            r'\b(trade-?off|balance|tension\s+between)\b',
            r'\b(while|whereas|conversely|alternatively)\b',
            r'\b(in\s+some\s+cases|in\s+certain|under\s+certain)\b',
        ]
        
        # Personal experience / anecdotal markers (can be good for appropriate topics)
        personal_patterns = [
            r'\b(in\s+my\s+experience|personally|from\s+what\s+i\'?ve?\s+seen)\b',
            r'\b(i\s+(think|believe|feel|suspect|imagine|reckon))\b',
            r'\b(i\'?ve?\s+(found|noticed|observed|seen))\b',
            r'\b(from\s+my\s+perspective|in\s+my\s+view)\b',
        ]
        
        def count_pattern_matches(text, patterns):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text, re.IGNORECASE))
            return count
        
        # Analyze each sentence
        sentence_scores = []
        hedge_counts = []
        overconfident_counts = []
        evidence_counts = []
        nuance_counts = []
        personal_counts = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            h = count_pattern_matches(sent_lower, hedge_patterns)
            o = count_pattern_matches(sent_lower, overconfident_patterns)
            e = count_pattern_matches(sent_lower, evidence_patterns)
            n = count_pattern_matches(sent_lower, nuance_patterns)
            p = count_pattern_matches(sent_lower, personal_patterns)
            
            hedge_counts.append(h)
            overconfident_counts.append(o)
            evidence_counts.append(e)
            nuance_counts.append(n)
            personal_counts.append(p)
            
            # Per-sentence calibration score
            # Positive: hedging, evidence, nuance, appropriate personal markers
            # Negative: overconfidence
            sent_score = (h * 1.5) + (e * 1.2) + (n * 1.8) + (p * 0.5) - (o * 2.0)
            sentence_scores.append(sent_score)
        
        num_sentences = len(sentences)
        
        # ---- Aggregate metrics ----
        total_hedges = sum(hedge_counts)
        total_overconfident = sum(overconfident_counts)
        total_evidence = sum(evidence_counts)
        total_nuance = sum(nuance_counts)
        total_personal = sum(personal_counts)
        
        # Proportion of sentences with at least one hedge
        hedge_sentence_ratio = sum(1 for h in hedge_counts if h > 0) / num_sentences
        
        # Proportion of sentences with overconfident markers
        overconfident_sentence_ratio = sum(1 for o in overconfident_counts if o > 0) / num_sentences
        
        # Proportion of sentences with evidence references
        evidence_sentence_ratio = sum(1 for e in evidence_counts if e > 0) / num_sentences
        
        # Proportion of sentences with nuance markers
        nuance_sentence_ratio = sum(1 for n in nuance_counts if n > 0) / num_sentences
        
        # ---- Epistemic flow analysis ----
        # Check if the response transitions between different epistemic modes
        # (e.g., stating a fact, then hedging, then providing evidence)
        # This is unique to this variant
        
        epistemic_modes = []
        for i in range(num_sentences):
            if hedge_counts[i] > 0 and evidence_counts[i] > 0:
                mode = 'evidenced_hedge'  # Best: hedged claim with evidence
            elif nuance_counts[i] > 0:
                mode = 'nuance'
            elif hedge_counts[i] > 0:
                mode = 'hedge'
            elif evidence_counts[i] > 0:
                mode = 'evidence'
            elif overconfident_counts[i] > 0:
                mode = 'overconfident'
            elif personal_counts[i] > 0:
                mode = 'personal'
            else:
                mode = 'neutral'
        
            epistemic_modes.append(mode)
        
        # Count mode transitions (more variety = better epistemic flow)
        mode_transitions = 0
        for i in range(1, len(epistemic_modes)):
            if epistemic_modes[i] != epistemic_modes[i-1]:
                mode_transitions += 1
        
        transition_rate = mode_transitions / max(1, num_sentences - 1) if num_sentences > 1 else 0
        
        # Count unique modes used
        unique_modes = len(set(epistemic_modes))
        mode_diversity = unique_modes / 7.0  # 7 possible modes
        
        # ---- Claim density analysis ----
        # Count assertive declarative patterns (claims without hedging)
        response_lower = response_text.lower()
        
        declarative_patterns = [
            r'\b(is|are|was|were)\s+(the|a|an)\s+\w+',
            r'\b(this|that|it)\s+(is|was|means|shows|proves)\b',
        ]
        
        declarative_count = count_pattern_matches(response_lower, declarative_patterns)
        
        # Ratio of hedged to total claims
        total_claims_proxy = declarative_count + total_hedges + total_overconfident
        if total_claims_proxy > 0:
            hedge_to_claim_ratio = (total_hedges + total_nuance) / total_claims_proxy
        else:
            hedge_to_claim_ratio = 0.3  # neutral default
        
        # ---- Conditional/qualifying phrase analysis ----
        conditional_patterns = [
            r'\bif\b.*\bthen\b',
            r'\b(assuming|given\s+that|provided\s+that|in\s+the\s+case)\b',
            r'\b(when|whenever|wherever)\b.*\b(may|might|could|can)\b',
        ]
        conditional_count = count_pattern_matches(response_lower, conditional_patterns)
        
        # ---- Question acknowledgment ----
        # Does the response acknowledge question complexity or ask clarifying questions?
        question_marks_in_response = response_text.count('?')
        acknowledges_complexity = bool(re.search(
            r'\b(good\s+question|that\'?s?\s+(complex|complicated|nuanced)|depends\s+on\s+what)\b',
            response_lower
        ))
        
        # ---- Length and depth scoring ----
        word_count = len(response_text.split())
        
        # Moderate length bonus (not too short, not just padding)
        if word_count < 20:
            length_factor = 0.5
        elif word_count < 50:
            length_factor = 0.7
        elif word_count < 150:
            length_factor = 0.9
        elif word_count < 400:
            length_factor = 1.0
        else:
            length_factor = 0.95
        
        # ---- Multiple perspective detection ----
        perspective_markers = re.findall(
            r'\b(on\s+one\s+hand|on\s+the\s+other|another\s+(view|perspective|argument)|'
            r'some\s+(argue|believe|think|say)|others\s+(argue|believe|think|say)|'
            r'alternatively|conversely|in\s+contrast)\b',
            response_lower
        )
        perspective_count = len(perspective_markers)
        
        # ---- Scoring composition ----
        score = 50.0  # Start at midpoint
        
        # 1. Hedging quality (0 to +15)
        # Reward appropriate hedging but not excessive
        if is_factual_query:
            # For factual queries, less hedging needed
            ideal_hedge_ratio = 0.1
        else:
            ideal_hedge_ratio = 0.25
        
        hedge_quality = 1.0 - abs(hedge_sentence_ratio - ideal_hedge_ratio) / max(ideal_hedge_ratio, 0.3)
        hedge_quality = max(0, min(1, hedge_quality))
        score += hedge_quality * 10
        
        # Bonus for having any hedging at all on subjective topics
        if is_subjective_query and total_hedges > 0:
            score += 3
        
        # 2. Overconfidence penalty (0 to -20)
        if is_subjective_query:
            score -= overconfident_sentence_ratio * 20
        else:
            score -= overconfident_sentence_ratio * 10
        
        # 3. Evidence and sourcing (+0 to +12)
        score += min(evidence_sentence_ratio * 15, 12)
        
        # 4. Nuance recognition (+0 to +12)
        score += min(nuance_sentence_ratio * 18, 12)
        
        # 5. Epistemic flow / transitions (+0 to +8)
        score += transition_rate * 5
        score += mode_diversity * 3
        
        # 6. Multiple perspectives (+0 to +6)
        score += min(perspective_count * 2, 6)
        
        # 7. Conditional reasoning (+0 to +5)
        score += min(conditional_count * 1.5, 5)
        
        # 8. Length factor
        score *= length_factor
        
        # 9. Depth and substance bonus
        # Longer, more detailed responses tend to be better calibrated
        if word_count > 80 and num_sentences > 3:
            # Check for substantive content (not just filler)
            unique_words = len(set(response_lower.split()))
            vocab_ratio = unique_words / word_count if word_count > 0 else 0
            if vocab_ratio > 0.4:  # Reasonably diverse vocabulary
                score += 5
        
        # 10. Penalize very short, dismissive responses
        if word_count < 30 and total_hedges == 0 and total_evidence == 0:
            score -= 10
        
        # 11. Bonus for acknowledging complexity
        if acknowledges_complexity:
            score += 3
        
        # 12. Bonus for rhetorical questions that engage with the topic
        if question_marks_in_response > 0 and word_count > 50:
            score += min(question_marks_in_response * 1, 3)
        
        # 13. Personal experience framing (good when appropriate)
        if total_personal > 0 and is_subjective_query:
            score += min(total_personal * 1, 3)
        
        # 14. Mean sentence-level calibration score bonus
        if sentence_scores:
            mean_sent_score = sum(sentence_scores) / len(sentence_scores)
            # Normalize: typical range is -2 to +4
            normalized_mean = (mean_sent_score + 2) / 6.0  # maps [-2,4] to [0,1]
            normalized_mean = max(0, min(1, normalized_mean))
            score += normalized_mean * 8
        
        # 15. Check for "essentially both" type nuanced framing
        both_sides = bool(re.search(
            r'\b(essentially|in\s+a\s+sense|it\'?s?\s+a\s+bit\s+of\s+both|both)\b',
            response_lower
        ))
        if both_sides and is_subjective_query:
            score += 2
        
        # Clamp final score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0