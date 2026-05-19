def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker / connective analysis (causal, contrastive, additive, temporal)
    - Argument depth via clause chain analysis
    - Internal consistency checks (contradiction detection)
    - Reasoning pattern detection (if-then, because-therefore, premise-conclusion)
    - Information density and elaboration ratio
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.5
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        # Tokenize into sentences more carefully
        sentence_endings = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response_stripped)
        # Also split on newlines that seem to separate thoughts
        sentences = []
        for s in sentence_endings:
            parts = re.split(r'\n\s*\n', s)
            sentences.extend(parts)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if not sentences:
            return 1.0
        
        num_sentences = len(sentences)
        words = re.findall(r'\b[a-zA-Z]+\b', response_stripped.lower())
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # ============================================================
        # 1. DISCOURSE CONNECTIVE ANALYSIS (weighted by type)
        # ============================================================
        
        # Causal connectives - strongest indicator of logical reasoning
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bit follows\b', r'\baccordingly\b', r'\bfor this reason\b',
            r'\bthat\'s why\b', r'\bthis is why\b', r'\bgiven that\b',
            r'\bin light of\b'
        ]
        
        # Contrastive connectives - show nuanced thinking
        contrastive_connectives = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\bin contrast\b',
            r'\byet\b', r'\bstill\b', r'\bnonetheless\b', r'\beven though\b',
            r'\bthat said\b', r'\bthat being said\b', r'\brather\b',
            r'\binstead\b', r'\bon the contrary\b'
        ]
        
        # Additive/elaboration connectives
        elaboration_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bbesides\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bto illustrate\b', r'\bnotably\b'
        ]
        
        # Sequential/structural connectives
        sequential_connectives = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\bin summary\b', r'\boverall\b',
            r'\bto begin with\b', r'\blast(?:ly)?\b', r'\bin the end\b',
            r'\bultimately\b'
        ]
        
        # Conditional/hypothetical connectives
        conditional_connectives = [
            r'\bif\b.*\bthen\b', r'\bif\b', r'\bassuming\b', r'\bprovided that\b',
            r'\bunless\b', r'\bin case\b', r'\bsuppose\b', r'\bwould\b',
            r'\bcould\b', r'\bmight\b', r'\bwhen\b.*\bthen\b'
        ]
        
        response_lower = response_stripped.lower()
        
        def count_patterns(patterns, text):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text))
            return count
        
        causal_count = count_patterns(causal_connectives, response_lower)
        contrastive_count = count_patterns(contrastive_connectives, response_lower)
        elaboration_count = count_patterns(elaboration_connectives, response_lower)
        sequential_count = count_patterns(sequential_connectives, response_lower)
        conditional_count = count_patterns(conditional_connectives, response_lower)
        
        # Normalize by number of sentences
        norm_factor = max(num_sentences, 1)
        
        # Weighted connective density (per sentence)
        connective_score = (
            causal_count * 3.0 +
            contrastive_count * 2.5 +
            elaboration_count * 1.5 +
            sequential_count * 2.0 +
            conditional_count * 1.8
        ) / norm_factor
        
        # Cap and scale: ideal is around 1-2 connectives per sentence
        connective_score = min(connective_score, 4.0) / 4.0 * 25.0
        
        # ============================================================
        # 2. ARGUMENT CHAIN DEPTH
        # ============================================================
        # Detect chains of reasoning: premise -> inference -> conclusion
        
        # Count reasoning chains (sequences of causal/logical connectives)
        chain_markers = (
            [r'\bbecause\b', r'\bsince\b', r'\bgiven\b', r'\bas\b'] +  # premises
            [r'\btherefore\b', r'\bthus\b', r'\bso\b', r'\bhence\b'] +  # conclusions
            [r'\bwhich means\b', r'\bthis means\b', r'\bit follows\b', r'\bimplying\b']  # inferences
        )
        
        # Find positions of reasoning markers
        marker_positions = []
        for pattern in chain_markers:
            for m in re.finditer(pattern, response_lower):
                marker_positions.append(m.start())
        
        marker_positions.sort()
        
        # Count chains: markers that appear within reasonable proximity
        chain_count = 0
        if len(marker_positions) >= 2:
            for i in range(1, len(marker_positions)):
                gap = marker_positions[i] - marker_positions[i-1]
                if 20 < gap < 500:  # reasonable distance for connected reasoning
                    chain_count += 1
        
        chain_score = min(chain_count, 8) / 8.0 * 15.0
        
        # ============================================================
        # 3. EXPLANATION DEPTH - clause complexity
        # ============================================================
        
        # Subordinate clauses indicate more complex reasoning
        subordinators = [
            r'\bwho\b', r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhile\b', r'\balthough\b', r'\bif\b', r'\bsince\b',
            r'\bbecause\b', r'\bunless\b', r'\bwhereas\b', r'\bwhether\b'
        ]
        
        subordinate_count = count_patterns(subordinators, response_lower)
        subordinate_density = subordinate_count / norm_factor
        
        # Ideal: 1-3 subordinate clauses per sentence
        clause_score = min(subordinate_density, 3.0) / 3.0 * 10.0
        
        # ============================================================
        # 4. INTERNAL CONSISTENCY CHECK
        # ============================================================
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bcan\b', r'\bcannot\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\bshould\b', r'\bshould not\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\beveryone\b', r'\bno one\b'),
            (r'\ball\b', r'\bnone\b'),
        ]
        
        # Simple contradiction penalty: if both extremes appear in close proximity
        contradiction_penalty = 0
        for pos_pat, neg_pat in contradiction_patterns:
            pos_matches = list(re.finditer(pos_pat, response_lower))
            neg_matches = list(re.finditer(neg_pat, response_lower))
            if pos_matches and neg_matches:
                # Check if they're discussing the same topic (within proximity)
                for pm in pos_matches:
                    for nm in neg_matches:
                        dist = abs(pm.start() - nm.start())
                        if dist < 100:
                            # Could be legitimate contrast or contradiction
                            # Check if there's a contrastive marker nearby
                            region_start = min(pm.start(), nm.start())
                            region_end = max(pm.end(), nm.end())
                            region = response_lower[max(0, region_start-30):min(len(response_lower), region_end+30)]
                            has_contrast = any(re.search(p, region) for p in [
                                r'\bhowever\b', r'\bbut\b', r'\balthough\b',
                                r'\bwhile\b', r'\bwhereas\b', r'\brather\b'
                            ])
                            if not has_contrast:
                                contradiction_penalty += 1
        
        contradiction_score = max(0, 5.0 - contradiction_penalty * 2.0)
        
        # ============================================================
        # 5. RESPONSE COMPLETENESS AND STRUCTURE
        # ============================================================
        
        # Measure if response has introduction, body, conclusion pattern
        # Check for topic sentence behavior (first sentence relates to query)
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) if query else set()
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'her', 'was', 'one', 'our', 'out', 'has', 'have', 'had', 'been',
            'this', 'that', 'with', 'they', 'from', 'what', 'which', 'when',
            'how', 'who', 'where', 'why', 'does', 'will', 'would', 'could',
            'should', 'about', 'there', 'their', 'them', 'than', 'then',
            'some', 'into', 'just', 'like', 'more', 'also', 'very', 'much',
            'most', 'other', 'any', 'each', 'every', 'being', 'through'
        }
        query_content_words = query_words - stopwords
        
        # First sentence relevance
        if sentences and query_content_words:
            first_sent_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentences[0].lower()))
            first_sent_overlap = len(first_sent_words & query_content_words) / max(len(query_content_words), 1)
        else:
            first_sent_overlap = 0.3  # neutral
        
        topic_intro_score = min(first_sent_overlap, 0.5) / 0.5 * 5.0
        
        # ============================================================
        # 6. PROGRESSIVE DEVELOPMENT
        # ============================================================
        # Check if new information is introduced progressively
        # (not just repeating the same words)
        
        if num_sentences >= 2:
            cumulative_vocab = set()
            new_word_ratios = []
            for sent in sentences:
                sent_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())) - stopwords
                if sent_words:
                    new_words = sent_words - cumulative_vocab
                    ratio = len(new_words) / max(len(sent_words), 1)
                    new_word_ratios.append(ratio)
                    cumulative_vocab.update(sent_words)
            
            if new_word_ratios:
                # Good responses introduce new concepts throughout
                avg_new_ratio = sum(new_word_ratios) / len(new_word_ratios)
                # Ideal: 30-70% new words per sentence (not too repetitive, not too scattered)
                if avg_new_ratio > 0.7:
                    progressive_score = 8.0  # slightly scattered but still good
                elif avg_new_ratio > 0.3:
                    progressive_score = 10.0  # ideal range
                elif avg_new_ratio > 0.15:
                    progressive_score = 6.0  # somewhat repetitive
                else:
                    progressive_score = 3.0  # very repetitive
            else:
                progressive_score = 5.0
        else:
            progressive_score = 4.0  # single sentence = limited development
        
        # ============================================================
        # 7. SPECIFICITY AND EVIDENCE
        # ============================================================
        
        # Detect specific claims, examples, references
        specificity_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\baccording to\b',
            r'\bresearch\b', r'\bstud(?:y|ies)\b', r'\bevidence\b',
            r'\bdata\b', r'\bstatistic\b', r'\bpercent\b', r'\b\d+%',
            r'\bin \d{4}\b',  # year references
            r'\b[A-Z][a-z]+\'s\b',  # possessive proper nouns (citing people)
        ]
        
        specificity_count = count_patterns(specificity_markers, response_stripped.lower())
        # Also count numbers as evidence of specificity
        number_count = len(re.findall(r'\b\d+\.?\d*\b', response_stripped))
        
        specificity_score = min((specificity_count * 1.5 + number_count * 0.5), 10.0)
        
        # ============================================================
        # 8. RESPONSE LENGTH AND SUBSTANCE
        # ============================================================
        # Longer responses tend to be more developed, but with diminishing returns
        
        length_score = 0
        if num_words < 15:
            length_score = 2.0
        elif num_words < 30:
            length_score = 4.0
        elif num_words < 60:
            length_score = 6.0
        elif num_words < 120:
            length_score = 8.0
        elif num_words < 250:
            length_score = 10.0
        else:
            length_score = 10.0
        
        # ============================================================
        # 9. HEDGING VS ASSERTION BALANCE
        # ============================================================
        # Good arguments balance confident claims with appropriate hedging
        
        strong_assertions = [
            r'\bclearly\b', r'\bobviously\b', r'\bcertainly\b', r'\bdefinitely\b',
            r'\bundoubtedly\b', r'\bwithout doubt\b', r'\bof course\b',
            r'\bthe fact is\b', r'\bthe truth is\b'
        ]
        
        hedging_words = [
            r'\bperhaps\b', r'\bmaybe\b', r'\bpossibly\b', r'\bmight\b',
            r'\bcould\b', r'\btends to\b', r'\bgenerally\b', r'\busually\b',
            r'\boften\b', r'\bsometimes\b', r'\blikely\b', r'\bprobably\b',
            r'\bit seems\b', r'\bappears to\b'
        ]
        
        assertion_count = count_patterns(strong_assertions, response_lower)
        hedge_count = count_patterns(hedging_words, response_lower)
        
        total_modality = assertion_count + hedge_count
        if total_modality > 0:
            # Balance: ideal is having some of both
            balance = 1.0 - abs(assertion_count - hedge_count) / total_modality
            modality_score = (balance * 3.0 + min(total_modality, 5) / 5.0 * 2.0)
        else:
            modality_score = 2.0  # neutral, no modality markers
        
        # ============================================================
        # 10. COHERENT REFERENCE CHAINS
        # ============================================================
        # Check for anaphoric references (this, that, these, those, it, they)
        # which indicate connected discourse
        
        anaphora_patterns = [
            r'\bthis\s+(?:means|implies|suggests|shows|indicates|is)\b',
            r'\bthat\s+(?:means|implies|suggests|shows|indicates|is)\b',
            r'\bthese\s+\w+\b',
            r'\bthose\s+\w+\b',
            r'\bas (?:mentioned|noted|stated|discussed)\b',
            r'\bthe (?:above|former|latter|previous)\b',
            r'\bin this (?:case|way|sense|context)\b',
        ]
        
        anaphora_count = count_patterns(anaphora_patterns, response_lower)
        anaphora_score = min(anaphora_count, 5) / 5.0 * 5.0
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        total_score = (
            connective_score +       # max 25
            chain_score +            # max 15
            clause_score +           # max 10
            contradiction_score +    # max 5
            topic_intro_score +      # max 5
            progressive_score +      # max 10
            specificity_score +      # max 10
            length_score +           # max 10
            modality_score +         # max 5
            anaphora_score           # max 5
        )
        
        # Total max: 100, normalize to 0-10
        final_score = total_score / 10.0
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0