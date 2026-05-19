def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a substantially different approach:
    - N-gram based analysis for citation/reference patterns
    - Epistemic stance detection (certainty vs uncertainty calibration)
    - Red-flag pattern detection (hallucination indicators)
    - Structural coherence scoring via transition/discourse markers
    - Specificity-to-hedging ratio analysis
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = re.findall(r'\b[a-z]+\b', response_lower)
        
        if len(words) < 3:
            return 1.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if not sentences:
            return 1.0
        
        score = 5.0  # Start at midpoint
        
        # ============================================================
        # FEATURE 1: Epistemic Calibration Score
        # Measures whether the response appropriately calibrates 
        # certainty vs uncertainty using epistemic markers
        # ============================================================
        
        # Appropriate hedging phrases (good - shows epistemic awareness)
        appropriate_hedges = [
            r'\bit\'?s\s+(completely\s+)?(understandable|natural|normal|okay|fine)\b',
            r'\b(can|could|may|might)\s+help\b',
            r'\b(consider|perhaps|possibly|potentially)\b',
            r'\b(it\s+seems?|it\s+appears?)\b',
            r'\b(in\s+general|generally|typically|usually|often)\b',
            r'\b(one\s+way|one\s+approach|an?\s+option)\b',
            r'\b(depending\s+on|it\s+depends)\b',
            r'\bremember\s+(that|to)\b',
            r'\bkeep\s+in\s+mind\b',
        ]
        
        hedge_count = 0
        for pattern in appropriate_hedges:
            hedge_count += len(re.findall(pattern, response_lower))
        
        # Overconfident absolute claims (bad - hallucination red flags)
        absolute_claims = [
            r'\b(always|never|definitely|absolutely|certainly|undoubtedly)\b',
            r'\b(guaranteed|100\s*%|proven\s+fact)\b',
            r'\b(everyone\s+knows|obviously|clearly)\b',
            r'\b(without\s+(a\s+)?doubt|no\s+question)\b',
            r'\b(the\s+only\s+way|the\s+best\s+way)\b',
            r'\b(impossible|inevitable)\b',
        ]
        
        absolute_count = 0
        for pattern in absolute_claims:
            absolute_count += len(re.findall(pattern, response_lower))
        
        # Calibration: reward hedging, penalize overconfidence
        hedge_ratio = hedge_count / max(len(sentences), 1)
        absolute_ratio = absolute_count / max(len(sentences), 1)
        
        # Good calibration: some hedging, not too many absolutes
        calibration_score = min(hedge_ratio * 1.5, 1.5) - absolute_ratio * 1.0
        score += calibration_score
        
        # ============================================================
        # FEATURE 2: Discourse Coherence via Transition Markers
        # Good responses use logical connectors showing structured thinking
        # ============================================================
        
        # Categorized discourse markers (different from simple bullet detection)
        causal_markers = [
            r'\b(because|since|therefore|thus|hence|consequently|as\s+a\s+result)\b',
            r'\b(this\s+(means|implies|suggests|indicates))\b',
            r'\b(due\s+to|owing\s+to|thanks\s+to)\b',
        ]
        
        contrastive_markers = [
            r'\b(however|although|though|nevertheless|nonetheless|on\s+the\s+other\s+hand)\b',
            r'\b(but|yet|still|instead|rather)\b',
            r'\b(despite|in\s+spite\s+of|while)\b',
        ]
        
        elaboration_markers = [
            r'\b(for\s+(example|instance)|such\s+as|specifically|in\s+particular)\b',
            r'\b(moreover|furthermore|additionally|in\s+addition)\b',
            r'\b(in\s+other\s+words|that\s+is|namely)\b',
        ]
        
        sequential_markers = [
            r'\b(first(ly)?|second(ly)?|third(ly)?|finally|next|then|afterward)\b',
            r'\b(to\s+begin|to\s+start|moving\s+on)\b',
        ]
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_markers)
        contrastive_count = sum(len(re.findall(p, response_lower)) for p in contrastive_markers)
        elaboration_count = sum(len(re.findall(p, response_lower)) for p in elaboration_markers)
        sequential_count = sum(len(re.findall(p, response_lower)) for p in sequential_markers)
        
        # Variety of discourse markers used (not just count but diversity)
        marker_types_used = sum([
            1 if causal_count > 0 else 0,
            1 if contrastive_count > 0 else 0,
            1 if elaboration_count > 0 else 0,
            1 if sequential_count > 0 else 0,
        ])
        
        total_markers = causal_count + contrastive_count + elaboration_count + sequential_count
        marker_density = total_markers / max(len(sentences), 1)
        
        # Reward diversity of marker types and reasonable density
        coherence_score = marker_types_used * 0.2 + min(marker_density * 0.5, 0.8)
        score += coherence_score
        
        # ============================================================
        # FEATURE 3: Empathy & Engagement Signals
        # For many queries, good responses acknowledge the user's situation
        # ============================================================
        
        # Acknowledgment patterns
        acknowledgment_patterns = [
            r'\b(i\s+(understand|hear|see|can\s+see|can\s+hear|recognize))\b',
            r'\b(i\'?m\s+(sorry|glad|happy)\s+to\s+hear)\b',
            r'\b(that\'?s?\s+(understandable|valid|reasonable|a\s+great))\b',
            r'\b(your\s+(feelings?|concerns?|frustration|experience))\b',
            r'\b(it\'?s\s+(okay|fine|natural|normal|perfectly))\b',
        ]
        
        ack_count = sum(len(re.findall(p, response_lower)) for p in acknowledgment_patterns)
        
        # Check if query seems to need empathy
        emotional_query_signals = [
            r'\b(feeling|frustrated|stressed|sad|struggling|difficult|heartbroken|devastated)\b',
            r'\b(help|comfort|advice|support|cope)\b',
            r'\b(problem|issue|trouble|concern)\b',
        ]
        
        query_emotional = sum(len(re.findall(p, query_lower)) for p in emotional_query_signals)
        
        if query_emotional > 0:
            # Query needs empathy - reward acknowledgment
            empathy_score = min(ack_count * 0.4, 1.2)
            score += empathy_score
        else:
            # Non-emotional query - slight reward for acknowledgment, less weight
            score += min(ack_count * 0.15, 0.4)
        
        # ============================================================
        # FEATURE 4: Specificity & Actionability Analysis
        # Good responses provide concrete, actionable content
        # ============================================================
        
        # Detect specific/concrete language (numbers, named entities patterns, action verbs)
        number_pattern = re.findall(r'\b\d+\.?\d*\b', response)
        number_count = len(number_pattern)
        
        # Action-oriented language
        action_patterns = [
            r'\b(try|start|begin|create|make|build|write|use|apply|implement)\b',
            r'\b(step\s+\d|here\'?s?\s+how|here\s+are|you\s+can|you\s+could|you\s+might)\b',
            r'\b(technique|method|approach|strategy|tip|suggestion)\b',
        ]
        
        action_count = sum(len(re.findall(p, response_lower)) for p in action_patterns)
        
        # Vague/filler language (penalize)
        vague_patterns = [
            r'\b(stuff|things?|something|somehow|whatever|etc)\b',
            r'\b(kind\s+of|sort\s+of|like|basically|just)\b',
            r'\b(you\s+know|i\s+mean|i\s+guess)\b',
        ]
        
        vague_count = sum(len(re.findall(p, response_lower)) for p in vague_patterns)
        
        specificity_ratio = (action_count + number_count * 0.5) / max(vague_count + 1, 1)
        specificity_score = min(math.log1p(specificity_ratio) * 0.5, 1.0)
        score += specificity_score
        
        # ============================================================
        # FEATURE 5: Dismissiveness Detection (strong negative signal)
        # Responses that dismiss the user's concerns are low quality
        # ============================================================
        
        dismissive_patterns = [
            r'\b(just\s+(get\s+over|deal\s+with|move\s+on|stop|forget))\b',
            r'\b(you\s+should\s+(be\s+able|know|handle))\b',
            r'\b(it\'?s\s+(not\s+a\s+big\s+deal|no\s+big\s+deal|not\s+that\s+bad))\b',
            r'\b(maybe\s+you\'?re?\s+(just|not))\b',
            r'\b(get\s+(yourself|it)\s+together)\b',
            r'\b(stop\s+(being|feeling|worrying))\b',
        ]
        
        dismissive_count = sum(len(re.findall(p, response_lower)) for p in dismissive_patterns)
        score -= dismissive_count * 0.8
        
        # ============================================================
        # FEATURE 6: Response Completeness & Depth
        # Using information-theoretic approach: character bigram entropy
        # as a proxy for content richness (different from word-level diversity)
        # ============================================================
        
        # Character trigram entropy (measures linguistic richness differently)
        if len(response) >= 10:
            trigrams = [response_lower[i:i+3] for i in range(len(response_lower) - 2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            trigram_entropy = 0.0
            for count in trigram_counts.values():
                p = count / total_trigrams
                if p > 0:
                    trigram_entropy -= p * math.log2(p)
            
            # Normalize entropy (typical range 5-10 for English text)
            normalized_entropy = min(trigram_entropy / 10.0, 1.0)
            score += normalized_entropy * 0.8
        
        # ============================================================
        # FEATURE 7: Query Relevance via Semantic Field Matching
        # Extract content words from query and check topical coverage
        # using word stems/roots (poor man's stemming via prefix matching)
        # ============================================================
        
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
        
        # Prefix-based fuzzy matching (first 5 chars)
        query_prefixes = set(w[:5] for w in query_words if len(w) >= 5)
        response_prefixes = set(w[:5] for w in response_words if len(w) >= 5)
        
        if query_prefixes:
            prefix_overlap = len(query_prefixes & response_prefixes) / len(query_prefixes)
            score += prefix_overlap * 0.8
        
        # ============================================================
        # FEATURE 8: Tone Consistency & Professionalism
        # Detect if response maintains appropriate register
        # ============================================================
        
        # Sensationalism / conspiracy red flags
        sensational_patterns = [
            r'\b(shocking|unbelievable|mind-?blowing|insane|crazy)\b',
            r'\b(they\s+don\'?t\s+want\s+you\s+to\s+know)\b',
            r'\b(secret|hidden\s+truth|cover-?up|conspiracy)\b',
            r'\b(wake\s+up|sheeple|mainstream\s+media)\b',
        ]
        
        sensational_count = sum(len(re.findall(p, response_lower)) for p in sensational_patterns)
        score -= sensational_count * 1.0
        
        # ============================================================
        # FEATURE 9: Structural Sophistication
        # Check for enumerated points, explanatory structure
        # Using regex for numbered items and structured explanations
        # ============================================================
        
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        has_structure = len(numbered_items) >= 2
        
        # Check for explanatory patterns (X: Y or X - Y format)
        explanatory_patterns = re.findall(r'[A-Z][a-z]+(?:\s[a-z]+)*\s*[:–—-]\s', response)
        
        if has_structure:
            score += 0.5
        if len(explanatory_patterns) >= 2:
            score += 0.3
        
        # ============================================================
        # FEATURE 10: Sentence Complexity Distribution
        # Good responses have varied sentence lengths (not all same)
        # Using coefficient of variation of sentence word counts
        # ============================================================
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len  # coefficient of variation
                
                # Moderate variation is good (0.3-0.7 range)
                if 0.2 <= cv <= 0.8:
                    score += 0.4
                elif cv < 0.2:
                    # Too uniform - robotic
                    score -= 0.2
                # Very high CV is okay (mix of short and long)
        
        # ============================================================
        # FEATURE 11: Response adequacy (not too short, not padded)
        # ============================================================
        
        word_count = len(words)
        if word_count < 20:
            score -= 1.0
        elif word_count < 40:
            score -= 0.3
        
        # Check for padding/repetition using unique word ratio in sliding windows
        if word_count >= 20:
            window_size = 20
            min_unique_ratio = 1.0
            for i in range(0, word_count - window_size + 1, 10):
                window = words[i:i + window_size]
                unique_ratio = len(set(window)) / len(window)
                min_unique_ratio = min(min_unique_ratio, unique_ratio)
            
            if min_unique_ratio < 0.4:
                score -= 0.5  # Repetitive window detected
        
        # ============================================================
        # FEATURE 12: Negative capability indicators
        # Ability to say "I don't know" or ask for clarification when appropriate
        # ============================================================
        
        ambiguity_in_query = bool(re.search(
            r'\b(ambiguous|unclear|vague|no\s+(previous\s+)?context)\b', query_lower
        ))
        
        clarification_patterns = [
            r'\b(could\s+you\s+(provide|give|share|clarify))\b',
            r'\b(can\s+you\s+(specify|tell\s+me\s+more|elaborate))\b',
            r'\b(without\s+(further|more)\s+(details?|information|context))\b',
            r'\b(what\s+(exactly|specifically))\b',
            r'\b(more\s+information|more\s+details)\b',
        ]
        
        clarification_count = sum(len(re.findall(p, response_lower)) for p in clarification_patterns)
        
        if ambiguity_in_query and clarification_count > 0:
            score += 1.0  # Appropriately asks for clarification
        elif ambiguity_in_query and clarification_count == 0:
            score -= 0.5  # Failed to recognize ambiguity
        
        # ============================================================
        # Final score normalization to 1-5 range
        # ============================================================
        
        # Clamp to reasonable range
        score = max(1.0, min(score, 9.0))
        
        # Map to 1-5 scale
        final_score = 1.0 + (score - 1.0) * (4.0 / 8.0)
        final_score = max(1.0, min(5.0, round(final_score, 2)))
        
        return final_score
        
    except Exception:
        return 3.0