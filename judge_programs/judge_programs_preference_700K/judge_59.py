def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant uses a sentence-level analysis approach:
    - Classifies each sentence by its epistemic stance (certain, hedged, speculative, qualified)
    - Analyzes the ratio and distribution of epistemic markers across the response
    - Rewards appropriate epistemic diversity (mixing certainty with hedging)
    - Penalizes monotonic overconfidence
    - Uses sentence complexity and information density as proxies for thoughtfulness
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 1.0
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 1.0
        
        # === SENTENCE-LEVEL EPISTEMIC CLASSIFICATION ===
        
        # Category 1: Hedging / uncertainty markers (good for uncertain topics)
        hedge_patterns = [
            r'\b(might|may|could|perhaps|possibly|potentially)\b',
            r'\b(it\s+seems?|appears?\s+to|tends?\s+to)\b',
            r'\b(generally|typically|usually|often|sometimes)\b',
            r'\b(likely|unlikely|probable|improbable)\b',
            r'\b(suggest(?:s|ed|ing)?|indicat(?:e|es|ed|ing))\b',
            r'\b(in\s+(?:my|some)\s+(?:experience|opinion|view))\b',
            r'\b(arguably|debatab(?:le|ly))\b',
            r'\b(one\s+(?:could|might|may)\s+argue)\b',
            r'\b(to\s+(?:some|a\s+certain)\s+(?:extent|degree))\b',
            r'\b(not\s+(?:necessarily|always|entirely))\b',
            r'\b(can\s+be|tend(?:s)?\s+to\s+be)\b',
        ]
        
        # Category 2: Evidence-based qualifiers (good)
        evidence_patterns = [
            r'\b(research|studies|evidence|data)\s+(suggest|show|indicat|support)',
            r'\b(according\s+to)\b',
            r'\b(historically|empirically)\b',
            r'\b(in\s+practice|in\s+theory)\b',
            r'\b(the\s+(?:research|literature|evidence)\s+(?:is|shows|suggests))\b',
            r'\b(peer[\s-]reviewed|meta[\s-]analysis)\b',
            r'\b(experts?\s+(?:say|believe|suggest|argue))\b',
            r'\b(there\s+is\s+(?:some|strong|limited|growing)\s+evidence)\b',
        ]
        
        # Category 3: Overconfidence markers (bad)
        overconfidence_patterns = [
            r'\b(obviously|clearly|undoubtedly|undeniably)\b',
            r'\b(everyone\s+knows|it\s+is\s+(?:a\s+)?fact)\b',
            r'\b(always|never|impossible|guaranteed|certainly|definitely)\b',
            r'\b(there\s+is\s+no\s+(?:doubt|question))\b',
            r'\b(without\s+(?:a\s+)?doubt)\b',
            r'\b(absolutely|unquestionably|indisputably)\b',
            r'\b(the\s+truth\s+is|the\s+fact\s+is)\b',
            r'\b(100\s*%|zero\s+chance)\b',
        ]
        
        # Category 4: Nuance / contrast markers (good)
        nuance_patterns = [
            r'\b(however|although|though|nevertheless|nonetheless)\b',
            r'\b(on\s+the\s+other\s+hand|that\s+said|having\s+said\s+that)\b',
            r'\b(while|whereas|conversely)\b',
            r'\b(it\s+depends|depends\s+on)\b',
            r'\b(both|either|neither)\b.*\b(and|or|nor)\b',
            r'\b(trade[\s-]?off|balance|nuanc)\b',
            r'\b(pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b',
            r'\b(on\s+one\s+hand)\b',
            r'\b(there\s+are\s+(?:several|multiple|different)\s+(?:views?|perspectives?|approaches?))\b',
        ]
        
        # Category 5: Acknowledgment of limits (good)
        limits_patterns = [
            r'\b(i\s+(?:don\'t|do\s+not)\s+know)\b',
            r'\b(i\'m\s+not\s+(?:sure|certain))\b',
            r'\b(it\'s\s+(?:hard|difficult)\s+to\s+(?:say|know|tell))\b',
            r'\b(this\s+is\s+(?:just\s+)?(?:my|one)\s+(?:opinion|perspective|take))\b',
            r'\b(i\s+(?:could|may|might)\s+be\s+wrong)\b',
            r'\b(correct\s+me\s+if)\b',
            r'\b(as\s+far\s+as\s+I\s+know)\b',
            r'\b(YMMV|your\s+mileage\s+may\s+vary)\b',
            r'\b(take\s+(?:this|it)\s+with\s+a\s+grain)\b',
        ]
        
        response_lower = response_text.lower()
        
        def count_pattern_matches(text, patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text, re.IGNORECASE))
            return total
        
        hedge_count = count_pattern_matches(response_lower, hedge_patterns)
        evidence_count = count_pattern_matches(response_lower, evidence_patterns)
        overconfidence_count = count_pattern_matches(response_lower, overconfidence_patterns)
        nuance_count = count_pattern_matches(response_lower, nuance_patterns)
        limits_count = count_pattern_matches(response_lower, limits_patterns)
        
        # === SENTENCE-LEVEL ANALYSIS ===
        
        sentence_classifications = []
        for sent in sentences:
            sent_lower = sent.lower()
            sent_class = {
                'hedge': count_pattern_matches(sent_lower, hedge_patterns),
                'evidence': count_pattern_matches(sent_lower, evidence_patterns),
                'overconfident': count_pattern_matches(sent_lower, overconfidence_patterns),
                'nuance': count_pattern_matches(sent_lower, nuance_patterns),
                'limits': count_pattern_matches(sent_lower, limits_patterns),
            }
            sentence_classifications.append(sent_class)
        
        num_sentences = len(sentences)
        
        # Count sentences with each type
        hedge_sentences = sum(1 for sc in sentence_classifications if sc['hedge'] > 0)
        evidence_sentences = sum(1 for sc in sentence_classifications if sc['evidence'] > 0)
        overconfident_sentences = sum(1 for sc in sentence_classifications if sc['overconfident'] > 0)
        nuance_sentences = sum(1 for sc in sentence_classifications if sc['nuance'] > 0)
        limits_sentences = sum(1 for sc in sentence_classifications if sc['limits'] > 0)
        
        # === RESPONSE LENGTH AND DEPTH METRICS ===
        
        words = response_text.split()
        word_count = len(words)
        
        # Unique word ratio (vocabulary richness)
        unique_words = len(set(w.lower() for w in words))
        vocab_richness = unique_words / max(word_count, 1)
        
        # Average sentence length (complexity proxy)
        avg_sent_len = word_count / max(num_sentences, 1)
        
        # Count subordinate clauses (complexity)
        subordinate_markers = len(re.findall(
            r'\b(because|since|although|while|whereas|if|unless|when|where|which|that|who)\b',
            response_lower
        ))
        clause_density = subordinate_markers / max(num_sentences, 1)
        
        # === QUERY ANALYSIS ===
        # Determine if the query is opinion-seeking, factual, or ambiguous
        
        query_lower = query.lower()
        
        opinion_query_signals = len(re.findall(
            r'\b(what\s+do\s+you\s+think|opinion|should\s+i|how\s+(?:do|would)\s+you|'
            r'is\s+it\s+(?:worth|good|bad|better)|recommend|advice|thoughts?\s+on|'
            r'how\s+has\s+it|what\s+does\s+it\s+mean|contemplating|ethical|argument\s+for)\b',
            query_lower
        ))
        
        speculative_query_signals = len(re.findall(
            r'\b(imagine|what\s+if|hypothetical|could\s+it|is\s+there|went\s+wrong|'
            r'what\s+went|why\s+(?:did|does|do)|how\s+(?:did|does))\b',
            query_lower
        ))
        
        factual_query_signals = len(re.findall(
            r'\b(how\s+(?:much|many|long|far)|what\s+is\s+the|when\s+(?:did|was|is)|'
            r'where\s+(?:did|was|is)|who\s+(?:is|was|did))\b',
            query_lower
        ))
        
        # Query type weighting
        is_opinion_query = opinion_query_signals > 0 or speculative_query_signals > 0
        is_factual_query = factual_query_signals > 0 and not is_opinion_query
        
        # === SCORING ===
        
        score = 50.0  # Start at midpoint
        
        # 1. Epistemic diversity score (0-15 points)
        # Reward having multiple types of epistemic markers
        categories_present = sum([
            hedge_count > 0,
            evidence_count > 0,
            nuance_count > 0,
            limits_count > 0,
        ])
        epistemic_diversity = min(categories_present * 3.75, 15.0)
        score += epistemic_diversity
        
        # 2. Hedging quality score (-5 to +10)
        hedge_ratio = hedge_sentences / max(num_sentences, 1)
        if is_opinion_query:
            # For opinion queries, moderate hedging is ideal
            if 0.1 <= hedge_ratio <= 0.5:
                score += 8.0
            elif hedge_ratio > 0.5:
                score += 4.0  # Too much hedging
            elif hedge_ratio > 0:
                score += 5.0
        else:
            # For factual queries, some hedging is still good
            if 0.05 <= hedge_ratio <= 0.3:
                score += 6.0
            elif hedge_ratio > 0.3:
                score += 3.0
            elif hedge_ratio > 0:
                score += 4.0
        
        # 3. Evidence-based reasoning bonus (0-8)
        evidence_ratio = evidence_sentences / max(num_sentences, 1)
        score += min(evidence_ratio * 40, 8.0)
        
        # 4. Overconfidence penalty (-15 to 0)
        overconfidence_ratio = overconfident_sentences / max(num_sentences, 1)
        # Penalize more heavily if many sentences are overconfident
        overconfidence_penalty = min(overconfidence_ratio * 50, 15.0)
        score -= overconfidence_penalty
        
        # Also penalize raw overconfidence count
        score -= min(overconfidence_count * 1.5, 8.0)
        
        # 5. Nuance bonus (0-10)
        nuance_ratio = nuance_sentences / max(num_sentences, 1)
        score += min(nuance_ratio * 50, 10.0)
        
        # 6. Acknowledgment of limits bonus (0-5)
        if limits_count > 0:
            score += min(limits_count * 2.5, 5.0)
        
        # 7. Response substance and depth
        # Longer, more detailed responses tend to be more epistemically careful
        length_score = 0.0
        if word_count < 20:
            length_score = -5.0
        elif word_count < 50:
            length_score = 0.0
        elif word_count < 100:
            length_score = 3.0
        elif word_count < 200:
            length_score = 5.0
        elif word_count < 400:
            length_score = 7.0
        else:
            length_score = 8.0
        score += length_score
        
        # 8. Clause density bonus (complexity of reasoning)
        clause_bonus = min(clause_density * 3, 6.0)
        score += clause_bonus
        
        # 9. Vocabulary richness (moderate bonus)
        if vocab_richness > 0.6:
            score += 3.0
        elif vocab_richness > 0.4:
            score += 1.5
        
        # 10. Check for personal experience / anecdotal framing
        # Sharing personal experience with appropriate framing is epistemically honest
        personal_experience = len(re.findall(
            r'\b(in\s+my\s+experience|personally|from\s+what\s+i\'ve\s+seen|'
            r'i\'ve\s+found|i\s+found|for\s+me|in\s+my\s+case)\b',
            response_lower
        ))
        if personal_experience > 0:
            score += min(personal_experience * 2.0, 4.0)
        
        # 11. Conditional/contextual reasoning
        conditional_markers = len(re.findall(
            r'\b(if\s+you|depending\s+on|in\s+(?:some|certain|many)\s+cases|'
            r'for\s+(?:some|most|many)\s+(?:people|cases|situations)|context|'
            r'it\s+varies|varies\s+by|case[\s-]by[\s-]case)\b',
            response_lower
        ))
        score += min(conditional_markers * 2.0, 6.0)
        
        # 12. Check for multiple perspectives or examples
        perspective_markers = len(re.findall(
            r'\b(for\s+example|for\s+instance|such\s+as|e\.g\.|'
            r'another\s+(?:way|perspective|view|approach|example)|'
            r'some\s+(?:people|argue|say|believe)|others?\s+(?:argue|say|believe|think))\b',
            response_lower
        ))
        score += min(perspective_markers * 2.0, 6.0)
        
        # 13. Detect dismissive or low-effort responses (penalty)
        dismissive_patterns = len(re.findall(
            r'\b(just\s+(?:do|google|look)|simple|easy|duh|obviously)\b',
            response_lower
        ))
        score -= min(dismissive_patterns * 3.0, 9.0)
        
        # 14. Question-asking in response (shows epistemic humility)
        questions_in_response = len(re.findall(r'\?', response_text))
        if questions_in_response > 0:
            score += min(questions_in_response * 1.0, 3.0)
        
        # 15. Structural markers (organized thinking)
        has_structure = 0
        if re.search(r'(?:^|\n)\s*[-*•]\s', response_text):
            has_structure += 1
        if re.search(r'(?:^|\n)\s*\d+[.)]\s', response_text):
            has_structure += 1
        if re.search(r'(?:first|second|third|finally|additionally|moreover|furthermore)', response_lower):
            has_structure += 1
        score += min(has_structure * 1.5, 4.5)
        
        # Clamp to range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0