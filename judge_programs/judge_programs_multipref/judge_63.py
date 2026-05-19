def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant focuses on:
    1. Claim density analysis (ratio of assertive claims to total content)
    2. Source/evidence attribution patterns
    3. Conditional/nuanced reasoning detection
    4. Epistemic verb analysis (know vs believe vs suggest)
    5. Quantified uncertainty (use of approximate numbers, ranges)
    6. Discourse structure quality (logical flow markers)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        words = re.findall(r'[a-z]+(?:\'[a-z]+)?', response_lower)
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 5]
        
        if len(words) < 3:
            return 1.0
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # === 1. Epistemic Verb Spectrum Analysis ===
        # Categorize verbs by epistemic strength
        strong_epistemic = [  # high certainty claims
            'is definitely', 'is certainly', 'is absolutely', 'is undoubtedly',
            'without doubt', 'without question', 'there is no doubt',
            'it is clear that', 'it is obvious', 'everyone knows',
            'always', 'never', 'impossible', 'guaranteed', 'proven fact',
            'undeniable', 'unquestionable', 'indisputable'
        ]
        
        moderate_epistemic = [  # appropriate confidence
            'generally', 'typically', 'usually', 'in most cases',
            'commonly', 'often', 'frequently', 'tends to',
            'is known to', 'is recognized', 'is understood',
            'established', 'well-documented', 'widely accepted'
        ]
        
        cautious_epistemic = [  # appropriate hedging
            'may', 'might', 'could', 'possibly', 'perhaps',
            'it seems', 'it appears', 'likely', 'unlikely',
            'suggests', 'indicates', 'implies', 'potentially',
            'in some cases', 'depending on', 'it depends',
            'not necessarily', 'can vary', 'varies'
        ]
        
        evidence_markers = [
            'research', 'studies', 'according to', 'evidence',
            'data', 'findings', 'experts', 'scientists',
            'literature', 'analysis', 'survey', 'experiment',
            'observation', 'reported', 'documented', 'peer-reviewed'
        ]
        
        strong_count = sum(1 for phrase in strong_epistemic if phrase in response_lower)
        moderate_count = sum(1 for phrase in moderate_epistemic if phrase in response_lower)
        cautious_count = sum(1 for phrase in cautious_epistemic if phrase in response_lower)
        evidence_count = sum(1 for phrase in evidence_markers if phrase in response_lower)
        
        # === 2. Conditional Reasoning Depth ===
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*,', r'\bwhen\b.*\bmight\b',
            r'\bassuming\b', r'\bgiven that\b', r'\bprovided that\b',
            r'\bin the case', r'\bdepending on\b', r'\bwhether\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\balthough\b',
            r'\bwhile\b.*\b(also|still|yet)\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bconversely\b', r'\balternatively\b'
        ]
        conditional_count = sum(1 for pat in conditional_patterns 
                               if re.search(pat, response_lower))
        
        # === 3. Quantified Uncertainty Detection ===
        # Look for ranges, approximations, qualified numbers
        approx_patterns = [
            r'approximately\s+\d', r'about\s+\d', r'around\s+\d',
            r'roughly\s+\d', r'estimated', r'\d+\s*-\s*\d+',
            r'between\s+\d+\s+and\s+\d+', r'up to\s+\d+',
            r'at least\s+\d+', r'more than\s+\d+', r'less than\s+\d+'
        ]
        approx_count = sum(1 for pat in approx_patterns 
                          if re.search(pat, response_lower))
        
        # Bare unqualified numbers (potential overconfidence)
        bare_numbers = len(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\s*[-–])', response))
        qualified_numbers = sum(1 for pat in approx_patterns 
                               if re.search(pat, response_lower))
        
        # === 4. Discourse Coherence Markers ===
        discourse_markers = [
            'first', 'second', 'third', 'additionally', 'moreover',
            'furthermore', 'in addition', 'finally', 'in conclusion',
            'to summarize', 'in summary', 'overall', 'specifically',
            'for example', 'for instance', 'such as', 'namely',
            'in particular', 'notably', 'importantly'
        ]
        discourse_count = sum(1 for m in discourse_markers if m in response_lower)
        
        # === 5. Perspective Acknowledgment ===
        perspective_markers = [
            'from one perspective', 'some argue', 'others believe',
            'there are different', 'various viewpoints', 'some people',
            'on one hand', 'on the other hand', 'proponents',
            'critics', 'supporters', 'opponents', 'debate',
            'controversial', 'disputed', 'mixed opinions',
            'different perspectives', 'multiple factors'
        ]
        perspective_count = sum(1 for m in perspective_markers if m in response_lower)
        
        # === 6. Assertiveness Density ===
        # Count declarative assertions per sentence
        assertion_words = ['is', 'are', 'was', 'were', 'will', 'must', 'shall', 'should']
        assertion_count = sum(1 for w in words if w in assertion_words)
        assertion_density = assertion_count / num_words if num_words > 0 else 0
        
        # === 7. Topic Sensitivity Detection ===
        # Some topics inherently need more hedging
        sensitive_topics = [
            'opinion', 'believe', 'think', 'should', 'best', 'worst',
            'politics', 'religion', 'moral', 'ethical', 'controversial',
            'debate', 'health', 'medical', 'diagnosis', 'treatment',
            'future', 'predict', 'forecast', 'will happen'
        ]
        query_lower = query.lower()
        topic_sensitivity = sum(1 for t in sensitive_topics if t in query_lower)
        is_sensitive = topic_sensitivity >= 1
        
        # Factual/procedural queries
        factual_markers = ['how', 'what is', 'explain', 'describe', 'recipe',
                          'steps', 'directions', 'instructions', 'calculate']
        is_factual = any(m in query_lower for m in factual_markers)
        
        # === 8. Self-awareness markers ===
        self_aware = [
            "i'm not sure", "i don't know", "i'm not aware",
            "i cannot confirm", "i may be wrong", "to my knowledge",
            "as far as i know", "i believe", "in my opinion",
            "this is my understanding", "i think"
        ]
        self_aware_count = sum(1 for m in self_aware if m in response_lower)
        
        # === 9. Absolute language detection (penalize) ===
        absolute_terms = [
            'always', 'never', 'every', 'none', 'all', 'nothing',
            'everything', 'completely', 'totally', 'entirely',
            'absolutely', 'perfectly', 'exactly', 'precisely',
            'the only', 'the best', 'the worst', 'no one', 'everyone'
        ]
        # Count but be careful about context
        absolute_count = 0
        for term in absolute_terms:
            matches = re.findall(r'\b' + term + r'\b', response_lower)
            absolute_count += len(matches)
        
        # === SCORING ===
        score = 50.0  # baseline
        
        # Reward cautious epistemic language (scaled by response length)
        cautious_per_100 = (cautious_count / num_words) * 100
        score += min(cautious_per_100 * 15, 12)
        
        # Reward moderate epistemic language
        moderate_per_100 = (moderate_count / num_words) * 100
        score += min(moderate_per_100 * 10, 8)
        
        # Penalize strong overconfident language
        strong_per_100 = (strong_count / num_words) * 100
        score -= min(strong_per_100 * 25, 15)
        
        # Reward evidence attribution
        score += min(evidence_count * 2.0, 8)
        
        # Reward conditional reasoning
        score += min(conditional_count * 1.5, 10)
        
        # Reward quantified uncertainty
        score += min(approx_count * 2.0, 6)
        
        # Reward discourse coherence
        score += min(discourse_count * 0.8, 5)
        
        # Reward perspective acknowledgment
        score += min(perspective_count * 2.0, 8)
        
        # Reward self-awareness
        score += min(self_aware_count * 2.5, 6)
        
        # Penalize absolute language (especially on sensitive topics)
        abs_penalty = absolute_count * (2.0 if is_sensitive else 1.0)
        abs_per_100 = (absolute_count / num_words) * 100
        score -= min(abs_per_100 * 8, 10)
        
        # === Response Structure Quality ===
        # Reward structured responses (numbered lists, sections)
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,4}\s', response)) or bool(re.search(r'\*\*[^*]+\*\*', response))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-•*]\s', response))
        
        structure_score = 0
        if has_numbering:
            structure_score += 2
        if has_headers:
            structure_score += 2
        if has_bullets:
            structure_score += 1
        score += min(structure_score, 4)
        
        # === Response Length Quality ===
        # Moderate length responses tend to be more thoughtful
        if num_words > 50:
            score += 2
        if num_words > 100:
            score += 2
        if num_words > 200:
            score += 1
        
        # === Sentence Complexity ===
        # Longer, more complex sentences can indicate nuanced thinking
        avg_sentence_len = num_words / num_sentences
        if 12 <= avg_sentence_len <= 25:
            score += 3  # good range for nuanced writing
        elif avg_sentence_len > 30:
            score -= 1  # too complex
        elif avg_sentence_len < 8:
            score -= 2  # too simple
        
        # === Topic-adaptive scoring ===
        if is_sensitive:
            # Extra reward for hedging on sensitive topics
            if cautious_count >= 2:
                score += 4
            if perspective_count >= 1:
                score += 3
            # Extra penalty for overconfidence on sensitive topics
            if strong_count > 0 and cautious_count == 0:
                score -= 5
        
        if is_factual:
            # For factual queries, moderate confidence is fine
            if moderate_count >= 1:
                score += 2
            # But still reward acknowledging limitations
            if cautious_count >= 1:
                score += 1
        
        # === Engagement and Helpfulness ===
        # Opening acknowledgment of the query
        engagement_starters = [
            'great question', 'good question', "that's a",
            'certainly', 'of course', 'absolutely', "let's",
            'happy to help', 'glad to help', "here's"
        ]
        has_engagement = any(response_lower.startswith(s) or 
                           response_lower[:50].find(s) >= 0 
                           for s in engagement_starters)
        if has_engagement:
            score += 2
        
        # === Ratio-based calibration score ===
        # Good calibration = mix of confident and hedged language
        total_epistemic = strong_count + moderate_count + cautious_count
        if total_epistemic > 0:
            calibration_ratio = (cautious_count + moderate_count) / total_epistemic
            score += calibration_ratio * 5
        
        # Clamp score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0