def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication.
    
    This variant uses a DIFFERENT approach: analyzing the ratio of claim types,
    detecting evidential reasoning patterns, measuring qualification density
    per assertion, and evaluating discourse-level uncertainty framing.
    
    Key unique features:
    - Assertion density analysis (claims per sentence)
    - Evidential source attribution detection
    - Qualification-to-assertion ratio
    - Discourse marker analysis for reasoning chains
    - Absolutism penalty based on categorical statement detection
    - Topic ambiguity estimation from query features
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        resp_len = len(response)
        
        # Split into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # === FEATURE 1: Assertion Density Analysis ===
        # Count strong declarative assertions (sentences with "is", "are", "was", "will", etc.)
        strong_assertion_patterns = [
            r'\b(?:is|are|was|were)\s+(?:the|a|an)\s+\w+',  # "is the best"
            r'\b(?:always|never|certainly|definitely|absolutely|undoubtedly|without doubt)\b',
            r'\b(?:it is clear|clearly|obviously|of course|no doubt|undeniably)\b',
            r'\b(?:everyone knows|as we all know|it goes without saying)\b',
            r'\b(?:the fact is|the truth is|in fact|actually)\b',
            r'\b(?:must be|has to be|cannot be|could not be)\b',
            r'\b(?:proven|established fact|indisputable|unquestionable)\b',
            r'\b(?:100%|guaranteed|impossible|inevitable)\b',
        ]
        
        strong_assertion_count = 0
        for pat in strong_assertion_patterns:
            strong_assertion_count += len(re.findall(pat, response_lower))
        
        assertion_density = strong_assertion_count / num_sentences
        # Penalize high assertion density (overconfidence)
        assertion_penalty = min(assertion_density * 3.0, 8.0)
        
        # === FEATURE 2: Evidential Source Attribution ===
        # Detect references to evidence, studies, sources
        evidence_patterns = [
            r'\b(?:research|studies|evidence|data|findings)\s+(?:suggest|indicate|show|demonstrate|support)',
            r'\b(?:according to|based on|as reported|as noted)\b',
            r'\b(?:literature|meta-analysis|survey|experiment|trial)\b',
            r'\b(?:experts|researchers|scientists|scholars)\s+(?:believe|argue|suggest|note|have found)',
            r'\b(?:peer-reviewed|published|documented)\b',
            r'\b(?:source|citation|reference|bibliography)\b',
            r'\b(?:empirical|statistical|quantitative|qualitative)\b',
        ]
        
        evidence_count = 0
        for pat in evidence_patterns:
            evidence_count += len(re.findall(pat, response_lower))
        
        evidence_score = min(evidence_count * 1.5, 6.0)
        
        # === FEATURE 3: Qualification-to-Assertion Ratio ===
        # Qualifiers that appropriately hedge claims
        qualifier_phrases = [
            r'\b(?:may|might|could|can)\s+(?:be|have|lead|cause|result)',
            r'\b(?:tends? to|likely|unlikely|probable|improbable)\b',
            r'\b(?:in some cases|in many cases|in certain|under certain)\b',
            r'\b(?:it depends|depending on|varies|variable)\b',
            r'\b(?:generally|typically|usually|often|sometimes|occasionally|rarely)\b',
            r'\b(?:to some extent|to a degree|somewhat|partially|partly)\b',
            r'\b(?:one perspective|some argue|others suggest|another view)\b',
            r'\b(?:it\'s worth noting|it should be noted|keep in mind|bear in mind)\b',
            r'\b(?:not necessarily|not always|not entirely|not guaranteed)\b',
            r'\b(?:appears? to|seems? to|tends? to)\b',
            r'\b(?:roughly|approximately|around|about|estimated)\b',
            r'\b(?:as far as|to my knowledge|from what|based on available)\b',
            r'\b(?:there is debate|controversial|disputed|contested)\b',
            r'\b(?:potential|potentially|possible|possibly)\b',
        ]
        
        qualifier_count = 0
        for pat in qualifier_phrases:
            qualifier_count += len(re.findall(pat, response_lower))
        
        # Ratio of qualifiers to sentences
        qual_ratio = qualifier_count / num_sentences
        # Sweet spot: some qualification is good, too much is wishy-washy
        if qual_ratio < 0.1:
            qual_score = 1.0
        elif qual_ratio < 0.3:
            qual_score = 3.0
        elif qual_ratio < 0.6:
            qual_score = 5.0
        elif qual_ratio < 1.0:
            qual_score = 4.0
        else:
            qual_score = 2.5  # Over-hedging
        
        # === FEATURE 4: Discourse Reasoning Chain Detection ===
        # Detect logical reasoning and causal explanation markers
        reasoning_markers = [
            r'\b(?:because|since|therefore|thus|hence|consequently)\b',
            r'\b(?:this means|this implies|this suggests|as a result)\b',
            r'\b(?:for example|for instance|such as|e\.g\.|i\.e\.)\b',
            r'\b(?:however|although|though|nevertheless|on the other hand)\b',
            r'\b(?:in contrast|conversely|alternatively|whereas)\b',
            r'\b(?:first|second|third|finally|moreover|furthermore|additionally)\b',
            r'\b(?:if .+ then|assuming|given that|provided that)\b',
            r'\b(?:consider|note that|importantly|significantly)\b',
        ]
        
        reasoning_count = 0
        for pat in reasoning_markers:
            reasoning_count += len(re.findall(pat, response_lower))
        
        reasoning_density = reasoning_count / num_sentences
        reasoning_score = min(reasoning_density * 4.0, 6.0)
        
        # === FEATURE 5: Categorical/Absolutist Statement Penalty ===
        # Detect overly categorical statements
        absolutist_patterns = [
            r'\b(?:all|every|none|no one|nobody|nothing|everything)\s+(?:is|are|will|can|should|must)\b',
            r'\b(?:always|never)\b',
            r'\b(?:the only|the best|the worst|the most|the least)\b',
            r'\b(?:completely|totally|entirely|utterly|wholly)\s+(?:\w+)',
            r'\b(?:there is no|there are no|there\'s no)\s+(?:way|chance|possibility|doubt|question)\b',
            r'\b(?:without exception|in all cases|under all circumstances)\b',
        ]
        
        absolutist_count = 0
        for pat in absolutist_patterns:
            absolutist_count += len(re.findall(pat, response_lower))
        
        absolutist_penalty = min(absolutist_count * 1.5, 6.0)
        
        # === FEATURE 6: Topic Ambiguity Estimation ===
        # Estimate whether the query is about an ambiguous/debatable topic
        ambiguity_signals = [
            r'\b(?:what do you think|do you believe|should|opinion|thoughts on)\b',
            r'\b(?:is it true|is it possible|can .+ really|does .+ actually)\b',
            r'\b(?:best|worst|better|worse|prefer|favorite)\b',
            r'\b(?:controversial|debate|argument|disagree)\b',
            r'\b(?:why|how come|what if|could .+ be)\b',
            r'\b(?:future|predict|forecast|will .+ happen)\b',
        ]
        
        ambiguity_level = 0
        for pat in ambiguity_signals:
            ambiguity_level += len(re.findall(pat, query_lower))
        
        ambiguity_level = min(ambiguity_level, 5)
        
        # If topic is ambiguous, qualifiers and hedging matter more
        ambiguity_multiplier = 1.0 + (ambiguity_level * 0.15)
        
        # For ambiguous topics, penalize overconfidence more
        if ambiguity_level >= 2:
            assertion_penalty *= 1.3
            absolutist_penalty *= 1.3
        
        # === FEATURE 7: Perspective Acknowledgment ===
        # Does the response acknowledge multiple viewpoints?
        perspective_patterns = [
            r'\b(?:on one hand|on the other hand)\b',
            r'\b(?:some people|some argue|others believe|others suggest)\b',
            r'\b(?:there are .+ views|there are .+ perspectives|there are .+ opinions)\b',
            r'\b(?:proponents|opponents|critics|supporters|advocates)\b',
            r'\b(?:pros and cons|advantages and disadvantages|benefits and drawbacks)\b',
            r'\b(?:it can be argued|one could argue|an argument could be made)\b',
            r'\b(?:different|various|multiple|several)\s+(?:perspectives|viewpoints|opinions|interpretations)\b',
        ]
        
        perspective_count = 0
        for pat in perspective_patterns:
            perspective_count += len(re.findall(pat, response_lower))
        
        perspective_score = min(perspective_count * 2.0, 5.0)
        
        # === FEATURE 8: Epistemic Verb Usage ===
        # Verbs that signal appropriate epistemic stance
        epistemic_verbs = [
            r'\b(?:believe|think|suppose|assume|suspect|hypothesize)\b',
            r'\b(?:estimate|predict|speculate|conjecture|infer)\b',
            r'\b(?:suggest|indicate|imply|point to|hint at)\b',
            r'\b(?:consider|acknowledge|recognize|understand)\b',
        ]
        
        epistemic_verb_count = 0
        for pat in epistemic_verbs:
            epistemic_verb_count += len(re.findall(pat, response_lower))
        
        epistemic_score = min(epistemic_verb_count * 1.0, 4.0)
        
        # === FEATURE 9: Structural Sophistication ===
        # Well-structured responses tend to be more thoughtful
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,4}\s', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        
        structure_score = 0
        if has_numbered_list: structure_score += 1.0
        if has_bullets: structure_score += 0.5
        if has_headers: structure_score += 1.0
        if has_bold: structure_score += 0.5
        
        # === FEATURE 10: Response Completeness ===
        # Very short responses may lack nuance; truncated responses lose points
        word_count = len(response.split())
        
        if word_count < 20:
            completeness_score = 1.0
        elif word_count < 50:
            completeness_score = 3.0
        elif word_count < 100:
            completeness_score = 4.5
        elif word_count < 200:
            completeness_score = 5.0
        else:
            completeness_score = 5.0
        
        # Check for truncation (ends mid-sentence)
        truncated = not response.rstrip().endswith(('.', '!', '?', ':', '"', "'", ')', ']'))
        if truncated:
            completeness_score *= 0.85
        
        # === FEATURE 11: Conditional Language ===
        # "If... then..." constructions show nuanced thinking
        conditional_patterns = [
            r'\bif\b.{5,60}\b(?:then|,)\b',
            r'\b(?:when|where|while)\b.{5,60}\b(?:,|then)\b',
            r'\b(?:in case|provided that|assuming that|given that)\b',
        ]
        
        conditional_count = 0
        for pat in conditional_patterns:
            conditional_count += len(re.findall(pat, response_lower))
        
        conditional_score = min(conditional_count * 1.5, 4.0)
        
        # === COMBINE SCORES ===
        # Positive contributions
        positive_total = (
            qual_score * ambiguity_multiplier * 1.2 +
            evidence_score * 1.0 +
            reasoning_score * 0.8 +
            perspective_score * ambiguity_multiplier * 0.9 +
            epistemic_score * 0.8 +
            structure_score * 0.6 +
            completeness_score * 0.5 +
            conditional_score * 0.7
        )
        
        # Negative contributions (penalties)
        negative_total = (
            assertion_penalty * 1.2 +
            absolutist_penalty * 1.0
        )
        
        # Base score
        raw_score = 50.0 + positive_total - negative_total
        
        # === BONUS: Engagement and Framing ===
        # Opening that acknowledges the question's complexity
        opening_text = response_lower[:200]
        engagement_bonus = 0
        
        engagement_phrases = [
            r'\b(?:great question|interesting|complex|nuanced|depends)\b',
            r'\b(?:let\'s|let me|i\'d|we can|worth considering)\b',
            r'\b(?:there are several|there are multiple|a few|various)\b',
        ]
        for pat in engagement_phrases:
            if re.search(pat, opening_text):
                engagement_bonus += 0.8
        
        raw_score += min(engagement_bonus, 3.0)
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, raw_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 50.0