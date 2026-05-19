def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, so, as a result)
    2. Sentence-level progression analysis (do sentences build on each other?)
    3. Explanation depth via subordinate clause detection
    4. Rhetorical structure markers (firstly, on one hand, in contrast, however)
    5. Evidence of qualification and nuance (trade-off, depends, distinction, difference)
    6. Question-response alignment via structural complexity matching
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower()
        resp_stripped = response.strip()
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', resp_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', resp_lower)
        num_words = max(len(words), 1)
        
        # ---- Feature 1: Causal/Logical Connective Density ----
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bthis means\b', r'\bwhich means\b',
            r'\bso that\b', r'\bin order to\b', r'\bthe reason\b',
            r'\bthis is why\b', r'\bthat\'s why\b', r'\bleading to\b',
            r'\bresulting in\b', r'\bit follows\b', r'\bimplies\b',
            r'\bgiven that\b', r'\bassuming\b', r'\bif\b.*\bthen\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 3.0, 10.0)
        
        # ---- Feature 2: Subordinate Clause Complexity ----
        subordinators = [
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\beven though\b',
            r'\bwhen\b', r'\bwhere\b', r'\bwhich\b', r'\bthat\b',
            r'\bwho\b', r'\bwhom\b', r'\bunless\b', r'\bprovided that\b',
            r'\bin case\b', r'\bso long as\b',
        ]
        sub_count = 0
        for pattern in subordinators:
            sub_count += len(re.findall(pattern, resp_lower))
        
        sub_density = sub_count / num_sentences
        subordination_score = min(sub_density * 2.5, 8.0)
        
        # ---- Feature 3: Rhetorical/Discourse Structure Markers ----
        discourse_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bon one hand\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bto illustrate\b', r'\bconsider\b',
            r'\bin summary\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto sum up\b', r'\bin short\b', r'\bultimately\b',
            r'\bessentially\b', r'\bthe key\b', r'\bthe point\b',
            r'\bimportantly\b', r'\bnotably\b', r'\bsignificantly\b',
        ]
        discourse_count = 0
        for pattern in discourse_markers:
            discourse_count += len(re.findall(pattern, resp_lower))
        
        discourse_density = discourse_count / num_sentences
        discourse_score = min(discourse_density * 3.5, 10.0)
        
        # ---- Feature 4: Qualification and Nuance Language ----
        nuance_terms = [
            r'\btrade-?off\b', r'\bdepends\b', r'\bdistinction\b',
            r'\bdifference\b', r'\bnuance\b', r'\bcomplex\b',
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\bsometimes\b',
            r'\bin some cases\b', r'\bnot always\b', r'\bnot necessarily\b',
            r'\bit depends\b', r'\bcan vary\b', r'\bvaries\b',
            r'\bmore or less\b', r'\broughly\b', r'\bapproximately\b',
            r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bpotentially\b', r'\barguably\b',
            r'\bon the other hand\b', r'\balternatively\b',
        ]
        nuance_count = 0
        for pattern in nuance_terms:
            nuance_count += len(re.findall(pattern, resp_lower))
        
        nuance_density = nuance_count / num_sentences
        nuance_score = min(nuance_density * 3.0, 8.0)
        
        # ---- Feature 5: Sentence-to-Sentence Cohesion (progressive reasoning) ----
        # Measure how much consecutive sentences share content words (building on each other)
        cohesion_scores = []
        if len(sentences) >= 2:
            for i in range(len(sentences) - 1):
                words_a = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentences[i].lower()))
                words_b = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentences[i+1].lower()))
                if words_a and words_b:
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    cohesion_scores.append(overlap / union if union > 0 else 0)
            
            avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
        else:
            avg_cohesion = 0
        
        cohesion_score = min(avg_cohesion * 25.0, 8.0)
        
        # ---- Feature 6: Explanation Depth - Average sentence length ----
        avg_sent_len = num_words / num_sentences
        # Sweet spot: 15-30 words per sentence indicates detailed explanation
        if avg_sent_len < 8:
            depth_score = 1.0
        elif avg_sent_len < 15:
            depth_score = 3.0
        elif avg_sent_len <= 30:
            depth_score = 6.0
        elif avg_sent_len <= 45:
            depth_score = 4.5
        else:
            depth_score = 3.0
        
        # ---- Feature 7: Response Length (longer = more room for reasoning) ----
        # Logarithmic scaling to avoid over-rewarding pure length
        length_score = min(math.log(num_words + 1) / math.log(500) * 6.0, 8.0)
        
        # ---- Feature 8: Explicit Reasoning Verbs ----
        reasoning_verbs = [
            r'\bexplain\b', r'\breason\b', r'\banalyze\b', r'\banalyse\b',
            r'\bargue\b', r'\bsuggest\b', r'\bindicate\b', r'\bdemonstrate\b',
            r'\bshow\b', r'\bprove\b', r'\bimply\b', r'\bsupport\b',
            r'\bcontribute\b', r'\bcause\b', r'\baffect\b', r'\binfluence\b',
            r'\brelate\b', r'\bcompare\b', r'\bcontrast\b', r'\bdistinguish\b',
            r'\bclarify\b', r'\belaborate\b', r'\billustrate\b',
        ]
        reasoning_verb_count = 0
        for pattern in reasoning_verbs:
            reasoning_verb_count += len(re.findall(pattern, resp_lower))
        
        rv_density = reasoning_verb_count / num_sentences
        reasoning_verb_score = min(rv_density * 4.0, 8.0)
        
        # ---- Feature 9: Parenthetical/Aside Explanations ----
        # Count parenthetical remarks (often used for clarification)
        parens = len(re.findall(r'\([^)]{5,}\)', response))
        dashes = len(re.findall(r' --? [^-]+ --? ', response))
        aside_count = parens + dashes
        aside_score = min(aside_count * 1.5, 5.0)
        
        # ---- Feature 10: Multi-perspective / Contrast Reasoning ----
        contrast_patterns = [
            r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bdespite\b', r'\byet\b', r'\bstill\b', r'\bnevertheless\b',
            r'\brather\b', r'\binstead\b',
        ]
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, resp_lower))
        
        contrast_density = contrast_count / num_sentences
        contrast_score = min(contrast_density * 3.0, 7.0)
        
        # ---- Feature 11: Explicit "why" explanations ----
        why_patterns = [
            r'\bthis is because\b', r'\bthe reason (?:is|being)\b',
            r'\bwhy\b.*\bbecause\b', r'\bexplains why\b',
            r'\baccounts for\b', r'\bhere\'s why\b', r'\bthat\'s because\b',
        ]
        why_count = 0
        for pattern in why_patterns:
            why_count += len(re.findall(pattern, resp_lower))
        why_score = min(why_count * 2.5, 6.0)
        
        # ---- Feature 12: Structural formatting (colons, semicolons used for elaboration) ----
        colons = response.count(':')
        semicolons = response.count(';')
        structural_punct = colons + semicolons
        structural_score = min(structural_punct * 0.8, 5.0)
        
        # ---- Combine all features with weights ----
        total = (
            causal_score * 1.8 +          # Causal reasoning is key
            subordination_score * 1.2 +    # Complex sentence structure
            discourse_score * 1.5 +        # Discourse organization
            nuance_score * 1.3 +           # Qualification/nuance
            cohesion_score * 1.0 +         # Progressive reasoning
            depth_score * 1.0 +            # Sentence complexity
            length_score * 1.2 +           # Sufficient elaboration
            reasoning_verb_score * 1.0 +   # Explicit reasoning
            aside_score * 0.8 +            # Clarifying asides
            contrast_score * 1.2 +         # Multi-perspective
            why_score * 1.5 +              # Explicit why
            structural_score * 0.5         # Structural punctuation
        )
        
        # Normalize: max possible ~= 18 + 9.6 + 15 + 10.4 + 8 + 6 + 9.6 + 8 + 4 + 8.4 + 9 + 2.5 = ~108.5
        # Scale to 0-10
        max_theoretical = 110.0
        normalized = (total / max_theoretical) * 10.0
        
        # Apply slight bonus for multi-sentence responses (reasoning needs space)
        if num_sentences >= 3:
            normalized += 0.5
        if num_sentences >= 5:
            normalized += 0.5
        if num_sentences >= 8:
            normalized += 0.3
        
        # Penalize very short responses (hard to show reasoning in < 20 words)
        if num_words < 20:
            normalized *= 0.4
        elif num_words < 40:
            normalized *= 0.7
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 3)
    
    except Exception:
        return 0.0