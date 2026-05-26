def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, etc.)
    2. Explanation depth via subordinate clause patterns
    3. Progressive reasoning markers (first...then...finally, step N, etc.)
    4. Explicit acknowledgment/restatement of the problem before answering
    5. Ratio of "reasoning sentences" vs "assertion sentences"
    6. Conditional reasoning patterns (if...then, when...would, etc.)
    
    This is fundamentally different from other variants that use bullet/list detection,
    word overlap, sentence length, headers, paragraph analysis, hedging language, or
    transition words as primary features.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        # Tokenize into sentences (rough but effective)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = response_lower.split()
        num_words = max(len(words), 1)
        
        # ============================================================
        # FEATURE 1: Causal/Explanatory Connective Density
        # These words signal that the response is EXPLAINING WHY
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bas a result\b', r'\bdue to\b', r'\bthis means\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bconsequently\b',
            r'\bso that\b', r'\bin order to\b', r'\bwhich means\b',
            r'\bwhich is why\b', r'\bthat\'s why\b', r'\bthis leads to\b',
            r'\bcaused by\b', r'\bresulting in\b', r'\bit follows\b',
            r'\bfor this reason\b', r'\bon account of\b',
            r'\bthis is due to\b', r'\bowing to\b',
        ]
        
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, response_lower))
        
        # Normalize by number of sentences
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 12, 10.0)
        
        # ============================================================
        # FEATURE 2: Subordinate Clause Complexity
        # Measures depth of explanation through clause embedding
        # ============================================================
        subordinate_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhile\b', r'\balthough\b', r'\beven though\b',
            r'\bwhereas\b', r'\bprovided that\b', r'\bunless\b',
            r'\bso long as\b',
        ]
        
        subordinate_count = 0
        for pattern in subordinate_markers:
            subordinate_count += len(re.findall(pattern, response_lower))
        
        subordinate_density = subordinate_count / num_sentences
        subordinate_score = min(subordinate_density * 4, 8.0)
        
        # ============================================================
        # FEATURE 3: Progressive/Sequential Reasoning Markers
        # Detects step-by-step structure through sequence words
        # ============================================================
        progressive_patterns = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bafter that\b',
            r'\bonce\s+(?:you|we|this|that|it)\b', r'\bfollowing that\b',
            r'\bsubsequently\b', r'\bstep\s+\d', r'\bphase\s+\d',
            r'\bfrom there\b', r'\bmoving on\b', r'\bat this point\b',
            r'\bnow\s*,', r'\balright\s*,',
        ]
        
        progressive_count = 0
        for pattern in progressive_patterns:
            progressive_count += len(re.findall(pattern, response_lower))
        
        progressive_score = min(progressive_count * 1.8, 10.0)
        
        # ============================================================
        # FEATURE 4: Problem Acknowledgment / Restatement
        # Does the response acknowledge and restate the user's situation
        # before jumping into an answer?
        # ============================================================
        
        # Extract key content words from query
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                     'than', 'too', 'very', 'just', 'how', 'what', 'when', 'where',
                     'who', 'whom', 'which', 'that', 'this', 'these', 'those',
                     'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                     'he', 'she', 'they', 'them', 'their', 'about', 'up', 'out',
                     'if', 'then', 'there', 'here', 'need', 'person'}
        
        query_words = set(re.findall(r'\b[a-z]+\b', query_lower)) - stopwords
        
        # Check if first 2 sentences reference query concepts
        first_part = ' '.join(sentences[:2]).lower() if len(sentences) >= 2 else response_lower[:200]
        first_part_words = set(re.findall(r'\b[a-z]+\b', first_part))
        
        if query_words:
            acknowledgment_overlap = len(query_words & first_part_words) / max(len(query_words), 1)
        else:
            acknowledgment_overlap = 0.5
        
        # Check for empathy/acknowledgment phrases at the start
        ack_patterns = [
            r'^i\s+(?:can\s+)?(?:see|hear|understand|sense)',
            r'^(?:it\'?s?|that\'?s?)\s+(?:completely\s+)?(?:understandable|natural|normal|okay|fine)',
            r'^i\'?m\s+(?:genuinely\s+|truly\s+|really\s+)?sorry',
            r'^(?:great|good)\s+question',
            r'^let\'?s\s+(?:break|work|figure|think)',
            r'^imagine\b',
        ]
        
        ack_bonus = 0
        first_100 = response_lower[:150]
        for pat in ack_patterns:
            if re.search(pat, first_100):
                ack_bonus = 2.0
                break
        
        acknowledgment_score = min(acknowledgment_overlap * 6 + ack_bonus, 8.0)
        
        # ============================================================
        # FEATURE 5: Reasoning Sentence Ratio
        # Classify each sentence as "reasoning" or "bare assertion"
        # A reasoning sentence contains explanatory language
        # ============================================================
        reasoning_indicators = re.compile(
            r'\b(?:because|since|therefore|thus|hence|so|means|implies|leads|'
            r'result|cause|reason|explain|understand|consider|note|important|'
            r'key|essentially|fundamentally|basically|specifically|particularly|'
            r'in other words|put simply|think of|imagine|for example|for instance|'
            r'such as|like when|this way|helps|allows|enables|ensures|makes)\b'
        )
        
        assertion_only = re.compile(
            r'^(?:you\s+(?:should|need|must|can|could|might)|'
            r'just\s+|try\s+|get\s+|do\s+|don\'t\s+|remember\s+to|'
            r'maybe\s+you)'
        )
        
        reasoning_sentences = 0
        assertion_sentences = 0
        
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if reasoning_indicators.search(sent_lower):
                reasoning_sentences += 1
            elif assertion_only.search(sent_lower):
                assertion_sentences += 1
        
        if num_sentences > 0:
            reasoning_ratio = reasoning_sentences / num_sentences
        else:
            reasoning_ratio = 0
        
        reasoning_ratio_score = reasoning_ratio * 10.0
        
        # ============================================================
        # FEATURE 6: Conditional/Hypothetical Reasoning
        # if...then patterns, when...would patterns
        # ============================================================
        conditional_patterns = [
            r'\bif\s+.{5,40}(?:then|,)',
            r'\bwhen\s+.{5,30}(?:would|will|can|should)',
            r'\bin case\b',
            r'\bsuppose\b',
            r'\bassume\b',
            r'\bwhat if\b',
            r'\beven if\b',
            r'\bprovided\b',
        ]
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        conditional_score = min(conditional_count * 2.5, 7.0)
        
        # ============================================================
        # FEATURE 7: Elaboration Depth
        # Longer explanatory passages (not just long sentences)
        # Look for sentences with both a claim and its support
        # ============================================================
        # Count sentences with comma-separated clauses (indicates elaboration)
        elaborated_sentences = 0
        for sent in sentences:
            commas = sent.count(',')
            words_in_sent = len(sent.split())
            # A well-elaborated sentence has commas and reasonable length
            if commas >= 2 and words_in_sent >= 12:
                elaborated_sentences += 1
            elif commas >= 1 and words_in_sent >= 15:
                elaborated_sentences += 1
        
        elaboration_ratio = elaborated_sentences / num_sentences
        elaboration_score = elaboration_ratio * 8.0
        
        # ============================================================
        # FEATURE 8: Contrast/Comparison Reasoning
        # Shows nuanced thinking by comparing alternatives
        # ============================================================
        contrast_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bwhereas\b', r'\binstead of\b', r'\brather than\b',
            r'\bunlike\b', r'\bwhile\b.*\b(?:but|however)\b',
            r'\bnot\s+just\b.*\bbut\s+also\b', r'\bboth\b.*\band\b',
            r'\balternatively\b',
        ]
        
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, response_lower))
        
        contrast_score = min(contrast_count * 2.0, 6.0)
        
        # ============================================================
        # FEATURE 9: Specificity/Concreteness
        # Specific examples, numbers, concrete nouns indicate depth
        # ============================================================
        # Count numbers/quantities
        number_matches = len(re.findall(r'\b\d+(?:\.\d+)?\b', response))
        
        # Count example indicators
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\blike\b', r'\be\.g\.\b', r'\bimagine\b', r'\bpicture\b',
            r'\bthink of\b', r'\bconsider\b', r'\bsay\b',
        ]
        example_count = 0
        for pat in example_patterns:
            example_count += len(re.findall(pat, response_lower))
        
        specificity_score = min((number_matches * 0.3 + example_count * 1.5), 6.0)
        
        # ============================================================
        # PENALTY: Dismissive/Opaque Language
        # Penalize responses that are dismissive or jump to conclusions
        # ============================================================
        dismissive_patterns = [
            r'\bjust\s+(?:do|get|try|go|buy|make|keep)\b',
            r'\byou\s+(?:should|need)\s+to\s+(?:just|simply)\b',
            r'\bit\'?s?\s+(?:just|simply|only)\s+a\b',
            r'\bget over\b', r'\bmove on\b', r'\bdon\'t\s+(?:worry|let)\b',
            r'\bremember\s+(?:it\'?s?|that)\s+just\b',
        ]
        
        dismissive_count = 0
        for pat in dismissive_patterns:
            dismissive_count += len(re.findall(pat, response_lower))
        
        dismissive_penalty = min(dismissive_count * 1.5, 6.0)
        
        # ============================================================
        # PENALTY: Very short responses lack reasoning depth
        # ============================================================
        length_factor = 1.0
        if num_words < 20:
            length_factor = 0.3
        elif num_words < 40:
            length_factor = 0.6
        elif num_words < 60:
            length_factor = 0.8
        elif num_words > 150:
            length_factor = 1.05  # slight bonus for thorough responses
        
        # ============================================================
        # COMBINE SCORES with weights
        # ============================================================
        raw_score = (
            causal_score * 0.15 +           # Why-explanations
            subordinate_score * 0.08 +       # Clause complexity
            progressive_score * 0.12 +       # Step-by-step structure
            acknowledgment_score * 0.12 +    # Problem restatement
            reasoning_ratio_score * 0.18 +   # Proportion of reasoning
            conditional_score * 0.07 +       # If-then reasoning
            elaboration_score * 0.10 +       # Elaborated sentences
            contrast_score * 0.08 +          # Nuanced comparison
            specificity_score * 0.10 -        # Concrete examples
            dismissive_penalty * 0.15        # Penalty for dismissiveness
        )
        
        # Apply length factor
        raw_score *= length_factor
        
        # Scale to 1-5 range
        # raw_score typically ranges from about 0 to 8
        final_score = 1.0 + (raw_score / 7.5) * 4.0
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0