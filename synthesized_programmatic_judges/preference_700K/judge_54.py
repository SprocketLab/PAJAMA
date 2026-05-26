def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, etc.)
    2. Sentence-level progression analysis (do sentences build on each other?)
    3. Explanation depth via subordinate clause detection
    4. Rhetorical structure markers (firstly, on one hand, in contrast, etc.)
    5. Evidence of qualification and nuance (distinguishing from simple hedging)
    6. Question-response alignment through topic threading
    
    This is distinct from other variants that use bullet detection, vocabulary diversity,
    concreteness, hedging language, or simple word overlap.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        from collections import Counter
        
        resp = response.strip()
        resp_lower = resp.lower()
        
        if len(resp) < 10:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', resp)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = resp_lower.split()
        num_words = max(len(words), 1)
        
        # ============================================================
        # FEATURE 1: Causal/Logical Connective Density
        # These words signal explicit reasoning chains
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bso that\b',
            r'\bwhich means\b', r'\bwhich leads to\b', r'\bthis means\b',
            r'\bthis implies\b', r'\bit follows\b', r'\bfor this reason\b',
            r'\bgiven that\b', r'\bgiven this\b', r'\bin turn\b',
            r'\baccordingly\b', r'\bas such\b', r'\bthat\'s why\b',
            r'\bthe reason\b', r'\bthis is because\b', r'\bthis is why\b',
        ]
        
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        # Normalize by sentence count
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 5.0, 10.0)  # 0-10
        
        # ============================================================
        # FEATURE 2: Subordinate Clause Complexity
        # Complex sentences with subordinate clauses indicate deeper explanation
        # ============================================================
        subordinate_markers = [
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\beven though\b',
            r'\beven if\b', r'\bunless\b', r'\bprovided that\b',
            r'\bin order to\b', r'\bso as to\b', r'\bwhether\b',
            r'\bwhenever\b', r'\bwherever\b', r'\bif\b',
            r'\bwho\b', r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
        ]
        
        subordinate_count = 0
        for pattern in subordinate_markers:
            subordinate_count += len(re.findall(pattern, resp_lower))
        
        # Average subordinate clauses per sentence
        sub_per_sentence = subordinate_count / num_sentences
        subordinate_score = min(sub_per_sentence * 2.5, 8.0)  # 0-8
        
        # ============================================================
        # FEATURE 3: Discourse/Rhetorical Structure Markers
        # These indicate organized, structured reasoning
        # ============================================================
        structure_markers = [
            # Sequencing
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bfinally\b', r'\blastly\b', r'\bto begin\b',
            r'\bto start\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\badditionally\b',
            # Contrast/comparison
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bthat said\b', r'\bthat being said\b', r'\bon one hand\b',
            r'\balternatively\b', r'\binstead\b', r'\brather\b',
            # Elaboration
            r'\bin other words\b', r'\bthat is\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bfor instance\b', r'\bfor example\b',
            r'\bsuch as\b', r'\bto illustrate\b', r'\bnamely\b',
            r'\be\.g\.\b', r'\bi\.e\.\b',
            # Summary/conclusion
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\boverall\b', r'\bin short\b', r'\bultimately\b',
            r'\bthe key point\b', r'\bthe main\b', r'\bin essence\b',
            r'\bessentially\b',
        ]
        
        structure_count = 0
        for pattern in structure_markers:
            structure_count += len(re.findall(pattern, resp_lower))
        
        structure_density = structure_count / num_sentences
        structure_score = min(structure_density * 4.0, 10.0)  # 0-10
        
        # ============================================================
        # FEATURE 4: Sentence-to-Sentence Coherence Threading
        # Measure how sentences reference/build on previous content
        # using pronoun references and demonstrative references
        # ============================================================
        threading_patterns = [
            r'^this\b', r'^that\b', r'^these\b', r'^those\b',
            r'^it\b', r'^its\b', r'^they\b', r'^their\b',
            r'^such\b', r'^the same\b', r'^the above\b',
            r'^as mentioned\b', r'^as noted\b', r'^as stated\b',
            r'^building on\b', r'^following\b', r'^from this\b',
        ]
        
        threading_count = 0
        for sent in sentences[1:]:  # skip first sentence
            sent_lower = sent.lower().strip()
            for pattern in threading_patterns:
                if re.match(pattern, sent_lower):
                    threading_count += 1
                    break
        
        threading_ratio = threading_count / max(num_sentences - 1, 1)
        threading_score = threading_ratio * 8.0  # 0-8
        
        # ============================================================
        # FEATURE 5: Explanation Depth - Average Sentence Length Variance
        # Good reasoning has mix of short topic sentences and longer explanatory ones
        # Pure short sentences = shallow; all same length = less structured
        # ============================================================
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / num_sentences
        
        if num_sentences > 1:
            variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / num_sentences
            std_dev = variance ** 0.5
            # Some variance is good (mix of setup and explanation)
            length_variety_score = min(std_dev / 3.0, 3.0)  # 0-3
        else:
            length_variety_score = 0.0
        
        # Reward moderate-to-long average sentence length (reasoning needs space)
        avg_len_score = min(max(avg_sent_len - 5, 0) / 10.0, 3.0)  # 0-3
        
        depth_score = length_variety_score + avg_len_score  # 0-6
        
        # ============================================================
        # FEATURE 6: Explicit Reasoning Phrases
        # Phrases that show the "why" and intermediate reasoning
        # ============================================================
        reasoning_phrases = [
            r'\bthe reason (?:is|being|for)\b',
            r'\bthis (?:is|works|happens) because\b',
            r'\bwhat this means\b', r'\bwhat that means\b',
            r'\bthe (?:key|important|crucial) (?:thing|point|factor|issue)\b',
            r'\bto understand (?:this|why|how)\b',
            r'\bthink of it (?:as|like|this way)\b',
            r'\bthe trade-?off\b', r'\bthe distinction\b',
            r'\bthe difference (?:is|between|here)\b',
            r'\bin practice\b', r'\bin theory\b',
            r'\bon (?:a |the )?(?:practical|theoretical) level\b',
            r'\bthe implication\b', r'\bthe consequence\b',
            r'\bmore (?:precisely|specifically|accurately)\b',
            r'\bto put it (?:simply|differently|another way)\b',
            r'\blet me explain\b', r'\bhere\'s (?:why|how|the thing)\b',
            r'\bthe point (?:is|here|being)\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bimportantly\b',
            r'\bconsider\b', r'\bsuppose\b', r'\bimagine\b',
            r'\bif you (?:think|look|consider)\b',
        ]
        
        reasoning_count = 0
        for pattern in reasoning_phrases:
            reasoning_count += len(re.findall(pattern, resp_lower))
        
        reasoning_phrase_score = min(reasoning_count * 2.0, 10.0)  # 0-10
        
        # ============================================================
        # FEATURE 7: Qualification and Nuance Indicators
        # Shows careful, non-dogmatic reasoning
        # ============================================================
        nuance_patterns = [
            r'\btends to\b', r'\bin general\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bit depends\b', r'\bdepending on\b',
            r'\bnot (?:always|necessarily|entirely)\b',
            r'\bto some (?:extent|degree)\b', r'\bmore or less\b',
            r'\bin most cases\b', r'\bin some cases\b',
            r'\bceteris paribus\b', r'\ball (?:else|things) (?:being )?equal\b',
            r'\bwith the caveat\b', r'\bwith the exception\b',
            r'\bboth\b.*\band\b', r'\bon balance\b',
        ]
        
        nuance_count = 0
        for pattern in nuance_patterns:
            nuance_count += len(re.findall(pattern, resp_lower))
        
        nuance_score = min(nuance_count * 1.5, 6.0)  # 0-6
        
        # ============================================================
        # FEATURE 8: Content Substantiveness
        # Longer, more developed responses tend to have more reasoning
        # But diminishing returns after a point
        # ============================================================
        import math
        # Log scale for length benefit
        length_score = min(math.log(max(num_words, 1) + 1) / math.log(300), 1.0) * 6.0  # 0-6
        
        # ============================================================
        # FEATURE 9: Multi-perspective or Multi-case Analysis
        # Detecting when response considers multiple angles
        # ============================================================
        multi_perspective_patterns = [
            r'\bon (?:the )?one hand\b', r'\bon (?:the )?other hand\b',
            r'\bfrom (?:one|another|a different|this|that) (?:perspective|angle|viewpoint|standpoint)\b',
            r'\bthere are (?:several|multiple|a few|two|three|many) (?:reasons|factors|aspects|considerations|ways|perspectives)\b',
            r'\balternatively\b', r'\banother (?:way|perspective|angle|factor|consideration|point)\b',
            r'\bconversely\b', r'\bin contrast\b',
            r'\bbut also\b', r'\bat the same time\b',
        ]
        
        multi_count = 0
        for pattern in multi_perspective_patterns:
            multi_count += len(re.findall(pattern, resp_lower))
        
        multi_score = min(multi_count * 2.5, 8.0)  # 0-8
        
        # ============================================================
        # FEATURE 10: Parenthetical/Aside Explanations
        # Using parentheses or dashes to add clarifying info
        # ============================================================
        parenthetical_count = len(re.findall(r'\([^)]{5,}\)', resp))
        dash_aside_count = len(re.findall(r' --? [^-]{5,} --? ', resp))
        aside_score = min((parenthetical_count + dash_aside_count) * 1.5, 5.0)  # 0-5
        
        # ============================================================
        # PENALTY: Extremely short or vacuous responses
        # ============================================================
        shortness_penalty = 0.0
        if num_words < 20:
            shortness_penalty = 5.0
        elif num_words < 40:
            shortness_penalty = 2.0
        
        # Detect if response is mostly a redirect (link, "see this other thing")
        redirect_patterns = [
            r'\byou might be interested in\b',
            r'\bcheck out\b',
            r'\bplease read\b',
            r'\bsee (?:this|the|our)\b',
        ]
        redirect_count = sum(1 for p in redirect_patterns if re.search(p, resp_lower))
        redirect_penalty = redirect_count * 2.0 if num_words < 60 else 0.0
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        raw_score = (
            causal_score * 1.5 +        # weight: 15 max
            subordinate_score * 0.8 +    # weight: 6.4 max
            structure_score * 1.3 +      # weight: 13 max
            threading_score * 0.8 +      # weight: 6.4 max
            depth_score * 0.7 +          # weight: 4.2 max
            reasoning_phrase_score * 1.2 +  # weight: 12 max
            nuance_score * 0.8 +         # weight: 4.8 max
            length_score * 1.0 +         # weight: 6 max
            multi_score * 1.0 +          # weight: 8 max
            aside_score * 0.6 +          # weight: 3 max
            - shortness_penalty -
            - redirect_penalty
        )
        # Theoretical max ~78.8
        
        # Normalize to 0-10 range
        normalized = max(raw_score / 7.5, 0.0)
        final_score = min(normalized, 10.0)
        
        return round(final_score, 3)
    
    except Exception:
        try:
            # Fallback: simple length-based score
            return min(len(response.split()) / 30.0, 5.0)
        except Exception:
            return 1.0