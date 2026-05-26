def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a novel approach:
    - Analyzes causal/logical connective density and diversity
    - Measures explanation depth through clause complexity
    - Detects explicit reasoning markers (because, therefore, since, thus, etc.)
    - Evaluates the ratio of "reasoning sentences" vs "assertion sentences"
    - Checks for progressive elaboration patterns (building on previous points)
    - Measures question-engagement (how well the response addresses the query's complexity)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        from collections import Counter
        
        resp_lower = response.lower()
        resp_stripped = response.strip()
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Causal/Logical Connective Density and Diversity
        # Focus on words that EXPLAIN reasoning, not just organize
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bwhich implies\b', r'\bimplying\b',
            r'\bit follows\b', r'\bso that\b', r'\bin order to\b',
            r'\bthe reason\b', r'\bthis is because\b', r'\bthis is why\b',
            r'\baccordingly\b', r'\bas such\b', r'\bgiven that\b',
            r'\bassuming\b', r'\bif we\b', r'\bwhen we\b',
        ]
        
        connective_counts = {}
        total_causal = 0
        for pattern in causal_connectives:
            count = len(re.findall(pattern, resp_lower))
            if count > 0:
                connective_counts[pattern] = count
                total_causal += count
        
        causal_density = total_causal / num_sentences
        causal_diversity = len(connective_counts)
        
        # Score: density up to 15 points, diversity up to 10 points
        score += min(causal_density * 8, 15.0)
        score += min(causal_diversity * 2.5, 10.0)
        
        # ============================================================
        # FEATURE 2: Reasoning Sentence Ratio
        # A "reasoning sentence" contains explanatory language
        # vs an "assertion sentence" that just states facts
        # ============================================================
        reasoning_markers = [
            r'\bbecause\b', r'\bsince\b', r'\bthis means\b', r'\bwhich means\b',
            r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bso\b',
            r'\bthe reason\b', r'\bdue to\b', r'\bas a result\b',
            r'\bin other words\b', r'\bspecifically\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\blike\b',
            r'\bthis is\b', r'\bthis allows\b', r'\bthis ensures\b',
            r'\bthis helps\b', r'\bthis makes\b', r'\bthis provides\b',
            r'\bwhich allows\b', r'\bwhich helps\b', r'\bwhich provides\b',
            r'\bwhich ensures\b', r'\bwhich makes\b',
            r'\benabling\b', r'\ballowing\b', r'\bensuring\b',
            r'\bby doing\b', r'\bby using\b', r'\bin this way\b',
            r'\bto understand\b', r'\bto see why\b', r'\bto illustrate\b',
            r'\blet\'s\b', r'\bwe can\b', r'\bwe need\b', r'\bwe know\b',
            r'\bnote that\b', r'\bnotice that\b', r'\bobserve that\b',
            r'\brecall that\b', r'\bremember that\b',
        ]
        
        reasoning_sentence_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            for marker in reasoning_markers:
                if re.search(marker, sent_lower):
                    reasoning_sentence_count += 1
                    break
        
        reasoning_ratio = reasoning_sentence_count / num_sentences
        # Up to 15 points
        score += reasoning_ratio * 15.0
        
        # ============================================================
        # FEATURE 3: Clause Complexity (multi-clause sentences indicate
        # more elaborate reasoning, not just simple assertions)
        # ============================================================
        subordinating_conjunctions = [
            r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhile\b', r'\balthough\b', r'\beven though\b',
            r'\bwhereas\b', r'\bunless\b', r'\bprovided\b',
            r'\bif\b', r'\bwhether\b',
        ]
        
        complex_sentence_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            clause_markers = 0
            for conj in subordinating_conjunctions:
                clause_markers += len(re.findall(conj, sent_lower))
            # Also count commas as potential clause separators
            comma_count = sent.count(',')
            # A complex sentence has multiple clauses
            if clause_markers >= 1 and comma_count >= 1:
                complex_sentence_count += 1
            elif clause_markers >= 2:
                complex_sentence_count += 1
        
        complexity_ratio = complex_sentence_count / num_sentences
        # Up to 10 points
        score += complexity_ratio * 10.0
        
        # ============================================================
        # FEATURE 4: Progressive Elaboration Patterns
        # Detect when the response builds on previous points
        # ============================================================
        progressive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\bbuilding on\b', r'\bexpanding on\b',
            r'\bnot only\b', r'\bbut also\b', r'\bon top of\b',
            r'\bbeyond that\b', r'\bbeyond this\b',
            r'\banother\b.*\breason\b', r'\banother\b.*\bfactor\b',
            r'\banother\b.*\baspect\b', r'\banother\b.*\bpoint\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b',
            r'\bfinally\b', r'\blast\b.*\bly\b',
            r'\bnext\b', r'\bthen\b', r'\bafter\b.*\bthat\b',
            r'\bonce\b.*\b(done|complete|finished)\b',
        ]
        
        progressive_count = 0
        for pattern in progressive_markers:
            progressive_count += len(re.findall(pattern, resp_lower))
        
        progressive_score = min(progressive_count * 1.5, 8.0)
        score += progressive_score
        
        # ============================================================
        # FEATURE 5: Explicit Problem Decomposition
        # Detect when the response explicitly breaks down the problem
        # ============================================================
        decomposition_phrases = [
            r'\blet\'s break\b', r'\bbreak this down\b', r'\bbreak it down\b',
            r'\bstep by step\b', r'\bstep-by-step\b',
            r'\bfirst,?\s*(we|let|i|you)\b', r'\bto start\b',
            r'\bthe first\b.*\b(step|thing|part)\b',
            r'\bthere are\b.*\b(several|multiple|a few|many)\b.*\b(reasons|factors|steps|parts|aspects|considerations)\b',
            r'\blet\'s consider\b', r'\blet\'s think\b', r'\blet\'s look\b',
            r'\blet\'s examine\b', r'\blet\'s analyze\b', r'\blet\'s explore\b',
            r'\bwe can approach\b', r'\bwe can think\b',
            r'\bto understand this\b', r'\bto answer this\b',
            r'\bthe key\b.*\b(is|here|point)\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\bto recap\b', r'\boverall\b',
            r'\bidentify\b', r'\bcalculate\b', r'\bdetermine\b',
        ]
        
        decomp_count = 0
        for pattern in decomposition_phrases:
            decomp_count += len(re.findall(pattern, resp_lower))
        
        decomp_score = min(decomp_count * 2.0, 10.0)
        score += decomp_score
        
        # ============================================================
        # FEATURE 6: Intermediate Conclusion Markers
        # Detect when the response makes visible intermediate conclusions
        # ============================================================
        intermediate_markers = [
            r'\bso\s*,?\s*(we|this|that|it)\b',
            r'\bthis tells us\b', r'\bthis shows\b', r'\bthis suggests\b',
            r'\bthis indicates\b', r'\bthis demonstrates\b',
            r'\bwe can see\b', r'\bwe can conclude\b', r'\bwe find\b',
            r'\bwe get\b', r'\bwe have\b', r'\bwe obtain\b',
            r'\bputting it\b.*\btogether\b', r'\bcombining\b',
            r'\bfrom this\b', r'\bfrom here\b', r'\bfrom the above\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bmeaning\b', r'\bnamely\b',
            r'\bnow\s*,?\s*(we|let|that)\b',
            r'\bat this point\b',
        ]
        
        intermediate_count = 0
        for pattern in intermediate_markers:
            intermediate_count += len(re.findall(pattern, resp_lower))
        
        intermediate_score = min(intermediate_count * 2.0, 10.0)
        score += intermediate_score
        
        # ============================================================
        # FEATURE 7: Conversational Engagement / Pedagogical Tone
        # Responses that engage the reader tend to be more transparent
        # ============================================================
        engagement_markers = [
            r'\byou\b', r'\byour\b', r'\byou\'ll\b', r'\byou\'re\b',
            r'\blet\'s\b', r'\bwe\b', r'\bour\b', r'\bwe\'ll\b',
            r'\bimagine\b', r'\bconsider\b', r'\bthink of\b', r'\bthink about\b',
            r'\bkeep in mind\b', r'\bremember\b', r'\bnote\b',
            r'\bhere\'s\b', r'\bhere is\b', r'\bhere are\b',
        ]
        
        engagement_count = 0
        unique_engagement = set()
        for pattern in engagement_markers:
            matches = re.findall(pattern, resp_lower)
            if matches:
                unique_engagement.add(pattern)
                engagement_count += len(matches)
        
        engagement_density = engagement_count / max(len(response.split()), 1)
        engagement_diversity = len(unique_engagement)
        
        # Up to 7 points
        score += min(engagement_density * 30, 4.0)
        score += min(engagement_diversity * 0.5, 3.0)
        
        # ============================================================
        # FEATURE 8: Sentence Length Variance
        # Good reasoning often alternates between short summary statements
        # and longer explanatory sentences
        # ============================================================
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = variance ** 0.5
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Moderate variation is good (0.3-0.7 CV)
            if 0.25 <= cv <= 0.8:
                score += 4.0
            elif 0.15 <= cv < 0.25 or 0.8 < cv <= 1.0:
                score += 2.0
            else:
                score += 0.5
        
        # ============================================================
        # FEATURE 9: Labeled/Named Steps or Sections
        # Using **bold**, numbered steps with names, etc.
        # ============================================================
        # Bold markers (markdown)
        bold_patterns = re.findall(r'\*\*[^*]+\*\*', response)
        bold_count = len(bold_patterns)
        
        # Named steps like "Step 1:", "Phase 1:", etc.
        named_steps = re.findall(r'(?:step|phase|stage|part)\s*\d+\s*[:\-]', resp_lower)
        named_step_count = len(named_steps)
        
        # Labeled enumerations with content
        labeled_enum = re.findall(r'\d+\.\s*\*\*', response)
        labeled_enum_count = len(labeled_enum)
        
        # Up to 8 points
        structure_score = min(bold_count * 0.8, 4.0) + min(named_step_count * 1.5, 2.0) + min(labeled_enum_count * 1.0, 2.0)
        score += structure_score
        
        # ============================================================
        # FEATURE 10: Response Length Adequacy
        # Very short responses rarely show reasoning; very long ones may ramble
        # ============================================================
        word_count = len(response.split())
        if word_count < 20:
            score *= 0.3
        elif word_count < 50:
            score *= 0.6
        elif word_count < 80:
            score *= 0.85
        elif word_count > 500:
            # Slight bonus for substantial responses that aren't penalized
            score += 2.0
        
        # ============================================================
        # FEATURE 11: Contrast and Comparison Indicators
        # Good reasoning often involves comparing alternatives
        # ============================================================
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bwhereas\b', r'\binstead\b',
            r'\bunlike\b', r'\brather than\b', r'\bas opposed to\b',
            r'\bbut\b', r'\byet\b', r'\bnevertheless\b',
            r'\bdespite\b', r'\balthough\b', r'\beven though\b',
        ]
        
        contrast_count = 0
        unique_contrast = set()
        for pattern in contrast_markers:
            matches = re.findall(pattern, resp_lower)
            if matches:
                unique_contrast.add(pattern)
                contrast_count += len(matches)
        
        # Up to 5 points
        score += min(contrast_count * 1.0, 3.0) + min(len(unique_contrast) * 0.5, 2.0)
        
        # ============================================================
        # FEATURE 12: Quantitative Reasoning Indicators
        # Numbers, calculations, formulas suggest transparent reasoning
        # ============================================================
        # Check if query seems to require quantitative reasoning
        quant_query_words = ['calculate', 'find', 'compute', 'how much', 'how many', 
                            'what is the', 'speed', 'distance', 'force', 'energy',
                            'probability', 'percent', 'ratio']
        is_quant_query = any(w in query.lower() for w in quant_query_words)
        
        if is_quant_query:
            # Count mathematical expressions
            math_expressions = re.findall(r'[=×÷\+\-\*/]\s*\d', response)
            formula_patterns = re.findall(r'\b\w+\s*=\s*[\d\w\(\)]+', response)
            
            quant_score = min(len(math_expressions) * 1.5 + len(formula_patterns) * 1.0, 5.0)
            score += quant_score
        
        # ============================================================
        # FEATURE 13: Opening Framing Quality
        # Good reasoning starts by framing the problem/approach
        # ============================================================
        first_100_chars = resp_lower[:min(200, len(resp_lower))]
        framing_phrases = [
            r'\blet\'s\b', r'\bto answer\b', r'\bto understand\b',
            r'\bgreat question\b', r'\bgood question\b',
            r'\bthere are several\b', r'\bthis is\b.*\bquestion\b',
            r'\bthe key\b', r'\bfirst\b', r'\bto start\b',
            r'\bhere\'s how\b', r'\bhere is how\b',
            r'\bthe answer\b.*\bdepends\b', r'\bit depends\b',
            r'\bto solve\b', r'\bwe need to\b', r'\bwe can\b',
        ]
        
        framing_count = 0
        for pattern in framing_phrases:
            if re.search(pattern, first_100_chars):
                framing_count += 1
        
        score += min(framing_count * 2.0, 4.0)
        
        # ============================================================
        # Normalize to 0-100 range
        # ============================================================
        # Theoretical max is around 90-100 for an excellent response
        final_score = max(0.0, min(score, 100.0))
        
        return round(final_score, 2)
    
    except Exception as e:
        # Never crash - return a neutral score
        return 25.0