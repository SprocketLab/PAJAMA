def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    CAUSAL CONNECTIVE CHAIN analysis approach.
    
    This variant focuses on:
    1. Causal/logical connective density and diversity (because, therefore, since, thus, hence...)
    2. Explicit reasoning markers ("this means", "this is because", "the reason is")
    3. Question-then-answer patterns (rhetorical questions followed by explanations)
    4. Progressive elaboration depth (do later sentences build on earlier ones?)
    5. Ratio of explanatory vs declarative sentences
    6. Self-referential reasoning (references to own prior points)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower().strip()
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        words = response_lower.split()
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # ============================================================
        # 1. CAUSAL CONNECTIVE CHAIN ANALYSIS
        # Measure density and diversity of causal/logical connectives
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bso that\b',
            r'\bin order to\b', r'\bwhich means\b', r'\bwhich leads to\b',
            r'\bthis causes\b', r'\bthis results in\b',
            r'\bfor this reason\b', r'\bthat is why\b',
            r'\bit follows that\b', r'\baccordingly\b',
            r'\bgiven that\b', r'\bassuming\b', r'\bif\b.*\bthen\b',
        ]
        
        connective_counts = {}
        total_connectives = 0
        for pattern in causal_connectives:
            matches = re.findall(pattern, response_lower)
            if matches:
                connective_counts[pattern] = len(matches)
                total_connectives += len(matches)
        
        connective_diversity = len(connective_counts)
        # Density per 100 words
        connective_density = (total_connectives / num_words) * 100
        
        # Score: density (up to 10) + diversity bonus (up to 8)
        score += min(connective_density * 3.0, 10.0)
        score += min(connective_diversity * 1.5, 8.0)
        
        # ============================================================
        # 2. EXPLICIT REASONING MARKERS
        # Phrases that signal the author is explaining WHY
        # ============================================================
        reasoning_markers = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis means\b',
            r'\bthis implies\b', r'\bthis suggests\b', r'\bin other words\b',
            r'\bto put it\b', r'\bwhat this tells us\b', r'\bthe key\b.*\bis\b',
            r'\bthe idea\b.*\bis\b', r'\bthe point\b.*\bis\b',
            r'\blet me explain\b', r'\bhere\'s why\b', r'\bhere is why\b',
            r'\bto understand\b', r'\bto see why\b', r'\bto clarify\b',
            r'\bnotice that\b', r'\bobserve that\b', r'\bnote that\b',
            r'\brecall that\b', r'\bkeep in mind\b',
            r'\bthe logic\b', r'\bthe reasoning\b', r'\bintuitively\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bwhat does this mean\b', r'\bwhy does this matter\b',
            r'\bwhy is this\b', r'\bhow does this\b',
            r'\blet\'s\b.*\bbreak\b', r'\blet\'s\b.*\bthink\b',
            r'\blet\'s\b.*\bconsider\b', r'\blet\'s\b.*\blook\b',
            r'\blet us\b', r'\bwe can see\b', r'\bwe know\b',
            r'\bthis works because\b', r'\bthe way this works\b',
        ]
        
        marker_count = 0
        marker_diversity = 0
        for pattern in reasoning_markers:
            matches = re.findall(pattern, response_lower)
            if matches:
                marker_count += len(matches)
                marker_diversity += 1
        
        marker_density = (marker_count / num_words) * 100
        score += min(marker_density * 5.0, 10.0)
        score += min(marker_diversity * 1.0, 6.0)
        
        # ============================================================
        # 3. QUESTION-THEN-ANSWER PATTERNS
        # Rhetorical questions that set up explanations
        # ============================================================
        question_marks = response.count('?')
        # Check if questions are followed by explanatory text (not just trailing)
        qa_patterns = re.findall(r'\?[^.!?]*[.!]', response)
        qa_score = min(len(qa_patterns) * 2.0, 6.0)
        # Also reward "What/Why/How" questions used rhetorically
        rhetorical_q = re.findall(r'\b(?:what|why|how|when|where)\b[^?]*\?', response_lower)
        qa_score += min(len(rhetorical_q) * 1.0, 4.0)
        score += qa_score
        
        # ============================================================
        # 4. PROGRESSIVE ELABORATION / SENTENCE-LEVEL BUILD-UP
        # Do sentences reference or build upon previous content?
        # ============================================================
        back_references = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\babove\b', r'\bprevious\b', r'\bearlier\b',
            r'\bas mentioned\b', r'\bas noted\b', r'\bas stated\b',
            r'\bfrom\b.*\bstep\b', r'\bin step\b',
            r'\bbuilding on\b', r'\bfollowing from\b',
        ]
        
        # Count sentences that start with or contain back-references
        back_ref_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            for pat in back_references:
                if re.search(pat, sent_lower):
                    back_ref_sentences += 1
                    break
        
        back_ref_ratio = back_ref_sentences / num_sentences
        score += min(back_ref_ratio * 12.0, 8.0)
        
        # ============================================================
        # 5. EXPLANATORY vs DECLARATIVE sentence ratio
        # Explanatory sentences contain reasoning language
        # Declarative sentences are bare assertions
        # ============================================================
        explanatory_indicators = [
            r'\bbecause\b', r'\bsince\b', r'\bso\b', r'\bthus\b',
            r'\btherefore\b', r'\bwhich\b', r'\bwhere\b', r'\bwhen\b',
            r'\bif\b', r'\bas\b', r'\bby\b', r'\bthrough\b',
            r'\busing\b', r'\bwith\b', r'\bfor\b.*\breason\b',
            r'\bmeans\b', r'\bimplies\b', r'\bresults\b',
            r'\ballows\b', r'\benables\b', r'\bcauses\b',
            r'\bensures\b', r'\bprevents\b', r'\brequires\b',
        ]
        
        explanatory_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            expl_matches = sum(1 for pat in explanatory_indicators if re.search(pat, sent_lower))
            if expl_matches >= 2:  # At least 2 explanatory indicators
                explanatory_count += 1
        
        explanatory_ratio = explanatory_count / num_sentences
        score += min(explanatory_ratio * 15.0, 10.0)
        
        # ============================================================
        # 6. SENTENCE COMPLEXITY PROGRESSION
        # Reasoning often involves sentences that grow in complexity
        # (longer sentences with more clauses as reasoning develops)
        # ============================================================
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) >= 3:
            # Check for variety in sentence length (reasoning mixes short setup + long explanation)
            if len(sent_lengths) > 1:
                mean_len = sum(sent_lengths) / len(sent_lengths)
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Moderate variance is good (mix of setup and explanation)
                cv = std_dev / max(mean_len, 1)
                if 0.2 <= cv <= 0.8:
                    score += 4.0
                elif cv > 0.1:
                    score += 2.0
        
        # ============================================================
        # 7. NUMBERED/LABELED STEP DETECTION (but different from list detection)
        # Focus on whether steps have EXPLANATORY content, not just existence
        # ============================================================
        step_patterns = re.findall(
            r'(?:step\s*\d|#\d|\*\*step\b|\bfirst(?:ly)?\b|\bsecond(?:ly)?\b|\bthird(?:ly)?\b|\bnext\b|\bfinally\b|\bthen\b)',
            response_lower
        )
        num_step_markers = len(step_patterns)
        
        # Check if numbered items (1., 2., etc.) contain explanatory content
        numbered_items = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', response, re.DOTALL)
        explained_items = 0
        for item in numbered_items:
            item_lower = item.lower()
            has_explanation = any(re.search(pat, item_lower) for pat in explanatory_indicators[:10])
            if has_explanation and len(item.split()) > 10:
                explained_items += 1
        
        score += min(num_step_markers * 0.8, 5.0)
        score += min(explained_items * 1.5, 5.0)
        
        # ============================================================
        # 8. CONTRAST AND COMPARISON REASONING
        # Using contrast to explain (however, on the other hand, whereas, unlike)
        # ============================================================
        contrast_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bwhereas\b',
            r'\bunlike\b', r'\bin contrast\b', r'\bconversely\b',
            r'\balternatively\b', r'\binstead\b', r'\brather than\b',
            r'\bwhile\b.*\b(?:is|are|was|were)\b', r'\bbut\b',
            r'\bdespite\b', r'\bnevertheless\b', r'\bnonetheless\b',
        ]
        
        contrast_count = 0
        for pat in contrast_patterns:
            contrast_count += len(re.findall(pat, response_lower))
        
        contrast_density = (contrast_count / num_words) * 100
        score += min(contrast_density * 4.0, 6.0)
        
        # ============================================================
        # 9. MATHEMATICAL/LOGICAL NOTATION (for technical queries)
        # Equations, formulas, calculations shown
        # ============================================================
        math_patterns = re.findall(r'[=+\-*/^]', response)
        has_equations = len(re.findall(r'\w+\s*=\s*[\d\w(]', response))
        if has_equations > 0:
            score += min(has_equations * 1.0, 4.0)
        
        # ============================================================
        # 10. LENGTH-ADJUSTED SCORING
        # Very short responses can't show much reasoning
        # But don't reward length alone - reward density
        # ============================================================
        if num_words < 20:
            score *= 0.3
        elif num_words < 50:
            score *= 0.6
        elif num_words < 80:
            score *= 0.8
        # For very long responses, slightly penalize if reasoning density is low
        if num_words > 200:
            overall_density = score / (num_words / 100)
            if overall_density < 5:
                score *= 0.9
        
        # ============================================================
        # 11. INTRODUCTORY FRAMING
        # Does the response set up the problem/approach before diving in?
        # ============================================================
        framing_patterns = [
            r'^(?:to|in order to|let\'s|let me|we need to|first,? (?:we|let|I))',
            r'^(?:there are several|this is a|the (?:key|main|primary))',
            r'^(?:great question|good question|interesting)',
            r'(?:approach|method|strategy|way to think about)',
            r'(?:break.*down|analyze|consider|examine|look at)',
        ]
        
        first_two_sentences = ' '.join(sentences[:2]).lower() if sentences else ''
        framing_score = 0
        for pat in framing_patterns:
            if re.search(pat, first_two_sentences):
                framing_score += 1.5
        score += min(framing_score, 4.0)
        
        # ============================================================
        # 12. CONCLUSION/SUMMARY PRESENCE
        # Does the response wrap up with a synthesis?
        # ============================================================
        conclusion_patterns = [
            r'\bin summary\b', r'\bin conclusion\b', r'\bto summarize\b',
            r'\boverall\b', r'\bin short\b', r'\bto sum up\b',
            r'\bthe bottom line\b', r'\bultimately\b', r'\ball in all\b',
            r'\bso,?\s+(?:the|in|to|we)\b',
        ]
        
        last_portion = ' '.join(sentences[-3:]).lower() if len(sentences) >= 3 else response_lower
        conclusion_found = any(re.search(pat, last_portion) for pat in conclusion_patterns)
        if conclusion_found:
            score += 3.0
        
        # ============================================================
        # 13. CONDITIONAL REASONING
        # "if X then Y" style reasoning
        # ============================================================
        conditional_patterns = re.findall(
            r'\bif\b[^.?!]{5,}\b(?:then|,)\b',
            response_lower
        )
        score += min(len(conditional_patterns) * 1.5, 4.0)
        
        # ============================================================
        # 14. EXAMPLE USAGE for illustration
        # "for example", "for instance", "such as", "e.g."
        # ============================================================
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\blike\b.*\band\b',
            r'\bconsider\b.*\bcase\b', r'\bimagine\b', r'\bsuppose\b',
        ]
        example_count = sum(len(re.findall(pat, response_lower)) for pat in example_patterns)
        score += min(example_count * 1.5, 5.0)
        
        # Normalize to 0-100 range
        # Maximum theoretical score is roughly: 10+8+10+6+10+8+10+4+5+5+6+4+4+3+4+5 = ~102
        # But in practice scores will be much lower
        # Scale so that good responses get 60-80 and poor ones get 20-40
        final_score = max(0.0, min(score, 100.0))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.split()) > 20:
                return 25.0
            return 10.0
        except:
            return 10.0