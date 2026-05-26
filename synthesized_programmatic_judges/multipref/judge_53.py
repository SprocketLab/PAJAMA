def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical chain analysis approach. This variant focuses on:
    1. Explicit reasoning connectives and causal language
    2. Sequential reasoning markers (ordinal progression)
    3. Explanation depth via subordinate clause analysis
    4. Question-answer self-dialogue patterns
    5. Evidence/example integration signals
    6. Ratio of explanatory vs declarative sentences
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.0
        
        query = query.strip() if query else ""
        
        # Tokenize into sentences (rough but effective)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = response_stripped.split()
        num_words = max(len(words), 1)
        
        response_lower = response_stripped.lower()
        
        score = 0.0
        
        # ===== 1. CAUSAL CONNECTIVE DENSITY =====
        # These words/phrases explicitly show reasoning chains
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bleads to\b', r'\bleading to\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b',
            r'\bit follows\b', r'\bgiven that\b', r'\bgiven this\b',
            r'\baccordingly\b', r'\bas such\b', r'\bthat\'s why\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bresulting in\b',
            r'\bwhich causes\b', r'\bwhich leads\b', r'\bif\s+\w+\s+then\b',
        ]
        
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, response_lower))
        
        # Normalize by number of sentences
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 8.0, 15.0)
        score += causal_score
        
        # ===== 2. SEQUENTIAL REASONING MARKERS =====
        # Detect ordinal/sequential progression showing step-by-step thinking
        ordinal_patterns = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bfourth(?:ly)?\b', r'\bfifth(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bmoving on\b',
            r'\bafter that\b', r'\bfollowing this\b', r'\bsubsequently\b',
        ]
        
        ordinal_count = 0
        for pattern in ordinal_patterns:
            ordinal_count += len(re.findall(pattern, response_lower))
        
        # Check for numbered steps like "Step 1:", "1.", "1)" etc.
        numbered_steps = re.findall(r'(?:^|\n)\s*(?:step\s+)?\d+[\.\)\:]', response_lower)
        ordinal_count += len(numbered_steps) * 1.5
        
        # Check for progression of distinct ordinals (not just repetition)
        distinct_ordinals = set()
        for i, pattern in enumerate(ordinal_patterns[:5]):  # first through fifth
            if re.search(pattern, response_lower):
                distinct_ordinals.add(i)
        
        progression_bonus = len(distinct_ordinals) * 1.5
        
        ordinal_score = min(ordinal_count * 1.2 + progression_bonus, 12.0)
        score += ordinal_score
        
        # ===== 3. SUBORDINATE CLAUSE / EXPLANATION DEPTH =====
        # Sentences with subordinate clauses tend to be more explanatory
        subordinate_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhile\b', r'\balthough\b', r'\beven though\b',
            r'\bwhereas\b', r'\bunless\b', r'\bprovided that\b',
            r'\bas long as\b', r'\bin case\b',
        ]
        
        explanatory_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            clause_count = 0
            for marker in subordinate_markers:
                if re.search(marker, sent_lower):
                    clause_count += 1
            # A sentence with multiple subordinate clauses is deeply explanatory
            if clause_count >= 2:
                explanatory_sentences += 1.5
            elif clause_count >= 1:
                explanatory_sentences += 1.0
        
        explanation_ratio = explanatory_sentences / num_sentences
        explanation_score = min(explanation_ratio * 10.0, 10.0)
        score += explanation_score
        
        # ===== 4. "WHY" EXPLANATION PATTERNS =====
        # Detect explicit explanations of reasoning
        why_patterns = [
            r'\bthe reason (?:is|being|for)\b',
            r'\bthis is because\b', r'\bthis is due to\b',
            r'\bhere\'?s why\b', r'\bhere is why\b',
            r'\bthe logic\b', r'\bthe thinking\b',
            r'\bto understand (?:why|how|this)\b',
            r'\blet me explain\b', r'\blet\'?s (?:break|think|consider|look|examine|analyze)\b',
            r'\bto put it (?:simply|another way)\b',
            r'\bin other words\b', r'\bwhat this means\b',
            r'\bthe key (?:point|insight|idea|takeaway)\b',
            r'\bwe can (?:see|observe|note|conclude|determine)\b',
            r'\bnotice that\b', r'\bobserve that\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bsignificantly\b',
        ]
        
        why_count = 0
        for pattern in why_patterns:
            why_count += len(re.findall(pattern, response_lower))
        
        why_score = min(why_count * 2.5, 12.0)
        score += why_score
        
        # ===== 5. INTERMEDIATE CONCLUSION MARKERS =====
        # Phrases that signal the response is building up conclusions
        intermediate_markers = [
            r'\bso far\b', r'\bat this point\b', r'\bfrom (?:this|the above)\b',
            r'\bwe (?:can|now) (?:see|conclude|determine|say)\b',
            r'\bthis (?:tells|shows|demonstrates|indicates|suggests|means|implies)\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\boverall\b', r'\btaken together\b', r'\bcombining\b',
            r'\bputting (?:it|this) (?:all )?together\b',
            r'\bnow (?:that|we)\b', r'\bwith this in mind\b',
            r'\bhaving (?:established|determined|found|shown)\b',
            r'\bbuilding on\b', r'\bbased on (?:this|the above|these)\b',
        ]
        
        intermediate_count = 0
        for pattern in intermediate_markers:
            intermediate_count += len(re.findall(pattern, response_lower))
        
        intermediate_score = min(intermediate_count * 3.0, 10.0)
        score += intermediate_score
        
        # ===== 6. CONTRAST AND COMPARISON REASONING =====
        # Shows nuanced thinking by comparing alternatives
        contrast_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bdespite\b', r'\brather than\b', r'\binstead of\b',
            r'\bunlike\b', r'\bwhile\b.*\b(?:but|however)\b',
            r'\balternatively\b', r'\bon one hand\b',
            r'\bthe (?:difference|distinction)\b', r'\bcompared to\b',
        ]
        
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, response_lower))
        
        contrast_score = min(contrast_count * 2.0, 8.0)
        score += contrast_score
        
        # ===== 7. EVIDENCE/EXAMPLE INTEGRATION =====
        evidence_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bconsider\b', r'\bimagine\b', r'\bsuppose\b',
            r'\btake (?:for example|the case)\b',
            r'\b(?:research|studies|data|evidence) (?:shows?|suggests?|indicates?)\b',
            r'\baccording to\b', r'\bas (?:shown|demonstrated|illustrated)\b',
            r'\be\.g\.\b', r'\bi\.e\.\b',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, response_lower))
        
        evidence_score = min(evidence_count * 2.0, 8.0)
        score += evidence_score
        
        # ===== 8. SENTENCE COMPLEXITY VARIANCE =====
        # Good reasoning has a mix of short (conclusions) and long (explanations) sentences
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            
            # Moderate variance is good (mix of summary and detailed sentences)
            if 3 < std_dev < 20:
                complexity_score = min(std_dev * 0.5, 5.0)
            else:
                complexity_score = 1.0
            score += complexity_score
        
        # ===== 9. MATHEMATICAL/LOGICAL NOTATION =====
        # Presence of equations, calculations shows explicit reasoning
        math_patterns = [
            r'[=+\-*/]', r'\d+\s*[×÷±]', r'\^', r'\\frac', r'\\times',
            r'\bequation\b', r'\bformula\b', r'\bcalculat\w+\b',
            r'\bsubstitut\w+\b', r'\bsolv\w+\b', r'\bderiv\w+\b',
        ]
        
        math_count = 0
        for pattern in math_patterns:
            matches = re.findall(pattern, response_lower)
            math_count += min(len(matches), 5)  # Cap each pattern
        
        # Only give credit if query seems to need math
        query_lower = query.lower()
        is_math_query = any(w in query_lower for w in [
            'calculat', 'find', 'solve', 'equation', 'speed', 'distance',
            'how many', 'how much', 'what is', 'kg', 'm/s', 'degrees'
        ])
        
        if is_math_query:
            math_score = min(math_count * 0.5, 8.0)
        else:
            math_score = min(math_count * 0.15, 3.0)
        score += math_score
        
        # ===== 10. STRUCTURAL SCAFFOLDING (different from bullet/header counting) =====
        # Look for explicit framing/setup language that scaffolds reasoning
        scaffolding_patterns = [
            r'\blet\'?s (?:start|begin|first|think|consider|break|look|dive|explore)\b',
            r'\bto (?:answer|address|tackle|approach|solve|understand) this\b',
            r'\bthere are (?:several|a few|multiple|many|two|three|four|five) (?:reasons|factors|steps|ways|aspects|considerations|points)\b',
            r'\bi\'?ll (?:break|walk|go through|explain|outline)\b',
            r'\bhere\'?s (?:how|what|why|the|a|my)\b',
            r'\bhere are\b',
            r'\bkeep in mind\b', r'\bnote that\b', r'\bit\'?s (?:important|worth|helpful) to\b',
            r'\bbefore we\b', r'\bnow let\'?s\b',
        ]
        
        scaffolding_count = 0
        for pattern in scaffolding_patterns:
            scaffolding_count += len(re.findall(pattern, response_lower))
        
        scaffolding_score = min(scaffolding_count * 2.5, 8.0)
        score += scaffolding_score
        
        # ===== 11. DECLARATIVE vs EXPLANATORY SENTENCE RATIO =====
        # Penalize responses that are purely declarative without explanation
        declarative_count = 0
        explanatory_count = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence contains explanatory elements
            has_explanation = any(re.search(p, sent_lower) for p in [
                r'\bbecause\b', r'\bsince\b', r'\bdue to\b', r'\bas\b',
                r'\bwhich\b', r'\bthis\b.*\b(?:means|shows|indicates)\b',
                r'\bso\b', r'\btherefore\b', r'\bthus\b',
            ])
            if has_explanation:
                explanatory_count += 1
            else:
                declarative_count += 1
        
        if num_sentences > 2:
            explanatory_ratio = explanatory_count / num_sentences
            ratio_score = explanatory_ratio * 6.0
            score += min(ratio_score, 6.0)
        
        # ===== 12. RESPONSE LENGTH CONSIDERATION =====
        # Longer responses have more room for reasoning, but with diminishing returns
        length_factor = math.log(num_words + 1) / math.log(500)
        length_score = min(max(length_factor, 0) * 3.0, 4.0)
        score += length_score
        
        # ===== 13. CONVERSATIONAL ENGAGEMENT SIGNALS =====
        # Phrases that engage the reader in the reasoning process
        engagement_patterns = [
            r'\byou (?:can|might|may|could|should|would|will)\b',
            r'\bas you (?:can|might|may) (?:see|notice|imagine|expect)\b',
            r'\bthink of\b', r'\bpicture\b',
            r'\bremember\b', r'\brecall\b',
            r'\bask yourself\b', r'\bwhy\?\b', r'\bhow\?\b',
        ]
        
        engagement_count = 0
        for pattern in engagement_patterns:
            engagement_count += len(re.findall(pattern, response_lower))
        
        engagement_score = min(engagement_count * 1.0, 4.0)
        score += engagement_score
        
        # Normalize final score to 0-100 range
        # Max theoretical ≈ 15+12+10+12+10+8+8+5+8+8+6+4+4 = 110
        final_score = min(max(score, 0.0), 100.0)
        
        return round(final_score, 2)
        
    except Exception:
        return 0.0