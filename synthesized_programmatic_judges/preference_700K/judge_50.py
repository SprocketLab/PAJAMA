def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, etc.)
    2. Conditional reasoning markers (if...then patterns)
    3. Evidence of qualification/nuance (hedging, acknowledging complexity)
    4. Depth of elaboration measured via clause complexity
    5. Progressive disclosure patterns (building from premise to conclusion)
    6. Ratio of explanatory content vs bare assertions
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        resp = response.strip()
        if len(resp) < 10:
            return 0.5
        
        import re
        import math
        
        resp_lower = resp.lower()
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = max(len(sentences), 1)
        
        # 1. CAUSAL CONNECTIVES — words that reveal reasoning chains
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bwhich means\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bit follows\b',
            r'\bfor this reason\b', r'\bthat\'s why\b', r'\bwhich is why\b',
            r'\bleading to\b', r'\bresulting in\b', r'\bcaused by\b',
            r'\bgiven that\b', r'\bin order to\b', r'\bso\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        # Normalize by sentence count — causal density
        causal_density = causal_count / sentence_count
        causal_score = min(causal_density * 5.0, 10.0)  # max 10
        
        # 2. CONDITIONAL REASONING — if/then, suppose, assume, consider
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\bwould\b', r'\bif\b.*\bcould\b',
            r'\bif\b.*\bmight\b', r'\bif\b.*\bwill\b',
            r'\bsuppose\b', r'\bassume\b', r'\bconsider\b',
            r'\bimagine\b', r'\bwhat if\b', r'\beven if\b',
            r'\bunless\b', r'\bprovided that\b', r'\bin the case\b',
        ]
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, resp_lower))
        
        conditional_score = min(conditional_count * 1.5, 8.0)
        
        # 3. QUALIFICATION & NUANCE markers — shows intellectual honesty
        nuance_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bon the other hand\b',
            r'\bthat said\b', r'\bnevertheless\b', r'\bnuance\b',
            r'\bit depends\b', r'\bnot necessarily\b', r'\btypically\b',
            r'\bgenerally\b', r'\btends to\b', r'\bin some cases\b',
            r'\bto some extent\b', r'\barguably\b', r'\bpotentially\b',
            r'\boften\b', r'\busually\b', r'\bsometimes\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bthe trade-?off\b', r'\bon one hand\b', r'\bbut\b',
            r'\brather\b', r'\binstead\b', r'\bactually\b',
        ]
        nuance_count = 0
        for pattern in nuance_markers:
            nuance_count += len(re.findall(pattern, resp_lower))
        
        nuance_density = nuance_count / sentence_count
        nuance_score = min(nuance_density * 3.0, 8.0)
        
        # 4. EXPLANATORY DEPTH — clause complexity via commas, semicolons, subordinate clauses
        # More complex sentences with dependent clauses = more reasoning shown
        comma_count = resp.count(',')
        semicolon_count = resp.count(';')
        colon_count = resp.count(':')
        dash_count = resp.count('—') + resp.count('--') + resp.count(' - ')
        
        clause_markers = comma_count + semicolon_count * 2 + colon_count * 1.5 + dash_count
        avg_clauses_per_sentence = clause_markers / sentence_count
        clause_score = min(avg_clauses_per_sentence * 1.8, 8.0)
        
        # 5. PROGRESSIVE DISCLOSURE — does the response build up?
        # Check if later sentences reference earlier concepts (anaphoric references)
        anaphoric_words = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bsuch\b', r'\bthe above\b', r'\bas mentioned\b',
            r'\bas noted\b', r'\bas I said\b', r'\bbuilding on\b',
            r'\bfrom this\b', r'\bwith this in mind\b',
        ]
        anaphoric_count = 0
        for pattern in anaphoric_words:
            anaphoric_count += len(re.findall(pattern, resp_lower))
        
        # Only meaningful if there are multiple sentences
        if sentence_count > 1:
            anaphoric_density = anaphoric_count / sentence_count
            progressive_score = min(anaphoric_density * 3.0, 6.0)
        else:
            progressive_score = 0.0
        
        # 6. ASSERTION vs EXPLANATION ratio
        # Bare assertions are short declarative sentences without connectives
        # Explanatory sentences contain reasoning markers
        reasoning_keywords = set()
        for pattern in causal_connectives + conditional_patterns + nuance_markers:
            reasoning_keywords.add(pattern)
        
        explanatory_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_reasoning = False
            for pattern in causal_connectives + conditional_patterns + nuance_markers[:15]:
                if re.search(pattern, sent_lower):
                    has_reasoning = True
                    break
            if has_reasoning:
                explanatory_sentences += 1
        
        explanation_ratio = explanatory_sentences / sentence_count
        explanation_score = explanation_ratio * 10.0  # max 10
        
        # 7. CONTENT RICHNESS — unique vocabulary relative to length
        # More diverse vocabulary often correlates with deeper reasoning
        unique_words = len(set(words))
        if word_count > 0:
            vocab_diversity = unique_words / word_count
        else:
            vocab_diversity = 0
        # Typical range 0.4-0.8; normalize
        vocab_score = min(max((vocab_diversity - 0.3) * 8.0, 0), 5.0)
        
        # 8. LENGTH BONUS — longer responses tend to show more reasoning
        # But with diminishing returns (log scale)
        length_score = min(math.log(word_count + 1, 2) * 0.8, 8.0)
        
        # 9. EXAMPLE/ILLUSTRATION usage — concrete reasoning
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bto illustrate\b', r'\blike\b',
            r'\bconsider the case\b', r'\btake for example\b',
        ]
        example_count = 0
        for pattern in example_patterns:
            example_count += len(re.findall(pattern, resp_lower))
        example_score = min(example_count * 2.0, 6.0)
        
        # 10. EXPLICIT REASONING STRUCTURE markers
        structure_patterns = [
            r'\bfirst(ly)?\b', r'\bsecond(ly)?\b', r'\bthird(ly)?\b',
            r'\bnext\b', r'\bfinally\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\bin summary\b', r'\boverall\b',
            r'\bthe reason\b', r'\bthe key\b', r'\bthe point\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bnotably\b',
            r'\bessentially\b', r'\bfundamentally\b',
            r'\bthe trade-?off\b', r'\bthe difference\b',
            r'\bon balance\b', r'\bin short\b',
        ]
        structure_count = 0
        for pattern in structure_patterns:
            structure_count += len(re.findall(pattern, resp_lower))
        structure_score = min(structure_count * 1.5, 6.0)
        
        # 11. QUESTION ENGAGEMENT — does response address the query directly?
        query_lower = query.lower() if query else ""
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        resp_words_set = set(re.findall(r'\b[a-z]{4,}\b', resp_lower))
        
        if query_words:
            overlap = len(query_words & resp_words_set) / len(query_words)
        else:
            overlap = 0.5
        engagement_score = overlap * 4.0  # max ~4
        
        # 12. MULTI-PERSPECTIVE reasoning — presenting multiple viewpoints
        perspective_markers = [
            r'\bon the other hand\b', r'\balternatively\b', r'\bconversely\b',
            r'\bin contrast\b', r'\bsome.*while others\b', r'\bboth\b',
            r'\bnot only.*but also\b', r'\bdepends on\b', r'\bfrom.*perspective\b',
            r'\bone.*another\b', r'\beither.*or\b',
        ]
        perspective_count = 0
        for pattern in perspective_markers:
            perspective_count += len(re.findall(pattern, resp_lower))
        perspective_score = min(perspective_count * 2.5, 6.0)
        
        # AGGREGATE with weights emphasizing reasoning transparency
        total = (
            causal_score * 1.8 +        # causal reasoning is core
            conditional_score * 1.3 +    # conditional logic
            nuance_score * 1.2 +         # qualification
            clause_score * 0.8 +         # sentence complexity
            progressive_score * 1.0 +    # building arguments
            explanation_score * 1.5 +    # explanation ratio
            vocab_score * 0.5 +          # vocabulary richness
            length_score * 0.7 +         # length (diminishing)
            example_score * 1.0 +        # concrete examples
            structure_score * 1.2 +      # explicit structure
            engagement_score * 0.6 +     # query relevance
            perspective_score * 1.0      # multiple viewpoints
        )
        
        # Normalize to 0-10 range
        # Max theoretical raw: ~18 + 10.4 + 9.6 + 6.4 + 6 + 15 + 2.5 + 5.6 + 6 + 7.2 + 2.4 + 6 = ~95
        # Typical good response: ~30-50
        # Typical poor response: ~5-15
        
        normalized = total / 8.0  # scale down
        final_score = min(max(normalized, 0.0), 10.0)
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This makes the middle range more discriminative
        midpoint = 4.5
        steepness = 0.6
        transformed = 10.0 / (1.0 + math.exp(-steepness * (final_score - midpoint)))
        
        return round(transformed, 3)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            return min(len(response.split()) / 30.0, 5.0)
        except Exception:
            return 2.5