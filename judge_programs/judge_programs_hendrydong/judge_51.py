def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, etc.)
    2. Reasoning chain detection (if...then patterns, premise-conclusion structures)
    3. Hedging and qualification language (suggests epistemic care)
    4. Explanation depth via clause complexity (commas, semicolons, parentheticals)
    5. Progressive elaboration detection (sentence-level information accumulation)
    6. Question-response alignment through shared conceptual tokens
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        resp = response.strip()
        resp_lower = resp.lower()
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # 1. CAUSAL/LOGICAL CONNECTIVES — signals that reasoning is being shown
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bso that\b',
            r'\bdue to\b', r'\bowing to\b', r'\bgiven that\b',
            r'\bfor this reason\b', r'\bit follows\b', r'\baccordingly\b',
            r'\bin turn\b', r'\bleading to\b', r'\bresulting in\b',
            r'\bthe reason\b', r'\bthis is why\b', r'\bthat\'s why\b',
            r'\bwhich is why\b', r'\bas such\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        causal_density = causal_count / max(word_count, 1) * 100  # per 100 words
        causal_score = min(causal_density * 5.0, 15.0)  # max 15
        
        # 2. CONDITIONAL/HYPOTHETICAL REASONING (if...then, suppose, consider, assume)
        conditional_patterns = [
            r'\bif\b.*?\bthen\b', r'\bif\b.*?\bwould\b', r'\bif\b.*?\bcould\b',
            r'\bif\b.*?\bshould\b', r'\bif\b.*?\bmight\b',
            r'\bsuppose\b', r'\bassume\b', r'\bconsider\b.*?\bthat\b',
            r'\bimagine\b.*?\bthat\b', r'\bwere\b.*?\bto\b',
            r'\bin the case\b', r'\bwhat if\b',
        ]
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, resp_lower))
        
        conditional_score = min(conditional_count * 2.0, 10.0)  # max 10
        
        # 3. HEDGING & EPISTEMIC MARKERS — shows intellectual honesty and nuance
        hedging_words = [
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bprobably\b', r'\blikely\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bmight\b', r'\bcould\b', r'\bseems?\b', r'\bappears?\b',
            r'\bargue\b', r'\bsuggest\b', r'\bindicate\b', r'\bimply\b',
            r'\bin my (experience|view|opinion)\b', r'\bto some extent\b',
            r'\bnot necessarily\b', r'\bnot always\b', r'\bit depends\b',
            r'\bthe trade-?off\b', r'\bon the other hand\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bnevertheless\b', r'\bnonetheless\b',
        ]
        hedge_count = 0
        for pattern in hedging_words:
            hedge_count += len(re.findall(pattern, resp_lower))
        
        hedge_density = hedge_count / max(word_count, 1) * 100
        hedge_score = min(hedge_density * 3.0, 10.0)  # max 10
        
        # 4. ELABORATION MARKERS — "for example", "in other words", "specifically"
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bnamely\b', r'\bthat is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bto illustrate\b', r'\bto put it\b', r'\bwhat this means\b',
            r'\bin practice\b', r'\bin fact\b', r'\bmore specifically\b',
            r'\bto clarify\b', r'\bto be (more )?specific\b',
        ]
        elab_count = 0
        for pattern in elaboration_markers:
            elab_count += len(re.findall(pattern, resp_lower))
        
        elab_score = min(elab_count * 2.5, 10.0)  # max 10
        
        # 5. CLAUSE COMPLEXITY — commas, semicolons, parenthetical expressions
        # More complex sentences often indicate more nuanced reasoning
        comma_count = resp.count(',')
        semicolon_count = resp.count(';')
        paren_count = resp.count('(')
        dash_count = resp.count(' - ') + resp.count(' -- ') + resp.count('—')
        
        clause_markers = comma_count + semicolon_count * 2 + paren_count * 1.5 + dash_count * 1.5
        clause_density = clause_markers / max(num_sentences, 1)
        clause_score = min(clause_density * 1.5, 8.0)  # max 8
        
        # 6. PROGRESSIVE ELABORATION — do sentences build on each other?
        # Measure by checking if later sentences reference concepts from earlier ones
        if num_sentences >= 2:
            progressive_score_raw = 0
            prev_words_set = set()
            for i, sent in enumerate(sentences):
                sent_words = set(re.findall(r'\b[a-z]{4,}\b', sent.lower()))
                if i > 0 and prev_words_set:
                    overlap = len(sent_words & prev_words_set)
                    new_words = len(sent_words - prev_words_set)
                    # Good reasoning: references old concepts AND introduces new ones
                    if overlap >= 1 and new_words >= 1:
                        progressive_score_raw += 1
                prev_words_set = prev_words_set | sent_words
            
            progressive_ratio = progressive_score_raw / max(num_sentences - 1, 1)
            progressive_score = progressive_ratio * 10.0  # max 10
        else:
            progressive_score = 0.0
        
        # 7. STRUCTURAL REASONING MARKERS — numbered steps, sequential language
        sequential_markers = [
            r'\bfirst(ly)?\b', r'\bsecond(ly)?\b', r'\bthird(ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\badditionally\b', r'\bin addition\b',
            r'\banother\b', r'\balso\b', r'\bin summary\b', r'\bin conclusion\b',
            r'\boverall\b', r'\bto summarize\b',
        ]
        seq_count = 0
        for pattern in sequential_markers:
            seq_count += len(re.findall(pattern, resp_lower))
        
        seq_density = seq_count / max(word_count, 1) * 100
        seq_score = min(seq_density * 3.0, 8.0)  # max 8
        
        # 8. EXPLANATION OF "WHY" — direct why-explanations
        why_patterns = [
            r'\bwhy\b', r'\bthe reason\b', r'\bthis is because\b',
            r'\bexplain\b', r'\bexplanation\b', r'\baccount for\b',
            r'\bjustif(y|ication)\b', r'\brationale\b',
        ]
        why_count = 0
        for pattern in why_patterns:
            why_count += len(re.findall(pattern, resp_lower))
        
        why_score = min(why_count * 1.5, 7.0)  # max 7
        
        # 9. RESPONSE LENGTH — longer responses tend to show more reasoning
        # But with diminishing returns
        length_score = min(math.log(max(word_count, 1) + 1) * 1.5, 10.0)  # max ~10
        
        # 10. SENTENCE COUNT BONUS — multiple sentences suggest step-by-step
        sent_bonus = min(num_sentences * 0.5, 5.0)  # max 5
        
        # 11. CONTRAST/COMPARISON language — shows weighing of alternatives
        contrast_patterns = [
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhereas\b',
            r'\bwhile\b.*?\b(also|however)\b', r'\bunlike\b',
            r'\bcompared to\b', r'\bthe difference\b', r'\bbut\b',
            r'\brather\b', r'\binstead\b', r'\balternatively\b',
            r'\bboth\b', r'\beither\b.*?\bor\b',
        ]
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, resp_lower))
        
        contrast_score = min(contrast_count * 1.5, 7.0)  # max 7
        
        # COMBINE ALL SCORES
        total = (
            causal_score +          # max 15
            conditional_score +      # max 10
            hedge_score +           # max 10
            elab_score +            # max 10
            clause_score +          # max 8
            progressive_score +     # max 10
            seq_score +             # max 8
            why_score +             # max 7
            length_score +          # max 10
            sent_bonus +            # max 5
            contrast_score          # max 7
        )
        # Theoretical max ~100
        
        # Normalize to 0-10 scale
        normalized = total / 10.0
        final_score = min(max(normalized, 0.0), 10.0)
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Minimal fallback: just use length
            return min(len(response.split()) / 20.0, 5.0)
        except Exception:
            return 1.0