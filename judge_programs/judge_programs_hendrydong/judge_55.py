def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, hence, etc.)
    2. Sequential/procedural markers (first, then, next, finally, step)
    3. Explanation depth via clause complexity (subordinate clauses)
    4. Rhetorical question usage (engaging reader in reasoning)
    5. Conditional reasoning (if/then patterns)
    6. Evidence/example markers (for example, such as, e.g., consider)
    7. Qualification and nuance markers (however, although, on the other hand)
    8. Ratio of reasoning content to total content
    9. Sentence-level progression (do later sentences build on earlier ones?)
    10. Parenthetical elaborations as a sign of transparent thinking
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        resp = response.strip()
        if len(resp) < 10:
            return 0.5
        
        resp_lower = resp.lower()
        words = re.findall(r'[a-z\']+', resp_lower)
        word_count = len(words)
        if word_count < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # 1. Causal connectives - the backbone of reasoning transparency
        causal_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bwhich means\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bit follows\b',
            r'\bthe reason\b', r'\bgiven that\b', r'\bowing to\b',
            r'\bin order to\b', r'\bso \b', r'\bthat\'s why\b',
            r'\bwhich leads\b', r'\bleading to\b', r'\bresulting in\b',
            r'\bcaused by\b', r'\benabled by\b', r'\baccounts for\b',
        ]
        causal_count = 0
        for pat in causal_patterns:
            causal_count += len(re.findall(pat, resp_lower))
        
        # Normalize per 100 words
        causal_density = (causal_count / word_count) * 100 if word_count > 0 else 0
        causal_score = min(causal_density * 3.5, 15)  # max 15 points
        
        # 2. Sequential/procedural markers
        sequential_patterns = [
            r'\bfirst\b', r'\bfirstly\b', r'\bsecond\b', r'\bsecondly\b',
            r'\bthird\b', r'\bthirdly\b', r'\bnext\b', r'\bthen\b',
            r'\bfinally\b', r'\blastly\b', r'\bstep\b', r'\bto begin\b',
            r'\bto start\b', r'\bafter that\b', r'\bsubsequently\b',
            r'\bin the first place\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin addition\b', r'\balso\b',
            r'\bon top of\b', r'\bbeyond that\b',
        ]
        seq_count = 0
        for pat in sequential_patterns:
            seq_count += len(re.findall(pat, resp_lower))
        
        seq_density = (seq_count / word_count) * 100 if word_count > 0 else 0
        seq_score = min(seq_density * 2.5, 10)  # max 10 points
        
        # 3. Subordinate clause complexity - measured by subordinating conjunctions
        # and relative pronouns indicating elaboration
        subordinate_patterns = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b',
            r'\bwhere\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bwhereas\b', r'\bunless\b', r'\buntil\b', r'\beven though\b',
            r'\beven if\b', r'\bwhether\b', r'\bwhereby\b',
        ]
        sub_count = 0
        for pat in subordinate_patterns:
            sub_count += len(re.findall(pat, resp_lower))
        
        sub_density = (sub_count / word_count) * 100 if word_count > 0 else 0
        # Optimal range: moderate complexity, not too low, not too high
        sub_score = min(sub_density * 1.0, 8)  # max 8 points
        
        # 4. Conditional reasoning (if/then patterns)
        conditional_patterns = [
            r'\bif\b.*?\bthen\b', r'\bif\b.*?\bwould\b', r'\bif\b.*?\bcould\b',
            r'\bif\b.*?\bmight\b', r'\bif\b.*?\bshould\b',
            r'\bif you\b', r'\bif we\b', r'\bif the\b', r'\bif a\b',
            r'\bassuming\b', r'\bsuppose\b', r'\bsupposing\b',
            r'\bin the case\b', r'\bwere to\b', r'\bwould be\b',
        ]
        cond_count = 0
        for pat in conditional_patterns:
            cond_count += len(re.findall(pat, resp_lower))
        
        cond_density = (cond_count / word_count) * 100 if word_count > 0 else 0
        cond_score = min(cond_density * 4.0, 8)  # max 8 points
        
        # 5. Evidence/example markers
        evidence_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\bnamely\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bconsider\b', r'\btake for example\b',
            r'\billustrat', r'\bdemonstrat', r'\bevidence\b', r'\bdata\b',
            r'\bresearch\b', r'\bstudy\b', r'\bstudies\b', r'\baccording to\b',
            r'\bas shown\b', r'\bas seen\b', r'\bin practice\b',
            r'\bin fact\b', r'\bactually\b',
        ]
        evidence_count = 0
        for pat in evidence_patterns:
            evidence_count += len(re.findall(pat, resp_lower))
        
        evidence_density = (evidence_count / word_count) * 100 if word_count > 0 else 0
        evidence_score = min(evidence_density * 5.0, 10)  # max 10 points
        
        # 6. Qualification and nuance (shows careful reasoning, not jumping to conclusions)
        nuance_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bthat said\b', r'\bbut\b',
            r'\byet\b', r'\bstill\b', r'\bdespite\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bnonetheless\b', r'\bwhile\b',
            r'\bthat being said\b', r'\bto be fair\b', r'\badmittedly\b',
            r'\bargubly\b', r'\bit depends\b', r'\bnot necessarily\b',
            r'\bnot always\b', r'\btends to\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bthe trade-off\b', r'\bthe tradeoff\b',
        ]
        nuance_count = 0
        for pat in nuance_patterns:
            nuance_count += len(re.findall(pat, resp_lower))
        
        nuance_density = (nuance_count / word_count) * 100 if word_count > 0 else 0
        nuance_score = min(nuance_density * 3.0, 10)  # max 10 points
        
        # 7. Parenthetical elaborations (parentheses, dashes used for inline explanation)
        paren_count = len(re.findall(r'\([^)]{5,}\)', resp))  # meaningful parentheticals
        dash_elaborations = len(re.findall(r' -- | — | - [^-]+ - ', resp))
        
        elaboration_density = ((paren_count + dash_elaborations) / num_sentences) if num_sentences > 0 else 0
        elaboration_score = min(elaboration_density * 5.0, 6)  # max 6 points
        
        # 8. Sentence-level progression: check if sentences reference concepts from earlier
        # sentences (cohesion via pronoun/demonstrative usage at sentence starts)
        progression_markers = [
            r'^this ', r'^that ', r'^these ', r'^those ',
            r'^it ', r'^they ', r'^such ', r'^the same ',
            r'^as a result', r'^consequently', r'^therefore',
            r'^in other words', r'^to put it', r'^what this means',
        ]
        progression_count = 0
        for sent in sentences:
            sent_lower = sent.strip().lower()
            for pat in progression_markers:
                if re.match(pat, sent_lower):
                    progression_count += 1
                    break
        
        progression_ratio = progression_count / num_sentences if num_sentences > 0 else 0
        progression_score = min(progression_ratio * 15, 8)  # max 8 points
        
        # 9. Explanation verbs - verbs that indicate explaining/reasoning process
        explanation_verbs = [
            r'\bmeans\b', r'\bimplies\b', r'\bsuggests\b', r'\bindicates\b',
            r'\bexplains\b', r'\bshows\b', r'\breveals\b', r'\bworks\b',
            r'\bfunctions\b', r'\boperates\b', r'\bdepends\b',
            r'\brequires\b', r'\binvolves\b', r'\bentails\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bimportantly\b',
            r'\bthe key\b', r'\bthe point\b', r'\bthe idea\b',
            r'\bin essence\b', r'\bessentially\b', r'\bfundamentally\b',
        ]
        expl_count = 0
        for pat in explanation_verbs:
            expl_count += len(re.findall(pat, resp_lower))
        
        expl_density = (expl_count / word_count) * 100 if word_count > 0 else 0
        expl_score = min(expl_density * 3.0, 8)  # max 8 points
        
        # 10. Length bonus - longer responses tend to show more reasoning
        # but with diminishing returns
        length_score = min(math.log(word_count + 1) / math.log(300) * 5, 7)  # max 7 points
        
        # 11. Multi-sentence reasoning chains: count sequences of 3+ sentences
        # that contain reasoning markers
        reasoning_keywords = set()
        all_reasoning_patterns = causal_patterns + sequential_patterns + evidence_patterns + nuance_patterns
        chain_score = 0
        if num_sentences >= 3:
            reasoning_sentences = 0
            for sent in sentences:
                sent_lower = sent.lower()
                has_reasoning = False
                for pat in all_reasoning_patterns:
                    if re.search(pat, sent_lower):
                        has_reasoning = True
                        break
                if has_reasoning:
                    reasoning_sentences += 1
            
            reasoning_ratio = reasoning_sentences / num_sentences
            chain_score = min(reasoning_ratio * 12, 8)  # max 8 points
        
        # 12. Absence of opaque/conclusory patterns (penalize)
        opaque_patterns = [
            r'^(yes|no|sure|okay|ok)[.,!]?\s*$',
            r'\bjust\b.*\bjust\b',  # overly simplistic
            r'\bobviously\b', r'\bclearly\b',  # assuming without explaining
            r'\beveryone knows\b', r'\bit\'s common knowledge\b',
            r'\bnuff said\b', r'\bperiod\b\.\s*$',
        ]
        opaque_count = 0
        for pat in opaque_patterns:
            opaque_count += len(re.findall(pat, resp_lower))
        
        opaque_penalty = min(opaque_count * 2, 6)
        
        # 13. Comma density as proxy for clause-rich sentences
        comma_count = resp.count(',')
        comma_per_sentence = comma_count / num_sentences if num_sentences > 0 else 0
        # Optimal: 1-3 commas per sentence
        if comma_per_sentence < 0.5:
            comma_score = 0
        elif comma_per_sentence <= 3:
            comma_score = comma_per_sentence * 2
        else:
            comma_score = 6 - (comma_per_sentence - 3) * 0.5
        comma_score = max(0, min(comma_score, 6))  # max 6 points
        
        # Aggregate score
        total = (
            causal_score +
            seq_score +
            sub_score +
            cond_score +
            evidence_score +
            nuance_score +
            elaboration_score +
            progression_score +
            expl_score +
            length_score +
            chain_score +
            comma_score -
            opaque_penalty
        )
        
        # Normalize to 0-10 range
        # Max theoretical: 15+10+8+8+10+10+6+8+8+7+8+6 = 104, but realistic max ~50-60
        normalized = total / 8.0  # Scale so good responses get ~6-9
        
        # Clamp
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 3)
    
    except Exception:
        return 2.0