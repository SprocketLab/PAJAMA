def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    discourse structure analysis approach based on:
    1. Causal/logical connective density and diversity
    2. Explanation depth via subordinate clause patterns
    3. Progressive reasoning markers (sequential logic flow)
    4. Evidence of intermediate conclusions before final claims
    5. Rhetorical question / engagement patterns
    6. Ratio of explanatory vs assertive sentences
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        resp = response.strip()
        if len(resp) < 10:
            return 0.5
        
        resp_lower = resp.lower()
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-z\']+\b', resp_lower)
        num_words = max(len(words), 1)
        
        # ---- Feature 1: Causal/Logical Connective Diversity and Density ----
        # Different categories of reasoning connectives
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\bas a result\b', r'\bdue to\b',
            r'\bcaused by\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bso that\b', r'\bowing to\b', r'\bgiven that\b',
            r'\bfor this reason\b', r'\bit follows\b', r'\bwhich means\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bleading to\b',
            r'\bresulting in\b', r'\baccordingly\b'
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bwhereas\b', r'\bwhile\b', r'\bdespite\b',
            r'\bin contrast\b', r'\byet\b', r'\bstill\b', r'\bnonetheless\b',
            r'\beven though\b', r'\bthat said\b', r'\bconversely\b'
        ]
        
        elaborative_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bconsider\b', r'\btake\b.*\bfor example\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bto put it\b', r'\bwhat this means\b'
        ]
        
        conditional_connectives = [
            r'\bif\b', r'\bassuming\b', r'\bsuppose\b', r'\bprovided that\b',
            r'\bin the case\b', r'\bwhen\b.*\bthen\b', r'\bwould\b.*\bif\b',
            r'\bgiven\b', r'\bunder\b.*\bcircumstances\b'
        ]
        
        def count_pattern_matches(patterns, text):
            total = 0
            unique = 0
            for p in patterns:
                matches = len(re.findall(p, text))
                if matches > 0:
                    unique += 1
                    total += matches
            return total, unique
        
        causal_count, causal_unique = count_pattern_matches(causal_connectives, resp_lower)
        contrast_count, contrast_unique = count_pattern_matches(contrastive_connectives, resp_lower)
        elab_count, elab_unique = count_pattern_matches(elaborative_connectives, resp_lower)
        cond_count, cond_unique = count_pattern_matches(conditional_connectives, resp_lower)
        
        total_connective_count = causal_count + contrast_count + elab_count + cond_count
        total_connective_unique = causal_unique + contrast_unique + elab_unique + cond_unique
        
        # Categories used (0-4)
        categories_used = sum(1 for c in [causal_unique, contrast_unique, elab_unique, cond_unique] if c > 0)
        
        # Connective density per 100 words
        connective_density = (total_connective_count / num_words) * 100
        
        # Score: density + diversity bonus
        connective_score = min(connective_density * 3.0, 10) + categories_used * 2.0 + min(total_connective_unique * 0.5, 5)
        connective_score = min(connective_score, 25)
        
        # ---- Feature 2: Subordinate Clause Depth (explanation depth) ----
        # Count sentences with subordinate clauses (which, that, who, where, when used mid-sentence)
        subordinate_patterns = [
            r',\s*which\b', r',\s*where\b', r',\s*who\b', r',\s*when\b',
            r'\bthat\s+\w+\s+\w+', r'\bwho\s+\w+\s+\w+',
            r'\bwhere\s+\w+\s+\w+', r'--\s*\w+', r'\(.*?\)'
        ]
        
        subordinate_count = 0
        for p in subordinate_patterns:
            subordinate_count += len(re.findall(p, resp_lower))
        
        subordinate_ratio = subordinate_count / num_sentences
        subordinate_score = min(subordinate_ratio * 5, 10)
        
        # ---- Feature 3: Progressive Reasoning Markers ----
        # Detect sequential/progressive reasoning flow
        progressive_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bin summary\b',
            r'\bto begin\b', r'\bto start\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin addition\b', r'\balso\b',
            r'\banother\b.*\b(?:point|reason|factor|aspect|consideration)\b',
            r'\bon top of\b', r'\bbeyond that\b', r'\bnot only\b.*\bbut\b',
            r'\bbuilding on\b', r'\bfollowing from\b'
        ]
        
        progressive_count = 0
        progressive_unique = 0
        for p in progressive_markers:
            matches = len(re.findall(p, resp_lower))
            if matches > 0:
                progressive_unique += 1
                progressive_count += matches
        
        progressive_score = min(progressive_count * 1.5 + progressive_unique * 1.0, 15)
        
        # ---- Feature 4: Intermediate Conclusions ----
        # Detect phrases that signal intermediate reasoning steps
        intermediate_patterns = [
            r'\bthis (?:means|suggests|implies|indicates|shows)\b',
            r'\bfrom this\b', r'\bwe can (?:see|conclude|infer|deduce)\b',
            r'\bit follows that\b', r'\bso\b', r'\bin effect\b',
            r'\bthe (?:key|important|main|crucial) (?:point|thing|takeaway|insight)\b',
            r'\bwhat this tells us\b', r'\bthe reason (?:is|being)\b',
            r'\bthe point (?:is|here|being)\b', r'\bessentially\b',
            r'\bin short\b', r'\bto summarize\b', r'\bthe upshot\b',
            r'\bthe trade-?off\b', r'\bthe result\b', r'\bthe implication\b',
            r'\bput (?:differently|another way|simply)\b'
        ]
        
        intermediate_count = 0
        for p in intermediate_patterns:
            intermediate_count += len(re.findall(p, resp_lower))
        
        intermediate_score = min(intermediate_count * 2.5, 12)
        
        # ---- Feature 5: Explanatory vs Assertive Sentence Ratio ----
        # Explanatory sentences contain reasoning markers; assertive ones don't
        explanatory_indicators = re.compile(
            r'\bbecause\b|\bsince\b|\bdue to\b|\bas\b.*\bresult\b|'
            r'\bfor example\b|\bfor instance\b|\bif\b|\bwhen\b|'
            r'\bthis means\b|\bthe reason\b|\bin order to\b|\bso that\b|'
            r'\bwhich\b|\bwhere\b|\btherefore\b|\bthus\b|\bhowever\b|'
            r'\balthough\b|\bwhile\b|\bwhereas\b'
        )
        
        explanatory_sentences = 0
        for s in sentences:
            if explanatory_indicators.search(s.lower()):
                explanatory_sentences += 1
        
        explanatory_ratio = explanatory_sentences / num_sentences
        explanatory_score = explanatory_ratio * 15  # max ~15
        
        # ---- Feature 6: Average Sentence Complexity (words per sentence) ----
        # Reasoning tends to produce moderately complex sentences
        avg_words_per_sentence = num_words / num_sentences
        # Sweet spot: 15-30 words per sentence
        if avg_words_per_sentence < 8:
            complexity_score = 1
        elif avg_words_per_sentence < 15:
            complexity_score = 3
        elif avg_words_per_sentence <= 30:
            complexity_score = 6
        elif avg_words_per_sentence <= 45:
            complexity_score = 4
        else:
            complexity_score = 2
        
        # ---- Feature 7: Response Substantiveness ----
        # Longer, more developed responses tend to show more reasoning
        length_score = 0
        if num_words < 20:
            length_score = 0
        elif num_words < 40:
            length_score = 2
        elif num_words < 80:
            length_score = 4
        elif num_words < 150:
            length_score = 6
        elif num_words < 300:
            length_score = 8
        else:
            length_score = 9
        
        # ---- Feature 8: Qualification and Nuance Markers ----
        # Shows careful reasoning rather than absolute claims
        nuance_patterns = [
            r'\btends to\b', r'\bgenerally\b', r'\btypically\b',
            r'\bin most cases\b', r'\bit depends\b', r'\bnot necessarily\b',
            r'\bto some extent\b', r'\bmore or less\b', r'\broughly\b',
            r'\bapproximately\b', r'\bmight\b', r'\bcould\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\blikely\b', r'\bunlikely\b',
            r'\bon balance\b', r'\bin principle\b', r'\bcaveats?\b',
            r'\bnuance\b', r'\bsubtlet(?:y|ies)\b', r'\bdistinction\b',
            r'\bimportant(?:ly)?\b.*\bnote\b', r'\bkeep in mind\b',
            r'\bbear in mind\b', r'\bworth noting\b', r'\bto be fair\b',
            r'\bthat said\b', r'\badmittedly\b'
        ]
        
        nuance_count = 0
        nuance_unique = 0
        for p in nuance_patterns:
            matches = len(re.findall(p, resp_lower))
            if matches > 0:
                nuance_unique += 1
                nuance_count += matches
        
        nuance_score = min(nuance_count * 1.0 + nuance_unique * 0.5, 8)
        
        # ---- Feature 9: Multi-perspective / Alternative Consideration ----
        perspective_patterns = [
            r'\bon (?:the )?one hand\b', r'\bon the other\b',
            r'\balternatively\b', r'\banother (?:way|perspective|view|approach)\b',
            r'\bsome (?:people|argue|say|believe|think)\b',
            r'\bothers (?:argue|say|believe|think|might)\b',
            r'\bthere(?:\'s| is) (?:also|another)\b',
            r'\bboth\b.*\band\b', r'\bnot just\b.*\bbut\b',
            r'\bfrom (?:a|the|one|another) (?:perspective|standpoint|viewpoint|angle)\b'
        ]
        
        perspective_count = 0
        for p in perspective_patterns:
            perspective_count += len(re.findall(p, resp_lower))
        
        perspective_score = min(perspective_count * 2.5, 8)
        
        # ---- Feature 10: Structural Variety ----
        # Mix of short and long sentences suggests structured reasoning
        if num_sentences >= 3:
            sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variation is good (indicates mix of topic sentences and elaboration)
            if std_dev < 2:
                variety_score = 1
            elif std_dev < 5:
                variety_score = 3
            elif std_dev < 10:
                variety_score = 5
            else:
                variety_score = 3
        else:
            variety_score = 1
        
        # ---- Aggregate Score ----
        raw_score = (
            connective_score * 1.0 +      # max ~25
            subordinate_score * 0.8 +      # max ~8
            progressive_score * 0.7 +      # max ~10.5
            intermediate_score * 1.0 +     # max ~12
            explanatory_score * 1.0 +      # max ~15
            complexity_score * 0.8 +       # max ~4.8
            length_score * 0.6 +           # max ~5.4
            nuance_score * 0.8 +           # max ~6.4
            perspective_score * 0.7 +      # max ~5.6
            variety_score * 0.5            # max ~2.5
        )
        # Theoretical max ~95, but realistic max much lower
        
        # Normalize to 0-10 scale with sigmoid-like mapping
        # Use a logistic function centered around a reasonable midpoint
        midpoint = 20
        steepness = 0.08
        normalized = 10.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))
        
        # Ensure minimum score for non-empty responses
        final_score = max(0.5, min(10.0, normalized))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0