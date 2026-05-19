def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical chain analysis approach.
    
    This variant focuses on:
    1. Causal connective density (because, therefore, since, thus, hence, so that, etc.)
    2. Conditional reasoning markers (if...then, when...would, suppose...then)
    3. Explicit acknowledgment/restatement of the problem before answering
    4. Progressive elaboration patterns (first...then...finally, moreover, furthermore)
    5. Qualification and nuance markers (however, although, while, on the other hand)
    6. Ratio of explanatory vs declarative sentences
    7. Evidence of perspective-taking / empathy framing before advice
    8. Meta-reasoning signals ("let's think about", "consider", "the reason is", "this means")
    """
    try:
        if not query or not response:
            return 0.0
        
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        import re
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        
        # Tokenize into sentences (simple split)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b\w+\b', resp_lower)
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # ---- 1. Causal Connectives (why behind claims) ----
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bwhich means\b',
            r'\bthis means\b', r'\bleading to\b', r'\bresulting in\b',
            r'\bfor this reason\b', r'\bthat\'s why\b', r'\bthis is because\b',
            r'\bthe reason\b', r'\bin order to\b', r'\bso you can\b',
            r'\bwhich allows\b', r'\benabling\b', r'\bthis helps\b',
            r'\bthis ensures\b', r'\bmaking it\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        causal_density = causal_count / num_sentences
        score += min(causal_density * 6.0, 12.0)
        
        # ---- 2. Conditional Reasoning Markers ----
        conditional_patterns = [
            r'\bif\b.*?\b(?:then|would|could|should|might|can|will)\b',
            r'\bwhen\b.*?\b(?:would|could|should|might|will)\b',
            r'\bsuppose\b', r'\bassuming\b', r'\bin case\b',
            r'\bwhat if\b', r'\beven if\b',
        ]
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += min(len(re.findall(pattern, resp_lower)), 3)
        
        score += min(conditional_count * 2.0, 8.0)
        
        # ---- 3. Problem Acknowledgment / Restatement ----
        # Check if the response references the user's situation early on
        first_third = response[:len(response) // 3].lower() if len(response) > 30 else resp_lower
        
        acknowledgment_patterns = [
            r'\bi (?:can |)(?:see|hear|understand|sense|notice)\b',
            r'\bit(?:\'s| is) (?:completely |totally |absolutely |perfectly )?(?:understandable|natural|normal|okay|fine|clear)\b',
            r'\bi\'m (?:sorry|glad|happy)\b',
            r'\bthat\'s (?:a |)(?:great|good|tough|hard|difficult|understandable)\b',
            r'\byou(?:\'re| are) (?:feeling|experiencing|going through|dealing with|facing|struggling)\b',
            r'\bimagine\b', r'\blet\'s\b', r'\bconsider\b',
            r'\bsounds like\b', r'\bit seems\b',
        ]
        ack_count = 0
        for pattern in acknowledgment_patterns:
            if re.search(pattern, first_third):
                ack_count += 1
        
        score += min(ack_count * 3.0, 9.0)
        
        # ---- 4. Progressive Elaboration / Sequential Structure ----
        sequential_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bafter (?:that|this)\b',
            r'\bfinally\b', r'\blast(?:ly)?\b', r'\bto (?:start|begin)\b',
            r'\bonce\b.*?\bthen\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bin addition\b', r'\balso\b',
            r'\banother\b', r'\bon top of\b',
        ]
        seq_count = 0
        for pattern in sequential_markers:
            seq_count += len(re.findall(pattern, resp_lower))
        
        seq_density = seq_count / num_sentences
        score += min(seq_density * 5.0, 10.0)
        
        # ---- 5. Qualification and Nuance (shows balanced reasoning) ----
        nuance_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bthat said\b', r'\bbut\b',
            r'\bdespite\b', r'\beven though\b', r'\bregardless\b',
            r'\bstill\b', r'\byet\b', r'\bnot necessarily\b',
            r'\bit depends\b', r'\bin some cases\b',
        ]
        nuance_count = 0
        for pattern in nuance_markers:
            nuance_count += len(re.findall(pattern, resp_lower))
        
        nuance_density = nuance_count / num_sentences
        score += min(nuance_density * 4.0, 6.0)
        
        # ---- 6. Explanatory vs Declarative Sentence Ratio ----
        # Explanatory sentences contain reasoning words
        explanatory_indicators = [
            'because', 'since', 'therefore', 'means', 'helps', 'allows',
            'ensures', 'enables', 'reason', 'way', 'how', 'why',
            'imagine', 'think of', 'consider', 'for example', 'for instance',
            'such as', 'like', 'similar to', 'in other words', 'essentially',
            'basically', 'specifically', 'particularly',
        ]
        explanatory_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            if any(ind in sent_lower for ind in explanatory_indicators):
                explanatory_count += 1
        
        explanatory_ratio = explanatory_count / num_sentences
        score += explanatory_ratio * 10.0
        
        # ---- 7. Meta-reasoning Signals ----
        meta_reasoning = [
            r'\blet\'s (?:think|consider|look|take|break|start|explore|examine)\b',
            r'\bthe (?:key|important|main|crucial|critical) (?:thing|point|idea|concept|aspect)\b',
            r'\bwhat this means\b', r'\bput (?:simply|differently)\b',
            r'\bin other words\b', r'\bto put it\b', r'\bthink of it\b',
            r'\bhere\'s (?:the|a|why|how|what)\b', r'\bthe idea (?:is|here)\b',
            r'\bremember\b', r'\bkeep in mind\b', r'\bnote that\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bnotably\b',
            r'\bthis is (?:where|when|how|why|what)\b',
            r'\bstep\b', r'\bbreak\b.*?\bdown\b',
        ]
        meta_count = 0
        for pattern in meta_reasoning:
            meta_count += len(re.findall(pattern, resp_lower))
        
        score += min(meta_count * 2.5, 10.0)
        
        # ---- 8. Analogies and Examples (making reasoning visible) ----
        analogy_patterns = [
            r'\bjust like\b', r'\bsimilar to\b', r'\bthink of\b',
            r'\bimagine\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\blike (?:a|an|the|when|how)\b',
            r'\banalog(?:y|ous)\b', r'\bcompar(?:e|ing|ison)\b',
            r'\bas if\b', r'\bpicture\b', r'\bvisualize\b',
            r'\bsay(?:,| you| for)\b',
        ]
        analogy_count = 0
        for pattern in analogy_patterns:
            analogy_count += len(re.findall(pattern, resp_lower))
        
        score += min(analogy_count * 2.0, 8.0)
        
        # ---- 9. Engagement with Complexity (not oversimplifying) ----
        # Longer responses with structure tend to show more reasoning
        # But we measure information density, not just length
        avg_sentence_length = num_words / num_sentences
        # Optimal range: 12-25 words per sentence (shows developed thoughts)
        if 12 <= avg_sentence_length <= 25:
            score += 3.0
        elif 8 <= avg_sentence_length < 12 or 25 < avg_sentence_length <= 35:
            score += 1.5
        
        # ---- 10. Numbered/structured reasoning (different from bullet detection) ----
        # Look for explicit reasoning structure like "1.", "2.", "Step 1", etc.
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        step_items = re.findall(r'\bstep\s*\d+\b', resp_lower)
        structured_count = len(numbered_items) + len(step_items)
        if structured_count >= 3:
            score += 5.0
        elif structured_count >= 2:
            score += 3.0
        elif structured_count >= 1:
            score += 1.5
        
        # ---- 11. Absence of opaque/dismissive patterns (penalize) ----
        dismissive_patterns = [
            r'\bjust\b.*?\b(?:do|get|try|move|stop|keep)\b',
            r'\byou (?:need|should|must) to (?:get|just|stop)\b',
            r'\bmaybe you(?:\'re| are) (?:just|not)\b',
            r'\bit\'s (?:just|only|simply) a\b',
        ]
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, resp_lower))
        
        score -= min(dismissive_count * 1.5, 6.0)
        
        # ---- 12. Imperative without explanation penalty ----
        # Sentences that are pure commands without reasoning
        imperative_without_reason = 0
        for sent in sentences:
            sent_stripped = sent.strip()
            sent_lower = sent_stripped.lower()
            # Starts with a verb (imperative)
            imperative_starts = ['get ', 'try ', 'do ', 'make ', 'keep ', 'stop ',
                                 'go ', 'take ', 'find ', 'buy ', 'call ']
            is_imperative = any(sent_lower.startswith(v) for v in imperative_starts)
            has_reasoning = any(ind in sent_lower for ind in ['because', 'since', 'so that',
                                                               'which', 'this will', 'this can',
                                                               'as it', 'as this', 'helping'])
            if is_imperative and not has_reasoning:
                imperative_without_reason += 1
        
        score -= min(imperative_without_reason * 1.0, 4.0)
        
        # ---- 13. Response substantiveness ----
        # Very short responses can't show much reasoning
        if num_words < 20:
            score *= 0.3
        elif num_words < 40:
            score *= 0.6
        elif num_words < 60:
            score *= 0.8
        
        # ---- 14. Paragraph structure (multiple developed paragraphs) ----
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        if len(paragraphs) >= 3:
            score += 3.0
        elif len(paragraphs) >= 2:
            score += 1.5
        
        # ---- 15. Coherence: Does response address the query? ----
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
        resp_words = set(re.findall(r'\b\w{4,}\b', resp_lower))
        if query_words:
            relevance = len(query_words & resp_words) / len(query_words)
            score += relevance * 4.0
        
        # Normalize to 1-5 scale
        # Based on feature analysis, raw scores typically range from ~5 to ~50
        raw_min, raw_max = 2.0, 45.0
        normalized = (score - raw_min) / (raw_max - raw_min)
        normalized = max(0.0, min(1.0, normalized))
        
        final_score = 1.0 + normalized * 4.0  # Maps to 1-5
        
        return round(final_score, 2)
    
    except Exception:
        return 2.5