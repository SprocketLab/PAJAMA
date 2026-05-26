def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    Focuses on:
    - Explicit step-by-step breakdowns
    - Visible intermediate reasoning/conclusions
    - Explanations of 'why' behind claims
    - Logical flow that readers can follow and verify
    - Penalizes opaque, jump-to-conclusion answers
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # ============================================================
        # FEATURE 1: Step-by-step structure indicators
        # ============================================================
        
        # Numbered lists (1. 2. 3. or 1) 2) 3))
        numbered_pattern = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        num_numbered_steps = len(numbered_pattern)
        
        # Bullet points
        bullet_pattern = re.findall(r'(?:^|\n)\s*[-•*]\s', response)
        num_bullets = len(bullet_pattern)
        
        # Sequential/transition words indicating step progression
        step_words = [
            'first', 'second', 'third', 'fourth', 'fifth',
            'firstly', 'secondly', 'thirdly',
            'next', 'then', 'after that', 'following this',
            'finally', 'lastly', 'to begin', 'to start',
            'step 1', 'step 2', 'step 3', 'step 4',
            'the first', 'the second', 'the third', 'the next',
            'moving on', 'building on', 'from there',
            'once you', 'once this', 'after you', 'now that',
            'at this point', 'from here'
        ]
        step_word_count = sum(1 for w in step_words if w in response_lower)
        
        structure_score = min(10, (num_numbered_steps * 1.5 + num_bullets * 1.2 + step_word_count * 0.8))
        
        # ============================================================
        # FEATURE 2: Reasoning/causal language
        # ============================================================
        
        reasoning_phrases = [
            'because', 'since', 'therefore', 'thus', 'hence',
            'as a result', 'consequently', 'this means',
            'the reason', 'this is because', 'this is due to',
            'which means', 'which leads to', 'which results in',
            'in other words', 'that is to say', 'put simply',
            'this implies', 'it follows that', 'given that',
            'due to', 'owing to', 'on account of',
            'so that', 'in order to', 'for this reason',
            'this suggests', 'this indicates', 'this shows',
            'the key point', 'importantly', 'notably',
            'this is why', 'that\'s why', "here's why",
            'the underlying', 'fundamentally', 'essentially',
            'to understand why', 'to see why', 'let me explain',
            'let\'s break', 'let\'s look at', 'let\'s consider',
            'let\'s think', 'consider this', 'think of it',
            'what this means', 'why this matters',
        ]
        reasoning_count = sum(1 for phrase in reasoning_phrases if phrase in response_lower)
        reasoning_score = min(10, reasoning_count * 1.2)
        
        # ============================================================
        # FEATURE 3: Intermediate conclusions / signposting
        # ============================================================
        
        intermediate_markers = [
            'so far', 'at this point', 'in summary', 'to summarize',
            'in short', 'the takeaway', 'key takeaway',
            'what we know', 'what this tells us',
            'having established', 'with that in mind',
            'now we can', 'now let\'s', 'now that we',
            'this brings us to', 'this leads to',
            'based on this', 'building on this',
            'with this understanding', 'keeping this in mind',
            'remember that', 'recall that', 'note that',
            'it\'s important to note', 'it\'s worth noting',
            'keep in mind', 'bear in mind',
            'here\'s the thing', 'the point is',
            'in essence', 'the bottom line',
        ]
        intermediate_count = sum(1 for m in intermediate_markers if m in response_lower)
        intermediate_score = min(10, intermediate_count * 1.5)
        
        # ============================================================
        # FEATURE 4: Elaboration depth - sentence complexity and variety
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences > 0:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / num_sentences
            # Moderate sentence length is good for explanation (10-25 words)
            if 10 <= avg_sentence_length <= 25:
                sentence_quality = 1.0
            elif 7 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
                sentence_quality = 0.6
            else:
                sentence_quality = 0.3
        else:
            sentence_quality = 0.1
        
        elaboration_score = min(10, num_sentences * 0.5 * sentence_quality)
        
        # ============================================================
        # FEATURE 5: Acknowledging complexity / showing awareness
        # ============================================================
        
        awareness_phrases = [
            'it\'s understandable', 'it\'s completely understandable',
            'i understand', 'i can see', 'i can hear',
            'it\'s natural', 'it\'s normal', 'it\'s okay',
            'it\'s perfectly', 'it\'s absolutely',
            'you might wonder', 'you may wonder',
            'you might think', 'you might feel',
            'this might seem', 'this may seem',
            'on one hand', 'on the other hand',
            'however', 'that said', 'having said that',
            'while', 'although', 'even though',
            'there are several', 'there are a few',
            'some ways', 'here are some', 'here are a few',
            'one approach', 'another approach',
            'one way', 'another way',
        ]
        awareness_count = sum(1 for p in awareness_phrases if p in response_lower)
        awareness_score = min(10, awareness_count * 1.3)
        
        # ============================================================
        # FEATURE 6: Response engagement with query specifics
        # ============================================================
        
        # Extract meaningful words from query (4+ chars, not stop words)
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'them', 'their',
            'have', 'been', 'were', 'will', 'would', 'could', 'should',
            'about', 'which', 'there', 'these', 'those', 'what', 'when',
            'where', 'being', 'some', 'more', 'very', 'just', 'also',
            'into', 'over', 'such', 'than', 'other', 'each', 'most',
            'only', 'after', 'before', 'does', 'doing', 'during',
            'here', 'your', 'need', 'must', 'the', 'and', 'for',
        }
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower)) - stop_words
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
        
        if query_words:
            relevance_ratio = len(query_words & response_words) / len(query_words)
        else:
            relevance_ratio = 0.5
        
        relevance_score = relevance_ratio * 8
        
        # ============================================================
        # FEATURE 7: Paragraph/section structure
        # ============================================================
        
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        num_paragraphs = len(paragraphs)
        
        # Headers or labeled sections
        header_patterns = re.findall(r'(?:^|\n)\s*(?:[A-Z][a-zA-Z\s]+:|\*\*[^*]+\*\*|#{1,3}\s)', response)
        num_headers = len(header_patterns)
        
        structure_organization = min(8, num_paragraphs * 1.0 + num_headers * 1.5)
        
        # ============================================================
        # FEATURE 8: Analogy and example usage (aids transparency)
        # ============================================================
        
        analogy_phrases = [
            'for example', 'for instance', 'such as',
            'imagine', 'think of', 'picture this',
            'it\'s like', 'it\'s similar to', 'just like',
            'consider', 'suppose', 'say you',
            'in the same way', 'analogous to',
            'to illustrate', 'to put it', 'to give you',
            'like a', 'like an', 'like when',
            'as if', 'as though', 'pretend',
        ]
        analogy_count = sum(1 for p in analogy_phrases if p in response_lower)
        analogy_score = min(8, analogy_count * 1.5)
        
        # ============================================================
        # FEATURE 9: Penalize dismissive / opaque responses
        # ============================================================
        
        dismissive_phrases = [
            'just do it', 'just try', 'you should be able',
            'it\'s simple', 'it\'s easy', 'obviously',
            'just remember', 'just keep', 'just make sure',
            'might not', 'probably won\'t', 'can\'t really',
            'maybe you\'re just', 'you\'re just not',
        ]
        dismissive_count = sum(1 for p in dismissive_phrases if p in response_lower)
        dismissive_penalty = min(5, dismissive_count * 1.5)
        
        # Overly short responses lack reasoning transparency
        response_len = len(response.strip())
        if response_len < 50:
            brevity_penalty = 4
        elif response_len < 100:
            brevity_penalty = 2.5
        elif response_len < 150:
            brevity_penalty = 1.0
        else:
            brevity_penalty = 0
        
        # Negative/unhelpful tone detection
        negative_patterns = [
            'you should be able to handle',
            'read the manual',
            'not my problem',
            'figure it out',
            'deal with it',
        ]
        negative_count = sum(1 for p in negative_patterns if p in response_lower)
        negative_penalty = negative_count * 2
        
        # ============================================================
        # FEATURE 10: Conditional/nuanced language (shows careful reasoning)
        # ============================================================
        
        nuance_phrases = [
            'depending on', 'it depends', 'in some cases',
            'typically', 'generally', 'usually',
            'if you', 'when you', 'should you',
            'may vary', 'can vary', 'might differ',
            'one option', 'alternatively', 'another option',
            'pros and cons', 'trade-off', 'balance',
            'it\'s crucial', 'it\'s essential', 'it\'s important',
            'make sure', 'ensure that', 'be careful',
            'don\'t forget', 'keep in mind',
        ]
        nuance_count = sum(1 for p in nuance_phrases if p in response_lower)
        nuance_score = min(8, nuance_count * 1.0)
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        # Weighted combination
        raw_score = (
            structure_score * 0.15 +        # Step-by-step structure
            reasoning_score * 0.18 +         # Causal/reasoning language
            intermediate_score * 0.10 +      # Intermediate conclusions
            elaboration_score * 0.10 +       # Elaboration depth
            awareness_score * 0.10 +         # Acknowledging complexity
            relevance_score * 0.08 +         # Query relevance
            structure_organization * 0.08 +  # Paragraph structure
            analogy_score * 0.08 +           # Examples/analogies
            nuance_score * 0.08 +            # Nuanced language
            # Small baseline for non-empty coherent response
            1.5
        )
        
        # Apply penalties
        total_penalty = dismissive_penalty + brevity_penalty + negative_penalty
        raw_score = max(0, raw_score - total_penalty)
        
        # Bonus for having multiple reasoning features present simultaneously
        features_present = sum(1 for s in [
            structure_score, reasoning_score, intermediate_score,
            awareness_score, analogy_score, nuance_score
        ] if s > 1.0)
        
        if features_present >= 4:
            raw_score *= 1.15
        elif features_present >= 3:
            raw_score *= 1.08
        
        # Scale to 1-5 range to match examples
        # Raw scores typically range from 0 to ~10
        final_score = 1.0 + (raw_score / 10.0) * 4.0
        final_score = max(1.0, min(5.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception as e:
        # Never crash - return neutral score
        try:
            if response and len(response.strip()) > 100:
                return 2.5
            return 1.5
        except:
            return 2.0