def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication in LLM responses.
    
    Focuses on:
    - Appropriate hedging language for uncertain claims
    - Distinguishing established facts from speculation
    - Avoiding overconfident claims on ambiguous topics
    - Depth and thoroughness of response
    - Proper attribution and sourcing signals
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # ---- Feature 1: Response length and substance ----
        resp_len = len(response)
        word_count = len(response.split())
        
        # Reward substantive responses, diminishing returns after a point
        import math
        if word_count < 5:
            length_score = 0.0
        elif word_count < 20:
            length_score = 1.0
        elif word_count < 50:
            length_score = 2.0
        elif word_count < 100:
            length_score = 3.5
        elif word_count < 200:
            length_score = 5.0
        elif word_count < 400:
            length_score = 6.0
        else:
            length_score = 6.5
        
        # ---- Feature 2: Hedging and uncertainty language ----
        hedging_phrases = [
            'likely', 'unlikely', 'probably', 'possibly', 'perhaps',
            'might', 'may be', 'could be', 'it seems', 'it appears',
            'tends to', 'generally', 'typically', 'often', 'usually',
            'in most cases', 'research suggests', 'studies suggest',
            'evidence suggests', 'it is thought', 'it is believed',
            'some argue', 'one could argue', 'arguably',
            'to some extent', 'in some cases', 'not always',
            'it depends', 'depending on', 'varies',
            'approximately', 'roughly', 'around',
            'as far as', 'to my knowledge', 'from what',
            'i think', 'i believe', 'in my experience',
            'not necessarily', 'not entirely', 'somewhat',
            'can be', 'there may be', 'there might be',
            'potential', 'potentially', 'plausible',
            'if i recall', 'if i remember', 'iirc',
            'i\'m not sure', 'i\'m not certain',
            'it\'s worth noting', 'worth mentioning',
            'keep in mind', 'bear in mind',
            'on the other hand', 'however', 'although',
            'that said', 'having said that', 'nonetheless',
            'while', 'whereas', 'but',
            'caveat', 'nuance', 'nuanced',
            'debatable', 'controversial', 'contested',
            'open question', 'unclear', 'uncertain',
            'speculation', 'speculative', 'hypothetical',
        ]
        
        hedging_count = 0
        for phrase in hedging_phrases:
            hedging_count += response_lower.count(phrase)
        
        # Normalize hedging by word count
        if word_count > 0:
            hedging_density = hedging_count / word_count
        else:
            hedging_density = 0
        
        # Sweet spot: some hedging is good, too much is wishy-washy
        if hedging_density == 0:
            hedging_score = 0.0
        elif hedging_density < 0.01:
            hedging_score = 1.0
        elif hedging_density < 0.03:
            hedging_score = 2.5
        elif hedging_density < 0.06:
            hedging_score = 3.5
        elif hedging_density < 0.10:
            hedging_score = 3.0
        else:
            hedging_score = 2.0  # Too much hedging
        
        # ---- Feature 3: Overconfidence indicators ----
        overconfident_phrases = [
            'definitely', 'absolutely', 'certainly', 'without a doubt',
            'no question', 'undoubtedly', 'unquestionably',
            'always', 'never', 'impossible', 'guaranteed',
            'everyone knows', 'obviously', 'clearly',
            'there is no', 'it is impossible', 'you must',
            'the only way', 'the only reason', 'the only answer',
            'plain and simple', 'period.', 'end of story',
            'no debate', 'fact is', 'the fact is',
            'trust me', 'believe me',
        ]
        
        overconfident_count = 0
        for phrase in overconfident_phrases:
            overconfident_count += response_lower.count(phrase)
        
        if word_count > 0:
            overconfident_density = overconfident_count / word_count
        else:
            overconfident_density = 0
        
        # Penalize overconfidence
        overconfidence_penalty = min(overconfident_count * 0.8, 4.0)
        
        # ---- Feature 4: Nuance and multi-perspective indicators ----
        nuance_phrases = [
            'on the other hand', 'from another perspective',
            'alternatively', 'conversely', 'in contrast',
            'some people', 'some argue', 'others argue',
            'one view', 'another view', 'different perspective',
            'it\'s complicated', 'it\'s complex', 'multifaceted',
            'trade-off', 'tradeoff', 'trade off',
            'pros and cons', 'advantages and disadvantages',
            'both', 'either', 'neither',
            'for example', 'for instance', 'such as',
            'e.g.', 'i.e.',
            'specifically', 'in particular',
            'context', 'contextual', 'situation',
            'distinction', 'distinguish', 'differentiate',
        ]
        
        nuance_count = 0
        for phrase in nuance_phrases:
            nuance_count += response_lower.count(phrase)
        
        nuance_score = min(nuance_count * 0.6, 4.0)
        
        # ---- Feature 5: Evidence and sourcing signals ----
        evidence_phrases = [
            'according to', 'research shows', 'studies show',
            'research suggests', 'studies suggest', 'data suggests',
            'evidence', 'experiment', 'study found',
            'published', 'journal', 'paper',
            'source', 'reference', 'cited',
            'historically', 'in history',
            'tradition', 'traditionally',
            'as noted by', 'as described by', 'as mentioned',
            'literature', 'scholarly', 'academic',
            'peer-reviewed', 'meta-analysis',
            'survey', 'poll', 'statistics',
            'documented', 'recorded',
        ]
        
        evidence_count = 0
        for phrase in evidence_phrases:
            evidence_count += response_lower.count(phrase)
        
        evidence_score = min(evidence_count * 0.5, 3.0)
        
        # ---- Feature 6: Structural quality ----
        # Check for structured thinking (lists, paragraphs, examples)
        import re
        
        structure_score = 0.0
        
        # Bullet points or numbered lists
        bullet_pattern = re.findall(r'(?:^|\n)\s*[-*•]\s', response)
        numbered_pattern = re.findall(r'(?:^|\n)\s*\d+[.)]\s', response)
        if bullet_pattern or numbered_pattern:
            structure_score += 1.0
        
        # Multiple paragraphs (indicates organized thinking)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            structure_score += 0.5
        if len(paragraphs) >= 3:
            structure_score += 0.5
        
        # Sentences count (proxy for completeness)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_count = len(sentences)
        
        if sentence_count >= 3:
            structure_score += 0.5
        if sentence_count >= 6:
            structure_score += 0.5
        
        structure_score = min(structure_score, 3.0)
        
        # ---- Feature 7: Engagement with the query ----
        # Check how many query terms appear in the response
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        # Remove very common words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
            'their', 'what', 'when', 'where', 'which', 'would', 'could',
            'should', 'about', 'there', 'these', 'those', 'than', 'then',
            'them', 'some', 'more', 'most', 'other', 'into', 'over',
            'such', 'your', 'just', 'also', 'very', 'much', 'many',
            'does', 'like', 'will', 'each', 'make', 'made', 'know',
        }
        query_words = query_words - stop_words
        
        if query_words:
            response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
            overlap = len(query_words & response_words)
            relevance_ratio = overlap / len(query_words)
        else:
            relevance_ratio = 0.5
        
        relevance_score = relevance_ratio * 3.0  # max 3.0
        
        # ---- Feature 8: Conditional and qualifying statements ----
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\b,\b',
            r'\bunless\b', r'\bprovided that\b',
            r'\bassuming\b', r'\bgiven that\b',
            r'\bin the case\b', r'\bwhen\b.*\bthen\b',
            r'\bdepends on\b', r'\bcontingent\b',
        ]
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        conditional_score = min(conditional_count * 0.5, 2.0)
        
        # ---- Feature 9: Self-awareness and epistemic humility ----
        humility_phrases = [
            'i could be wrong', 'i may be wrong', 'i might be wrong',
            'correct me if', 'take this with',
            'grain of salt', 'your mileage may vary', 'ymmv',
            'not an expert', 'not my area',
            'someone else might', 'others may disagree',
            'this is just my', 'in my opinion', 'imo',
            'as i understand it', 'my understanding',
            'i\'m not entirely sure', 'don\'t quote me',
        ]
        
        humility_count = 0
        for phrase in humility_phrases:
            humility_count += response_lower.count(phrase)
        
        humility_score = min(humility_count * 1.0, 2.0)
        
        # ---- Feature 10: Explanatory depth ----
        # Causal and explanatory language
        explanatory_phrases = [
            'because', 'therefore', 'thus', 'hence',
            'as a result', 'consequently', 'due to',
            'the reason', 'this means', 'this implies',
            'in other words', 'essentially', 'fundamentally',
            'the key', 'the main', 'the important',
            'explains', 'explanation', 'mechanism',
        ]
        
        explanatory_count = 0
        for phrase in explanatory_phrases:
            explanatory_count += response_lower.count(phrase)
        
        explanatory_score = min(explanatory_count * 0.4, 2.5)
        
        # ---- Feature 11: Detect if response is a non-answer ----
        non_answer_phrases = [
            'welcome to', 'please read our rules',
            'your post has been', 'this post was removed',
            'i am a bot', 'automod',
            'can you please describe', 'can you clarify',
        ]
        
        non_answer_penalty = 0.0
        for phrase in non_answer_phrases:
            if phrase in response_lower:
                non_answer_penalty += 3.0
        non_answer_penalty = min(non_answer_penalty, 6.0)
        
        # ---- Feature 12: Appropriate confidence calibration ----
        # For questions that are inherently uncertain/debatable, hedging should be rewarded more
        uncertain_query_signals = [
            'ethical', 'moral', 'opinion', 'argue', 'debate',
            'should', 'best', 'worst', 'recommend',
            'think', 'feel', 'believe', 'philosophy',
            'controversial', 'subjective',
            'how much', 'how often', 'how likely',
            'is it possible', 'can you', 'is there',
            'what if', 'imagine', 'hypothetical',
        ]
        
        query_uncertainty = 0
        for signal in uncertain_query_signals:
            if signal in query_lower:
                query_uncertainty += 1
        
        # If query is about uncertain topics, boost hedging score
        if query_uncertainty >= 2:
            hedging_score *= 1.3
            overconfidence_penalty *= 1.3
        
        # ---- Feature 13: Specificity and concreteness ----
        # Specific examples, numbers, names suggest informed response
        has_numbers = len(re.findall(r'\b\d+\b', response)) > 0
        has_proper_nouns = len(re.findall(r'\b[A-Z][a-z]{2,}\b', response)) > 2
        has_quotes = '"' in response or "'" in response
        has_parenthetical = '(' in response and ')' in response
        
        specificity_score = 0.0
        if has_numbers:
            specificity_score += 0.5
        if has_proper_nouns:
            specificity_score += 0.5
        if has_quotes:
            specificity_score += 0.3
        if has_parenthetical:
            specificity_score += 0.3
        specificity_score = min(specificity_score, 1.5)
        
        # ---- Feature 14: Sentence complexity and vocabulary diversity ----
        words = response_lower.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            vocab_score = unique_ratio * 2.0  # Higher diversity = better
        else:
            vocab_score = 0.5
        
        # ---- Combine all features ----
        total_score = (
            length_score           # 0 - 6.5
            + hedging_score        # 0 - 4.55 (with boost)
            + nuance_score         # 0 - 4.0
            + evidence_score       # 0 - 3.0
            + structure_score      # 0 - 3.0
            + relevance_score      # 0 - 3.0
            + conditional_score    # 0 - 2.0
            + humility_score       # 0 - 2.0
            + explanatory_score    # 0 - 2.5
            + specificity_score    # 0 - 1.5
            + vocab_score          # 0 - 2.0
            - overconfidence_penalty  # 0 - 5.2
            - non_answer_penalty      # 0 - 6.0
        )
        
        # Normalize to 0-10 range
        # Theoretical max is around 34, theoretical min around -11
        # Practical range is roughly -5 to 25
        normalized = (total_score + 5) / 30 * 10
        
        # Clamp to [0, 10]
        normalized = max(0.0, min(10.0, normalized))
        
        return round(normalized, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        return 3.0