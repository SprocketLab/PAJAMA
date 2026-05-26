def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a multi-signal approach based on:
    1. Specificity signals (names, dates, numbers, technical terms)
    2. Appropriate hedging vs. overconfidence detection
    3. Structural coherence and completeness
    4. Hallucination red-flag detection
    5. Information density and diversity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        query = query.strip() if query else ""
        
        if len(response) < 3:
            return 0.5
        
        score = 50.0  # Start at midpoint
        
        # === 1. SPECIFICITY SIGNALS (0-15 points) ===
        # Detect concrete, verifiable information markers
        specificity_score = 0.0
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response)
        num_count = min(len(numbers), 8)
        specificity_score += num_count * 1.0
        
        # Year-like patterns
        years = re.findall(r'\b(?:1[0-9]{3}|2[0-9]{3})\b', response)
        specificity_score += min(len(years), 4) * 1.5
        
        # Capitalized proper nouns (not at sentence start)
        sentences = re.split(r'[.!?]\s+', response)
        proper_noun_count = 0
        for sent in sentences:
            words = sent.split()
            for i, w in enumerate(words):
                if i > 0 and w and w[0].isupper() and len(w) > 1 and w.isalpha():
                    proper_noun_count += 1
        specificity_score += min(proper_noun_count, 6) * 0.5
        
        # Technical/domain terms (longer words suggest more specific vocabulary)
        words = response.split()
        long_words = [w for w in words if len(w) > 8 and w.isalpha()]
        specificity_score += min(len(long_words), 8) * 0.3
        
        score += min(specificity_score, 15.0)
        
        # === 2. HEDGING & CONFIDENCE CALIBRATION (-8 to +8) ===
        response_lower = response.lower()
        
        # Appropriate hedging phrases (good)
        appropriate_hedges = [
            'generally', 'typically', 'often', 'usually', 'tends to',
            'in many cases', 'may', 'might', 'can be', 'it is possible',
            'according to', 'research suggests', 'evidence indicates',
            'it appears', 'likely', 'approximately', 'roughly',
            'in some cases', 'depending on', 'varies'
        ]
        hedge_count = sum(1 for h in appropriate_hedges if h in response_lower)
        score += min(hedge_count, 5) * 1.6
        
        # Overconfidence / absolutism red flags (bad)
        absolutist_phrases = [
            'always', 'never', 'definitely', 'absolutely', 'without a doubt',
            'guaranteed', 'proven fact', 'everyone knows', 'obviously',
            'undeniably', 'there is no question', 'it is certain',
            'no one can deny', '100%', 'the truth is'
        ]
        abs_count = sum(1 for a in absolutist_phrases if a in response_lower)
        score -= min(abs_count, 4) * 2.0
        
        # === 3. HALLUCINATION RED FLAGS (-12 to 0) ===
        hallucination_penalty = 0.0
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{2,}\.\d+%', response)
        hallucination_penalty -= len(precise_stats) * 2.0
        
        # Sensationalism markers
        sensational = [
            'shocking', 'unbelievable', 'mind-blowing', 'incredible',
            'you won\'t believe', 'secret', 'they don\'t want you to know',
            'exposed', 'bombshell', 'game-changer', 'revolutionary',
            'conspiracy', 'cover-up', 'mainstream media', 'wake up',
            'sheeple', 'big pharma', 'deep state', 'hoax'
        ]
        sens_count = sum(1 for s in sensational if s in response_lower)
        hallucination_penalty -= sens_count * 3.0
        
        # Repetitive phrasing (sign of degenerate output / low quality)
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(words[i].lower() + ' ' + words[i+1].lower())
        if bigrams:
            bigram_counts = Counter(bigrams)
            max_bigram_repeat = max(bigram_counts.values()) if bigram_counts else 1
            total_bigrams = len(bigrams)
            if total_bigrams > 4 and max_bigram_repeat > 2:
                repeat_ratio = max_bigram_repeat / total_bigrams
                hallucination_penalty -= repeat_ratio * 25.0
        
        score += max(hallucination_penalty, -12.0)
        
        # === 4. INFORMATION DENSITY & DIVERSITY (0-12 points) ===
        
        # Unique word ratio (lexical diversity)
        word_tokens = [w.lower().strip('.,!?;:()[]"\'') for w in words if w.strip('.,!?;:()[]"\'')]
        if len(word_tokens) > 3:
            unique_ratio = len(set(word_tokens)) / len(word_tokens)
            # Optimal diversity is around 0.5-0.8
            if unique_ratio > 0.85:
                diversity_score = 4.0  # Very diverse but maybe too scattered
            elif unique_ratio > 0.6:
                diversity_score = 6.0  # Good diversity
            elif unique_ratio > 0.4:
                diversity_score = 4.0  # Moderate
            elif unique_ratio > 0.25:
                diversity_score = 2.0  # Repetitive
            else:
                diversity_score = 0.0  # Very repetitive
        else:
            diversity_score = 1.0
        score += diversity_score
        
        # Sentence count and variety (structural complexity)
        sentence_list = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = len(sentence_list)
        
        if num_sentences >= 3:
            # Sentence length variance (good writing has varied sentence lengths)
            sent_lengths = [len(s.split()) for s in sentence_list]
            if len(sent_lengths) > 1:
                mean_len = sum(sent_lengths) / len(sent_lengths)
                variance = sum((l - mean_len)**2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Some variance is good (2-8 std dev)
                if 2 <= std_dev <= 10:
                    score += 3.0
                elif std_dev > 0.5:
                    score += 1.5
            score += min(num_sentences, 5) * 0.6
        
        # === 5. RESPONSE COMPLETENESS & STRUCTURE (0-10 points) ===
        
        # Length appropriateness
        resp_len = len(response)
        word_count = len(words)
        
        if word_count < 5:
            score -= 8.0
        elif word_count < 15:
            score -= 3.0
        elif word_count < 30:
            score += 2.0
        elif word_count < 100:
            score += 5.0
        elif word_count < 250:
            score += 4.0
        else:
            score += 3.0  # Very long might be unfocused
        
        # Check for truncation (response cut off mid-sentence)
        if response and response[-1] not in '.!?"\')':
            last_sentence = sentence_list[-1] if sentence_list else response
            if len(last_sentence.split()) > 3:
                score -= 4.0  # Likely truncated
        
        # Explanatory connectives (shows reasoning structure)
        connectives = [
            'because', 'therefore', 'however', 'furthermore', 'in addition',
            'for example', 'for instance', 'such as', 'specifically',
            'in contrast', 'on the other hand', 'as a result',
            'this means', 'in other words', 'while', 'whereas',
            'although', 'moreover', 'consequently'
        ]
        conn_count = sum(1 for c in connectives if c in response_lower)
        score += min(conn_count, 5) * 1.2
        
        # === 6. QUERY RELEVANCE (0-5 points) ===
        # Check that response addresses the query
        if query:
            query_words = set(w.lower().strip('.,!?;:()[]"\'') for w in query.split() 
                            if len(w) > 3)
            resp_words = set(w.lower().strip('.,!?;:()[]"\'') for w in words)
            
            if query_words:
                overlap = len(query_words & resp_words)
                relevance = overlap / len(query_words)
                score += min(relevance * 5.0, 5.0)
        
        # === 7. CITATION-LIKE PATTERNS (bonus 0-4) ===
        citation_patterns = [
            r'according to\s+\w+',
            r'(?:study|research|survey|report|analysis)\s+(?:by|from|published)',
            r'(?:source|reference|citation)',
            r'\(\d{4}\)',  # Year in parentheses like citations
            r'et\s+al\.',
            r'(?:published|reported|found)\s+(?:in|by|that)',
        ]
        citation_hits = sum(1 for p in citation_patterns if re.search(p, response_lower))
        score += min(citation_hits, 3) * 1.3
        
        # === 8. PENALIZE EMPTY/NONSENSE PATTERNS ===
        # Check for <noinput> or similar non-responses
        if re.search(r'<noinput>|<nooutput>|n/a|not applicable', response_lower):
            score -= 15.0
        
        # Check for excessive repetition of single words
        if word_tokens:
            word_freq = Counter(word_tokens)
            most_common_word, most_common_count = word_freq.most_common(1)[0]
            if most_common_count > len(word_tokens) * 0.3 and len(word_tokens) > 5:
                if most_common_word not in {'the', 'a', 'an', 'is', 'to', 'and', 'of', 'in', 'for', 'that', 'it', 'with'}:
                    score -= 8.0
        
        # Clamp to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0