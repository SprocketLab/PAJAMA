def judging_function(query, response):
    """
    Evaluates response quality using a structural and linguistic analysis approach
    focused on factual accuracy indicators.
    
    This variant uses:
    - Sentence-level analysis (variety, structure complexity)
    - Information density metrics (unique content ratio, entity-like patterns)
    - Repetition penalty (n-gram repetition detection)
    - Specificity signals (numbers, proper nouns, technical terms)
    - Appropriate uncertainty language vs overconfidence
    - Response completeness relative to query
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # === 1. SENTENCE-LEVEL STRUCTURAL ANALYSIS ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Sentence length variety (std dev of sentence lengths)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        if len(sent_lengths) > 1:
            variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            sent_len_std = math.sqrt(variance)
        else:
            sent_len_std = 0
        
        # Score sentence variety (some variety is good, shows structured thinking)
        sent_variety_score = min(sent_len_std / 5.0, 1.0) * 5  # 0-5
        
        # === 2. REPETITION PENALTY (n-gram based) ===
        words = re.findall(r'\b[a-zA-Z]+\b', response.lower())
        total_words = len(words)
        
        if total_words == 0:
            return 0.5
        
        # Bigram repetition
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)] if len(words) > 1 else []
        bigram_counts = Counter(bigrams)
        if bigrams:
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
            bigram_repetition_ratio = repeated_bigrams / max(len(bigram_counts), 1)
        else:
            bigram_repetition_ratio = 0
        
        # Trigram repetition
        trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)] if len(words) > 2 else []
        trigram_counts = Counter(trigrams)
        if trigrams:
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition_ratio = repeated_trigrams / max(len(trigram_counts), 1)
        else:
            trigram_repetition_ratio = 0
        
        # Word-level repetition (unique ratio)
        unique_words = set(words)
        # Filter out very common words for uniqueness calculation
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                     'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
                     'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
                     'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'he',
                     'him', 'his', 'she', 'her', 'i', 'me', 'my', 'you', 'your'}
        
        content_words = [w for w in words if w not in stopwords]
        unique_content = set(content_words)
        content_uniqueness = len(unique_content) / max(len(content_words), 1)
        
        repetition_penalty = (bigram_repetition_ratio * 8 + trigram_repetition_ratio * 12)
        repetition_penalty += max(0, (1 - content_uniqueness) - 0.3) * 15  # penalize if <70% unique content words
        repetition_penalty = min(repetition_penalty, 25)
        
        # === 3. SPECIFICITY AND FACTUAL INDICATORS ===
        # Count specific factual indicators in the response
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+\.?\d*\b', response)
        num_count = len(numbers)
        
        # Proper noun patterns (capitalized words not at sentence start)
        proper_noun_pattern = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response)
        proper_noun_count = len(proper_noun_pattern)
        
        # Parenthetical information (often used for clarification/citation)
        parentheticals = re.findall(r'\([^)]+\)', response)
        paren_count = len(parentheticals)
        
        # Quotation marks (referencing specific phrases)
        quotes = re.findall(r'"[^"]*"', response) + re.findall(r'"[^"]*"', response)
        quote_count = len(quotes)
        
        # Technical/specific vocabulary (longer words tend to be more specific)
        long_words = [w for w in content_words if len(w) >= 8]
        long_word_ratio = len(long_words) / max(len(content_words), 1)
        
        specificity_score = (
            min(num_count * 0.5, 3) +
            min(proper_noun_count * 0.4, 2) +
            min(paren_count * 0.8, 2) +
            min(quote_count * 0.5, 2) +
            long_word_ratio * 5
        )  # 0-14 range roughly
        specificity_score = min(specificity_score, 12)
        
        # === 4. APPROPRIATE UNCERTAINTY vs OVERCONFIDENCE ===
        # Appropriate hedging phrases
        appropriate_hedging = [
            'suggests', 'indicates', 'may', 'might', 'could', 'potentially',
            'generally', 'typically', 'often', 'usually', 'tends to',
            'in many cases', 'it is possible', 'research suggests',
            'according to', 'evidence suggests', 'it appears',
            'commonly', 'frequently', 'likely'
        ]
        
        # Overconfidence / sensationalism red flags
        overconfidence_flags = [
            'always', 'never', 'absolutely', 'definitely', 'undoubtedly',
            'without a doubt', 'guaranteed', 'proven fact', 'everyone knows',
            'obviously', 'clearly the best', 'the only', 'no one',
            'shocking', 'unbelievable', 'mind-blowing', 'they don\'t want you to know',
            'exposed', 'secret', 'conspiracy', 'cover-up', 'wake up',
            'mainstream media', 'big pharma', 'the truth is',
            'exactly', '100%', 'impossible'
        ]
        
        response_lower = response.lower()
        
        hedge_count = sum(1 for h in appropriate_hedging if h in response_lower)
        overconfidence_count = sum(1 for o in overconfidence_flags if o in response_lower)
        
        # Balanced hedging is good (some but not excessive)
        hedge_score = min(hedge_count * 0.8, 4)
        overconfidence_penalty = overconfidence_count * 1.5
        
        certainty_score = hedge_score - overconfidence_penalty  # can be negative
        
        # === 5. STRUCTURAL COMPLETENESS ===
        # Does the response have proper structure?
        
        # Check if response appears truncated
        truncated = 0
        if response[-1] not in '.!?")\']' and len(response) > 50:
            truncated = 3  # penalty for truncation
        
        # Check for enumeration / organized structure
        has_list = bool(re.search(r'(\d+[.)]\s|\b(first|second|third|finally|additionally|moreover|furthermore)\b)', response_lower))
        has_structure = bool(re.search(r'(for example|such as|including|specifically|in particular)', response_lower))
        
        structure_score = (has_list * 2 + has_structure * 2) - truncated
        
        # === 6. RESPONSE LENGTH AND INFORMATION DENSITY ===
        # Not just length, but information per unit length
        
        # Optimal length scoring (not too short, not too long with filler)
        length_score = 0
        if total_words < 5:
            length_score = -5
        elif total_words < 10:
            length_score = -2
        elif total_words < 20:
            length_score = 2
        elif total_words < 60:
            length_score = 5
        elif total_words < 120:
            length_score = 4
        elif total_words < 200:
            length_score = 3
        else:
            length_score = 2
        
        # Information density: unique content words per sentence
        info_density = len(unique_content) / num_sentences if num_sentences > 0 else 0
        density_score = min(info_density / 5.0, 1.0) * 5  # 0-5
        
        # === 7. QUERY-RESPONSE RELEVANCE (without simple word overlap) ===
        # Use character n-gram similarity instead of word overlap
        def char_ngrams(text, n=3):
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        query_ngrams = char_ngrams(query, 4)
        response_ngrams = char_ngrams(response, 4)
        
        if query_ngrams and response_ngrams:
            # Jaccard-like but weighted toward query coverage
            query_covered = len(query_ngrams & response_ngrams) / max(len(query_ngrams), 1)
            relevance_score = query_covered * 8  # 0-8
        else:
            relevance_score = 2  # neutral
        
        # Also check if response addresses the query type
        query_lower = query.lower()
        query_types = {
            'explain': ['means', 'refers to', 'is defined', 'describes', 'signifies', 'implies'],
            'compare': ['both', 'while', 'whereas', 'however', 'unlike', 'similar', 'differ', 'contrast'],
            'describe': ['is a', 'involves', 'includes', 'consists', 'features', 'characterized'],
            'list': ['first', 'second', 'third', 'also', 'additionally', 'another', 'furthermore'],
            'how': ['step', 'process', 'method', 'by', 'through', 'using', 'procedure'],
            'why': ['because', 'reason', 'due to', 'since', 'therefore', 'as a result'],
            'what': ['is', 'refers', 'means', 'defined', 'known as']
        }
        
        query_type_bonus = 0
        for qtype, indicators in query_types.items():
            if qtype in query_lower:
                matches = sum(1 for ind in indicators if ind in response_lower)
                query_type_bonus = min(matches * 0.7, 3)
                break
        
        # === 8. EXPLANATORY DEPTH ===
        # Causal and explanatory connectors indicate deeper analysis
        explanatory_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'this means', 'in other words', 'for instance',
            'for example', 'specifically', 'in particular', 'namely',
            'that is', 'which means', 'this suggests', 'this indicates',
            'leads to', 'results in', 'caused by', 'due to'
        ]
        
        explanation_count = sum(1 for m in explanatory_markers if m in response_lower)
        explanation_score = min(explanation_count * 1.2, 6)
        
        # === FINAL SCORE COMPOSITION ===
        raw_score = (
            sent_variety_score * 0.8 +      # 0-4
            specificity_score * 1.0 +         # 0-12
            certainty_score * 1.0 +           # -inf to 4
            structure_score * 1.0 +           # -3 to 4
            length_score * 1.2 +              # -6 to 6
            density_score * 1.0 +             # 0-5
            relevance_score * 1.0 +           # 0-8
            query_type_bonus * 1.0 +          # 0-3
            explanation_score * 0.8 -          # 0-4.8
            repetition_penalty * 1.0           # 0-25
        )
        
        # Normalize to 0-100 range
        # Theoretical range: roughly -30 to 50
        # Map to 0-100
        final_score = (raw_score + 30) * (100 / 80)
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 25.0
            return 0.0
        except:
            return 0.0