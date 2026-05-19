def judging_function(query, response):
    """
    Evaluates relevance of an LLM response to a query.
    Uses keyword overlap, topic alignment, structural quality, and direct address detection.
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        import re
        import math
        from collections import Counter
        
        # Handle edge cases
        if not query or not response:
            return 0.0
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        query = query.strip()
        response = response.strip()
        
        if len(response) < 5:
            return 0.0
        if len(query) < 3:
            return 5.0  # Can't evaluate relevance without a real query
        
        # --- Tokenization helpers ---
        def tokenize(text):
            """Lowercase tokenization, removing punctuation."""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            tokens = text.split()
            return tokens
        
        def get_ngrams(tokens, n):
            """Get n-grams from token list."""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Common English stopwords
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'that',
            'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his',
            'she', 'her', 'hers', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'am', 'also', 'like', 'get', 'got', 'much', 'many', 'well',
            'really', 'even', 'still', 'back', 'one', 'two', 'make', 'made',
            'know', 'think', 'see', 'come', 'go', 'going', 'say', 'said',
            'don', 'doesn', 'didn', 'won', 'wouldn', 'couldn', 'shouldn',
            've', 're', 'll', 't', 's', 'd', 'm'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content = [t for t in query_tokens if t not in STOPWORDS and len(t) > 2]
        response_content = [t for t in response_tokens if t not in STOPWORDS and len(t) > 2]
        
        if not query_content:
            query_content = [t for t in query_tokens if len(t) > 1]
        
        # --- Feature 1: Content word overlap (Jaccard + weighted) ---
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        if query_content_set:
            overlap = query_content_set & response_content_set
            # Jaccard similarity
            union = query_content_set | response_content_set
            jaccard = len(overlap) / len(union) if union else 0
            
            # Query coverage: what fraction of query terms appear in response
            query_coverage = len(overlap) / len(query_content_set)
            
            # Weighted overlap: terms that appear more in query are more important
            query_content_counts = Counter(query_content)
            weighted_overlap = 0
            total_weight = sum(query_content_counts.values())
            for term in overlap:
                weighted_overlap += query_content_counts[term]
            weighted_coverage = weighted_overlap / total_weight if total_weight > 0 else 0
        else:
            jaccard = 0
            query_coverage = 0
            weighted_coverage = 0
        
        # --- Feature 2: Bigram overlap ---
        query_bigrams = set(get_ngrams(query_content, 2))
        response_bigrams = set(get_ngrams(response_content, 2))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0
        
        # --- Feature 3: TF-IDF inspired relevance ---
        # Use term frequency in response weighted by "importance" (inverse frequency proxy)
        # Rare query terms found in response are more indicative of relevance
        all_tokens_combined = query_content + response_content
        combined_counts = Counter(all_tokens_combined)
        total_unique = len(set(all_tokens_combined))
        
        tfidf_score = 0
        if query_content_set and total_unique > 0:
            for term in query_content_set:
                if term in response_content_set:
                    # IDF proxy: rarer terms in combined text get higher weight
                    idf = math.log(1 + total_unique / (1 + combined_counts[term]))
                    # TF in response
                    tf = response_content.count(term) / (len(response_content) + 1)
                    tfidf_score += tf * idf
            # Normalize
            max_possible = len(query_content_set) * math.log(1 + total_unique)
            tfidf_score = tfidf_score / max_possible if max_possible > 0 else 0
        
        # --- Feature 4: Topic/domain keyword detection ---
        # Extract likely topic words (longer, less common words from query)
        topic_words = [t for t in query_content if len(t) > 4]
        if topic_words:
            topic_set = set(topic_words)
            topic_found = sum(1 for t in topic_set if t in response_content_set)
            topic_coverage = topic_found / len(topic_set)
        else:
            topic_coverage = query_coverage  # fallback
        
        # --- Feature 5: Response length quality ---
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Moderate length is good; too short is bad; very long can be okay
        if resp_len < 10:
            length_score = 0.2
        elif resp_len < 20:
            length_score = 0.4
        elif resp_len < 50:
            length_score = 0.7
        elif resp_len < 150:
            length_score = 1.0
        elif resp_len < 300:
            length_score = 0.95
        else:
            length_score = 0.85
        
        # --- Feature 6: Direct address detection ---
        # Check if response seems to directly engage with the query
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Detect question type and check if response addresses it
        direct_address_score = 0.5  # neutral baseline
        
        # Check for question words and appropriate response patterns
        question_patterns = {
            'how': ['by ', 'through ', 'using ', 'step', 'method', 'way', 'process', 'you can', 'first'],
            'why': ['because', 'reason', 'due to', 'since', 'result of', 'caused by', 'explanation'],
            'what': ['is ', 'are ', 'refers to', 'means', 'defined', 'essentially', 'basically'],
            'when': ['during', 'after', 'before', 'in ', 'year', 'time', 'period', 'century'],
            'where': ['in ', 'at ', 'location', 'place', 'region', 'area', 'country'],
            'who': ['person', 'people', 'individual', 'group', 'team', 'name'],
            'is there': ['yes', 'no', 'there is', 'there are', 'indeed', 'argument', 'exist'],
            'can': ['yes', 'no', 'possible', 'able', 'capable', 'you can'],
            'do': ['yes', 'no', 'typically', 'usually', 'generally', 'indeed'],
        }
        
        for q_word, r_patterns in question_patterns.items():
            if q_word in query_lower[:80]:
                matches = sum(1 for p in r_patterns if p in response_lower[:300])
                if matches > 0:
                    direct_address_score = min(1.0, 0.5 + matches * 0.15)
                break
        
        # --- Feature 7: Substantiveness ---
        # Penalize very generic / boilerplate responses
        boilerplate_phrases = [
            'welcome to', 'please read our rules', 'your comments will be removed',
            'this is a bot', 'i am a bot', 'auto-generated', 'moderator',
            'removed if they', 'before commenting', 'up to standard',
            'can you please describe', 'do not fear'
        ]
        boilerplate_penalty = 0
        for phrase in boilerplate_phrases:
            if phrase in response_lower:
                boilerplate_penalty += 0.15
        boilerplate_penalty = min(boilerplate_penalty, 0.6)
        
        # --- Feature 8: Specificity and detail ---
        # Count specific indicators: numbers, proper nouns (capitalized words), technical terms
        specificity_score = 0
        
        # Numbers in response (dates, quantities, etc.)
        numbers = re.findall(r'\d+', response)
        if numbers:
            specificity_score += min(0.15, len(numbers) * 0.03)
        
        # Quoted terms, code blocks, or references
        if '```' in response or '`' in response:
            specificity_score += 0.1
        if '"' in response or "'" in response:
            specificity_score += 0.05
        
        # Capitalized words (potential proper nouns / specific references)
        caps_words = re.findall(r'\b[A-Z][a-z]+\b', response[1:])  # skip first word
        if caps_words:
            specificity_score += min(0.15, len(set(caps_words)) * 0.02)
        
        # Longer content words suggest more specific/technical language
        long_words = [t for t in response_content if len(t) > 7]
        if response_content:
            long_word_ratio = len(long_words) / (len(response_content) + 1)
            specificity_score += min(0.15, long_word_ratio * 0.5)
        
        specificity_score = min(specificity_score, 0.5)
        
        # --- Feature 9: Semantic coherence via shared context ---
        # Check if response shares multi-word phrases or concepts with query
        query_trigrams = set(get_ngrams(query_tokens, 3))
        response_trigrams = set(get_ngrams(response_tokens, 3))
        if query_trigrams:
            trigram_overlap = len(query_trigrams & response_trigrams) / len(query_trigrams)
        else:
            trigram_overlap = 0
        
        # --- Feature 10: Engagement depth ---
        # Responses that explain, elaborate, or provide examples are better
        explanation_markers = [
            'for example', 'for instance', 'such as', 'in other words',
            'specifically', 'essentially', 'in particular', 'this means',
            'the reason', 'because', 'therefore', 'however', 'moreover',
            'additionally', 'furthermore', 'in fact', 'trade-off',
            'on the other hand', 'in contrast', 'as a result'
        ]
        explanation_count = sum(1 for m in explanation_markers if m in response_lower)
        explanation_score = min(0.3, explanation_count * 0.06)
        
        # --- Combine all features ---
        # Weighted combination
        score = (
            query_coverage * 2.5 +          # 0-2.5: core relevance
            weighted_coverage * 1.5 +        # 0-1.5: weighted relevance
            jaccard * 1.0 +                  # 0-1.0: overall similarity
            bigram_overlap * 1.5 +           # 0-1.5: phrase-level match
            trigram_overlap * 1.0 +          # 0-1.0: exact phrase match
            tfidf_score * 15.0 +             # 0-~2.0: tf-idf relevance
            topic_coverage * 2.0 +           # 0-2.0: topic alignment
            length_score * 1.0 +             # 0-1.0: appropriate length
            direct_address_score * 1.5 +     # 0-1.5: addresses the question
            specificity_score * 2.0 +        # 0-1.0: specific details
            explanation_score * 2.0 -         # 0-0.6: explanatory depth
            boilerplate_penalty * 3.0        # penalty for boilerplate
        )
        
        # Normalize to 0-10 range
        # Theoretical max is roughly: 2.5+1.5+1+1.5+1+2+2+1+1.5+1+0.6 = 15.6
        # Practical max is around 10-12
        score = max(0, min(10, score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This makes the middle range more discriminative
        normalized = score / 10.0  # 0 to 1
        # Gentle S-curve
        if normalized <= 0.5:
            adjusted = 2 * normalized * normalized
        else:
            adjusted = 1 - 2 * (1 - normalized) * (1 - normalized)
        
        final_score = adjusted * 10.0
        
        return round(final_score, 3)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            # Last resort: simple length-based heuristic
            if response and query:
                return 3.0
        except:
            pass
        return 2.0