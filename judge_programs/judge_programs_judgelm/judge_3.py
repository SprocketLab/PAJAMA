def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query term importance weighting, response coherence analysis, and 
    semantic field coverage scoring.
    
    This variant uses:
    - TF-IDF-like query term importance weighting
    - Query intent extraction and coverage measurement
    - Response informativeness scoring (entropy-based)
    - Penalty for repetition and off-topic content ratio
    - Sentence-level relevance aggregation
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 2:
            return 0.5
        
        # --- Tokenization helpers ---
        def tokenize(text):
            text = text.lower()
            tokens = re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
            return tokens
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 3]
        
        # Common English stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this',
            'these', 'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'i',
            'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
            'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their',
            'also', 'about', 'up', 'down', 'any', 'much', 'many', 'well', 'back',
            'make', 'like', 'get', 'got', 'go', 'going', 'come', 'know', 'take',
            'see', 'think', 'say', 'said', 'tell', 'told', 'give', 'given'
        }
        
        # Document frequency approximation - words that appear in many contexts
        # (higher DF = less discriminative)
        high_df_words = {
            'please', 'help', 'want', 'question', 'answer', 'know', 'find',
            'information', 'tell', 'explain', 'describe', 'provide', 'give',
            'example', 'show', 'write', 'create', 'make', 'list', 'identify',
            'name', 'following', 'based', 'using', 'use', 'new', 'way', 'good',
            'best', 'different', 'first', 'last', 'one', 'two', 'three',
            'thing', 'things', 'people', 'time', 'year', 'day', 'world',
            'number', 'part', 'place', 'case', 'point', 'work', 'right'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # --- 1. TF-IDF-like Query Term Importance and Coverage ---
        # Assign importance weights to query terms
        query_content_words = [w for w in query_tokens if w not in stopwords]
        if not query_content_words:
            query_content_words = query_tokens  # fallback
        
        # IDF-like weighting: rarer query words are more important
        word_importance = {}
        for w in query_content_words:
            if w in high_df_words:
                word_importance[w] = 1.0
            elif len(w) <= 3:
                word_importance[w] = 1.5
            elif len(w) <= 6:
                word_importance[w] = 2.5
            else:
                word_importance[w] = 3.5  # longer, rarer words are more important
        
        # Boost proper nouns / capitalized words in original query
        original_query_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        for w in original_query_words:
            wl = w.lower()
            if wl in word_importance:
                word_importance[wl] *= 1.5
        
        response_word_set = set(response_tokens)
        response_word_counts = Counter(response_tokens)
        
        # Calculate weighted coverage
        total_importance = sum(word_importance.values())
        if total_importance == 0:
            total_importance = 1.0
        
        covered_importance = 0.0
        for w, imp in word_importance.items():
            if w in response_word_set:
                # Boost if the word appears multiple times (up to a point)
                freq_boost = min(1.0 + 0.2 * (response_word_counts[w] - 1), 1.5)
                covered_importance += imp * freq_boost
        
        coverage_score = min(covered_importance / total_importance, 1.5)
        
        # --- 2. Semantic Field / Topic Alignment ---
        # Build "semantic fields" from query content words using character trigrams
        def char_trigrams(word):
            if len(word) < 3:
                return {word}
            return {word[i:i+3] for i in range(len(word) - 2)}
        
        query_trigrams = set()
        for w in query_content_words:
            query_trigrams.update(char_trigrams(w))
        
        response_content_words = [w for w in response_tokens if w not in stopwords]
        if not response_content_words:
            response_content_words = response_tokens
        
        response_trigrams = set()
        for w in response_content_words:
            response_trigrams.update(char_trigrams(w))
        
        if query_trigrams:
            trigram_overlap = len(query_trigrams & response_trigrams) / len(query_trigrams)
        else:
            trigram_overlap = 0.0
        
        # Also check for partial word matches (stemming approximation)
        def get_stem(word):
            """Very rough stemming by taking first 5+ chars"""
            if len(word) <= 4:
                return word
            # Remove common suffixes
            for suffix in ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful', 
                          'less', 'ous', 'ive', 'ing', 'ied', 'ies', 'ers', 'est',
                          'ity', 'ism', 'ist', 'ent', 'ant', 'ure', 'ate', 'ize',
                          'ise', 'ify', 'fy', 'ly', 'ed', 'er', 'es', 'al', 'en', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        query_stems = set(get_stem(w) for w in query_content_words)
        response_stems = Counter(get_stem(w) for w in response_content_words)
        
        stem_coverage = 0
        for qs in query_stems:
            if qs in response_stems:
                stem_coverage += 1
            else:
                # Check if any response stem starts with the query stem or vice versa
                for rs in response_stems:
                    if (len(qs) >= 3 and len(rs) >= 3 and 
                        (rs.startswith(qs[:3]) or qs.startswith(rs[:3]))):
                        stem_coverage += 0.3
                        break
        
        if query_stems:
            stem_score = stem_coverage / len(query_stems)
        else:
            stem_score = 0.0
        
        # --- 3. Sentence-Level Relevance Aggregation ---
        response_sentences = get_sentences(response)
        if not response_sentences:
            response_sentences = [response]
        
        query_content_set = set(query_content_words)
        
        sentence_relevance_scores = []
        for sent in response_sentences:
            sent_tokens = tokenize(sent)
            sent_content = [w for w in sent_tokens if w not in stopwords]
            if not sent_content:
                sentence_relevance_scores.append(0.0)
                continue
            
            # What fraction of this sentence's content words relate to the query?
            relevant_count = sum(1 for w in sent_content if w in query_content_set or 
                               get_stem(w) in query_stems)
            sent_relevance = relevant_count / len(sent_content) if sent_content else 0
            sentence_relevance_scores.append(sent_relevance)
        
        if sentence_relevance_scores:
            # Weight earlier sentences more heavily
            weighted_sent_relevance = 0.0
            total_weight = 0.0
            for i, score in enumerate(sentence_relevance_scores):
                weight = 1.0 / (1.0 + 0.3 * i)  # Decay weight for later sentences
                weighted_sent_relevance += score * weight
                total_weight += weight
            avg_sent_relevance = weighted_sent_relevance / total_weight if total_weight > 0 else 0
            
            # Fraction of sentences that have ANY relevance
            relevant_sent_fraction = sum(1 for s in sentence_relevance_scores if s > 0.05) / len(sentence_relevance_scores)
        else:
            avg_sent_relevance = 0.0
            relevant_sent_fraction = 0.0
        
        # --- 4. Response Quality Indicators ---
        
        # 4a. Response length appropriateness
        query_len = len(query_tokens)
        resp_len = len(response_tokens)
        
        # Very short responses are often low quality
        if resp_len < 3:
            length_score = 0.2
        elif resp_len < 8:
            length_score = 0.5
        elif resp_len < 15:
            length_score = 0.75
        else:
            length_score = 1.0
        
        # 4b. Repetition penalty
        if resp_len > 10:
            unique_ratio = len(set(response_tokens)) / resp_len
            # Check for repeated phrases (bigrams)
            bigrams = [(response_tokens[i], response_tokens[i+1]) for i in range(len(response_tokens)-1)]
            if bigrams:
                bigram_counts = Counter(bigrams)
                max_bigram_repeat = max(bigram_counts.values())
                bigram_unique_ratio = len(set(bigrams)) / len(bigrams)
            else:
                max_bigram_repeat = 1
                bigram_unique_ratio = 1.0
            
            repetition_penalty = 1.0
            if unique_ratio < 0.3:
                repetition_penalty *= 0.4
            elif unique_ratio < 0.5:
                repetition_penalty *= 0.7
            
            if max_bigram_repeat > 5:
                repetition_penalty *= 0.5
            elif max_bigram_repeat > 3:
                repetition_penalty *= 0.7
            
            if bigram_unique_ratio < 0.3:
                repetition_penalty *= 0.6
        else:
            repetition_penalty = 1.0
        
        # 4c. Off-topic content detection
        # Check if response contains lots of code, HTML, or other non-text artifacts
        code_patterns = [r'import\s+\w+', r'def\s+\w+\s*\(', r'class\s+\w+',
                        r'\{[^}]{20,}\}', r'<[a-z]+>[^<]*</[a-z]+>',
                        r'```', r'#include', r'function\s*\(']
        
        # Only penalize code if the query doesn't ask for code
        query_asks_code = any(w in query.lower() for w in ['code', 'program', 'function', 'html', 'script', 'python', 'tag', 'css', 'javascript'])
        
        code_penalty = 1.0
        if not query_asks_code:
            code_matches = sum(1 for p in code_patterns if re.search(p, response))
            if code_matches >= 3:
                code_penalty = 0.3
            elif code_matches >= 2:
                code_penalty = 0.6
            elif code_matches >= 1:
                code_penalty = 0.85
        
        # 4d. Response informativeness (entropy-like measure)
        if response_content_words:
            word_freq = Counter(response_content_words)
            total = len(response_content_words)
            entropy = 0.0
            for count in word_freq.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalize entropy
            max_entropy = math.log2(max(total, 2))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            norm_entropy = 0.0
        
        # 4e. Check if response seems to address the query type
        # Detect query type
        query_lower = query.lower()
        is_question = any(query_lower.strip().startswith(w) for w in 
                         ['what', 'where', 'when', 'who', 'why', 'how', 'is ', 'are ', 
                          'can ', 'do ', 'does ', 'did ', 'will ', 'would ', 'could ',
                          'should ']) or '?' in query
        
        is_instruction = any(query_lower.strip().startswith(w) for w in 
                            ['write', 'create', 'make', 'list', 'identify', 'explain',
                             'describe', 'rewrite', 'generate', 'find', 'tell', 'give',
                             'provide', 'summarize', 'translate', 'convert', 'calculate',
                             'determine', 'compare', 'analyze', 'remove', 'add'])
        
        # For questions, check if response provides an answer-like structure
        answer_quality = 1.0
        if is_question:
            # Very short non-committal responses to questions are bad
            if resp_len < 5 and not any(w in response_word_set for w in query_content_words[:3]):
                answer_quality = 0.4
        
        if is_instruction:
            # Check if the response actually attempts the task
            if resp_len < 3:
                answer_quality = 0.3
        
        # --- 5. Combine Scores ---
        # Weighted combination of all signals
        
        # Core relevance signals
        relevance_composite = (
            0.30 * coverage_score +          # Direct query term coverage (TF-IDF weighted)
            0.15 * trigram_overlap +          # Character-level topic similarity
            0.20 * stem_score +              # Stem-based coverage
            0.20 * avg_sent_relevance +      # Sentence-level relevance
            0.15 * relevant_sent_fraction    # Fraction of on-topic sentences
        )
        
        # Quality modifiers
        quality_composite = (
            length_score * 
            repetition_penalty * 
            code_penalty * 
            answer_quality
        )
        
        # Informativeness bonus (small)
        info_bonus = 0.1 * norm_entropy
        
        # Final score
        raw_score = relevance_composite * quality_composite + info_bonus
        
        # Scale to 0-10 range
        # The raw_score typically ranges from 0 to about 1.2
        final_score = raw_score * 8.5
        
        # Clamp
        final_score = max(0.5, min(10.0, final_score))
        
        # Small bonus for responses that are substantive but not excessively long
        if 20 <= resp_len <= 200 and relevance_composite > 0.2:
            final_score = min(10.0, final_score + 0.5)
        
        return round(final_score, 2)
        
    except Exception:
        return 3.0