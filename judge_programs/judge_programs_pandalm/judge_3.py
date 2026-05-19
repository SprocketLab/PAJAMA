def judging_function(query, response):
    """
    Evaluates response relevance using a TF-IDF-inspired weighted term matching approach
    combined with query intent coverage analysis and response quality signals.
    
    This variant focuses on:
    1. IDF-weighted query term coverage (not simple overlap or Jaccard)
    2. Query intent decomposition (breaking query into semantic components)
    3. Response coherence and informativeness signals
    4. Penalty for repetition, off-topic filler, and low information density
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if not response or len(response) < 2:
            return 0.0
        
        # --- Tokenization helpers ---
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'get', 'got', 'like', 'make', 'made', 'go', 'going',
            'come', 'take', 'give', 'given', 'well', 'back', 'even', 'still',
        }
        
        # Common filler / low-info phrases
        FILLER_PHRASES = [
            'it is important to note', 'it should be noted', 'in conclusion',
            'as we can see', 'it goes without saying', 'needless to say',
            'at the end of the day', 'in other words', 'that being said',
        ]
        
        def tokenize(text):
            """Tokenize into lowercase words."""
            return re.findall(r'[a-z]+(?:\'[a-z]+)?', text.lower())
        
        def content_words(tokens):
            """Filter to content words only."""
            return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        
        def get_stems(tokens):
            """Very simple suffix-stripping pseudo-stemmer."""
            stemmed = []
            for t in tokens:
                # Simple suffix removal for matching purposes
                for suffix in ['tion', 'sion', 'ment', 'ness', 'ious', 'eous', 
                               'ling', 'ting', 'ing', 'ful', 'less', 'able', 'ible',
                               'ated', 'ized', 'ally', 'ely', 'ity',
                               'ed', 'er', 'ly', 'es', 's']:
                    if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                        t = t[:-len(suffix)]
                        break
                stemmed.append(t)
            return stemmed
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not response_tokens:
            return 0.0
        
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        query_stems = get_stems(query_content)
        response_stems = get_stems(response_content)
        
        # --- Component 1: IDF-weighted query term coverage ---
        # Simulate IDF: rarer words in general English get higher weight
        # Use word length as a rough proxy for specificity, plus frequency-based weighting
        COMMON_WORDS_FREQ = {
            'make': 0.1, 'time': 0.1, 'people': 0.1, 'way': 0.1, 'work': 0.1,
            'know': 0.1, 'think': 0.1, 'good': 0.1, 'new': 0.1, 'use': 0.1,
            'say': 0.1, 'help': 0.1, 'tell': 0.1, 'ask': 0.1, 'need': 0.1,
            'want': 0.1, 'look': 0.1, 'find': 0.1, 'thing': 0.1, 'part': 0.1,
            'place': 0.1, 'case': 0.1, 'point': 0.1, 'hand': 0.1, 'high': 0.1,
            'keep': 0.1, 'let': 0.1, 'begin': 0.1, 'seem': 0.1, 'show': 0.1,
            'hear': 0.1, 'play': 0.1, 'run': 0.1, 'move': 0.1, 'try': 0.1,
            'long': 0.1, 'great': 0.1, 'small': 0.1, 'large': 0.1, 'old': 0.1,
            'different': 0.1, 'important': 0.1, 'world': 0.1, 'life': 0.1,
        }
        
        def idf_weight(word):
            """Estimate IDF weight for a word."""
            if word in COMMON_WORDS_FREQ:
                return 1.0
            # Longer, rarer words are more topically specific
            base = 1.5
            if len(word) >= 6:
                base = 2.5
            elif len(word) >= 8:
                base = 3.0
            elif len(word) >= 10:
                base = 3.5
            return base
        
        response_stem_set = set(response_stems)
        response_content_set = set(response_content)
        
        if query_stems:
            total_weight = 0.0
            matched_weight = 0.0
            for i, stem in enumerate(query_stems):
                w = idf_weight(query_content[i] if i < len(query_content) else stem)
                total_weight += w
                # Check both stem match and exact match
                if stem in response_stem_set or (i < len(query_content) and query_content[i] in response_content_set):
                    matched_weight += w
            
            idf_coverage = matched_weight / total_weight if total_weight > 0 else 0.0
        else:
            # Fall back to all-token overlap
            q_set = set(query_tokens)
            r_set = set(response_tokens)
            overlap = q_set & r_set
            idf_coverage = len(overlap) / len(q_set) if q_set else 0.0
        
        # --- Component 2: Query intent decomposition ---
        # Extract key phrases (bigrams and trigrams) from query
        def get_phrases(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        query_bigrams = get_phrases(query_content, 2)
        query_trigrams = get_phrases(query_content, 3)
        response_text_lower = response.lower()
        
        # Check how many query phrases appear in response
        phrase_hits = 0
        phrase_total = len(query_bigrams) + len(query_trigrams)
        
        for phrase in query_bigrams:
            # Check if both words appear near each other in response
            words = phrase.split()
            if all(w in response_text_lower for w in words):
                phrase_hits += 1
        
        for phrase in query_trigrams:
            words = phrase.split()
            if all(w in response_text_lower for w in words):
                phrase_hits += 1.5  # Trigram matches are more valuable
        
        phrase_coverage = phrase_hits / phrase_total if phrase_total > 0 else 0.5
        
        # --- Component 3: Topic coherence via response-to-query semantic density ---
        # What fraction of response content words relate to query topics?
        query_stem_set = set(query_stems)
        query_content_set = set(query_content)
        
        if response_content:
            on_topic_count = 0
            for i, stem in enumerate(response_stems):
                word = response_content[i] if i < len(response_content) else stem
                if stem in query_stem_set or word in query_content_set:
                    on_topic_count += 1
            
            # Also count words that share a common root (3+ char prefix) with query words
            query_prefixes = set()
            for w in query_content:
                if len(w) >= 4:
                    query_prefixes.add(w[:4])
                if len(w) >= 5:
                    query_prefixes.add(w[:5])
            
            for word in response_content:
                if word not in query_content_set:
                    for plen in [5, 4]:
                        if len(word) >= plen and word[:plen] in query_prefixes:
                            on_topic_count += 0.5
                            break
            
            topic_density = min(on_topic_count / len(response_content), 1.0)
        else:
            topic_density = 0.0
        
        # --- Component 4: Response quality signals ---
        
        # 4a. Repetition penalty
        response_word_counter = Counter(response_tokens)
        total_resp_words = len(response_tokens)
        unique_resp_words = len(response_word_counter)
        
        # Type-token ratio (adjusted for length)
        if total_resp_words > 0:
            ttr = unique_resp_words / total_resp_words
            # Adjust for text length (longer texts naturally have lower TTR)
            adjusted_ttr = ttr * math.log(total_resp_words + 1) / math.log(50)
            adjusted_ttr = min(adjusted_ttr, 1.0)
        else:
            adjusted_ttr = 0.0
        
        # Check for excessive word repetition
        if total_resp_words > 5:
            max_freq = max(response_word_counter.values())
            # Exclude stop words from max freq check
            content_counter = Counter(response_content)
            max_content_freq = max(content_counter.values()) if content_counter else 0
            repetition_ratio = max_content_freq / total_resp_words if total_resp_words > 0 else 0
            repetition_penalty = max(0, repetition_ratio - 0.15) * 3  # Penalize if any content word > 15% of text
        else:
            repetition_penalty = 0.0
        
        # 4b. Response length appropriateness
        query_len = len(query_tokens)
        resp_len = total_resp_words
        
        # Very short responses are usually worse
        if resp_len < 3:
            length_score = 0.1
        elif resp_len < 8:
            length_score = 0.4
        elif resp_len < 15:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # But extremely long responses relative to query might be padded
        if resp_len > query_len * 15 and resp_len > 200:
            length_score *= 0.8
        
        # 4c. Sentence structure - responses with multiple sentences tend to be more thorough
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences >= 3:
            structure_score = 1.0
        elif num_sentences == 2:
            structure_score = 0.8
        elif num_sentences == 1:
            structure_score = 0.6
        else:
            structure_score = 0.3
        
        # 4d. Check if response is just echoing the query without adding info
        echo_penalty = 0.0
        if response_content and query_content:
            response_only = set(response_content) - set(query_content)
            if len(response_only) < 3 and len(response_content) > 3:
                echo_penalty = 0.3
        
        # 4e. Filler phrase penalty
        filler_count = 0
        for phrase in FILLER_PHRASES:
            if phrase in response_text_lower:
                filler_count += 1
        filler_penalty = min(filler_count * 0.05, 0.2)
        
        # 4f. Check for noinput / empty-like responses
        if re.match(r'^<noinput>|^n/a|^none|^no response|^no answer', response.strip().lower()):
            return 0.5
        
        # --- Component 5: Direct address of query action verbs ---
        # Extract the main action/intent words from query
        ACTION_VERBS = {
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'generate', 'create', 'write', 'rewrite', 'summarize', 'analyze',
            'evaluate', 'discuss', 'define', 'identify', 'classify', 'suggest',
            'recommend', 'outline', 'illustrate', 'demonstrate', 'elaborate',
            'translate', 'convert', 'calculate', 'determine', 'name', 'give',
            'crop', 'reduce', 'add', 'remove', 'change', 'modify', 'edit',
            'come', 'find', 'show', 'tell', 'mean', 'meant',
        }
        
        query_actions = [t for t in query_tokens if t in ACTION_VERBS]
        
        # Check if the response format matches the query intent
        intent_bonus = 0.0
        query_lower = query.lower()
        
        if 'compare' in query_lower or 'contrast' in query_lower:
            # Look for comparison language in response
            comparison_words = {'both', 'while', 'whereas', 'however', 'unlike', 'similar',
                              'different', 'differ', 'same', 'contrast', 'compare', 'but',
                              'although', 'on the other hand', 'in contrast'}
            comp_found = sum(1 for w in comparison_words if w in response_text_lower)
            intent_bonus = min(comp_found * 0.08, 0.3)
        
        elif 'list' in query_lower or 'provide examples' in query_lower or 'examples of' in query_lower:
            # Look for list-like structure
            commas = response.count(',')
            bullets = response.count('•') + response.count('-') + response.count('*')
            numbers = len(re.findall(r'\d+[.)]\s', response))
            list_signals = commas + bullets * 2 + numbers * 2
            intent_bonus = min(list_signals * 0.05, 0.3)
        
        elif 'explain' in query_lower or 'describe' in query_lower or 'what' in query_lower:
            # Look for explanatory language
            explain_words = {'means', 'refers', 'implies', 'suggests', 'indicates',
                           'because', 'therefore', 'thus', 'hence', 'since', 'result',
                           'process', 'involves', 'when', 'allows'}
            exp_found = sum(1 for w in explain_words if w in response_text_lower)
            intent_bonus = min(exp_found * 0.06, 0.25)
        
        elif 'rewrite' in query_lower or 'rephrase' in query_lower:
            # The response should be different from any input text
            # Just give a small bonus for having content
            if resp_len >= 5:
                intent_bonus = 0.15
        
        # --- Combine all components ---
        # Weights for each component
        score = (
            idf_coverage * 30.0 +          # 0-30: How well query terms are covered
            phrase_coverage * 10.0 +         # 0-10: Query phrase preservation
            topic_density * 15.0 +           # 0-15: How on-topic the response stays
            length_score * 10.0 +            # 0-10: Appropriate length
            structure_score * 8.0 +          # 0-8: Good sentence structure
            adjusted_ttr * 7.0 +             # 0-7: Vocabulary diversity
            intent_bonus * 10.0 +            # 0-3: Matches query intent format
            - repetition_penalty * 15.0 +    # Penalty for repetition
            - echo_penalty * 10.0 +          # Penalty for just echoing query
            - filler_penalty * 5.0           # Penalty for filler
        )
        
        # Ensure score is in reasonable range [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
    
    except Exception as e:
        # Fallback: return a minimal score based on length
        try:
            if response and len(response.strip()) > 10:
                return 5.0
            return 0.0
        except:
            return 0.0