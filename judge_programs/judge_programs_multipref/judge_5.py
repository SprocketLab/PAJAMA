def judging_function(query, response):
    """
    Evaluate response relevance using a question-type intent matching and 
    semantic coverage approach based on:
    1. Query intent classification and fulfillment detection
    2. TF-IDF-inspired weighted term matching with IDF approximation
    3. Query decomposition into sub-questions/aspects and coverage measurement
    4. Response coherence flow analysis (sentence-to-sentence topic drift)
    5. Penalty for filler/generic content ratio
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.0
        
        # ---- Utility functions ----
        def tokenize(text):
            """Lowercase tokenization, removing punctuation."""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return [w for w in text.split() if len(w) > 0]
        
        def get_stopwords():
            return set([
                'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                'just', 'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that',
                'this', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
                'what', 'which', 'who', 'whom', 'am', 'about', 'up', 'down', 'also',
                're', 've', 'll', 'd', 'm'
            ])
        
        stopwords = get_stopwords()
        
        def content_words(tokens):
            return [t for t in tokens if t not in stopwords and len(t) > 1]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        query_content = content_words(query_tokens)
        response_content = content_words(response_tokens)
        
        if not query_content or not response_content:
            return 1.0
        
        # ---- 1. Query Intent Classification & Fulfillment ----
        query_lower = query.lower()
        
        # Classify query type
        intent_type = 'informational'  # default
        if re.search(r'^(how|what steps|can you (show|explain|help).*how)', query_lower):
            intent_type = 'how_to'
        elif re.search(r'^(what (is|are|was|were|does|do))', query_lower):
            intent_type = 'definition'
        elif re.search(r'^(why|what.*reason)', query_lower):
            intent_type = 'explanation'
        elif re.search(r'^(do you|should|is it|are there|can|would|could)', query_lower):
            intent_type = 'yes_no'
        elif re.search(r'^(list|what are (some|ideas|ways)|give me|suggest|any suggestions|ideas)', query_lower):
            intent_type = 'list'
        elif re.search(r'(recipe|how to (make|cook|bake|brew|prepare))', query_lower):
            intent_type = 'procedural'
        elif re.search(r'(find|calculate|solve|compute|determine)', query_lower):
            intent_type = 'problem_solving'
        
        # Check if response fulfills intent
        response_lower = response.lower()
        intent_score = 0.5  # baseline
        
        if intent_type == 'how_to':
            # Look for steps, instructions, numbered items
            has_steps = bool(re.search(r'(\d+[\.\):]|\bstep\b|\bfirst\b.*\bthen\b)', response_lower))
            has_action_verbs = len(re.findall(r'\b(start|begin|next|then|finally|first|second|third)\b', response_lower))
            intent_score = 0.3 + 0.4 * has_steps + 0.1 * min(has_action_verbs / 3.0, 1.0)
        elif intent_type == 'definition':
            has_definition = bool(re.search(r'\b(is|are|refers to|means|defined as)\b', response_lower))
            intent_score = 0.4 + 0.4 * has_definition
        elif intent_type == 'explanation':
            has_because = bool(re.search(r'\b(because|reason|due to|since|therefore|thus|cause)\b', response_lower))
            intent_score = 0.3 + 0.5 * has_because
        elif intent_type == 'yes_no':
            has_direct_answer = bool(re.search(r'^(yes|no|i do|i don|i think|i believe|absolutely|certainly|not necessarily)', response_lower))
            intent_score = 0.3 + 0.5 * has_direct_answer
        elif intent_type == 'list':
            list_items = len(re.findall(r'(?:^|\n)\s*[\-\*\•]|\d+[\.\)]', response))
            intent_score = 0.3 + 0.5 * min(list_items / 4.0, 1.0)
        elif intent_type == 'procedural':
            has_ingredients = bool(re.search(r'\b(ingredient|recipe|cup|tablespoon|teaspoon|tbsp|tsp|oz)\b', response_lower))
            has_steps = bool(re.search(r'(\d+[\.\):]|\bstep\b)', response_lower))
            intent_score = 0.3 + 0.3 * has_ingredients + 0.3 * has_steps
        elif intent_type == 'problem_solving':
            has_math = bool(re.search(r'[=×÷\+\-\*\/]\s*\d|\\frac|\^2|\bformula\b|\bcalculate\b', response_lower))
            has_steps = bool(re.search(r'(\bstep\b|\d+[\.\):])', response_lower))
            intent_score = 0.3 + 0.35 * has_math + 0.25 * has_steps
        else:
            intent_score = 0.5
        
        # ---- 2. TF-IDF-inspired Weighted Term Matching ----
        # Approximate IDF: rarer words in a "pseudo-corpus" get higher weight
        # Use word length and character patterns as proxy for specificity
        def pseudo_idf(word):
            """Approximate IDF based on word characteristics."""
            base = 1.0
            # Longer words tend to be more specific
            if len(word) >= 8:
                base += 0.5
            elif len(word) >= 6:
                base += 0.3
            # Words with numbers are often specific
            if re.search(r'\d', word):
                base += 0.4
            # Very common short words
            if len(word) <= 3:
                base -= 0.3
            return max(base, 0.3)
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        # Weighted recall of query terms in response
        weighted_hits = 0.0
        weighted_total = 0.0
        for word in query_content_set:
            w = pseudo_idf(word)
            weighted_total += w
            if word in response_content_set:
                weighted_hits += w
        
        weighted_recall = weighted_hits / weighted_total if weighted_total > 0 else 0.0
        
        # ---- 3. Query Decomposition & Aspect Coverage ----
        # Extract key noun phrases / aspects from the query
        def extract_aspects(text, tokens_content):
            """Extract meaningful bigrams and trigrams as aspects."""
            tokens = tokenize(text)
            aspects = []
            # Bigrams from content words in sequence
            for i in range(len(tokens) - 1):
                if tokens[i] not in stopwords and tokens[i+1] not in stopwords:
                    aspects.append(tokens[i] + ' ' + tokens[i+1])
            # Trigrams
            for i in range(len(tokens) - 2):
                non_stop = sum(1 for t in tokens[i:i+3] if t not in stopwords)
                if non_stop >= 2:
                    aspects.append(' '.join(tokens[i:i+3]))
            return aspects
        
        query_aspects = extract_aspects(query, query_content)
        response_text_lower = ' '.join(response_tokens)
        
        if query_aspects:
            aspects_covered = sum(1 for asp in query_aspects if asp in response_text_lower)
            aspect_coverage = aspects_covered / len(query_aspects)
        else:
            aspect_coverage = weighted_recall  # fallback
        
        # ---- 4. Sentence-level Topic Coherence (anti-drift measure) ----
        def split_sentences(text):
            """Split text into sentences."""
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 10]
        
        response_sentences = split_sentences(response)
        
        if len(response_sentences) >= 2:
            # Measure how many sentences contain query-relevant terms
            relevant_sentence_count = 0
            for sent in response_sentences:
                sent_tokens = set(content_words(tokenize(sent)))
                overlap = sent_tokens & query_content_set
                if len(overlap) >= 1:
                    relevant_sentence_count += 1
            
            coherence_score = relevant_sentence_count / len(response_sentences)
            
            # Also check if topic drifts: compare first half vs second half relevance
            mid = len(response_sentences) // 2
            first_half = response_sentences[:mid] if mid > 0 else response_sentences[:1]
            second_half = response_sentences[mid:] if mid > 0 else response_sentences[1:]
            
            def half_relevance(sents):
                if not sents:
                    return 0.5
                rel = 0
                for s in sents:
                    st = set(content_words(tokenize(s)))
                    if st & query_content_set:
                        rel += 1
                return rel / len(sents)
            
            first_rel = half_relevance(first_half)
            second_rel = half_relevance(second_half)
            
            # Penalize if second half drifts away
            drift_penalty = max(0, (first_rel - second_rel) * 0.3)
        else:
            coherence_score = 0.5
            drift_penalty = 0.0
        
        # ---- 5. Filler / Generic Content Ratio ----
        filler_phrases = [
            'great question', 'that\'s a great', 'glad you asked', 'absolutely',
            'of course', 'no problem', 'sure thing', 'happy to help',
            'let me know if', 'hope this helps', 'feel free to',
            'don\'t hesitate', 'i hope', 'good luck'
        ]
        
        generic_phrases = [
            'it is important to note', 'it should be noted', 'in conclusion',
            'there are many', 'there are several', 'it depends on',
            'as you can see', 'as mentioned above', 'in general',
            'on the other hand', 'having said that', 'that being said'
        ]
        
        filler_count = sum(1 for p in filler_phrases if p in response_lower)
        generic_count = sum(1 for p in generic_phrases if p in response_lower)
        
        total_sentences = max(len(response_sentences), 1)
        filler_ratio = (filler_count + generic_count * 0.5) / total_sentences
        filler_penalty = min(filler_ratio * 0.15, 0.2)
        
        # ---- 6. Engagement & Directness ----
        # Does the response address the user directly and get to the point?
        first_sentence = response_sentences[0] if response_sentences else response[:200]
        first_sent_tokens = set(content_words(tokenize(first_sentence)))
        first_sent_overlap = len(first_sent_tokens & query_content_set)
        
        directness_score = min(first_sent_overlap / max(len(query_content_set), 1), 1.0)
        
        # ---- 7. Semantic Field Expansion ----
        # Check if response introduces related/synonymous terms (indicates deeper understanding)
        # Use character-level similarity between response words and query words
        def char_trigrams(word):
            if len(word) < 3:
                return set()
            return set(word[i:i+3] for i in range(len(word) - 2))
        
        query_trigrams = set()
        for w in query_content_set:
            query_trigrams |= char_trigrams(w)
        
        if query_trigrams:
            response_only_words = response_content_set - query_content_set
            semantically_related = 0
            for w in response_only_words:
                w_trigrams = char_trigrams(w)
                if w_trigrams:
                    overlap_ratio = len(w_trigrams & query_trigrams) / len(w_trigrams)
                    if overlap_ratio > 0.3:
                        semantically_related += 1
            
            semantic_expansion = min(semantically_related / max(len(response_only_words), 1) * 2, 0.5)
        else:
            semantic_expansion = 0.0
        
        # ---- 8. Response Substance ----
        # Penalize very short or truncated responses less if they're highly relevant
        # Reward responses with good length that maintain relevance
        word_count = len(response_tokens)
        if word_count < 20:
            length_factor = 0.6
        elif word_count < 50:
            length_factor = 0.8
        elif word_count < 300:
            length_factor = 1.0
        else:
            length_factor = 0.95  # slight penalty for very long (might be padded)
        
        # ---- Combine all signals ----
        # Weights chosen to emphasize direct relevance
        score = (
            intent_score * 25.0 +           # Intent fulfillment (0-25)
            weighted_recall * 20.0 +         # Weighted term recall (0-20)
            aspect_coverage * 15.0 +         # Aspect coverage (0-15)
            coherence_score * 12.0 +         # Sentence-level coherence (0-12)
            directness_score * 10.0 +        # Gets to point quickly (0-10)
            semantic_expansion * 8.0 +        # Related term expansion (0-4 effectively)
            length_factor * 5.0 -            # Length appropriateness (0-5)
            drift_penalty * 10.0 -           # Topic drift penalty
            filler_penalty * 10.0            # Filler penalty
        )
        
        # Normalize to 0-100 range
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
    
    except Exception:
        return 25.0  # Safe fallback