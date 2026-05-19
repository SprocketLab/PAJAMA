def judging_function(query, response):
    """
    Evaluates response relevance to query using TF-IDF-inspired cosine similarity,
    query intent coverage analysis, and semantic coherence scoring.
    
    This variant uses:
    - TF-IDF weighted cosine similarity between query and response
    - Query keyword coverage ratio with IDF weighting
    - N-gram overlap analysis (bigrams and trigrams)
    - Intent detection and fulfillment scoring
    - Penalty for generic/filler content ratio
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 5:
            return 0.0
        
        # --- Text preprocessing ---
        def tokenize(text):
            """Lowercase and extract alphanumeric tokens."""
            return re.findall(r'[a-z][a-z\']*', text.lower())
        
        # Common stopwords
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
            'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'about', 'up', 'also', 'get', 'got',
            'like', 'make', 'made', 'much', 'many', 'well', 'back', 'even',
            'still', 'way', 'take', 'come', 'go', 'see', 'know', 'say', 'said',
            'one', 'two', 'first', 'new', 'because', 'thing', 'things'
        }
        
        # Filler/generic phrases that indicate low-quality generic responses
        generic_phrases = [
            'keep trying', 'you\'ll get there', 'just remember', 'don\'t worry',
            'it\'s fine', 'no big deal', 'that\'s life', 'move on',
            'get yourself together', 'just keep', 'you should be able',
            'maybe you\'re just', 'it\'s noted that'
        ]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        query_content = [w for w in query_tokens if w not in stopwords and len(w) > 2]
        response_content = [w for w in response_tokens if w not in stopwords and len(w) > 2]
        
        if not query_content:
            query_content = query_tokens
        if not response_content:
            response_content = response_tokens
        
        # --- Component 1: TF-IDF Cosine Similarity ---
        # Build a pseudo-corpus from query and response for IDF
        all_tokens = set(query_content + response_content)
        
        # Compute term frequencies
        query_tf = Counter(query_content)
        response_tf = Counter(response_content)
        
        # Pseudo-IDF: terms appearing in both get lower weight, unique terms get higher
        query_set = set(query_content)
        response_set = set(response_content)
        
        def compute_idf(term):
            doc_count = 0
            if term in query_set:
                doc_count += 1
            if term in response_set:
                doc_count += 1
            return math.log(2.0 / (doc_count + 0.5) + 1.0)
        
        # Build TF-IDF vectors
        dot_product = 0.0
        query_norm = 0.0
        response_norm = 0.0
        
        for term in all_tokens:
            idf = compute_idf(term)
            q_tfidf = query_tf.get(term, 0) * idf
            r_tfidf = response_tf.get(term, 0) * idf
            dot_product += q_tfidf * r_tfidf
            query_norm += q_tfidf ** 2
            response_norm += r_tfidf ** 2
        
        if query_norm > 0 and response_norm > 0:
            cosine_sim = dot_product / (math.sqrt(query_norm) * math.sqrt(response_norm))
        else:
            cosine_sim = 0.0
        
        # --- Component 2: Query Keyword Coverage with importance weighting ---
        # Words that appear less frequently in general text are more important
        # Approximate importance by word length and uniqueness
        def word_importance(word):
            base = 1.0
            if len(word) > 6:
                base += 0.5
            if len(word) > 9:
                base += 0.5
            return base
        
        total_importance = 0.0
        covered_importance = 0.0
        
        for word in set(query_content):
            imp = word_importance(word)
            total_importance += imp
            if word in response_set:
                covered_importance += imp
        
        if total_importance > 0:
            keyword_coverage = covered_importance / total_importance
        else:
            keyword_coverage = 0.0
        
        # --- Component 3: N-gram overlap (bigrams and trigrams) ---
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        query_bigrams = set(get_ngrams(query_content, 2))
        response_bigrams = set(get_ngrams(response_content, 2))
        
        query_trigrams = set(get_ngrams(query_content, 3))
        response_trigrams = set(get_ngrams(response_content, 3))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        if query_trigrams:
            trigram_overlap = len(query_trigrams & response_trigrams) / len(query_trigrams)
        else:
            trigram_overlap = 0.0
        
        ngram_score = 0.6 * bigram_overlap + 0.4 * trigram_overlap
        
        # --- Component 4: Topic/Intent Alignment ---
        # Extract likely topic words from query (most distinctive words)
        # Check if response addresses the same topic space
        
        # Detect question type / intent from query
        query_lower = query.lower()
        
        intent_signals = {
            'how': ['steps', 'way', 'method', 'process', 'approach', 'guide', 'first', 'then', 'next', 'start', 'begin'],
            'why': ['because', 'reason', 'due', 'cause', 'result', 'therefore', 'since', 'explanation'],
            'what': ['definition', 'means', 'refers', 'concept', 'idea', 'basically', 'essentially'],
            'explain': ['imagine', 'think', 'consider', 'example', 'instance', 'means', 'basically', 'simply'],
            'help': ['suggest', 'recommend', 'try', 'consider', 'approach', 'strategy', 'tip'],
            'feeling': ['understand', 'sorry', 'hear', 'feel', 'empathy', 'support', 'comfort', 'okay', 'natural', 'valid'],
            'advice': ['suggest', 'recommend', 'try', 'consider', 'might', 'could', 'approach'],
        }
        
        intent_score = 0.0
        intent_matches = 0
        intent_total = 0
        
        response_lower = response.lower()
        
        for intent_key, signal_words in intent_signals.items():
            if intent_key in query_lower:
                intent_total += 1
                matched = sum(1 for w in signal_words if w in response_lower)
                if matched > 0:
                    intent_matches += 1
                    intent_score += min(matched / len(signal_words), 1.0)
        
        if intent_total > 0:
            intent_alignment = intent_score / intent_total
        else:
            # Fallback: check if response uses empathetic/helpful language when query seems emotional
            emotional_words = ['stress', 'frustrat', 'sad', 'lonely', 'heartbr', 'devastat', 'upset', 'worried', 'anxious', 'fear', 'exhaust', 'struggle']
            empathetic_words = ['understand', 'sorry', 'hear', 'feel', 'okay', 'natural', 'valid', 'completely', 'genuinely', 'acknowledge']
            
            is_emotional = any(w in query_lower for w in emotional_words)
            if is_emotional:
                emp_count = sum(1 for w in empathetic_words if w in response_lower)
                intent_alignment = min(emp_count / 3.0, 1.0)
            else:
                intent_alignment = 0.5  # neutral
        
        # --- Component 5: Dismissiveness / Generic Penalty ---
        response_lower_clean = response.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower_clean)
        generic_penalty = min(generic_count * 0.08, 0.3)
        
        # Dismissive tone detection
        dismissive_patterns = [
            r'\bjust\b.*\bjust\b', r'not a big deal', r'get over it',
            r'you should be able', r'maybe you\'re just not', r'it\'s noted',
            r'read the manual', r'that\'s a bummer'
        ]
        dismissive_count = sum(1 for p in dismissive_patterns if re.search(p, response_lower_clean))
        dismissive_penalty = min(dismissive_count * 0.12, 0.3)
        
        # --- Component 6: Response Substance and Depth ---
        # Longer, more detailed responses that stay on topic tend to be better
        response_sentences = re.split(r'[.!?]+', response)
        response_sentences = [s.strip() for s in response_sentences if len(s.strip()) > 10]
        num_sentences = len(response_sentences)
        
        # Check how many sentences contain query-relevant content
        relevant_sentences = 0
        for sent in response_sentences:
            sent_tokens = set(tokenize(sent)) - stopwords
            overlap = sent_tokens & query_set
            if len(overlap) >= 1 or any(w in sent.lower() for w in list(query_content)[:10]):
                relevant_sentences += 1
        
        if num_sentences > 0:
            sentence_relevance_ratio = relevant_sentences / num_sentences
        else:
            sentence_relevance_ratio = 0.0
        
        # Depth bonus: structured responses (numbered lists, clear paragraphs)
        has_structure = bool(re.search(r'\d+[.)]\s', response) or re.search(r'[-•]\s', response))
        structure_bonus = 0.08 if has_structure else 0.0
        
        # --- Component 7: Semantic field matching ---
        # Extract the most important query terms and check for synonyms/related terms in response
        # Use a simple approach: check if response discusses the same domain
        
        # Build word co-occurrence context from response
        response_word_set = set(response_content)
        query_word_set = set(query_content)
        
        # Direct overlap ratio (different from coverage - this is symmetric)
        intersection = response_word_set & query_word_set
        union = response_word_set | query_word_set
        if union:
            symmetric_overlap = len(intersection) / math.sqrt(len(query_word_set) * max(len(response_word_set), 1))
        else:
            symmetric_overlap = 0.0
        
        symmetric_overlap = min(symmetric_overlap, 1.0)
        
        # --- Combine all components ---
        # Weights chosen to emphasize different aspects of relevance
        score = (
            cosine_sim * 2.5 +           # TF-IDF cosine similarity (0-2.5)
            keyword_coverage * 2.0 +       # Query keyword coverage (0-2.0)
            ngram_score * 1.0 +            # N-gram overlap (0-1.0)
            intent_alignment * 1.5 +       # Intent fulfillment (0-1.5)
            sentence_relevance_ratio * 1.0 + # Sentence-level relevance (0-1.0)
            symmetric_overlap * 1.0 +      # Semantic field overlap (0-1.0)
            structure_bonus                # Structure bonus (0-0.08)
        )
        
        # Apply penalties
        score = score * (1.0 - generic_penalty) * (1.0 - dismissive_penalty)
        
        # Bonus for appropriate response length (not too short, not padding)
        word_count = len(response_tokens)
        if word_count < 20:
            score *= 0.7
        elif word_count < 40:
            score *= 0.85
        
        # Normalize to approximately 1-5 scale based on observed patterns
        # Raw score range is roughly 0-9
        # Map to 1-5
        normalized = 1.0 + (score / 9.0) * 4.0
        
        # Clamp
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        return 2.5  # Safe middle-ground fallback