def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a semantic field coverage approach.
    
    Algorithm: Query Decomposition & Semantic Field Coverage
    - Decomposes the query into semantic "fields" (question intent, topic keywords, entity references)
    - Measures how many distinct semantic fields from the query are addressed in the response
    - Uses BM25-inspired term weighting (not TF-IDF) for importance scoring
    - Analyzes question-type alignment (does the response match what's being asked?)
    - Measures information density and directness of address
    - Penalizes responses that are meta/procedural rather than substantive
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        # --- Preprocessing ---
        def tokenize(text):
            text = text.lower()
            # Keep alphanumeric and some meaningful punctuation
            tokens = re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text)
            return tokens
        
        def get_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 5]
        
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'am',
            'about', 'up', 'also', 'like', 'get', 'got', 'much', 'many',
            'really', 'think', 'know', 'see', 'make', 'way', 'thing', 'things',
            'been', 'going', 'went', 'come', 'came', 'even', 'still', 'well',
            'back', 'any', 'don', 'doesn', 'didn', 'won', 'isn', 'aren', 'wasn',
            'weren', 'hasn', 'haven', 'hadn', 'wouldn', 'couldn', 'shouldn',
        }
        
        META_PHRASES = [
            'welcome to', 'please read', 'our rules', 'before commenting',
            'your comments will be removed', 'while we do not require',
            'you might be interested', 'an earlier answer', 'this response by',
            'while you wait', 'removed if they', 'up to standard',
        ]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not response_tokens:
            return 0.0
        
        query_content = [t for t in query_tokens if t not in STOP_WORDS and len(t) > 1]
        response_content = [t for t in response_tokens if t not in STOP_WORDS and len(t) > 1]
        
        if not query_content:
            return 5.0  # Can't evaluate relevance without query content
        
        # --- 1. BM25-inspired term relevance ---
        # Instead of TF-IDF, use BM25 scoring with document = response, query = query terms
        k1 = 1.5
        b = 0.75
        avg_dl = 150  # assumed average document length
        dl = len(response_content)
        
        response_freq = Counter(response_content)
        query_freq = Counter(query_content)
        
        # Estimate IDF using query term frequency as a proxy for commonality
        # Rarer query terms (appearing once) get higher weight
        total_query_terms = len(query_content)
        
        bm25_score = 0.0
        matched_query_terms = set()
        
        for term, qf in query_freq.items():
            if term in response_freq:
                matched_query_terms.add(term)
                tf = response_freq[term]
                # IDF approximation: terms that are less common in query get higher weight
                idf = math.log(1 + (total_query_terms - qf + 0.5) / (qf + 0.5))
                idf = max(idf, 0.1)
                # BM25 term score
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
                bm25_score += idf * (numerator / denominator)
        
        # Normalize BM25 score
        max_possible_bm25 = len(query_freq) * 3.0  # rough upper bound
        bm25_normalized = min(bm25_score / max(max_possible_bm25, 1), 1.0)
        
        # --- 2. Query Semantic Field Coverage ---
        # Extract "semantic fields" from query: clusters of related content words
        # Use positional proximity to group words into fields
        query_content_positions = []
        for i, t in enumerate(query_tokens):
            if t not in STOP_WORDS and len(t) > 1:
                query_content_positions.append((i, t))
        
        # Create fields based on proximity (words within 3 positions of each other)
        fields = []
        current_field = set()
        last_pos = -10
        for pos, token in query_content_positions:
            if pos - last_pos <= 4:
                current_field.add(token)
            else:
                if current_field:
                    fields.append(current_field)
                current_field = {token}
            last_pos = pos
        if current_field:
            fields.append(current_field)
        
        # Score: what fraction of fields have at least one term covered?
        if fields:
            fields_covered = 0
            for field in fields:
                coverage = len(field & set(response_content)) / len(field)
                if coverage > 0:
                    fields_covered += coverage
            field_coverage_score = fields_covered / len(fields)
        else:
            field_coverage_score = 0.5
        
        # --- 3. Question Type Alignment ---
        # Detect what type of question is being asked and whether response matches
        query_lower = query.lower()
        
        question_types = {
            'how': ['step', 'process', 'method', 'way', 'approach', 'technique', 'by', 'through', 'using', 'first', 'then', 'next'],
            'why': ['because', 'reason', 'cause', 'due', 'since', 'result', 'therefore', 'thus', 'consequence', 'led', 'explain'],
            'what': ['definition', 'meaning', 'refers', 'called', 'known', 'type', 'kind', 'form', 'category', 'essentially'],
            'when': ['year', 'date', 'time', 'period', 'era', 'century', 'during', 'after', 'before', 'since'],
            'where': ['location', 'place', 'region', 'area', 'country', 'city', 'site', 'position'],
            'who': ['person', 'name', 'individual', 'people', 'group', 'organization', 'author', 'founder'],
            'is_there': ['yes', 'no', 'indeed', 'certainly', 'argument', 'case', 'position', 'perspective', 'view'],
            'imagine': ['scene', 'character', 'dialogue', 'story', 'narrative', 'setting', 'action', 'said', 'replied'],
        }
        
        detected_type = None
        for qtype, _ in question_types.items():
            if qtype == 'is_there' and ('is there' in query_lower or 'are there' in query_lower):
                detected_type = qtype
                break
            elif qtype == 'imagine' and 'imagine' in query_lower:
                detected_type = qtype
                break
            elif query_lower.lstrip().startswith(qtype) or f' {qtype} ' in query_lower[:80]:
                detected_type = qtype
                break
        
        type_alignment_score = 0.5  # default neutral
        if detected_type and detected_type in question_types:
            response_lower = response.lower()
            alignment_words = question_types[detected_type]
            matches = sum(1 for w in alignment_words if w in response_lower)
            type_alignment_score = min(matches / max(len(alignment_words) * 0.3, 1), 1.0)
        
        # --- 4. Substantiveness vs Meta-response Detection ---
        response_lower = response.lower()
        meta_penalty = 0.0
        for phrase in META_PHRASES:
            if phrase in response_lower:
                meta_penalty += 0.15
        meta_penalty = min(meta_penalty, 0.6)
        
        # Detect if response is mostly redirecting rather than answering
        redirect_patterns = [
            r'you might (want to|be interested)',
            r'check out',
            r'this (answer|response|thread) by',
            r'while you wait',
            r'i would (suggest|recommend) (looking|checking|reading)',
        ]
        redirect_count = sum(1 for p in redirect_patterns if re.search(p, response_lower))
        redirect_penalty = min(redirect_count * 0.1, 0.3)
        
        # --- 5. Response Depth & Engagement Score ---
        response_sentences = get_sentences(response)
        
        # Measure how many response sentences contain query terms
        sentences_with_query_terms = 0
        for sent in response_sentences:
            sent_tokens = set(tokenize(sent))
            if sent_tokens & set(query_content):
                sentences_with_query_terms += 1
        
        if response_sentences:
            engagement_ratio = sentences_with_query_terms / len(response_sentences)
        else:
            engagement_ratio = 0.0
        
        # --- 6. Unique Query Term Coverage (distinct terms covered) ---
        unique_query_terms = set(query_content)
        if unique_query_terms:
            term_coverage = len(matched_query_terms) / len(unique_query_terms)
        else:
            term_coverage = 0.5
        
        # --- 7. Response length adequacy ---
        # Very short responses are less likely to be comprehensive
        resp_len = len(response_tokens)
        if resp_len < 10:
            length_factor = 0.4
        elif resp_len < 25:
            length_factor = 0.7
        elif resp_len < 50:
            length_factor = 0.85
        elif resp_len < 300:
            length_factor = 1.0
        else:
            length_factor = 0.95  # slight penalty for very long (potential rambling)
        
        # --- 8. First-sentence relevance bonus ---
        # If the first sentence of the response is relevant, it's a good sign of direct address
        first_sent_bonus = 0.0
        if response_sentences:
            first_sent_tokens = set(tokenize(response_sentences[0])) - STOP_WORDS
            if first_sent_tokens and query_content:
                first_overlap = len(first_sent_tokens & set(query_content)) / max(len(first_sent_tokens), 1)
                first_sent_bonus = min(first_overlap * 2, 0.3)
        
        # --- 9. Bigram overlap (contextual relevance) ---
        def get_content_bigrams(tokens, stop_words):
            bigrams = []
            content_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
            for i in range(len(content_tokens) - 1):
                bigrams.append((content_tokens[i], content_tokens[i+1]))
            return bigrams
        
        query_bigrams = set(get_content_bigrams(query_tokens, STOP_WORDS))
        response_bigrams = set(get_content_bigrams(response_tokens, STOP_WORDS))
        
        if query_bigrams:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        # --- 10. Specificity score ---
        # Responses with specific details (numbers, proper nouns, technical terms) score higher
        specificity_indicators = 0
        # Numbers
        specificity_indicators += len(re.findall(r'\b\d+\b', response)) * 0.5
        # Capitalized words (potential proper nouns, excluding sentence starts)
        caps = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]{2,}', response)
        specificity_indicators += len(caps) * 0.3
        # Long/technical words (>8 chars)
        long_words = [t for t in response_content if len(t) > 8]
        specificity_indicators += len(long_words) * 0.2
        
        specificity_score = min(specificity_indicators / max(len(response_sentences), 1) / 3, 1.0)
        
        # --- Combine scores with weights ---
        # BM25 relevance: 20%
        # Field coverage: 20%
        # Question type alignment: 10%
        # Term coverage: 15%
        # Engagement ratio: 10%
        # Bigram overlap: 10%
        # Specificity: 5%
        # First sentence bonus: added
        # Length factor: multiplied
        # Penalties: subtracted
        
        raw_score = (
            bm25_normalized * 0.20 +
            field_coverage_score * 0.20 +
            type_alignment_score * 0.10 +
            term_coverage * 0.15 +
            engagement_ratio * 0.10 +
            bigram_overlap * 0.10 +
            specificity_score * 0.05 +
            first_sent_bonus
        )
        
        # Apply length factor
        raw_score *= length_factor
        
        # Apply penalties
        raw_score -= meta_penalty
        raw_score -= redirect_penalty
        
        # --- 11. Direct address bonus ---
        # If response directly references key query concepts in a structured way
        # (e.g., answering "what does X mean to you" with personal experience)
        personal_markers = ['i ', 'my ', "i'm ", "i've ", 'we ', 'our ']
        query_asks_personal = any(p in query_lower for p in ['your ', 'you ', 'your career', 'your experience', 'to you'])
        response_is_personal = any(response_lower.startswith(p) or f' {p}' in response_lower[:100] for p in personal_markers)
        
        if query_asks_personal and response_is_personal:
            raw_score += 0.08
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, raw_score * 10))
        
        return round(final_score, 3)
        
    except Exception:
        return 5.0