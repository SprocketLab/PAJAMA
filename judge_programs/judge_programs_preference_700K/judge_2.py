def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using semantic vector similarity
    based on character n-gram and word embedding approximation via co-occurrence hashing.
    
    This approach uses:
    1. Query intent extraction (question words, key phrases, named entities)
    2. Semantic field matching via distributional word clustering
    3. Response structure analysis (does it answer vs deflect)
    4. Topic coherence via sentence-level relevance scoring
    """
    try:
        import re
        import math
        from collections import Counter, defaultdict
        
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        if len(response.strip()) < 5:
            return 0.0
        
        # --- Preprocessing ---
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s\'-]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def get_words(text):
            return clean_text(text).split()
        
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
            'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
            'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'am', 'about', 'up', 'also', 'much',
            'like', 'even', 'still', 'get', 'got', 'going', 'go', 'really',
            'thing', 'things', 'one', 'two', 'know', 'think', 'make', 'see',
        }
        
        def content_words(text):
            words = get_words(text)
            return [w for w in words if w not in STOPWORDS and len(w) > 2]
        
        def get_char_ngrams(text, n=3):
            """Get character n-grams for fuzzy matching."""
            text = clean_text(text)
            ngrams = Counter()
            for word in text.split():
                padded = f"#{word}#"
                for i in range(len(padded) - n + 1):
                    ngrams[padded[i:i+n]] += 1
            return ngrams
        
        def cosine_sim_counters(c1, c2):
            """Cosine similarity between two Counter objects."""
            if not c1 or not c2:
                return 0.0
            common = set(c1.keys()) & set(c2.keys())
            dot = sum(c1[k] * c2[k] for k in common)
            mag1 = math.sqrt(sum(v*v for v in c1.values()))
            mag2 = math.sqrt(sum(v*v for v in c2.values()))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)
        
        # --- Feature 1: Character n-gram cosine similarity (fuzzy semantic overlap) ---
        q_trigrams = get_char_ngrams(query, 3)
        r_trigrams = get_char_ngrams(response, 3)
        q_fourgrams = get_char_ngrams(query, 4)
        r_fourgrams = get_char_ngrams(response, 4)
        
        char3_sim = cosine_sim_counters(q_trigrams, r_trigrams)
        char4_sim = cosine_sim_counters(q_fourgrams, r_fourgrams)
        char_sim = 0.4 * char3_sim + 0.6 * char4_sim
        
        # --- Feature 2: Query intent coverage ---
        # Extract the core question/intent from the query
        q_content = content_words(query)
        r_content = content_words(response)
        
        if not q_content:
            q_content = get_words(query)
        if not r_content:
            r_content = get_words(response)
        
        q_content_set = set(q_content)
        r_content_set = set(r_content)
        
        # Word stem approximation (simple suffix stripping)
        def pseudo_stem(word):
            word = word.lower()
            for suffix in ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful', 
                          'less', 'ous', 'ive', 'ing', 'ated', 'ting', 'ed', 'ly', 
                          'er', 'est', 'al', 'ity', 'ies', 'es', 's']:
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    return word[:-len(suffix)]
            return word
        
        q_stems = set(pseudo_stem(w) for w in q_content)
        r_stems = set(pseudo_stem(w) for w in r_content)
        
        if q_stems:
            stem_coverage = len(q_stems & r_stems) / len(q_stems)
        else:
            stem_coverage = 0.0
        
        # --- Feature 3: Semantic field expansion via word co-occurrence hashing ---
        # Build "semantic neighborhoods" by hashing words to buckets based on
        # their character composition - words with similar structure often relate
        def word_hash_vector(word, dim=64):
            """Create a pseudo-embedding based on character features."""
            vec = [0.0] * dim
            word = word.lower()
            for i, ch in enumerate(word):
                idx = (ord(ch) * 7 + i * 13) % dim
                vec[idx] += 1.0
                # bigram feature
                if i < len(word) - 1:
                    bigram_idx = (ord(ch) * 31 + ord(word[i+1]) * 17) % dim
                    vec[bigram_idx] += 0.5
            # normalize
            mag = math.sqrt(sum(v*v for v in vec))
            if mag > 0:
                vec = [v/mag for v in vec]
            return vec
        
        def avg_vector(words, dim=64):
            if not words:
                return [0.0] * dim
            vecs = [word_hash_vector(w, dim) for w in words]
            avg = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    avg[i] += v[i]
            n = len(vecs)
            return [x/n for x in avg]
        
        def vec_cosine(v1, v2):
            dot = sum(a*b for a, b in zip(v1, v2))
            m1 = math.sqrt(sum(a*a for a in v1))
            m2 = math.sqrt(sum(b*b for b in v2))
            if m1 == 0 or m2 == 0:
                return 0.0
            return dot / (m1 * m2)
        
        q_vec = avg_vector(q_content)
        r_vec = avg_vector(r_content)
        embedding_sim = vec_cosine(q_vec, r_vec)
        
        # --- Feature 4: Sentence-level relevance (each response sentence vs query) ---
        def split_sentences(text):
            sents = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 10]
        
        r_sentences = split_sentences(response)
        if r_sentences:
            sent_scores = []
            q_vec_full = avg_vector(q_content)
            for sent in r_sentences:
                s_content = content_words(sent)
                if s_content:
                    s_vec = avg_vector(s_content)
                    sim = vec_cosine(q_vec_full, s_vec)
                    # Also check direct word overlap
                    s_set = set(pseudo_stem(w) for w in s_content)
                    overlap = len(q_stems & s_set) / max(len(q_stems), 1)
                    sent_scores.append(0.5 * sim + 0.5 * overlap)
                else:
                    sent_scores.append(0.0)
            
            # Proportion of sentences that are relevant (score > threshold)
            relevant_sents = sum(1 for s in sent_scores if s > 0.15)
            sent_relevance_ratio = relevant_sents / len(sent_scores)
            avg_sent_relevance = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
            max_sent_relevance = max(sent_scores) if sent_scores else 0.0
        else:
            sent_relevance_ratio = 0.0
            avg_sent_relevance = 0.0
            max_sent_relevance = 0.0
        
        # --- Feature 5: Response directness / engagement signals ---
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Detect deflection patterns (lower score)
        deflection_patterns = [
            r'\bwelcome to\b', r'\bplease read\b', r'\brules before\b',
            r'\byour (post|comment) (has|was)\b', r'\bthis is (a )?bot\b',
            r'\bremoved\b.*\brule', r'\bmoderat', r'\bautomod',
        ]
        deflection_score = 0
        for pat in deflection_patterns:
            if re.search(pat, response_lower):
                deflection_score += 1
        
        # Detect engagement patterns (higher score)
        engagement_patterns = [
            r'\bbecause\b', r'\bfor example\b', r'\bspecifically\b',
            r'\bin (this|that) case\b', r'\bthe reason\b', r'\bto answer\b',
            r'\bessentially\b', r'\bin short\b', r'\bthe key\b',
        ]
        engagement_score = 0
        for pat in engagement_patterns:
            if re.search(pat, response_lower):
                engagement_score += 1
        
        # --- Feature 6: Query type detection and response format matching ---
        is_question = bool(re.search(r'\?', query))
        is_how = bool(re.search(r'\bhow\b', query_lower))
        is_why = bool(re.search(r'\bwhy\b', query_lower))
        is_what = bool(re.search(r'\bwhat\b', query_lower))
        is_creative = bool(re.search(r'\b(imagine|write|create|generate|story|poem|dialogue)\b', query_lower))
        is_technical = bool(re.search(r'\b(code|sql|function|table|create|select|api|program)\b', query_lower))
        
        # Check if response attempts to answer the type of question
        format_bonus = 0.0
        if is_creative:
            # Creative tasks: look for narrative elements
            if re.search(r'["\*]', response) or re.search(r'\b(said|replied|asked|whispered|shouted)\b', response_lower):
                format_bonus += 0.15
        if is_technical:
            # Technical: look for code-like content
            if re.search(r'```|SELECT|FROM|def |class |function|return', response):
                format_bonus += 0.1
        if is_how or is_why:
            # Explanatory: look for causal/explanatory language
            if re.search(r'\bbecause\b|\bdue to\b|\bthe reason\b|\bthis (means|is because)\b', response_lower):
                format_bonus += 0.1
        
        # --- Feature 7: Response substantiveness ---
        r_words = get_words(response)
        q_words = get_words(query)
        
        # Length ratio - responses should generally be substantial
        resp_len = len(r_words)
        if resp_len < 10:
            length_score = 0.2
        elif resp_len < 30:
            length_score = 0.5
        elif resp_len < 80:
            length_score = 0.8
        elif resp_len < 200:
            length_score = 1.0
        else:
            length_score = 0.95  # Very long might include fluff
        
        # --- Feature 8: Key entity/concept matching ---
        # Extract likely important terms (capitalized words, quoted terms, technical terms)
        def extract_key_terms(text):
            terms = set()
            # Capitalized words (potential entities)
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for c in caps:
                terms.add(c.lower())
            # Quoted terms
            quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
            for q in quoted:
                terms.add(q.lower())
            # Technical terms (words with special chars)
            tech = re.findall(r'\b\w+[_\.]\w+\b', text)
            for t in tech:
                terms.add(t.lower())
            return terms
        
        q_key_terms = extract_key_terms(query)
        r_key_terms = extract_key_terms(response)
        r_text_lower = response.lower()
        
        if q_key_terms:
            key_term_hits = sum(1 for t in q_key_terms if t in r_text_lower)
            key_term_coverage = key_term_hits / len(q_key_terms)
        else:
            key_term_coverage = stem_coverage  # fall back
        
        # --- Feature 9: Topical word frequency correlation ---
        # Build word frequency profiles and compare distributions
        def freq_profile(words, top_n=30):
            c = Counter(words)
            total = sum(c.values())
            if total == 0:
                return {}
            return {w: count/total for w, count in c.most_common(top_n)}
        
        q_profile = freq_profile(q_content, 20)
        r_profile = freq_profile(r_content, 40)
        
        # Check how many of query's top words appear in response's profile
        if q_profile:
            profile_overlap = sum(
                min(q_profile.get(w, 0), r_profile.get(w, 0)) 
                for w in set(q_profile) | set(r_profile)
            )
        else:
            profile_overlap = 0.0
        
        # --- Combine all features ---
        score = (
            char_sim * 12.0 +                    # Character n-gram similarity (0-12)
            stem_coverage * 15.0 +                # Stem-based query coverage (0-15)
            embedding_sim * 10.0 +                # Pseudo-embedding similarity (0-10)
            avg_sent_relevance * 12.0 +           # Average sentence relevance (0-12)
            max_sent_relevance * 5.0 +            # Best sentence relevance (0-5)
            sent_relevance_ratio * 8.0 +          # Ratio of relevant sentences (0-8)
            key_term_coverage * 15.0 +            # Key entity coverage (0-15)
            profile_overlap * 10.0 +              # Frequency profile overlap (0-10)
            engagement_score * 2.0 +              # Engagement signals (0-6)
            format_bonus * 10.0 +                 # Format matching bonus (0-1.5)
            length_score * 5.0 +                  # Length adequacy (0-5)
            - deflection_score * 8.0              # Deflection penalty
        )
        
        # Normalize to 0-100 range
        # Theoretical max is around 100, but typical good answers score 40-70
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
        
    except Exception as e:
        # Fallback: simple word overlap ratio
        try:
            q_words = set(str(query).lower().split())
            r_words = set(str(response).lower().split())
            if q_words:
                return round(len(q_words & r_words) / len(q_words) * 50, 3)
            return 25.0
        except:
            return 25.0