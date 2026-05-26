def judging_function(query, response):
    """
    Evaluate clarity and conciseness using a structural coherence approach:
    - Measures signal-to-noise ratio via functional vs filler word density
    - Detects broken/corrupted text patterns
    - Evaluates response completeness relative to query
    - Checks for structural anomalies (repeated blocks, off-topic tangents, HTML/code artifacts)
    - Uses compression ratio as a proxy for information density
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            query = ""
        
        resp_clean = response.strip()
        query_clean = query.strip()
        
        # === Feature 1: Response length adequacy ===
        resp_len = len(resp_clean)
        word_count = len(resp_clean.split())
        
        if word_count <= 1:
            return 0.5
        
        # Very short responses are usually bad unless query asks for something simple
        query_words = query_clean.lower().split()
        is_simple_query = any(w in query_clean.lower() for w in ['identify', 'which', 'biggest', 'name'])
        
        length_score = 1.0
        if word_count < 3 and not is_simple_query:
            length_score = 0.15
        elif word_count < 5 and not is_simple_query:
            length_score = 0.4
        elif word_count >= 5:
            length_score = min(1.0, 0.5 + word_count / 30.0)
        
        # === Feature 2: Compression ratio (repetition detector) ===
        # Simulate compression by counting unique character trigrams vs total
        def compression_ratio(text):
            if len(text) < 4:
                return 1.0
            trigrams = [text[i:i+3] for i in range(len(text) - 2)]
            unique = len(set(trigrams))
            total = len(trigrams)
            return unique / total if total > 0 else 1.0
        
        comp_ratio = compression_ratio(resp_clean)
        # Low compression ratio = lots of repetition = bad
        # Typical good text: 0.4-0.8, repetitive: < 0.3
        repetition_score = min(1.0, comp_ratio / 0.5) if comp_ratio < 0.5 else 1.0
        
        # === Feature 3: Duplicate sentence/phrase detection ===
        sentences = re.split(r'[.!?\n]+', resp_clean)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        duplicate_ratio = 0.0
        if len(sentences) > 1:
            seen = set()
            duplicates = 0
            for s in sentences:
                # Normalize whitespace
                normalized = re.sub(r'\s+', ' ', s)
                if normalized in seen:
                    duplicates += 1
                seen.add(normalized)
            duplicate_ratio = duplicates / len(sentences)
        
        dedup_score = 1.0 - duplicate_ratio
        
        # === Feature 4: Structural artifact detection ===
        artifact_patterns = [
            r'<[a-zA-Z/][^>]*>',  # HTML tags
            r'```',                 # code blocks
            r'import\s+\w+',       # code imports
            r'def\s+\w+\s*\(',    # function definitions
            r'Input:\s*$',         # empty input/output patterns
            r'Output:.*Output:',   # repeated output markers
            r'Question:.*Answer:.*Question:', # Q&A chains
        ]
        
        artifact_count = 0
        for pattern in artifact_patterns:
            matches = re.findall(pattern, resp_clean, re.MULTILINE)
            artifact_count += len(matches)
        
        # Check if query asked for HTML/code
        query_asks_code = any(w in query_clean.lower() for w in ['html', 'code', 'python', 'tag', 'program'])
        
        if query_asks_code:
            artifact_penalty = 0.0
        else:
            artifact_penalty = min(0.6, artifact_count * 0.1)
        
        artifact_score = 1.0 - artifact_penalty
        
        # === Feature 5: Functional word ratio (signal density) ===
        # Content words vs function/filler words
        filler_words = {
            'very', 'really', 'quite', 'rather', 'somewhat', 'basically',
            'actually', 'literally', 'honestly', 'just', 'simply',
            'obviously', 'clearly', 'of course', 'needless to say',
            'it is worth noting', 'it should be noted', 'as a matter of fact',
            'in terms of', 'with regard to', 'in order to', 'due to the fact',
            'at the end of the day', 'all things considered',
            'essentially', 'fundamentally', 'importantly'
        }
        
        resp_lower = resp_clean.lower()
        words_lower = re.findall(r'\b[a-z]+\b', resp_lower)
        
        if len(words_lower) == 0:
            return 1.0
        
        filler_count = sum(1 for w in words_lower if w in filler_words)
        # Also check multi-word fillers
        for phrase in ['of course', 'needless to say', 'it is worth noting',
                       'it should be noted', 'as a matter of fact', 'in terms of',
                       'with regard to', 'in order to', 'due to the fact',
                       'at the end of the day', 'all things considered']:
            filler_count += resp_lower.count(phrase)
        
        filler_ratio = filler_count / len(words_lower)
        filler_score = max(0.3, 1.0 - filler_ratio * 3)
        
        # === Feature 6: Topical relevance via keyword coverage ===
        # Extract meaningful query words (not stopwords)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'between',
            'through', 'after', 'before', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'make', 'please', 'want', 'know',
            'tell', 'give', 'get', 'find', 'many', 'much'
        }
        
        query_content_words = [w for w in re.findall(r'\b[a-z]+\b', query_clean.lower()) 
                               if w not in stopwords and len(w) > 2]
        
        if query_content_words:
            resp_word_set = set(words_lower)
            matched = sum(1 for w in query_content_words if w in resp_word_set)
            relevance_score = matched / len(query_content_words) if query_content_words else 0.5
            relevance_score = min(1.0, relevance_score + 0.3)  # baseline boost
        else:
            relevance_score = 0.7
        
        # === Feature 7: Sentence structure quality ===
        # Check for well-formed sentences (start with capital, end with punctuation)
        raw_sentences = re.split(r'\n+', resp_clean)
        raw_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 0]
        
        well_formed = 0
        total_checked = 0
        for sent in raw_sentences[:10]:  # Check first 10 lines
            total_checked += 1
            starts_cap = sent[0].isupper() if sent else False
            # Ends with punctuation or is reasonably structured
            ends_punct = sent[-1] in '.!?:;)' if sent else False
            if starts_cap:
                well_formed += 0.5
            if ends_punct:
                well_formed += 0.5
        
        structure_score = well_formed / total_checked if total_checked > 0 else 0.5
        structure_score = min(1.0, structure_score + 0.2)  # slight boost
        
        # === Feature 8: Broken text detection ===
        # Look for signs of truncation, garbled text, random characters
        broken_indicators = 0
        
        # Repeated single characters or spaces
        if re.search(r'(.)\1{4,}', resp_clean):
            broken_indicators += 1
        
        # Lines that are just whitespace or single chars
        lines = resp_clean.split('\n')
        empty_lines = sum(1 for l in lines if len(l.strip()) <= 1)
        if len(lines) > 2 and empty_lines / len(lines) > 0.5:
            broken_indicators += 1
        
        # Starts with lowercase (unusual for proper response)
        if resp_clean[0].islower() and not resp_clean.startswith('http'):
            broken_indicators += 0.5
        
        # Contains lots of '#' or special formatting
        special_char_ratio = sum(1 for c in resp_clean if c in '#*_~`|') / max(1, len(resp_clean))
        if special_char_ratio > 0.05 and not query_asks_code:
            broken_indicators += 1
        
        broken_score = max(0.2, 1.0 - broken_indicators * 0.2)
        
        # === Feature 9: Unique information density ===
        # Ratio of unique content-bearing words to total words
        content_words = [w for w in words_lower if w not in stopwords and len(w) > 2]
        if content_words:
            unique_content = len(set(content_words))
            total_content = len(content_words)
            # Higher ratio = more diverse vocabulary = less repetitive
            info_density = unique_content / total_content
            # Also consider absolute count of unique content words
            info_richness = min(1.0, unique_content / 15.0)
            info_score = 0.5 * info_density + 0.5 * info_richness
        else:
            info_score = 0.2
        
        # === Feature 10: Response coherence - check if response addresses query type ===
        coherence_score = 0.7  # default
        
        # If query is a question, response should contain declarative content
        is_question = '?' in query_clean
        if is_question:
            # Response should not just be another question
            resp_questions = resp_clean.count('?')
            resp_statements = len(re.findall(r'[.!]', resp_clean))
            if resp_questions > resp_statements and resp_questions > 1:
                coherence_score = 0.4
            elif word_count >= 5:
                coherence_score = 0.85
        
        # If query asks to rewrite/create, response should contain the output
        if any(w in query_clean.lower() for w in ['rewrite', 'create', 'write', 'generate']):
            if word_count >= 3:
                coherence_score = 0.85
        
        # === Combine all features with weights ===
        weights = {
            'length': 0.12,
            'repetition': 0.15,
            'dedup': 0.12,
            'artifact': 0.10,
            'filler': 0.08,
            'relevance': 0.12,
            'structure': 0.08,
            'broken': 0.08,
            'info': 0.08,
            'coherence': 0.07,
        }
        
        scores = {
            'length': length_score,
            'repetition': repetition_score,
            'dedup': dedup_score,
            'artifact': artifact_score,
            'filler': filler_score,
            'relevance': relevance_score,
            'structure': structure_score,
            'broken': broken_score,
            'info': info_score,
            'coherence': coherence_score,
        }
        
        weighted_sum = sum(weights[k] * scores[k] for k in weights)
        total_weight = sum(weights.values())
        
        raw_score = weighted_sum / total_weight
        
        # Scale to 0-10
        final_score = raw_score * 10.0
        
        # Apply floor/ceiling
        final_score = max(0.5, min(10.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception:
        # Fallback: basic length-based score
        try:
            resp_len = len(response.strip()) if response else 0
            if resp_len == 0:
                return 0.0
            elif resp_len < 5:
                return 1.0
            elif resp_len < 20:
                return 3.0
            else:
                return 5.0
        except Exception:
            return 3.0