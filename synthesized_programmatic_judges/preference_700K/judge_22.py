def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a 
    question-decomposition and topical coverage approach.
    
    Strategy: Decompose the query into sub-topics/aspects (using keyword extraction
    and question word analysis), then measure how many of those aspects are addressed
    in the response. Also evaluate structural depth, explanation density, and 
    information richness through unique concept counting and clause analysis.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.5
        
        # ---- Helper functions ----
        
        def extract_content_words(text):
            """Extract meaningful content words (not stopwords)."""
            stopwords = {
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
                'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
                'myself', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
                'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs',
                'what', 'which', 'who', 'whom', 'whose', 'am', 'about', 'up', 'down',
                'also', 'like', 'get', 'got', 'much', 'many', 'really', 'well',
                'even', 'back', 'still', 'way', 'take', 'come', 'make', 'know',
                'think', 'see', 'look', 'want', 'give', 'use', 'find', 'tell',
                'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'going',
                'been', 'being', 'dont', 'im', 'ive', 'youre', 'thats', 'its',
                'one', 'two', 'first', 'new', 'now', 'people', 'thing', 'things',
                'something', 'anything', 'nothing', 'everything', 'someone',
            }
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        def get_ngrams(words, n):
            """Get n-grams from word list."""
            return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        def count_clauses(text):
            """Estimate number of independent clauses/ideas."""
            # Split on clause boundaries
            clause_markers = r'[.!?;]|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\bwhereas\b|\bmoreover\b|\bfurthermore\b|\badditionally\b|\bnevertheless\b|\bconsequently\b|\btherefore\b|\bhence\b'
            parts = re.split(clause_markers, text)
            return len([p for p in parts if len(p.strip()) > 15])
        
        # ---- 1. Query Decomposition: Extract query aspects/topics ----
        
        query_content_words = extract_content_words(query)
        query_bigrams = get_ngrams(query_content_words, 2)
        
        # Identify question types in the query
        question_patterns = re.findall(
            r'\b(what|how|why|when|where|who|which|can|does|do|is|are|should|would|could)\b',
            query.lower()
        )
        num_questions = len(re.findall(r'\?', query))
        # Also count implicit questions (sentences starting with question words)
        implicit_q = len(re.findall(r'(?:^|\.\s+)(what|how|why|when|where|who|which)', query.lower()))
        total_question_aspects = max(num_questions + implicit_q, 1)
        
        # Extract key query topics (unique content words, weighted by frequency)
        query_word_freq = Counter(query_content_words)
        # Top query terms (the most important aspects to cover)
        query_key_terms = set()
        for word, count in query_word_freq.most_common(30):
            query_key_terms.add(word)
        
        # Also extract quoted terms, capitalized terms, and technical terms from query
        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        capitalized = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
        special_terms = set()
        for q in quoted:
            special_terms.update(q.lower().split())
        for c in capitalized:
            special_terms.add(c.lower())
        
        # ---- 2. Response Analysis ----
        
        response_lower = response.lower()
        response_content_words = extract_content_words(response)
        response_word_set = set(response_content_words)
        response_bigrams = set(get_ngrams(response_content_words, 2))
        
        # ---- 3. Topic Coverage Score ----
        # What fraction of query key terms appear in the response?
        if query_key_terms:
            covered = sum(1 for t in query_key_terms if t in response_word_set)
            topic_coverage = covered / len(query_key_terms)
        else:
            topic_coverage = 0.5
        
        # Bigram overlap (captures phrase-level coverage)
        if query_bigrams:
            query_bigram_set = set(query_bigrams)
            bigram_covered = len(query_bigram_set & response_bigrams)
            bigram_coverage = bigram_covered / len(query_bigram_set)
        else:
            bigram_coverage = 0.5
        
        # Special terms coverage
        if special_terms:
            special_covered = sum(1 for t in special_terms if t in response_lower)
            special_coverage = special_covered / len(special_terms)
        else:
            special_coverage = 0.5
        
        # ---- 4. Information Density & Richness ----
        
        # Unique concepts: ratio of unique content words to total words
        total_words = len(re.findall(r'\S+', response))
        unique_content = len(set(response_content_words))
        
        if total_words > 0:
            vocab_richness = unique_content / (total_words ** 0.5)  # type-token with sqrt normalization
        else:
            vocab_richness = 0
        
        # Clause count as a proxy for number of distinct ideas
        clause_count = count_clauses(response)
        
        # ---- 5. Depth Indicators ----
        
        # Causal/explanatory language (indicates depth of explanation)
        explanation_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b', r'\bleads to\b',
            r'\bcauses?\b', r'\beffects?\b', r'\bimplications?\b', r'\bmeans that\b',
            r'\bin other words\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bsuch as\b', r'\bincluding\b', r'\be\.g\.\b', r'\bi\.e\.\b',
        ]
        explanation_count = sum(
            len(re.findall(pat, response_lower)) for pat in explanation_markers
        )
        explanation_density = min(explanation_count / max(total_words / 50, 1), 3.0)
        
        # Qualification/nuance markers (indicates thoroughness)
        nuance_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bon the other hand\b',
            r'\bnevertheless\b', r'\bwhile\b', r'\bdepends\b', r'\bnot necessarily\b',
            r'\bin some cases\b', r'\btypically\b', r'\bgenerally\b', r'\busually\b',
            r'\bsometimes\b', r'\boften\b', r'\brarely\b', r'\bexception\b',
            r'\bnuance\b', r'\bcaveat\b', r'\btrade-?off\b', r'\balternative\b',
        ]
        nuance_count = sum(
            len(re.findall(pat, response_lower)) for pat in nuance_markers
        )
        nuance_score = min(nuance_count / max(total_words / 80, 1), 2.5)
        
        # Evidence/reference markers
        evidence_markers = [
            r'\baccording to\b', r'\bresearch\b', r'\bstudy\b', r'\bstudies\b',
            r'\bevidence\b', r'\bdata\b', r'\bstatistic\b', r'\bpercent\b',
            r'\bsource\b', r'\bcited?\b', r'\breference\b', r'\bexpert\b',
            r'\b\d{4}\b',  # years
            r'\bfigure\b', r'\btable\b', r'\bsurvey\b',
        ]
        evidence_count = sum(
            len(re.findall(pat, response_lower)) for pat in evidence_markers
        )
        evidence_score = min(evidence_count * 0.3, 2.0)
        
        # ---- 6. Structural Completeness ----
        
        # Multiple perspectives or aspects addressed
        # Count distinct semantic segments (paragraphs or major sections)
        paragraphs = [p.strip() for p in response.split('\n') if len(p.strip()) > 30]
        num_paragraphs = max(len(paragraphs), 1)
        
        # Enumeration/listing (suggests covering multiple points)
        enum_patterns = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*|-|•)', response)
        has_enumeration = len(enum_patterns)
        
        # ---- 7. Response Length Adequacy ----
        # Longer responses tend to be more complete, but with diminishing returns
        # Use log scale relative to query complexity
        query_complexity = len(query_content_words) + num_questions * 5
        
        length_score = math.log(max(total_words, 1) + 1) / math.log(max(query_complexity, 5) + 1)
        length_score = min(length_score, 3.0)  # cap
        
        # Penalize very short responses relative to query complexity
        if total_words < 20:
            length_penalty = 0.3
        elif total_words < 50:
            length_penalty = 0.6
        elif total_words < 80:
            length_penalty = 0.8
        else:
            length_penalty = 1.0
        
        # ---- 8. Novel Information Score ----
        # How much new information does the response add beyond just echoing the query?
        query_word_set = set(query_content_words)
        novel_words = response_word_set - query_word_set
        if response_word_set:
            novelty_ratio = len(novel_words) / len(response_word_set)
        else:
            novelty_ratio = 0
        
        # ---- 9. Directness/Relevance Check ----
        # Does the response actually attempt to answer (vs deflect)?
        deflection_patterns = [
            r'\bi don\'?t know\b', r'\bi\'?m not sure\b', r'\bi can\'?t\b',
            r'\byou should ask\b', r'\bconsult a\b', r'\bgoogle\b',
            r'\bcheck out\b.*\bfor more\b', r'\boutside my\b',
        ]
        deflection_count = sum(
            len(re.findall(pat, response_lower)) for pat in deflection_patterns
        )
        deflection_penalty = max(0, 1.0 - deflection_count * 0.15)
        
        # ---- 10. Composite Scoring ----
        
        # Weight the components
        score = 0.0
        
        # Topic coverage (0-1) -> weighted heavily (max ~25 points)
        score += topic_coverage * 25.0
        
        # Bigram coverage (0-1) -> (max ~10 points)
        score += bigram_coverage * 10.0
        
        # Special terms coverage (max ~5 points)
        score += special_coverage * 5.0
        
        # Explanation density (0-3) -> (max ~10 points)
        score += explanation_density * 3.3
        
        # Nuance (0-2.5) -> (max ~8 points)
        score += nuance_score * 3.2
        
        # Evidence (0-2) -> (max ~6 points)
        score += evidence_score * 3.0
        
        # Clause richness (log scale, max ~10 points)
        score += min(math.log(clause_count + 1) * 3.5, 10.0)
        
        # Vocabulary richness (typically 1-5) -> (max ~8 points)
        score += min(vocab_richness * 2.0, 8.0)
        
        # Length adequacy (0-3) -> (max ~8 points)
        score += length_score * 2.7
        
        # Enumeration bonus (max ~5 points)
        score += min(has_enumeration * 1.0, 5.0)
        
        # Paragraph structure (max ~5 points)
        score += min(math.log(num_paragraphs + 1) * 2.5, 5.0)
        
        # Novelty (0-1) -> (max ~5 points)
        score += novelty_ratio * 5.0
        
        # Apply penalties
        score *= length_penalty
        score *= deflection_penalty
        
        # Normalize to 0-10 range
        # Theoretical max is around 100+, but practical max is ~70-80
        score = score / 10.0
        
        # Clamp
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            return min(len(str(response).split()) / 30.0, 5.0)
        except Exception:
            return 2.5