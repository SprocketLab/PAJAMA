def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a 
    question-awareness and structural depth analysis approach.
    
    This variant focuses on:
    1. Query decomposition - identifying sub-questions/aspects in the query
    2. Response addressing ratio - how many query aspects are addressed
    3. Explanation depth via clause density and elaboration patterns
    4. Example/evidence density
    5. Perspective diversity (multiple viewpoints/angles)
    6. Truncation/incompleteness detection
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        import math
        from collections import Counter
        
        response_text = response.strip()
        query_text = query.strip()
        
        if len(response_text) < 10:
            return 0.5
        
        # =====================================================
        # 1. QUERY DECOMPOSITION & ASPECT COVERAGE (0-25 pts)
        # =====================================================
        # Extract question words and key phrases from query
        query_lower = query_text.lower()
        response_lower = response_text.lower()
        
        # Find explicit questions in query (sentences ending with ?)
        query_sentences = re.split(r'[.!?\n]+', query_text)
        question_sentences = [s.strip() for s in query_sentences if '?' in s and len(s.strip()) > 5]
        num_questions = max(len(question_sentences), 1)
        
        # Extract key topic words from query (nouns, verbs - approximated by longer words)
        query_words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        # Remove very common words
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will',
            'would', 'could', 'should', 'about', 'which', 'their', 'there',
            'they', 'them', 'then', 'than', 'what', 'when', 'where', 'does',
            'some', 'into', 'also', 'just', 'more', 'most', 'very', 'much',
            'many', 'such', 'like', 'other', 'only', 'your', 'know', 'make',
            'being', 'seem', 'seem', 'really', 'think', 'want', 'here',
            'these', 'those', 'each', 'every', 'both', 'even', 'after',
            'before', 'because', 'between', 'through', 'during', 'without',
            'again', 'further', 'once', 'doing', 'having', 'asking',
        }
        query_content_words = [w for w in query_words if w not in stopwords]
        query_content_set = set(query_content_words)
        
        # Check how many query content words appear in response
        if query_content_set:
            addressed_words = sum(1 for w in query_content_set if w in response_lower)
            query_coverage_ratio = addressed_words / len(query_content_set)
        else:
            query_coverage_ratio = 0.5
        
        # Extract key bigrams from query for more precise topic matching
        query_tokens = query_lower.split()
        query_bigrams = set()
        for i in range(len(query_tokens) - 1):
            w1 = re.sub(r'[^a-z]', '', query_tokens[i])
            w2 = re.sub(r'[^a-z]', '', query_tokens[i + 1])
            if len(w1) > 3 and len(w2) > 3 and w1 not in stopwords and w2 not in stopwords:
                query_bigrams.add((w1, w2))
        
        resp_tokens = response_lower.split()
        resp_bigrams = set()
        for i in range(len(resp_tokens) - 1):
            w1 = re.sub(r'[^a-z]', '', resp_tokens[i])
            w2 = re.sub(r'[^a-z]', '', resp_tokens[i + 1])
            if len(w1) > 2 and len(w2) > 2:
                resp_bigrams.add((w1, w2))
        
        if query_bigrams:
            bigram_coverage = len(query_bigrams & resp_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.3
        
        aspect_score = (query_coverage_ratio * 15) + (bigram_coverage * 10)
        
        # =====================================================
        # 2. CLAUSE DENSITY & ELABORATION DEPTH (0-20 pts)
        # =====================================================
        # Count subordinate clauses and elaboration markers
        elaboration_markers = [
            r'\bbecause\b', r'\bsince\b', r'\balthough\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bhowever\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bsuch as\b',
            r'\bwhich means\b', r'\bthis means\b', r'\bas a result\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bnevertheless\b',
            r'\bdespite\b', r'\bgiven that\b', r'\bprovided that\b',
            r'\bif you\b', r'\bwhen you\b', r'\bthe reason\b',
            r'\bthe trade-off\b', r'\bessentially\b', r'\bin fact\b',
        ]
        
        clause_count = 0
        for pattern in elaboration_markers:
            clause_count += len(re.findall(pattern, response_lower))
        
        # Normalize by response length (per 100 words)
        word_count = len(resp_tokens)
        if word_count > 0:
            clause_density = clause_count / (word_count / 100.0)
        else:
            clause_density = 0
        
        # Also count commas as a proxy for clause complexity
        comma_count = response_text.count(',')
        comma_density = comma_count / max(word_count / 100.0, 1)
        
        depth_score = min(clause_density * 3.5, 12) + min(comma_density * 0.8, 8)
        
        # =====================================================
        # 3. EVIDENCE & EXAMPLE DENSITY (0-15 pts)
        # =====================================================
        evidence_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\blike\s+\w+\s+and\s+\w+',
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bdata\b', r'\bevidence\b', r'\bstatistic', r'\bpercent\b',
            r'\b\d+%', r'\$\d+', r'\b\d{4}\b',  # years
            r'\bcite[ds]?\b', r'\bsource\b', r'\brefer(?:ence|ring)\b',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, response_lower))
        
        # Detect specific names, proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response_text)
        evidence_count += min(len(proper_nouns), 5)
        
        # Detect quoted text or references
        quotes = re.findall(r'["\u201c].*?["\u201d]', response_text)
        evidence_count += len(quotes)
        
        # Detect code blocks or technical content
        code_blocks = re.findall(r'```', response_text)
        evidence_count += len(code_blocks) * 2
        
        # Detect book/work references with asterisks or italics
        book_refs = re.findall(r'\*[A-Z][^*]+\*', response_text)
        evidence_count += len(book_refs)
        
        evidence_score = min(evidence_count * 1.5, 15)
        
        # =====================================================
        # 4. PERSPECTIVE DIVERSITY (0-10 pts)
        # =====================================================
        perspective_markers = [
            r'\bon the other hand\b', r'\balternatively\b', r'\bconversely\b',
            r'\banother (?:view|perspective|approach|way|option|argument)\b',
            r'\bsome (?:people|argue|say|think|believe)\b',
            r'\bothers (?:argue|say|think|believe)\b',
            r'\bfrom (?:a|the|one) (?:\w+ )?perspective\b',
            r'\bhowever\b', r'\bbut\b', r'\bnevertheless\b',
            r'\bwhile (?:some|others|this|that)\b',
            r'\bboth\b', r'\bnot only\b', r'\balso\b',
            r'\bin contrast\b', r'\bthat said\b',
            r'\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b',
        ]
        
        perspective_count = 0
        for pattern in perspective_markers:
            perspective_count += min(len(re.findall(pattern, response_lower)), 2)
        
        perspective_score = min(perspective_count * 1.5, 10)
        
        # =====================================================
        # 5. STRUCTURAL COMPLETENESS (0-15 pts)
        # =====================================================
        # Check for response length relative to query complexity
        query_complexity = num_questions + len(query_content_set) / 5.0
        
        # Ideal response length scales with query complexity
        ideal_min_words = max(30, query_complexity * 20)
        
        # Length score with diminishing returns
        if word_count < 10:
            length_score = 0
        elif word_count < ideal_min_words:
            length_score = (word_count / ideal_min_words) * 8
        else:
            # Diminishing returns after ideal minimum
            length_score = 8 + min(math.log(word_count / ideal_min_words + 1) * 4, 7)
        
        length_score = min(length_score, 15)
        
        # =====================================================
        # 6. TRUNCATION & INCOMPLETENESS PENALTY (0 to -10)
        # =====================================================
        truncation_penalty = 0
        
        # Check if response ends mid-sentence
        last_chars = response_text[-3:] if len(response_text) >= 3 else response_text
        proper_endings = {'.', '!', '?', ')', '"', "'", '`', ':', ']', '}'}
        if last_chars[-1] not in proper_endings and not response_text.endswith('...'):
            truncation_penalty -= 4
        
        # Check if response ends with an incomplete word or thought
        if re.search(r'\b\w{1,2}$', response_text) and not re.search(r'\b(?:I|a|to|is|it|or|so|no|do|am|an)\s*$', response_text):
            truncation_penalty -= 2
        
        # Check for "..." at end suggesting continuation
        if response_text.rstrip().endswith('...'):
            truncation_penalty -= 1
        
        # Very short responses for complex queries
        if word_count < 25 and query_complexity > 3:
            truncation_penalty -= 3
        
        # =====================================================
        # 7. DIRECT ADDRESS BONUS (0-5 pts)
        # =====================================================
        # Does the response directly engage with the specific scenario/question?
        direct_address_score = 0
        
        # Check for second person engagement
        if re.search(r'\byou\b', response_lower):
            direct_address_score += 1
        
        # Check for conditional/nuanced responses
        nuance_patterns = [
            r'\bit depends\b', r'\bgenerally\b', r'\btypically\b',
            r'\bin most cases\b', r'\bthere are (?:several|multiple|a few)\b',
            r'\bthe (?:key|main|important)\b', r'\bkeep in mind\b',
            r'\bnote that\b', r'\bimportant(?:ly)?\b',
        ]
        for pattern in nuance_patterns:
            if re.search(pattern, response_lower):
                direct_address_score += 0.5
        
        direct_address_score = min(direct_address_score, 5)
        
        # =====================================================
        # 8. INFORMATION DENSITY (0-10 pts)
        # =====================================================
        # Unique meaningful words ratio (type-token ratio adjusted)
        meaningful_words = [w for w in re.findall(r'\b[a-z]{3,}\b', response_lower) if w not in stopwords]
        if meaningful_words:
            unique_meaningful = len(set(meaningful_words))
            # Adjusted TTR using root TTR to handle length variation
            if len(meaningful_words) > 0:
                root_ttr = unique_meaningful / math.sqrt(len(meaningful_words))
            else:
                root_ttr = 0
            info_density_score = min(root_ttr * 1.5, 10)
        else:
            info_density_score = 0
        
        # =====================================================
        # COMBINE ALL SCORES
        # =====================================================
        total = (
            aspect_score +           # 0-25: query aspect coverage
            depth_score +            # 0-20: elaboration depth
            evidence_score +         # 0-15: evidence/examples
            perspective_score +      # 0-10: multiple perspectives
            length_score +           # 0-15: structural completeness
            truncation_penalty +     # -10 to 0: truncation penalty
            direct_address_score +   # 0-5: direct engagement
            info_density_score       # 0-10: information density
        )
        
        # Normalize to 0-10 scale
        # Max theoretical: 25+20+15+10+15+0+5+10 = 100
        # Practical max is around 60-70
        normalized = max(0, min(total / 7.0, 10.0))
        
        return round(normalized, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score based on length
        try:
            return min(max(len(str(response)) / 100.0, 0.5), 5.0)
        except:
            return 2.0