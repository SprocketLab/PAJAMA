def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Query decomposition (extracting sub-questions/aspects from query)
    - Information density via sentence-level analysis
    - Structural progression detection (intro -> body -> conclusion flow)
    - Specificity scoring via named entities, numbers, and concrete details
    - Redundancy penalty
    - Coverage breadth via unique semantic clusters (using word co-occurrence patterns)
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
        
        # ---- Helper functions ----
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if len(s.strip()) > 5]
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'that', 'this', 'these', 'those', 'it',
            'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'also',
            'like', 'get', 'got', 'make', 'made', 'well', 'much', 'many'
        }
        
        def content_words(text):
            tokens = tokenize(text)
            return [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # ---- 1. Query Decomposition & Aspect Coverage ----
        # Extract question words and key aspects from the query
        query_content = content_words(query)
        query_bigrams = set()
        for i in range(len(query_content) - 1):
            query_bigrams.add((query_content[i], query_content[i+1]))
        
        # Detect multiple aspects/sub-questions in query
        query_aspects = []
        # Split by conjunctions, commas, question marks for sub-questions
        aspect_splits = re.split(r'[,;]|\band\b|\bor\b|\balso\b|\badditionally\b', query.lower())
        for asp in aspect_splits:
            words = [w for w in re.findall(r'[a-zA-Z]+', asp) if w.lower() not in stopwords and len(w) > 2]
            if words:
                query_aspects.append(set(words))
        
        if not query_aspects:
            query_aspects = [set(query_content)]
        
        response_content = content_words(response)
        response_content_set = set(response_content)
        
        # Measure coverage of each query aspect
        aspect_scores = []
        for aspect_words in query_aspects:
            if not aspect_words:
                continue
            covered = len(aspect_words & response_content_set)
            total = len(aspect_words)
            if total > 0:
                aspect_scores.append(covered / total)
        
        aspect_coverage = sum(aspect_scores) / len(aspect_scores) if aspect_scores else 0.5
        
        # ---- 2. Sentence-level Information Density ----
        sentences = get_sentences(response)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 1.0
        
        # Measure unique content words per sentence (information density)
        sentence_densities = []
        sentence_content_sets = []
        for sent in sentences:
            cw = content_words(sent)
            unique_cw = set(cw)
            words_in_sent = len(tokenize(sent))
            if words_in_sent > 0:
                density = len(unique_cw) / words_in_sent
                sentence_densities.append(density)
            sentence_content_sets.append(unique_cw)
        
        avg_density = sum(sentence_densities) / len(sentence_densities) if sentence_densities else 0
        
        # ---- 3. Structural Progression Detection ----
        # Check if response has intro, body, conclusion flow
        # Intro indicators
        intro_patterns = [
            r'^(yes|no|certainly|sure|absolutely|great|awesome|that\'s)',
            r'^(here|let|i\'ll|i will|i can|to answer)',
            r'(overview|introduction|let me explain|here\'s how)',
        ]
        has_intro = 0
        first_two_sents = ' '.join(sentences[:2]).lower() if len(sentences) >= 2 else response[:200].lower()
        for pat in intro_patterns:
            if re.search(pat, first_two_sents):
                has_intro = 1
                break
        
        # Conclusion indicators
        conclusion_patterns = [
            r'(in conclusion|in summary|overall|to summarize|finally|in short|remember)',
            r'(hope this helps|good luck|enjoy|happy|feel free)',
            r'(key takeaway|important to note|keep in mind)',
        ]
        has_conclusion = 0
        last_two_sents = ' '.join(sentences[-2:]).lower() if len(sentences) >= 2 else response[-200:].lower()
        for pat in conclusion_patterns:
            if re.search(pat, last_two_sents):
                has_conclusion = 1
                break
        
        # Body structure: numbered steps, bullets, or logical connectors
        step_markers = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        bullet_markers = len(re.findall(r'(?:^|\n)\s*[-*•]\s', response))
        has_structured_body = min(1.0, (step_markers + bullet_markers) / 3.0)
        
        structural_score = (has_intro * 0.25 + has_conclusion * 0.25 + has_structured_body * 0.5)
        
        # ---- 4. Specificity: numbers, examples, named entities ----
        # Count numbers (years, measurements, quantities)
        numbers = re.findall(r'\b\d+[\.\d]*\b', response)
        num_count = len(numbers)
        
        # Count capitalized words (potential proper nouns / named entities), excluding sentence starts
        proper_nouns = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\n)[A-Z][a-z]+', response)
        mid_sentence_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response)
        named_entity_count = len(mid_sentence_caps)
        
        # Example indicators
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\blike\b(?=\s+\w+\s+\w+)', r'\bincluding\b',
            r'\bspecifically\b', r'\bin particular\b'
        ]
        example_count = sum(len(re.findall(p, response.lower())) for p in example_patterns)
        
        # Specificity score
        specificity = min(1.0, (num_count * 0.15 + named_entity_count * 0.1 + example_count * 0.2))
        
        # ---- 5. Redundancy Penalty ----
        # Detect repeated content across sentences using Jaccard-like overlap
        redundancy_pairs = 0
        total_pairs = 0
        for i in range(len(sentence_content_sets)):
            for j in range(i + 2, len(sentence_content_sets)):  # skip adjacent
                if sentence_content_sets[i] and sentence_content_sets[j]:
                    intersection = len(sentence_content_sets[i] & sentence_content_sets[j])
                    union = len(sentence_content_sets[i] | sentence_content_sets[j])
                    if union > 0:
                        sim = intersection / union
                        if sim > 0.6:
                            redundancy_pairs += 1
                        total_pairs += 1
        
        redundancy_ratio = redundancy_pairs / total_pairs if total_pairs > 0 else 0
        redundancy_penalty = redundancy_ratio * 0.3  # up to 0.3 penalty
        
        # ---- 6. Coverage Breadth via Topic Clusters ----
        # Group content words into clusters based on co-occurrence within sentences
        # Count how many distinct "topic clusters" the response covers
        word_to_sentences = {}
        for idx, cw_set in enumerate(sentence_content_sets):
            for w in cw_set:
                if w not in word_to_sentences:
                    word_to_sentences[w] = set()
                word_to_sentences[w].add(idx)
        
        # Simple clustering: words that only appear in similar sentence sets are one cluster
        # Use unique sentence distribution patterns as proxy for topic diversity
        distribution_patterns = set()
        for w, sent_ids in word_to_sentences.items():
            pattern = frozenset(sent_ids)
            distribution_patterns.add(pattern)
        
        # More unique distribution patterns = more diverse coverage
        num_patterns = len(distribution_patterns)
        topic_diversity = min(1.0, num_patterns / max(15, num_sentences * 2))
        
        # ---- 7. Response Length & Depth Score ----
        total_words = len(tokenize(response))
        # Logarithmic length scoring - diminishing returns
        length_score = min(1.0, math.log(1 + total_words) / math.log(300))
        
        # Depth: average sentence length in content words
        avg_content_per_sentence = len(response_content) / num_sentences if num_sentences > 0 else 0
        depth_score = min(1.0, avg_content_per_sentence / 12.0)
        
        # ---- 8. Explanation Markers (showing reasoning depth) ----
        explanation_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bdue to\b', r'\bthis means\b', r'\bwhich means\b',
            r'\bthe reason\b', r'\bthis is because\b', r'\bconsequently\b',
            r'\bin other words\b', r'\bto clarify\b', r'\bimportantly\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bit\'s important\b'
        ]
        explanation_count = sum(len(re.findall(p, response.lower())) for p in explanation_markers)
        explanation_score = min(1.0, explanation_count / 6.0)
        
        # ---- 9. Edge Case / Caveat Handling ----
        caveat_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bexcept\b', r'\bunless\b',
            r'\bbe aware\b', r'\bnote that\b', r'\bcaution\b', r'\bwarning\b',
            r'\bdepending on\b', r'\bin some cases\b', r'\bnot always\b',
            r'\bif\b.*\bthen\b', r'\bedge case\b', r'\bexception\b',
            r'\balternatively\b', r'\banother option\b', r'\byou could also\b'
        ]
        caveat_count = sum(len(re.findall(p, response.lower())) for p in caveat_patterns)
        caveat_score = min(1.0, caveat_count / 4.0)
        
        # ---- 10. Truncation Detection ----
        # Check if response appears cut off
        truncation_penalty = 0.0
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?:"\')]}':
            # Likely truncated
            truncation_penalty = 0.15
        # Check for incomplete sentences at end
        last_sentence = sentences[-1] if sentences else ""
        last_words = tokenize(last_sentence)
        if len(last_words) < 4 and len(sentences) > 2:
            truncation_penalty += 0.05
        
        # ---- Combine Scores ----
        # Weighted combination emphasizing completeness dimensions
        score = (
            aspect_coverage * 1.5 +       # Query aspect coverage (0-1.5)
            structural_score * 1.5 +       # Structural organization (0-1.5)
            length_score * 1.2 +           # Sufficient length/depth (0-1.2)
            depth_score * 1.0 +            # Content depth per sentence (0-1.0)
            specificity * 1.2 +            # Concrete details (0-1.2)
            explanation_score * 1.3 +      # Reasoning depth (0-1.3)
            topic_diversity * 1.0 +        # Breadth of coverage (0-1.0)
            caveat_score * 0.8 +           # Edge cases/caveats (0-0.8)
            avg_density * 0.5 -            # Information density bonus (0-0.5)
            redundancy_penalty -           # Redundancy penalty (0-0.3)
            truncation_penalty             # Truncation penalty (0-0.2)
        )
        # Max theoretical: ~10.0
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        return round(score, 4)
    
    except Exception as e:
        # Fallback: return a basic length-based score
        try:
            return min(5.0, len(str(response)) / 200.0)
        except:
            return 0.0