def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response.
    
    This variant uses a question-decomposition and topic-coverage approach:
    1. Extract implicit sub-questions/topics from the query
    2. Check how many query topics are addressed in the response
    3. Measure structural depth via clause complexity and explanation patterns
    4. Detect hedging, examples, counterpoints, and elaboration markers
    5. Penalize truncation and superficiality
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # ---- 1. Query Decomposition: Extract topic chunks and question words ----
        # Split query into meaningful chunks (phrases around punctuation, question marks, etc.)
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract question fragments (split by ?, newlines, bullet points)
        question_fragments = re.split(r'[?\n\r•\-\*]', query_lower)
        question_fragments = [f.strip() for f in question_fragments if len(f.strip()) > 10]
        
        # Extract content words from query (nouns, verbs, adjectives - approximated by length and stopword filtering)
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
            'if', 'while', 'about', 'up', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'it', 'its', 'my', 'your', 'his',
            'her', 'our', 'their', 'me', 'him', 'them', 'i', 'you', 'he', 'she',
            'we', 'they', 'like', 'also', 'really', 'think', 'know', 'get', 'got',
            'much', 'many', 'any', 'been', 'going', 'make', 'made', 'even', 'still',
            'well', 'back', 'way', 'thing', 'things', 'something', 'anything',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        query_content_words = extract_content_words(query)
        response_content_words = extract_content_words(response)
        
        # Build bigrams from query for topic phrase matching
        query_words_all = re.findall(r'[a-z]+', query_lower)
        query_bigrams = set()
        for i in range(len(query_words_all) - 1):
            if query_words_all[i] not in stopwords or query_words_all[i+1] not in stopwords:
                query_bigrams.add((query_words_all[i], query_words_all[i+1]))
        
        response_words_all = re.findall(r'[a-z]+', response_lower)
        response_bigrams = set()
        for i in range(len(response_words_all) - 1):
            response_bigrams.add((response_words_all[i], response_words_all[i+1]))
        
        # ---- 2. Topic Coverage Score ----
        # What fraction of query content words appear in the response?
        if query_content_words:
            query_word_set = set(query_content_words)
            response_word_set = set(response_content_words)
            word_coverage = len(query_word_set & response_word_set) / len(query_word_set)
        else:
            word_coverage = 0.5
        
        # Bigram coverage
        if query_bigrams:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        # Fragment coverage: for each query fragment, check if key words appear in response
        fragment_scores = []
        for frag in question_fragments:
            frag_words = extract_content_words(frag)
            if len(frag_words) >= 2:
                covered = sum(1 for w in frag_words if w in response_lower)
                fragment_scores.append(covered / len(frag_words))
        
        if fragment_scores:
            fragment_coverage = sum(fragment_scores) / len(fragment_scores)
        else:
            fragment_coverage = word_coverage  # fallback
        
        topic_score = (word_coverage * 0.4 + bigram_coverage * 0.3 + fragment_coverage * 0.3)
        
        # ---- 3. Structural Depth: Clause complexity and explanation density ----
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Average clause count per sentence (approximated by commas, semicolons, conjunctions)
        clause_markers = re.findall(r'[,;]|\b(?:because|although|while|whereas|since|unless|however|therefore|moreover|furthermore|additionally|consequently|nevertheless|thus|hence|accordingly)\b', response_lower)
        avg_clauses_per_sentence = len(clause_markers) / num_sentences
        clause_complexity = min(avg_clauses_per_sentence / 3.0, 1.0)  # normalize to [0,1]
        
        # Explanation patterns: "this means", "for example", "in other words", "specifically"
        explanation_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bthis means\b', r'\bin other words\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bto illustrate\b', r'\bconsider\b', r'\bsuppose\b',
            r'\bthe reason\b', r'\bthis is because\b', r'\bdue to\b',
            r'\bas a result\b', r'\bwhich means\b', r'\bin practice\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\balternatively\b',
            r'\bmore importantly\b', r'\bnotably\b', r'\bcrucially\b',
        ]
        explanation_count = sum(len(re.findall(p, response_lower)) for p in explanation_patterns)
        explanation_density = min(explanation_count / 5.0, 1.0)
        
        # ---- 4. Elaboration and Multi-perspective Detection ----
        # Detect examples, counterpoints, caveats, qualifications
        example_markers = re.findall(r'\bexample\b|\binstance\b|\bcase\b|\bscenario\b|\billustrat', response_lower)
        counterpoint_markers = re.findall(r'\bhowever\b|\bon the other hand\b|\bthat said\b|\balthough\b|\bdespite\b|\bnevertheless\b|\bconversely\b|\bbut\b|\bwhile\b.*\b(?:also|however)\b', response_lower)
        caveat_markers = re.findall(r'\bit depends\b|\bnot always\b|\bin some cases\b|\btypically\b|\bgenerally\b|\busually\b|\bsometimes\b|\boften\b|\bmay not\b|\bmight not\b|\bnot necessarily\b', response_lower)
        
        elaboration_score = min((len(example_markers) * 0.3 + len(counterpoint_markers) * 0.3 + len(caveat_markers) * 0.2) / 3.0, 1.0)
        
        # ---- 5. Information Density ----
        # Unique content words per 100 words (type-token ratio variant)
        if response_content_words:
            unique_ratio = len(set(response_content_words)) / max(len(response_content_words), 1)
        else:
            unique_ratio = 0.0
        
        # Penalize very repetitive responses
        if response_content_words:
            word_freq = Counter(response_content_words)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            repetition_penalty = max(0, 1.0 - (most_common_freq / max(len(response_content_words), 1)) * 5)
        else:
            repetition_penalty = 0.5
        
        info_density = unique_ratio * 0.6 + repetition_penalty * 0.4
        
        # ---- 6. Length and Substance Score ----
        response_len = len(response)
        word_count = len(response.split())
        
        # Length score with diminishing returns (log-based)
        # Very short responses (<50 words) are penalized heavily
        if word_count < 10:
            length_score = 0.05
        elif word_count < 30:
            length_score = 0.2
        elif word_count < 60:
            length_score = 0.4
        elif word_count < 100:
            length_score = 0.6
        elif word_count < 200:
            length_score = 0.8
        else:
            length_score = min(0.8 + 0.2 * math.log(word_count / 200 + 1), 1.0)
        
        # ---- 7. Truncation Detection ----
        truncation_penalty = 0.0
        # Check if response ends mid-sentence or mid-word
        if response[-1] not in '.!?"\')]}':
            truncation_penalty += 0.1
        # Check for abrupt ending patterns
        if re.search(r'\b(the|a|an|to|of|in|for|and|but|or|is|are|was|were|that|this|with)\s*$', response_lower):
            truncation_penalty += 0.15
        # Ends with comma or incomplete
        if response.rstrip()[-1:] in (',', ';', ':'):
            truncation_penalty += 0.1
        # Very short relative to query length
        if len(response) < len(query) * 0.3 and len(query) > 100:
            truncation_penalty += 0.1
        
        truncation_penalty = min(truncation_penalty, 0.3)
        
        # ---- 8. Specificity Score ----
        # Detect specific details: numbers, proper nouns (capitalized words), technical terms
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response)
        # Proper nouns (capitalized words not at sentence start, excluding "I")
        proper_nouns = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]{2,}', response)
        proper_nouns = [p for p in proper_nouns if p.lower() not in stopwords]
        
        # Quoted terms, book titles, technical references
        quoted_terms = re.findall(r'["\*][^"\*]+["\*]', response)
        
        specificity_raw = len(numbers) * 0.2 + len(proper_nouns) * 0.15 + len(quoted_terms) * 0.3
        specificity_score = min(specificity_raw / 4.0, 1.0)
        
        # ---- 9. Direct Address Score ----
        # Does the response directly address the query type?
        # Check for question-type alignment
        is_how_question = bool(re.search(r'\bhow\b', query_lower[:50]))
        is_why_question = bool(re.search(r'\bwhy\b', query_lower[:50]))
        is_what_question = bool(re.search(r'\bwhat\b', query_lower[:50]))
        is_opinion_question = bool(re.search(r'\bshould\b|\bwould\b|\bcould\b|\bopinion\b|\bthink\b|\brecommend\b', query_lower))
        
        direct_address = 0.5  # default
        if is_how_question:
            # Check for procedural/explanatory language
            how_markers = re.findall(r'\bstep\b|\bfirst\b|\bthen\b|\bnext\b|\bprocess\b|\bmethod\b|\bway\b|\bapproach\b|\bby\b.*\bing\b', response_lower)
            direct_address = min(0.3 + len(how_markers) * 0.15, 1.0)
        elif is_why_question:
            why_markers = re.findall(r'\bbecause\b|\breason\b|\bdue to\b|\bcaused\b|\bresult\b|\bsince\b|\btherefore\b|\bexplain\b', response_lower)
            direct_address = min(0.3 + len(why_markers) * 0.15, 1.0)
        elif is_what_question:
            # Check for definitional/descriptive language
            what_markers = re.findall(r'\bis\b|\bare\b|\brefers to\b|\bmeans\b|\bdefined\b|\bconsists\b|\bincludes\b|\binvolves\b', response_lower)
            direct_address = min(0.3 + len(what_markers) * 0.1, 1.0)
        
        # ---- 10. Engagement with Nuance ----
        # Does the response engage with multiple aspects or just give a flat answer?
        paragraph_breaks = response.count('\n\n') + response.count('\n')
        distinct_sections = max(1, paragraph_breaks + 1)
        
        # Multiple distinct points
        point_indicators = re.findall(r'(?:^|\n)\s*(?:\d+[.)]|\*|\-|•)\s', response)
        has_structured_points = len(point_indicators) >= 2
        
        nuance_score = min((distinct_sections * 0.15 + (0.3 if has_structured_points else 0) + explanation_density * 0.3 + elaboration_score * 0.3), 1.0)
        
        # ---- 11. Superficiality Penalty ----
        # Detect responses that are just agreeing, redirecting, or giving meta-commentary
        superficial_patterns = [
            r'^(yes|no|sure|okay|right)[.,!]?\s',
            r'\bi would (suggest|recommend) (looking|checking|searching)\b',
            r'\byou (should|could|might) (try|check|look|search|google)\b',
            r'\bthat\'s a (great|good|interesting) question\b',
            r'\bwelcome to\b.*\bplease read\b',
            r'\bwhile you wait\b',
        ]
        superficiality = sum(1 for p in superficial_patterns if re.search(p, response_lower))
        superficiality_penalty = min(superficiality * 0.15, 0.4)
        
        # Detect if response is mostly meta/moderation rather than substantive
        meta_ratio = 0.0
        meta_phrases = ['please read', 'rules before', 'welcome to', 'your comments will be', 'while you wait', 'you might be interested in this']
        for mp in meta_phrases:
            if mp in response_lower:
                meta_ratio += 0.2
        meta_ratio = min(meta_ratio, 0.5)
        
        # ---- FINAL SCORE COMPUTATION ----
        # Weighted combination
        raw_score = (
            topic_score * 2.0 +          # How well query topics are covered
            clause_complexity * 1.0 +     # Syntactic depth
            explanation_density * 1.5 +   # Explanation richness
            elaboration_score * 1.5 +     # Examples, counterpoints, caveats
            info_density * 1.0 +          # Vocabulary richness
            length_score * 2.0 +          # Sufficient length
            specificity_score * 1.5 +     # Concrete details
            direct_address * 1.0 +        # Addresses question type
            nuance_score * 1.5            # Multi-faceted engagement
        )
        
        # Max possible raw = 2+1+1.5+1.5+1+2+1.5+1+1.5 = 13.0
        normalized = raw_score / 13.0  # [0, 1]
        
        # Apply penalties
        final = normalized - truncation_penalty - superficiality_penalty - meta_ratio
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, final * 10.0))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on length
        try:
            return min(max(len(str(response)) / 200.0, 0.5), 5.0)
        except Exception:
            return 2.5