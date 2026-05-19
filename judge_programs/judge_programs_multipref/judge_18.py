def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and semantic coverage approach. This variant focuses on:
    1. Decomposing the query into sub-questions/aspects and checking coverage
    2. Measuring information density and depth via unique concept tracking
    3. Evaluating structural completeness (intro, body, conclusion patterns)
    4. Checking for hedging/uncertainty vs. confident comprehensive answers
    5. Measuring lexical diversity relative to query terms
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.0
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # ---- 1. Query Aspect Decomposition and Coverage ----
        # Extract key aspects from the query that should be addressed
        
        # Extract question words and their associated phrases
        question_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhich\b', r'\bwho\b', r'\bcan\b', r'\bdo\b', r'\bshould\b',
            r'\bis\b', r'\bare\b', r'\bwill\b', r'\bwould\b'
        ]
        
        # Count distinct question types in query
        question_count = sum(1 for p in question_patterns if re.search(p, query_lower))
        question_count = max(question_count, 1)
        
        # Extract noun phrases / key concepts from query using simple chunking
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'i', 'me', 'my', 'myself',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
            'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'im', 'get', 'got', 'need',
            'want', 'like', 'think', 'know', 'bit', 'any', 'also'
        }
        
        # Extract meaningful query terms
        query_words = re.findall(r'[a-z]+', query_lower)
        query_content_words = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        # Calculate what fraction of query content words appear in response
        if query_content_words:
            covered_query_terms = sum(1 for w in set(query_content_words) if w in response_lower)
            query_coverage_ratio = covered_query_terms / len(set(query_content_words))
        else:
            query_coverage_ratio = 0.5
        
        # ---- 2. Information Density via Unique Concept Tracking ----
        response_words = re.findall(r'[a-z]+', response_lower)
        response_content_words = [w for w in response_words if w not in stop_words and len(w) > 2]
        
        # Unique content words as measure of breadth
        unique_content = set(response_content_words)
        total_content = len(response_content_words)
        
        # Information density: unique concepts per 100 words
        if total_content > 0:
            info_density = len(unique_content) / max(total_content, 1)
        else:
            info_density = 0
        
        # Absolute count of unique concepts (breadth)
        concept_breadth = len(unique_content)
        
        # ---- 3. Depth Analysis: Multi-faceted explanation detection ----
        # Look for explanatory patterns that indicate depth
        depth_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bas a result\b', r'\bdue to\b', r'\bthis means\b', r'\bin order to\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b', r'\blike\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bhowever\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bwhile\b', r'\balthough\b', r'\bdespite\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\badditionally\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bimportant\b', r'\bkey\b', r'\bessential\b', r'\bcritical\b',
            r'\bnote that\b', r'\bkeep in mind\b', r'\bremember\b',
        ]
        
        depth_count = sum(1 for p in depth_markers if re.search(p, response_lower))
        
        # ---- 4. Structural Completeness ----
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_count = len(sentences)
        
        # Check for introduction (first sentence addresses query topic)
        has_intro = False
        if sentences:
            first_sent_lower = sentences[0].lower()
            query_key_terms = set(query_content_words[:5]) if query_content_words else set()
            intro_overlap = sum(1 for t in query_key_terms if t in first_sent_lower)
            has_intro = intro_overlap >= min(2, len(query_key_terms))
        
        # Check for conclusion/summary patterns
        conclusion_patterns = [
            r'\bin summary\b', r'\boverall\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\bin short\b', r'\bultimately\b',
            r'\bhope this\b', r'\bgood luck\b', r'\bhappy\b', r'\benjoy\b',
            r'\bremember\b', r'\bkey takeaway\b'
        ]
        has_conclusion = any(re.search(p, response_lower[-300:]) for p in conclusion_patterns) if len(response_lower) > 50 else False
        
        # ---- 5. Truncation Detection ----
        # Detect if response was cut off mid-sentence
        is_truncated = False
        stripped_response = response.rstrip()
        if stripped_response:
            last_char = stripped_response[-1]
            if last_char not in '.!?"\')]}:;':
                is_truncated = True
            # Also check if it ends mid-word or mid-thought
            last_50 = stripped_response[-50:] if len(stripped_response) > 50 else stripped_response
            if re.search(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|is|are|was|were|this|that|these|those)\s*$', last_50.lower()):
                is_truncated = True
        
        # ---- 6. Hedging and Uncertainty Detection ----
        hedge_patterns = [
            r"\bi'm not sure\b", r"\bi don't know\b", r"\bi'm not aware\b",
            r"\bi cannot\b", r"\bi can't\b", r"\bpossibly\b", r"\bperhaps\b",
            r"\bmaybe\b", r"\bmight be\b", r"\bcould be\b",
            r"\bnot certain\b", r"\bnot clear\b", r"\bhard to say\b",
            r"\bi think\b", r"\bi believe\b", r"\bprobably\b",
            r"\bnot necessarily\b", r"\bit depends\b"
        ]
        hedge_count = sum(1 for p in hedge_patterns if re.search(p, response_lower))
        
        # ---- 7. Specificity Score ----
        # Numbers, proper nouns, technical terms indicate specificity
        numbers = re.findall(r'\d+\.?\d*', response)
        number_count = len(numbers)
        
        # Words with capital letters (potential proper nouns/technical terms)
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', response)
        # Filter out sentence starters
        proper_noun_estimate = max(0, len(capitalized_words) - sentence_count)
        
        # Technical/specific vocabulary (longer words tend to be more specific)
        long_words = [w for w in response_content_words if len(w) > 7]
        specificity_ratio = len(long_words) / max(len(response_content_words), 1)
        
        # ---- 8. Enumeration and Multi-point Coverage ----
        # Detect numbered lists, lettered lists
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[.)\-]|\*|\-|•|[a-z][.)\-])', response)
        enum_count = len(numbered_items)
        
        # Detect bold/emphasized points (markdown)
        bold_items = re.findall(r'\*\*[^*]+\*\*', response)
        bold_count = len(bold_items)
        
        # ---- 9. Response Length Adequacy ----
        # Longer queries typically need longer responses
        query_complexity = len(query_content_words) + question_count
        response_length = len(response)
        
        # Minimum expected length based on query complexity
        min_expected_length = min(query_complexity * 30, 500)
        length_adequacy = min(response_length / max(min_expected_length, 1), 3.0)
        
        # ---- 10. Paragraph/Section Analysis ----
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        para_count = len(paragraphs)
        
        # Check for section headers (markdown or plain)
        headers = re.findall(r'(?:^|\n)#{1,4}\s+.+', response)
        header_count = len(headers)
        
        # ---- SCORING ----
        score = 0.0
        
        # Query coverage (0-15 points)
        score += query_coverage_ratio * 15
        
        # Information breadth (0-15 points)
        breadth_score = min(concept_breadth / 40, 1.0) * 15
        score += breadth_score
        
        # Depth markers (0-12 points)
        depth_score = min(depth_count / 8, 1.0) * 12
        score += depth_score
        
        # Sentence count / content volume (0-12 points)
        sentence_score = min(sentence_count / 10, 1.0) * 12
        score += sentence_score
        
        # Structural completeness (0-8 points)
        if has_intro:
            score += 4
        if has_conclusion:
            score += 4
        
        # Specificity (0-10 points)
        spec_score = 0
        spec_score += min(number_count / 5, 1.0) * 3
        spec_score += min(proper_noun_estimate / 5, 1.0) * 3
        spec_score += specificity_ratio * 4
        score += spec_score
        
        # Enumeration / structured coverage (0-8 points)
        enum_score = min((enum_count + bold_count) / 6, 1.0) * 8
        score += enum_score
        
        # Length adequacy (0-8 points)
        score += min(length_adequacy, 1.5) * (8 / 1.5)
        
        # Multi-section organization (0-5 points)
        org_score = min((para_count + header_count) / 5, 1.0) * 5
        score += org_score
        
        # Information density bonus (0-4 points) - reward moderate density
        # Too high density = repetitive, too low = filler
        ideal_density = 0.45
        density_deviation = abs(info_density - ideal_density)
        density_score = max(0, 1 - density_deviation / 0.3) * 4
        score += density_score
        
        # ---- PENALTIES ----
        
        # Truncation penalty (significant - incomplete response)
        if is_truncated:
            score *= 0.75
        
        # Heavy hedging penalty
        if hedge_count >= 3:
            score -= (hedge_count - 2) * 2
        
        # Very short response penalty
        if response_length < 100:
            score *= 0.4
        elif response_length < 200:
            score *= 0.65
        elif response_length < 300:
            score *= 0.8
        
        # Empty/near-empty penalty
        if sentence_count <= 1:
            score *= 0.3
        elif sentence_count <= 2:
            score *= 0.6
        
        # Clamp score
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)
        
    except Exception:
        return 0.0