def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a structural
    depth analysis approach based on:
    - Question/aspect detection in query and coverage in response
    - Information density via unique concept/entity tracking
    - Structural diversity (different rhetorical moves)
    - Response-to-query proportionality
    - Specificity scoring via concrete details detection
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.5
        
        score = 0.0
        
        # === 1. Query Aspect Detection & Coverage (0-25 points) ===
        # Extract key aspects/topics from the query
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract meaningful content words from query (nouns, verbs, adjectives - approximated)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'them', 'they', 'their', 'this',
            'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'she', 'him', 'her', 'his', 'who', 'which',
            'what', 'whom', 'whose', 'also', 'any', 'much', 'many', 'well',
            'get', 'got', 'like', 'make', 'made', 'know', 'need', 'want',
            'thing', 'things', 'way', 'ways', 'person', 'people', 'one', 'two',
            'must', 'ai', 'model', 'response', 'provide', 'given', 'using',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        query_content = extract_content_words(query)
        response_content = extract_content_words(response)
        
        # Build bigrams from query for topic detection
        query_bigrams = set()
        for i in range(len(query_content) - 1):
            query_bigrams.add((query_content[i], query_content[i+1]))
        
        # Check coverage of query content words in response
        query_unique = set(query_content)
        response_word_set = set(response_content)
        
        if query_unique:
            covered = sum(1 for w in query_unique if w in response_word_set)
            coverage_ratio = covered / len(query_unique)
        else:
            coverage_ratio = 0.5
        
        # Bigram coverage
        if query_bigrams:
            response_bigrams = set()
            for i in range(len(response_content) - 1):
                response_bigrams.add((response_content[i], response_content[i+1]))
            bigram_covered = sum(1 for bg in query_bigrams if bg in response_bigrams)
            bigram_coverage = bigram_covered / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        aspect_score = (coverage_ratio * 15) + (bigram_coverage * 10)
        score += aspect_score
        
        # === 2. Information Density & Unique Concepts (0-20 points) ===
        # Count unique meaningful words relative to total - higher ratio = more diverse info
        response_words = re.findall(r'[a-z]+', response_lower)
        total_words = len(response_words)
        
        if total_words == 0:
            return 1.0
        
        unique_content = set(response_content)
        content_richness = len(unique_content)
        
        # Diminishing returns on content richness
        richness_score = min(20, math.log1p(content_richness) * 3.5)
        score += richness_score
        
        # === 3. Rhetorical Move Diversity (0-15 points) ===
        # Detect different types of rhetorical moves in the response
        moves_detected = 0
        
        # Acknowledgment/empathy
        empathy_patterns = [
            r'\bi (?:understand|hear|see|can see|recognize|appreciate)\b',
            r'\bthat\'s (?:completely |totally |absolutely )?(?:understandable|okay|fine|normal|natural)\b',
            r'\bi\'m (?:sorry|genuinely sorry)\b',
            r'\bit\'s (?:completely |totally |absolutely )?(?:okay|fine|normal|natural|understandable)\b',
        ]
        if any(re.search(p, response_lower) for p in empathy_patterns):
            moves_detected += 1
        
        # Explanation/reasoning
        explanation_patterns = [
            r'\bbecause\b', r'\bdue to\b', r'\bthis (?:is|means|allows|enables)\b',
            r'\bthe reason\b', r'\bin other words\b', r'\bwhich means\b',
            r'\bthink of (?:it )?as\b', r'\bimagine\b',
        ]
        if any(re.search(p, response_lower) for p in explanation_patterns):
            moves_detected += 1
        
        # Actionable advice/instructions
        action_patterns = [
            r'\btry to\b', r'\byou (?:can|could|should|might|may)\b',
            r'\bstart (?:by|with)\b', r'\bfirst\b.*\bthen\b',
            r'\bstep \d\b', r'\bhere\'s how\b', r'\bfollow\b',
            r'\bmake sure\b', r'\bdon\'t forget\b', r'\bremember to\b',
        ]
        if any(re.search(p, response_lower) for p in action_patterns):
            moves_detected += 1
        
        # Examples or analogies
        example_patterns = [
            r'\bfor (?:example|instance)\b', r'\bsuch as\b',
            r'\blike (?:a|an|the|when|how)\b', r'\bimagine (?:if|you|a|that)\b',
            r'\bjust like\b', r'\bsimilar to\b', r'\bthink of\b',
            r'\banalog\b', r'\bmetaphor\b',
        ]
        if any(re.search(p, response_lower) for p in example_patterns):
            moves_detected += 1
        
        # Qualification/nuance
        nuance_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b.*\b(?:also|still)\b',
            r'\bon the other hand\b', r'\bthat said\b', r'\bkeep in mind\b',
            r'\bit\'s (?:important|worth|crucial)\b', r'\bnot always\b',
        ]
        if any(re.search(p, response_lower) for p in nuance_patterns):
            moves_detected += 1
        
        # Encouragement/positive framing
        encourage_patterns = [
            r'\byou\'(?:ll|re) (?:going to |gonna )?(?:be fine|get through|do great)\b',
            r'\bdon\'t (?:hesitate|worry|be afraid|be shy)\b',
            r'\bremember\b.*\b(?:important|okay|fine|normal)\b',
            r'\byou\'ve got\b', r'\bkeep (?:going|working|trying)\b',
        ]
        if any(re.search(p, response_lower) for p in encourage_patterns):
            moves_detected += 1
        
        # Questioning/clarification seeking
        question_patterns = [
            r'\?', r'\bcould you (?:tell|share|provide|clarify)\b',
            r'\bcan you (?:give|share|provide)\b', r'\bwhat (?:is|are|do)\b',
        ]
        if any(re.search(p, response_lower) for p in question_patterns):
            moves_detected += 1
        
        # Summary/conclusion
        summary_patterns = [
            r'\bin (?:summary|conclusion|short)\b', r'\boverall\b',
            r'\bto (?:sum|summarize|wrap)\b', r'\bmost importantly\b',
            r'\bthe (?:key|main|bottom) (?:point|takeaway|line)\b',
        ]
        if any(re.search(p, response_lower) for p in summary_patterns):
            moves_detected += 1
        
        moves_score = min(15, moves_detected * 2.5)
        score += moves_score
        
        # === 4. Structural Depth via Clause Analysis (0-15 points) ===
        # Count subordinate clauses, compound sentences - indicates depth
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return max(1.0, score * 0.1)
        
        # Average clauses per sentence (approximated by commas, semicolons, conjunctions)
        clause_markers = [',', ';', ' and ', ' but ', ' or ', ' while ', ' although ',
                         ' because ', ' since ', ' when ', ' where ', ' which ', ' that ',
                         ' however ', ' therefore ']
        
        total_clauses = 0
        for sent in sentences:
            clause_count = 1  # base clause
            for marker in clause_markers:
                clause_count += sent.lower().count(marker)
            total_clauses += clause_count
        
        avg_clauses = total_clauses / num_sentences if num_sentences > 0 else 1
        # Ideal avg clauses: 2-4
        clause_quality = min(1.0, avg_clauses / 3.0)
        
        # Sentence count contribution (more sentences = more complete, with diminishing returns)
        sentence_depth = min(1.0, math.log1p(num_sentences) / math.log1p(12))
        
        depth_score = (clause_quality * 7.5) + (sentence_depth * 7.5)
        score += depth_score
        
        # === 5. Specificity via Concrete Detail Detection (0-15 points) ===
        # Detect numbers, proper nouns, specific terms, technical vocabulary
        specificity_count = 0
        
        # Numbers and quantities
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        specificity_count += min(5, len(numbers))
        
        # Quoted terms or emphasized words
        quoted = re.findall(r'["\']([^"\']+)["\']', response)
        specificity_count += min(3, len(quoted))
        
        # Technical/domain terms (words with capital letters mid-text, excluding sentence starts)
        # Approximate proper nouns
        words_raw = response.split()
        capitalized_mid = 0
        for i, w in enumerate(words_raw):
            if i > 0 and w and w[0].isupper() and not re.match(r'^[.!?]', words_raw[i-1][-1:] if words_raw[i-1] else ''):
                capitalized_mid += 1
        specificity_count += min(5, capitalized_mid)
        
        # Parenthetical explanations (indicates thoroughness)
        parens = re.findall(r'\([^)]+\)', response)
        specificity_count += min(3, len(parens))
        
        # Enumeration patterns (1., 2., a), b), etc.)
        enumerations = re.findall(r'(?:^|\n)\s*(?:\d+[.):]|[a-z][.):]|\-|\*|•)', response)
        specificity_count += min(5, len(enumerations))
        
        # Conditional/edge case handling
        conditionals = len(re.findall(r'\bif\b|\bin case\b|\bunless\b|\bwhen\b.*\bthen\b|\bdepending\b', response_lower))
        specificity_count += min(3, conditionals)
        
        specificity_score = min(15, specificity_count * 1.2)
        score += specificity_score
        
        # === 6. Response Proportionality & Engagement (0-10 points) ===
        # Longer, more detailed responses tend to be more complete
        # But must be proportional to query complexity
        
        query_words = len(query.split())
        response_words_count = total_words
        
        # Response should generally be longer than query for completeness
        if query_words > 0:
            ratio = response_words_count / query_words
        else:
            ratio = response_words_count / 10.0
        
        # Sweet spot: response is 1.5x to 5x the query length
        if ratio < 0.5:
            proportion_score = ratio * 4  # penalize very short
        elif ratio < 1.0:
            proportion_score = 2 + (ratio - 0.5) * 6
        elif ratio < 5.0:
            proportion_score = 5 + min(5, (ratio - 1.0) * 1.25)
        else:
            proportion_score = 10  # cap at max
        
        score += proportion_score
        
        # === 7. Tone Appropriateness Bonus (0-5 points) ===
        # Check if response tone matches what the query seems to need
        
        # Detect if query is emotional/seeking comfort
        emotional_query = bool(re.search(
            r'\bfeel(?:ing|s)?\b|\bfrustrat\w+\b|\bstress\w*\b|\bsad\b|\bupset\b|\bworr\w+\b|\blonely\b|\bheartbrok\w+\b|\bdevast\w+\b|\bexhaust\w+\b|\bdespair\b',
            query_lower
        ))
        
        # Detect if response shows emotional awareness
        emotional_response = bool(re.search(
            r'\bunderstand\w*\b|\bsorry\b|\bhear\b|\bfeel\w*\b|\bokay\b|\bnormal\b|\bnatural\b|\bvalid\b|\bcomfort\b',
            response_lower
        ))
        
        # Detect if query is technical/informational
        technical_query = bool(re.search(
            r'\bhow (?:to|do|does|can|would)\b|\bexplain\b|\bguide\b|\bsteps?\b|\bprocess\b|\bmethod\b',
            query_lower
        ))
        
        # Detect if response provides structured info
        structured_response = bool(enumerations) or bool(re.search(r'\bfirst\b.*\bthen\b', response_lower))
        
        tone_score = 0
        if emotional_query and emotional_response:
            tone_score += 3
        if technical_query and structured_response:
            tone_score += 2
        if not emotional_query and not technical_query:
            tone_score += 2  # neutral baseline
        
        tone_score = min(5, tone_score)
        score += tone_score
        
        # === Final normalization to 1-5 scale ===
        # Max possible raw score: 25 + 20 + 15 + 15 + 15 + 10 + 5 = 105
        # Normalize to 1-5
        normalized = 1.0 + (score / 105.0) * 4.0
        
        # Clamp
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        return 2.5