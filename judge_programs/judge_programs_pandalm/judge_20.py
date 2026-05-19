def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question decomposition: detecting sub-questions/aspects in the query
    - Information density via unique information units (clause-level analysis)
    - Structural depth (nested explanations, examples, elaborations)
    - Repetition penalty (heavy penalty for repeated content)
    - Query-aspect coverage through semantic field matching
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        if not query or not query.strip():
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        response_words = response.lower().split()
        query_words = query.lower().split()
        
        if len(response_words) == 0:
            return 0.0
        
        # ========== 1. REPETITION PENALTY ==========
        # Detect repeated phrases (n-grams) - a strong signal of low quality
        def compute_repetition_ratio(words, n=3):
            if len(words) < n:
                return 0.0
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(tuple(words[i:i+n]))
            total = len(ngrams)
            unique = len(set(ngrams))
            if total == 0:
                return 0.0
            return 1.0 - (unique / total)
        
        rep_3 = compute_repetition_ratio(response_words, 3)
        rep_5 = compute_repetition_ratio(response_words, 5)
        
        # Also check word-level repetition
        word_counts = Counter(response_words)
        # Filter out common stop words for repetition check
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                      'than', 'too', 'very', 'just', 'because', 'if', 'when', 'where',
                      'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                      'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
                      'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'i',
                      'me', 'my', 'also', 'then'}
        
        content_words = [w for w in response_words if w not in stop_words and len(w) > 2]
        content_counts = Counter(content_words)
        
        if content_words:
            max_content_freq = max(content_counts.values()) if content_counts else 0
            content_unique_ratio = len(set(content_words)) / len(content_words)
        else:
            max_content_freq = 0
            content_unique_ratio = 0
        
        # Heavy repetition penalty
        repetition_penalty = 0.0
        if rep_3 > 0.5:
            repetition_penalty += (rep_3 - 0.5) * 30
        if rep_5 > 0.3:
            repetition_penalty += (rep_5 - 0.3) * 40
        if max_content_freq > 5 and content_words:
            excess = max_content_freq - 5
            repetition_penalty += excess * 2
        if content_unique_ratio < 0.3 and len(content_words) > 10:
            repetition_penalty += (0.3 - content_unique_ratio) * 30
        
        # ========== 2. QUERY DECOMPOSITION - Identify aspects ==========
        # Extract key aspects from the query
        query_lower = query.lower()
        
        # Count explicit sub-tasks (comma-separated actions, "and" conjunctions, numbered items)
        # Look for action verbs and their objects as separate aspects
        action_patterns = [
            r'\b(compare|contrast|explain|describe|discuss|analyze|evaluate|list|provide|give|write|create|generate|show|define|identify|outline)\b',
            r'\b(what|how|why|when|where|who|which)\b',
        ]
        
        num_query_aspects = 1
        # Count conjunctions suggesting multiple aspects
        num_query_aspects += query_lower.count(' and ')
        num_query_aspects += query_lower.count(',')
        # Count question words
        question_words_found = len(re.findall(r'\b(what|how|why|when|where|who|which)\b', query_lower))
        num_query_aspects += max(0, question_words_found - 1)
        
        num_query_aspects = min(num_query_aspects, 8)  # cap
        
        # ========== 3. INFORMATION UNITS (clause-level analysis) ==========
        # Split response into clauses/sentences and count unique information units
        sentences = re.split(r'[.!?;]\s*', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Count unique clauses (split by commas and conjunctions too)
        clauses = re.split(r'[.!?;,]\s*|\b(?:and|but|while|whereas|although|however|moreover|furthermore|additionally)\b', response)
        clauses = [c.strip() for c in clauses if c.strip() and len(c.strip().split()) >= 3]
        
        # Deduplicate clauses by checking similarity
        unique_clauses = []
        for clause in clauses:
            clause_words = set(clause.lower().split())
            is_duplicate = False
            for existing in unique_clauses:
                existing_words = set(existing.lower().split())
                if clause_words and existing_words:
                    overlap = len(clause_words & existing_words) / max(len(clause_words), len(existing_words))
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            if not is_duplicate:
                unique_clauses.append(clause)
        
        num_unique_clauses = len(unique_clauses)
        
        # ========== 4. STRUCTURAL DEPTH INDICATORS ==========
        # Look for elaboration patterns: examples, explanations, qualifications
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bthis means\b', r'\bin other words\b', r'\bthat is\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\balso\b', r'\bin addition\b', r'\bnot only\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\bwhereas\b',
            r'\bwhile\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bbecause\b', r'\bsince\b', r'\bdue to\b', r'\bas a result\b',
            r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bimportantly\b', r'\bsignificantly\b',
        ]
        
        response_lower = response.lower()
        elaboration_count = 0
        for pattern in elaboration_markers:
            elaboration_count += len(re.findall(pattern, response_lower))
        
        # ========== 5. QUERY KEYWORD COVERAGE ==========
        # Extract meaningful query keywords and check coverage
        query_content_words = [w for w in query_words if w.lower() not in stop_words and len(w) > 2]
        # Remove common instruction words
        instruction_words = {'following', 'given', 'input', 'output', 'please', 'provide',
                           'write', 'create', 'generate', 'describe', 'explain', 'make',
                           'come', 'rewrite', 'sentence', 'words'}
        query_keywords = [w.lower().strip('.,!?:;()[]"\'') for w in query_content_words 
                         if w.lower().strip('.,!?:;()[]"\'') not in instruction_words]
        
        if query_keywords:
            response_word_set = set(w.lower().strip('.,!?:;()[]"\'') for w in response_words)
            covered = sum(1 for kw in query_keywords if kw in response_word_set)
            keyword_coverage = covered / len(query_keywords)
        else:
            keyword_coverage = 0.5  # neutral if no keywords extractable
        
        # ========== 6. SEMANTIC FIELD EXPANSION ==========
        # Check if response introduces NEW relevant content words beyond query
        response_content = set(w.lower().strip('.,!?:;()[]"\'') for w in response_words 
                              if w.lower() not in stop_words and len(w) > 2)
        query_content_set = set(w.lower().strip('.,!?:;()[]"\'') for w in query_words)
        
        new_terms = response_content - query_content_set
        semantic_expansion = len(new_terms)
        
        # ========== 7. RESPONSE LENGTH SCORING (diminishing returns) ==========
        word_count = len(response_words)
        # Use logarithmic scaling for length - diminishing returns
        if word_count <= 3:
            length_score = word_count * 0.5
        elif word_count <= 20:
            length_score = 1.5 + math.log(word_count, 2) * 1.5
        elif word_count <= 100:
            length_score = 5.0 + math.log(word_count / 20, 2) * 3.0
        elif word_count <= 300:
            length_score = 8.0 + math.log(word_count / 100, 2) * 1.5
        else:
            length_score = 9.5 + math.log(word_count / 300, 2) * 0.5
        
        length_score = min(length_score, 12.0)
        
        # ========== 8. SPECIFICITY SCORE ==========
        # Longer words and technical terms suggest more specific/detailed content
        if content_words:
            avg_word_len = sum(len(w) for w in content_words) / len(content_words)
            long_words = sum(1 for w in content_words if len(w) >= 7)
            specificity = (avg_word_len - 3.0) * 1.5 + (long_words / max(len(content_words), 1)) * 5
        else:
            specificity = 0
        specificity = max(0, min(specificity, 8))
        
        # ========== 9. TRUNCATION DETECTION ==========
        truncation_penalty = 0.0
        if response[-1] not in '.!?")\']' and len(response) > 50:
            # Likely truncated
            truncation_penalty = 3.0
        # Check for incomplete sentences at the end
        last_sentence = sentences[-1] if sentences else ""
        last_words = last_sentence.split()
        if last_words and last_words[-1].lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 
                                                       'more', 'and', 'or', 'but', 'to', 'in', 'of',
                                                       'for', 'with', 'that', 'this']:
            truncation_penalty += 2.0
        
        # ========== 10. EMPTY/NOINPUT DETECTION ==========
        if response.strip().lower() in ['<noinput>', 'noinput', 'n/a', 'none', '']:
            return 0.5
        
        # ========== COMPOSITE SCORING ==========
        # Clause richness (0-15)
        clause_score = min(num_unique_clauses * 1.8, 15.0)
        
        # Elaboration depth (0-10)
        elaboration_score = min(elaboration_count * 1.2, 10.0)
        
        # Keyword coverage (0-10)
        coverage_score = keyword_coverage * 10.0
        
        # Semantic expansion (0-8)
        expansion_score = min(semantic_expansion * 0.4, 8.0)
        
        # Sentence count bonus (0-8)
        sentence_score = min(num_sentences * 1.2, 8.0)
        
        # Aspect coverage estimate: reward responses that seem to address multiple aspects
        # Use ratio of unique clauses to expected aspects
        if num_query_aspects > 0:
            aspect_ratio = min(num_unique_clauses / num_query_aspects, 3.0)
            aspect_score = aspect_ratio * 4.0
        else:
            aspect_score = 4.0
        aspect_score = min(aspect_score, 12.0)
        
        # Final composite
        raw_score = (
            length_score * 1.5 +        # max ~18
            clause_score * 1.2 +         # max ~18
            elaboration_score * 0.8 +    # max ~8
            coverage_score * 1.0 +       # max ~10
            expansion_score * 0.7 +      # max ~5.6
            sentence_score * 0.8 +       # max ~6.4
            aspect_score * 0.8 +         # max ~9.6
            specificity * 0.6            # max ~4.8
        )
        # Theoretical max around 80
        
        # Apply penalties
        raw_score -= repetition_penalty
        raw_score -= truncation_penalty
        
        # Normalize to 0-100
        final_score = max(0.0, min(raw_score * 1.25, 100.0))
        
        return round(final_score, 2)
    
    except Exception:
        try:
            # Absolute fallback: simple word count
            if response and response.strip():
                return min(len(response.strip().split()) * 0.5, 50.0)
            return 0.0
        except Exception:
            return 0.0