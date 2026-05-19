def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a novel approach:
    - Query decomposition: identifies sub-topics/aspects in the query
    - Response segmentation: analyzes distinct "information units" 
    - Depth analysis: measures explanation depth via causal/reasoning chains
    - Coverage breadth: checks for multiple perspectives/approaches
    - Structural completeness signals: opening context, body, conclusion patterns
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 0.0
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        response_words = response_lower.split()
        query_words = query_lower.split()
        
        if len(response_words) < 3:
            return 0.5
        
        score = 0.0
        
        # ============================================================
        # 1. INFORMATION UNIT DENSITY (novel: sentence-level info units)
        # ============================================================
        # Split response into sentences and measure how many carry distinct info
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = max(len(sentences), 1)
        
        # Measure distinctness of sentences via unique content word sets
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                     'than', 'too', 'very', 'just', 'because', 'if', 'when', 'where',
                     'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                     'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
                     'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
                     'also', 'then', 'there', 'here', 'about', 'up', 'out', 'down',
                     'over', 'under', 'again', 'once', 'between', 'while'}
        
        def content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stopwords and len(w) > 2)
        
        sentence_content_sets = [content_words(s) for s in sentences]
        
        # Calculate information novelty: each sentence adds new content words
        seen_words = set()
        novelty_scores = []
        for cset in sentence_content_sets:
            if not cset:
                novelty_scores.append(0)
                continue
            new_words = cset - seen_words
            novelty = len(new_words) / max(len(cset), 1)
            novelty_scores.append(novelty)
            seen_words.update(cset)
        
        avg_novelty = sum(novelty_scores) / max(len(novelty_scores), 1)
        # High novelty means each sentence adds new info (good for coverage)
        info_density_score = min(avg_novelty * 12, 10)
        score += info_density_score * 1.0
        
        # ============================================================
        # 2. QUERY ASPECT COVERAGE (novel: extract question facets)
        # ============================================================
        # Extract potential aspects/facets from the query
        query_content = content_words(query_lower)
        
        # Also look for question words suggesting multiple aspects
        question_indicators = re.findall(r'\b(what|how|why|when|where|who|which|can|should|would|does|do)\b', query_lower)
        
        # Check how many query content words appear in response
        response_content = content_words(response_lower)
        
        if query_content:
            direct_coverage = len(query_content & response_content) / len(query_content)
        else:
            direct_coverage = 0.5
        
        # Also check for semantic expansion: response should contain related terms
        # beyond just the query terms (shows deeper coverage)
        expansion_words = response_content - query_content
        expansion_ratio = len(expansion_words) / max(len(response_content), 1)
        
        aspect_score = (direct_coverage * 6) + (expansion_ratio * 4)
        aspect_score = min(aspect_score, 10)
        score += aspect_score * 1.5
        
        # ============================================================
        # 3. CAUSAL/REASONING CHAIN DEPTH (novel approach)
        # ============================================================
        # Detect reasoning connectors that indicate depth of explanation
        reasoning_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bleading to\b',
            r'\bresulting in\b', r'\bfor this reason\b', r'\bin order to\b',
            r'\bso that\b', r'\bif\.\.\.then\b', r'\bsince\b',
            r'\bgiven that\b', r'\bit follows\b', r'\bimplying\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bincluding\b',
            r'\bnamely\b', r'\bin other words\b', r'\bthat is\b',
        ]
        
        reasoning_count = 0
        for pattern in reasoning_markers:
            reasoning_count += len(re.findall(pattern, response_lower))
        
        # Normalize by response length to avoid just rewarding length
        reasoning_density = reasoning_count / max(num_sentences, 1)
        reasoning_score = min(reasoning_density * 8, 10)
        score += reasoning_score * 1.2
        
        # ============================================================
        # 4. MULTI-PERSPECTIVE / APPROACH COVERAGE (novel)
        # ============================================================
        # Detect signals that multiple angles/approaches are covered
        perspective_markers = [
            r'\bon the other hand\b', r'\balternatively\b', r'\banother\b',
            r'\bin contrast\b', r'\bhowever\b', r'\bconversely\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bdifferent\b',
            r'\bfirst\b.*\bsecond\b', r'\bone\b.*\banother\b',
            r'\bmultiple\b', r'\bseveral\b', r'\bvarious\b',
            r'\boption\b', r'\bapproach\b', r'\bmethod\b',
            r'\bperspective\b', r'\baspect\b', r'\bdimension\b',
            r'\badditionally\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\bnot only\b', r'\bbut also\b', r'\bbeyond\b',
        ]
        
        perspective_count = 0
        for pattern in perspective_markers:
            perspective_count += len(re.findall(pattern, response_lower))
        
        perspective_density = perspective_count / max(num_sentences, 1)
        perspective_score = min(perspective_density * 6, 10)
        score += perspective_score * 1.0
        
        # ============================================================
        # 5. STRUCTURAL COMPLETENESS PATTERN (novel: intro-body-conclusion)
        # ============================================================
        # Check for structural completeness signals
        
        # Opening/context-setting (first 20% of response)
        first_portion = response_lower[:max(len(response_lower) // 5, 50)]
        has_opening = any(re.search(p, first_portion) for p in [
            r'\b(certainly|sure|great|absolutely|yes|no|here)\b',
            r'\b(let me|let\'s|i\'ll|we can|to understand)\b',
            r'\b(overview|introduction|first|begin)\b',
        ])
        
        # Body with structured content (middle portion)
        has_enumeration = bool(re.search(r'(\d+[\.\):]|\*\s|-\s|•)', response))
        has_formatting = bool(re.search(r'(\*\*[^*]+\*\*|#{1,3}\s)', response))
        
        # Closing/summary signals (last 20% of response)
        last_portion = response_lower[max(len(response_lower) * 4 // 5, len(response_lower) - 200):]
        has_closing = any(re.search(p, last_portion) for p in [
            r'\b(overall|in summary|in conclusion|finally|remember|key)\b',
            r'\b(hope this|good luck|enjoy|happy|feel free)\b',
            r'\b(important|essential|crucial|keep in mind)\b',
        ])
        
        structural_score = 0
        if has_opening:
            structural_score += 2.5
        if has_enumeration:
            structural_score += 3.0
        if has_formatting:
            structural_score += 2.0
        if has_closing:
            structural_score += 2.5
        
        structural_score = min(structural_score, 10)
        score += structural_score * 0.8
        
        # ============================================================
        # 6. TOPIC SEGMENTATION BREADTH (novel: distinct topic clusters)
        # ============================================================
        # Divide response into chunks and measure topical diversity
        chunk_size = max(len(response_words) // 4, 10)
        chunks = []
        for i in range(0, len(response_words), chunk_size):
            chunk = response_words[i:i + chunk_size]
            chunks.append(set(w for w in chunk if w not in stopwords and len(w) > 2))
        
        if len(chunks) >= 2:
            # Measure average pairwise Jaccard distance between chunks
            distances = []
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    if chunks[i] or chunks[j]:
                        intersection = len(chunks[i] & chunks[j])
                        union = len(chunks[i] | chunks[j])
                        if union > 0:
                            jaccard_dist = 1 - (intersection / union)
                            distances.append(jaccard_dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                # Higher distance = more diverse topics covered
                topic_breadth_score = min(avg_distance * 12, 10)
            else:
                topic_breadth_score = 3
        else:
            topic_breadth_score = 2
        
        score += topic_breadth_score * 0.8
        
        # ============================================================
        # 7. QUANTITATIVE/SPECIFIC DETAIL SIGNALS (novel)
        # ============================================================
        # Count specific details: numbers, proper nouns, technical terms
        numbers = re.findall(r'\b\d+[\.\d]*\b', response)
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response)
        # Parenthetical clarifications (show thoroughness)
        parentheticals = re.findall(r'\([^)]+\)', response)
        # Quoted terms or emphasized terms
        emphasized = re.findall(r'["\'][^"\']+["\']', response)
        
        detail_count = len(numbers) + len(proper_nouns) * 0.5 + len(parentheticals) * 1.5 + len(emphasized) * 0.5
        detail_density = detail_count / max(num_sentences, 1)
        detail_score = min(detail_density * 5, 10)
        score += detail_score * 0.7
        
        # ============================================================
        # 8. RESPONSE LENGTH ADEQUACY (relative to query complexity)
        # ============================================================
        # Estimate query complexity
        query_complexity = len(query_words) + len(question_indicators) * 2
        
        # Expected minimum response length scales with query complexity
        expected_min_words = max(query_complexity * 3, 30)
        length_ratio = len(response_words) / expected_min_words
        
        # Sigmoid-like scaling: reward up to a point, diminishing returns
        if length_ratio < 0.5:
            length_score = length_ratio * 6
        elif length_ratio < 1.5:
            length_score = 3 + (length_ratio - 0.5) * 5
        elif length_ratio < 3:
            length_score = 8 + (length_ratio - 1.5) * 1.33
        else:
            length_score = 10
        
        length_score = min(max(length_score, 0), 10)
        score += length_score * 0.5
        
        # ============================================================
        # 9. EDGE CASE / CAVEAT AWARENESS (novel)
        # ============================================================
        caveat_patterns = [
            r'\bhowever\b', r'\bnote that\b', r'\bkeep in mind\b',
            r'\bbe aware\b', r'\bcaution\b', r'\bwarning\b',
            r'\bexception\b', r'\bunless\b', r'\bdepending on\b',
            r'\bin some cases\b', r'\bnot always\b', r'\bmay vary\b',
            r'\bimportant to\b', r'\bmake sure\b', r'\bensure\b',
            r'\bconsider\b', r'\btake into account\b', r'\bpotential\b',
            r'\brisk\b', r'\blimitation\b', r'\bdrawback\b',
            r'\bcaveat\b', r'\btrade-?off\b', r'\balthough\b',
        ]
        
        caveat_count = 0
        for pattern in caveat_patterns:
            caveat_count += len(re.findall(pattern, response_lower))
        
        caveat_score = min(caveat_count * 1.5, 10)
        score += caveat_score * 0.5
        
        # ============================================================
        # 10. ACTIONABILITY / CONCRETE GUIDANCE (novel metric)
        # ============================================================
        # Detect imperative/instructional language showing concrete guidance
        action_patterns = [
            r'\b(choose|select|pick|decide|determine)\b',
            r'\b(start|begin|proceed|continue|finish)\b',
            r'\b(add|mix|combine|place|put|set)\b',
            r'\b(check|verify|confirm|test|measure)\b',
            r'\b(use|apply|implement|create|make|build)\b',
            r'\b(avoid|prevent|reduce|minimize|eliminate)\b',
            r'\b(try|attempt|experiment|explore)\b',
            r'\b(need|require|must|essential)\b',
            r'\b(step|process|procedure|method|technique)\b',
        ]
        
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, response_lower))
        
        action_density = action_count / max(num_sentences, 1)
        action_score = min(action_density * 4, 10)
        score += action_score * 0.5
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Max possible raw: ~10*1.0 + 10*1.5 + 10*1.2 + 10*1.0 + 10*0.8 + 10*0.8 + 10*0.7 + 10*0.5 + 10*0.5 + 10*0.5
        # = 10 + 15 + 12 + 10 + 8 + 8 + 7 + 5 + 5 + 5 = 85
        max_raw = 85.0
        normalized = (score / max_raw) * 100.0
        
        return round(min(max(normalized, 0), 100), 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            return min(max(len(response.split()) * 0.1, 1), 50)
        except Exception:
            return 25.0