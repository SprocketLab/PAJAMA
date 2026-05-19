def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a
    topic-threading and structural depth analysis approach.
    
    This variant focuses on:
    1. Query decomposition - identifying distinct question components/aspects
    2. Response threading - measuring how many query aspects are addressed
    3. Explanation depth via clause density and elaboration patterns
    4. Information density via unique concept introduction rate
    5. Reasoning chain detection (causal/logical connectors forming chains)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 10:
            return 0.5
        
        # ============================================================
        # 1. QUERY DECOMPOSITION: Extract distinct aspects/sub-questions
        # ============================================================
        
        # Extract question marks as explicit sub-questions
        explicit_questions = re.findall(r'[^.!?]*\?', query)
        num_explicit_questions = max(len(explicit_questions), 1)
        
        # Extract query topic words (nouns, verbs, key terms)
        # Remove common stop words to find substantive query terms
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
            'they', 'them', 'this', 'that', 'these', 'those', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'shall', 'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'not', 'so',
            'if', 'then', 'than', 'too', 'very', 'just', 'about', 'above',
            'after', 'again', 'all', 'also', 'am', 'any', 'as', 'at', 'back',
            'because', 'before', 'between', 'both', 'by', 'came', 'come',
            'down', 'each', 'for', 'from', 'get', 'got', 'go', 'going', 'here',
            'her', 'him', 'his', 'how', 'in', 'into', 'its', 'know', 'like',
            'make', 'many', 'more', 'most', 'much', 'must', 'new', 'no', 'now',
            'of', 'on', 'one', 'only', 'or', 'other', 'out', 'over', 'own',
            'said', 'same', 'see', 'some', 'still', 'such', 'take', 'tell',
            'their', 'there', 'to', 'up', 'us', 'use', 'want', 'way', 'what',
            'when', 'where', 'which', 'while', 'who', 'why', 'with', 'been',
            'im', 'ive', 'dont', 'doesnt', 'didnt', 'wont', 'cant', 'couldnt',
            'shouldnt', 'wouldnt', 've', 're', 'll', 'd', 's', 't', 'm',
        }
        
        def tokenize(text):
            return re.findall(r"[a-z][a-z']*", text.lower())
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        query_content_words = [w for w in query_tokens if w not in stop_words and len(w) > 2]
        response_content_words = [w for w in response_tokens if w not in stop_words and len(w) > 2]
        
        # Identify query "aspect clusters" using bigrams from the query
        query_bigrams = set()
        for i in range(len(query_content_words) - 1):
            query_bigrams.add((query_content_words[i], query_content_words[i+1]))
        
        # ============================================================
        # 2. QUERY ASPECT COVERAGE (threading)
        # ============================================================
        
        query_content_set = set(query_content_words)
        response_content_set = set(response_content_words)
        
        if len(query_content_set) > 0:
            # What fraction of unique query content words appear in response
            word_coverage = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            word_coverage = 0.5
        
        # Check bigram coverage
        if len(query_bigrams) > 0:
            response_bigrams = set()
            for i in range(len(response_content_words) - 1):
                response_bigrams.add((response_content_words[i], response_content_words[i+1]))
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.0
        
        # Combined topic coverage score
        topic_coverage_score = 0.6 * word_coverage + 0.4 * bigram_coverage
        
        # ============================================================
        # 3. CLAUSE DENSITY & ELABORATION DEPTH
        # ============================================================
        # Count subordinate clauses and elaboration markers as indicators
        # of depth of explanation
        
        elaboration_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bconsequently\b',
            r'\bas a result\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\binstead\b',
            r'\bmeanwhile\b', r'\blikewise\b', r'\bsimilarly\b',
            r'\bin fact\b', r'\bindeed\b', r'\bcertainly\b',
            r'\bultimately\b', r'\bessentially\b', r'\bbasically\b',
        ]
        
        response_lower = response.lower()
        elaboration_count = 0
        for pattern in elaboration_markers:
            elaboration_count += len(re.findall(pattern, response_lower))
        
        # Count commas as rough proxy for clause complexity
        comma_count = response.count(',')
        semicolon_count = response.count(';')
        colon_count = response.count(':')
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Clauses per sentence (using commas, semicolons as splitters)
        clauses_per_sentence = (comma_count + semicolon_count + num_sentences) / num_sentences
        
        # Elaboration density (elaboration markers per sentence)
        elaboration_density = elaboration_count / num_sentences
        
        # Depth score: combines clause complexity and elaboration
        depth_score = min(1.0, (clauses_per_sentence - 1.0) / 3.0) * 0.4 + \
                      min(1.0, elaboration_density / 0.8) * 0.6
        
        # ============================================================
        # 4. INFORMATION DENSITY: Unique concept introduction rate
        # ============================================================
        # Measure how many NEW unique content words are introduced as
        # we progress through the response (vs repetition)
        
        if len(response_content_words) > 0:
            seen = set()
            new_introductions = []
            window = max(10, len(response_content_words) // 5)
            
            for i, w in enumerate(response_content_words):
                if w not in seen:
                    new_introductions.append(1)
                    seen.add(w)
                else:
                    new_introductions.append(0)
            
            # Overall type-token ratio
            ttr = len(set(response_content_words)) / len(response_content_words)
            
            # Measure if new concepts keep being introduced in later portions
            # Split response into thirds
            third = max(1, len(new_introductions) // 3)
            first_third_rate = sum(new_introductions[:third]) / max(third, 1)
            last_third_rate = sum(new_introductions[2*third:]) / max(third, 1)
            
            # Sustained novelty: if last third still introduces concepts, 
            # the response is covering more ground
            sustained_novelty = last_third_rate / max(first_third_rate, 0.01)
            sustained_novelty = min(sustained_novelty, 1.0)
            
            info_density_score = 0.4 * ttr + 0.3 * sustained_novelty + \
                                 0.3 * min(1.0, len(set(response_content_words)) / 50.0)
        else:
            info_density_score = 0.0
        
        # ============================================================
        # 5. REASONING CHAIN DETECTION
        # ============================================================
        # Detect sequences of causal/logical reasoning
        
        causal_patterns = [
            r'\bif\b.*?\bthen\b', r'\bbecause\b.*?,', r'\bthis means\b',
            r'\bwhich leads to\b', r'\bas a result\b', r'\btherefore\b',
            r'\bconsequently\b', r'\bso that\b', r'\bin order to\b',
            r'\bthe reason\b', r'\bdue to\b', r'\bthis is why\b',
            r'\bimplies that\b', r'\bit follows\b', r'\bgiven that\b',
        ]
        
        reasoning_count = 0
        for pattern in causal_patterns:
            reasoning_count += len(re.findall(pattern, response_lower))
        
        reasoning_score = min(1.0, reasoning_count / 4.0)
        
        # ============================================================
        # 6. STRUCTURAL COMPLETENESS SIGNALS
        # ============================================================
        
        # Check for multiple distinct points/perspectives
        # Count sentence-initial discourse markers that signal new points
        new_point_markers = [
            r'(?:^|\n)\s*(?:first|second|third|fourth|finally|lastly|additionally|also|another|next)',
            r'(?:^|\n)\s*(?:\d+[\.\)]\s)',
            r'(?:^|\n)\s*[-*•]\s',
            r'(?:^|\n)\s*(?:on one hand|on the other|alternatively|conversely)',
        ]
        
        point_count = 0
        for pattern in new_point_markers:
            point_count += len(re.findall(pattern, response_lower))
        
        multi_point_score = min(1.0, point_count / 4.0)
        
        # ============================================================
        # 7. RESPONSE LENGTH CALIBRATION
        # ============================================================
        # Longer responses tend to be more complete, but with diminishing returns
        
        char_len = len(response)
        word_count = len(response_tokens)
        
        # Sigmoid-like length score
        # Calibrated so ~200 words gets ~0.7, ~400 words gets ~0.9
        length_score = 1.0 - math.exp(-word_count / 150.0)
        length_score = max(0.0, min(1.0, length_score))
        
        # Penalty for very short responses
        if word_count < 20:
            length_penalty = word_count / 20.0
        elif word_count < 50:
            length_penalty = 0.7 + 0.3 * (word_count - 20) / 30.0
        else:
            length_penalty = 1.0
        
        # ============================================================
        # 8. SPECIFICITY & CONCRETENESS (different approach: named entities, numbers, quotes)
        # ============================================================
        
        # Count specific evidence types
        number_count = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', response))
        quoted_count = len(re.findall(r'["\*].*?["\*]', response))
        # Capitalized multi-word phrases (likely proper nouns/names)
        proper_nouns = re.findall(r'(?<!\. )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response)
        
        specificity_signals = number_count + quoted_count + len(proper_nouns)
        specificity_score = min(1.0, specificity_signals / 6.0)
        
        # ============================================================
        # 9. QUESTION-TYPE ALIGNMENT
        # ============================================================
        # Check if response format matches query type
        
        query_lower = query.lower()
        
        # Is it a "how" question? Response should have procedural language
        is_how = 'how' in query_lower[:50]
        if is_how:
            procedural_words = len(re.findall(
                r'\b(?:step|first|then|next|after|before|start|begin|process|method|way)\b',
                response_lower
            ))
            alignment_bonus = min(0.15, procedural_words * 0.03)
        # Is it a "why" question? Response should have explanatory language
        elif 'why' in query_lower[:50]:
            explanatory_words = len(re.findall(
                r'\b(?:because|reason|cause|due|since|therefore|result|led|origin)\b',
                response_lower
            ))
            alignment_bonus = min(0.15, explanatory_words * 0.03)
        # Is it asking for comparison/contrast?
        elif any(w in query_lower for w in ['difference', 'compare', 'versus', 'vs', 'or']):
            comparison_words = len(re.findall(
                r'\b(?:whereas|while|however|contrast|unlike|similar|different|both|neither)\b',
                response_lower
            ))
            alignment_bonus = min(0.15, comparison_words * 0.03)
        else:
            alignment_bonus = 0.0
        
        # ============================================================
        # 10. TRUNCATION PENALTY
        # ============================================================
        # Check if response appears truncated (ends mid-sentence)
        
        truncation_penalty = 0.0
        last_chars = response[-10:] if len(response) >= 10 else response
        if not re.search(r'[.!?"\)}\]]\s*$', response):
            # Doesn't end with sentence-ending punctuation
            truncation_penalty = 0.05
        
        # ============================================================
        # FINAL SCORE COMPOSITION
        # ============================================================
        
        # Weighted combination emphasizing completeness-related features
        raw_score = (
            topic_coverage_score * 1.5 +    # How well query topics are addressed
            depth_score * 2.0 +              # Depth of explanation
            info_density_score * 2.0 +       # Breadth of information
            reasoning_score * 1.5 +          # Logical reasoning chains
            multi_point_score * 1.5 +        # Multiple distinct points
            specificity_score * 1.0 +        # Concrete details
            length_score * 1.5 +             # Sufficient length
            alignment_bonus * 5.0            # Query-type alignment (already small)
        )
        
        # Apply length penalty and truncation penalty
        raw_score *= length_penalty
        raw_score -= truncation_penalty
        
        # Normalize: max possible raw ~11.75, target range 0-10
        max_possible = 1.5 + 2.0 + 2.0 + 1.5 + 1.5 + 1.0 + 1.5 + 0.75
        normalized = (raw_score / max_possible) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: use simple length heuristic
        try:
            return min(5.0, len(str(response)) / 200.0)
        except Exception:
            return 0.0