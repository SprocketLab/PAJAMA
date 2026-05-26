def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a 
    sentence-level semantic coverage approach combined with structural depth analysis.
    
    Algorithm: Query decomposition into semantic components + response sentence-level
    coverage mapping + depth/specificity scoring + coherence penalty.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        query = query.strip()
        response = response.strip()
        
        # === 1. Query Decomposition: Extract semantic "demands" from the query ===
        # Split query into question words, key nouns/verbs, and sub-questions
        
        def extract_question_aspects(text):
            """Extract what the query is asking about - the aspects that need covering."""
            text_lower = text.lower()
            
            # Detect question types
            question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which',
                            'can', 'could', 'should', 'would', 'is', 'are', 'do', 'does',
                            'explain', 'describe', 'identify', 'list', 'name', 'compare',
                            'create', 'write', 'rewrite', 'generate', 'make', 'find']
            
            found_qwords = [w for w in question_words if w in text_lower.split() or 
                           any(text_lower.startswith(w))]
            
            # Count sub-questions (multiple question marks, semicolons, "and", "also")
            num_questions = max(1, text_lower.count('?'))
            conjunctions = len(re.findall(r'\b(and|also|additionally|plus|as well|moreover)\b', text_lower))
            sub_parts = num_questions + conjunctions
            
            # Extract content words (nouns, verbs, adjectives - approximated by filtering stopwords)
            stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                        'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'could',
                        'would', 'please', 'also', 'make', 'want', 'know', 'need'}
            
            words = re.findall(r'[a-z]+', text_lower)
            content_words = [w for w in words if w not in stopwords and len(w) > 2]
            
            return {
                'question_types': found_qwords,
                'sub_parts': sub_parts,
                'content_words': content_words,
                'num_content_words': len(content_words)
            }
        
        query_aspects = extract_question_aspects(query)
        
        # === 2. Response Sentence Decomposition ===
        def split_into_sentences(text):
            """Split text into sentences."""
            # Handle various sentence endings
            sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
            return sentences
        
        response_sentences = split_into_sentences(response)
        num_sentences = len(response_sentences)
        
        # === 3. Content Word Coverage Score ===
        response_lower = response.lower()
        response_words = set(re.findall(r'[a-z]+', response_lower))
        
        query_content = query_aspects['content_words']
        if query_content:
            covered = sum(1 for w in query_content if w in response_words)
            # Also check for partial/synonym coverage via substring matching
            uncovered = [w for w in query_content if w not in response_words]
            partial_covered = 0
            for w in uncovered:
                if len(w) > 4 and any(w[:4] in rw for rw in response_words):
                    partial_covered += 0.5
            content_coverage = (covered + partial_covered) / len(query_content)
        else:
            content_coverage = 0.5  # neutral if no content words extracted
        
        # === 4. Response Depth & Specificity ===
        # Measure information density per sentence
        
        def sentence_info_density(sentence):
            """Estimate information content of a sentence."""
            words = re.findall(r'[a-z]+', sentence.lower())
            if not words:
                return 0
            
            stopwords_light = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'it', 'its', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                             'and', 'or', 'but', 'not', 'this', 'that'}
            
            content = [w for w in words if w not in stopwords_light and len(w) > 2]
            # Unique content ratio
            if not content:
                return 0
            unique_ratio = len(set(content)) / max(len(content), 1)
            # Longer content words suggest more specific/technical language
            avg_word_len = sum(len(w) for w in content) / len(content)
            specificity = min(avg_word_len / 8.0, 1.0)
            
            return (0.6 * unique_ratio + 0.4 * specificity)
        
        if response_sentences:
            densities = [sentence_info_density(s) for s in response_sentences]
            avg_density = sum(densities) / len(densities)
            # Count "substantive" sentences (above threshold)
            substantive_count = sum(1 for d in densities if d > 0.3)
        else:
            avg_density = 0
            substantive_count = 0
        
        # === 5. Detect if response is on-topic vs off-topic/garbage ===
        # Check for repetition (sign of low quality)
        def repetition_score(text):
            """Detect repetitive content. Returns 0 (no repetition) to 1 (fully repetitive)."""
            sentences = split_into_sentences(text)
            if len(sentences) <= 1:
                return 0
            
            # Check for duplicate sentences
            unique = set()
            duplicates = 0
            for s in sentences:
                normalized = re.sub(r'\s+', ' ', s.lower().strip())
                if normalized in unique:
                    duplicates += 1
                unique.add(normalized)
            
            dup_ratio = duplicates / len(sentences)
            
            # Check for repeated phrases (trigrams)
            words = re.findall(r'[a-z]+', text.lower())
            if len(words) >= 3:
                trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                trigram_counts = Counter(trigrams)
                if trigrams:
                    repeated = sum(1 for c in trigram_counts.values() if c > 2)
                    trigram_rep = repeated / max(len(trigram_counts), 1)
                else:
                    trigram_rep = 0
            else:
                trigram_rep = 0
            
            return min(1.0, dup_ratio * 0.6 + trigram_rep * 0.4)
        
        rep_score = repetition_score(response)
        
        # === 6. Detect garbage/irrelevant content ===
        # Signs of garbage: HTML tags where not expected, code where not expected, random characters
        garbage_indicators = 0
        
        # Check if query asks for code/HTML
        query_lower = query.lower()
        expects_code = any(w in query_lower for w in ['code', 'program', 'function', 'html', 'tag', 'script'])
        expects_html = any(w in query_lower for w in ['html', 'tag', 'webpage', 'web page'])
        
        if not expects_code:
            code_patterns = len(re.findall(r'(def |import |class |function\(|var |let |const )', response))
            if code_patterns > 2:
                garbage_indicators += 0.3
        
        if not expects_html:
            html_tags = len(re.findall(r'<[a-z]+[^>]*>', response.lower()))
            if html_tags > 3:
                garbage_indicators += 0.2
        
        # Check for "Output:" repetition pattern (seen in bad examples)
        output_pattern = len(re.findall(r'(?:Output|Input|Question|Answer):', response))
        if output_pattern > 3:
            garbage_indicators += 0.2
        
        # === 7. Length Appropriateness ===
        response_word_count = len(response.split())
        query_word_count = len(query.split())
        
        # Estimate expected complexity from query
        expected_complexity = query_aspects['sub_parts']
        
        # Very short responses are usually incomplete
        if response_word_count <= 2:
            length_score = 0.05
        elif response_word_count <= 5:
            length_score = 0.15
        elif response_word_count <= 10:
            length_score = 0.3
        elif response_word_count <= 20:
            length_score = 0.5
        elif response_word_count <= 50:
            length_score = 0.7
        elif response_word_count <= 150:
            length_score = 0.9
        elif response_word_count <= 300:
            length_score = 1.0
        else:
            # Very long responses may have padding/repetition
            length_score = max(0.6, 1.0 - (response_word_count - 300) / 2000)
        
        # But for simple queries, short answers can be fine
        simple_query_indicators = ['identify', 'name', 'which', 'biggest', 'largest', 'smallest']
        is_simple_query = any(w in query_lower for w in simple_query_indicators) and query_word_count < 15
        
        if is_simple_query and response_word_count >= 3:
            # For simple queries, don't penalize short answers as much
            length_score = max(length_score, 0.6)
        
        # === 8. Topical Relevance via Shared Vocabulary ===
        # Use Jaccard-like similarity but weighted by word importance
        query_words_set = set(re.findall(r'[a-z]{3,}', query_lower))
        response_words_set = set(re.findall(r'[a-z]{3,}', response_lower))
        
        if query_words_set:
            intersection = query_words_set & response_words_set
            # Weight by word length (longer words are more topically relevant)
            weighted_overlap = sum(len(w) for w in intersection)
            weighted_query = sum(len(w) for w in query_words_set)
            topical_relevance = weighted_overlap / weighted_query if weighted_query > 0 else 0
        else:
            topical_relevance = 0.5
        
        # === 9. Structural Completeness Indicators ===
        # Check for explanatory markers suggesting thorough coverage
        explanation_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhowever\b', r'\balthough\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bin addition\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnotably\b',
            r'\bincluding\b', r'\bas well as\b'
        ]
        
        explanation_count = sum(1 for pattern in explanation_markers 
                               if re.search(pattern, response_lower))
        explanation_score = min(1.0, explanation_count / 4.0)
        
        # === 10. Truncation Detection ===
        # Check if response appears cut off
        truncation_penalty = 0
        if response.rstrip()[-1:] not in '.!?")\']}>':
            # Doesn't end with proper punctuation
            last_sentence = response_sentences[-1] if response_sentences else response
            last_words = last_sentence.split()
            if last_words and len(last_words[-1]) < 4 and last_words[-1].isalpha():
                truncation_penalty = 0.1  # Mild penalty for apparent truncation
        
        # === 11. Combine All Signals ===
        # Weights chosen to emphasize completeness dimensions
        
        # Core completeness signals
        w_content_coverage = 0.20
        w_length = 0.18
        w_density = 0.12
        w_substantive = 0.10
        w_topical = 0.15
        w_explanation = 0.10
        w_repetition_penalty = 0.08
        w_garbage_penalty = 0.07
        
        # Normalize substantive count
        if expected_complexity > 0:
            substantive_ratio = min(1.0, substantive_count / max(expected_complexity * 2, 3))
        else:
            substantive_ratio = min(1.0, substantive_count / 3)
        
        raw_score = (
            w_content_coverage * content_coverage +
            w_length * length_score +
            w_density * avg_density +
            w_substantive * substantive_ratio +
            w_topical * topical_relevance +
            w_explanation * explanation_score
        )
        
        # Apply penalties
        repetition_penalty = w_repetition_penalty * rep_score
        garbage_penalty_val = w_garbage_penalty * garbage_indicators
        
        raw_score = raw_score - repetition_penalty - garbage_penalty_val - truncation_penalty
        
        # Ensure in [0, 1] range
        raw_score = max(0.0, min(1.0, raw_score))
        
        # === 12. Apply non-linear scaling for discrimination ===
        # Use sigmoid-like mapping to spread scores
        
        # Map to 0-10 scale
        score = raw_score * 10.0
        
        # Apply floor for extremely short/empty responses
        if response_word_count <= 2:
            score = min(score, 1.5)
        elif response_word_count <= 5 and content_coverage < 0.3:
            score = min(score, 2.5)
        
        # Boost for clearly comprehensive responses
        if (content_coverage > 0.7 and substantive_count >= 3 and 
            rep_score < 0.2 and garbage_indicators < 0.1):
            score = max(score, 5.0)
        
        # Cap responses with high garbage/repetition
        if rep_score > 0.5 or garbage_indicators > 0.4:
            score = min(score, 4.0)
        
        # Final clamp
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: basic length-based score
        try:
            words = len(response.split()) if response else 0
            if words == 0:
                return 0.5
            elif words <= 3:
                return 1.5
            elif words <= 10:
                return 3.0
            elif words <= 50:
                return 5.0
            else:
                return 6.0
        except:
            return 3.0