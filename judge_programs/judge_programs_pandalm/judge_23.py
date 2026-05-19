def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response.
    
    This variant uses a STRUCTURAL DECOMPOSITION approach:
    1. Decomposes the query into interrogative components and implied sub-tasks
    2. Measures information density via unique information units (clause-level)
    3. Checks for elaboration patterns (explanations, examples, qualifications)
    4. Penalizes repetition and hollow filler content
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
        
        response_words = response.lower().split()
        query_words = query.lower().split()
        
        if len(response_words) == 0:
            return 0.0
        
        # ============================================================
        # 1. QUERY DECOMPOSITION: Count implied sub-tasks/sub-questions
        # ============================================================
        
        # Count explicit sub-questions (question marks, question words)
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whom', 'whose'}
        query_lower = query.lower()
        
        # Count question marks
        num_questions = query.count('?')
        
        # Count imperative verbs suggesting sub-tasks
        task_verbs = {
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'give', 'write', 'create', 'generate', 'discuss', 'analyze',
            'evaluate', 'summarize', 'identify', 'define', 'outline',
            'rewrite', 'convert', 'translate', 'classify', 'categorize',
            'suggest', 'recommend', 'show', 'demonstrate', 'illustrate',
            'crop', 'reduce', 'add', 'remove', 'modify', 'come up'
        }
        
        num_tasks = 0
        for verb in task_verbs:
            if verb in query_lower:
                num_tasks += 1
        
        # Count conjunctions suggesting multiple parts: "and", commas in query
        num_conjunctions = query_lower.count(' and ') + query_lower.count(',')
        
        # Estimate query complexity (number of expected sub-answers)
        query_complexity = max(1, num_questions + num_tasks + num_conjunctions * 0.5)
        query_complexity = min(query_complexity, 8)  # cap
        
        # ============================================================
        # 2. CLAUSE-LEVEL INFORMATION UNITS
        # ============================================================
        
        # Split response into clauses (by punctuation and conjunctions)
        clause_splitters = r'[.!?;]|\band\b|\bbut\b|\bhowever\b|\bwhile\b|\bwhereas\b|\balso\b|\bfurthermore\b|\bmoreover\b|\bin addition\b|\bwhich\b|\bthat\b'
        clauses = re.split(clause_splitters, response.lower())
        clauses = [c.strip() for c in clauses if c and len(c.strip()) > 3]
        
        num_clauses = len(clauses)
        
        # Measure uniqueness of clauses using Jaccard distance between consecutive clauses
        if num_clauses > 1:
            unique_clause_score = 0
            for i in range(len(clauses)):
                clause_words_i = set(clauses[i].split())
                is_unique = True
                for j in range(len(clauses)):
                    if i != j:
                        clause_words_j = set(clauses[j].split())
                        if len(clause_words_i) > 0 and len(clause_words_j) > 0:
                            intersection = len(clause_words_i & clause_words_j)
                            union = len(clause_words_i | clause_words_j)
                            jaccard = intersection / union if union > 0 else 0
                            if jaccard > 0.8:
                                is_unique = False
                                break
                if is_unique:
                    unique_clause_score += 1
            clause_uniqueness_ratio = unique_clause_score / num_clauses
        else:
            clause_uniqueness_ratio = 0.5 if num_clauses == 1 else 0.0
        
        # ============================================================
        # 3. REPETITION PENALTY (word-level and phrase-level)
        # ============================================================
        
        # Word repetition: ratio of unique words to total words
        word_counts = Counter(response_words)
        unique_words = len(word_counts)
        total_words = len(response_words)
        
        # Filter out common stop words for repetition analysis
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'than', 'too', 'very',
            'just', 'also', 'it', 'its', 'this', 'that', 'these', 'those', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'they', 'them', 'their', 'which', 'who', 'whom'
        }
        
        content_words = [w for w in response_words if w not in stop_words and len(w) > 2]
        content_counts = Counter(content_words)
        
        if len(content_words) > 0:
            # Check for extreme repetition of any single content word
            max_content_freq = max(content_counts.values()) if content_counts else 0
            repetition_ratio = max_content_freq / len(content_words)
            repetition_penalty = max(0, (repetition_ratio - 0.15) * 3)  # penalize if >15% is one word
        else:
            repetition_penalty = 0.5
        
        # Bigram repetition check
        if len(response_words) >= 2:
            bigrams = [f"{response_words[i]} {response_words[i+1]}" for i in range(len(response_words)-1)]
            bigram_counts = Counter(bigrams)
            # Filter to content bigrams
            content_bigrams = {bg: c for bg, c in bigram_counts.items() 
                             if not all(w in stop_words for w in bg.split())}
            if content_bigrams:
                max_bigram_freq = max(content_bigrams.values())
                if max_bigram_freq > 3:
                    repetition_penalty += min(2.0, (max_bigram_freq - 3) * 0.3)
        
        # ============================================================
        # 4. ELABORATION PATTERNS
        # ============================================================
        
        response_lower = response.lower()
        
        # Example indicators
        example_markers = [
            'for example', 'for instance', 'such as', 'e.g.', 'like ',
            'including', 'specifically', 'in particular', 'namely'
        ]
        num_examples = sum(1 for m in example_markers if m in response_lower)
        
        # Explanation indicators
        explanation_markers = [
            'because', 'since', 'therefore', 'thus', 'this means',
            'in other words', 'that is', 'i.e.', 'as a result',
            'consequently', 'due to', 'the reason', 'suggests that',
            'implies that', 'indicates that'
        ]
        num_explanations = sum(1 for m in explanation_markers if m in response_lower)
        
        # Qualification/nuance indicators
        nuance_markers = [
            'however', 'although', 'while', 'on the other hand', 'nevertheless',
            'despite', 'in contrast', 'conversely', 'whereas', 'but ',
            'yet ', 'still ', 'even though', 'regardless'
        ]
        num_nuances = sum(1 for m in nuance_markers if m in response_lower)
        
        # Structural completeness markers
        structure_markers = [
            'first', 'second', 'third', 'finally', 'additionally',
            'furthermore', 'moreover', 'in addition', 'also ',
            'another', 'lastly', 'in conclusion', 'to summarize',
            'overall', 'in summary'
        ]
        num_structure = sum(1 for m in structure_markers if m in response_lower)
        
        elaboration_score = (
            min(num_examples, 3) * 1.0 +
            min(num_explanations, 3) * 0.8 +
            min(num_nuances, 3) * 0.7 +
            min(num_structure, 4) * 0.5
        )
        
        # ============================================================
        # 5. QUERY-RESPONSE ALIGNMENT (semantic coverage)
        # ============================================================
        
        # Extract key concepts from query (non-stop content words)
        query_content = [w for w in query_words if w.lower() not in stop_words and len(w) > 2]
        query_content = [re.sub(r'[^\w]', '', w) for w in query_content]
        query_content = [w for w in query_content if w]
        
        if query_content:
            response_word_set = set(response_words)
            # Check how many query concepts appear in response
            covered = sum(1 for w in query_content if w in response_word_set)
            query_coverage = covered / len(query_content)
        else:
            query_coverage = 0.5
        
        # ============================================================
        # 6. RESPONSE LENGTH APPROPRIATENESS
        # ============================================================
        
        # Adaptive length scoring based on query complexity
        expected_min_words = max(10, query_complexity * 12)
        expected_ideal_words = max(30, query_complexity * 30)
        
        if total_words < 5:
            length_score = 0.1
        elif total_words < expected_min_words:
            length_score = 0.3 + 0.4 * (total_words / expected_min_words)
        elif total_words < expected_ideal_words:
            length_score = 0.7 + 0.3 * ((total_words - expected_min_words) / 
                                          max(1, expected_ideal_words - expected_min_words))
        else:
            # Diminishing returns for very long responses
            length_score = 1.0
        
        # ============================================================
        # 7. SENTENCE COUNT AND VARIETY
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        # Variety in sentence length (standard deviation)
        if num_sentences > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            sent_length_std = math.sqrt(variance)
            # Some variety is good; normalize
            sentence_variety = min(1.0, sent_length_std / 8.0)
        else:
            sentence_variety = 0.0
        
        # ============================================================
        # 8. SPECIFICITY: presence of specific details
        # ============================================================
        
        # Check for numbers, proper nouns (capitalized words mid-sentence), 
        # technical terms (longer words)
        has_numbers = len(re.findall(r'\b\d+\b', response)) > 0
        
        # Words longer than 8 chars (likely more specific/technical)
        long_words = [w for w in content_words if len(w) > 8]
        specificity_ratio = len(long_words) / max(1, len(content_words))
        
        specificity_score = (
            (0.3 if has_numbers else 0.0) +
            min(0.7, specificity_ratio * 3)
        )
        
        # ============================================================
        # 9. TRUNCATION DETECTION
        # ============================================================
        
        truncation_penalty = 0.0
        # Check if response seems cut off
        if response[-1] not in '.!?"\')' and len(response) > 50:
            truncation_penalty = 0.5
        # Check for incomplete last sentence
        last_sentence = sentences[-1] if sentences else ""
        if last_sentence and len(last_sentence.split()) < 3 and len(sentences) > 1:
            truncation_penalty += 0.2
        
        # ============================================================
        # 10. EMPTINESS / NOINPUT DETECTION
        # ============================================================
        
        emptiness_penalty = 0.0
        noinput_patterns = ['<noinput>', 'noinput', 'n/a', 'none', 'no input']
        if any(p in response_lower for p in noinput_patterns) and total_words < 5:
            emptiness_penalty = 5.0
        
        # ============================================================
        # FINAL SCORE COMPOSITION
        # ============================================================
        
        # Base score from information content
        clause_score = min(1.0, num_clauses / max(2, query_complexity * 1.5))
        
        score = (
            # Core completeness metrics (0-10 scale contributions)
            clause_score * clause_uniqueness_ratio * 2.5 +       # unique info units: 0-2.5
            length_score * 2.0 +                                   # appropriate length: 0-2.0
            query_coverage * 1.5 +                                 # query concept coverage: 0-1.5
            elaboration_score * 0.4 +                              # elaboration depth: 0-~3.0
            min(1.0, num_sentences / max(2, query_complexity)) * 1.0 +  # sentence count: 0-1.0
            sentence_variety * 0.5 +                               # sentence variety: 0-0.5
            specificity_score * 0.8 +                              # specificity: 0-0.8
            
            # Penalties
            - repetition_penalty * 1.5 +                           # repetition: 0-~3.0
            - truncation_penalty * 1.0 +                           # truncation: 0-0.7
            - emptiness_penalty                                     # emptiness: 0-5.0
        )
        
        # Clamp to 0-10
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
        
    except Exception:
        # Fallback: simple length-based score
        try:
            words = len(response.split()) if response else 0
            return min(5.0, words * 0.1)
        except Exception:
            return 0.0