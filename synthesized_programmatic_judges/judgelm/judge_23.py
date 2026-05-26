def judging_function(query, response):
    """
    Evaluates completeness and coverage using a sentence-level semantic coverage approach.
    
    Algorithm: Decomposes the query into constituent "information needs" (approximated by
    extracting question words, noun phrases, and key content tokens), then measures how
    many of these needs are addressed in the response through sentence-level coverage
    analysis. Also evaluates structural completeness via sentence count diversity,
    topic transition detection, and information density metrics.
    
    This is fundamentally different from:
    - paragraph analysis (not counting paragraphs)
    - bullet/list detection (not looking for bullets)
    - word overlap / n-gram (not using n-gram matching)
    - word length / Jaccard similarity (not using set similarity)
    
    Instead uses: query decomposition into information needs + sentence-level coverage mapping
    + information density + topic transition analysis
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 2:
            return 0.0
        
        # --- Step 1: Extract "information needs" from the query ---
        # Split query into sub-questions (by ?, newline, semicolon, "and", "also")
        sub_questions = re.split(r'[?\n;]|(?:\band\b)|(?:\balso\b)', query.lower())
        sub_questions = [sq.strip() for sq in sub_questions if len(sq.strip()) > 3]
        num_sub_questions = max(len(sub_questions), 1)
        
        # Extract content words from query (nouns, verbs, adjectives - approximated by
        # filtering out stopwords and short words)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'it', 'its', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'make', 'please', 'also', 'get', 'got', 'am', 'im',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        query_content_words = extract_content_words(query)
        query_content_set = set(query_content_words)
        
        # --- Step 2: Split response into sentences ---
        # Use multiple delimiters for sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+|(?:\n\s*\n)|\n(?=[A-Z])', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5
        
        # --- Step 3: Measure sub-question coverage ---
        # For each sub-question, check if at least one sentence addresses it
        def sub_question_covered(sq, sentences_lower):
            sq_words = extract_content_words(sq)
            if not sq_words:
                return 1.0  # trivial sub-question
            
            best_coverage = 0.0
            for sent in sentences_lower:
                sent_words = set(re.findall(r'[a-z]+', sent))
                matched = sum(1 for w in sq_words if w in sent_words)
                coverage = matched / len(sq_words)
                best_coverage = max(best_coverage, coverage)
                if best_coverage >= 0.5:
                    # Also check across multiple sentences combined
                    break
            
            # Also check combined text
            all_resp_words = set(re.findall(r'[a-z]+', ' '.join(sentences_lower)))
            combined_coverage = sum(1 for w in sq_words if w in all_resp_words) / len(sq_words)
            
            # Weight: sentence-level coverage matters but combined also counts
            return 0.4 * best_coverage + 0.6 * combined_coverage
        
        sentences_lower = [s.lower() for s in sentences]
        
        sq_coverages = [sub_question_covered(sq, sentences_lower) for sq in sub_questions]
        avg_sq_coverage = sum(sq_coverages) / len(sq_coverages) if sq_coverages else 0.5
        
        # --- Step 4: Topic transition / breadth analysis ---
        # Count how many distinct "topic clusters" appear in the response
        # by looking at which sentences introduce new content words
        seen_content = set()
        new_topic_introductions = 0
        for sent in sentences_lower:
            sent_content = set(extract_content_words(sent))
            new_words = sent_content - seen_content
            if len(new_words) >= 2:
                new_topic_introductions += 1
            seen_content.update(sent_content)
        
        # Normalize: more topic introductions = broader coverage
        topic_breadth = min(new_topic_introductions / max(num_sub_questions, 2), 2.0)
        topic_breadth_score = min(topic_breadth, 1.0)
        
        # --- Step 5: Information density ---
        # Ratio of unique content words to total words (penalizes repetition)
        resp_words = re.findall(r'[a-z]+', response.lower())
        resp_content_words = [w for w in resp_words if w not in stopwords and len(w) > 2]
        
        if len(resp_words) == 0:
            return 0.5
        
        unique_content = set(resp_content_words)
        total_words = len(resp_words)
        
        # Information density: unique content words / total words
        info_density = len(unique_content) / max(total_words, 1)
        # Ideal range is 0.3-0.6; too low = repetitive, too high = might be keyword spam
        density_score = 1.0 - abs(info_density - 0.4) * 2
        density_score = max(0.0, min(1.0, density_score))
        
        # --- Step 6: Repetition penalty ---
        # Detect repeated phrases (3+ word sequences appearing multiple times)
        trigrams = []
        for i in range(len(resp_words) - 2):
            trigrams.append(tuple(resp_words[i:i+3]))
        
        trigram_counts = Counter(trigrams)
        if trigrams:
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            repetition_ratio = repeated_trigrams / max(len(set(trigrams)), 1)
        else:
            repetition_ratio = 0.0
        
        repetition_penalty = min(repetition_ratio * 3, 1.0)
        
        # --- Step 7: Response substantiveness ---
        # Check if response actually contains substantive content vs filler
        filler_patterns = [
            r'\b(um|uh|well|okay|ok|sure)\b',
            r'^(yes|no|maybe)[.!]?$',
        ]
        
        is_trivial = len(response.strip()) < 20
        trivial_penalty = 0.0
        if is_trivial:
            trivial_penalty = 0.5
        
        # Check for very short non-answers
        resp_stripped = re.sub(r'[^a-z\s]', '', response.lower()).strip()
        if len(resp_stripped.split()) <= 3:
            trivial_penalty = 0.7
        
        # --- Step 8: Length appropriateness ---
        # Longer responses tend to be more complete, but with diminishing returns
        # Use log scale
        word_count = len(resp_words)
        length_score = min(math.log(max(word_count, 1) + 1) / math.log(200), 1.0)
        
        # --- Step 9: Relevance check ---
        # Ensure response is actually about the query topic
        resp_content_set = set(resp_content_words)
        if query_content_set:
            query_words_in_resp = sum(1 for w in query_content_set if w in resp_content_set)
            relevance = query_words_in_resp / len(query_content_set)
        else:
            relevance = 0.5
        
        # --- Step 10: Detect garbage/off-topic content ---
        # Check for code dumps, HTML, repeated patterns that suggest broken output
        code_pattern_count = len(re.findall(r'(import |def |class |<[a-z]+>|</[a-z]+>|\{|\}|=>|==)', response))
        # Only penalize if query doesn't seem to ask for code/HTML
        query_asks_code = bool(re.search(r'(code|program|html|script|function|implement|write a)', query.lower()))
        
        garbage_penalty = 0.0
        if not query_asks_code and code_pattern_count > 5:
            garbage_penalty = min(code_pattern_count * 0.05, 0.4)
        
        # Check for excessive repetition of lines
        lines = response.split('\n')
        if len(lines) > 3:
            line_counter = Counter(line.strip().lower() for line in lines if len(line.strip()) > 10)
            max_line_repeat = max(line_counter.values()) if line_counter else 1
            if max_line_repeat > 2:
                garbage_penalty += min((max_line_repeat - 2) * 0.1, 0.3)
        
        # --- Step 11: Structural completeness signals ---
        # Does the response have an introduction/conclusion pattern?
        has_structure = 0.0
        if num_sentences >= 3:
            has_structure += 0.3
        if num_sentences >= 5:
            has_structure += 0.2
        # Check for transitional/connective words suggesting organized thought
        connectives = re.findall(
            r'\b(however|therefore|furthermore|additionally|moreover|first|second|third|'
            r'finally|in conclusion|for example|for instance|specifically|in addition|'
            r'on the other hand|as a result|consequently|nevertheless)\b',
            response.lower()
        )
        has_structure += min(len(connectives) * 0.1, 0.3)
        has_structure = min(has_structure, 1.0)
        
        # --- Step 12: Truncation detection ---
        # Check if response appears to be cut off mid-sentence
        truncation_penalty = 0.0
        last_chars = response.strip()[-5:] if len(response.strip()) >= 5 else response.strip()
        if not re.search(r'[.!?"\']$', last_chars):
            # Might be truncated
            truncation_penalty = 0.1
        
        # --- Combine all signals ---
        # Weights emphasize coverage and relevance for "completeness" evaluation
        score = (
            avg_sq_coverage * 3.0 +        # Sub-question coverage (0-3)
            topic_breadth_score * 1.5 +     # Topic breadth (0-1.5)
            density_score * 0.8 +           # Information density (0-0.8)
            length_score * 1.5 +            # Length appropriateness (0-1.5)
            relevance * 1.5 +               # Relevance to query (0-1.5)
            has_structure * 0.7 -            # Structural organization (0-0.7)
            repetition_penalty * 1.5 -      # Repetition penalty
            trivial_penalty * 3.0 -         # Trivial response penalty
            garbage_penalty * 2.0 -         # Garbage content penalty
            truncation_penalty * 0.5        # Truncation penalty
        )
        
        # Normalize to 0-10 scale
        # Max possible ≈ 3 + 1.5 + 0.8 + 1.5 + 1.5 + 0.7 = 9.0
        # Min possible ≈ 0 - 1.5 - 3.0 - 0.8 - 0.5 = -5.8
        
        # Map to 0-10
        score = max(0.0, score)
        score = (score / 9.0) * 10.0
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: basic length-based score
        try:
            if not response or len(response.strip()) < 3:
                return 0.0
            words = len(response.split())
            return min(max(words / 20.0, 0.5), 5.0)
        except Exception:
            return 1.0