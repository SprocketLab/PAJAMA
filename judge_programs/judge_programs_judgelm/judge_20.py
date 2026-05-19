def judging_function(query, response):
    """
    Evaluates completeness and coverage using a semantic overlap and structural depth analysis.
    
    This variant uses:
    - Query decomposition into question components/aspects
    - Information density measurement via unique concept coverage
    - Sentence-level diversity and topic spread analysis
    - Depth scoring through explanation patterns and elaboration detection
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        query_clean = query.strip()
        response_clean = response.strip()
        
        # === 1. Query Aspect Extraction ===
        # Extract key concepts/aspects from the query that should be addressed
        
        # Common stop words to filter out
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'it',
            'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'also', 'make',
            'please', 'tell', 'give', 'know', 'want', 'get', 'like',
        }
        
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def content_words(text):
            tokens = tokenize(text)
            return [t for t in tokens if t not in stop_words and len(t) > 1]
        
        query_content = content_words(query_clean)
        response_content = content_words(response_clean)
        response_tokens = tokenize(response_clean)
        
        # === 2. Query Concept Coverage ===
        # What fraction of query's key concepts appear in the response?
        query_concepts = set(query_content)
        response_word_set = set(response_content)
        
        if query_concepts:
            concept_coverage = len(query_concepts & response_word_set) / len(query_concepts)
        else:
            concept_coverage = 0.5
        
        # === 3. Bigram/trigram overlap for phrase-level coverage ===
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        query_bigrams = set(get_ngrams(tokenize(query_clean.lower()), 2))
        response_bigrams = set(get_ngrams(tokenize(response_clean.lower()), 2))
        
        if query_bigrams:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.0
        
        # === 4. Information Density: Unique concepts per sentence ===
        response_sentences = re.split(r'[.!?]+', response_clean)
        response_sentences = [s.strip() for s in response_sentences if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(response_sentences), 1)
        
        unique_concepts_per_sentence = []
        all_response_concepts = set()
        new_concepts_per_sentence = []
        
        for sent in response_sentences:
            sent_concepts = set(content_words(sent))
            unique_concepts_per_sentence.append(len(sent_concepts))
            new_info = sent_concepts - all_response_concepts
            new_concepts_per_sentence.append(len(new_info))
            all_response_concepts.update(sent_concepts)
        
        # Measure how many sentences contribute NEW information (not just repeating)
        if new_concepts_per_sentence:
            informative_sentences = sum(1 for nc in new_concepts_per_sentence if nc >= 2)
            info_progression_ratio = informative_sentences / len(new_concepts_per_sentence)
        else:
            info_progression_ratio = 0.0
        
        total_unique_concepts = len(all_response_concepts)
        
        # === 5. Detect sub-questions in query ===
        question_marks = query_clean.count('?')
        # Also detect implicit sub-questions via conjunctions and commas in query
        query_parts = re.split(r'[,;]|\band\b|\balso\b|\badditionally\b', query_clean.lower())
        query_parts = [p.strip() for p in query_parts if len(p.strip()) > 3]
        num_query_aspects = max(len(query_parts), max(question_marks, 1))
        
        # Check how many query parts have some content word overlap with response
        parts_addressed = 0
        for part in query_parts:
            part_concepts = set(content_words(part))
            if part_concepts:
                overlap = len(part_concepts & response_word_set)
                if overlap >= 1:
                    parts_addressed += 1
            else:
                parts_addressed += 0.5
        
        if len(query_parts) > 0:
            aspect_coverage = parts_addressed / len(query_parts)
        else:
            aspect_coverage = 0.5
        
        # === 6. Depth indicators: explanations, examples, reasoning ===
        depth_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bthis means\b', r'\bin other words\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bhowever\b', r'\balthough\b', r'\bon the other hand\b', r'\bwhile\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bincluding\b', r'\bnote that\b', r'\bimportant\b', r'\bkey\b',
            r'\breason\b', r'\bexplain\b', r'\bconsider\b', r'\bcontext\b',
        ]
        
        response_lower = response_clean.lower()
        depth_count = 0
        for pattern in depth_patterns:
            matches = re.findall(pattern, response_lower)
            depth_count += len(matches)
        
        # Normalize depth score (diminishing returns)
        depth_score = min(1.0, depth_count / 8.0)
        
        # === 7. Response length adequacy ===
        response_word_count = len(response_tokens)
        query_word_count = len(tokenize(query_clean))
        
        # Very short responses are likely incomplete
        if response_word_count <= 3:
            length_score = 0.05
        elif response_word_count <= 8:
            length_score = 0.15
        elif response_word_count <= 15:
            length_score = 0.35
        elif response_word_count <= 30:
            length_score = 0.55
        elif response_word_count <= 60:
            length_score = 0.75
        elif response_word_count <= 150:
            length_score = 0.9
        else:
            length_score = 1.0
        
        # === 8. Repetition penalty ===
        # Detect if response is repetitive (repeating same phrases/sentences)
        if len(response_sentences) >= 2:
            sentence_texts = [s.lower().strip() for s in response_sentences]
            unique_sentences = set(sentence_texts)
            repetition_ratio = len(unique_sentences) / len(sentence_texts)
        else:
            repetition_ratio = 1.0
        
        # Also check word-level repetition
        if response_content:
            word_freq = Counter(response_content)
            most_common_freq = word_freq.most_common(1)[0][1]
            word_diversity = len(set(response_content)) / len(response_content)
        else:
            word_diversity = 0.0
            most_common_freq = 0
        
        repetition_penalty = min(1.0, repetition_ratio * 0.5 + word_diversity * 0.5)
        
        # === 9. Irrelevance / off-topic detection ===
        # If response contains a lot of content unrelated to query
        # Check if response seems to go off on tangents
        off_topic_indicators = [
            r'\bquestion:', r'\binput:', r'\boutput:', r'\bexercise\b',
            r'```', r'import\s+\w+', r'def\s+\w+\(',
        ]
        
        off_topic_count = 0
        for pattern in off_topic_indicators:
            if re.search(pattern, response_clean):
                off_topic_count += 1
        
        # Check if query is about code - if so, code in response is fine
        query_is_code = bool(re.search(r'\bcode\b|\bprogram\b|\bfunction\b|\bhtml\b|\bpython\b|\bscript\b', query_clean.lower()))
        if query_is_code:
            off_topic_count = 0
        
        off_topic_penalty = max(0.3, 1.0 - off_topic_count * 0.15)
        
        # === 10. Truncation detection ===
        # If response appears cut off mid-sentence
        truncation_penalty = 1.0
        if response_clean and response_clean[-1] not in '.!?"\')}>':
            last_sentence = response_sentences[-1] if response_sentences else response_clean
            if len(last_sentence) > 20:
                truncation_penalty = 0.85
        
        # === 11. Direct answer detection ===
        # For questions, check if response actually provides an answer vs deflects
        is_question = '?' in query_clean or any(
            query_clean.lower().startswith(w) for w in 
            ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'is', 'are', 'do', 'does', 'which', 'identify', 'name', 'list', 'describe', 'explain']
        )
        
        # Check for deflection patterns
        deflection_patterns = [
            r'^no$', r'^yes$', r'^i don\'t know',
            r'you can tell us', r'let me know',
        ]
        
        is_deflection = False
        for pattern in deflection_patterns:
            if re.match(pattern, response_clean.lower().strip()):
                is_deflection = True
                break
        
        deflection_penalty = 0.1 if is_deflection else 1.0
        
        # === COMBINE SCORES ===
        # Weighted combination of all signals
        
        # Core coverage (40%)
        coverage_combined = (
            concept_coverage * 0.35 +
            bigram_coverage * 0.15 +
            aspect_coverage * 0.50
        )
        
        # Information richness (25%)
        concept_richness = min(1.0, total_unique_concepts / max(15, num_query_aspects * 5))
        richness_combined = (
            concept_richness * 0.4 +
            info_progression_ratio * 0.3 +
            depth_score * 0.3
        )
        
        # Structural adequacy (15%)
        structural_score = length_score
        
        # Quality modifiers (20%)
        quality_modifier = (
            repetition_penalty * 0.3 +
            off_topic_penalty * 0.25 +
            truncation_penalty * 0.2 +
            deflection_penalty * 0.25
        )
        
        raw_score = (
            coverage_combined * 0.40 +
            richness_combined * 0.25 +
            structural_score * 0.15 +
            quality_modifier * 0.20
        )
        
        # Scale to 0-10
        final_score = raw_score * 10.0
        
        # Apply floor and ceiling
        final_score = max(0.5, min(10.0, final_score))
        
        # Boost for very short but correct/complete answers to simple queries
        # (e.g., "What is X?" -> "Y" is fine if query is simple)
        if num_query_aspects <= 1 and query_word_count <= 15 and response_word_count <= 10:
            # Don't penalize too much for brevity on simple queries if coverage is decent
            if concept_coverage >= 0.3 and not is_deflection:
                final_score = max(final_score, 4.5)
        
        # Strong penalty for near-empty or clearly non-responsive answers
        if response_word_count <= 2:
            final_score = min(final_score, 1.5)
        
        if is_deflection:
            final_score = min(final_score, 2.0)
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: basic length heuristic
        try:
            resp_len = len(response.strip()) if response else 0
            if resp_len == 0:
                return 0.0
            elif resp_len < 10:
                return 1.5
            elif resp_len < 50:
                return 3.5
            elif resp_len < 200:
                return 5.5
            else:
                return 6.5
        except:
            return 3.0