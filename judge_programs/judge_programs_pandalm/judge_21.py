def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question decomposition (detecting sub-questions/aspects in query)
    - Information density (unique information units per sentence)
    - Structural depth (nested explanations, examples, elaborations)
    - Repetition penalty
    - Coverage of query entities/concepts
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
        
        # ---- Helper functions ----
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 3]
        
        def get_clauses(text):
            """Split into clauses using commas, semicolons, conjunctions"""
            parts = re.split(r'[,;]|\band\b|\bbut\b|\bwhile\b|\bwhereas\b|\balthough\b|\bwhich\b|\bthat\b|\bwhere\b|\bwhen\b', text.lower())
            return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if len(response_tokens) < 2:
            return 0.5
        
        sentences = get_sentences(response)
        clauses = get_clauses(response)
        
        # ---- 1. Query Aspect Decomposition & Coverage ----
        # Extract meaningful query concepts (nouns, verbs, key phrases)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'that', 'which', 'who',
            'whom', 'this', 'these', 'those', 'what', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'about', 'up', 'also', 'give', 'given', 'following',
            'describe', 'explain', 'provide', 'write', 'generate', 'create', 'make',
            'come', 'rewrite', 'input', 'output', 'using', 'use'
        }
        
        query_concepts = [t for t in query_tokens if t not in stopwords and len(t) > 2]
        query_concept_set = set(query_concepts)
        
        # Check how many query concepts appear in response
        response_token_set = set(response_tokens)
        if query_concept_set:
            direct_coverage = len(query_concept_set & response_token_set) / len(query_concept_set)
        else:
            direct_coverage = 0.5
        
        # ---- 2. Detect query complexity (sub-questions, multiple aspects) ----
        # Count question marks, "and" conjunctions, commas in query, listed items
        question_marks = query.count('?')
        query_aspects = max(1, question_marks) if question_marks > 0 else 1
        
        # Detect multi-part queries
        multi_part_indicators = len(re.findall(
            r'\band\b|\balso\b|\badditionally\b|\bfurthermore\b|\bmoreover\b|\bthen\b',
            query.lower()
        ))
        query_aspects += multi_part_indicators
        
        # Detect comparison queries
        comparison_words = re.findall(
            r'\bcompare\b|\bcontrast\b|\bdifference\b|\bsimilar\b|\bversus\b|\bvs\b|\bboth\b',
            query.lower()
        )
        is_comparison = len(comparison_words) > 0
        if is_comparison:
            query_aspects = max(query_aspects, 3)  # comparisons need multiple aspects
        
        # ---- 3. Information Density ----
        # Unique content words per sentence (measures how much new info each sentence adds)
        seen_content = set()
        new_info_per_sentence = []
        for sent in sentences:
            sent_tokens = [t for t in tokenize(sent) if t not in stopwords and len(t) > 2]
            new_tokens = set(sent_tokens) - seen_content
            if sent_tokens:
                new_info_per_sentence.append(len(new_tokens) / max(len(sent_tokens), 1))
            seen_content.update(sent_tokens)
        
        if new_info_per_sentence:
            avg_new_info = sum(new_info_per_sentence) / len(new_info_per_sentence)
        else:
            avg_new_info = 0.0
        
        # Total unique content words
        content_tokens = [t for t in response_tokens if t not in stopwords and len(t) > 2]
        unique_content = set(content_tokens)
        unique_content_count = len(unique_content)
        
        # ---- 4. Repetition Penalty ----
        if content_tokens:
            content_freq = Counter(content_tokens)
            max_freq = max(content_freq.values())
            total_content = len(content_tokens)
            unique_ratio = len(unique_content) / total_content
            
            # Severe repetition: same word appears many times
            repetition_penalty = 0.0
            for word, freq in content_freq.items():
                if freq > 3:
                    repetition_penalty += (freq - 3) * 0.05
            
            # Check for repeated phrases (3-grams)
            trigrams = []
            for i in range(len(response_tokens) - 2):
                trigrams.append(tuple(response_tokens[i:i+3]))
            trigram_freq = Counter(trigrams)
            phrase_repetition = sum(1 for _, c in trigram_freq.items() if c > 2)
            repetition_penalty += phrase_repetition * 0.15
            
            # Check for repeated sentences
            sent_texts = [' '.join(tokenize(s)) for s in sentences]
            sent_freq = Counter(sent_texts)
            sent_repetition = sum(c - 1 for c in sent_freq.values() if c > 1)
            repetition_penalty += sent_repetition * 0.3
        else:
            unique_ratio = 0.0
            repetition_penalty = 0.0
        
        repetition_penalty = min(repetition_penalty, 5.0)  # cap
        
        # ---- 5. Structural Depth Indicators ----
        # Elaboration markers (signals that the response goes deeper)
        elaboration_markers = len(re.findall(
            r'\bfor example\b|\bfor instance\b|\bsuch as\b|\bspecifically\b|\bin particular\b'
            r'|\bnamely\b|\bincluding\b|\be\.g\.\b|\bi\.e\.\b|\bin other words\b'
            r'|\bthis means\b|\bthis suggests\b|\bin addition\b|\bfurthermore\b'
            r'|\bmoreover\b|\balso\b|\badditionally\b',
            response.lower()
        ))
        
        # Causal/explanatory depth
        causal_markers = len(re.findall(
            r'\bbecause\b|\btherefore\b|\bthus\b|\bhence\b|\bas a result\b'
            r'|\bconsequently\b|\bdue to\b|\bleading to\b|\bwhich means\b'
            r'|\bso that\b|\bin order to\b|\bthis leads\b|\bcaused by\b',
            response.lower()
        ))
        
        # Contrast/nuance markers (important for completeness)
        contrast_markers = len(re.findall(
            r'\bhowever\b|\bon the other hand\b|\bwhile\b|\bwhereas\b|\balthough\b'
            r'|\bdespite\b|\bnevertheless\b|\bin contrast\b|\bconversely\b'
            r'|\bunlike\b|\bdiffers?\b|\bdifferent\b',
            response.lower()
        ))
        
        depth_score = min(elaboration_markers * 0.4 + causal_markers * 0.5 + contrast_markers * 0.4, 4.0)
        
        # ---- 6. Clause-level richness ----
        # More clauses = more information units
        num_clauses = len(clauses)
        clause_score = min(math.log1p(num_clauses) * 1.5, 4.0)
        
        # ---- 7. Response proportionality to query complexity ----
        # More complex queries should have longer, more detailed responses
        expected_min_sentences = max(2, query_aspects * 2)
        actual_sentences = len(sentences)
        
        if actual_sentences >= expected_min_sentences:
            proportionality = 1.0
        else:
            proportionality = actual_sentences / expected_min_sentences
        
        # ---- 8. Specificity indicators ----
        # Numbers, proper nouns, technical terms suggest concrete/specific answers
        numbers = len(re.findall(r'\b\d+\b', response))
        capitalized_words = len(re.findall(r'\b[A-Z][a-z]{2,}\b', response))
        # Filter out sentence starters
        sentence_starters = len(sentences)
        specific_caps = max(0, capitalized_words - sentence_starters)
        
        specificity_score = min((numbers * 0.3 + specific_caps * 0.2), 2.0)
        
        # ---- 9. Truncation detection ----
        truncation_penalty = 0.0
        if response[-1] not in '.!?")\']':
            # Response might be truncated
            truncation_penalty = 1.5
        # Check if last sentence is incomplete (very short compared to others)
        if sentences:
            last_sent_words = len(tokenize(sentences[-1]))
            if last_sent_words < 3 and len(sentences) > 1:
                truncation_penalty += 0.5
        
        # ---- 10. Length score (logarithmic, diminishing returns) ----
        word_count = len(response_tokens)
        length_score = min(math.log1p(word_count) * 1.2, 5.0)
        
        # ---- 11. Empty/garbage detection ----
        if word_count < 5:
            return max(0.5, length_score - repetition_penalty)
        
        # Check if response is mostly the same as query (just echoing)
        if query_tokens:
            query_set = set(query_tokens)
            resp_only = response_token_set - query_set
            novelty = len(resp_only) / max(len(response_token_set), 1)
        else:
            novelty = 1.0
        
        # ---- Composite Score ----
        score = 0.0
        
        # Coverage of query concepts (0-2)
        score += direct_coverage * 2.0
        
        # Information density & uniqueness (0-2)
        score += avg_new_info * 1.0
        score += min(unique_ratio * 1.5, 1.5)
        
        # Structural depth (0-4)
        score += depth_score
        
        # Clause richness (0-4)
        score += clause_score
        
        # Length (0-5)
        score += length_score
        
        # Proportionality (0-2)
        score += proportionality * 2.0
        
        # Specificity (0-2)
        score += specificity_score
        
        # Novelty beyond query (0-1)
        score += novelty * 1.0
        
        # Penalties
        score -= repetition_penalty
        score -= truncation_penalty
        
        # Bonus for comparison queries that actually compare
        if is_comparison and contrast_markers >= 1:
            score += 1.0
        
        # Ensure score is in reasonable range [0, 20] -> normalize to [0, 10]
        score = max(0.0, min(score, 20.0))
        score = score / 2.0  # normalize to 0-10
        
        return round(score, 3)
        
    except Exception as e:
        # Fallback: simple length-based score
        try:
            words = len(response.split())
            return min(words / 10.0, 5.0)
        except:
            return 0.0