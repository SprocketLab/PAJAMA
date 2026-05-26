def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    
    Algorithm: Analyzes each sentence individually for information density signals,
    computes per-sentence "evidence scores", then aggregates using distributional
    statistics (median, variance, proportion of high-evidence sentences).
    
    This is fundamentally different from prior variants by:
    - Operating at sentence granularity rather than document/word level
    - Using character-class ratios within sentences
    - Computing distributional statistics over sentence-level scores
    - Using a taxonomy of specificity markers vs vagueness markers with contextual weighting
    - Measuring "information compression" (ratio of content-bearing tokens per sentence length)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?;:])\s+|\n+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 0.5
        
        # === SPECIFICITY MARKERS (per-sentence) ===
        # These are patterns that indicate concrete, specific information
        
        # Numeric patterns
        num_pattern = re.compile(r'\b\d+[\d,\.]*\b')
        # Percentage/measurement patterns
        measure_pattern = re.compile(r'\b\d+\s*(%|percent|pounds?|kg|lbs?|oz|ounces?|cups?|tbsp|tsp|ml|liters?|hours?|minutes?|seconds?|degrees?|steps?|times?|miles?|km|feet|inches?|cm|mm)\b', re.IGNORECASE)
        # Proper nouns (capitalized words not at sentence start)
        proper_noun_pattern = re.compile(r'(?<!^)(?<!\. )(?<!\.\s)\b[A-Z][a-z]{2,}\b')
        # Quoted terms or technical terms
        quoted_pattern = re.compile(r'["\'][^"\']+["\']|`[^`]+`')
        # Enumeration patterns (1., 2., a), b), First, Second, etc.)
        enum_pattern = re.compile(r'^\s*(\d+[\.\):]|[a-z][\.\)]|[-•*])\s', re.IGNORECASE)
        # Causal/explanatory connectors suggesting reasoning
        causal_pattern = re.compile(r'\b(because|therefore|thus|hence|since|due to|as a result|this means|which means|this is because|the reason|caused by|leads to|results in)\b', re.IGNORECASE)
        # Action verbs suggesting concrete instructions
        action_pattern = re.compile(r'\b(add|remove|click|press|type|enter|select|choose|open|close|heat|cook|stir|mix|pour|cut|place|set|turn|move|grab|take|put|apply|use|install|download|run|execute|create|build|make|write|read|check|verify|ensure|confirm)\b', re.IGNORECASE)
        # Comparative/superlative specifics
        comparative_pattern = re.compile(r'\b(more than|less than|at least|at most|up to|between \d+ and \d+|from \d+ to \d+|approximately|roughly|exactly|precisely)\b', re.IGNORECASE)
        
        # === VAGUENESS MARKERS ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bvarious factors\b', r'\bthere are (many|various|several|different)\b',
            r'\bin general\b', r'\bgenerally speaking\b', r'\bfor the most part\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b', r'\bmight\b',
            r'\bcould be\b', r'\bmay or may not\b',
            r'\band stuff\b', r'\band things\b', r'\betc\.?\b',
            r'\byou know\b', r'\bI guess\b', r'\bI think\b',
            r'\bwhatever\b', r'\bsomehow\b', r'\bsomewhere\b',
            r'\bjust\b.*\bjust\b',  # repeated "just"
            r'\bbasically\b', r'\bessentially\b', r'\bfundamentally\b',
            r'\bsupposedly\b', r'\ballegedly\b',
        ]
        vague_patterns = [re.compile(p, re.IGNORECASE) for p in vague_phrases]
        
        # Dismissive/minimizing patterns
        dismissive_phrases = [
            r'\bjust do\b', r'\bjust try\b', r'\bjust get\b', r'\bjust make\b',
            r'\bdon\'t worry\b', r'\bno big deal\b', r'\bit\'s fine\b',
            r'\byou\'ll be fine\b', r'\bget over it\b', r'\bmove on\b',
            r'\bkeep trying\b', r'\bkeep working\b',
        ]
        dismissive_patterns = [re.compile(p, re.IGNORECASE) for p in dismissive_phrases]
        
        # === SENTENCE-LEVEL SCORING ===
        sentence_scores = []
        
        for sent in sentences:
            score = 0.0
            words = sent.split()
            word_count = len(words)
            if word_count == 0:
                continue
            
            # 1. Numeric density
            nums = num_pattern.findall(sent)
            score += min(len(nums) * 1.5, 5.0)
            
            # 2. Measurement specifics (bonus on top of numbers)
            measures = measure_pattern.findall(sent)
            score += min(len(measures) * 2.0, 4.0)
            
            # 3. Proper nouns / named entities
            proper_nouns = proper_noun_pattern.findall(sent)
            score += min(len(proper_nouns) * 1.0, 3.0)
            
            # 4. Quoted/technical terms
            quoted = quoted_pattern.findall(sent)
            score += min(len(quoted) * 1.5, 3.0)
            
            # 5. Enumeration (structured information)
            if enum_pattern.match(sent):
                score += 2.0
            
            # 6. Causal/explanatory reasoning
            causal_matches = causal_pattern.findall(sent)
            score += min(len(causal_matches) * 1.5, 3.0)
            
            # 7. Action verbs (concrete instructions)
            action_matches = action_pattern.findall(sent)
            score += min(len(action_matches) * 0.8, 3.0)
            
            # 8. Comparative specifics
            comp_matches = comparative_pattern.findall(sent)
            score += min(len(comp_matches) * 1.5, 3.0)
            
            # 9. Information compression: ratio of "long" (content) words to total
            long_words = [w for w in words if len(w) > 5]
            compression_ratio = len(long_words) / max(word_count, 1)
            score += compression_ratio * 3.0
            
            # 10. Unique word ratio within sentence (lexical density)
            unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
            score += unique_ratio * 2.0
            
            # 11. Penalize vagueness
            for vp in vague_patterns:
                if vp.search(sent):
                    score -= 1.5
            
            # 12. Penalize dismissiveness
            for dp in dismissive_patterns:
                if dp.search(sent):
                    score -= 2.0
            
            # 13. Penalize very short sentences (often filler)
            if word_count < 5:
                score -= 1.0
            
            # 14. Bonus for parenthetical clarifications (e.g., "qubits (quantum bits)")
            parens = re.findall(r'\([^)]+\)', sent)
            score += min(len(parens) * 1.5, 3.0)
            
            # 15. Bonus for colon-based definitions or explanations
            if ':' in sent and word_count > 5:
                score += 1.0
            
            sentence_scores.append(max(score, 0.0))
        
        if not sentence_scores:
            return 1.0
        
        # === AGGREGATE SCORING ===
        
        # Median sentence score (robust to outliers)
        sorted_scores = sorted(sentence_scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0
        else:
            median_score = sorted_scores[n // 2]
        
        # Mean sentence score
        mean_score = sum(sentence_scores) / n
        
        # Proportion of "high evidence" sentences (score > 4)
        high_evidence_ratio = sum(1 for s in sentence_scores if s > 4.0) / n
        
        # Top quartile average (captures peak evidence density)
        top_quarter_idx = max(1, n // 4)
        top_quarter_avg = sum(sorted_scores[-top_quarter_idx:]) / top_quarter_idx
        
        # === DOCUMENT-LEVEL FEATURES ===
        
        response_lower = response_clean.lower()
        
        # Structural organization bonus
        structure_score = 0.0
        # Numbered lists
        numbered_items = re.findall(r'^\s*\d+[\.\):]', response_clean, re.MULTILINE)
        if len(numbered_items) >= 2:
            structure_score += min(len(numbered_items) * 0.5, 3.0)
        
        # Bullet points
        bullet_items = re.findall(r'^\s*[-•*]\s', response_clean, re.MULTILINE)
        if len(bullet_items) >= 2:
            structure_score += min(len(bullet_items) * 0.4, 2.5)
        
        # Paragraph count (well-organized responses have multiple paragraphs)
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if len(p.strip()) > 20]
        if len(paragraphs) >= 2:
            structure_score += min(len(paragraphs) * 0.3, 1.5)
        
        # === EMPATHY/ENGAGEMENT DETECTION ===
        # For emotional queries, good responses show empathy WITH substance
        emotional_query_signals = re.findall(
            r'\b(feeling|frustrated|stress|sad|lonely|heartbroken|devastated|upset|angry|anxious|worried|struggling|difficult|hard time)\b',
            query.lower()
        )
        
        empathy_score = 0.0
        if emotional_query_signals:
            # Check for empathetic acknowledgment
            empathy_markers = re.findall(
                r'\b(understand|sorry to hear|completely|absolutely|natural|okay to feel|valid|understandable|it\'s okay|perfectly fine|normal)\b',
                response_lower
            )
            empathy_score += min(len(empathy_markers) * 0.8, 3.0)
            
            # Check for actionable advice alongside empathy
            advice_markers = re.findall(
                r'\b(try|consider|recommend|suggest|here are|you can|you could|one way|another way|first|second|step)\b',
                response_lower
            )
            empathy_score += min(len(advice_markers) * 0.5, 2.0)
        
        # === QUERY RELEVANCE (lightweight) ===
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
        
        if query_words:
            relevance = len(query_words & response_words) / len(query_words)
        else:
            relevance = 0.5
        
        # === NEGATIVE SIGNALS (document level) ===
        negative_score = 0.0
        
        # Excessive hedging at document level
        hedge_count = len(re.findall(
            r'\b(might|maybe|perhaps|possibly|could be|not sure|hard to say|difficult to determine)\b',
            response_lower
        ))
        negative_score += min(hedge_count * 0.5, 3.0)
        
        # Contradictions or inability statements
        inability_count = len(re.findall(
            r'\b(can\'t|cannot|won\'t be able|might not|may not|probably won\'t|not able to)\b',
            response_lower
        ))
        negative_score += min(inability_count * 0.3, 2.0)
        
        # Repetitive phrases (sign of low information density)
        words_list = re.findall(r'\b[a-z]+\b', response_lower)
        if len(words_list) > 10:
            # Check for repeated bigrams
            bigrams = [f"{words_list[i]} {words_list[i+1]}" for i in range(len(words_list) - 1)]
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 2)
            negative_score += min(repeated_bigrams * 0.3, 2.0)
        
        # === FINAL SCORE COMPOSITION ===
        
        # Weight the components
        final_score = (
            median_score * 0.25 +          # Central tendency of sentence evidence
            mean_score * 0.20 +             # Average evidence density
            high_evidence_ratio * 8.0 * 0.15 +  # Proportion of evidence-rich sentences
            top_quarter_avg * 0.15 +        # Peak evidence quality
            structure_score * 0.10 +        # Organizational quality
            empathy_score * 0.08 +          # Emotional intelligence (when relevant)
            relevance * 3.0 * 0.07 +        # Query relevance
            - negative_score * 0.15         # Penalties
        )
        
        # Length bonus (longer responses can contain more evidence, but diminishing returns)
        word_count_total = len(words_list) if words_list else 0
        length_factor = min(math.log(max(word_count_total, 1) + 1) / math.log(200), 1.2)
        final_score *= (0.7 + 0.3 * length_factor)
        
        # Normalize to 1-5 range
        # Empirically, raw scores tend to range from ~0 to ~8
        normalized = 1.0 + (final_score / 7.0) * 4.0
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
    
    except Exception:
        return 2.5