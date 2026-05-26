def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level structural analysis approach.
    
    This variant focuses on:
    1. Sentence-level coherence and information density
    2. Citation/evidence pattern detection (dates, numbers, proper nouns, specific references)
    3. Hallucination red-flag detection (absolute claims, unsourced precision, sensationalism)
    4. Appropriate epistemic calibration (hedging vs. overconfidence ratio)
    5. Repetition/degeneration detection
    6. Query-response alignment via keyword extraction and coverage
    
    This is fundamentally different from word overlap, Jaccard, n-gram, paragraph analysis,
    vocabulary diversity, or simple hedging/concreteness approaches.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # ============================================================
        # 1. SENTENCE-LEVEL STRUCTURAL ANALYSIS
        # ============================================================
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b\w+\b', response.lower())
        num_words = max(len(words), 1)
        
        # Average sentence length (information density proxy)
        avg_sent_len = num_words / num_sentences
        # Ideal range: 10-25 words per sentence for factual content
        if avg_sent_len < 3:
            sent_len_score = 0.1
        elif avg_sent_len < 8:
            sent_len_score = 0.4
        elif avg_sent_len <= 30:
            # Peak around 15-20
            sent_len_score = 1.0 - abs(avg_sent_len - 17) / 30
            sent_len_score = max(sent_len_score, 0.3)
        else:
            sent_len_score = 0.3  # Run-on sentences
        
        # ============================================================
        # 2. REPETITION / DEGENERATION DETECTION
        # ============================================================
        # Check for repeated phrases (sliding window of 3-grams)
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i:i+3]))
        
        trigram_counts = Counter(trigrams)
        if trigrams:
            max_trigram_freq = max(trigram_counts.values())
            total_trigrams = len(trigrams)
            repetition_ratio = max_trigram_freq / max(total_trigrams, 1)
        else:
            repetition_ratio = 0
            max_trigram_freq = 0
        
        # Check for repeated sentences
        sent_lower = [s.lower().strip() for s in sentences]
        unique_sents = len(set(sent_lower))
        sent_repetition = 1.0 - (unique_sents / max(len(sent_lower), 1))
        
        # Check for repeated words (bigram level too)
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        if bigrams:
            max_bigram_freq = max(bigram_counts.values())
            bigram_rep_ratio = max_bigram_freq / max(len(bigrams), 1)
        else:
            bigram_rep_ratio = 0
            max_bigram_freq = 0
        
        # Severe repetition penalty
        repetition_penalty = 0.0
        if max_trigram_freq > 3 and repetition_ratio > 0.15:
            repetition_penalty += min(repetition_ratio * 3, 1.0)
        if sent_repetition > 0.3:
            repetition_penalty += sent_repetition
        if max_bigram_freq > 5 and bigram_rep_ratio > 0.2:
            repetition_penalty += min(bigram_rep_ratio * 2, 0.8)
        
        # Word-level repetition (single word dominance)
        word_counts = Counter(words)
        # Exclude common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them',
            'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
        }
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_count = Counter(content_words)
        if content_words:
            max_content_freq = max(content_count.values())
            if max_content_freq > 4 and max_content_freq / len(content_words) > 0.15:
                repetition_penalty += 0.3
        
        repetition_penalty = min(repetition_penalty, 2.0)
        
        # ============================================================
        # 3. EVIDENCE / SPECIFICITY PATTERN DETECTION
        # ============================================================
        # Dates (years, full dates)
        date_patterns = re.findall(r'\b(?:19|20)\d{2}\b', response)
        full_date_patterns = re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            response, re.IGNORECASE
        )
        
        # Numbers / statistics
        number_patterns = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b', response)
        # Filter out trivially small numbers that aren't really statistics
        meaningful_numbers = [n for n in number_patterns if len(n) > 1 or '%' in n or 'percent' in n.lower()]
        
        # Proper nouns (capitalized words not at sentence start)
        proper_noun_count = 0
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                if i > 0 and w and w[0].isupper() and w.lower() not in stop_words:
                    proper_noun_count += 1
        
        # Citation-like patterns
        citation_patterns = len(re.findall(
            r'(?:according to|research (?:shows|suggests|indicates)|studies? (?:show|suggest|indicate|found)|'
            r'data (?:shows|suggests|indicates)|evidence (?:shows|suggests)|'
            r'reported|published|journal|university|institute|professor|dr\.|'
            r'source|reference|cited|documented)',
            response, re.IGNORECASE
        ))
        
        # Explanation/reasoning patterns
        explanation_patterns = len(re.findall(
            r'(?:because|therefore|consequently|as a result|this means|for example|'
            r'for instance|such as|specifically|in particular|namely|'
            r'this is due to|the reason|which means|in other words|that is)',
            response, re.IGNORECASE
        ))
        
        # Calculate specificity score
        specificity_score = 0.0
        specificity_score += min(len(date_patterns) * 0.15, 0.5)
        specificity_score += min(len(full_date_patterns) * 0.25, 0.5)
        specificity_score += min(len(meaningful_numbers) * 0.08, 0.4)
        specificity_score += min(proper_noun_count * 0.06, 0.4)
        specificity_score += min(citation_patterns * 0.2, 0.6)
        specificity_score += min(explanation_patterns * 0.1, 0.5)
        
        # ============================================================
        # 4. HALLUCINATION RED-FLAG DETECTION
        # ============================================================
        red_flag_score = 0.0
        
        # Absolute/overconfident claims
        absolute_patterns = len(re.findall(
            r'\b(?:always|never|every single|without exception|guaranteed|'
            r'100%|absolutely certain|undeniable|unquestionable|'
            r'no doubt whatsoever|proven beyond|irrefutable|'
            r'everyone knows|nobody can deny|it is a fact that)\b',
            response, re.IGNORECASE
        ))
        red_flag_score += min(absolute_patterns * 0.15, 0.6)
        
        # Sensationalism / conspiracy language
        sensational_patterns = len(re.findall(
            r'\b(?:shocking|bombshell|explosive|mind-blowing|unbelievable|'
            r'they don\'t want you to know|the truth is being hidden|'
            r'wake up|sheeple|cover-?up|mainstream media lies|'
            r'big pharma|deep state|conspiracy|hoax|scam|'
            r'exposed|revealed|secret(?:ly)?|suppressed|banned|censored|'
            r'miracle|cure-?all|revolutionary breakthrough)\b',
            response, re.IGNORECASE
        ))
        red_flag_score += min(sensational_patterns * 0.25, 0.8)
        
        # Overly precise unsourced statistics (e.g., "exactly 73.847%")
        overly_precise = re.findall(r'\b\d+\.\d{2,}\s*%', response)
        red_flag_score += min(len(overly_precise) * 0.2, 0.4)
        
        # ============================================================
        # 5. EPISTEMIC CALIBRATION
        # ============================================================
        # Appropriate hedging for uncertain claims
        hedging_words = len(re.findall(
            r'\b(?:may|might|could|possibly|perhaps|likely|unlikely|'
            r'probably|approximately|roughly|estimated|around|about|'
            r'it seems|it appears|suggests|tends to|generally|'
            r'in many cases|often|sometimes|typically|usually|'
            r'some (?:experts|researchers|studies)|it is (?:thought|believed)|'
            r'one possible|there is debate|not entirely clear)\b',
            response, re.IGNORECASE
        ))
        
        # Calibration: some hedging is good, but too much is wishy-washy
        hedge_ratio = hedging_words / max(num_sentences, 1)
        if hedge_ratio == 0:
            calibration_score = 0.3  # No hedging at all - slightly concerning
        elif hedge_ratio <= 0.5:
            calibration_score = 0.6
        elif hedge_ratio <= 1.5:
            calibration_score = 1.0  # Good amount of hedging
        elif hedge_ratio <= 3.0:
            calibration_score = 0.7  # A bit too much
        else:
            calibration_score = 0.4  # Excessively hedged
        
        # ============================================================
        # 6. QUERY-RESPONSE ALIGNMENT (Keyword Coverage)
        # ============================================================
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_content = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        if query_content:
            response_word_set = set(words)
            covered = sum(1 for w in query_content if w in response_word_set)
            coverage_score = covered / len(query_content)
        else:
            coverage_score = 0.5  # neutral if no meaningful query words
        
        # ============================================================
        # 7. STRUCTURAL COMPLETENESS
        # ============================================================
        # Check if response ends mid-sentence (truncation)
        truncation_penalty = 0.0
        stripped_resp = response.rstrip()
        if stripped_resp and stripped_resp[-1] not in '.!?")\']':
            truncation_penalty = 0.3
            # Check if it seems severely truncated
            if len(stripped_resp) > 50 and stripped_resp[-1].isalpha():
                truncation_penalty = 0.5
        
        # Check for very short/empty responses
        length_score = 0.0
        if num_words < 3:
            length_score = 0.05
        elif num_words < 10:
            length_score = 0.3
        elif num_words < 20:
            length_score = 0.6
        elif num_words < 40:
            length_score = 0.8
        elif num_words <= 200:
            length_score = 1.0
        elif num_words <= 500:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ============================================================
        # 8. INFORMATION STRUCTURE (connectives, logical flow)
        # ============================================================
        # Causal/logical connectives
        logical_connectives = len(re.findall(
            r'\b(?:however|moreover|furthermore|additionally|in addition|'
            r'on the other hand|conversely|nevertheless|nonetheless|'
            r'while|whereas|although|despite|in contrast|similarly|'
            r'first|second|third|finally|lastly|overall|in conclusion|'
            r'to summarize|in summary)\b',
            response, re.IGNORECASE
        ))
        
        structure_score = min(logical_connectives * 0.12, 0.6)
        
        # Definitional/explanatory structure
        definitional = len(re.findall(
            r'(?:is (?:a|an|the|defined as)|refers to|means that|involves|'
            r'can be described as|is known as|is characterized by)',
            response, re.IGNORECASE
        ))
        structure_score += min(definitional * 0.1, 0.4)
        
        # ============================================================
        # 9. UNIQUE CONTENT RICHNESS (not vocab diversity but semantic breadth)
        # ============================================================
        # Count distinct semantic clusters (rough proxy: unique content word stems)
        if content_words:
            # Simple pseudo-stemming: take first 5 chars
            pseudo_stems = set(w[:min(5, len(w))] for w in content_words)
            semantic_breadth = len(pseudo_stems) / max(len(content_words), 1)
            # Also consider absolute count of unique content
            unique_content_ratio = len(set(content_words)) / max(len(content_words), 1)
            richness_score = (semantic_breadth + unique_content_ratio) / 2
        else:
            richness_score = 0.1
        
        # ============================================================
        # FINAL SCORING COMBINATION
        # ============================================================
        # Weighted combination with emphasis on factual accuracy indicators
        
        score = 0.0
        
        # Base quality (length and structure)
        score += length_score * 2.0          # 0-2.0
        score += sent_len_score * 1.0        # 0-1.0
        
        # Factual reliability indicators
        score += specificity_score * 1.5     # 0-variable, capped contributions
        score += calibration_score * 1.0     # 0-1.0
        score += structure_score * 1.0       # 0-1.0
        
        # Content quality
        score += coverage_score * 1.5        # 0-1.5
        score += richness_score * 1.5        # 0-1.5
        
        # Penalties
        score -= repetition_penalty * 2.0    # 0-4.0 penalty
        score -= red_flag_score * 1.5        # 0-variable penalty
        score -= truncation_penalty * 1.5    # 0-0.75 penalty
        
        # Normalize to 0-10 range
        # Theoretical max ~9.5, min could be negative
        score = max(score, 0.0)
        score = min(score, 10.0)
        
        return round(score, 3)
        
    except Exception:
        # Never crash - return neutral score
        try:
            if response and len(response.strip()) > 0:
                return 3.0
            return 0.0
        except Exception:
            return 0.0