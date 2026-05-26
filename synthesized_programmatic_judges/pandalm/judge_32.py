def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level structure quality (well-formed sentences with subject-verb patterns)
    2. Information density via unique content words per sentence
    3. Citation/evidence language patterns (regex-based pattern matching)
    4. Hallucination red-flag detection (absolute claims, unsourced precision)
    5. Appropriate uncertainty language calibration
    6. Anti-repetition scoring via compression ratio analogy
    7. Query coverage via semantic field matching
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 3:
            return 0.5
        
        # ---- 1. Sentence-level analysis ----
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # Score: well-formed sentences (start with capital, end with punctuation)
        well_formed_count = 0
        for s in sentences:
            if s and s[0].isupper() and s[-1] in '.!?':
                well_formed_count += 1
        sentence_quality = well_formed_count / num_sentences if num_sentences > 0 else 0
        
        # ---- 2. Information density via unique content words per sentence ----
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'they', 'them', 'their', 'also', 'there', 'here', 'up',
        }
        
        all_words = re.findall(r'[a-z]+', response.lower())
        total_words = len(all_words) if all_words else 1
        content_words = [w for w in all_words if w not in stopwords and len(w) > 2]
        unique_content = set(content_words)
        
        info_density = len(unique_content) / max(num_sentences, 1)
        # Normalize: ideal ~5-15 unique content words per sentence
        info_density_score = min(info_density / 10.0, 1.0)
        
        # ---- 3. Citation/evidence language patterns ----
        # Look for patterns that suggest factual grounding
        citation_patterns = [
            r'\b\d{4}\b',  # years
            r'\b\d+%\b',  # percentages
            r'\b\d+\.\d+\b',  # decimal numbers
            r'\baccording\s+to\b',
            r'\bresearch\b',
            r'\bstud(?:y|ies)\b',
            r'\bexample\b',
            r'\bfor\s+instance\b',
            r'\bsuch\s+as\b',
            r'\bincluding\b',
            r'\bspecifically\b',
            r'\bin\s+particular\b',
            r'\bevidence\b',
            r'\bdata\b',
            r'\bfound\s+that\b',
            r'\bdemonstrat\w+\b',
            r'\bshow(?:s|n|ed)?\s+that\b',
        ]
        
        citation_hits = 0
        for pat in citation_patterns:
            matches = re.findall(pat, response.lower())
            citation_hits += len(matches)
        
        # Normalize: diminishing returns after several citation markers
        citation_score = min(citation_hits / 5.0, 1.0)
        
        # ---- 4. Hallucination red-flags ----
        red_flag_patterns = [
            r'\bexactly\s+\d+\b',  # overly precise unsourced stats
            r'\b100%\b',
            r'\balways\b',
            r'\bnever\b',
            r'\beveryone\s+knows\b',
            r'\bobviously\b',
            r'\bclearly\b',  # can be fine but often signals unsupported confidence
            r'\bundeniably\b',
            r'\bwithout\s+(?:a\s+)?doubt\b',
            r'\bno\s+one\s+can\s+deny\b',
            r'\bproven\s+(?:fact|true)\b',
            r'\bthey\s+don\'t\s+want\s+you\s+to\s+know\b',
            r'\bsecretly\b',
            r'\bhidden\s+(?:truth|agenda)\b',
            r'\bconspiracy\b',
            r'\bwake\s+up\b',
            r'\bsheeple\b',
            r'\bmainstream\s+media\b',
            r'\bcover[\s-]?up\b',
            r'\bguaranteed\b',
            r'\babsolutely\s+(?:certain|sure|true)\b',
        ]
        
        red_flag_count = 0
        for pat in red_flag_patterns:
            matches = re.findall(pat, response.lower())
            red_flag_count += len(matches)
        
        # Penalty increases with more red flags
        red_flag_penalty = min(red_flag_count * 0.15, 1.0)
        
        # ---- 5. Appropriate hedging/uncertainty calibration ----
        hedging_patterns = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends?\s+to\b', r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bpossibly\b', r'\bprobably\b', r'\blikely\b', r'\bapproximately\b',
            r'\babout\b', r'\baround\b', r'\broughly\b', r'\bin\s+some\s+cases\b',
            r'\bit\s+(?:is\s+)?(?:possible|likely)\b', r'\bsuggests?\b',
            r'\bindicates?\b', r'\bappears?\b', r'\bseems?\b',
            r'\bin\s+general\b', r'\bmost(?:ly)?\b', r'\bsome\b',
        ]
        
        hedge_count = 0
        for pat in hedging_patterns:
            matches = re.findall(pat, response.lower())
            hedge_count += len(matches)
        
        # Hedging is good but too much is wishy-washy
        # Ideal: ~1-4 hedges per 100 words
        hedge_rate = hedge_count / max(total_words / 100.0, 0.5)
        if hedge_rate <= 0:
            hedge_score = 0.3  # no hedging at all is slightly penalized
        elif hedge_rate <= 5:
            hedge_score = 0.5 + 0.1 * hedge_rate  # linear increase up to 1.0
        else:
            hedge_score = max(1.0 - (hedge_rate - 5) * 0.1, 0.3)  # too much hedging
        hedge_score = min(hedge_score, 1.0)
        
        # ---- 6. Repetition penalty via compression ratio analogy ----
        # Count repeated n-grams (trigrams)
        if len(all_words) >= 3:
            trigrams = [tuple(all_words[i:i+3]) for i in range(len(all_words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            total_trigrams = max(len(trigrams), 1)
            repetition_ratio = repeated_trigrams / total_trigrams
        else:
            repetition_ratio = 0
        
        # Also check for repeated words (extreme repetition)
        word_counts = Counter(all_words)
        max_word_freq = max(word_counts.values()) if word_counts else 1
        # If any single word appears way too often relative to response length
        dominance = max_word_freq / max(total_words, 1)
        
        repetition_penalty = min(repetition_ratio * 3.0 + max(0, dominance - 0.15) * 2.0, 1.0)
        
        # ---- 7. Query coverage via semantic field matching ----
        query_words = re.findall(r'[a-z]+', query.lower())
        query_content = set(w for w in query_words if w not in stopwords and len(w) > 2)
        
        if query_content:
            covered = query_content.intersection(unique_content)
            coverage = len(covered) / len(query_content)
        else:
            coverage = 0.5  # neutral if query has no content words
        
        # ---- 8. Response length appropriateness ----
        # Too short is bad, moderate length is good, very long can be fine
        if total_words < 5:
            length_score = 0.1
        elif total_words < 15:
            length_score = 0.3
        elif total_words < 30:
            length_score = 0.6
        elif total_words < 200:
            length_score = 1.0
        elif total_words < 500:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ---- 9. Structural indicators ----
        # Check for explanatory structure (cause-effect, comparison, listing)
        structural_patterns = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bin\s+contrast\b', r'\bhowever\b',
            r'\bon\s+the\s+other\s+hand\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bconsequently\b', r'\btherefore\b',
            r'\bas\s+a\s+result\b', r'\bthis\s+means\b', r'\bfor\s+example\b',
            r'\bin\s+other\s+words\b', r'\bnamely\b',
            r'\bbecause\b', r'\bsince\b', r'\bdue\s+to\b',
        ]
        
        struct_hits = 0
        for pat in structural_patterns:
            if re.search(pat, response.lower()):
                struct_hits += 1
        
        structural_score = min(struct_hits / 4.0, 1.0)
        
        # ---- 10. Specificity score ----
        # Longer content words and more diverse vocabulary suggest specificity
        avg_content_len = sum(len(w) for w in content_words) / max(len(content_words), 1)
        specificity = min((avg_content_len - 3) / 5.0, 1.0)  # words avg 3-8 chars
        specificity = max(specificity, 0.0)
        
        # Unique word ratio (type-token ratio on content words)
        ttr = len(unique_content) / max(len(content_words), 1)
        vocab_diversity = min(ttr * 1.2, 1.0)
        
        # ---- COMBINE SCORES ----
        # Weighted combination
        score = (
            sentence_quality * 1.5 +      # well-formed sentences
            info_density_score * 2.0 +     # information-rich
            citation_score * 1.5 +         # evidence language
            (1.0 - red_flag_penalty) * 2.0 +  # no hallucination flags
            hedge_score * 1.0 +            # appropriate hedging
            (1.0 - repetition_penalty) * 2.0 + # no excessive repetition
            coverage * 1.5 +               # addresses the query
            length_score * 1.0 +           # appropriate length
            structural_score * 1.0 +       # good structure
            specificity * 0.75 +           # specific vocabulary
            vocab_diversity * 0.75         # vocabulary diversity
        )
        
        # Max possible: 1.5+2+1.5+2+1+2+1.5+1+1+0.75+0.75 = 15.0
        # Normalize to 0-10 scale
        max_possible = 15.0
        normalized = (score / max_possible) * 10.0
        
        # Clamp
        return max(0.0, min(10.0, round(normalized, 3)))
        
    except Exception:
        try:
            # Minimal fallback: score based on response length
            return min(len(response.split()) / 10.0, 5.0)
        except Exception:
            return 2.5