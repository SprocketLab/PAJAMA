def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    Focuses on: specificity signals, hedging appropriateness, hallucination red flags,
    structural credibility markers, and information density.
    
    Uses sentence-level decomposition rather than simple word overlap.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        query = query.strip() if query and isinstance(query, str) else ""
        
        score = 50.0  # Start at midpoint
        
        # === 1. Sentence-level analysis ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # === 2. Specificity & factual grounding signals (per sentence) ===
        # Look for specific factual indicators at sentence level
        
        # Numbers, dates, measurements
        number_pattern = re.compile(r'\b\d+[\d,]*\.?\d*\b')
        date_pattern = re.compile(r'\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}')
        
        sentences_with_specifics = 0
        for sent in sentences:
            has_number = bool(number_pattern.search(sent))
            has_date = bool(date_pattern.search(sent))
            has_proper_noun = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent))
            if has_number or has_date or has_proper_noun:
                sentences_with_specifics += 1
        
        specificity_ratio = sentences_with_specifics / num_sentences
        score += specificity_ratio * 5  # Up to +5
        
        # === 3. Appropriate hedging detection ===
        # Hedging phrases that indicate epistemic awareness
        hedging_phrases = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\btend(?:s)?\s+to\b',
            r'\bin\s+(?:many|most|some)\s+cases\b', r'\bapproximately\b',
            r'\babout\b', r'\baround\b', r'\broughly\b',
            r'\blikely\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\baccording\s+to\b', r'\bresearch\s+(?:suggests|shows|indicates)\b',
            r'\bit\s+(?:appears|seems)\b', r'\bcan\s+be\b',
            r'\bin\s+general\b', r'\bfor\s+(?:example|instance)\b',
        ]
        
        hedging_count = 0
        response_lower = response.lower()
        for pattern in hedging_phrases:
            hedging_count += len(re.findall(pattern, response_lower))
        
        # Moderate hedging is good; too much is bad
        hedging_per_sentence = hedging_count / num_sentences
        if hedging_per_sentence <= 0.3:
            score += hedging_per_sentence * 8  # Reward some hedging
        elif hedging_per_sentence <= 1.0:
            score += 2.5  # Moderate hedging is fine
        else:
            score += max(0, 2.5 - (hedging_per_sentence - 1.0) * 2)  # Penalize excessive hedging
        
        # === 4. Hallucination red flags ===
        # Overly precise unsourced statistics
        overly_precise_stats = re.findall(r'\b\d{1,3}\.\d{2,}%\b', response)
        score -= len(overly_precise_stats) * 3
        
        # Absolute/extreme claims
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bevery\s+single\b',
            r'\bwithout\s+(?:any\s+)?exception\b', r'\babsolutely\b',
            r'\bdefinitely\b', r'\bundeniably\b', r'\bunquestionably\b',
            r'\bwithout\s+a\s+doubt\b', r'\b100%\b', r'\bguaranteed?\b',
            r'\bno\s+one\b(?!\s+(?:knows|is sure))', r'\beveryone\s+knows\b',
        ]
        
        absolute_count = 0
        for pattern in absolute_patterns:
            absolute_count += len(re.findall(pattern, response_lower))
        
        score -= absolute_count * 2.0
        
        # === 5. Sensationalism and conspiracy language ===
        sensational_patterns = [
            r'\bshocking\b', r'\bunbelievable\b', r'\bmind[\s-]?blowing\b',
            r'\binsane\b', r'\bcrazy\b', r'\bthey\s+don\'?t\s+want\s+you\s+to\s+know\b',
            r'\bhidden\s+truth\b', r'\bcover[\s-]?up\b', r'\bwake\s+up\b',
            r'\bsheep(?:le)?\b', r'\bthe\s+elites?\b', r'\bnew\s+world\s+order\b',
            r'\bdeep\s+state\b', r'\bsecret(?:ly)?\s+(?:control|manipulat)\b',
            r'\bexpos[eé]\b', r'\bbombshell\b', r'\bbreaking\b',
            r'\byou\s+won\'?t\s+believe\b', r'\bthis\s+changes\s+everything\b',
        ]
        
        sensational_count = 0
        for pattern in sensational_patterns:
            sensational_count += len(re.findall(pattern, response_lower))
        
        score -= sensational_count * 4.0
        
        # === 6. Information density analysis ===
        # Unique content words per sentence (avoid repetition)
        words = re.findall(r'\b[a-z]+\b', response_lower)
        total_words = len(words)
        
        if total_words == 0:
            return 1.0
        
        # Stop words for content analysis
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'they',
            'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'i', 'me', 'my', 'who', 'which', 'what', 'also',
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        # Type-token ratio for content words (vocabulary richness)
        if len(content_words) > 0:
            ttr = len(unique_content) / len(content_words)
        else:
            ttr = 0
        
        # Very low TTR suggests extreme repetition (hallucination-like)
        if ttr < 0.2:
            score -= 15
        elif ttr < 0.4:
            score -= 5
        elif ttr > 0.6:
            score += 3
        
        # === 7. Repetition detection (strong hallucination signal) ===
        # Check for repeated phrases (n-grams)
        if total_words >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            max_trigram_repeat = max(trigram_counts.values()) if trigram_counts else 1
            
            if max_trigram_repeat > 5:
                score -= 20
            elif max_trigram_repeat > 3:
                score -= 10
            elif max_trigram_repeat > 2:
                score -= 3
        
        # Check for repeated sentences
        sentence_set = set()
        repeated_sentences = 0
        for sent in sentences:
            normalized = re.sub(r'\s+', ' ', sent.lower().strip())
            if normalized in sentence_set:
                repeated_sentences += 1
            sentence_set.add(normalized)
        
        if num_sentences > 1:
            repeat_ratio = repeated_sentences / num_sentences
            score -= repeat_ratio * 15
        
        # === 8. Structural credibility ===
        # Causal/explanatory connectors suggest reasoning
        explanatory_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas\s+a\s+result\b', r'\bconsequently\b', r'\bdue\s+to\b',
            r'\bthis\s+(?:means|implies|suggests|indicates)\b',
            r'\bin\s+other\s+words\b', r'\bspecifically\b',
            r'\bfor\s+example\b', r'\bfor\s+instance\b', r'\bsuch\s+as\b',
            r'\bincluding\b', r'\bnamely\b',
        ]
        
        explanatory_count = 0
        for pattern in explanatory_patterns:
            explanatory_count += len(re.findall(pattern, response_lower))
        
        # Reward explanatory language (shows reasoning, not just assertion)
        score += min(explanatory_count * 1.5, 6)
        
        # === 9. Comparative/contrastive structure (shows nuance) ===
        contrast_patterns = [
            r'\bhowever\b', r'\bon\s+the\s+other\s+hand\b', r'\bwhereas\b',
            r'\bwhile\b', r'\balthough\b', r'\bdespite\b', r'\bin\s+contrast\b',
            r'\bnevertheless\b', r'\bnot\s+(?:all|every|always)\b',
            r'\bdepends?\s+on\b', r'\bvaries?\b',
        ]
        
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, response_lower))
        
        score += min(contrast_count * 1.5, 5)
        
        # === 10. Response length & completeness ===
        # Very short responses are usually lower quality
        if total_words < 5:
            score -= 15
        elif total_words < 10:
            score -= 8
        elif total_words < 20:
            score -= 3
        elif total_words >= 30:
            score += 3
        elif total_words >= 50:
            score += 5
        
        # Penalize truncated responses (ends mid-word or mid-sentence)
        if response[-1] not in '.!?"\')':
            # Check if it seems truncated
            if len(response) > 50:
                score -= 5
        
        # === 11. Query relevance via key term overlap ===
        query_words = set(re.findall(r'\b[a-z]+\b', query.lower())) - stop_words
        response_content = set(content_words)
        
        if query_words:
            overlap = len(query_words & response_content) / len(query_words)
            score += overlap * 5  # Up to +5 for relevance
        
        # === 12. Definitional/explanatory quality ===
        # Does the response explain rather than just restate?
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['explain', 'describe', 'what is', 'what are', 'how does', 'why']):
            # For explanatory queries, reward longer, more detailed responses
            if num_sentences >= 3:
                score += 3
            if total_words >= 40:
                score += 2
            # Penalize if response is just restating the query
            if total_words < 15:
                score -= 5
        
        # === 13. Penalize empty/placeholder responses ===
        placeholder_patterns = [
            r'^<noinput>$', r'^\[.*\]$', r'^n/a$', r'^none$',
            r'^no\s+(?:response|answer|input)', r'^\s*$',
        ]
        for pattern in placeholder_patterns:
            if re.match(pattern, response.strip(), re.IGNORECASE):
                return 1.0
        
        # === 14. Citation-like patterns (positive signal) ===
        citation_patterns = [
            r'according\s+to', r'studies\s+(?:show|suggest|indicate|have\s+found)',
            r'research\s+(?:shows|suggests|indicates|has\s+found)',
            r'\(\d{4}\)', r'\[\d+\]',  # Year citations, numbered refs
            r'et\s+al\.', r'published\s+in',
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, response_lower))
        
        score += min(citation_count * 2, 6)
        
        # === 15. Sentence complexity variance ===
        # Good responses have varied sentence lengths
        if num_sentences >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Some variance is good (not all same-length sentences)
                cv = std_dev / mean_len if mean_len > 0 else 0
                if 0.2 <= cv <= 0.8:
                    score += 2  # Good variety
                elif cv < 0.1:
                    score -= 1  # Too uniform
        
        # Clamp score to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Never crash - return neutral score
        return 25.0