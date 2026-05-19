def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant uses sentence-level decomposition and scores each sentence for
    factual reliability signals, then aggregates. It also analyzes response structure,
    information density, and discourse coherence markers.
    
    Different from other variants by:
    - Sentence-level scoring with per-sentence feature extraction
    - Information density ratio (content words vs function words)
    - Discourse coherence markers (logical connectors, topic continuity)
    - Named entity density approximation via capitalization patterns
    - Penalizing repetition at the sentence level (not just word overlap)
    - Using entropy of character distribution as a quality signal
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses are almost always low quality
        if len(response_stripped) < 5:
            return 0.5
        
        # === Split into sentences ===
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'[a-zA-Z]+', response_stripped.lower())
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # === 1. Character-level entropy (measures information richness) ===
        char_counts = Counter(response_stripped.lower())
        total_chars = max(len(response_stripped), 1)
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Good text typically has entropy between 3.5 and 5.0
        # Very low entropy = repetitive, very high = random
        entropy_score = 0.0
        if char_entropy < 2.0:
            entropy_score = 0.0
        elif char_entropy < 3.5:
            entropy_score = (char_entropy - 2.0) / 1.5
        elif char_entropy <= 5.0:
            entropy_score = 1.0
        else:
            entropy_score = max(0.0, 1.0 - (char_entropy - 5.0) / 2.0)
        score += entropy_score * 1.0  # max 1.0
        
        # === 2. Information density: content words vs function words ===
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'while', 'where', 'how', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
            'their', 'there', 'here', 'up', 'out', 'about', 'then', 'also'
        }
        content_words = [w for w in words if w not in function_words]
        content_ratio = len(content_words) / num_words if num_words > 0 else 0
        # Good content ratio is around 0.4-0.7
        density_score = min(content_ratio / 0.55, 1.0) if content_ratio <= 0.55 else max(0, 1.0 - (content_ratio - 0.55) / 0.45)
        score += density_score * 0.8  # max 0.8
        
        # === 3. Named entity density approximation ===
        # Look for capitalized words that aren't sentence starters
        original_words = re.findall(r'\b[A-Za-z]+\b', response_stripped)
        named_entity_count = 0
        for i, w in enumerate(original_words):
            if i == 0:
                continue
            # Check if previous character context suggests sentence start
            if w[0].isupper() and len(w) > 1:
                # Check it's not after a sentence boundary
                prefix = response_stripped[:response_stripped.find(w, max(0, response_stripped.find(w) - 1))]
                if not prefix.rstrip().endswith(('.', '!', '?', ':')):
                    named_entity_count += 1
        
        ne_density = named_entity_count / num_words if num_words > 0 else 0
        # Some named entities are good (0.02-0.15 range), too many might be noise
        ne_score = min(ne_density / 0.08, 1.0) if ne_density <= 0.08 else max(0, 1.0 - (ne_density - 0.08) / 0.2)
        score += ne_score * 0.7  # max 0.7
        
        # === 4. Specific numbers and dates ===
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', response_stripped)
        date_patterns = re.findall(r'\b(?:19|20)\d{2}\b', response_stripped)
        
        num_count = len(numbers)
        date_count = len(date_patterns)
        
        specificity_score = 0.0
        if num_count > 0:
            specificity_score += min(num_count / 3.0, 1.0) * 0.4
        if date_count > 0:
            specificity_score += min(date_count / 2.0, 1.0) * 0.4
        
        # Check for overly precise unsourced statistics (red flag)
        overly_precise = re.findall(r'\b\d{1,3}\.\d{3,}\s*%', response_stripped)
        if overly_precise:
            specificity_score -= 0.3 * len(overly_precise)
        
        score += max(0, min(specificity_score, 0.8))  # max 0.8
        
        # === 5. Appropriate hedging language ===
        hedging_phrases = [
            'it is difficult to', 'it is hard to', 'approximately', 'roughly',
            'estimated', 'generally', 'typically', 'often', 'usually',
            'in most cases', 'it depends', 'may vary', 'can vary',
            'it is believed', 'according to', 'research suggests',
            'studies show', 'evidence suggests', 'it appears',
            'likely', 'unlikely', 'possibly', 'perhaps', 'arguably',
            'in general', 'tends to', 'on average', 'it seems',
            'however', 'although', 'while', 'despite', 'nevertheless',
            'subject to', 'depending on', 'varies depending'
        ]
        response_lower = response_stripped.lower()
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
        hedge_score = min(hedge_count / 3.0, 1.0)
        score += hedge_score * 0.8  # max 0.8
        
        # === 6. Sensationalism and conspiracy red flags ===
        red_flag_patterns = [
            r'\b(?:shocking|unbelievable|mind-?blowing|insane|crazy)\b',
            r'\b(?:they don\'t want you to know|hidden truth|wake up|sheeple)\b',
            r'\b(?:exposed|bombshell|breaking|jaw-?dropping)\b',
            r'\b(?:100%|totally|absolutely|completely|definitely)\s+(?:true|false|wrong|right|certain)\b',
            r'\b(?:always|never|every single|without exception)\b',
            r'!!!+',
            r'\?\?\?+',
            r'ALL CAPS [A-Z]{5,}',
        ]
        red_flag_count = 0
        for pattern in red_flag_patterns:
            red_flag_count += len(re.findall(pattern, response_stripped, re.IGNORECASE))
        
        # Penalize
        red_flag_penalty = min(red_flag_count * 0.3, 1.5)
        score -= red_flag_penalty
        
        # === 7. Discourse coherence markers ===
        coherence_markers = [
            'therefore', 'consequently', 'as a result', 'furthermore',
            'moreover', 'in addition', 'additionally', 'for example',
            'for instance', 'specifically', 'in particular', 'namely',
            'first', 'second', 'third', 'finally', 'in conclusion',
            'to summarize', 'in summary', 'on the other hand',
            'conversely', 'similarly', 'likewise', 'in contrast',
            'meanwhile', 'subsequently', 'previously', 'notably',
            'importantly', 'significantly', 'essentially'
        ]
        coherence_count = sum(1 for m in coherence_markers if m in response_lower)
        coherence_score = min(coherence_count / 3.0, 1.0)
        score += coherence_score * 0.6  # max 0.6
        
        # === 8. Sentence-level repetition penalty ===
        if num_sentences > 1:
            sentence_set = set()
            duplicate_count = 0
            for s in sentences:
                s_normalized = ' '.join(s.lower().split())
                if s_normalized in sentence_set:
                    duplicate_count += 1
                sentence_set.add(s_normalized)
            
            # Also check for near-duplicates using word overlap
            near_dup_count = 0
            sent_word_sets = [set(re.findall(r'[a-z]+', s.lower())) for s in sentences]
            for i in range(len(sent_word_sets)):
                for j in range(i + 1, len(sent_word_sets)):
                    if sent_word_sets[i] and sent_word_sets[j]:
                        overlap = len(sent_word_sets[i] & sent_word_sets[j])
                        union = len(sent_word_sets[i] | sent_word_sets[j])
                        if union > 0 and overlap / union > 0.8:
                            near_dup_count += 1
            
            repetition_ratio = (duplicate_count + near_dup_count * 0.5) / num_sentences
            score -= repetition_ratio * 2.0
        
        # === 9. Response length adequacy ===
        # Very short responses to substantive queries are usually bad
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_len = len(query_words)
        
        length_score = 0.0
        if num_words < 3:
            length_score = 0.0
        elif num_words < 10:
            length_score = 0.3
        elif num_words < 30:
            length_score = 0.6
        elif num_words < 200:
            length_score = 1.0
        elif num_words < 500:
            length_score = 0.8
        else:
            length_score = 0.6
        
        score += length_score * 1.2  # max 1.2
        
        # === 10. Citation-like patterns ===
        citation_patterns = [
            r'according to',
            r'cited by',
            r'source[s]?:',
            r'reference[s]?:',
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2020) style
            r'et al\.',
            r'published in',
            r'reported by',
            r'as noted by',
            r'as stated in',
        ]
        citation_count = 0
        for pat in citation_patterns:
            citation_count += len(re.findall(pat, response_lower))
        
        citation_score = min(citation_count / 2.0, 1.0)
        score += citation_score * 0.6  # max 0.6
        
        # === 11. HTML/code contamination penalty ===
        html_tags = re.findall(r'<[a-zA-Z/][^>]*>', response_stripped)
        code_indicators = re.findall(r'(?:import |def |class |print\(|console\.log)', response_stripped)
        
        # Only penalize if query doesn't ask for code/HTML
        query_asks_code = any(w in query.lower() for w in ['code', 'html', 'program', 'script', 'tag', 'function'])
        if not query_asks_code:
            contamination = len(html_tags) + len(code_indicators)
            if contamination > 2:
                score -= min(contamination * 0.3, 2.0)
        
        # === 12. Topical relevance via shared content words ===
        query_content = set(w for w in query_words if w not in function_words and len(w) > 2)
        response_content = set(w for w in [w.lower() for w in re.findall(r'[a-zA-Z]+', response_stripped)] if w not in function_words and len(w) > 2)
        
        if query_content and response_content:
            relevance = len(query_content & response_content) / len(query_content)
            score += relevance * 1.0  # max 1.0
        
        # === 13. Structural quality: presence of clear structure ===
        has_list = bool(re.search(r'(?:^|\n)\s*[\-\*\•]\s', response_stripped)) or bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response_stripped))
        has_paragraphs = response_stripped.count('\n\n') >= 1
        
        structure_score = 0.0
        if has_list:
            structure_score += 0.3
        if has_paragraphs and num_words > 50:
            structure_score += 0.2
        score += structure_score  # max 0.5
        
        # === 14. Word-level diversity (unique trigrams) ===
        if num_words >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            if trigrams:
                trigram_diversity = len(set(trigrams)) / len(trigrams)
                # High diversity is good (> 0.7), low is repetitive
                diversity_bonus = max(0, (trigram_diversity - 0.5) * 1.5)
                score += min(diversity_bonus, 0.8)
        
        # === Normalize to 0-10 range ===
        # Max theoretical score is approximately: 1.0 + 0.8 + 0.7 + 0.8 + 0.8 + 0.6 + 1.2 + 0.6 + 1.0 + 0.5 + 0.8 = 8.8
        # Plus some bonuses, minus penalties
        # Scale to 0-10
        final_score = max(0.0, min(10.0, score * 1.15))
        
        # Round to 1 decimal
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0