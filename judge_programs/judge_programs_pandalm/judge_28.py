def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level structure quality (well-formed informative sentences)
    2. Specificity signals (named entities, numbers, technical terms)
    3. Hallucination red-flags (repetition, absolute claims, sensationalism)
    4. Appropriate epistemic calibration (hedging vs. overconfidence)
    5. Information density and diversity
    """
    try:
        if not response or not isinstance(response, str) or response.strip() == "":
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        # === 1. SENTENCE-LEVEL ANALYSIS ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = response_clean.split()
        num_words = len(words)
        
        if num_words < 2:
            return 1.0
        
        # Average sentence length (prefer moderate: 10-25 words)
        avg_sent_len = num_words / num_sentences
        if 10 <= avg_sent_len <= 25:
            sent_len_score = 1.0
        elif 5 <= avg_sent_len < 10 or 25 < avg_sent_len <= 40:
            sent_len_score = 0.6
        else:
            sent_len_score = 0.2
        
        # === 2. REPETITION DETECTION (hallucination red-flag) ===
        # Bigram repetition rate
        words_lower = [w.lower().strip('.,!?;:()[]"\'') for w in words]
        words_lower = [w for w in words_lower if w]
        
        if len(words_lower) >= 2:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            unique_bigrams = len(bigram_counts)
            bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
            
            # Check for highly repeated bigrams
            max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
            bigram_repeat_ratio = max_bigram_freq / total_bigrams if total_bigrams > 0 else 0
        else:
            bigram_diversity = 0.5
            bigram_repeat_ratio = 0
        
        # Trigram repetition
        if len(words_lower) >= 3:
            trigrams = [(words_lower[i], words_lower[i+1], words_lower[i+2]) for i in range(len(words_lower)-2)]
            trigram_counts = Counter(trigrams)
            max_trigram_freq = max(trigram_counts.values()) if trigram_counts else 0
            trigram_repeat_ratio = max_trigram_freq / len(trigrams) if trigrams else 0
        else:
            trigram_repeat_ratio = 0
        
        # Severe repetition penalty
        repetition_penalty = 0
        if bigram_repeat_ratio > 0.3:
            repetition_penalty += 3.0
        elif bigram_repeat_ratio > 0.15:
            repetition_penalty += 1.5
        
        if trigram_repeat_ratio > 0.2:
            repetition_penalty += 3.0
        elif trigram_repeat_ratio > 0.1:
            repetition_penalty += 1.5
        
        # Sentence-level repetition
        sent_lower = [s.lower().strip() for s in sentences]
        sent_counter = Counter(sent_lower)
        if sent_lower:
            unique_sent_ratio = len(sent_counter) / len(sent_lower)
        else:
            unique_sent_ratio = 1.0
        
        if unique_sent_ratio < 0.5:
            repetition_penalty += 2.0
        
        # Word-level vocabulary diversity
        if len(words_lower) > 0:
            vocab_diversity = len(set(words_lower)) / len(words_lower)
        else:
            vocab_diversity = 0
        
        # === 3. SPECIFICITY SIGNALS ===
        # Numbers and dates
        number_pattern = re.findall(r'\b\d+[\d,.]*\b', response_clean)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', response_clean)
        
        # Capitalized multi-word phrases (potential named entities)
        named_entity_pattern = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response_clean)
        
        # Technical/specific vocabulary (longer words, often more specific)
        technical_words = [w for w in words_lower if len(w) > 8 and w.isalpha()]
        
        specificity_score = 0
        specificity_score += min(len(number_pattern) * 0.3, 1.5)
        specificity_score += min(len(year_pattern) * 0.5, 1.5)
        specificity_score += min(len(named_entity_pattern) * 0.4, 1.5)
        specificity_score += min(len(technical_words) / max(num_words, 1) * 10, 1.5)
        
        # === 4. EPISTEMIC CALIBRATION ===
        response_lower = response_clean.lower()
        
        # Appropriate hedging phrases (good sign)
        hedging_phrases = [
            'may ', 'might ', 'could ', 'possibly', 'likely', 'unlikely',
            'suggests that', 'indicates that', 'it appears', 'it seems',
            'generally', 'typically', 'often', 'usually', 'tends to',
            'in some cases', 'depending on', 'according to', 'research suggests',
            'evidence suggests', 'studies show', 'approximately', 'roughly',
            'estimated', 'can vary', 'it is believed', 'widely considered'
        ]
        
        hedge_count = sum(1 for h in hedging_phrases if h in response_lower)
        hedge_score = min(hedge_count * 0.4, 2.0)
        
        # Overconfidence / absolute claims (red flag)
        absolute_phrases = [
            'always ', 'never ', 'definitely ', 'absolutely ', 'certainly ',
            'without a doubt', 'undeniably', 'unquestionably', 'guaranteed',
            'proven fact', 'everyone knows', 'nobody can', 'it is certain',
            '100%', 'impossible to', 'the only way'
        ]
        
        absolute_count = sum(1 for a in absolute_phrases if a in response_lower)
        overconfidence_penalty = min(absolute_count * 0.4, 2.0)
        
        # Sensationalism red-flags
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'insane', 'crazy',
            'devastating', 'explosive', 'bombshell', 'conspiracy', 'cover-up',
            'they don\'t want you to know', 'wake up', 'sheeple', 'mainstream media',
            'big pharma', 'deep state', 'hoax', 'scam', 'fraud',
            'exposed', 'revealed', 'secret', 'hidden truth', 'suppressed'
        ]
        
        sensational_count = sum(1 for s in sensational_words if s in response_lower)
        sensational_penalty = min(sensational_count * 0.8, 3.0)
        
        # === 5. STRUCTURAL QUALITY ===
        # Causal/logical connectors indicate reasoning
        reasoning_connectors = [
            'because', 'therefore', 'however', 'although', 'while',
            'in contrast', 'on the other hand', 'for example', 'for instance',
            'such as', 'specifically', 'in particular', 'moreover', 'furthermore',
            'as a result', 'consequently', 'this means', 'in addition',
            'similarly', 'conversely', 'nevertheless', 'thus'
        ]
        
        connector_count = sum(1 for c in reasoning_connectors if c in response_lower)
        structure_score = min(connector_count * 0.3, 2.0)
        
        # === 6. INFORMATION DENSITY ===
        # Ratio of content words to function words
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'not', 'no', 'nor', 'so', 'yet', 'both', 'either', 'neither',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'our', 'their', 'which', 'who', 'whom', 'what', 'where',
            'when', 'how', 'than', 'then', 'also', 'very', 'just', 'more'
        }
        
        content_words = [w for w in words_lower if w not in function_words and w.isalpha()]
        content_ratio = len(content_words) / max(len(words_lower), 1)
        info_density_score = content_ratio * 3.0  # scale to ~1.5 max
        
        # === 7. QUERY RELEVANCE (semantic overlap) ===
        query_words = set(query_clean.lower().split())
        query_words -= function_words
        query_words = {w.strip('.,!?;:()[]"\'') for w in query_words if len(w) > 2}
        
        response_word_set = set(words_lower)
        
        if query_words:
            relevance = len(query_words & response_word_set) / len(query_words)
        else:
            relevance = 0.5
        
        relevance_score = relevance * 2.0
        
        # === 8. RESPONSE LENGTH APPROPRIATENESS ===
        # Not too short, not excessively long
        if num_words < 5:
            length_score = 0.2
        elif num_words < 15:
            length_score = 0.5
        elif num_words < 30:
            length_score = 0.8
        elif num_words <= 200:
            length_score = 1.0
        elif num_words <= 400:
            length_score = 0.9
        else:
            length_score = 0.7
        
        # === 9. CITATION-LIKE INDICATORS ===
        citation_patterns = [
            r'according to', r'research\s+(shows?|suggests?|indicates?)',
            r'studies?\s+(show|suggest|indicate|find|found)',
            r'data\s+(shows?|suggests?|indicates?)',
            r'\([^)]*\d{4}[^)]*\)',  # parenthetical with year
            r'et al\.', r'source:', r'reference:',
            r'published in', r'journal of'
        ]
        
        citation_count = sum(1 for p in citation_patterns if re.search(p, response_lower))
        citation_score = min(citation_count * 0.5, 1.5)
        
        # === 10. EXPLANATION DEPTH ===
        # Does the response explain WHY or HOW, not just WHAT
        explanation_markers = [
            'this means', 'this is because', 'the reason', 'in other words',
            'which means', 'which allows', 'which enables', 'resulting in',
            'this suggests', 'this indicates', 'this implies', 'to clarify',
            'to explain', 'put simply', 'essentially'
        ]
        
        explanation_count = sum(1 for e in explanation_markers if e in response_lower)
        explanation_score = min(explanation_count * 0.4, 1.5)
        
        # === COMPOSITE SCORE ===
        score = 0.0
        
        # Positive contributions
        score += sent_len_score * 1.0          # 0-1.0
        score += specificity_score              # 0-6.0
        score += hedge_score                    # 0-2.0
        score += structure_score                # 0-2.0
        score += info_density_score             # 0-3.0
        score += relevance_score                # 0-2.0
        score += length_score * 2.0             # 0-2.0
        score += citation_score                 # 0-1.5
        score += explanation_score              # 0-1.5
        score += bigram_diversity * 2.0         # 0-2.0
        score += unique_sent_ratio * 1.0        # 0-1.0
        score += vocab_diversity * 2.0          # 0-2.0
        
        # Negative contributions
        score -= repetition_penalty             # 0-8.0
        score -= overconfidence_penalty          # 0-2.0
        score -= sensational_penalty             # 0-3.0
        
        # Normalize to 0-10 range
        # Max positive ~26, min negative ~-13
        # Typical good response: ~12-18, typical bad: ~3-8
        score = max(0.0, score)
        score = min(score, 26.0)
        final_score = (score / 26.0) * 10.0
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Minimal fallback
            if response and len(response.strip()) > 10:
                return 3.0
            return 1.0
        except Exception:
            return 1.0