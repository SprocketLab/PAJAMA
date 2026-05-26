def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant uses a fundamentally different approach: it analyzes the response
    at the SENTENCE level, scoring each sentence for factual reliability signals,
    then aggregates using weighted statistics. It also uses character-level analysis
    (punctuation ratios, capitalization patterns) and structural coherence metrics
    rather than simple word overlap or vocabulary diversity.
    """
    try:
        import re
        import math
        from collections import Counter
        
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
        
        # ========== SENTENCE-LEVEL ANALYSIS ==========
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 1.0
        
        sentence_scores = []
        
        for sent in sentences:
            sent_score = 5.0  # baseline per sentence
            
            # --- Factual specificity signals ---
            # Dates (years, full dates)
            year_matches = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', sent)
            sent_score += min(len(year_matches) * 0.8, 2.4)
            
            # Specific numbers (not just years)
            number_matches = re.findall(r'\b\d+[,.]?\d*\b', sent)
            non_year_numbers = [n for n in number_matches if n not in [m for m in year_matches]]
            sent_score += min(len(non_year_numbers) * 0.3, 1.5)
            
            # Proper nouns (capitalized words not at sentence start)
            words_in_sent = sent.split()
            if len(words_in_sent) > 1:
                proper_nouns = [w for w in words_in_sent[1:] if w and w[0].isupper() and w.isalpha()]
                sent_score += min(len(proper_nouns) * 0.4, 2.0)
            
            # --- Appropriate hedging (good for uncertain claims) ---
            hedging_phrases = [
                r'\bit is (difficult|hard|challenging) to\b',
                r'\b(may|might|could|possibly|perhaps|likely|unlikely)\b',
                r'\b(generally|typically|often|usually|approximately|roughly|around)\b',
                r'\b(according to|research suggests|studies indicate|evidence suggests)\b',
                r'\b(it appears|it seems|tends to)\b',
                r'\bhowever\b',
                r'\b(some|many|most|few|several) (experts|researchers|scholars|historians)\b',
                r'\bdepending on\b',
                r'\bnot without\b',
                r'\bcan vary\b',
                r'\bsubjective\b',
            ]
            hedging_count = sum(1 for p in hedging_phrases if re.search(p, sent, re.IGNORECASE))
            sent_score += min(hedging_count * 0.7, 2.1)
            
            # --- Red flags: hallucination indicators ---
            # Overly precise unsourced statistics
            precise_stats = re.findall(r'\b\d{1,3}\.\d{2,}%\b', sent)
            sent_score -= len(precise_stats) * 1.0
            
            # Absolute claims
            absolute_patterns = [
                r'\b(always|never|absolutely|definitely|certainly|undoubtedly|without (a )?doubt)\b',
                r'\b(everyone knows|it is (a )?fact that|clearly|obviously)\b',
                r'\b(the truth is|in reality|wake up)\b',
            ]
            absolute_count = sum(1 for p in absolute_patterns if re.search(p, sent, re.IGNORECASE))
            sent_score -= absolute_count * 0.8
            
            # Sensationalism / conspiracy language
            sensational_patterns = [
                r'\b(shocking|bombshell|explosive|devastating|mind-blowing)\b',
                r'\b(cover.?up|conspiracy|they don\'t want you to know|hidden truth|secret(ly)?)\b',
                r'\b(exposed|revealed|unmasked|whistleblow)\b',
                r'\b(mainstream media|big pharma|deep state|elites)\b',
                r'!!!|!!\s*$',
            ]
            sensational_count = sum(1 for p in sensational_patterns if re.search(p, sent, re.IGNORECASE))
            sent_score -= sensational_count * 1.5
            
            # --- Citation-like references ---
            citation_patterns = [
                r'\b(according to|as (noted|stated|described|reported) (by|in))\b',
                r'\b(published|journal|study|research|survey|report)\b',
                r'\((19|20)\d{2}\)',  # parenthetical year citations
                r'\b(University|Institute|Foundation|Organization)\b',
            ]
            citation_count = sum(1 for p in citation_patterns if re.search(p, sent, re.IGNORECASE))
            sent_score += min(citation_count * 0.6, 1.8)
            
            # Sentence too short to be informative
            if len(words_in_sent) < 3:
                sent_score -= 2.0
            
            sentence_scores.append(max(0.0, min(10.0, sent_score)))
        
        # Weighted aggregation: longer sentences contribute more
        if sentence_scores:
            weights = [max(1, len(s.split())) for s in sentences[:len(sentence_scores)]]
            total_weight = sum(weights)
            weighted_avg = sum(s * w for s, w in zip(sentence_scores, weights)) / total_weight if total_weight > 0 else 5.0
        else:
            weighted_avg = 3.0
        
        # ========== STRUCTURAL COHERENCE ==========
        structural_score = 0.0
        
        # Sentence count bonus (more complete responses tend to be better)
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 4]
        if len(meaningful_sentences) >= 3:
            structural_score += 1.5
        elif len(meaningful_sentences) >= 2:
            structural_score += 0.8
        elif len(meaningful_sentences) >= 1:
            structural_score += 0.3
        
        # Check for repetition at sentence level (bad sign)
        if len(sentences) >= 2:
            normalized_sents = [re.sub(r'\s+', ' ', s.lower().strip()) for s in sentences]
            unique_ratio = len(set(normalized_sents)) / len(normalized_sents)
            if unique_ratio < 0.5:
                structural_score -= 2.0
            elif unique_ratio < 0.75:
                structural_score -= 1.0
        
        # Check for incoherent artifacts (HTML, code, repeated patterns)
        artifact_patterns = [
            r'<[a-z]+[^>]*>',  # HTML tags
            r'```',  # code blocks
            r'(Input:|Output:){2,}',  # repeated I/O markers
            r'(Question:.*?Answer:.*?){2,}',  # Q&A spam
            r'def\s+\w+\s*\(',  # function definitions
            r'import\s+\w+',  # import statements
        ]
        artifact_count = sum(1 for p in artifact_patterns if re.search(p, response, re.IGNORECASE))
        structural_score -= artifact_count * 0.8
        
        # ========== CHARACTER-LEVEL ANALYSIS ==========
        char_score = 0.0
        
        # Punctuation ratio (well-written text has moderate punctuation)
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        punct_chars = sum(1 for c in response_stripped if c in '.,;:!?()-"\'')
        total_chars = len(response_stripped)
        
        if alpha_chars > 0:
            punct_ratio = punct_chars / alpha_chars
            # Good range: 0.03 to 0.15
            if 0.03 <= punct_ratio <= 0.15:
                char_score += 0.8
            elif punct_ratio > 0.3:
                char_score -= 1.0  # Too much punctuation
        
        # ALL CAPS ratio (shouting = bad)
        words = response_stripped.split()
        if len(words) > 3:
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha())
            caps_ratio = caps_words / len(words)
            if caps_ratio > 0.3:
                char_score -= 1.5
        
        # ========== RESPONSE COMPLETENESS ==========
        completeness_score = 0.0
        
        # Does response end mid-sentence? (truncation)
        last_char = response_stripped[-1] if response_stripped else ''
        if last_char in '.!?)"\':':
            completeness_score += 0.5
        elif last_char.isalpha():
            # Might be truncated
            completeness_score -= 0.3
        
        # Response length adequacy relative to query complexity
        query_words = query.split()
        response_words = response_stripped.split()
        
        if len(response_words) < 3:
            completeness_score -= 2.0
        elif len(response_words) < 10:
            completeness_score -= 0.5
        elif len(response_words) > 20:
            completeness_score += 0.5
        
        # ========== QUERY RELEVANCE (lightweight) ==========
        # Use character trigram overlap instead of word overlap
        relevance_score = 0.0
        
        def get_trigrams(text):
            text = text.lower()
            return set(text[i:i+3] for i in range(len(text) - 2))
        
        if len(query) >= 3 and len(response_stripped) >= 3:
            q_trigrams = get_trigrams(query)
            r_trigrams = get_trigrams(response_stripped)
            if q_trigrams:
                trigram_overlap = len(q_trigrams & r_trigrams) / len(q_trigrams)
                relevance_score += trigram_overlap * 1.5
        
        # Check if response contains key nouns from query
        # Extract content words (longer words likely to be content-bearing)
        query_content = set(w.lower() for w in query_words if len(w) > 4 and w.isalpha())
        response_lower = response_stripped.lower()
        if query_content:
            content_hits = sum(1 for w in query_content if w in response_lower)
            content_ratio = content_hits / len(query_content)
            relevance_score += content_ratio * 1.0
        
        # ========== INFORMATION DENSITY ==========
        # Ratio of unique content words to total words
        density_score = 0.0
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
                      'not', 'no', 'if', 'then', 'than', 'that', 'this', 'these', 'those',
                      'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my',
                      'your', 'his', 'her', 'our', 'their', 'who', 'what', 'which', 'when',
                      'where', 'how', 'why', 'all', 'each', 'every', 'both', 'some', 'any'}
        
        content_words = [w.lower() for w in response_words if w.lower() not in stop_words and w.isalpha()]
        if len(response_words) > 5:
            if content_words:
                unique_content = set(content_words)
                diversity = len(unique_content) / len(content_words) if content_words else 0
                # Moderate diversity is good (0.4-0.8)
                if 0.4 <= diversity <= 0.85:
                    density_score += 1.0
                elif diversity < 0.3:
                    density_score -= 1.0  # Very repetitive
                
                # Content word ratio
                content_ratio = len(content_words) / len(response_words)
                if content_ratio > 0.35:
                    density_score += 0.5
        
        # ========== COMBINE ALL SCORES ==========
        # weighted_avg is on 0-10 scale, others are adjustments
        final_score = (
            weighted_avg * 0.55 +          # Sentence-level factual analysis (main signal)
            structural_score * 0.8 +        # Structural coherence
            char_score * 0.6 +              # Character-level quality
            completeness_score * 0.7 +      # Response completeness
            relevance_score * 0.5 +         # Query relevance
            density_score * 0.5             # Information density
        )
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This maps the range more discriminatively
        midpoint = 5.0
        steepness = 0.5
        transformed = 10.0 / (1.0 + math.exp(-steepness * (final_score - midpoint)))
        
        return round(transformed, 2)
        
    except Exception:
        # Fallback: return a neutral score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0