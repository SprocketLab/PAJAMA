def judging_function(query, response):
    """
    Evaluate response quality based on evidence density and specificity.
    
    This variant uses a unique approach based on:
    1. Named entity density (capitalized multi-word phrases)
    2. Numeric/quantitative information density
    3. Sentence-level information completeness scoring
    4. Specificity lexicon vs vagueness lexicon ratio
    5. Unique noun-like token diversity
    6. Response structural coherence (not just lists/bullets but sentence quality)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        resp = response.strip()
        query_clean = (query or "").strip()
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', resp)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        resp_lower = resp.lower()
        words_lower = [w.lower() for w in words]
        
        # ============================================================
        # FEATURE 1: Capitalized phrase density (proxy for named entities)
        # Look for sequences of capitalized words (2+ words) that aren't sentence starters
        # ============================================================
        sentences = re.split(r'[.!?]+', resp)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        capitalized_phrases = []
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) < 2:
                continue
            # Skip first word (sentence starter), look for capitalized sequences
            i = 1
            while i < len(sent_words):
                clean_w = re.sub(r'[^a-zA-Z]', '', sent_words[i])
                if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                    phrase = [clean_w]
                    j = i + 1
                    while j < len(sent_words):
                        cw2 = re.sub(r'[^a-zA-Z]', '', sent_words[j])
                        if cw2 and cw2[0].isupper() and len(cw2) > 1:
                            phrase.append(cw2)
                            j += 1
                        else:
                            break
                    if len(phrase) >= 1:
                        capitalized_phrases.append(' '.join(phrase))
                    i = j
                else:
                    i += 1
        
        # Count unique named-entity-like phrases
        unique_cap_phrases = set(capitalized_phrases)
        named_entity_score = min(len(unique_cap_phrases) * 1.5, 15.0)
        
        # ============================================================
        # FEATURE 2: Quantitative information density
        # Numbers, percentages, dates, measurements, ranges
        # ============================================================
        # Find all numeric patterns
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', resp)
        percentages = re.findall(r'\d+\.?\d*\s*%', resp)
        dates = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', resp)
        measurements = re.findall(r'\d+\.?\d*\s*(?:km|mi|lb|kg|ft|m|cm|mm|oz|mg|ml|mph|kph|GB|MB|TB|Hz|GHz|MHz)', resp, re.IGNORECASE)
        money = re.findall(r'[\$€£¥]\s*\d[\d,]*\.?\d*|\d[\d,]*\.?\d*\s*(?:dollars|euros|pounds|cents)', resp, re.IGNORECASE)
        
        quant_items = len(set(numbers)) + len(set(percentages)) * 2 + len(set(dates)) * 2 + len(set(measurements)) * 2 + len(set(money)) * 2
        quant_score = min(quant_items * 1.8, 15.0)
        
        # ============================================================
        # FEATURE 3: Specificity lexicon ratio
        # Count specific/precise words vs vague/hedge words
        # ============================================================
        vague_phrases = [
            'many people', 'some people', 'a lot of', 'various factors',
            'it depends', 'there are many', 'there are various', 'in general',
            'generally speaking', 'for the most part', 'more or less',
            'kind of', 'sort of', 'a number of', 'quite a few',
            'several factors', 'many factors', 'some say', 'others say',
            'it is said', 'people say', 'they say', 'arguably',
            'some argue', 'many argue', 'could be', 'might be',
            'perhaps', 'maybe', 'possibly', 'probably',
            'in some cases', 'in many cases', 'sometimes', 'often',
            'usually', 'typically', 'tend to', 'seems to',
            'appears to', 'likely', 'unlikely', 'may or may not',
            'hard to say', 'difficult to say', 'not clear',
            'a variety of', 'numerous', 'countless', 'innumerable',
        ]
        
        vague_count = 0
        for vp in vague_phrases:
            vague_count += len(re.findall(re.escape(vp), resp_lower))
        
        # Specific/precise indicator words
        specific_indicators = [
            'specifically', 'exactly', 'precisely', 'namely',
            'for example', 'for instance', 'such as', 'including',
            'in particular', 'according to', 'based on', 'defined as',
            'known as', 'referred to as', 'located in', 'located at',
            'founded in', 'established in', 'published in', 'created by',
            'developed by', 'invented by', 'designed by', 'built by',
            'written by', 'composed by', 'directed by', 'produced by',
        ]
        
        specific_count = 0
        for sp in specific_indicators:
            specific_count += len(re.findall(re.escape(sp), resp_lower))
        
        vagueness_penalty = min(vague_count * 1.0, 10.0)
        specificity_bonus = min(specific_count * 2.0, 10.0)
        
        # ============================================================
        # FEATURE 4: Unique content word diversity (type-token ratio for content words)
        # ============================================================
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'and', 'but', 'or', 'if', 'while', 'because', 'that', 'this', 'these',
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'up', 'about', 'also', 'just', 'like',
        }
        
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        content_count = len(content_words)
        
        if content_count > 0:
            # Hapax legomena ratio (words appearing only once) - indicates richness
            content_freq = Counter(content_words)
            hapax = sum(1 for w, c in content_freq.items() if c == 1)
            hapax_ratio = hapax / max(len(unique_content), 1)
            
            # Unique content density
            unique_density = len(unique_content) / max(content_count, 1)
            
            diversity_score = (unique_density * 5.0 + hapax_ratio * 3.0)
        else:
            diversity_score = 0.0
        
        diversity_score = min(diversity_score, 8.0)
        
        # ============================================================
        # FEATURE 5: Sentence-level completeness and information density
        # Good sentences have subject-verb-object structure with specific content
        # ============================================================
        valid_sentences = [s for s in sentences if len(s.split()) >= 3]
        
        if len(valid_sentences) > 0:
            # Average content words per sentence
            avg_content_per_sent = content_count / len(valid_sentences)
            sentence_richness = min(avg_content_per_sent / 3.0, 3.0)  # cap at 3
        else:
            sentence_richness = 0.0
        
        # ============================================================
        # FEATURE 6: Repetition penalty
        # Detect repeated phrases/sentences (sign of low quality)
        # ============================================================
        # Check for repeated trigrams
        if len(words_lower) >= 3:
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            repetition_penalty = min(repetition_ratio * 20.0, 10.0)
        else:
            repetition_penalty = 0.0
        
        # ============================================================
        # FEATURE 7: Response length adequacy
        # Too short = likely lacking detail; reasonable length = good
        # ============================================================
        if word_count <= 2:
            length_score = 0.5
        elif word_count <= 5:
            length_score = 1.5
        elif word_count <= 15:
            length_score = 3.0
        elif word_count <= 40:
            length_score = 5.0
        elif word_count <= 100:
            length_score = 6.0
        elif word_count <= 200:
            length_score = 5.5
        else:
            length_score = 5.0
        
        # ============================================================
        # FEATURE 8: Query relevance via shared rare content words
        # ============================================================
        query_words = set(re.findall(r'[a-zA-Z]+', query_clean.lower())) - stop_words
        query_words = {w for w in query_words if len(w) > 2}
        
        if query_words:
            overlap = unique_content & query_words
            relevance_score = min(len(overlap) / max(len(query_words), 1) * 5.0, 5.0)
        else:
            relevance_score = 2.5  # neutral if query has no content words
        
        # ============================================================
        # FEATURE 9: Presence of garbage/HTML/code artifacts penalty
        # ============================================================
        garbage_patterns = [
            r'<[a-zA-Z]+[^>]*>',  # HTML tags
            r'```',  # code blocks
            r'def\s+\w+\s*\(',  # Python functions
            r'import\s+\w+',  # import statements
            r'Input:\s*$',  # empty input prompts
            r'Output:',  # output markers suggesting template
        ]
        
        garbage_count = 0
        for pat in garbage_patterns:
            garbage_count += len(re.findall(pat, resp))
        
        # Only penalize if it seems like the response is mostly garbage
        garbage_ratio = garbage_count / max(word_count / 10, 1)
        garbage_penalty = min(garbage_ratio * 3.0, 8.0)
        
        # Check if query asks for HTML/code - reduce penalty
        code_query_words = {'html', 'tag', 'code', 'program', 'script', 'function', 'css', 'javascript'}
        if query_words & code_query_words:
            garbage_penalty *= 0.2
        
        # ============================================================
        # FEATURE 10: Assertiveness score
        # Direct, declarative statements vs wishy-washy hedging
        # ============================================================
        declarative_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bhas\b', r'\bhave\b', r'\bhad\b',
            r'\bcan\b', r'\bwill\b',
        ]
        
        hedge_words = ['maybe', 'perhaps', 'possibly', 'probably', 'might', 
                       'could', 'seems', 'appears', 'somewhat', 'rather',
                       'fairly', 'quite', 'relatively']
        
        hedge_count = sum(1 for w in words_lower if w in hedge_words)
        hedge_ratio = hedge_count / max(word_count, 1)
        assertiveness_penalty = min(hedge_ratio * 30.0, 5.0)
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        raw_score = (
            named_entity_score * 0.8      # up to 12
            + quant_score * 0.8            # up to 12
            + specificity_bonus * 0.7      # up to 7
            + diversity_score * 0.8        # up to 6.4
            + sentence_richness * 1.5      # up to 4.5
            + length_score * 1.0           # up to 6
            + relevance_score * 0.8        # up to 4
            - vagueness_penalty * 0.6      # up to -6
            - repetition_penalty * 0.8     # up to -8
            - garbage_penalty * 0.7        # up to -5.6
            - assertiveness_penalty * 0.5  # up to -2.5
        )
        
        # Normalize to 0-10 range
        # Theoretical max ~52, typical good response ~25-35
        score = (raw_score / 40.0) * 10.0
        
        # Clamp
        score = max(0.5, min(10.0, score))
        
        # Special case: very short responses that are just noise
        if word_count <= 3 and quant_score == 0 and named_entity_score == 0:
            score = min(score, 2.0)
        
        # Special case: single word/very minimal responses
        if word_count == 1:
            # Could be a correct one-word answer
            if query_words and any(w in resp_lower for w in query_words):
                score = max(score, 3.0)
            else:
                score = min(score, 1.5)
        
        return round(score, 2)
        
    except Exception:
        return 3.0