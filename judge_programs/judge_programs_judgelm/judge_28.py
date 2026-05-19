def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level structure analysis (well-formed sentences indicate higher quality)
    2. Information density via named entity patterns and specific references
    3. Epistemic calibration (appropriate confidence levels)
    4. Red flag detection for hallucination patterns
    5. Response coherence relative to query
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses are almost always low quality
        if len(response_stripped) < 5:
            return 0.5
        
        score = 0.0
        
        # ============================================================
        # 1. SENTENCE STRUCTURE ANALYSIS
        # Split into sentences and analyze each one
        # ============================================================
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        # Score based on having multiple well-formed sentences
        if num_sentences >= 3:
            score += 1.5
        elif num_sentences >= 2:
            score += 1.0
        elif num_sentences >= 1:
            score += 0.5
        
        # Average sentence length (well-formed sentences tend to be 8-25 words)
        if num_sentences > 0:
            avg_words_per_sentence = sum(len(s.split()) for s in sentences) / num_sentences
            if 8 <= avg_words_per_sentence <= 25:
                score += 1.0
            elif 5 <= avg_words_per_sentence < 8 or 25 < avg_words_per_sentence <= 35:
                score += 0.5
            # Very short or very long sentences get nothing
        
        # ============================================================
        # 2. INFORMATION DENSITY - Named entities and specifics
        # ============================================================
        # Capitalized multi-word phrases (potential proper nouns/names)
        proper_noun_pattern = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response_stripped)
        proper_noun_score = min(len(proper_noun_pattern) * 0.3, 1.5)
        score += proper_noun_score
        
        # Dates (years, full dates)
        date_patterns = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', response_stripped)
        date_score = min(len(date_patterns) * 0.25, 0.75)
        score += date_score
        
        # Numbers used in context (not just random digits)
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand|km|miles|years|months|days|hours|people|characters|pages))', response_stripped.lower())
        number_score = min(len(numbers) * 0.2, 0.6)
        score += number_score
        
        # Specific reference indicators (titles, quoted works, etc.)
        reference_indicators = re.findall(r"['\"][\w\s]+['\"]|(?:known as|called|titled|named|referred to as)", response_stripped, re.IGNORECASE)
        ref_score = min(len(reference_indicators) * 0.2, 0.6)
        score += ref_score
        
        # ============================================================
        # 3. EPISTEMIC CALIBRATION
        # Appropriate hedging and confidence calibration
        # ============================================================
        response_lower = response_stripped.lower()
        
        # Good epistemic markers (showing appropriate uncertainty)
        good_hedging = [
            'it is difficult to', 'it is hard to', 'approximately', 'roughly',
            'generally', 'typically', 'often', 'usually', 'tends to',
            'may vary', 'can vary', 'depending on', 'it depends',
            'is considered', 'is widely regarded', 'is often cited',
            'according to', 'research suggests', 'studies show',
            'it is believed', 'scholars suggest', 'historians note',
            'is thought to', 'is estimated', 'however', 'although',
            'on the other hand', 'it should be noted', 'importantly',
            'in some cases', 'for example', 'for instance', 'such as',
            'including', 'among others', 'one of the', 'is also known as'
        ]
        
        hedge_count = sum(1 for h in good_hedging if h in response_lower)
        hedge_score = min(hedge_count * 0.25, 1.2)
        score += hedge_score
        
        # ============================================================
        # 4. RED FLAG DETECTION
        # ============================================================
        penalties = 0.0
        
        # Overly absolute claims without evidence
        absolute_phrases = [
            'everyone knows', 'it is obvious', 'clearly the best',
            'without a doubt', 'undeniably', 'the truth is',
            'wake up', 'they don\'t want you to know', 'exposed',
            'the real truth', 'mainstream media', 'cover up',
            'conspiracy', 'sheeple', 'big pharma', 'illuminati',
            'secret agenda', 'guaranteed', '100% certain',
            'always works', 'never fails', 'proven fact that'
        ]
        
        absolute_count = sum(1 for a in absolute_phrases if a in response_lower)
        penalties += absolute_count * 0.5
        
        # Repetition detection (hallucination indicator)
        words = response_lower.split()
        if len(words) > 10:
            # Check for repeated phrases (trigrams)
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            if trigram_counts:
                max_trigram_repeat = max(trigram_counts.values())
                total_trigrams = len(trigrams)
                if max_trigram_repeat > 3 and total_trigrams > 0:
                    repeat_ratio = max_trigram_repeat / total_trigrams
                    penalties += min(repeat_ratio * 10, 3.0)
        
        # Repeated sentences (copy-paste indicator)
        if num_sentences > 2:
            sentence_set = set(s.lower().strip() for s in sentences if len(s.strip()) > 10)
            if len(sentence_set) < num_sentences * 0.6:
                penalties += 1.5
        
        # HTML/code in non-code responses
        query_lower = query.lower()
        expects_code = any(kw in query_lower for kw in ['html', 'code', 'program', 'script', 'function', 'tag'])
        if not expects_code:
            html_tags = re.findall(r'<[a-z/][^>]*>', response_lower)
            if len(html_tags) > 2:
                penalties += 1.0
        
        # Random question-answer pairs appended (a specific bad pattern)
        qa_pattern = re.findall(r'(?:question|answer|input|output)\s*:', response_lower)
        if len(qa_pattern) > 2:
            penalties += 1.5
        
        # Response is just echoing the query without adding info
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        if len(response_words) > 0 and len(query_words) > 3:
            overlap = len(query_words & response_words) / len(response_words)
            if overlap > 0.8 and len(response_words) < len(query_words) * 1.5:
                penalties += 1.0
        
        # ============================================================
        # 5. COHERENCE & RELEVANCE SIGNALS
        # ============================================================
        
        # Content words from query appearing in response (topical relevance)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                     'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                     'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which',
                     'who', 'whom', 'where', 'when', 'why', 'how', 'and', 'but',
                     'or', 'not', 'no', 'if', 'then', 'than', 'so', 'as', 'up',
                     'out', 'about', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'also', 'just', 'more', 'most',
                     'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many',
                     'much', 'own', 'other', 'another', 'such', 'only', 'same',
                     'there', 'here', 'very', 'too', 'quite'}
        
        query_content = [w for w in re.findall(r'\b[a-z]+\b', query_lower) if w not in stopwords and len(w) > 2]
        if query_content:
            content_overlap = sum(1 for w in query_content if w in response_lower) / len(query_content)
            score += content_overlap * 1.5
        
        # ============================================================
        # 6. STRUCTURAL QUALITY INDICATORS
        # ============================================================
        
        # Proper capitalization at sentence starts
        capital_starts = re.findall(r'(?:^|[.!?]\s+)[A-Z]', response_stripped)
        if len(capital_starts) >= 1:
            score += 0.3
        
        # Ends with proper punctuation
        if response_stripped[-1] in '.!?)':
            score += 0.2
        
        # Contains explanatory connectors
        connectors = ['because', 'therefore', 'thus', 'consequently', 'as a result',
                      'in addition', 'furthermore', 'moreover', 'specifically',
                      'in particular', 'notably', 'while', 'whereas', 'despite']
        connector_count = sum(1 for c in connectors if c in response_lower)
        score += min(connector_count * 0.2, 0.8)
        
        # ============================================================
        # 7. LENGTH CALIBRATION
        # ============================================================
        word_count = len(words)
        
        # Very short responses are usually bad
        if word_count < 3:
            score *= 0.2
        elif word_count < 8:
            score *= 0.5
        elif word_count < 15:
            score *= 0.7
        # Moderate length is good
        elif 15 <= word_count <= 200:
            score += 0.5
        # Very long responses might be rambling
        elif word_count > 500:
            score *= 0.85
        
        # ============================================================
        # 8. UNIQUE CONTENT RATIO (not just filler)
        # ============================================================
        if len(words) > 5:
            unique_words = set(words)
            unique_ratio = len(unique_words) / len(words)
            if unique_ratio > 0.6:
                score += 0.5
            elif unique_ratio < 0.3:
                penalties += 0.5
        
        # Apply penalties
        score -= penalties
        
        # Normalize to 0-10 range
        # The max theoretical score is around 10-11, so we scale accordingly
        score = max(0.0, score)
        score = min(10.0, score)
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This maps the raw score to a more discriminative range
        if score > 0:
            score = 10.0 * (1.0 - math.exp(-0.3 * score)) / (1.0 - math.exp(-3.0))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0