def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    focused on detecting specific entity types, structural completeness, and
    information-to-filler ratio.
    
    This variant uses:
    - Named entity pattern detection (capitalized multi-word phrases, proper nouns)
    - Numeric/quantitative expression detection
    - Filler/hedge phrase penalty scoring
    - Sentence-level information density (specific tokens per sentence)
    - Response completeness heuristics (truncation detection, coherence)
    - Relevance via character n-gram overlap with query
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        response_clean = response.strip()
        query_clean = query.strip() if query else ""
        
        # Tokenize
        words = re.findall(r'\b\w+\b', response_clean)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Very short responses are usually low quality
        if word_count <= 3:
            # Unless they directly and correctly answer a factual question
            # Give a small base score
            return 1.5
        
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Named entity density (capitalized phrases)
        # ============================================================
        # Detect capitalized words that aren't sentence starters
        # Find proper nouns / named entities via capitalization patterns
        cap_pattern = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response_clean)
        # Filter out sentence-starting words
        # Approximate: words at position 0 or after sentence-ending punctuation
        named_entities = []
        for match in cap_pattern:
            # Multi-word capitalized phrases are almost certainly named entities
            if ' ' in match:
                named_entities.append(match)
            elif match.lower() not in {
                'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'is',
                'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'shall', 'can', 'for', 'and', 'but', 'or', 'nor', 'not',
                'so', 'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
                'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'about', 'between',
                'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'if',
                'also', 'however', 'therefore', 'thus', 'hence', 'sure', 'yes',
                'no', 'comment', 'output', 'input', 'question', 'answer',
                'note', 'string', 'percussion', 'visit', 'determine', 'identify'
            }:
                named_entities.append(match)
        
        # Also detect specific patterns like "X of Y" with capitals
        specific_refs = re.findall(r'[A-Z][a-z]+(?:\s+(?:of|the|and|in|for|at|by)\s+[A-Z][a-z]+)+', response_clean)
        named_entities.extend(specific_refs)
        
        ne_count = len(named_entities)
        ne_density = ne_count / max(word_count, 1) * 100  # per 100 words
        ne_score = min(ne_density * 1.5, 10)  # cap at 10
        
        # ============================================================
        # FEATURE 2: Numeric and quantitative expressions
        # ============================================================
        # Detect numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', response_clean)
        percentages = re.findall(r'\d+\.?\d*\s*%', response_clean)
        dates = re.findall(r'\b(?:19|20)\d{2}\b', response_clean)
        measurements = re.findall(r'\b\d+\.?\d*\s*(?:km|miles?|meters?|feet|inches|lbs?|kg|oz|gallons?|liters?|hours?|minutes?|seconds?|days?|years?|months?|weeks?|degrees?|celsius|fahrenheit)\b', response_clean, re.IGNORECASE)
        money = re.findall(r'[\$€£¥]\s*\d[\d,]*\.?\d*|\d[\d,]*\.?\d*\s*(?:dollars?|euros?|pounds?|yen)', response_clean, re.IGNORECASE)
        
        quant_count = len(numbers) + len(percentages) * 2 + len(dates) * 1.5 + len(measurements) * 2 + len(money) * 2
        quant_density = quant_count / max(word_count, 1) * 100
        quant_score = min(quant_density * 3, 10)
        
        # ============================================================
        # FEATURE 3: Filler and hedge phrase penalty
        # ============================================================
        filler_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things)\b',
            r'\bin general\b', r'\boverall\b', r'\bbasically\b',
            r'\bvarious factors\b', r'\bmany factors\b',
            r'\bit\'?s? (?:important|worth|good) to (?:note|mention|remember|consider)\b',
            r'\bas (?:we all|everyone) know\b',
            r'\bneedless to say\b', r'\bof course\b',
            r'\bgenerally speaking\b', r'\bin most cases\b',
            r'\bthere are many\b', r'\bthere are several\b',
            r'\bcan vary\b', r'\bmay vary\b', r'\bdepends on\b',
            r'\band so on\b', r'\betc\.?\b', r'\band more\b',
            r'\bamong others\b', r'\bfor example\b(?!\s*[,:]?\s*[A-Z0-9])',
        ]
        
        filler_count = 0
        response_lower = response_clean.lower()
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / max(num_sentences, 1)
        filler_penalty = min(filler_ratio * 3, 5)  # max penalty of 5
        
        # ============================================================
        # FEATURE 4: Sentence-level information density
        # ============================================================
        # For each sentence, count "informative tokens" (not stopwords, not short)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'for', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'to', 'of', 'in', 'on', 'at', 'by', 'with',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'as', 'until', 'while', 'if', 'that', 'this',
            'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my',
            'also', 'however', 'which', 'who', 'whom', 'what'
        }
        
        info_densities = []
        for sent in sentences:
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            if len(sent_words) < 2:
                continue
            informative = [w for w in sent_words if w not in stopwords and len(w) > 2]
            density = len(informative) / max(len(sent_words), 1)
            info_densities.append(density)
        
        avg_info_density = sum(info_densities) / max(len(info_densities), 1) if info_densities else 0.3
        info_density_score = avg_info_density * 10  # scale to ~0-7 range
        
        # ============================================================
        # FEATURE 5: Lexical specificity (unique content words / total words)
        # ============================================================
        content_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
        unique_content = set(content_words)
        
        # Type-token ratio for content words (higher = more diverse vocabulary = more specific)
        if len(content_words) > 0:
            ttr = len(unique_content) / len(content_words)
        else:
            ttr = 0
        
        # Penalize very low TTR (repetitive) but also very high TTR in very short text is meaningless
        if word_count < 10:
            specificity_score = ttr * 3
        else:
            specificity_score = ttr * 6
        
        # ============================================================
        # FEATURE 6: Truncation and coherence detection
        # ============================================================
        truncation_penalty = 0
        # Check if response appears truncated (ends mid-sentence)
        if response_clean and response_clean[-1] not in '.!?"\')>}]':
            # Might be truncated
            last_chars = response_clean[-20:]
            if not re.search(r'[.!?]\s*$', last_chars):
                truncation_penalty = 1.0
        
        # Check for repetitive content (copy-paste artifacts)
        if num_sentences >= 4:
            sent_texts = [s.lower().strip() for s in sentences if len(s.strip()) > 10]
            if sent_texts:
                unique_sents = set(sent_texts)
                repetition_ratio = 1 - (len(unique_sents) / len(sent_texts))
                if repetition_ratio > 0.3:
                    truncation_penalty += repetition_ratio * 3
        
        # Check for garbage/irrelevant content (HTML, code when not asked)
        query_lower = query_clean.lower()
        is_code_query = any(kw in query_lower for kw in ['code', 'program', 'function', 'html', 'script', 'python', 'java', 'css'])
        
        garbage_penalty = 0
        if not is_code_query:
            html_tags = re.findall(r'<[^>]+>', response_clean)
            code_blocks = re.findall(r'(?:import |def |class |print\(|console\.log)', response_clean)
            if len(html_tags) > 3 or len(code_blocks) > 2:
                garbage_penalty = 2.0
        
        # ============================================================
        # FEATURE 7: Query-response relevance via character trigram overlap
        # ============================================================
        def char_trigrams(text):
            text = text.lower()
            return Counter(text[i:i+3] for i in range(len(text) - 2))
        
        if query_clean and len(query_clean) > 5:
            q_trigrams = char_trigrams(query_clean)
            r_trigrams = char_trigrams(response_clean[:500])  # first 500 chars
            
            # Cosine-like overlap
            common = set(q_trigrams.keys()) & set(r_trigrams.keys())
            if common:
                numerator = sum(q_trigrams[t] * r_trigrams[t] for t in common)
                denom_q = math.sqrt(sum(v**2 for v in q_trigrams.values()))
                denom_r = math.sqrt(sum(v**2 for v in r_trigrams.values()))
                relevance = numerator / (denom_q * denom_r) if denom_q * denom_r > 0 else 0
            else:
                relevance = 0
        else:
            relevance = 0.5  # neutral if no query
        
        relevance_score = relevance * 5  # scale to ~0-5
        
        # ============================================================
        # FEATURE 8: Actionable/concrete language detection
        # ============================================================
        concrete_patterns = [
            r'\b(?:located|founded|built|created|established|invented|discovered)\s+(?:in|at|by|on)\b',
            r'\b(?:called|named|known as|referred to as)\b',
            r'\b(?:specifically|precisely|exactly|approximately|roughly)\b',
            r'\b(?:according to|based on|as stated|as described)\b',
            r'\b(?:for instance|for example|such as|including|e\.g\.)\s+[A-Z0-9]',
            r'\b(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b',
            r'\b(?:step \d|phase \d|stage \d|part \d)\b',
        ]
        
        concrete_count = 0
        for pattern in concrete_patterns:
            concrete_count += len(re.findall(pattern, response_clean, re.IGNORECASE))
        
        concrete_density = concrete_count / max(num_sentences, 1)
        concrete_score = min(concrete_density * 4, 6)
        
        # ============================================================
        # FEATURE 9: Response length appropriateness
        # ============================================================
        # Not too short, not excessively long with filler
        if word_count < 5:
            length_score = 0.5
        elif word_count < 15:
            length_score = 2.0
        elif word_count < 30:
            length_score = 3.5
        elif word_count < 80:
            length_score = 4.5
        elif word_count < 200:
            length_score = 5.0
        else:
            # Longer isn't always better - diminishing returns
            length_score = 4.5
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        raw_score = (
            ne_score * 0.15 +          # Named entities
            quant_score * 0.12 +        # Quantitative info
            info_density_score * 0.15 +  # Information density per sentence
            specificity_score * 0.10 +   # Lexical specificity
            relevance_score * 0.13 +     # Query relevance
            concrete_score * 0.10 +      # Concrete language
            length_score * 0.15 +        # Length appropriateness
            - filler_penalty * 0.4 +     # Filler penalty
            - truncation_penalty * 0.5 + # Truncation penalty
            - garbage_penalty * 0.5      # Garbage penalty
        )
        
        # Add small bonus for structured responses (lists, colons)
        has_structure = bool(re.search(r'(?:^\s*[-•*]\s|\d+\.\s|:\s)', response_clean, re.MULTILINE))
        if has_structure:
            raw_score += 0.5
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score + 2.0))  # shift up a bit since most features are 0-based
        
        # Floor very short non-informative responses
        if word_count <= 5 and ne_count == 0 and len(numbers) == 0:
            final_score = min(final_score, 2.0)
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: simple length-based score
        try:
            wc = len(response.split())
            if wc == 0:
                return 0.0
            return min(max(wc / 20.0, 0.5), 5.0)
        except:
            return 2.0