def judging_function(query, response):
    """
    Evaluate evidence density and specificity using a unique approach:
    - Named entity detection via capitalization patterns
    - Numeric/quantitative density
    - Specificity of language (precise vs vague words)
    - Information-to-filler ratio
    - Structural richness (formatting, citations, parentheticals)
    - Action verb density
    - Unique proper nouns and technical terms
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.0
        
        words = response.split()
        total_words = len(words)
        if total_words < 3:
            return 0.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Named Entity / Proper Noun Density ===
        # Look for capitalized words NOT at sentence starts
        proper_nouns = set()
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                clean = re.sub(r'[^a-zA-Z]', '', w)
                if len(clean) > 1 and clean[0].isupper() and not clean.isupper():
                    if i > 0:  # not sentence start
                        proper_nouns.add(clean)
        
        proper_noun_density = len(proper_nouns) / max(total_words, 1) * 100
        score += min(proper_noun_density * 3, 12)
        
        # === 2. Numeric / Quantitative Information Density ===
        # Find all numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d+[\d,]*\.?\d*\b', response)
        percentages = re.findall(r'\d+\.?\d*\s*%', response)
        measurements = re.findall(r'\d+\.?\d*\s*(?:kg|lb|lbs|m|km|miles|feet|ft|cm|mm|inches|in|oz|g|mg|ml|L|°[CF]|mph|kph|m/s|Hz|kHz|MHz|GHz|watts|W|V|A|ohm)', response)
        dates = re.findall(r'\b(?:19|20)\d{2}\b', response)
        fractions = re.findall(r'\b\d+/\d+\b', response)
        
        quant_items = len(numbers) + len(percentages) * 2 + len(measurements) * 2.5 + len(dates) * 1.5 + len(fractions) * 1.5
        quant_density = quant_items / num_sentences
        score += min(quant_density * 4, 15)
        
        # === 3. Specificity Language Analysis ===
        # Reward precise/specific words, penalize vague/hedge words
        response_lower = response.lower()
        
        # Vague filler phrases (penalize)
        vague_phrases = [
            'many people', 'some people', 'a lot of', 'various factors',
            'it depends', 'there are many', 'there are various', 'in general',
            'generally speaking', 'more or less', 'kind of', 'sort of',
            'and so on', 'and stuff', 'things like that', 'etc etc',
            'you know', 'basically', 'essentially', 'arguably',
            'it is said', 'some say', 'people say', 'they say',
            'in some cases', 'in many cases', 'for the most part',
            'quite a few', 'a number of', 'a bunch of',
            'and more', 'among others', 'or something',
            'pretty much', 'more or less', 'to some extent',
            'can be', 'might be', 'could be', 'may vary',
            'tends to', 'it seems', 'it appears',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += len(re.findall(re.escape(phrase), response_lower))
        
        vague_penalty = min(vague_count * 2.5, 15)
        score -= vague_penalty
        
        # === 4. Precision Markers (reward) ===
        precision_markers = [
            r'\bspecifically\b', r'\bexactly\b', r'\bprecisely\b',
            r'\bin particular\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bnamely\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\baccording to\b', r'\bbased on\b',
            r'\bresearch shows\b', r'\bstudies show\b',
            r'\bthe reason is\b', r'\bthis means\b',
        ]
        
        precision_count = 0
        for pattern in precision_markers:
            precision_count += len(re.findall(pattern, response_lower))
        
        score += min(precision_count * 2, 10)
        
        # === 5. Parenthetical Details (e.g., abbreviations, clarifications) ===
        parentheticals = re.findall(r'\([^)]{2,80}\)', response)
        score += min(len(parentheticals) * 1.5, 8)
        
        # === 6. Technical / Domain-Specific Term Density ===
        # Words that are longer and less common tend to be more technical
        technical_words = 0
        for w in words:
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if len(clean) >= 8:
                technical_words += 1
            # Check for compound words with hyphens
            if '-' in w and len(w) > 5:
                technical_words += 0.5
        
        tech_density = technical_words / total_words * 100
        score += min(tech_density * 0.8, 8)
        
        # === 7. Enumeration and Structured Data ===
        # Detect numbered steps, lettered items, colon-separated key-value pairs
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        lettered_items = len(re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s', response))
        kv_pairs = len(re.findall(r'\*\*[^*]+\*\*\s*:', response))
        colon_defs = len(re.findall(r'(?:^|\n)[^:\n]{3,30}:\s', response))
        
        struct_score = numbered_items * 0.8 + lettered_items * 0.6 + kv_pairs * 1.2 + colon_defs * 0.5
        score += min(struct_score, 10)
        
        # === 8. Unique Information Tokens ===
        # Count unique content-bearing tokens (not stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
            'and', 'or', 'if', 'while', 'that', 'this', 'these', 'those',
            'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their',
            'he', 'she', 'his', 'her', 'i', 'me', 'my', 'also', 'which',
            'what', 'who', 'whom', 'about', 'up', 'down', 'any',
        }
        
        content_words = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if clean and clean not in stopwords and len(clean) > 2:
                content_words.append(clean)
        
        unique_content = len(set(content_words))
        content_richness = unique_content / max(total_words, 1) * 100
        score += min(content_richness * 0.3, 10)
        
        # === 9. Concrete Action Verbs ===
        action_verbs = [
            'install', 'configure', 'calculate', 'measure', 'combine',
            'mix', 'heat', 'cool', 'pour', 'cut', 'slice', 'chop',
            'connect', 'attach', 'remove', 'insert', 'apply', 'spread',
            'drive', 'navigate', 'travel', 'fly', 'sail', 'walk',
            'download', 'upload', 'click', 'select', 'type', 'enter',
            'preheat', 'bake', 'boil', 'simmer', 'roast', 'grill',
            'stir', 'whisk', 'fold', 'knead', 'marinate', 'season',
            'compile', 'execute', 'run', 'deploy', 'test', 'debug',
            'analyze', 'compare', 'evaluate', 'identify', 'specify',
            'implement', 'design', 'build', 'create', 'develop',
            'purchase', 'buy', 'order', 'gather', 'collect', 'obtain',
        ]
        
        action_count = 0
        for verb in action_verbs:
            action_count += len(re.findall(r'\b' + verb + r'(?:s|ed|ing|d)?\b', response_lower))
        
        action_density = action_count / num_sentences
        score += min(action_density * 3, 8)
        
        # === 10. Inline Examples and References ===
        # Detect patterns like "e.g.,", "such as X, Y, and Z", quoted terms
        example_patterns = len(re.findall(r'(?:e\.g\.|for example|for instance|such as)\s*[A-Za-z]', response_lower))
        quoted_terms = len(re.findall(r'"[^"]{2,50}"', response))
        bold_terms = len(re.findall(r'\*\*[^*]{2,50}\*\*', response))
        
        ref_score = example_patterns * 2 + quoted_terms * 1.5 + bold_terms * 0.5
        score += min(ref_score, 10)
        
        # === 11. URL/Link patterns (evidence of references) ===
        urls = len(re.findall(r'https?://\S+', response))
        score += min(urls * 3, 6)
        
        # === 12. Mathematical/Formula content ===
        math_expressions = len(re.findall(r'[=+\-*/^]', response))
        formula_density = math_expressions / max(total_words, 1) * 100
        score += min(formula_density * 0.5, 5)
        
        # === 13. Length consideration (moderate bonus for substantive length) ===
        # Not just length, but information-per-unit-length
        if total_words > 50:
            length_bonus = math.log2(total_words / 50) * 1.5
            score += min(length_bonus, 5)
        
        # === 14. Repetition penalty ===
        # Penalize responses that repeat the same phrases
        bigrams = [' '.join(words[i:i+2]).lower() for i in range(len(words)-1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated = sum(1 for c in bigram_counts.values() if c > 2)
            repetition_ratio = repeated / max(len(set(bigrams)), 1)
            score -= min(repetition_ratio * 10, 5)
        
        # === 15. Sentence-level information density ===
        # Average content words per sentence
        if num_sentences > 0:
            content_per_sentence = len(content_words) / num_sentences
            if content_per_sentence > 5:
                score += min((content_per_sentence - 5) * 0.5, 5)
        
        # Normalize to 0-100 range
        score = max(0, min(100, score * 1.5 + 30))
        
        return round(score, 2)
        
    except Exception:
        return 25.0