def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a different approach:
    - Named entity detection via capitalization patterns
    - Numeric/quantitative data density
    - Specificity of language (precise vs vague words)
    - Information-to-filler ratio
    - Structured knowledge delivery (step-by-step, labeled sections)
    - Unique noun density as proxy for concrete detail
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.0
        
        words = response_stripped.split()
        total_words = len(words)
        if total_words < 3:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Numeric/Quantitative Data Density ===
        # Count numbers, percentages, measurements, dates, ranges
        number_patterns = [
            r'\b\d+\.?\d*\s*%',           # percentages
            r'\b\d+\.?\d*\s*(kg|lb|m|km|cm|mm|ft|in|oz|g|mg|ml|L|mph|km/h|°[CF]|degrees)',  # measurements with units
            r'\b\d{4}\b',                   # years
            r'\b\d+/\d+\b',               # fractions
            r'\$\d+',                       # dollar amounts
            r'\b\d+\.?\d*\s*(hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b',  # time durations
            r'\b\d+\.?\d*\s*(?:x|×)\s*\d+',  # dimensions
        ]
        
        numeric_count = 0
        for pat in number_patterns:
            numeric_count += len(re.findall(pat, response_stripped, re.IGNORECASE))
        
        # Also count standalone numbers not already captured
        all_numbers = re.findall(r'\b\d+\.?\d*\b', response_stripped)
        numeric_count += len(all_numbers) * 0.3  # lower weight for bare numbers
        
        numeric_density = numeric_count / total_words * 100
        score += min(numeric_density * 8, 20)  # cap at 20
        
        # === 2. Named Entity Proxy: Capitalized multi-word phrases and proper nouns ===
        # Find capitalized words not at sentence start
        capitalized_entities = set()
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                if i > 0 and len(w) > 1 and w[0].isupper() and not w.isupper():
                    clean_w = re.sub(r'[^\w]', '', w)
                    if len(clean_w) > 1:
                        capitalized_entities.add(clean_w.lower())
        
        entity_score = len(capitalized_entities) / total_words * 100
        score += min(entity_score * 12, 15)
        
        # === 3. Specificity vs Vagueness Ratio ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bvarious factors\b', r'\bthere are many\b',
            r'\bthere are various\b', r'\bgenerally speaking\b',
            r'\bin general\b', r'\bsort of\b', r'\bkind of\b',
            r'\bmore or less\b', r'\bquite a few\b', r'\ba lot of\b',
            r'\bthings like that\b', r'\band so on\b', r'\betc\.?\b',
            r'\band more\b', r'\bamong others\b', r'\bvarious\b',
            r'\bnumerous\b', r'\bcountless\b', r'\bmyriad\b',
            r'\bseveral\b', r'\ba number of\b', r'\ba variety of\b',
            r'\bmany different\b', r'\ball sorts of\b',
            r'\bpotentially\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bmight be\b', r'\bcould be\b', r'\btend to\b',
            r'\bsomewhat\b', r'\brelatively\b', r'\bfairly\b',
            r'\bquite\b', r'\brather\b',
        ]
        
        vague_count = 0
        response_lower = response_stripped.lower()
        for vp in vague_phrases:
            vague_count += len(re.findall(vp, response_lower))
        
        vague_density = vague_count / num_sentences
        score -= min(vague_density * 5, 15)
        
        # Specific/precise language indicators
        precise_patterns = [
            r'\bspecifically\b', r'\bexactly\b', r'\bprecisely\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\bnamely\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bin particular\b',
        ]
        
        precise_count = 0
        for pp in precise_patterns:
            precise_count += len(re.findall(pp, response_lower))
        
        score += min(precise_count * 1.5, 8)
        
        # === 4. Unique Concrete Noun Density ===
        # Approximate: longer words that aren't common function words tend to be more content-rich
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'although',
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'what', 'which', 'who', 'whom', 'whose', 'also', 'about', 'up',
            'i', 'me', 'my', 'myself', 'yourself', 'himself', 'herself', 'itself',
            'themselves', 'ourselves', 'yourselves', 'am', 'much', 'many',
        }
        
        content_words = []
        for w in words:
            clean = re.sub(r'[^\w]', '', w).lower()
            if clean and clean not in function_words and len(clean) > 2 and not clean.isdigit():
                content_words.append(clean)
        
        unique_content = set(content_words)
        
        # Content word richness: ratio of unique content words to total words
        if total_words > 0:
            content_richness = len(unique_content) / total_words
            score += content_richness * 20  # typically 0.2-0.5 range -> 4-10 points
        
        # Long specific words (>8 chars) as proxy for technical/specific terms
        long_specific = [w for w in unique_content if len(w) > 8]
        long_word_density = len(long_specific) / total_words * 100
        score += min(long_word_density * 4, 10)
        
        # === 5. Structured Information Delivery ===
        # Markdown bold markers **text** 
        bold_items = re.findall(r'\*\*[^*]+\*\*', response_stripped)
        score += min(len(bold_items) * 0.8, 8)
        
        # Numbered/lettered steps with content
        numbered_steps = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*\S', response_stripped)
        score += min(len(numbered_steps) * 0.6, 5)
        
        # Colon-separated label:value pairs (e.g., "Temperature: 350°F")
        label_value_pairs = re.findall(r'\b[A-Z][a-zA-Z\s]{2,20}:\s*\S', response_stripped)
        score += min(len(label_value_pairs) * 0.5, 4)
        
        # === 6. Parenthetical clarifications (e.g., "also known as X", "abbreviated Y") ===
        parentheticals = re.findall(r'\([^)]{3,60}\)', response_stripped)
        meaningful_parens = [p for p in parentheticals if not re.match(r'^\(\d+\)$', p)]
        score += min(len(meaningful_parens) * 1.0, 5)
        
        # === 7. Direct actionable language ===
        action_verbs = [
            r'\buse\b', r'\badd\b', r'\bmix\b', r'\bplace\b', r'\bset\b',
            r'\bapply\b', r'\binstall\b', r'\bconnect\b', r'\bopen\b',
            r'\bclick\b', r'\btype\b', r'\benter\b', r'\bselect\b',
            r'\bchoose\b', r'\bcalculate\b', r'\bmeasure\b', r'\bcut\b',
            r'\bpour\b', r'\bheat\b', r'\bcook\b', r'\bbake\b',
            r'\bdrive\b', r'\btake\b', r'\bturn\b', r'\bfollow\b',
        ]
        
        action_count = 0
        for av in action_verbs:
            action_count += len(re.findall(av, response_lower))
        
        action_density = action_count / num_sentences
        score += min(action_density * 3, 6)
        
        # === 8. Information density: average content words per sentence ===
        content_per_sentence = len(content_words) / num_sentences
        # Sweet spot: 8-15 content words per sentence is dense but readable
        if content_per_sentence >= 8:
            score += min((content_per_sentence - 5) * 0.5, 5)
        
        # === 9. Penalty for excessive hedging sentences ===
        hedging_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            hedge_markers = ['i think', 'i believe', 'in my opinion', 'it seems',
                           'it appears', 'arguably', 'debatable', 'not necessarily',
                           'hard to say', 'difficult to determine']
            if any(h in sent_lower for h in hedge_markers):
                hedging_sentences += 1
        
        hedge_ratio = hedging_sentences / num_sentences
        score -= hedge_ratio * 8
        
        # === 10. Reward for comparative/contrastive specificity ===
        comparison_patterns = [
            r'\bin contrast\b', r'\bwhereas\b', r'\bunlike\b',
            r'\bcompared to\b', r'\bon the other hand\b',
            r'\bhowever\b', r'\binstead of\b', r'\brather than\b',
        ]
        
        comparison_count = 0
        for cp in comparison_patterns:
            comparison_count += len(re.findall(cp, response_lower))
        
        score += min(comparison_count * 1.0, 4)
        
        # === 11. URL, reference, or citation-like patterns ===
        urls = re.findall(r'https?://\S+|www\.\S+', response_stripped)
        citations = re.findall(r'\[\d+\]|\(\d{4}\)', response_stripped)
        score += min((len(urls) + len(citations)) * 1.5, 5)
        
        # === 12. Formula/equation patterns (LaTeX or plain) ===
        formulas = re.findall(r'\\[a-z]+\{|[a-zA-Z]+\s*=\s*[\d\(]|\\frac|\\sqrt', response_stripped)
        score += min(len(formulas) * 1.0, 5)
        
        # Normalize to 0-100 range
        # Typical raw scores range from about -5 to 70
        final_score = max(0.0, min(100.0, score * 1.3 + 10))
        
        return round(final_score, 2)
        
    except Exception:
        return 25.0