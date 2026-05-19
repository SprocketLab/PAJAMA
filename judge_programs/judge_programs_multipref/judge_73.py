def judging_function(query, response):
    """
    Evaluates evidence density and specificity in an LLM response.
    Higher scores indicate more concrete evidence, specific details, and actionable information.
    Uses a feature-based approach focusing on named entities, numbers, specific terms, and structure.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        words = response.split()
        total_words = len(words)
        
        if total_words < 3:
            return 0.5
        
        score = 0.0
        
        # ============================================================
        # 1. NUMERIC DENSITY: Count specific numbers, measurements, percentages
        # ============================================================
        # Match integers, decimals, percentages, fractions, ranges
        number_patterns = [
            r'\b\d+\.?\d*\s*%',           # percentages
            r'\b\d+\.?\d*\s*(kg|lb|lbs|m|km|mi|miles|ft|feet|inches|cm|mm|oz|g|mg|ml|L|liters|gallons)\b',  # measurements
            r'\$\d+[\d,]*\.?\d*',          # dollar amounts
            r'\b\d{4}\b',                  # years
            r'\b\d+/\d+\b',               # fractions
            r'\b\d+[\d,]*\.?\d+\b',       # decimal numbers
            r'\b\d+[\d,]+\b',             # large numbers with commas
            r'\b\d+\b',                    # plain integers
        ]
        
        numeric_count = 0
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            numeric_count += len(matches)
        
        # Density of numbers per 100 words
        numeric_density = (numeric_count / total_words) * 100
        score += min(numeric_density * 3.0, 15.0)
        
        # ============================================================
        # 2. NAMED ENTITIES / PROPER NOUNS: Capitalized words not at sentence start
        # ============================================================
        sentences = re.split(r'[.!?]\s+', response)
        proper_noun_count = 0
        
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) > 1:
                # Skip first word (sentence start), check for capitalized words
                for w in sent_words[1:]:
                    clean_w = w.strip(string.punctuation)
                    if clean_w and len(clean_w) > 1 and clean_w[0].isupper() and not clean_w.isupper():
                        proper_noun_count += 1
        
        proper_noun_density = (proper_noun_count / total_words) * 100
        score += min(proper_noun_density * 2.5, 10.0)
        
        # ============================================================
        # 3. SPECIFIC TECHNICAL/CONCRETE VOCABULARY
        # ============================================================
        # Words that indicate specificity
        specificity_indicators = set()
        
        # Check for specific material names, chemical terms, brand names, etc.
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word proper nouns
            r'\b\w+tion\b', r'\b\w+ment\b', r'\b\w+ity\b',  # Abstract nouns (moderate)
        ]
        
        # Concrete nouns and specific terms (longer, more specific words tend to be more informative)
        long_specific_words = [w for w in words if len(w.strip(string.punctuation)) > 8]
        long_word_density = (len(long_specific_words) / total_words) * 100
        score += min(long_word_density * 0.8, 8.0)
        
        # ============================================================
        # 4. PENALIZE VAGUE/HEDGE LANGUAGE
        # ============================================================
        vague_phrases = [
            'many people think', 'it depends', 'there are various factors',
            'some people', 'many people', 'it is important to note',
            'there are many', 'there are several', 'in general',
            'it varies', 'it can vary', 'depending on', 'various ways',
            'a number of', 'quite a few', 'a lot of', 'lots of',
            'and so on', 'and more', 'etc.', 'among others',
            'you might want to', 'you may want to', 'you could try',
            'there are numerous', 'can be quite', 'tends to be',
            'generally speaking', 'for the most part', 'in most cases',
            'as you may know', 'as we all know', 'needless to say',
            'it goes without saying', 'obviously', 'of course',
            'basically', 'essentially', 'simply put',
            'that being said', 'having said that', 'with that said',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += response_lower.count(phrase)
        
        vague_penalty = min(vague_count * 2.0, 12.0)
        score -= vague_penalty
        
        # Single vague words (lighter penalty)
        vague_words = ['maybe', 'perhaps', 'possibly', 'somewhat', 'fairly',
                       'rather', 'quite', 'pretty', 'kind of', 'sort of',
                       'stuff', 'things', 'something']
        vague_word_count = sum(1 for vw in vague_words if vw in response_lower)
        score -= min(vague_word_count * 0.5, 4.0)
        
        # ============================================================
        # 5. STRUCTURAL SPECIFICITY: Lists, steps, headers, formatting
        # ============================================================
        # Numbered lists (1., 2., 3., etc.) or bullet points
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-*•]\s', response))
        header_items = len(re.findall(r'(?:^|\n)\s*#{1,4}\s', response))
        bold_items = len(re.findall(r'\*\*[^*]+\*\*', response))
        
        structure_score = min(numbered_items * 1.0, 5.0) + min(bullet_items * 0.8, 4.0) + \
                         min(header_items * 1.0, 3.0) + min(bold_items * 0.5, 3.0)
        score += min(structure_score, 10.0)
        
        # ============================================================
        # 6. SPECIFIC EXAMPLES AND REFERENCES
        # ============================================================
        example_indicators = [
            'for example', 'for instance', 'such as', 'e.g.',
            'specifically', 'in particular', 'namely', 'including',
            'like the', 'consider the', 'take the case of',
        ]
        example_count = sum(1 for ei in example_indicators if ei in response_lower)
        score += min(example_count * 2.0, 8.0)
        
        # ============================================================
        # 7. ACTIONABLE DETAILS: Specific instructions, recipes, formulas
        # ============================================================
        actionable_patterns = [
            r'\b\d+\s*(tablespoons?|teaspoons?|cups?|tbsp|tsp|oz|ounces?|pounds?|grams?)\b',
            r'\b\d+\s*(minutes?|hours?|seconds?|days?|weeks?)\b',
            r'\b\d+\s*(degrees?|°[CF])\b',
            r'step\s+\d+',
            r'(?:route|highway|interstate|I-)\s*\d+',
        ]
        
        actionable_count = 0
        for pattern in actionable_patterns:
            actionable_count += len(re.findall(pattern, response, re.IGNORECASE))
        score += min(actionable_count * 2.0, 10.0)
        
        # ============================================================
        # 8. FORMULA / EQUATION DENSITY (for technical responses)
        # ============================================================
        equation_patterns = [
            r'[=×÷±∑∫√π]',
            r'\b\w+\s*=\s*[\d\w(]',  # variable = something
            r'\^\d+',                  # exponents
            r'\\frac',                 # LaTeX fractions
            r'\\times',               # LaTeX multiplication
        ]
        equation_count = sum(len(re.findall(p, response)) for p in equation_patterns)
        score += min(equation_count * 1.0, 8.0)
        
        # ============================================================
        # 9. UNIQUE VOCABULARY RICHNESS (Type-Token Ratio adjusted)
        # ============================================================
        clean_words = [w.strip(string.punctuation).lower() for w in words if w.strip(string.punctuation)]
        if len(clean_words) > 10:
            # Use a root TTR to normalize for length
            unique_words = len(set(clean_words))
            ttr = unique_words / math.sqrt(len(clean_words))
            score += min(ttr * 0.5, 5.0)
        
        # ============================================================
        # 10. INFORMATION DENSITY: Ratio of content words to total words
        # ============================================================
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if', 'while',
            'about', 'up', 'out', 'that', 'this', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
            'them', 'his', 'her', 'their', 'what', 'which', 'who', 'whom',
        }
        
        content_words = [w for w in clean_words if w not in stop_words and len(w) > 2]
        if clean_words:
            content_ratio = len(content_words) / len(clean_words)
            score += content_ratio * 8.0
        
        # ============================================================
        # 11. RESPONSE LENGTH BONUS (longer responses tend to have more detail, but diminishing returns)
        # ============================================================
        length_bonus = math.log(max(total_words, 1) + 1) * 1.2
        score += min(length_bonus, 8.0)
        
        # ============================================================
        # 12. PARENTHETICAL DETAILS (e.g., abbreviations, clarifications)
        # ============================================================
        parenthetical_count = len(re.findall(r'\([^)]+\)', response))
        score += min(parenthetical_count * 0.8, 4.0)
        
        # ============================================================
        # 13. COLON-INTRODUCED SPECIFICS (Label: detail pattern)
        # ============================================================
        colon_details = len(re.findall(r'\w+\s*:', response))
        score += min(colon_details * 0.5, 4.0)
        
        # ============================================================
        # 14. QUOTE USAGE (direct quotes indicate specific references)
        # ============================================================
        quote_count = len(re.findall(r'["""].*?["""]', response))
        score += min(quote_count * 1.5, 4.0)
        
        # ============================================================
        # 15. PENALIZE FILLER OPENINGS
        # ============================================================
        filler_openings = [
            "that's a great", "great question", "awesome!", "absolutely!",
            "sure!", "of course!", "certainly!", "definitely!",
            "a classic", "let's dive", "let me",
        ]
        for filler in filler_openings:
            if response_lower.startswith(filler):
                score -= 1.5
                break
        
        # Normalize to 0-100 range
        # Typical raw scores range from about -5 to 70
        score = max(0.0, score)
        score = min(100.0, score)
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a neutral score based on response length
        try:
            return min(len(response.split()) * 0.1, 50.0)
        except:
            return 25.0