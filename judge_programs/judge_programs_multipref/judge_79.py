def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a fundamentally different approach:
    - Named entity detection via capitalization patterns
    - Numeric/quantitative information density
    - Specificity ratio (specific words vs vague words)
    - Information-bearing token ratio
    - Parenthetical detail density (citations, clarifications, units)
    - Colon-introduced elaboration patterns
    - Quotation and proper noun density
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.0
        
        words = response_clean.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # === FEATURE 1: Named Entity Density ===
        # Detect capitalized multi-word phrases (likely proper nouns/named entities)
        # Exclude sentence starters
        named_entity_pattern = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', response_clean)
        # Also find standalone capitalized words mid-sentence
        mid_sentence_caps = re.findall(r'(?<=[a-z,;:]\s)[A-Z][a-zA-Z]+', response_clean)
        named_entity_count = len(named_entity_pattern) + len(mid_sentence_caps)
        named_entity_density = named_entity_count / word_count * 100
        
        # === FEATURE 2: Numeric Information Density ===
        # Count different types of numbers
        integers = re.findall(r'\b\d+\b', response_clean)
        decimals = re.findall(r'\b\d+\.\d+\b', response_clean)
        percentages = re.findall(r'\d+%', response_clean)
        fractions = re.findall(r'\b\d+/\d+\b', response_clean)
        ranges = re.findall(r'\d+\s*[-–—to]+\s*\d+', response_clean)
        years = re.findall(r'\b(?:19|20)\d{2}\b', response_clean)
        measurements = re.findall(r'\d+\s*(?:kg|lb|m|cm|mm|km|ft|inch|oz|g|mg|ml|L|mph|km/h|°[CF]|degrees|meters|feet|miles|pounds|kilograms|grams)', response_clean, re.IGNORECASE)
        
        numeric_count = len(integers) + len(decimals) * 2 + len(percentages) * 2 + len(fractions) * 1.5 + len(ranges) * 2 + len(years) * 1.5 + len(measurements) * 2.5
        numeric_density = numeric_count / word_count * 100
        
        # === FEATURE 3: Parenthetical Detail Density ===
        # Parenthetical expressions often contain clarifying specifics
        parentheticals = re.findall(r'\([^)]+\)', response_clean)
        paren_count = len(parentheticals)
        # Score based on content richness of parentheticals
        paren_richness = 0
        for p in parentheticals:
            if re.search(r'\d', p):
                paren_richness += 2
            if re.search(r'[A-Z]', p):
                paren_richness += 1
            if len(p) > 10:
                paren_richness += 1
        paren_density = (paren_count + paren_richness) / num_sentences
        
        # === FEATURE 4: Specificity Lexicon Score ===
        response_lower = response_clean.lower()
        
        # Vague/hedge phrases - penalize these
        vague_phrases = [
            'many people', 'some people', 'it depends', 'various factors',
            'there are many', 'there are various', 'a lot of', 'in general',
            'generally speaking', 'it is important', 'it\'s important',
            'can be', 'may be', 'might be', 'could be', 'tends to',
            'often times', 'sometimes', 'arguably', 'presumably',
            'more or less', 'kind of', 'sort of', 'a number of',
            'a variety of', 'several factors', 'many factors',
            'it is worth noting', 'it should be noted', 'keep in mind',
            'on the other hand', 'at the end of the day',
            'needless to say', 'goes without saying',
            'as we all know', 'everyone knows', 'obviously',
            'basically', 'essentially', 'fundamentally',
            'in some cases', 'in many cases', 'in most cases',
            'and so on', 'and so forth', 'etc', 'and more',
            'things like that', 'stuff like that', 'or something',
            'you know', 'i think', 'i believe', 'i feel',
            'not necessarily', 'not always', 'it varies',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += len(re.findall(re.escape(phrase), response_lower))
        
        vague_penalty = vague_count / num_sentences
        
        # === FEATURE 5: Concrete/Specific Action Words ===
        specific_indicators = [
            'specifically', 'for example', 'for instance', 'such as',
            'including', 'namely', 'in particular', 'e.g.',
            'i.e.', 'according to', 'known as', 'called',
            'defined as', 'referred to as', 'consists of',
            'comprised of', 'contains', 'located in', 'located at',
            'founded in', 'established in', 'developed by',
            'created by', 'invented by', 'designed by',
            'published in', 'released in', 'produced by',
            'measured in', 'calculated as', 'equal to',
            'approximately', 'exactly', 'precisely',
            'between', 'ranges from', 'up to',
        ]
        
        specific_count = 0
        for indicator in specific_indicators:
            specific_count += len(re.findall(re.escape(indicator), response_lower))
        
        specificity_density = specific_count / num_sentences
        
        # === FEATURE 6: Technical/Domain Vocabulary Density ===
        # Words that are longer and likely domain-specific
        technical_words = [w for w in words if len(w) > 8 and w.isalpha()]
        technical_density = len(technical_words) / word_count * 100
        
        # === FEATURE 7: Structural Specificity ===
        # Colon-introduced elaborations (often precede specific details)
        colon_elaborations = len(re.findall(r':\s*\S', response_clean))
        
        # Bold/formatted terms (markdown formatting indicates structured info)
        bold_terms = re.findall(r'\*\*[^*]+\*\*', response_clean)
        bold_count = len(bold_terms)
        
        # Inline code or quoted terms
        quoted_terms = re.findall(r'["\'][^"\']+["\']', response_clean)
        code_terms = re.findall(r'`[^`]+`', response_clean)
        quoted_count = len(quoted_terms) + len(code_terms)
        
        structural_score = (colon_elaborations * 0.5 + bold_count * 0.3 + quoted_count * 0.5) / num_sentences
        
        # === FEATURE 8: Information-to-Filler Ratio ===
        # Filler/empty words
        filler_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                       'it', 'this', 'that', 'these', 'those', 'and', 'or', 'but',
                       'if', 'then', 'so', 'as', 'not', 'no', 'yes', 'very',
                       'just', 'also', 'really', 'quite', 'much', 'well', 'here',
                       'there', 'when', 'where', 'how', 'what', 'which', 'who',
                       'your', 'you', 'i', 'we', 'they', 'he', 'she', 'my',
                       'their', 'its', 'our', 'his', 'her'}
        
        word_list_lower = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words]
        content_words = [w for w in word_list_lower if w and w not in filler_words and len(w) > 2]
        info_ratio = len(content_words) / max(word_count, 1)
        
        # === FEATURE 9: Unique Noun Phrase Density ===
        # Approximate by counting unique capitalized sequences and unique longer words
        unique_content = set(content_words)
        content_diversity = len(unique_content) / max(len(content_words), 1)
        
        # === FEATURE 10: Step/Process Detail Score ===
        # Numbered steps with actual content
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*.+', response_clean)
        lettered_items = re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s*.+', response_clean)
        dash_items = re.findall(r'(?:^|\n)\s*[-•]\s*.+', response_clean)
        
        list_items = numbered_items + lettered_items + dash_items
        # Score list items by their specificity (containing numbers, proper nouns, etc.)
        list_specificity = 0
        for item in list_items:
            if re.search(r'\d', item):
                list_specificity += 1.5
            if re.search(r'[A-Z][a-z]{2,}', item):
                list_specificity += 0.5
            if len(item.split()) > 5:
                list_specificity += 0.5
            list_specificity += 1  # base credit for having structure
        
        list_score = list_specificity / max(num_sentences, 1)
        
        # === FEATURE 11: Formula/Equation Density ===
        equations = re.findall(r'[=×÷±√∑∫]', response_clean)
        math_expressions = re.findall(r'\b\w+\s*[=]\s*\S+', response_clean)
        formula_density = (len(equations) + len(math_expressions) * 0.5) / num_sentences
        
        # === FEATURE 12: URL/Reference patterns ===
        urls = re.findall(r'https?://\S+|www\.\S+', response_clean)
        url_score = len(urls) * 2.0
        
        # === COMPOSITE SCORING ===
        # Weight each feature
        score = 0.0
        
        # Named entities (0-15 points)
        score += min(named_entity_density * 1.5, 15)
        
        # Numeric density (0-20 points)
        score += min(numeric_density * 3.0, 20)
        
        # Parenthetical richness (0-8 points)
        score += min(paren_density * 2.0, 8)
        
        # Vague penalty (-15 to 0 points)
        score -= min(vague_penalty * 5.0, 15)
        
        # Specificity indicators (0-12 points)
        score += min(specificity_density * 4.0, 12)
        
        # Technical vocabulary (0-10 points)
        score += min(technical_density * 0.8, 10)
        
        # Structural specificity (0-8 points)
        score += min(structural_score * 3.0, 8)
        
        # Information ratio (0-10 points)
        score += info_ratio * 10
        
        # Content diversity bonus (0-5 points)
        score += content_diversity * 5
        
        # List specificity (0-10 points)
        score += min(list_score * 2.0, 10)
        
        # Formula density (0-5 points)
        score += min(formula_density * 2.0, 5)
        
        # URL references (0-4 points)
        score += min(url_score, 4)
        
        # Length bonus: slightly favor responses that are substantive but not just verbose
        # Reward up to ~300 words, diminishing returns after
        length_factor = 1.0 + 0.1 * min(math.log(word_count + 1) / math.log(300), 1.0)
        score *= length_factor
        
        # Normalize to 0-100 range
        score = max(0, min(score, 100))
        
        return round(score, 3)
        
    except Exception as e:
        # Fallback: return a basic score based on response length
        try:
            return min(len(response.split()) / 20.0, 50.0)
        except:
            return 0.0