def judging_function(query, response):
    """
    Evaluates evidence density and specificity by analyzing the presence of
    concrete evidence markers: named entities (capitalized multi-word phrases),
    precise numbers/measurements, specific technical terms, actionable instructions,
    and real-world references. Penalizes vague hedging patterns.
    
    This variant focuses on:
    - Named entity detection (capitalized phrases)
    - Numeric precision (numbers with units, percentages, dates)
    - Specificity of nouns (proper nouns, technical terms)
    - Actionable detail density (imperative verbs with specific objects)
    - Ratio of concrete vs vague language
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 0.0
        
        words = response_text.split()
        word_count = len(words)
        if word_count < 3:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Named Entity Detection (capitalized multi-word phrases) ===
        # Find sequences of capitalized words that aren't at sentence starts
        named_entities = set()
        # Multi-word capitalized phrases (likely proper nouns/names)
        cap_phrases = re.findall(r'(?<=[.!?]\s|^)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', response_text)
        # Also find capitalized words mid-sentence (not after period)
        mid_sentence_caps = re.findall(r'(?<=[a-z]\s)([A-Z][a-z]{2,})', response_text)
        # Filter out common words that happen to be capitalized
        common_caps = {'the', 'this', 'that', 'these', 'those', 'here', 'there', 'however',
                       'also', 'first', 'second', 'third', 'next', 'finally', 'additionally',
                       'furthermore', 'moreover', 'therefore', 'thus', 'hence', 'meanwhile'}
        mid_sentence_caps = [w for w in mid_sentence_caps if w.lower() not in common_caps]
        
        named_entity_count = len(cap_phrases) + len(mid_sentence_caps)
        ne_density = named_entity_count / max(word_count, 1) * 100
        score += min(ne_density * 3.0, 12.0)
        
        # === 2. Numeric Precision Score ===
        # Numbers with units
        numbers_with_units = re.findall(
            r'\b\d+(?:\.\d+)?(?:\s*(?:kg|lb|lbs|g|mg|oz|m|km|mi|miles|ft|feet|cm|mm|'
            r'inch|inches|°[CF]|degrees|mph|km/h|m/s|hours?|minutes?|mins?|seconds?|secs?|'
            r'days?|weeks?|months?|years?|ml|liters?|L|gallons?|cups?|tbsp|tsp|'
            r'tablespoons?|teaspoons?|%|percent|dollars?|\$|watts?|volts?|amps?|'
            r'calories|cal|kcal|Hz|GHz|MHz|GB|MB|KB|TB|pixels?|px|dpi))\b',
            response_text, re.IGNORECASE
        )
        
        # Standalone numbers (dates, quantities, etc.)
        all_numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|°)?\b', response_text)
        
        # Dates
        dates = re.findall(r'\b(?:19|20)\d{2}\b', response_text)
        
        # Fractions and ratios
        fractions = re.findall(r'\b\d+/\d+\b', response_text)
        
        # Dollar amounts
        dollar_amounts = re.findall(r'\$\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|k|M|B))?', response_text)
        
        numeric_score = (
            len(numbers_with_units) * 2.5 +
            len(all_numbers) * 0.8 +
            len(dates) * 2.0 +
            len(fractions) * 1.5 +
            len(dollar_amounts) * 2.0
        )
        numeric_density = numeric_score / max(word_count, 1) * 100
        score += min(numeric_density * 2.5, 15.0)
        
        # === 3. Specificity Markers ===
        # Technical/specific terms - words that are longer and less common
        # Using word length and character patterns as proxy for specificity
        specific_word_patterns = re.findall(
            r'\b(?:[a-z]{2,}-[a-z]{2,})\b',  # Hyphenated compound words
            response_text.lower()
        )
        
        # Words with specific suffixes indicating technical terms
        technical_suffixes = re.findall(
            r'\b\w+(?:tion|sion|ment|ness|ity|ance|ence|ism|ist|ous|ive|ical|ular|ular)\b',
            response_text.lower()
        )
        
        # Quoted terms or emphasized terms
        quoted_terms = re.findall(r'["\']([^"\']+)["\']', response_text)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', response_text)
        
        specificity_count = (
            len(specific_word_patterns) * 1.0 +
            len(technical_suffixes) * 0.3 +
            len(quoted_terms) * 1.5 +
            len(bold_terms) * 1.0
        )
        specificity_density = specificity_count / max(word_count, 1) * 100
        score += min(specificity_density * 1.5, 12.0)
        
        # === 4. Concrete Noun Detection ===
        # Words that follow articles (a, an, the) are likely nouns - check their specificity
        noun_phrases = re.findall(r'\b(?:a|an|the)\s+(\w+(?:\s+\w+)?)\b', response_text.lower())
        
        # Vague nouns vs specific nouns
        vague_nouns = {'thing', 'things', 'stuff', 'way', 'ways', 'lot', 'lots',
                       'kind', 'kinds', 'type', 'types', 'sort', 'sorts', 'bit',
                       'matter', 'aspect', 'aspects', 'area', 'areas', 'factor',
                       'factors', 'issue', 'issues', 'situation', 'case', 'point',
                       'idea', 'concept', 'notion', 'element', 'elements', 'part', 'parts'}
        
        vague_noun_count = sum(1 for np in noun_phrases if np.split()[0] in vague_nouns)
        specific_noun_count = len(noun_phrases) - vague_noun_count
        
        if len(noun_phrases) > 0:
            noun_specificity_ratio = specific_noun_count / len(noun_phrases)
            score += noun_specificity_ratio * 5.0
        
        # === 5. Vagueness / Hedging Penalty ===
        hedging_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bit really depends\b',
            r'\bthere are (?:many|various|several|numerous|different) (?:factors|reasons|ways|things|aspects|elements)\b',
            r'\bgenerally speaking\b', r'\bin general\b',
            r'\bcan be (?:quite|very|really|somewhat)\b',
            r'\btend to\b', r'\bmight be\b', r'\bcould be\b',
            r'\bvarious\b', r'\bnumerous\b',
            r'\band so on\b', r'\band more\b', r'\betc\.?\b',
            r'\bsort of\b', r'\bkind of\b',
            r'\bbasically\b', r'\bessentially\b',
            r'\bpretty much\b', r'\bmore or less\b',
            r'\bin some cases\b', r'\bin many cases\b',
            r'\bfor the most part\b',
            r'\bas (?:we all|everyone) knows?\b',
            r'\bneedless to say\b',
            r'\bit is (?:important|worth|interesting) to (?:note|mention|consider)\b',
            r'\bthere are (?:a lot|lots) of\b',
        ]
        
        hedge_count = 0
        response_lower = response_text.lower()
        for pattern in hedging_phrases:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_density = hedge_count / max(sentence_count, 1)
        hedge_penalty = min(hedge_density * 4.0, 10.0)
        score -= hedge_penalty
        
        # === 6. Structural Evidence of Detail ===
        # Enumerated/numbered lists with content
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*.{15,}', response_text)
        bullet_items = re.findall(r'(?:^|\n)\s*[-*•]\s*.{15,}', response_text)
        
        list_item_count = len(numbered_items) + len(bullet_items)
        if list_item_count >= 2:
            score += min(list_item_count * 0.8, 6.0)
        
        # === 7. Information-to-Filler Ratio ===
        # Filler phrases that add no information
        filler_patterns = [
            r'\bthat\'s a (?:great|good|excellent|wonderful|fantastic) (?:question|idea|point)\b',
            r'\bI\'m glad you asked\b',
            r'\blet me (?:explain|tell you|help)\b',
            r'\babsolutely\b',
            r'\bdefinitely\b',
            r'\bcertainly\b',
            r'\bof course\b',
            r'\bwithout a doubt\b',
            r'\bno doubt\b',
            r'\bsuper\b',
            r'\bawesome\b',
            r'\bamazing\b',
            r'\bwonderful\b',
            r'\bfantastic\b',
            r'\bgreat\b(?!\s+(?:wall|barrier|depression|lakes?|plains?|britain|fire))',
        ]
        
        filler_count = 0
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_penalty = min(filler_count * 0.5, 4.0)
        score -= filler_penalty
        
        # === 8. Unique Information Tokens ===
        # Count unique "informative" words (longer words that carry more meaning)
        informative_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', response_text)]
        unique_informative = set(informative_words)
        
        if word_count > 0:
            info_richness = len(unique_informative) / max(word_count, 1) * 100
            score += min(info_richness * 0.5, 8.0)
        
        # === 9. Colon-introduced specifics ===
        # Patterns like "Name: value" or "Term: description" indicate specific details
        colon_specs = re.findall(r'(?:^|\n)\s*(?:\*\*)?[A-Z][^:]{2,30}(?:\*\*)?:\s*.{5,}', response_text)
        score += min(len(colon_specs) * 0.7, 5.0)
        
        # === 10. URL/reference patterns ===
        urls = re.findall(r'https?://\S+', response_text)
        score += min(len(urls) * 2.0, 4.0)
        
        # === 11. Parenthetical clarifications (show precision) ===
        parentheticals = re.findall(r'\([^)]{3,60}\)', response_text)
        meaningful_parens = [p for p in parentheticals if re.search(r'[a-zA-Z]{3,}', p)]
        score += min(len(meaningful_parens) * 0.8, 4.0)
        
        # === 12. Mathematical/formula content ===
        math_expressions = re.findall(r'[=×÷±≈<>≤≥]|\\frac|\\times|\^2|\^3|\bsqrt\b', response_text)
        formula_patterns = re.findall(r'\b\w+\s*=\s*[\d\w(]', response_text)
        score += min((len(math_expressions) + len(formula_patterns)) * 0.6, 5.0)
        
        # === 13. Step-by-step with substance ===
        step_patterns = re.findall(r'(?:step|phase|stage)\s*\d', response_lower)
        if len(step_patterns) >= 2:
            score += min(len(step_patterns) * 0.5, 3.0)
        
        # === 14. Length bonus (moderate - longer responses tend to have more evidence) ===
        length_bonus = math.log(max(word_count, 1)) * 0.8
        score += min(length_bonus, 5.0)
        
        # === 15. Sentence-level evidence density ===
        # What fraction of sentences contain at least one concrete detail?
        evidence_sentences = 0
        for sent in sentences:
            has_number = bool(re.search(r'\d', sent))
            has_proper_noun = bool(re.search(r'(?<!\.\s)(?<![.!?]\s)[A-Z][a-z]{2,}', sent))
            has_specific_term = bool(re.search(r'["\']|(?:\w+-\w+)', sent))
            if has_number or has_proper_noun or has_specific_term:
                evidence_sentences += 1
        
        evidence_sentence_ratio = evidence_sentences / max(sentence_count, 1)
        score += evidence_sentence_ratio * 6.0
        
        # Normalize to 0-100 range
        # Typical raw scores range from about -5 to 50
        normalized = max(0.0, min(100.0, score * 1.8 + 10))
        
        return round(normalized, 2)
        
    except Exception:
        return 25.0