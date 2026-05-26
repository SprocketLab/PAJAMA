def judging_function(query, response):
    """
    Evaluates evidence density and specificity by detecting concrete evidence markers:
    named entities (capitalized multi-word phrases), numbers/measurements, specific
    technical terms, actionable instructions, and penalizing vague hedging patterns.
    
    This variant focuses on:
    1. Named entity density (capitalized phrases, proper nouns)
    2. Numeric/quantitative information density
    3. Technical/domain-specific term detection
    4. Actionable detail markers (specific verbs, step references)
    5. Vague filler penalty (hedging, weasel words)
    6. Information-to-filler ratio
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
        if word_count == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        sentence_count = max(len(sentences), 1)
        
        # === 1. Named Entity Detection ===
        # Look for capitalized multi-word phrases (proper nouns, place names, etc.)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response_text)
        # Single capitalized words not at sentence start
        # Find words that are capitalized but not at the start of a sentence
        mid_sentence_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}\b', response_text)
        
        # Specific named things in bold/markdown
        bold_items = re.findall(r'\*\*([^*]+)\*\*', response_text)
        bold_count = len(bold_items)
        
        entity_score = (len(named_entities) * 2.0 + len(mid_sentence_caps) * 1.0) / max(word_count / 50, 1)
        entity_score = min(entity_score, 15.0)
        
        # === 2. Numeric/Quantitative Information ===
        # Specific numbers (not just "1.", "2." list markers)
        # Numbers with units or in context
        numbers_with_units = re.findall(
            r'\b\d+[\.,]?\d*\s*(?:kg|lb|lbs|m|km|mi|miles|feet|ft|cm|mm|inch|inches|'
            r'hours?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|'
            r'degrees?|°|%|percent|dollars?|\$|€|£|mph|km/h|m/s|'
            r'cups?|tbsp|tsp|tablespoons?|teaspoons?|oz|ounces?|grams?|g|mg|ml|liters?|'
            r'calories|cal|watts?|volts?|amps?|GB|MB|TB|GHz|MHz)\b',
            response_text, re.IGNORECASE
        )
        
        # Standalone specific numbers (like years, temperatures, etc.)
        specific_numbers = re.findall(r'\b(?:19|20)\d{2}\b', response_text)  # years
        decimal_numbers = re.findall(r'\b\d+\.\d+\b', response_text)
        large_numbers = re.findall(r'\b\d{3,}\b', response_text)
        fractions = re.findall(r'\b\d+/\d+\b', response_text)
        
        # All numbers in response
        all_numbers = re.findall(r'\b\d+[\.,]?\d*\b', response_text)
        # Subtract simple list markers (1. 2. 3. etc)
        list_markers = re.findall(r'(?:^|\n)\s*\d+[.)]\s', response_text)
        meaningful_numbers = max(len(all_numbers) - len(list_markers), 0)
        
        numeric_score = (
            len(numbers_with_units) * 3.0 +
            len(specific_numbers) * 2.5 +
            len(decimal_numbers) * 2.0 +
            len(large_numbers) * 1.5 +
            len(fractions) * 1.5 +
            meaningful_numbers * 0.5
        ) / max(word_count / 40, 1)
        numeric_score = min(numeric_score, 20.0)
        
        # === 3. Technical/Domain-Specific Terms ===
        # Detect terms that suggest domain expertise
        # Look for compound technical terms, parenthetical clarifications, formulas
        parenthetical = re.findall(r'\([^)]{3,50}\)', response_text)
        formulas = re.findall(r'[A-Za-z]+\s*[=<>≤≥]+\s*[\d\w(]', response_text)
        colon_definitions = re.findall(r'\b\w+\s*:\s*\w', response_text)
        
        # Words with special characters suggesting technical content
        technical_markers = re.findall(r'\b(?:e\.g\.|i\.e\.|etc\.|vs\.|approx\.)\b', response_text, re.IGNORECASE)
        
        # Specific ingredient/material/tool mentions (longer specific words)
        long_specific_words = [w for w in words if len(w) > 8 and w[0].islower() and w.isalpha()]
        
        technical_score = (
            len(parenthetical) * 1.5 +
            len(formulas) * 3.0 +
            len(colon_definitions) * 0.5 +
            len(technical_markers) * 1.5 +
            len(long_specific_words) * 0.3
        ) / max(word_count / 40, 1)
        technical_score = min(technical_score, 15.0)
        
        # === 4. Actionable Detail Markers ===
        # Specific action verbs and instructional language
        action_patterns = [
            r'\b(?:click|navigate|open|select|choose|type|enter|press|download|install|connect|attach|mix|stir|pour|heat|bake|cook|add|remove|cut|place|set|adjust|measure|calculate|apply|use|insert)\b',
        ]
        action_count = 0
        for pat in action_patterns:
            action_count += len(re.findall(pat, response_text, re.IGNORECASE))
        
        # Step-by-step structure with actual content
        step_references = re.findall(r'(?:step\s+\d|#\d|\bfirst\b|\bsecond\b|\bthird\b|\bthen\b|\bnext\b|\bfinally\b)', response_text, re.IGNORECASE)
        
        # Specific recommendations (named products, places, etc.)
        quoted_items = re.findall(r'"[^"]{3,50}"', response_text)
        
        actionable_score = (
            action_count * 0.5 +
            len(step_references) * 1.0 +
            len(quoted_items) * 2.0
        ) / max(word_count / 40, 1)
        actionable_score = min(actionable_score, 12.0)
        
        # === 5. Structural Richness ===
        # Markdown formatting suggests organized, detailed content
        has_headers = len(re.findall(r'#{1,4}\s+\w', response_text))
        has_bold = bold_count
        has_lists = len(re.findall(r'(?:^|\n)\s*[-*•]\s+\w', response_text))
        has_numbered = len(list_markers)
        
        # Sub-items or nested structure
        nested_items = len(re.findall(r'(?:^|\n)\s{2,}[-*•]\s+\w', response_text))
        
        structural_score = (
            min(has_headers, 5) * 1.5 +
            min(has_bold, 8) * 0.8 +
            min(has_lists + has_numbered, 10) * 0.5 +
            nested_items * 1.0
        )
        structural_score = min(structural_score, 15.0)
        
        # === 6. Vague/Hedging Language Penalty ===
        hedging_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bit varies\b', r'\bgenerally speaking\b',
            r'\bthere are (?:many|various|several|different|numerous) (?:factors|reasons|ways|things|aspects|elements)\b',
            r'\bin general\b', r'\boverall\b',
            r'\bcan be\b.*\bor\b', r'\bmay or may not\b',
            r'\bkind of\b', r'\bsort of\b',
            r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b',
            r'\btend to\b', r'\busually\b',
            r'\bit\'s important to note\b', r'\bit is important to\b',
            r'\bit\'s worth noting\b', r'\bit is worth\b',
            r'\bthere are many\b', r'\bthere are several\b',
            r'\bas you know\b', r'\bas we all know\b',
            r'\bin some cases\b', r'\bin many cases\b',
            r'\bto some extent\b', r'\bin a way\b',
            r'\bvarious\b', r'\bnumerous\b',
        ]
        
        hedge_count = 0
        for pat in hedging_patterns:
            hedge_count += len(re.findall(pat, response_text, re.IGNORECASE))
        
        # Generic filler phrases
        filler_patterns = [
            r'\bcan be a (?:great|good|fun|wonderful|rewarding)\b',
            r'\bhere are (?:some|a few)\b',
            r'\bthat\'s a great\b',
            r'\bgreat question\b',
            r'\blet me\b',
            r'\bI hope this helps\b',
        ]
        filler_count = 0
        for pat in filler_patterns:
            filler_count += len(re.findall(pat, response_text, re.IGNORECASE))
        
        hedge_penalty = (hedge_count * 1.5 + filler_count * 0.5) / max(sentence_count, 1)
        hedge_penalty = min(hedge_penalty, 10.0)
        
        # === 7. Information Density (unique content words per sentence) ===
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
            'those', 'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their',
            'them', 'he', 'she', 'his', 'her', 'i', 'me', 'my', 'which', 'what',
            'who', 'whom', 'also', 'well', 'like', 'even', 'still', 'much',
        }
        
        content_words = [w.lower().strip('.,;:!?()[]{}"\'-') for w in words 
                        if w.lower().strip('.,;:!?()[]{}"\'-') not in stop_words 
                        and len(w.strip('.,;:!?()[]{}"\'-')) > 2]
        
        unique_content = set(content_words)
        content_density = len(unique_content) / max(word_count, 1) * 100
        
        # Ratio of unique content words to total content words (repetition penalty)
        if len(content_words) > 0:
            uniqueness_ratio = len(unique_content) / len(content_words)
        else:
            uniqueness_ratio = 0
        
        density_score = content_density * 0.15 + uniqueness_ratio * 5.0
        density_score = min(density_score, 15.0)
        
        # === 8. Specificity of examples ===
        # Look for "for example", "such as", "like X" followed by specific items
        example_intros = re.findall(
            r'(?:for example|for instance|such as|e\.g\.|including|specifically)[,:]?\s+\w',
            response_text, re.IGNORECASE
        )
        
        # Specific comparisons
        comparisons = re.findall(r'\b(?:compared to|rather than|instead of|unlike|whereas)\b', 
                                response_text, re.IGNORECASE)
        
        specificity_score = (len(example_intros) * 2.5 + len(comparisons) * 2.0) / max(sentence_count / 3, 1)
        specificity_score = min(specificity_score, 10.0)
        
        # === 9. Response completeness signal ===
        # Responses that end mid-sentence are truncated
        truncation_penalty = 0
        if response_text and response_text[-1] not in '.!?"\')':
            truncation_penalty = 2.0
        
        # === FINAL SCORE COMPUTATION ===
        raw_score = (
            entity_score * 1.0 +
            numeric_score * 1.2 +
            technical_score * 1.0 +
            actionable_score * 0.8 +
            structural_score * 0.7 +
            density_score * 0.8 +
            specificity_score * 0.9 -
            hedge_penalty * 1.5 -
            truncation_penalty
        )
        
        # Length bonus: slightly reward longer responses (more room for evidence)
        # but with strong diminishing returns
        length_factor = math.log(max(word_count, 10)) / math.log(300)
        length_factor = min(max(length_factor, 0.5), 1.3)
        
        final_score = raw_score * length_factor
        
        # Normalize to 0-100 range
        final_score = max(0.0, min(100.0, final_score * 1.5))
        
        return round(final_score, 2)
        
    except Exception:
        return 0.0