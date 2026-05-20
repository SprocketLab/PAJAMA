def judging_function(query, response):
    """
    Evaluates evidence density and specificity in an LLM response.
    Higher scores indicate more concrete evidence, specific details, and actionable content.
    Returns a score roughly in the range 0-100.
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
        
        response_text = response.strip()
        if len(response_text) < 5:
            return 0.0
        
        words = response_text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. LENGTH & SUBSTANCE SCORE (0-15) ===
        # Longer responses tend to have more evidence, but with diminishing returns
        length_score = min(15, 3 * math.log(1 + word_count / 10))
        score += length_score
        
        # === 2. NUMERIC/DATA DENSITY (0-20) ===
        # Look for numbers, percentages, dates, quantities
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response_text)
        year_patterns = re.findall(r'\b(?:1[0-9]{3}|2[0-9]{3})\b', response_text)
        percentage_patterns = re.findall(r'\d+(?:\.\d+)?%', response_text)
        currency_patterns = re.findall(r'[\$€£¥]\s?\d+|\d+\s?(?:dollars|euros|pounds|USD|EUR|GBP)', response_text, re.IGNORECASE)
        measurement_patterns = re.findall(r'\d+\s?(?:mg|kg|g|lb|lbs|oz|ml|L|km|mi|miles|feet|ft|inches|in|cm|mm|m|hours?|minutes?|seconds?|days?|weeks?|months?|years?|GHz|MHz|GB|MB|TB|kW|MW|mph|kph)', response_text, re.IGNORECASE)
        
        num_count = len(numbers) + len(percentage_patterns) * 2 + len(currency_patterns) * 2 + len(measurement_patterns) * 2
        numeric_density = num_count / max(word_count, 1) * 100
        numeric_score = min(20, numeric_density * 15 + min(num_count, 10) * 1.0)
        score += numeric_score
        
        # === 3. NAMED ENTITY DENSITY (0-18) ===
        # Capitalized multi-word sequences (likely proper nouns/names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response_text)
        # Filter out sentence starters - rough heuristic
        # Look for sequences that appear mid-sentence
        proper_nouns = re.findall(r'(?<=[a-z,;:]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', response_text)
        # Also look for all-caps acronyms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', response_text)
        # Filter common non-entity acronyms
        common_acronyms = {'I', 'A', 'THE', 'AND', 'OR', 'BUT', 'NOT', 'IF', 'IT', 'IS', 'AS', 'AT', 'BY', 'TO', 'IN', 'ON', 'OF', 'FOR', 'SO', 'NO', 'DO', 'AM', 'AN', 'OK', 'OP', 'RE', 'VS', 'ID', 'IE', 'EG'}
        meaningful_acronyms = [a for a in acronyms if a not in common_acronyms]
        
        entity_count = len(proper_nouns) + len(meaningful_acronyms) * 0.5
        entity_density = entity_count / max(word_count, 1) * 100
        entity_score = min(18, entity_density * 8 + min(entity_count, 8) * 1.2)
        score += entity_score
        
        # === 4. SPECIFIC REFERENCE MARKERS (0-15) ===
        # Look for citations, book titles, URLs, specific references
        reference_patterns = [
            (r'\*[^*]+\*', 2.5),           # Italic text (often titles)
            (r'"[^"]{5,}"', 2.0),           # Quoted text
            (r'https?://\S+', 3.0),         # URLs
            (r'\b(?:Chapter|Section|Article|Page|Vol\.|Volume)\s+\d+', 3.0),  # Document references
            (r'\b(?:according to|as stated by|as noted by|as described in|published in|written by)\b', 2.0),  # Attribution phrases
            (r'\bu/\w+', 2.0),              # Reddit user references
            (r'\br/\w+', 1.5),              # Subreddit references
            (r'(?:Dr\.|Prof\.|Professor|St\.|Sir|Lord)\s+[A-Z]', 2.5),  # Titled people
            (r'\b(?:University|Institute|College|Department)\s+of\s+[A-Z]', 2.0),  # Institutions
        ]
        
        ref_score = 0
        for pattern, weight in reference_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE if 'according' in pattern.lower() else 0)
            ref_score += len(matches) * weight
        
        ref_score = min(15, ref_score)
        score += ref_score
        
        # === 5. CONCRETE LANGUAGE vs VAGUE LANGUAGE (0-15) ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bfor the most part\b',
            r'\bit\'s complicated\b', r'\bthere\'s no (?:simple|easy) answer\b',
            r'\beveryone knows\b', r'\bas we all know\b',
            r'\bvarious\b', r'\bnumerous\b', r'\bcountless\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bsomewhat\b', r'\bsomehow\b',
            r'\ba lot of\b', r'\btons of\b', r'\bplenty of\b',
            r'\btend to\b', r'\btends to\b',
            r'\bcan be\b', r'\bmight be\b', r'\bcould be\b',
            r'\bin some cases\b', r'\bin many cases\b',
            r'\bI think\b', r'\bI believe\b', r'\bI feel like\b',
            r'\byou should\b', r'\byou could\b', r'\byou might\b',
        ]
        
        vague_count = 0
        for vp in vague_phrases:
            vague_count += len(re.findall(vp, response_text, re.IGNORECASE))
        
        vague_density = vague_count / max(word_count, 1) * 100
        
        # Concrete/specific language markers
        concrete_phrases = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b',
            r'\bin particular\b', r'\bexactly\b', r'\bprecisely\b',
            r'\b\d+(?:st|nd|rd|th)\b',  # ordinal numbers
            r'\be\.g\.\b', r'\bi\.e\.\b',
            r'\bthe reason (?:is|was|being)\b',
            r'\bbecause\b', r'\bdue to\b', r'\bas a result of\b',
            r'\bthis means\b', r'\bwhich means\b',
        ]
        
        concrete_count = 0
        for cp in concrete_phrases:
            concrete_count += len(re.findall(cp, response_text, re.IGNORECASE))
        
        concrete_density = concrete_count / max(word_count, 1) * 100
        
        # Net specificity score
        specificity_raw = (concrete_density * 12) - (vague_density * 6)
        specificity_score = max(0, min(15, 7.5 + specificity_raw))
        score += specificity_score
        
        # === 6. STRUCTURAL EVIDENCE (0-10) ===
        # Lists, code blocks, structured data suggest concrete content
        bullet_points = len(re.findall(r'(?:^|\n)\s*[-*•]\s', response_text))
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s', response_text))
        code_blocks = len(re.findall(r'```', response_text)) // 2
        inline_code = len(re.findall(r'`[^`]+`', response_text))
        colons_as_definitions = len(re.findall(r'\w+\s*:\s*\w', response_text))
        
        structural_items = bullet_points + numbered_items + code_blocks * 3 + inline_code * 0.5 + colons_as_definitions * 0.3
        structural_score = min(10, structural_items * 1.2)
        score += structural_score
        
        # === 7. INFORMATION DENSITY (0-7) ===
        # Unique words / total words ratio — higher ratio means less repetition
        lower_words = [w.lower().strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        if lower_words:
            unique_ratio = len(set(lower_words)) / len(lower_words)
            # Also compute average word length (longer words often more specific)
            avg_word_len = sum(len(w) for w in lower_words) / len(lower_words)
            
            # Reward vocabulary richness and word specificity
            info_density_score = min(7, (unique_ratio * 4) + max(0, (avg_word_len - 3.5) * 1.5))
        else:
            info_density_score = 0
        score += info_density_score
        
        # === 8. CAUSAL/EXPLANATORY DEPTH (0-8) ===
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bthis leads to\b',
            r'\bthe reason\b', r'\bcaused by\b', r'\bresulting in\b',
            r'\bif\s+.+\s+then\b', r'\bwhen\s+.+\s+(?:it|this|the)\b',
            r'\bin order to\b', r'\bso that\b',
            r'\bhowever\b', r'\bnevertheless\b', r'\bon the other hand\b',
            r'\bwhile\b', r'\bwhereas\b', r'\balthough\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
        ]
        
        causal_count = 0
        for cm in causal_markers:
            causal_count += len(re.findall(cm, response_text, re.IGNORECASE))
        
        causal_density = causal_count / max(sentence_count, 1)
        causal_score = min(8, causal_density * 6 + min(causal_count, 6) * 0.5)
        score += causal_score
        
        # === 9. EXAMPLE DENSITY (0-7) ===
        example_markers = re.findall(
            r'\b(?:for example|for instance|e\.g\.|such as|like\s+(?:the|a|an)\s+\w+|consider\s+(?:the|a)|take\s+(?:the|a)|imagine\s+(?:the|a))\b',
            response_text, re.IGNORECASE
        )
        example_count = len(example_markers)
        example_score = min(7, example_count * 2.5)
        score += example_score
        
        # === 10. PENALTY FOR PURE META/FILLER RESPONSES (0 to -10) ===
        meta_patterns = [
            r'\bwelcome to\b', r'\bplease read our rules\b',
            r'\byour (?:post|comment|question) (?:has been|was|will be)\b',
            r'\bI (?:can|will|would be happy to) help\b',
            r'\bthat\'s a great question\b', r'\bgood question\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b',
            r'\bI\'m here to (?:help|assist)\b',
        ]
        
        meta_count = 0
        for mp in meta_patterns:
            meta_count += len(re.findall(mp, response_text, re.IGNORECASE))
        
        meta_penalty = min(10, meta_count * 3)
        score -= meta_penalty
        
        # === 11. RESPONSE COMPLETENESS BONUS ===
        # Responses that seem cut off get a small penalty
        if response_text.endswith(('...', '…')) or (len(response_text) > 50 and not response_text[-1] in '.!?"\')]}'):
            # Might be truncated, but still has content — minor penalty
            score -= 1
        
        # Ensure score is in reasonable range
        score = max(0, min(100, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Never crash — return a neutral score
        return 25.0