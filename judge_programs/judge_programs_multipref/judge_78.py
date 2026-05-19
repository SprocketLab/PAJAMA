def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a novel approach:
    - Named entity detection via capitalization patterns
    - Numeric/quantitative density
    - Specificity ratio (specific vs vague words)
    - Parenthetical/citation density
    - Technical term detection
    - Action verb density
    - Ratio of unique information tokens
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
        if total_words == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Named Entity Detection via capitalization patterns ===
        # Look for capitalized words that aren't sentence starters
        named_entities = 0
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                if i == 0:
                    continue
                cleaned = re.sub(r'[^a-zA-Z]', '', w)
                if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                    named_entities += 1
        
        # Also detect multi-word proper nouns (consecutive capitals mid-sentence)
        multi_word_entities = len(re.findall(r'(?<!\. )(?<!\.\s)[A-Z][a-z]+\s+[A-Z][a-z]+', response_stripped))
        
        entity_density = (named_entities + multi_word_entities * 1.5) / num_sentences
        score += min(entity_density * 2.0, 12.0)
        
        # === 2. Numeric/Quantitative Density ===
        # Count various numeric patterns
        pure_numbers = re.findall(r'\b\d+\.?\d*\b', response_stripped)
        percentages = re.findall(r'\d+\.?\d*\s*%', response_stripped)
        measurements = re.findall(r'\d+\.?\d*\s*(?:kg|lb|m|km|mi|ft|cm|mm|oz|g|mg|ml|L|mph|km/h|m/s|°[CF]|degrees|hours?|minutes?|seconds?|days?|years?|months?|weeks?)', response_stripped)
        fractions = re.findall(r'\d+/\d+', response_stripped)
        ranges = re.findall(r'\d+\s*[-–—to]+\s*\d+', response_stripped)
        
        numeric_items = len(pure_numbers) + len(percentages) * 2 + len(measurements) * 2.5 + len(fractions) * 1.5 + len(ranges) * 2
        numeric_density = numeric_items / max(total_words / 100, 1)
        score += min(numeric_density * 3.0, 15.0)
        
        # === 3. Specificity vs Vagueness Ratio ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bvarious factors\b', r'\bthere are (?:many|various|several|some)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\bsometimes\b', r'\bperhaps\b',
            r'\bprobably\b', r'\bmight be\b', r'\bcould be\b',
            r'\ba lot of\b', r'\ba number of\b', r'\bquite a few\b',
            r'\band so on\b', r'\betc\.?\b', r'\band more\b',
            r'\bin some cases\b', r'\bin many cases\b',
            r'\bcan vary\b', r'\bvaries\b', r'\bdepending on\b',
            r'\bsort of\b', r'\bkind of\b', r'\bmore or less\b',
            r'\bfor the most part\b', r'\bby and large\b',
            r'\bas you know\b', r'\bas we know\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bthings like\b', r'\bstuff like\b',
        ]
        
        vague_count = 0
        response_lower = response_stripped.lower()
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response_lower))
        
        vagueness_penalty = min(vague_count * 1.5, 12.0)
        score -= vagueness_penalty
        
        # === 4. Specific/Concrete Word Patterns ===
        # Words that signal specificity
        specific_signals = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b',
            r'\bin particular\b', r'\bexactly\b', r'\bprecisely\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies show\b',
            r'\bdata shows\b', r'\bevidence suggests\b',
        ]
        
        specific_count = 0
        for pattern in specific_signals:
            specific_count += len(re.findall(pattern, response_lower))
        
        score += min(specific_count * 2.0, 10.0)
        
        # === 5. Parenthetical/Elaboration Density ===
        # Parenthetical content often contains specific details
        parentheticals = re.findall(r'\([^)]+\)', response_stripped)
        paren_content_length = sum(len(p) for p in parentheticals)
        paren_score = min(len(parentheticals) * 1.0 + paren_content_length / 50.0, 8.0)
        score += paren_score
        
        # === 6. Technical/Domain-Specific Term Density ===
        # Detect words that are likely technical (longer, less common patterns)
        technical_pattern = re.findall(r'\b[a-z]{2,}(?:tion|sion|ment|ance|ence|ity|ous|ive|ical|ular|ular|esis|osis|ism|ist|oid|ase|ide|ate|ene|yne|ium)\b', response_lower)
        # Also detect compound technical terms with hyphens
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', response_stripped)
        
        tech_density = (len(set(technical_pattern)) + len(hyphenated) * 0.5) / max(total_words / 50, 1)
        score += min(tech_density * 2.0, 8.0)
        
        # === 7. Information Density via Unique Content Words ===
        stop_words = {
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
            'he', 'she', 'his', 'her', 'i', 'me', 'my', 'which', 'what',
            'who', 'whom', 'also', 'about', 'up', 'them', 'him',
        }
        
        content_words = [re.sub(r'[^a-z]', '', w.lower()) for w in words]
        content_words = [w for w in content_words if w and w not in stop_words and len(w) > 2]
        
        unique_content = len(set(content_words))
        total_content = max(len(content_words), 1)
        
        # Unique content word ratio (higher = more diverse information)
        diversity_ratio = unique_content / total_content
        # Also raw count of unique content words (more = more information)
        info_richness = unique_content / max(num_sentences, 1)
        
        score += min(info_richness * 1.5, 10.0)
        score += min(diversity_ratio * 5.0, 5.0)
        
        # === 8. Structured Enumeration Detection ===
        # Numbered steps, lettered items, markdown formatting
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_stripped))
        lettered_items = len(re.findall(r'(?:^|\n)\s*[a-zA-Z][\.\)]\s', response_stripped))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-*•]\s', response_stripped))
        bold_terms = len(re.findall(r'\*\*[^*]+\*\*', response_stripped))
        
        structure_score = (numbered_items * 0.8 + lettered_items * 0.5 + bullet_items * 0.6 + bold_terms * 0.7)
        score += min(structure_score, 10.0)
        
        # === 9. Concrete Action Verbs ===
        action_verbs = [
            r'\bcalculate\b', r'\bmeasure\b', r'\bcompute\b', r'\bdetermine\b',
            r'\bidentify\b', r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b',
            r'\bimplement\b', r'\bconfigure\b', r'\binstall\b', r'\bdownload\b',
            r'\bcreate\b', r'\bbuild\b', r'\bdesign\b', r'\bdevelop\b',
            r'\bapply\b', r'\buse\b', r'\bselect\b', r'\bchoose\b',
            r'\bcut\b', r'\bmix\b', r'\bstir\b', r'\bbake\b', r'\bcook\b',
            r'\bpreheat\b', r'\bseason\b', r'\bserve\b',
            r'\bdrive\b', r'\bturn\b', r'\bfollow\b', r'\btake\b',
            r'\bconnect\b', r'\battach\b', r'\bsecure\b', r'\bfasten\b',
        ]
        
        action_count = 0
        for pattern in action_verbs:
            action_count += len(re.findall(pattern, response_lower))
        
        action_density = action_count / max(num_sentences, 1)
        score += min(action_density * 2.0, 6.0)
        
        # === 10. URL/Reference Patterns ===
        urls = len(re.findall(r'https?://\S+|www\.\S+', response_stripped))
        references = len(re.findall(r'\[\d+\]|\(\d{4}\)', response_stripped))
        score += min((urls + references) * 2.0, 6.0)
        
        # === 11. Equation/Formula Detection ===
        equations = len(re.findall(r'[=<>≤≥]+', response_stripped))
        math_symbols = len(re.findall(r'[×÷∑∫∂√π±∞]|\^[\d{]', response_stripped))
        formula_score = min((equations * 0.5 + math_symbols * 0.8), 6.0)
        score += formula_score
        
        # === 12. Penalize filler/fluff sentences ===
        filler_starts = [
            r"^that'?s? (?:a )?great",
            r"^great question",
            r"^good question",
            r"^absolutely",
            r"^of course",
            r"^sure thing",
            r"^i'?d be happy to",
            r"^let me help",
            r"^a classic",
        ]
        
        filler_sentence_count = 0
        for sent in sentences[:3]:  # Check first 3 sentences
            sent_lower = sent.strip().lower()
            for fp in filler_starts:
                if re.match(fp, sent_lower):
                    filler_sentence_count += 1
                    break
        
        score -= filler_sentence_count * 1.5
        
        # === 13. Response completeness signal ===
        # Penalize responses that appear truncated mid-sentence
        if response_stripped and response_stripped[-1] not in '.!?"\')':
            score -= 2.0
        
        # === 14. Length bonus (moderate - longer responses tend to have more evidence) ===
        length_bonus = math.log(max(total_words, 1)) * 1.2
        score += min(length_bonus, 8.0)
        
        # Normalize to 0-100 range
        # Theoretical range roughly: -15 to ~80
        score = max(0.0, score)
        score = min(100.0, score)
        
        return round(score, 2)
        
    except Exception:
        return 0.0