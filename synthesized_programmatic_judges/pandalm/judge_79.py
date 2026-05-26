def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a fundamentally different approach:
    Named Entity / Proper Noun detection heuristics, numeric/data density, 
    sentence-level information gain analysis, and repetition penalty.
    
    This variant focuses on:
    1. Capitalized proper nouns / named entities detection
    2. Numeric and quantitative expression density
    3. Sentence-level novelty (new information per sentence)
    4. Specificity markers (technical terms, action verbs, precise language)
    5. Vagueness/filler penalty using pattern matching
    6. Repetition penalty at n-gram level
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        query_str = query.strip() if query and isinstance(query, str) else ""
        
        words = response.split()
        num_words = len(words)
        if num_words == 0:
            return 0.5
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Numeric / Quantitative Expression Density ----
        # Find numbers, percentages, dates, measurements, currencies
        numeric_patterns = [
            r'\b\d+\.?\d*%',           # percentages
            r'\$\d+',                    # dollar amounts
            r'\b\d{4}\b',              # years
            r'\b\d+\.?\d*\s*(kg|lb|km|mi|cm|mm|m|ft|inch|oz|gb|mb|tb|ghz|mhz|mph|kph)\b',  # measurements
            r'\b\d+[\d,]*\.?\d*\b',    # general numbers
        ]
        
        numeric_count = 0
        for pattern in numeric_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            numeric_count += len(matches)
        
        numeric_density = min(numeric_count / max(num_words, 1) * 50, 15)
        
        # ---- 2. Named Entity / Proper Noun Heuristic ----
        # Words that are capitalized but not at start of sentence
        query_words_lower = set(query_str.lower().split())
        
        proper_noun_count = 0
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                clean_w = re.sub(r'[^a-zA-Z]', '', w)
                if not clean_w:
                    continue
                if i > 0 and clean_w[0].isupper() and len(clean_w) > 1:
                    if clean_w.lower() not in query_words_lower:
                        proper_noun_count += 1
                # Also count acronyms
                if clean_w.isupper() and len(clean_w) >= 2 and len(clean_w) <= 6:
                    proper_noun_count += 1
        
        proper_noun_score = min(proper_noun_count * 1.5, 12)
        
        # ---- 3. Sentence-Level Novelty (Information Gain) ----
        # Each sentence should introduce new content words not seen before
        seen_content_words = set()
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'them', 'they', 'their', 'this', 'that',
            'these', 'those', 'it', 'its', 'he', 'she', 'we', 'you', 'i', 'me',
            'my', 'your', 'his', 'her', 'our', 'also', 'which', 'what', 'who',
        }
        
        novelty_scores = []
        for sent in sentences:
            sent_words = [w.lower().strip('.,;:!?()[]"\'') for w in sent.split()]
            content_words = [w for w in sent_words if w and w not in stop_words and len(w) > 2]
            if not content_words:
                novelty_scores.append(0)
                continue
            new_words = [w for w in content_words if w not in seen_content_words]
            novelty_ratio = len(new_words) / len(content_words) if content_words else 0
            novelty_scores.append(novelty_ratio)
            seen_content_words.update(content_words)
        
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        novelty_score = avg_novelty * 15
        
        # ---- 4. Specificity Markers ----
        # Technical/precise language indicators
        specificity_patterns = [
            r'\b(specifically|precisely|exactly|approximately|roughly)\b',
            r'\b(for example|for instance|such as|e\.g\.|i\.e\.)\b',
            r'\b(according to|based on|research shows|studies indicate|data suggests)\b',
            r'\b(first|second|third|fourth|fifth|step \d)\b',
            r'\b(increase|decrease|improve|reduce|enhance|optimize|maximize|minimize)\b',
            r'\b(percent|percentage|ratio|rate|average|median|total)\b',
            r'\b(method|technique|approach|strategy|algorithm|process|mechanism|protocol)\b',
            r'\b(including|consists of|comprises|contains|features)\b',
        ]
        
        specificity_count = 0
        for pattern in specificity_patterns:
            specificity_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        specificity_score = min(specificity_count / max(num_words, 1) * 80, 12)
        
        # ---- 5. Vagueness / Filler Penalty ----
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (many|various|several|different) (factors|reasons|ways|things)\b',
            r'\bin many ways\b', r'\bin various ways\b',
            r'\bgenerally speaking\b', r'\bin general\b',
            r'\bkind of\b', r'\bsort of\b',
            r'\bmore or less\b', r'\bto some extent\b',
            r'\ba lot of\b', r'\ba number of\b',
            r'\bquite\b', r'\brather\b', r'\bsomewhat\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\btend to\b', r'\bmight be\b',
            r'\band so on\b', r'\betc\b', r'\band more\b',
            r'\bstuff\b', r'\bthings\b',
        ]
        
        vague_count = 0
        for pattern in vague_patterns:
            vague_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        vague_density = vague_count / max(num_words, 1)
        vagueness_penalty = min(vague_density * 200, 12)
        
        # ---- 6. N-gram Repetition Penalty ----
        # Penalize repeated trigrams heavily (sign of low-quality/degenerate output)
        if num_words >= 3:
            trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(total_trigrams, 1)
        else:
            repetition_ratio = 0
        
        repetition_penalty = min(repetition_ratio * 40, 20)
        
        # ---- 7. Actionable Detail Score ----
        # Detect action verbs and concrete instructions
        action_patterns = [
            r'\b(click|select|choose|open|close|create|build|write|read|send|receive)\b',
            r'\b(install|download|upload|configure|setup|set up|enable|disable)\b',
            r'\b(track|monitor|measure|analyze|evaluate|compare|calculate)\b',
            r'\b(categorize|organize|manage|plan|schedule|prioritize)\b',
            r'\b(allows?|enables?|provides?|offers?|supports?|includes?)\b',
        ]
        
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        action_score = min(action_count / max(num_words, 1) * 60, 10)
        
        # ---- 8. Content Richness (unique content words / total words) ----
        all_content = [w.lower().strip('.,;:!?()[]"\'') for w in words 
                       if w.lower().strip('.,;:!?()[]"\'') not in stop_words 
                       and len(w.strip('.,;:!?()[]"\'')) > 2]
        
        unique_content = set(all_content)
        content_richness = len(unique_content) / max(num_words, 1)
        richness_score = content_richness * 15
        
        # ---- 9. Length Appropriateness ----
        # Reward substantive responses but with diminishing returns
        length_score = min(math.log(max(num_words, 1) + 1) * 2, 10)
        
        # ---- 10. Parenthetical/Elaboration Density ----
        # Parentheses, commas for appositive clauses, colons for elaboration
        elaboration_markers = len(re.findall(r'[(:;,]', response))
        elaboration_density = elaboration_markers / max(num_sentences, 1)
        elaboration_score = min(elaboration_density * 1.5, 8)
        
        # ---- Combine Scores ----
        total_score = (
            numeric_density       # 0-15: numbers and data
            + proper_noun_score   # 0-12: named entities
            + novelty_score       # 0-15: sentence-level information gain
            + specificity_score   # 0-12: precise language markers
            + action_score        # 0-10: actionable details
            + richness_score      # 0-15: content word diversity
            + length_score        # 0-10: substantive length
            + elaboration_score   # 0-8: elaboration markers
            - vagueness_penalty   # 0-12: vague filler penalty
            - repetition_penalty  # 0-20: repetition penalty
        )
        
        # Normalize to 0-100 range
        # Max theoretical: 15+12+15+12+10+15+10+8 = 97
        # Practical max around 60-70
        final_score = max(0.0, min(100.0, total_score))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            return max(0.0, min(10.0, len(str(response)) / 50.0))
        except Exception:
            return 1.0