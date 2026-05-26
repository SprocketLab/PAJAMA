def judging_function(query, response):
    """
    Evaluates evidence density and specificity by analyzing concrete details,
    named entities, numbers, structured content, and actionable information.
    
    This variant focuses on:
    - Named entity detection (capitalized multi-word phrases, proper nouns)
    - Numeric/quantitative density (numbers, measurements, percentages)
    - Specificity markers (technical terms, precise language)
    - Structural depth (nested lists, step-by-step with details)
    - Anti-vagueness scoring (penalizing filler phrases)
    - Information-to-fluff ratio
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
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        score = 0.0
        
        # === 1. Numeric/Quantitative Density ===
        # Find all numbers including decimals, fractions, percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|°|°C|°F|km|m|kg|lb|ft|mph|m/s|cm|mm|mg|ml|L|Hz|kHz|MHz|GHz)?\b', response_text)
        # Also find written-out specific quantities
        specific_quantities = re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|degrees|miles|kilometers|meters|feet|inches|pounds|kilograms|grams|liters|gallons|hours|minutes|seconds|days|weeks|months|years|dollars|euros|cents)\b', response_text, re.IGNORECASE)
        
        numeric_count = len(numbers) + len(specific_quantities)
        # Score: density of numbers per 100 words, capped
        numeric_density = (numeric_count / word_count) * 100
        numeric_score = min(numeric_density * 3.0, 15.0)
        score += numeric_score
        
        # === 2. Named Entity Detection (capitalized phrases) ===
        # Multi-word capitalized phrases (likely proper nouns/names)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response_text)
        # Single capitalized words not at sentence start (approximate)
        # Remove first word of each sentence
        sentence_starts = set()
        for s in sentences:
            sw = s.split()
            if sw:
                sentence_starts.add(sw[0])
        
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', response_text)
        # Filter out common sentence starters and generic words
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'Here', 'There', 'What', 'When', 
                       'Where', 'Why', 'How', 'Which', 'Who', 'Some', 'Many', 'Most', 'All',
                       'Any', 'Each', 'Every', 'Both', 'Few', 'Several', 'Other', 'Another',
                       'First', 'Second', 'Third', 'Next', 'Also', 'However', 'Therefore',
                       'Furthermore', 'Moreover', 'Additionally', 'Finally', 'Overall',
                       'Certainly', 'Absolutely', 'Yes', 'No', 'Not', 'But', 'And', 'Or',
                       'If', 'For', 'With', 'From', 'Into', 'Through', 'During', 'Before',
                       'After', 'Above', 'Below', 'Between', 'Under', 'Over', 'Let', 'You',
                       'Your', 'Our', 'His', 'Her', 'Its', 'Their', 'We', 'They', 'She',
                       'He', 'It', 'Awesome', 'Great', 'Good', 'Nice', 'Well', 'Sure',
                       'Okay', 'Right', 'True', 'False', 'Step', 'Note', 'Important',
                       'Remember', 'Consider', 'Think', 'Use', 'Try', 'Make', 'Take',
                       'Give', 'Get', 'Set', 'Put', 'Keep', 'Find', 'Start', 'Begin',
                       'Continue', 'Stop', 'End', 'Choose', 'Select', 'Pick', 'Gather',
                       'Prepare', 'Mix', 'Add', 'Remove', 'Place', 'Bring', 'Organize',
                       'Plan', 'Check', 'Identify', 'Determine', 'Decide', 'Once'}
        
        meaningful_caps = [w for w in capitalized_words if w not in common_caps]
        entity_count = len(named_entities) * 2 + len(meaningful_caps) * 0.5
        entity_density = (entity_count / word_count) * 100
        entity_score = min(entity_density * 2.5, 12.0)
        score += entity_score
        
        # === 3. Specificity Markers ===
        # Technical/domain terms (longer, more specific words)
        long_words = [w for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) >= 8]
        long_word_density = (len(long_words) / word_count) * 100
        specificity_score = min(long_word_density * 0.8, 10.0)
        
        # Compound/hyphenated terms (often more specific)
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', response_text)
        specificity_score += min(len(hyphenated) * 0.3, 3.0)
        
        # Parenthetical clarifications (e.g., "also known as", abbreviations)
        parentheticals = re.findall(r'\([^)]+\)', response_text)
        specificity_score += min(len(parentheticals) * 0.8, 4.0)
        
        # Specific connectors that introduce evidence
        evidence_phrases = re.findall(
            r'\b(?:for example|for instance|such as|specifically|in particular|namely|'
            r'e\.g\.|i\.e\.|according to|research shows|studies show|data shows|'
            r'evidence suggests|known as|referred to as|defined as|consisting of|'
            r'comprised of|including but not limited to)\b',
            response_text, re.IGNORECASE
        )
        specificity_score += min(len(evidence_phrases) * 1.5, 8.0)
        
        score += min(specificity_score, 20.0)
        
        # === 4. Structural Depth and Organization ===
        # Markdown headers
        headers = re.findall(r'#{1,4}\s+.+', response_text)
        header_score = min(len(headers) * 1.0, 5.0)
        
        # Numbered steps with actual content
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*.+', response_text)
        # Check average length of numbered items (longer = more detailed)
        if numbered_items:
            avg_item_len = sum(len(item.split()) for item in numbered_items) / len(numbered_items)
            # Reward more items and longer items
            list_detail_score = min(len(numbered_items) * 0.5, 4.0) + min(avg_item_len / 15.0 * 3.0, 4.0)
        else:
            list_detail_score = 0.0
        
        # Bold text (often highlights key terms)
        bold_items = re.findall(r'\*\*[^*]+\*\*', response_text)
        bold_score = min(len(bold_items) * 0.4, 4.0)
        
        # Bullet points
        bullets = re.findall(r'(?:^|\n)\s*[-*•]\s+.+', response_text)
        bullet_score = min(len(bullets) * 0.3, 3.0)
        
        structure_score = header_score + list_detail_score + bold_score + bullet_score
        score += min(structure_score, 15.0)
        
        # === 5. Anti-Vagueness Penalty ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are many\b', r'\bthere are several\b',
            r'\bthere are various\b', r'\bgenerally speaking\b',
            r'\bin general\b', r'\btypically\b', r'\busually\b',
            r'\boften\b', r'\bsometimes\b', r'\bmight be\b',
            r'\bcould be\b', r'\bmay or may not\b', r'\bit\'s hard to say\b',
            r'\bthat said\b', r'\bon the other hand\b',
            r'\bit really depends\b', r'\beveryone is different\b',
            r'\bthere\'s no one.size.fits.all\b', r'\bcan vary\b',
            r'\bwill vary\b', r'\bvaries\b', r'\bdepending on\b',
            r'\bcan be a\b', r'\bcan be an\b',
        ]
        
        vague_count = 0
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response_text, re.IGNORECASE))
        
        vague_density = (vague_count / sentence_count)
        vague_penalty = min(vague_density * 3.0, 8.0)
        score -= vague_penalty
        
        # Heavy filler phrases
        filler_phrases = [
            r'\bthat\'s a great\b', r'\bgreat question\b', r'\bgood question\b',
            r'\bI\'m glad you asked\b', r'\babsolutely\b', r'\bdefinitely\b',
            r'\bwonderful\b', r'\bamazing\b', r'\bfantastic\b',
            r'\bthat\'s awesome\b', r'\bsuper\b',
        ]
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_text, re.IGNORECASE))
        
        filler_penalty = min(filler_count * 0.5, 3.0)
        score -= filler_penalty
        
        # === 6. Information Density (content words ratio) ===
        # Ratio of content words to total words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their', 'he', 'she',
            'him', 'her', 'his', 'my', 'me', 'i', 'also', 'which', 'what', 'who',
            'whom', 'whose', 'them', 'us',
        }
        
        clean_words = [re.sub(r'[^a-zA-Z]', '', w).lower() for w in words]
        clean_words = [w for w in clean_words if w]
        content_words = [w for w in clean_words if w not in stop_words and len(w) > 2]
        
        if clean_words:
            content_ratio = len(content_words) / len(clean_words)
            # Higher content ratio = more informative
            info_density_score = content_ratio * 12.0
        else:
            info_density_score = 0.0
        
        score += min(info_density_score, 10.0)
        
        # === 7. Sentence-level specificity ===
        # Average words per sentence (very short sentences may lack detail)
        avg_sentence_length = word_count / sentence_count
        if avg_sentence_length >= 12 and avg_sentence_length <= 30:
            sentence_len_score = 3.0
        elif avg_sentence_length >= 8:
            sentence_len_score = 1.5
        else:
            sentence_len_score = 0.0
        score += sentence_len_score
        
        # === 8. Concrete action verbs and instructional language ===
        action_patterns = re.findall(
            r'\b(?:measure|calculate|multiply|divide|add|subtract|mix|stir|combine|'
            r'attach|connect|install|remove|insert|apply|spread|pour|heat|cool|'
            r'download|upload|click|select|navigate|open|close|save|delete|'
            r'drive|turn|exit|merge|follow|continue|proceed|cross|'
            r'preheat|bake|roast|grill|sauté|simmer|boil|chop|dice|slice|mince|'
            r'cut|trim|fold|wrap|seal|cover|uncover|drain|rinse|wash)\b',
            response_text, re.IGNORECASE
        )
        action_score = min(len(action_patterns) * 0.5, 8.0)
        score += action_score
        
        # === 9. Mathematical/formula content ===
        formulas = re.findall(r'[=×÷±≈≤≥<>]\s*\d', response_text)
        math_expressions = re.findall(r'\b\w+\s*[=]\s*[\d\w(]', response_text)
        latex_like = re.findall(r'\\[a-z]+\{', response_text)
        math_score = min((len(formulas) + len(math_expressions) + len(latex_like)) * 0.8, 6.0)
        score += math_score
        
        # === 10. URL/reference patterns ===
        urls = re.findall(r'https?://\S+', response_text)
        references = re.findall(r'\b(?:source|reference|citation|according to|as reported by|published in)\b', response_text, re.IGNORECASE)
        ref_score = min((len(urls) * 1.5 + len(references) * 0.8), 5.0)
        score += ref_score
        
        # === 11. Length bonus (longer responses tend to have more evidence, but diminishing returns) ===
        length_bonus = min(math.log(max(word_count, 1) + 1) * 1.2, 8.0)
        score += length_bonus
        
        # === 12. Unique information tokens ===
        # Count unique non-stopword tokens (more unique = more diverse info)
        unique_content = set(content_words)
        if content_words:
            uniqueness_ratio = len(unique_content) / len(content_words)
            unique_score = uniqueness_ratio * 5.0
        else:
            unique_score = 0.0
        score += min(unique_score, 5.0)
        
        # Normalize to 0-100 range
        # Theoretical max is around 80-90, so scale accordingly
        final_score = max(0.0, min(score * 1.1, 100.0))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 25.0
            return 5.0
        except:
            return 5.0