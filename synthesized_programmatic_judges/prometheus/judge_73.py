def judging_function(query, response):
    """
    Evaluates evidence density and specificity in an LLM response.
    Higher scores indicate more concrete evidence, specific details, and actionable content.
    Returns a score from 0 to 10.
    """
    try:
        import re
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        words = response_lower.split()
        word_count = len(words)
        
        if word_count < 3:
            return 0.5
        
        score = 0.0
        
        # === 1. NUMBERS AND QUANTITATIVE DATA (0-2 points) ===
        # Count specific numbers, percentages, dates, measurements
        number_patterns = [
            r'\b\d+\.?\d*\s*%',           # percentages
            r'\b\d{4}\b',                   # years
            r'\b\d+\.?\d*\s*(pounds?|lbs?|kg|grams?|oz|ounces?|cups?|tbsp|tsp|liters?|ml|gallons?)\b',  # measurements
            r'\b\d+\.?\d*\s*(minutes?|hours?|seconds?|days?|weeks?|months?|years?)\b',  # time
            r'\b\d+\.?\d*\s*(miles?|km|kilometers?|meters?|feet|inches?|cm)\b',  # distance
            r'\b\$\d+',                     # currency
            r'\b\d+(?:,\d{3})+\b',         # large numbers with commas
            r'\b\d+/\d+\b',                # fractions
        ]
        
        number_count = 0
        for pattern in number_patterns:
            number_count += len(re.findall(pattern, response_lower))
        
        # Also count standalone numbers that aren't just "1" or "2" in lists
        all_numbers = re.findall(r'\b\d+\.?\d*\b', response)
        # Filter out simple enumeration numbers (1, 2, 3 at start of lines or after periods)
        enum_numbers = re.findall(r'(?:^|\n)\s*(\d+)[.\):]', response)
        specific_numbers = len(all_numbers) - len(enum_numbers)
        number_count += max(0, specific_numbers)
        
        number_score = min(2.0, number_count * 0.3)
        score += number_score
        
        # === 2. NAMED ENTITIES AND SPECIFIC REFERENCES (0-2 points) ===
        # Detect capitalized multi-word phrases (likely proper nouns/named entities)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response)
        # Filter out sentence starters
        sentences = re.split(r'[.!?]\s+', response)
        sentence_starters = set()
        for s in sentences:
            s = s.strip()
            if s:
                first_words = s.split()[:2]
                if len(first_words) >= 2:
                    sentence_starters.add(' '.join(first_words))
        
        real_entities = [ne for ne in named_entities if ne not in sentence_starters]
        
        # Also detect technical terms, specific methodologies
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like AI, API, SQL
        ]
        acronym_count = len(re.findall(r'\b[A-Z]{2,}\b', response))
        
        entity_score = min(2.0, len(real_entities) * 0.25 + acronym_count * 0.15)
        score += entity_score
        
        # === 3. STRUCTURED/ACTIONABLE CONTENT (0-1.5 points) ===
        # Numbered lists, bullet points, step-by-step instructions
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[.\):]', response)
        bullet_items = re.findall(r'(?:^|\n)\s*[-•*]\s', response)
        colon_definitions = re.findall(r'\b\w+(?:\s+\w+){0,3}:\s', response)
        
        structure_count = len(numbered_items) + len(bullet_items) + len(colon_definitions) * 0.5
        structure_score = min(1.5, structure_count * 0.2)
        score += structure_score
        
        # === 4. CONCRETE ACTION VERBS AND SPECIFICITY MARKERS (0-1.5 points) ===
        specific_action_words = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'specifically', 'precisely', 'exactly', 'particularly',
            'for example', 'for instance', 'such as', 'including',
            'in particular', 'namely', 'e.g.', 'i.e.',
            'according to', 'research shows', 'studies show',
            'demonstrated', 'illustrated', 'evidenced',
        ]
        
        specificity_count = 0
        for marker in specific_action_words:
            specificity_count += response_lower.count(marker)
        
        specificity_score = min(1.5, specificity_count * 0.25)
        score += specificity_score
        
        # === 5. VAGUENESS PENALTY (0 to -2 points) ===
        vague_phrases = [
            'many people', 'some people', 'a lot of', 'various factors',
            'it depends', 'there are many', 'there are various',
            'in general', 'generally speaking', 'as we all know',
            'everyone knows', 'it is said', 'they say',
            'kind of', 'sort of', 'more or less', 'pretty much',
            'stuff like that', 'things like that', 'and so on',
            'and whatnot', 'you know', 'basically',
            'just', 'maybe', 'probably', 'perhaps', 'might',
            'could be', 'may or may not', 'hard to say',
            'it\'s complicated', 'who knows', 'not sure',
            'i guess', 'i suppose', 'somewhere', 'somehow',
            'something like', 'whatever', 'wherever',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += response_lower.count(phrase)
        
        # Normalize by word count
        vague_density = vague_count / max(1, word_count) * 100
        vague_penalty = min(2.0, vague_density * 0.8)
        score -= vague_penalty
        
        # === 6. DISMISSIVE/SHALLOW PENALTY (0 to -1 point) ===
        dismissive_phrases = [
            'just do', 'just try', 'just go', 'just get',
            'you should be able', 'it\'s not that hard',
            'simply', 'obviously', 'clearly', 'of course',
            'no big deal', 'don\'t worry about it',
            'you\'ll figure it out', 'you\'ll get there',
            'keep trying', 'just keep',
        ]
        
        dismissive_count = 0
        for phrase in dismissive_phrases:
            dismissive_count += response_lower.count(phrase)
        
        dismissive_penalty = min(1.0, dismissive_count * 0.2)
        score -= dismissive_penalty
        
        # === 7. DETAIL DENSITY - ratio of content words to filler (0-1.5 points) ===
        filler_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'but', 'and', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'about', 'up', 'out', 'if', 'then', 'that', 'this',
            'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'our', 'their', 'what', 'which', 'who', 'when', 'where', 'how',
            'there', 'here', 'also', 'like', 'well', 'back', 'even', 'still',
        }
        
        content_words = [w for w in words if w.strip(string.punctuation) not in filler_words and len(w) > 2]
        content_ratio = len(content_words) / max(1, word_count)
        detail_density_score = min(1.5, content_ratio * 3.0)
        score += detail_density_score
        
        # === 8. RESPONSE LENGTH AND COMPLETENESS (0-1 point) ===
        # Longer responses tend to have more detail, but diminishing returns
        if word_count < 20:
            length_score = 0.0
        elif word_count < 50:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.6
        elif word_count < 200:
            length_score = 0.8
        else:
            length_score = 1.0
        score += length_score
        
        # === 9. CONDITIONAL/CONTEXTUAL AWARENESS (0-1 point) ===
        # Does the response address the query specifically rather than generically?
        query_words = set(query.lower().split())
        query_content = {w.strip(string.punctuation) for w in query_words if w.strip(string.punctuation) not in filler_words and len(w) > 3}
        response_words_set = set(words)
        
        if query_content:
            overlap = len(query_content.intersection(response_words_set))
            relevance_ratio = overlap / max(1, len(query_content))
            relevance_score = min(1.0, relevance_ratio * 1.5)
        else:
            relevance_score = 0.5
        score += relevance_score
        
        # === 10. UNIQUE VOCABULARY RICHNESS (0-0.5 points) ===
        unique_words = set(words)
        vocab_richness = len(unique_words) / max(1, word_count)
        vocab_score = min(0.5, vocab_richness * 0.8)
        score += vocab_score
        
        # === 11. EMPATHY + SPECIFICITY COMBO BONUS (0-0.5 points) ===
        # Responses that combine empathy with specific advice score higher
        empathy_markers = ['understand', 'sorry', 'hear', 'feel', 'feelings', 
                          'completely', 'absolutely', 'genuinely', 'sincerely',
                          'acknowledge', 'recognize', 'appreciate', 'valid']
        empathy_count = sum(1 for m in empathy_markers if m in response_lower)
        
        if empathy_count >= 2 and specificity_count >= 1:
            score += 0.5
        elif empathy_count >= 1 and structure_count >= 2:
            score += 0.3
        
        # Clamp final score to [0, 10]
        final_score = max(0.0, min(10.0, score))
        
        # Scale to make it more discriminative (spread out the range)
        # Apply a slight sigmoid-like transformation to push scores apart
        midpoint = 5.0
        if final_score > midpoint:
            final_score = midpoint + (final_score - midpoint) * 1.2
        else:
            final_score = midpoint - (midpoint - final_score) * 1.2
        
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a neutral score
        return 3.0