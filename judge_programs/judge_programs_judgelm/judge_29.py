def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level structure analysis (well-formed sentences as credibility proxy)
    2. Information density scoring (ratio of informative tokens to filler)
    3. Red-flag pattern detection using regex patterns for hallucination indicators
    4. Appropriate uncertainty calibration detection
    5. Response coherence relative to query topic
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 2:
            return 0.0
        
        # === 1. Sentence-level structure analysis ===
        # Split into sentences and analyze each
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        num_sentences = max(len(sentences), 1)
        
        # Score sentences for well-formedness
        well_formed_count = 0
        for sent in sentences:
            words = sent.split()
            if len(words) >= 3:
                # Check starts with capital letter
                if sent[0].isupper():
                    well_formed_count += 1
                elif len(words) >= 5:
                    well_formed_count += 0.5
        
        sentence_quality = well_formed_count / num_sentences if num_sentences > 0 else 0
        
        # === 2. Information density scoring ===
        # Identify informative tokens: proper nouns, numbers, specific terms
        words = response.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Detect specific/informative tokens
        proper_noun_pattern = re.compile(r'\b[A-Z][a-z]{2,}\b')
        number_pattern = re.compile(r'\b\d[\d,.:/-]*\d?\b')
        
        proper_nouns = proper_noun_pattern.findall(response)
        numbers = number_pattern.findall(response)
        
        # Filter out sentence-starting capitals by checking position
        # Count named entities (sequences of capitalized words not at sentence start)
        named_entity_pattern = re.compile(r'(?<=[a-z.]\s)[A-Z][a-z]+(?:\s[A-Z][a-z]+)*')
        named_entities = named_entity_pattern.findall(response)
        
        # Information density: ratio of specific/informative content
        informative_tokens = len(numbers) * 2 + len(named_entities) * 1.5
        info_density = min(informative_tokens / max(total_words, 1) * 10, 3.0)
        
        # === 3. Red-flag pattern detection ===
        red_flag_score = 0.0
        
        # Overly absolute claims
        absolute_patterns = [
            r'\b(always|never|every single|100%|guaranteed|proven fact|undeniable)\b',
            r'\b(everyone knows|obviously true|no one can deny)\b',
            r'\b(exposed|cover.?up|they don\'t want you|mainstream media lies)\b',
            r'\b(shocking truth|secret|conspiracy|wake up|sheeple)\b',
        ]
        
        response_lower = response.lower()
        
        for pattern in absolute_patterns:
            matches = re.findall(pattern, response_lower)
            red_flag_score += len(matches) * 0.5
        
        # Sensationalism markers
        exclamation_count = response.count('!')
        all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', response))
        red_flag_score += min(exclamation_count * 0.3, 2.0)
        red_flag_score += min(all_caps_words * 0.4, 2.0)
        
        # Repetition as a red flag (copy-paste or looping)
        if total_words > 20:
            # Check for repeated phrases (trigrams)
            trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                most_common_freq = trigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(trigrams)
                if repetition_ratio > 0.15:
                    red_flag_score += 2.0
                elif repetition_ratio > 0.08:
                    red_flag_score += 1.0
        
        # === 4. Appropriate uncertainty calibration ===
        calibration_score = 0.0
        
        # Hedging phrases (good when appropriate)
        hedging_phrases = [
            r'\b(it is difficult to|it\'s difficult to)\b',
            r'\b(may|might|could|possibly|perhaps|likely|unlikely)\b',
            r'\b(according to|based on|research suggests|studies indicate)\b',
            r'\b(generally|typically|often|usually|in most cases)\b',
            r'\b(approximately|around|roughly|estimated|about)\b',
            r'\b(it depends|varies|subjective|interpretation)\b',
            r'\b(however|although|on the other hand|that said|nonetheless)\b',
        ]
        
        hedging_count = 0
        for pattern in hedging_phrases:
            hedging_count += len(re.findall(pattern, response_lower))
        
        # Hedging is good but not too much
        if hedging_count > 0:
            calibration_score = min(hedging_count * 0.4, 2.0)
        
        # Citation-like patterns
        citation_patterns = [
            r'\b(according to|cited by|published in|reported by)\b',
            r'\b(in \d{4})\b',
            r'\b(university|institute|journal|study|research)\b',
            r'\(\d{4}\)',  # (2023) style citations
            r'\b(professor|dr\.|ph\.d)\b',
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, response_lower))
        
        citation_score = min(citation_count * 0.3, 1.5)
        
        # === 5. Query-response topical coherence ===
        # Extract content words from query and check presence in response
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'between',
            'through', 'after', 'before', 'during', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'where',
            'when', 'why', 'how', 'if', 'then', 'than', 'more', 'most', 'very',
            'just', 'also', 'there', 'here', 'all', 'any', 'some', 'no', 'many',
            'much', 'few', 'little', 'other', 'another', 'such', 'only', 'own',
            'same', 'tell', 'please', 'know', 'want', 'make', 'like',
        }
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower())) - stop_words
        response_words_set = set(re.findall(r'\b[a-z]{3,}\b', response_lower))
        
        if query_words:
            overlap = len(query_words & response_words_set) / len(query_words)
            topical_coherence = overlap * 2.0
        else:
            topical_coherence = 1.0
        
        topical_coherence = min(topical_coherence, 2.0)
        
        # === 6. Response substantiveness ===
        # Penalize very short responses, reward moderate length
        length_score = 0.0
        if total_words < 3:
            length_score = -2.0
        elif total_words < 8:
            length_score = -0.5
        elif total_words < 15:
            length_score = 0.5
        elif total_words <= 200:
            length_score = 1.5
        elif total_words <= 400:
            length_score = 1.0
        else:
            length_score = 0.5  # Very long might indicate rambling
        
        # === 7. Garbage/noise detection ===
        noise_penalty = 0.0
        
        # HTML/code in non-code responses
        html_tags = len(re.findall(r'<[a-z/][^>]*>', response_lower))
        if html_tags > 3:
            # Check if query asks for HTML
            if not re.search(r'\b(html|tag|code|web|page)\b', query.lower()):
                noise_penalty += 1.5
        
        # Random code blocks when not asked for
        code_indicators = len(re.findall(r'\b(import |def |class |print\(|function |var |const )\b', response))
        if code_indicators > 2 and not re.search(r'\b(code|program|script|function|implement)\b', query.lower()):
            noise_penalty += 1.5
        
        # Repeated "Output:" or "Input:" patterns (template artifacts)
        template_artifacts = len(re.findall(r'(Output:|Input:|Question:|Answer:)', response))
        if template_artifacts > 3:
            noise_penalty += 1.0
        
        # Check for response being mostly non-alphabetic
        alpha_chars = sum(1 for c in response if c.isalpha())
        alpha_ratio = alpha_chars / max(len(response), 1)
        if alpha_ratio < 0.4:
            noise_penalty += 1.5
        
        # === 8. Coherent explanation detection ===
        # Presence of explanatory connectors indicates structured reasoning
        explanation_markers = [
            r'\b(because|therefore|thus|hence|as a result)\b',
            r'\b(for example|for instance|such as|including)\b',
            r'\b(first|second|third|finally|additionally|moreover)\b',
            r'\b(this means|in other words|specifically)\b',
        ]
        
        explanation_count = 0
        for pattern in explanation_markers:
            explanation_count += len(re.findall(pattern, response_lower))
        
        explanation_score = min(explanation_count * 0.3, 1.5)
        
        # === Combine all scores ===
        base_score = 4.0
        
        final_score = (
            base_score
            + sentence_quality * 1.5      # 0 to 1.5
            + info_density                  # 0 to 3.0
            - red_flag_score                # 0 to ~6
            + calibration_score             # 0 to 2.0
            + citation_score                # 0 to 1.5
            + topical_coherence             # 0 to 2.0
            + length_score                  # -2 to 1.5
            - noise_penalty                 # 0 to ~5
            + explanation_score             # 0 to 1.5
        )
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middle score based on response length
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except Exception:
            return 3.0