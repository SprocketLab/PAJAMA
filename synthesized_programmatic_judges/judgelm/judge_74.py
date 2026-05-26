def judging_function(query, response):
    """
    Evaluates response quality based on evidence density and specificity.
    
    This variant uses a different approach: it analyzes the ratio of "informative tokens"
    to total tokens, detects named entities via capitalization patterns, counts numeric
    references, measures sentence-level information density, and penalizes vague/filler
    language patterns. It also uses character-level entropy as a proxy for information content.
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) < 2:
            return 0.0
        
        # ---- Feature 1: Character-level entropy (information density) ----
        # Higher entropy suggests more diverse, information-rich text
        char_counts = Counter(response_stripped.lower())
        total_chars = len(response_stripped)
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Normalize: typical English text has entropy ~4.0-4.5, max ~log2(27)≈4.75
        entropy_score = min(char_entropy / 4.5, 1.0)
        
        # ---- Feature 2: Named entity density via capitalization patterns ----
        # Look for capitalized words that aren't at sentence starts
        sentences = re.split(r'[.!?]+', response_stripped)
        named_entity_count = 0
        words = response_stripped.split()
        total_words = max(len(words), 1)
        
        # Detect capitalized words not at sentence beginnings
        for sent in sentences:
            sent = sent.strip()
            sent_words = sent.split()
            if len(sent_words) > 1:
                for w in sent_words[1:]:
                    clean_w = w.strip(string.punctuation)
                    if clean_w and len(clean_w) > 1 and clean_w[0].isupper() and not clean_w.isupper():
                        named_entity_count += 1
        
        # Also count fully capitalized acronyms (2-5 chars)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', response_stripped)
        named_entity_count += len(acronyms)
        
        ne_density = min(named_entity_count / max(total_words, 1) * 10, 1.0)
        
        # ---- Feature 3: Numeric/quantitative information density ----
        # Count numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', response_stripped)
        percentages = re.findall(r'\d+\.?\d*\s*%', response_stripped)
        dates = re.findall(r'\b(?:19|20)\d{2}\b', response_stripped)
        measurements = re.findall(r'\b\d+\.?\d*\s*(?:km|mi|lb|kg|ft|m|cm|mm|oz|mg|ml|L|mph|kph|GB|MB|TB)\b', response_stripped, re.IGNORECASE)
        
        numeric_items = len(numbers) + len(percentages) * 2 + len(dates) * 1.5 + len(measurements) * 2
        numeric_density = min(numeric_items / max(total_words, 1) * 15, 1.0)
        
        # ---- Feature 4: Vagueness penalty ----
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are many\b', r'\bthere are various\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bsome say\b',
            r'\bpeople think\b', r'\bpeople believe\b', r'\bsome believe\b',
            r'\bcan vary\b', r'\bmay vary\b', r'\bdepends on\b',
            r'\ba lot of\b', r'\ba number of\b', r'\bseveral\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bbasically\b', r'\bessentially\b', r'\bprobably\b',
            r'\bmaybe\b', r'\bperhaps\b', r'\bmight be\b',
            r'\bcould be\b', r'\btend to\b', r'\boften\b',
            r'\busually\b', r'\bsometimes\b', r'\bquite\b',
            r'\brather\b', r'\bfairly\b', r'\bsomewhat\b',
        ]
        vague_count = 0
        response_lower = response_stripped.lower()
        for pattern in vague_patterns:
            vague_count += len(re.findall(pattern, response_lower))
        
        vagueness_ratio = min(vague_count / max(total_words, 1) * 20, 1.0)
        
        # ---- Feature 5: Specificity via unique content words ----
        # Use a set of common stop/filler words; measure ratio of content words
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
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'like', 'get', 'got', 'make', 'made',
        }
        
        clean_words = [w.strip(string.punctuation).lower() for w in words if w.strip(string.punctuation)]
        content_words = [w for w in clean_words if w and len(w) > 2 and w not in stop_words]
        unique_content = set(content_words)
        
        content_ratio = len(content_words) / max(len(clean_words), 1)
        unique_content_ratio = len(unique_content) / max(len(content_words), 1) if content_words else 0
        
        # Reward lexical diversity among content words
        specificity_score = content_ratio * 0.5 + unique_content_ratio * 0.5
        
        # ---- Feature 6: Sentence completeness and structure ----
        # Count complete sentences (ending with punctuation)
        complete_sentences = len(re.findall(r'[.!?](?:\s|$)', response_stripped))
        sentence_ratio = min(complete_sentences / max(total_words / 15, 1), 1.0)
        
        # ---- Feature 7: Long/specific words ratio ----
        # Words >= 7 chars tend to be more specific/technical
        long_words = [w for w in clean_words if len(w) >= 7]
        long_word_ratio = len(long_words) / max(len(clean_words), 1)
        long_word_score = min(long_word_ratio * 3, 1.0)
        
        # ---- Feature 8: Repetition penalty ----
        # Detect repeated phrases (trigrams)
        if len(clean_words) >= 3:
            trigrams = [' '.join(clean_words[i:i+3]) for i in range(len(clean_words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            repetition_penalty = min(repeated_trigrams / max(len(trigrams), 1) * 10, 1.0)
        else:
            repetition_penalty = 0.0
        
        # ---- Feature 9: Response relevance to query ----
        query_words = set(w.strip(string.punctuation).lower() for w in query.split() 
                         if w.strip(string.punctuation).lower() not in stop_words and len(w.strip(string.punctuation)) > 2)
        response_content_set = unique_content
        
        if query_words:
            overlap = len(query_words & response_content_set)
            relevance = min(overlap / max(len(query_words), 1), 1.0)
        else:
            relevance = 0.5
        
        # ---- Feature 10: Garbage/noise detection ----
        # Detect HTML tags, code artifacts, repeated characters
        html_tags = len(re.findall(r'<[^>]+>', response_stripped))
        code_artifacts = len(re.findall(r'(?:import |def |class |print\(|return )', response_stripped))
        
        # Detect if response is mostly repetitive junk
        if total_chars > 20:
            # Check character diversity relative to length
            char_diversity = len(char_counts) / min(total_chars, 100)
        else:
            char_diversity = 0.5
        
        noise_penalty = min((html_tags * 0.1 + code_artifacts * 0.05), 0.5)
        
        # ---- Feature 11: Proper length (not too short, not excessively padded) ----
        # Sweet spot: responses that are substantive but not padded
        if total_words < 3:
            length_score = 0.1
        elif total_words < 10:
            length_score = 0.3
        elif total_words < 25:
            length_score = 0.6
        elif total_words < 80:
            length_score = 0.9
        elif total_words < 200:
            length_score = 1.0
        else:
            length_score = 0.85  # Slight penalty for very long (might be padded)
        
        # ---- Feature 12: Actionable/concrete language markers ----
        concrete_markers = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b',
            r'\bin particular\b', r'\baccording to\b', r'\bresearch shows\b',
            r'\bstudies show\b', r'\bdata shows\b', r'\bevidence\b',
            r'\bexperiment\b', r'\bfound that\b', r'\bdemonstrated\b',
            r'\bproven\b', r'\bmeasured\b', r'\bcalculated\b',
            r'\bestimated at\b', r'\bapproximately\b', r'\bexactly\b',
            r'\bprecisely\b', r'\bknown as\b', r'\bcalled\b',
            r'\bnamed\b', r'\blocated in\b', r'\bfounded in\b',
        ]
        concrete_count = 0
        for pattern in concrete_markers:
            concrete_count += len(re.findall(pattern, response_lower))
        
        concrete_score = min(concrete_count / max(total_words / 20, 1), 1.0)
        
        # ---- Combine all features with weights ----
        score = (
            entropy_score * 1.2 +          # Information density via entropy
            ne_density * 1.5 +             # Named entities
            numeric_density * 1.8 +        # Numbers and data
            specificity_score * 1.3 +      # Content word ratio and diversity
            sentence_ratio * 0.6 +         # Sentence completeness
            long_word_score * 0.8 +        # Technical/specific vocabulary
            relevance * 1.0 +              # Query relevance
            length_score * 1.2 +           # Appropriate length
            concrete_score * 1.2 +         # Concrete language markers
            char_diversity * 0.4 -          # Character diversity
            vagueness_ratio * 2.0 -        # Vagueness penalty
            repetition_penalty * 1.5 -     # Repetition penalty
            noise_penalty * 1.5            # HTML/code noise penalty
        )
        
        # Normalize to 0-10 scale
        # Theoretical max is around 11, theoretical min is around -5
        # Map to 0-10
        score = max(0.0, score)
        score = (score / 8.0) * 10.0  # Scale so typical good responses get 7-9
        score = min(10.0, max(0.0, score))
        
        return round(score, 2)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            if response and len(response.strip()) > 0:
                return min(len(response.strip().split()) / 20.0 * 5.0, 5.0)
            return 0.0
        except Exception:
            return 0.0