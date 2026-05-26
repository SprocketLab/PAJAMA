def judging_function(query, response):
    """
    Evaluates clarity and conciseness of an LLM response.
    
    This variant focuses on:
    - Information density (ratio of content words to total words)
    - Transition/connective word usage (smooth flow)
    - Directness of opening (does it get to the point?)
    - Sentence structure variance (not monotonous)
    - Filler/weasel word penalty
    - Formatting effectiveness (bold, structured markers)
    - Compression ratio (unique information per unit length)
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
        if len(response) < 10:
            return 1.0
        
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. DIRECTNESS OF OPENING ===
        # Good responses get to the point quickly. Penalize overly hedging openings.
        first_50_words = ' '.join(words[:50])
        
        # Filler openings that add no value
        weak_openers = [
            r'^(well|so|okay|ok|um|uh|hmm)',
            r'^(that\'?s? (a |an )?(great|good|excellent|wonderful|fantastic|awesome|interesting))',
            r'^(what a (great|good|excellent))',
            r'^(i\'?m glad you asked)',
            r'^(thank you for)',
            r'^(absolutely)',
        ]
        opening_text = ' '.join(words[:15])
        opener_penalty = 0
        for pattern in weak_openers:
            if re.search(pattern, opening_text):
                opener_penalty += 2.0
        score -= min(opener_penalty, 6.0)
        
        # Reward direct engagement - starts with answer-relevant content
        # Check if first sentence contains a key query term
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                     'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'how', 'what',
                     'who', 'where', 'when', 'why', 'which', 'this', 'that', 'with', 'from',
                     'they', 'will', 'would', 'there', 'their', 'about', 'into', 'some', 'could',
                     'them', 'than', 'its', 'over', 'also', 'back', 'after', 'should', 'does',
                     'need', 'any', 'just', 'each', 'much', 'like', 'being', 'other', 'very',
                     'here', 'many', 'these', 'your', 'more'}
        query_content = query_words - stopwords
        if sentences and query_content:
            first_sent_words = set(re.findall(r'[a-zA-Z]{3,}', sentences[0].lower()))
            overlap = len(first_sent_words & query_content)
            if overlap >= 2:
                score += 3.0
            elif overlap >= 1:
                score += 1.5
        
        # === 2. FILLER AND WEASEL WORDS ===
        filler_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'it is worth noting', 'as you may know', 'as we all know',
            'needless to say', 'it goes without saying', 'at the end of the day',
            'in terms of', 'when it comes to', 'the fact that', 'due to the fact',
            'in order to', 'for the purpose of', 'with regard to', 'with respect to',
            'in the event that', 'on the other hand', 'having said that',
            'it is essential to', 'it is crucial to', 'it is vital to',
            'basically', 'essentially', 'actually', 'literally', 'honestly',
            'really', 'very', 'quite', 'rather', 'somewhat', 'pretty much',
            'kind of', 'sort of', 'more or less',
            'in my opinion', 'i think that', 'i believe that', 'i feel that',
            'as a matter of fact', 'to be honest', 'to tell you the truth',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for phrase in filler_phrases:
            filler_count += len(re.findall(re.escape(phrase), response_lower))
        
        filler_density = filler_count / max(total_words / 100, 1)
        score -= min(filler_density * 3.0, 12.0)
        
        # === 3. INFORMATION DENSITY ===
        # Content words vs function words ratio
        function_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                         'should', 'may', 'might', 'must', 'can', 'could', 'of', 'in', 'to',
                         'for', 'with', 'on', 'at', 'from', 'by', 'as', 'into', 'through',
                         'during', 'before', 'after', 'above', 'below', 'between', 'under',
                         'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                         'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
                         'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
                         'very', 'just', 'also', 'it', 'its', 'that', 'this', 'these', 'those',
                         'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
                         'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
                         'how', 'where', 'when', 'why', 'if', 'then', 'there', 'here', 'about'}
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        content_ratio = len(content_words) / max(total_words, 1)
        
        # Ideal content ratio is around 0.45-0.55
        if 0.40 <= content_ratio <= 0.60:
            score += 5.0
        elif 0.35 <= content_ratio <= 0.65:
            score += 2.5
        else:
            score -= 2.0
        
        # === 4. SENTENCE STRUCTURE VARIANCE ===
        # Good writing has varied sentence lengths - not monotonous
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 3:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(mean_len, 1)  # coefficient of variation
            
            # Some variance is good (cv 0.3-0.7), too little or too much is bad
            if 0.25 <= cv <= 0.75:
                score += 4.0
            elif 0.15 <= cv <= 0.90:
                score += 2.0
            else:
                score -= 2.0
            
            # Penalize very long average sentences (>30 words)
            if mean_len > 30:
                score -= 3.0
            elif mean_len > 25:
                score -= 1.5
            # Reward moderate sentence length (12-22 words)
            elif 12 <= mean_len <= 22:
                score += 2.0
        
        # === 5. REPETITION DETECTION (phrase-level) ===
        # Check for repeated 4-grams (same phrase used multiple times)
        if total_words >= 20:
            four_grams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
            four_gram_counts = Counter(four_grams)
            repeated_4grams = sum(1 for c in four_gram_counts.values() if c > 1)
            repetition_ratio = repeated_4grams / max(len(four_grams), 1)
            score -= min(repetition_ratio * 80, 10.0)
        
        # Also check for repeated trigrams
        if total_words >= 15:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            tri_counts = Counter(trigrams)
            # Only penalize non-trivial repeated trigrams
            trivial_trigrams = {'one of the', 'as well as', 'in order to', 'a lot of',
                               'you can also', 'this is a', 'there are many', 'it is a'}
            repeated_tris = sum(1 for gram, c in tri_counts.items() if c > 2 and gram not in trivial_trigrams)
            score -= min(repeated_tris * 1.5, 6.0)
        
        # === 6. FORMATTING EFFECTIVENESS ===
        # Bold markers (** **) indicate structured, scannable content
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        
        # Numbered lists
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        
        # Markdown headers
        header_count = len(re.findall(r'(?:^|\n)#{1,4}\s+\S', response))
        
        # Formatting bonus - structured content is clearer
        formatting_score = 0
        if bold_count >= 2:
            formatting_score += 3.0
        if numbered_items >= 2:
            formatting_score += 2.5
        if header_count >= 1:
            formatting_score += 2.0
        
        # But don't over-reward - cap it
        score += min(formatting_score, 6.0)
        
        # === 7. TRANSITION WORDS (flow quality) ===
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'nevertheless', 'alternatively',
            'specifically', 'particularly', 'notably', 'similarly',
            'conversely', 'instead', 'accordingly', 'thus', 'hence',
            'first', 'second', 'third', 'finally', 'next', 'then',
            'in contrast', 'for example', 'for instance', 'in addition',
            'as a result', 'on the contrary', 'in particular'
        ]
        
        transition_count = 0
        for tw in transition_words:
            transition_count += len(re.findall(r'\b' + re.escape(tw) + r'\b', response_lower))
        
        transition_density = transition_count / max(num_sentences, 1)
        # Ideal: about 0.2-0.5 transitions per sentence
        if 0.15 <= transition_density <= 0.6:
            score += 3.0
        elif transition_density > 0.6:
            score += 1.0  # Still okay, just a bit heavy
        
        # === 8. CONCISENESS: Unique info per word ===
        # Measure how many unique content words per total words
        unique_content = set(content_words)
        uniqueness_ratio = len(unique_content) / max(len(content_words), 1)
        
        # Higher uniqueness = less repetitive
        if uniqueness_ratio >= 0.65:
            score += 4.0
        elif uniqueness_ratio >= 0.50:
            score += 2.0
        elif uniqueness_ratio < 0.35:
            score -= 3.0
        
        # === 9. RESPONSE LENGTH APPROPRIATENESS ===
        # Very short responses for complex queries are bad, but bloated responses are also bad
        query_complexity = len(re.findall(r'[a-zA-Z]+', query))
        
        # For simple queries, shorter is better; for complex, moderate length is ideal
        if query_complexity <= 10:
            # Simple query - reward conciseness
            if total_words <= 150:
                score += 3.0
            elif total_words <= 250:
                score += 1.0
            elif total_words > 400:
                score -= 2.0
        else:
            # Complex query - moderate length is good
            if 80 <= total_words <= 350:
                score += 2.0
            elif total_words > 500:
                score -= 1.5
        
        # === 10. CLARITY SIGNALS ===
        # Parenthetical asides can reduce clarity
        paren_count = len(re.findall(r'\([^)]{20,}\)', response))
        score -= min(paren_count * 1.5, 4.0)
        
        # Very long words (jargon) can reduce clarity unless technical
        long_words = [w for w in words if len(w) > 12]
        long_word_ratio = len(long_words) / max(total_words, 1)
        if long_word_ratio > 0.05:
            score -= 2.0
        
        # Passive voice indicators (rough heuristic)
        passive_patterns = len(re.findall(r'\b(is|are|was|were|been|being)\s+\w+ed\b', response_lower))
        passive_density = passive_patterns / max(num_sentences, 1)
        if passive_density > 0.4:
            score -= 2.0
        
        # === 11. EXCLAMATION AND ENTHUSIASM PENALTY ===
        exclamation_count = response.count('!')
        if exclamation_count > 3:
            score -= min((exclamation_count - 3) * 0.5, 3.0)
        
        # === 12. COLON USAGE (often indicates clear definitions/explanations) ===
        colon_count = len(re.findall(r':\s', response))
        if 1 <= colon_count <= 8:
            score += 1.5
        
        # Clamp score to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0