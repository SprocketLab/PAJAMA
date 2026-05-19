def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    Uses a token-level and pattern-based approach focusing on:
    - Hedging and epistemic markers (appropriate uncertainty)
    - Specificity indicators (names, dates, numbers used appropriately)
    - Hallucination red flags (unsourced absolute claims, sensationalism)
    - Discourse quality markers (structured reasoning, acknowledgment patterns)
    - Empathy and engagement signals
    
    Returns a score where higher = better quality.
    """
    try:
        if not query or not response:
            return 1.0
        
        if not isinstance(query, str) or not isinstance(response, str):
            return 1.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        
        if len(words) < 3:
            return 1.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # === 1. APPROPRIATE HEDGING & EPISTEMIC MARKERS ===
        # Good responses show calibrated confidence
        hedging_phrases = [
            r'\bit\'?s\s+(completely\s+)?(understandable|natural|normal|okay|fine)\b',
            r'\b(it\s+seems|it\s+appears|it\s+looks\s+like)\b',
            r'\b(can\s+help|could\s+help|might\s+help|may\s+help)\b',
            r'\b(consider|you\s+might|you\s+could|you\s+may)\b',
            r'\b(often|usually|typically|generally|commonly)\b',
            r'\b(one\s+way|one\s+approach|an?\s+option)\b',
            r'\b(remember\s+that|keep\s+in\s+mind|it\'?s\s+worth\s+noting)\b',
            r'\b(perfectly\s+(fine|okay|normal|natural|understandable))\b',
        ]
        
        hedging_count = 0
        for pattern in hedging_phrases:
            hedging_count += len(re.findall(pattern, response_lower))
        
        # Reward moderate hedging (not too much, not too little)
        hedging_ratio = hedging_count / num_sentences
        if 0.1 <= hedging_ratio <= 0.8:
            score += hedging_count * 2.5
        elif hedging_ratio > 0.8:
            score += 5  # Too much hedging is less good but still okay
        
        # === 2. STRUCTURED REASONING INDICATORS ===
        # Numbered lists, step-by-step reasoning
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+', response)
        if 2 <= len(numbered_items) <= 15:
            score += len(numbered_items) * 2.0
        
        # Colon-separated explanations (e.g., "Concept: explanation")
        colon_explanations = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,3}:\s+[A-Z]', response)
        score += min(len(colon_explanations) * 1.5, 8)
        
        # === 3. EMPATHY & ENGAGEMENT SIGNALS ===
        empathy_patterns = [
            r'\bi\'?m\s+(sorry|glad|happy)\s+to\s+hear\b',
            r'\bi\s+can\s+(see|hear|understand|imagine)\b',
            r'\bthat\'?s\s+(completely|totally|absolutely|perfectly)\s+(understandable|normal|okay|fine|natural)\b',
            r'\b(i\s+understand|we\s+understand)\b',
            r'\b(your\s+(feelings?|concerns?|frustration|experience|situation))\b',
            r'\b(we\s+(value|appreciate|care))\b',
            r'\bsincerely\s+apologize\b',
            r'\blet\'?s\s+(take|work|figure|try)\b',
        ]
        
        empathy_score = 0
        for pattern in empathy_patterns:
            matches = re.findall(pattern, response_lower)
            empathy_score += len(matches)
        
        score += min(empathy_score * 3.0, 15)
        
        # === 4. ACTIONABLE ADVICE INDICATORS ===
        action_patterns = [
            r'\b(here\s+are\s+some|here\'?s\s+how|try\s+to|you\s+can|start\s+by|begin\s+with)\b',
            r'\b(first|second|third|next|then|finally|lastly)\b',
            r'\b(step\s+\d|phase\s+\d)\b',
            r'\b(for\s+example|for\s+instance|such\s+as|like\s+when)\b',
            r'\b(this\s+(means|helps|ensures|allows|enables))\b',
            r'\b(in\s+order\s+to|so\s+that|this\s+way)\b',
        ]
        
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, response_lower))
        
        score += min(action_count * 1.8, 12)
        
        # === 5. HALLUCINATION RED FLAGS (PENALIZE) ===
        # Overly absolute claims without hedging
        absolute_patterns = [
            r'\b(always|never|definitely|certainly|absolutely|guaranteed)\b',
            r'\b(everyone\s+knows|it\'?s\s+obvious|clearly)\b',
            r'\b(the\s+only\s+way|the\s+best\s+way|the\s+right\s+way)\b',
            r'\b(you\s+must|you\s+need\s+to|you\s+should\s+just)\b',
        ]
        
        absolute_count = 0
        for pattern in absolute_patterns:
            absolute_count += len(re.findall(pattern, response_lower))
        
        # Penalize excessive absolutism
        if absolute_count > 2:
            score -= (absolute_count - 2) * 2.0
        
        # === 6. DISMISSIVE LANGUAGE (PENALIZE) ===
        dismissive_patterns = [
            r'\bjust\s+(get\s+over|move\s+on|deal\s+with|stop)\b',
            r'\b(you\s+should\s+be\s+able\s+to)\b',
            r'\b(it\'?s\s+not\s+that\s+(big|hard|bad|difficult))\b',
            r'\b(maybe\s+you\'?re\s+(just|not))\b',
            r'\b(not\s+using\s+it\s+correctly)\b',
            r'\b(read\s+the\s+manual)\b',
            r'\b(that\'?s\s+a\s+bummer)\b',
            r'\b(get\s+yourself\s+together)\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        score -= dismissive_count * 5.0
        
        # === 7. RESPONSE DEPTH & COMPLETENESS ===
        # Longer, more detailed responses tend to be better (up to a point)
        word_count = len(words)
        if word_count < 20:
            score -= 10
        elif 20 <= word_count < 50:
            score += 2
        elif 50 <= word_count <= 200:
            score += 8
        elif word_count > 200:
            score += 6
        
        # === 8. VOCABULARY RICHNESS ===
        unique_words = set(words)
        if len(words) > 10:
            vocab_richness = len(unique_words) / len(words)
            if vocab_richness > 0.55:
                score += 5
            elif vocab_richness < 0.35:
                score -= 3
        
        # === 9. QUERY RELEVANCE - Semantic overlap ===
        # Check if response addresses key terms from the query
        query_words = set(query_lower.split())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'only', 'own',
                      'same', 'than', 'too', 'very', 'just', 'because', 'if',
                      'when', 'where', 'how', 'what', 'which', 'who', 'whom',
                      'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
                      'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                      'it', 'its', 'they', 'them', 'their', 'up', 'about'}
        
        query_content_words = query_words - stop_words
        response_words_set = set(words)
        
        if query_content_words:
            overlap = len(query_content_words & response_words_set) / len(query_content_words)
            score += overlap * 8
        
        # === 10. TONE CONSISTENCY & PROFESSIONALISM ===
        # Check for appropriate greeting/acknowledgment at start
        opening_patterns = [
            r'^(i\'?m\s+(sorry|glad)|i\s+can\s+(see|hear|understand)|it\'?s\s+(completely|totally)|imagine|hey\s+there|let\'?s)',
            r'^(i\s+understand|we\s+understand|thank\s+you|that\'?s)',
        ]
        
        response_start = response_lower[:100]
        for pattern in opening_patterns:
            if re.search(pattern, response_start):
                score += 3
                break
        
        # === 11. INAPPROPRIATE CONFIDENCE IN WRONG CONTEXT ===
        # If query is ambiguous/unclear, response should ask for clarification
        ambiguity_indicators = ['ambiguous', 'unclear', 'no previous context', 'no context', 'vague']
        query_is_about_ambiguity = any(ind in query_lower for ind in ambiguity_indicators)
        
        if query_is_about_ambiguity:
            # Reward asking for clarification
            clarification_patterns = [
                r'\b(can\s+you\s+(provide|give|share|tell)\s+(more|further))\b',
                r'\b(without\s+(further|more)\s+(details?|information|context))\b',
                r'\b(could\s+you\s+(clarify|specify|elaborate))\b',
                r'\b(what\s+(place|destination|location))\b',
            ]
            asks_clarification = False
            for pattern in clarification_patterns:
                if re.search(pattern, response_lower):
                    asks_clarification = True
                    break
            
            if asks_clarification:
                score += 10
            else:
                # Penalize making up directions/info without context
                score -= 8
        
        # === 12. SENSATIONALISM DETECTION (PENALIZE) ===
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'insane', 'crazy',
            'conspiracy', 'they don\'t want you to know', 'secret',
            'wake up', 'sheeple', 'mainstream media'
        ]
        
        sensational_count = sum(1 for w in sensational_words if w in response_lower)
        score -= sensational_count * 4.0
        
        # === 13. CONDITIONAL/NUANCED THINKING ===
        conditional_patterns = [
            r'\b(if\s+you|when\s+you|in\s+case|depending\s+on)\b',
            r'\b(however|on\s+the\s+other\s+hand|that\s+said|although)\b',
            r'\b(while\s+it|even\s+though|despite)\b',
        ]
        
        conditional_count = 0
        for pattern in conditional_patterns:
            conditional_count += len(re.findall(pattern, response_lower))
        
        score += min(conditional_count * 2.0, 8)
        
        # === 14. PARAGRAPH STRUCTURE ===
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += min(len(paragraphs) * 1.0, 5)
        
        # === 15. PERSONAL PRONOUN BALANCE ===
        # Good responses balance "you/your" (addressing user) with explanatory content
        you_count = len(re.findall(r'\byou(?:r|\'re|\'ll|\'ve)?\b', response_lower))
        i_we_count = len(re.findall(r'\b(i|we|our|my)\b', response_lower))
        
        # Addressing the user is generally good
        if you_count >= 2:
            score += min(you_count * 0.8, 5)
        
        # Too much "I/we" without "you" can be self-centered
        if i_we_count > you_count * 3 and i_we_count > 5:
            score -= 3
        
        # === 16. SPECIFIC TECHNICAL TERMS USED APPROPRIATELY ===
        # If query mentions technical topics, check if response uses relevant terms
        technical_query_terms = {
            'quantum': ['qubit', 'superposition', 'entanglement', 'quantum'],
            'ai': ['model', 'algorithm', 'training', 'neural', 'machine learning'],
            'cooking': ['heat', 'cook', 'ingredients', 'temperature', 'minutes'],
            'meeting': ['agenda', 'minutes', 'interruption', 'conversation', 'track'],
        }
        
        for topic, terms in technical_query_terms.items():
            if topic in query_lower:
                term_matches = sum(1 for t in terms if t in response_lower)
                score += min(term_matches * 1.5, 6)
                break
        
        # === FINAL NORMALIZATION ===
        # Clamp to 1-5 range to match expected output
        # Map from internal score range (~30-90) to 1-5
        normalized = (score - 30) / 60.0  # Maps 30->0, 90->1
        normalized = max(0, min(1, normalized))
        final_score = 1 + normalized * 4  # Maps to 1-5
        
        return round(final_score, 2)
        
    except Exception as e:
        return 3.0  # Return middle score on any error