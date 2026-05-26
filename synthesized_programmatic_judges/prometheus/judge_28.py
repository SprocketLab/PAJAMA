def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant focuses on:
    - Discourse structure and coherence markers
    - Empathy/acknowledgment patterns (appropriate for advisory queries)
    - Specificity vs vagueness ratio
    - Dismissive language detection
    - Logical connective density
    - Actionable content detection
    - Response completeness relative to query complexity
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 1.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        response_words = response_lower.split()
        query_words = query_lower.split()
        
        if len(response_words) < 3:
            return 1.0
        
        score = 5.0  # Start at midpoint
        
        # === 1. DISCOURSE COHERENCE MARKERS ===
        # Logical connectives that indicate structured reasoning
        logical_connectives = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'although', 'whereas', 'meanwhile',
            'in addition', 'as a result', 'on the other hand', 'in contrast',
            'for instance', 'for example', 'specifically', 'in particular',
            'thus', 'hence', 'accordingly', 'similarly', 'likewise',
            'in other words', 'that is to say', 'to illustrate'
        ]
        connective_count = sum(1 for c in logical_connectives if c in response_lower)
        connective_score = min(connective_count * 0.25, 1.5)
        score += connective_score
        
        # === 2. ACKNOWLEDGMENT & EMPATHY PATTERNS ===
        # Good responses often acknowledge the user's situation
        acknowledgment_phrases = [
            "i understand", "i can see", "it's understandable", "it's completely",
            "that's understandable", "i hear", "it sounds like", "i'm sorry",
            "it's natural", "it's okay", "it's perfectly", "absolutely okay",
            "completely understandable", "genuinely sorry", "i can hear",
            "we value", "we appreciate", "thank you for",
            "it seems like", "you're feeling", "your experience"
        ]
        ack_count = sum(1 for p in acknowledgment_phrases if p in response_lower)
        # Only reward if query seems to warrant empathy
        emotional_query_words = ['feeling', 'frustrated', 'struggling', 'stress', 
                                  'sad', 'lonely', 'heartbroken', 'difficult',
                                  'concern', 'worry', 'fear', 'problem', 'issue',
                                  'help', 'comfort', 'devastat', 'exhaust']
        query_is_emotional = sum(1 for w in emotional_query_words if w in query_lower) >= 1
        if query_is_emotional:
            score += min(ack_count * 0.35, 1.5)
        else:
            score += min(ack_count * 0.15, 0.5)
        
        # === 3. DISMISSIVE LANGUAGE DETECTION (penalize) ===
        dismissive_patterns = [
            r'\bjust\b.*\bget over\b', r'\bjust\b.*\bmove on\b',
            r'\bget yourself together\b', r'\bstop\b.*\bcomplaining\b',
            r'\bit\'s just\b', r'\bmaybe you\'re just not\b',
            r'\byou should be able to\b', r'\bthat\'s a bummer\b',
            r'\bjust\b.*\bbuy a new\b', r'\bget rid of\b.*\bnegative\b',
            r'\byou need to get\b', r'\bremember that all\b.*\bhave a\b',
            r'\bjust keep\b.*\btrying\b'
        ]
        dismissive_count = sum(1 for p in dismissive_patterns if re.search(p, response_lower))
        score -= dismissive_count * 0.6
        
        # Blunt imperative commands without softening
        blunt_imperatives = [
            r'^(just|stop|don\'t|quit|get)\b',
        ]
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        blunt_count = 0
        for sent in sentences:
            sent_lower = sent.lower().strip()
            for pat in blunt_imperatives:
                if re.match(pat, sent_lower):
                    blunt_count += 1
        score -= blunt_count * 0.15
        
        # === 4. SPECIFICITY INDICATORS ===
        # Specific details: numbers, proper nouns, technical terms
        number_matches = re.findall(r'\b\d+\.?\d*\b', response)
        number_score = min(len(number_matches) * 0.1, 0.5)
        score += number_score
        
        # Proper nouns / capitalized words (not sentence starters)
        words_in_response = response.split()
        proper_noun_count = 0
        for i, w in enumerate(words_in_response):
            if i > 0 and len(w) > 1 and w[0].isupper() and w[1:].islower():
                # Check it's not after a sentence boundary
                prev_char = response[response.index(w) - 2] if response.index(w) >= 2 else ''
                if prev_char not in '.!?\n':
                    proper_noun_count += 1
        score += min(proper_noun_count * 0.05, 0.3)
        
        # === 5. STRUCTURED CONTENT DETECTION ===
        # Numbered lists or clear structure
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        has_structure = len(numbered_items) >= 2
        if has_structure:
            score += 0.5
        
        # Paragraph breaks indicate organized thought
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.3
        
        # === 6. APPROPRIATE HEDGING vs OVERCONFIDENCE ===
        hedging_phrases = [
            'might', 'could', 'perhaps', 'possibly', 'it seems',
            'may be', 'it appears', 'likely', 'suggest', 'consider',
            'it\'s possible', 'one approach', 'you might want',
            'can help', 'could help', 'may help'
        ]
        hedging_count = sum(1 for h in hedging_phrases if h in response_lower)
        
        # Absolute/overconfident claims
        absolute_phrases = [
            'always', 'never', 'definitely', 'certainly', 'undoubtedly',
            'without a doubt', 'guaranteed', 'impossible', 'everyone knows',
            'obviously', 'clearly the only'
        ]
        absolute_count = sum(1 for a in absolute_phrases if a in response_lower)
        
        # Moderate hedging is good, too much is wishy-washy
        hedging_score = min(hedging_count * 0.15, 0.6) - absolute_count * 0.2
        score += hedging_score
        
        # === 7. RESPONSE ENGAGEMENT WITH QUERY ===
        # Measure how well the response addresses the query's key concepts
        # Extract meaningful query words (not stopwords)
        stopwords = {
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
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'about', 'up'
        }
        
        query_content_words = set()
        for w in query_words:
            cleaned = re.sub(r'[^a-z]', '', w)
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                query_content_words.add(cleaned)
        
        response_content_words = set()
        for w in response_words:
            cleaned = re.sub(r'[^a-z]', '', w)
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                response_content_words.add(cleaned)
        
        if query_content_words:
            # Direct overlap
            overlap = len(query_content_words & response_content_words)
            overlap_ratio = overlap / len(query_content_words)
            score += overlap_ratio * 1.0
            
            # Semantic expansion: response introduces related new concepts
            new_concepts = response_content_words - query_content_words
            expansion_ratio = len(new_concepts) / max(len(response_content_words), 1)
            # Some expansion is good (bringing new info), but not too much (going off topic)
            if 0.3 <= expansion_ratio <= 0.8:
                score += 0.3
        
        # === 8. SENTENCE QUALITY ANALYSIS ===
        # Average sentence length (too short = shallow, too long = rambling)
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Ideal range: 10-25 words per sentence
            if 10 <= avg_sent_len <= 25:
                score += 0.4
            elif 7 <= avg_sent_len < 10 or 25 < avg_sent_len <= 35:
                score += 0.1
            else:
                score -= 0.2
            
            # Sentence length variance (good writing has varied sentence lengths)
            if len(sent_lengths) > 1:
                mean_len = sum(sent_lengths) / len(sent_lengths)
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Moderate variance is good
                if 3 <= std_dev <= 12:
                    score += 0.3
        
        # === 9. RESPONSE LENGTH APPROPRIATENESS ===
        resp_len = len(response_words)
        query_len = len(query_words)
        
        # Longer queries generally deserve longer responses
        if query_len > 20:
            if resp_len >= 60:
                score += 0.4
            elif resp_len >= 40:
                score += 0.2
            elif resp_len < 20:
                score -= 0.5
        
        # Very short responses are usually low quality
        if resp_len < 15:
            score -= 1.0
        
        # === 10. TONE CONSISTENCY ===
        # Check for inappropriate casualness in serious contexts
        casual_markers = ['lol', 'haha', 'lmao', 'omg', 'gonna', 'wanna', 
                         'kinda', 'sorta', 'nah', 'yep', 'nope', 'dude', 'bro']
        formal_markers = ['furthermore', 'consequently', 'therefore', 'regarding',
                         'additionally', 'sincerely', 'acknowledge', 'appreciate']
        
        casual_count = sum(1 for c in casual_markers if c in response_lower)
        formal_count = sum(1 for f in formal_markers if f in response_lower)
        
        # Check if query expects formality
        formal_query_indicators = ['professional', 'formal', 'business', 'meeting',
                                    'customer service', 'ai model', 'system', 'scenario']
        query_expects_formal = sum(1 for f in formal_query_indicators if f in query_lower) >= 1
        
        if query_expects_formal and casual_count > 1:
            score -= casual_count * 0.3
        
        # === 11. ACTIONABLE ADVICE DETECTION ===
        actionable_phrases = [
            'try to', 'you can', 'consider', 'start by', 'begin with',
            'one way', 'first', 'next', 'then', 'finally',
            'step', 'approach', 'strategy', 'technique', 'method',
            'here are', 'here\'s', 'following', 'tips', 'suggestion',
            'recommend', 'advise', 'important to', 'make sure',
            'remember to', 'keep in mind', 'don\'t forget'
        ]
        
        # Only reward actionable content if query asks for help/advice
        help_indicators = ['how', 'help', 'advice', 'guide', 'explain', 'what',
                          'assist', 'manage', 'handle', 'cope', 'improve', 'need']
        query_seeks_help = sum(1 for h in help_indicators if h in query_lower) >= 1
        
        if query_seeks_help:
            actionable_count = sum(1 for a in actionable_phrases if a in response_lower)
            score += min(actionable_count * 0.15, 0.8)
        
        # === 12. QUESTION-ASKING (clarification) ===
        # When query is ambiguous, asking clarifying questions is good
        questions_in_response = re.findall(r'\?', response)
        ambiguity_words = ['ambiguous', 'unclear', 'context', 'vague']
        query_mentions_ambiguity = any(w in query_lower for w in ambiguity_words)
        
        if query_mentions_ambiguity and len(questions_in_response) >= 1:
            score += 0.5
        
        # === 13. NEGATIVE PATTERN: Fabrication red flags ===
        # Overly specific unsourced directions/instructions that seem made up
        fabrication_patterns = [
            r'turn (left|right) at the',
            r'you\'ll see a .* on your (left|right)',
            r'continue (straight|down) until',
            r'take the (first|second|third) exit',
        ]
        fabrication_count = sum(1 for p in fabrication_patterns if re.search(p, response_lower))
        
        # Only penalize if query doesn't actually ask for directions
        direction_query = any(w in query_lower for w in ['direction', 'navigate', 'route', 'map'])
        if fabrication_count >= 2 and not direction_query:
            score -= fabrication_count * 0.5
        
        # === 14. SELF-AWARENESS / LIMITATIONS ===
        # Good AI responses acknowledge limitations when appropriate
        limitation_phrases = [
            'without further', 'more information', 'could you clarify',
            'i\'m not sure', 'without context', 'can you provide',
            'i don\'t have', 'it would help if', 'please specify'
        ]
        limitation_count = sum(1 for l in limitation_phrases if l in response_lower)
        if query_mentions_ambiguity:
            score += min(limitation_count * 0.4, 1.0)
        
        # === 15. NEGATIVE: Contradictory or capability-denying language ===
        # "might not be able to" repeated = low confidence response
        inability_phrases = [
            'might not be able', 'may not be able', 'can\'t', 'cannot',
            'won\'t be able', 'probably won\'t', 'it probably', 'it might not'
        ]
        inability_count = sum(1 for p in inability_phrases if p in response_lower)
        if inability_count >= 3:
            score -= 0.8
        elif inability_count >= 2:
            score -= 0.3
        
        # Clamp score to reasonable range
        score = max(1.0, min(10.0, score))
        
        # Scale to 1-5 range to match expected output
        final_score = 1.0 + (score - 1.0) * (4.0 / 9.0)
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        return 3.0  # Safe middle score on error