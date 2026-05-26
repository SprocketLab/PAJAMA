def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response to a query.
    Returns a numeric score where HIGHER = BETTER quality.
    
    Strategy: Analyze multiple dimensions of completeness including:
    - Response length and depth relative to query complexity
    - Coverage of query topics/keywords
    - Structural completeness (lists, paragraphs, examples)
    - Presence of actionable content vs vague filler
    - Addressing multiple aspects/sub-questions
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.5
        
        # ============================================================
        # 1. QUERY COMPLEXITY ANALYSIS
        # ============================================================
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Count question marks and sub-questions
        question_marks = query.count('?')
        
        # Detect query complexity signals
        complexity_words = ['how', 'why', 'explain', 'describe', 'what', 'compare',
                          'analyze', 'evaluate', 'discuss', 'elaborate', 'detail',
                          'guide', 'help', 'advice', 'recommend', 'suggest',
                          'understand', 'concept', 'approach', 'strategy', 'method']
        query_complexity = sum(1 for w in complexity_words if w in query_lower)
        
        # Multi-aspect detection
        multi_aspect_signals = ['and', 'also', 'additionally', 'moreover', 'both',
                               'as well as', 'not only', 'but also', 'furthermore']
        multi_aspect_count = sum(1 for s in multi_aspect_signals if s in query_lower)
        
        estimated_query_complexity = max(1, query_complexity + multi_aspect_count + question_marks)
        
        # ============================================================
        # 2. RESPONSE LENGTH & DEPTH SCORE (0-20)
        # ============================================================
        resp_words = response.split()
        resp_word_count = len(resp_words)
        query_words = query.split()
        query_word_count = len(query_words)
        
        # Ideal response length scales with query complexity
        # More complex queries need longer responses
        ideal_min_words = max(50, estimated_query_complexity * 20)
        
        if resp_word_count < 20:
            length_score = 2.0
        elif resp_word_count < ideal_min_words * 0.5:
            length_score = 6.0 + 4.0 * (resp_word_count / (ideal_min_words * 0.5))
        elif resp_word_count < ideal_min_words:
            length_score = 10.0 + 6.0 * ((resp_word_count - ideal_min_words * 0.5) / (ideal_min_words * 0.5))
        else:
            # Diminishing returns for very long responses
            length_score = 16.0 + min(4.0, 4.0 * math.log(1 + (resp_word_count - ideal_min_words) / ideal_min_words))
        
        length_score = min(20.0, length_score)
        
        # ============================================================
        # 3. TOPIC COVERAGE SCORE (0-25)
        # ============================================================
        # Extract meaningful content words from query
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'out', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 'just', 'don',
                     'now', 'up', 'down', 'it', 'its', 'they', 'them', 'their',
                     'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
                     'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
                     'who', 'whom', 'which', 'what', 'but', 'and', 'or', 'if',
                     'while', 'about', 'against', 'because', 'until', 'although',
                     'however', 'also', 'need', 'must', 'way', 'person', 'one',
                     'make', 'get', 'like', 'know', 'take', 'come', 'think'}
        
        # Extract query content words (lowercased, alphabetic, non-stop)
        query_content_words = set()
        for word in re.findall(r'[a-zA-Z]+', query_lower):
            if word not in stop_words and len(word) > 2:
                query_content_words.add(word)
        
        if query_content_words:
            covered = sum(1 for w in query_content_words if w in response_lower)
            coverage_ratio = covered / len(query_content_words)
        else:
            coverage_ratio = 0.5
        
        # Also check for semantic coverage via related terms
        # Extract bigrams from query for phrase matching
        query_words_clean = [w for w in re.findall(r'[a-zA-Z]+', query_lower) if len(w) > 2]
        query_bigrams = set()
        for i in range(len(query_words_clean) - 1):
            bigram = query_words_clean[i] + ' ' + query_words_clean[i+1]
            if query_words_clean[i] not in stop_words or query_words_clean[i+1] not in stop_words:
                query_bigrams.add(bigram)
        
        if query_bigrams:
            bigram_covered = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_coverage = bigram_covered / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        topic_score = 25.0 * (0.6 * coverage_ratio + 0.4 * bigram_coverage)
        
        # ============================================================
        # 4. STRUCTURAL COMPLETENESS (0-20)
        # ============================================================
        struct_score = 0.0
        
        # Check for numbered lists or bullet points (indicates structured, thorough answer)
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response))
        list_items = numbered_items + bullet_items
        
        if list_items >= 3:
            struct_score += 6.0
        elif list_items >= 1:
            struct_score += 3.0
        
        # Check for paragraphs (multiple blocks of text)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 3:
            struct_score += 5.0
        elif len(paragraphs) >= 2:
            struct_score += 3.0
        else:
            struct_score += 1.0
        
        # Check sentence count - more sentences usually means more thorough
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = len(sentences)
        
        if sentence_count >= 8:
            struct_score += 5.0
        elif sentence_count >= 5:
            struct_score += 3.5
        elif sentence_count >= 3:
            struct_score += 2.0
        else:
            struct_score += 0.5
        
        # Check for examples or illustrations
        example_signals = ['for example', 'for instance', 'such as', 'e.g.',
                          'like when', 'imagine', 'consider', 'suppose',
                          'illustration', 'scenario', 'case in point']
        has_examples = any(sig in response_lower for sig in example_signals)
        if has_examples:
            struct_score += 4.0
        
        struct_score = min(20.0, struct_score)
        
        # ============================================================
        # 5. SUBSTANTIVE CONTENT vs FILLER (0-20)
        # ============================================================
        # Penalize vague, dismissive, or superficial responses
        
        # Dismissive/vague phrases
        dismissive_phrases = [
            "just do it", "you should be able", "it's not that hard",
            "just keep", "maybe you should", "you could try",
            "that's a bummer", "get over it", "move on",
            "it's just a", "nothing wrong with", "don't let it",
            "you'll be fine", "don't worry about it",
            "it's not a big deal", "might not be able",
            "probably won't", "might not have"
        ]
        dismissive_count = sum(1 for p in dismissive_phrases if p in response_lower)
        
        # Empathetic/substantive phrases
        substantive_phrases = [
            "i understand", "it's understandable", "completely understandable",
            "here are", "first", "second", "third", "next",
            "importantly", "specifically", "in particular",
            "the reason", "this means", "this is because",
            "let me explain", "to clarify", "in other words",
            "step by step", "one approach", "another approach",
            "keep in mind", "it's important to", "remember that",
            "this will help", "this can help", "you might find",
            "i can see", "i hear", "that's completely",
            "absolutely", "genuinely", "sincerely",
            "let's", "we can", "together"
        ]
        substantive_count = sum(1 for p in substantive_phrases if p in response_lower)
        
        # Hedging/uncertainty (too much = less helpful)
        hedge_phrases = ["maybe", "perhaps", "i guess", "i think maybe",
                        "not sure", "might", "possibly", "probably"]
        hedge_count = sum(1 for h in hedge_phrases if h in response_lower)
        
        # Calculate content quality
        content_score = 10.0  # baseline
        content_score += min(8.0, substantive_count * 1.5)
        content_score -= min(6.0, dismissive_count * 2.0)
        content_score -= min(3.0, max(0, hedge_count - 1) * 1.0)
        
        content_score = max(0.0, min(20.0, content_score))
        
        # ============================================================
        # 6. ADDRESSING THE ACTUAL NEED (0-15)
        # ============================================================
        # Does the response actually try to help with what was asked?
        
        need_score = 7.5  # baseline
        
        # Check if response acknowledges the situation/problem
        acknowledgment_phrases = [
            "i'm sorry", "i understand", "i can see", "it sounds like",
            "i hear you", "that must be", "it's natural to",
            "it's okay to", "it's perfectly", "you're right to",
            "that's a great", "good question", "great question",
            "absolutely", "of course"
        ]
        ack_count = sum(1 for a in acknowledgment_phrases if a in response_lower)
        if ack_count > 0:
            need_score += min(3.0, ack_count * 1.5)
        
        # Check if response provides actionable advice/information
        action_phrases = [
            "you can", "you could", "try to", "start by", "begin with",
            "make sure", "consider", "i recommend", "i suggest",
            "one way", "another way", "the best way", "a good approach",
            "here's how", "to do this", "follow these", "the steps"
        ]
        action_count = sum(1 for a in action_phrases if a in response_lower)
        need_score += min(4.5, action_count * 1.0)
        
        # Penalize if response contradicts the query's need
        # (e.g., being dismissive when comfort is needed)
        emotional_query = any(w in query_lower for w in ['feeling', 'frustrated', 'sad',
                                                          'stress', 'worried', 'anxious',
                                                          'heartbroken', 'devastated',
                                                          'lonely', 'loneliness', 'exhaustion',
                                                          'down', 'struggling'])
        if emotional_query:
            # Check for empathy in response
            empathy_words = ['understand', 'sorry', 'hear', 'feel', 'empathize',
                           'compassion', 'care', 'support', 'natural', 'valid',
                           'okay', 'normal', 'understandable']
            empathy_count = sum(1 for e in empathy_words if e in response_lower)
            if empathy_count == 0:
                need_score -= 3.0
            else:
                need_score += min(2.0, empathy_count * 0.5)
        
        need_score = max(0.0, min(15.0, need_score))
        
        # ============================================================
        # 7. COMBINE ALL SCORES
        # ============================================================
        # Weights: length(20) + topic(25) + structure(20) + content(20) + need(15) = 100
        raw_score = length_score + topic_score + struct_score + content_score + need_score
        
        # ============================================================
        # 8. PENALTY: Truncated/incomplete responses
        # ============================================================
        # If response appears cut off (no ending punctuation, ends mid-sentence)
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        if last_char not in '.!?:"\')':
            raw_score *= 0.90  # 10% penalty for apparent truncation
        
        # ============================================================
        # 9. PENALTY: Very short responses for complex queries
        # ============================================================
        if estimated_query_complexity >= 3 and resp_word_count < 40:
            raw_score *= 0.7
        
        # ============================================================
        # 10. MAP to 1-5 scale to match examples
        # ============================================================
        # raw_score is 0-100, map to approximately 1-5
        # Use a mapping that creates good discrimination
        
        if raw_score <= 20:
            final_score = 1.0
        elif raw_score <= 35:
            final_score = 1.0 + (raw_score - 20) / 15.0
        elif raw_score <= 50:
            final_score = 2.0 + (raw_score - 35) / 15.0
        elif raw_score <= 65:
            final_score = 3.0 + (raw_score - 50) / 15.0
        elif raw_score <= 80:
            final_score = 4.0 + (raw_score - 65) / 15.0
        else:
            final_score = 5.0
        
        final_score = max(0.5, min(5.0, round(final_score, 2)))
        
        return final_score
        
    except Exception as e:
        # Fallback: return a middle-of-the-road score
        try:
            # Simple fallback based on length ratio
            resp_len = len(str(response))
            if resp_len > 200:
                return 3.0
            elif resp_len > 50:
                return 2.0
            else:
                return 1.0
        except:
            return 2.5