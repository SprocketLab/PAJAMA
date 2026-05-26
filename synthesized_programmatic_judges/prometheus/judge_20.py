def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a topic-coverage
    and structural depth analysis approach.
    
    Strategy: Extract key concepts/topics from the query, then measure how many
    are addressed in the response. Also analyze structural indicators of thoroughness
    like enumeration patterns, explanation depth, and multi-faceted addressing.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        if len(response.strip()) < 10:
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # 1. QUERY CONCEPT COVERAGE (0-25 points)
        # Extract meaningful content words from query, check coverage
        # ============================================================
        
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'up',
            'about', 'also', 'much', 'many', 'well', 'get', 'got', 'like',
            'make', 'made', 'know', 'think', 'take', 'come', 'go', 'see',
            'way', 'even', 'new', 'want', 'give', 'use', 'find', 'tell',
            'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep',
            'let', 'begin', 'show', 'hear', 'play', 'run', 'move', 'live',
            'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand',
            'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn',
            'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create',
            'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk',
            'win', 'offer', 'remember', 'consider', 'appear', 'buy', 'wait',
            'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut',
            'reach', 'kill', 'remain', 'am', 'been', 'person', 'individual',
            'someone', 'something', 'anything', 'everything', 'nothing',
            'however', 'must', 'any', 'every', 'one', 'two', 'three',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        query_words = extract_content_words(query)
        response_lower = response.lower()
        
        if query_words:
            # Count unique query concepts found in response
            query_unique = list(set(query_words))
            covered = sum(1 for w in query_unique if w in response_lower)
            coverage_ratio = covered / max(len(query_unique), 1)
            score += coverage_ratio * 25
        else:
            score += 12.5  # neutral if we can't extract
        
        # ============================================================
        # 2. MULTI-ASPECT ADDRESSING (0-20 points)
        # Check if response addresses multiple facets/sub-questions
        # ============================================================
        
        # Detect sub-questions or multiple aspects in query
        query_question_words = re.findall(r'\b(how|what|why|when|where|who|which|can|could|should|would)\b', query.lower())
        query_aspects = len(set(query_question_words))
        
        # Detect query keywords that suggest multiple needs
        need_indicators = re.findall(
            r'\b(seeking|need|want|looking for|require|assist|advice|comfort|help|explain|understand|manage|handle|cope|track|provide|ensure)\b',
            query.lower()
        )
        num_needs = len(set(need_indicators))
        
        # Check how many of these needs are reflected in response
        response_words_set = set(re.findall(r'[a-z]+', response_lower))
        
        # Semantic addressing: look for response patterns that address needs
        addressing_patterns = [
            r'\b(first|second|third|next|then|also|additionally|furthermore|moreover)\b',
            r'\b(important|essential|crucial|key|critical|significant)\b',
            r'\b(suggest|recommend|advise|consider|try|approach)\b',
            r'\b(because|since|therefore|thus|hence|reason)\b',
            r'\b(example|instance|such as|like|for instance)\b',
            r'\b(however|although|but|yet|nevertheless|on the other hand)\b',
            r'\b(feel|emotion|understand|acknowledge|recognize|appreciate)\b',
            r'\b(step|process|method|technique|strategy|approach)\b',
        ]
        
        addressing_count = 0
        for pattern in addressing_patterns:
            if re.search(pattern, response_lower):
                addressing_count += 1
        
        aspect_score = min(addressing_count / max(len(addressing_patterns), 1) * 20, 20)
        score += aspect_score
        
        # ============================================================
        # 3. STRUCTURAL DEPTH ANALYSIS (0-20 points)
        # Analyze if response has layered structure indicating thoroughness
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Numbered/bulleted items (strong completeness signal)
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        bullet_items = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response))
        list_items = numbered_items + bullet_items
        
        # Paragraphs (separated by double newline or significant breaks)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response) if len(p.strip()) > 20]
        num_paragraphs = len(paragraphs)
        
        # Structural depth score
        struct_score = 0
        
        # Reward enumeration (structured coverage)
        if list_items >= 3:
            struct_score += 8
        elif list_items >= 1:
            struct_score += 4
        
        # Reward multiple paragraphs
        if num_paragraphs >= 3:
            struct_score += 6
        elif num_paragraphs >= 2:
            struct_score += 3
        
        # Reward sufficient sentence count (depth)
        if num_sentences >= 8:
            struct_score += 6
        elif num_sentences >= 5:
            struct_score += 4
        elif num_sentences >= 3:
            struct_score += 2
        
        score += min(struct_score, 20)
        
        # ============================================================
        # 4. RESPONSE LENGTH AND INFORMATION DENSITY (0-15 points)
        # Longer responses with unique content words = more complete
        # ============================================================
        
        response_content_words = extract_content_words(response)
        unique_content = set(response_content_words)
        
        # Length-based (diminishing returns)
        word_count = len(response.split())
        length_score = min(math.log(max(word_count, 1) + 1) / math.log(300) * 10, 10)
        
        # Vocabulary richness (unique content words relative to total)
        if len(response_content_words) > 0:
            vocab_richness = len(unique_content) / len(response_content_words)
            richness_score = vocab_richness * 5
        else:
            richness_score = 0
        
        score += length_score + richness_score
        
        # ============================================================
        # 5. EMPATHY AND ENGAGEMENT SIGNALS (0-10 points)
        # For queries seeking emotional support, check empathetic coverage
        # ============================================================
        
        emotional_query = bool(re.search(
            r'\b(feeling|emotion|stress|frustrat|sad|lonely|heartbroken|devastat|comfort|support|cope|struggle|difficult|pain|grief|exhaust|upset|anxious|worried|fear)\b',
            query.lower()
        ))
        
        if emotional_query:
            empathy_markers = [
                r'\b(understand|sorry|hear|feel|completely|absolutely|natural|okay|perfectly|valid)\b',
                r'\b(it\'s okay|it\'s fine|that\'s|understandable|normal|natural)\b',
                r'\b(take .{0,20} time|give yourself|allow yourself|permit yourself)\b',
                r'\b(here for|listen|support|care|matter)\b',
                r'\b(remember|don\'t forget|keep in mind)\b',
            ]
            empathy_count = sum(1 for p in empathy_markers if re.search(p, response_lower))
            empathy_score = min(empathy_count / len(empathy_markers) * 10, 10)
            score += empathy_score
        else:
            # For non-emotional queries, reward specificity/detail
            specific_markers = [
                r'\b(specifically|particular|detail|precise|exact)\b',
                r'\b(for example|for instance|such as|e\.g\.|i\.e\.)\b',
                r'\b(note that|keep in mind|importantly|remember)\b',
                r'\b(step|phase|stage|part|component)\b',
                r'\b(ensure|verify|check|confirm|make sure)\b',
            ]
            specific_count = sum(1 for p in specific_markers if re.search(p, response_lower))
            specific_score = min(specific_count / len(specific_markers) * 10, 10)
            score += specific_score
        
        # ============================================================
        # 6. NEGATIVE SIGNALS / PENALTIES (0 to -10 points)
        # Penalize signs of incompleteness or dismissiveness
        # ============================================================
        
        penalties = 0
        
        # Dismissive or oversimplified language
        dismissive_patterns = [
            r'\bjust\b.*\bjust\b',  # repeated "just" = oversimplification
            r'\b(simply|easy|obviously|clearly)\b',
            r'\b(you should be able to|you\'ll be fine|no big deal)\b',
            r'\b(maybe you\'re|probably not|might not)\b.*\b(correct|right|using)\b',
        ]
        for p in dismissive_patterns:
            if re.search(p, response_lower):
                penalties += 1.5
        
        # Very short response for complex query
        if word_count < 30 and len(query.split()) > 20:
            penalties += 5
        
        # Response says it can't do something (negative capability)
        negative_capability = len(re.findall(
            r'\b(can\'t|cannot|unable|might not|won\'t be able|not able|not capable)\b',
            response_lower
        ))
        if negative_capability >= 2:
            penalties += 3
        
        # Truncation detection (response seems cut off)
        if response.rstrip()[-1:] not in '.!?"\')\n' and len(response) > 100:
            # Might be truncated - mild penalty since many examples are truncated
            pass  # Don't penalize truncation since examples show it's common
        
        score -= min(penalties, 10)
        
        # ============================================================
        # 7. ACTIONABILITY AND SOLUTION ORIENTATION (0-10 points)
        # Check if response provides actionable guidance when appropriate
        # ============================================================
        
        action_query = bool(re.search(
            r'\b(how|guide|help|assist|advice|manage|handle|cope|solve|fix|improve|create|make|build|design|develop|explain|tell)\b',
            query.lower()
        ))
        
        if action_query:
            action_patterns = [
                r'\b(you can|you could|you might|you should|try|consider|start)\b',
                r'\b(first|begin|start by|step)\b',
                r'\b(this (will|can|helps?|allows?))\b',
                r'\b(make sure|ensure|be sure|don\'t forget)\b',
                r'\b(one way|another|alternative|option)\b',
            ]
            action_count = sum(1 for p in action_patterns if re.search(p, response_lower))
            action_score = min(action_count / len(action_patterns) * 10, 10)
            score += action_score
        else:
            score += 5  # neutral
        
        # ============================================================
        # FINAL SCALING: Map to 1-5 range
        # ============================================================
        
        # Theoretical max is ~100, typical range 20-80
        # Map to 1-5 scale
        raw_score = max(score, 0)
        
        # Normalize: use sigmoid-like mapping
        # Calibrate based on examples
        normalized = 1 + 4 * (1 / (1 + math.exp(-0.08 * (raw_score - 45))))
        
        # Clamp to 1-5
        final_score = max(1.0, min(5.0, round(normalized, 2)))
        
        return final_score
        
    except Exception as e:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 50:
                return 3.0
            return 1.5
        except:
            return 2.0