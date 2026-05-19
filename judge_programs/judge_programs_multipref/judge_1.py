def judging_function(query, response):
    """
    Evaluates relevance of an LLM response to a query.
    Uses keyword overlap, topic alignment, structural quality, and direct address detection.
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        import re
        import math
        from collections import Counter
        
        # Handle edge cases
        if not query or not response:
            return 0.0
        if not isinstance(query, str) or not isinstance(response, str):
            return 0.0
        
        query = query.strip()
        response = response.strip()
        
        if len(query) == 0 or len(response) == 0:
            return 0.0
        
        # ---- Tokenization helpers ----
        def tokenize(text):
            """Lowercase tokenization, removing punctuation."""
            text = text.lower()
            tokens = re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text)
            return tokens
        
        def get_ngrams(tokens, n):
            """Generate n-grams from token list."""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Common English stopwords
        stopwords = {
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
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'up', 'about', 'also',
            'like', 'get', 'got', 'make', 'made', 'much', 'many', 'well', 'back',
            'even', 'still', 'way', 'take', 'come', 'go', 'know', 'see', 'think',
            'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem',
            'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin', 'show',
            'hear', 'play', 'run', 'move', 'live', 'believe', 'bring', 'happen',
            'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
            'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch',
            'follow', 'stop', 'create', 'speak', 'read', 'add', 'spend', 'grow',
            'open', 'walk', 'win', 'offer', 'remember', 'consider', 'appear',
            'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay',
            'fall', 'cut', 'reach', 'kill', 'remain', 'am', 'an', 'don', 't',
            's', 're', 've', 'll', 'd', 'm', 'won', 'isn', 'aren', 'wasn',
            'weren', 'hasn', 'haven', 'hadn', 'doesn', 'didn', 'wouldn', 'couldn',
            'shouldn', 'mustn', 'needn'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if len(query_tokens) == 0 or len(response_tokens) == 0:
            return 0.0
        
        # Content words (non-stopwords)
        query_content = [t for t in query_tokens if t not in stopwords and len(t) > 2]
        response_content = [t for t in response_tokens if t not in stopwords and len(t) > 2]
        
        # ---- FEATURE 1: Content Word Recall (query keywords found in response) ----
        if len(query_content) > 0:
            query_content_set = set(query_content)
            response_content_set = set(response_content)
            content_recall = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            content_recall = 0.5  # neutral if no content words
        
        # ---- FEATURE 2: Weighted keyword coverage with frequency ----
        # Weight rarer query words higher
        query_content_counter = Counter(query_content)
        response_content_counter = Counter(response_content)
        
        if len(query_content_counter) > 0:
            weighted_coverage = 0.0
            total_weight = 0.0
            for word, count in query_content_counter.items():
                weight = 1.0 + math.log(1 + count)
                total_weight += weight
                if word in response_content_counter:
                    weighted_coverage += weight * min(1.0, response_content_counter[word] / max(count, 1))
            weighted_keyword_score = weighted_coverage / total_weight if total_weight > 0 else 0.0
        else:
            weighted_keyword_score = 0.5
        
        # ---- FEATURE 3: Bigram overlap (captures phrase-level relevance) ----
        query_bigrams = get_ngrams(query_tokens, 2)
        response_bigrams = get_ngrams(response_tokens, 2)
        
        if len(query_bigrams) > 0:
            query_bigram_set = set(query_bigrams)
            response_bigram_set = set(response_bigrams)
            bigram_recall = len(query_bigram_set & response_bigram_set) / len(query_bigram_set)
        else:
            bigram_recall = 0.0
        
        # ---- FEATURE 4: Topic coherence via cosine similarity of TF vectors ----
        all_words = set(query_content) | set(response_content)
        if len(all_words) > 0:
            q_vec = []
            r_vec = []
            for w in all_words:
                q_vec.append(query_content_counter.get(w, 0))
                r_vec.append(response_content_counter.get(w, 0))
            
            dot = sum(a * b for a, b in zip(q_vec, r_vec))
            mag_q = math.sqrt(sum(a * a for a in q_vec))
            mag_r = math.sqrt(sum(b * b for b in r_vec))
            
            if mag_q > 0 and mag_r > 0:
                cosine_sim = dot / (mag_q * mag_r)
            else:
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0
        
        # ---- FEATURE 5: Direct address detection ----
        # Check if response starts by addressing the query topic
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check for echo/acknowledgment patterns
        direct_address_score = 0.0
        
        # Check if first sentence of response contains query keywords
        first_sentence = re.split(r'[.!?\n]', response)[0].lower() if response else ""
        first_sent_tokens = set(tokenize(first_sentence))
        query_content_set = set(query_content)
        
        if len(query_content_set) > 0:
            first_sent_overlap = len(first_sent_tokens & query_content_set) / len(query_content_set)
            direct_address_score = min(1.0, first_sent_overlap * 1.5)
        
        # ---- FEATURE 6: Response structure quality ----
        structure_score = 0.0
        
        # Check for organized formatting (numbered lists, headers, bullet points)
        has_numbering = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bullets = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
        has_headers = bool(re.search(r'#{1,4}\s+\w+|^\*\*[^*]+\*\*', response, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        
        structure_indicators = sum([has_numbering, has_bullets, has_headers, has_bold])
        structure_score = min(1.0, structure_indicators * 0.3)
        
        # ---- FEATURE 7: Response length adequacy ----
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Responses should generally be longer than queries, but not absurdly so
        if resp_len < 10:
            length_score = 0.2
        elif resp_len < 30:
            length_score = 0.5
        elif resp_len < 50:
            length_score = 0.7
        elif resp_len < 200:
            length_score = 1.0
        elif resp_len < 500:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ---- FEATURE 8: Engagement/personalization signals ----
        engagement_score = 0.0
        
        # Acknowledging the user's situation/question
        engagement_patterns = [
            r'\bthat\'s\s+(a\s+)?(great|good|excellent|wonderful|fantastic)',
            r'\bcertainly\b', r'\babsolutely\b', r'\bof course\b',
            r'\bhere\s+are\b', r'\bhere\'s\b', r'\blet\'s\b',
            r'\byou\s+(can|could|might|should|may)\b',
            r'\byour\b',
        ]
        
        for pattern in engagement_patterns:
            if re.search(pattern, response_lower):
                engagement_score += 0.15
        engagement_score = min(1.0, engagement_score)
        
        # ---- FEATURE 9: Specificity / informativeness ----
        # Measure unique content words as a proxy for informativeness
        if len(response_content) > 0:
            unique_ratio = len(set(response_content)) / len(response_content)
            # Also count specific details: numbers, proper nouns, technical terms
            numbers_in_response = len(re.findall(r'\b\d+(?:\.\d+)?\b', response))
            specificity_score = min(1.0, unique_ratio * 0.7 + min(numbers_in_response * 0.05, 0.3))
        else:
            specificity_score = 0.0
        
        # ---- FEATURE 10: Semantic field coverage ----
        # Check if response covers related semantic concepts beyond exact matches
        # Use character trigram overlap as a fuzzy matching proxy
        def char_trigrams(text):
            text = text.lower()
            return set(text[i:i+3] for i in range(len(text) - 2))
        
        query_trigrams = char_trigrams(' '.join(query_content))
        response_trigrams = char_trigrams(' '.join(response_content))
        
        if len(query_trigrams) > 0:
            trigram_overlap = len(query_trigrams & response_trigrams) / len(query_trigrams)
        else:
            trigram_overlap = 0.0
        
        # ---- FEATURE 11: Completeness - does response seem to finish or is it cut off? ----
        completeness_score = 1.0
        if response.rstrip().endswith((',', 'a', 'an', 'the', 'and', 'or', 'but', 'to', 'of', 'in')):
            completeness_score = 0.6
        # Check if last char suggests truncation
        last_chars = response.rstrip()[-3:] if len(response) > 3 else response
        if not any(last_chars.endswith(c) for c in '.!?)"\':;0123456789*'): 
            completeness_score *= 0.8
        
        # ---- FEATURE 12: Question type alignment ----
        # Detect question type and check if response format matches
        question_type_bonus = 0.0
        
        # Yes/No questions
        yn_patterns = [r'^(do|does|did|is|are|was|were|can|could|should|would|will|has|have)\s', 
                       r'^(don\'t|doesn\'t|isn\'t|aren\'t)\s']
        is_yn_question = any(re.search(p, query_lower) for p in yn_patterns)
        
        if is_yn_question:
            # Response should start with yes/no or clearly take a stance
            if re.search(r'^(yes|no|i\s+(do|don\'t|believe|think))', response_lower.strip()):
                question_type_bonus = 0.3
        
        # How-to questions
        if re.search(r'^how\s+(can|do|to|should)', query_lower):
            # Should contain steps or instructions
            if has_numbering or re.search(r'\bstep\b', response_lower):
                question_type_bonus = 0.3
        
        # What questions
        if re.search(r'^what\s+(is|are|was|were)', query_lower):
            # Should contain definitions or explanations
            if re.search(r'\b(is|are|refers?\s+to|means?|defined)\b', response_lower):
                question_type_bonus = 0.2
        
        # ---- Combine all features with weights ----
        score = (
            content_recall * 18.0 +          # Core relevance via keyword recall
            weighted_keyword_score * 12.0 +   # Weighted keyword coverage
            bigram_recall * 8.0 +             # Phrase-level relevance
            cosine_sim * 10.0 +               # Topic alignment
            direct_address_score * 8.0 +      # Addresses query directly
            structure_score * 5.0 +           # Well-organized
            length_score * 4.0 +              # Appropriate length
            engagement_score * 5.0 +          # Engaging/personalized
            specificity_score * 6.0 +         # Specific and informative
            trigram_overlap * 8.0 +           # Fuzzy semantic coverage
            completeness_score * 4.0 +        # Not truncated
            question_type_bonus * 8.0         # Matches question type
        )
        
        # Normalize to 0-100 range
        # Max theoretical = 18+12+8+10+8+5+4+5+6+8+4+8 = 96 (but rarely all max)
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Never crash - return neutral score on error
        return 25.0