def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response to a query.
    Returns a numeric score where HIGHER = BETTER quality.
    
    Strategy: Analyze structural completeness, topic coverage, depth indicators,
    and how well the response addresses the query's components.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        # Handle edge cases
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0  # Can't evaluate relevance without query
        
        query = query.strip()
        response = response.strip()
        
        if len(response) < 10:
            return 0.5
        
        score = 0.0
        
        # ============================================================
        # 1. RESPONSE LENGTH AND SUBSTANCE (0-15 points)
        # ============================================================
        resp_len = len(response)
        word_count = len(response.split())
        
        # Reward adequate length but with diminishing returns
        if word_count < 20:
            length_score = word_count * 0.2
        elif word_count < 50:
            length_score = 4 + (word_count - 20) * 0.1
        elif word_count < 150:
            length_score = 7 + (word_count - 50) * 0.05
        elif word_count < 300:
            length_score = 12 + (word_count - 150) * 0.015
        else:
            length_score = 14.25 + min(0.75, (word_count - 300) * 0.002)
        
        score += min(15, length_score)
        
        # ============================================================
        # 2. STRUCTURAL ORGANIZATION (0-20 points)
        # ============================================================
        struct_score = 0.0
        
        # Check for numbered lists (indicates systematic coverage)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        num_list_count = len(numbered_items)
        if num_list_count >= 2:
            struct_score += min(5, num_list_count * 1.0)
        
        # Check for bullet points
        bullet_items = re.findall(r'(?:^|\n)\s*[-•\*]\s', response)
        bullet_count = len(bullet_items)
        if bullet_count >= 2:
            struct_score += min(4, bullet_count * 0.8)
        
        # Check for headers/sections (markdown or caps)
        headers = re.findall(r'(?:^|\n)\s*#{1,4}\s+.+', response)
        bold_headers = re.findall(r'\*\*[^*]+\*\*', response)
        header_count = len(headers) + len(bold_headers) // 2
        if header_count >= 1:
            struct_score += min(5, header_count * 1.2)
        
        # Check for paragraphs (multiple distinct blocks of text)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip() and len(p.strip()) > 20]
        if len(paragraphs) >= 2:
            struct_score += min(3, len(paragraphs) * 0.6)
        
        # Check for colons (often indicate explanations/definitions)
        colon_explanations = re.findall(r':\s+\w', response)
        if len(colon_explanations) >= 2:
            struct_score += min(3, len(colon_explanations) * 0.4)
        
        score += min(20, struct_score)
        
        # ============================================================
        # 3. QUERY COMPONENT COVERAGE (0-25 points)
        # ============================================================
        coverage_score = 0.0
        
        # Extract question words and key aspects from query
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Detect sub-questions (multiple question marks or conjunctions)
        question_marks = query.count('?')
        sub_questions = max(1, question_marks)
        
        # Check for "and", "also", "as well" suggesting multiple aspects
        multi_aspect_words = ['and', 'also', 'as well', 'additionally', 'plus', 'both', 'or']
        aspect_count = sum(1 for w in multi_aspect_words if f' {w} ' in query_lower)
        estimated_aspects = max(sub_questions, 1 + aspect_count)
        
        # Extract content words from query (nouns, verbs, adjectives)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'i', 'me', 'my', 'myself', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
            'them', 'their', 'any', 'think', 'need', 'want', 'get', 'help'
        }
        
        # Tokenize query
        query_tokens = re.findall(r'[a-zA-Z]+', query_lower)
        query_content_words = [w for w in query_tokens if w not in stop_words and len(w) > 2]
        
        # Check how many query content words appear in response
        if query_content_words:
            words_found = sum(1 for w in query_content_words if w in response_lower)
            word_coverage_ratio = words_found / len(query_content_words)
            coverage_score += word_coverage_ratio * 10
        
        # Check for query-specific bigrams in response
        query_bigrams = []
        for i in range(len(query_tokens) - 1):
            if query_tokens[i] not in stop_words or query_tokens[i+1] not in stop_words:
                bigram = query_tokens[i] + ' ' + query_tokens[i+1]
                query_bigrams.append(bigram)
        
        if query_bigrams:
            bigrams_found = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_ratio = bigrams_found / len(query_bigrams)
            coverage_score += bigram_ratio * 8
        
        # Reward responses that seem to address multiple facets
        # by checking for transition/enumeration patterns
        transition_words = [
            'first', 'second', 'third', 'fourth', 'fifth',
            'additionally', 'furthermore', 'moreover', 'also',
            'another', 'in addition', 'next', 'finally', 'lastly',
            'on the other hand', 'however', 'alternatively',
            'for example', 'for instance', 'such as',
            'in contrast', 'meanwhile', 'besides'
        ]
        transitions_found = sum(1 for tw in transition_words if tw in response_lower)
        coverage_score += min(7, transitions_found * 1.2)
        
        score += min(25, coverage_score)
        
        # ============================================================
        # 4. DEPTH AND DETAIL INDICATORS (0-20 points)
        # ============================================================
        depth_score = 0.0
        
        # Check for explanatory phrases (indicates depth)
        explanation_phrases = [
            'because', 'this means', 'in other words', 'specifically',
            'the reason', 'this is due to', 'as a result', 'therefore',
            'consequently', 'which means', 'this allows', 'this ensures',
            'important to note', 'keep in mind', 'it is worth',
            'this is important', 'the key', 'essentially', 'fundamentally'
        ]
        explanations_found = sum(1 for ep in explanation_phrases if ep in response_lower)
        depth_score += min(6, explanations_found * 1.2)
        
        # Check for specific details (numbers, proper nouns, technical terms)
        numbers_in_response = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        depth_score += min(3, len(numbers_in_response) * 0.4)
        
        # Check for parenthetical explanations
        parentheticals = re.findall(r'\([^)]{5,}\)', response)
        depth_score += min(2, len(parentheticals) * 0.7)
        
        # Average sentence length (longer sentences often = more detailed)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if sentences:
            avg_sentence_words = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_words > 10:
                depth_score += min(3, (avg_sentence_words - 10) * 0.3)
        
        # Vocabulary richness (type-token ratio on content words)
        resp_tokens = re.findall(r'[a-zA-Z]+', response_lower)
        resp_content = [w for w in resp_tokens if w not in stop_words and len(w) > 2]
        if len(resp_content) > 10:
            unique_ratio = len(set(resp_content)) / len(resp_content)
            depth_score += unique_ratio * 4
        
        # Check for examples
        example_indicators = ['for example', 'for instance', 'such as', 'e.g.', 'like ', 'including']
        examples_found = sum(1 for ei in example_indicators if ei in response_lower)
        depth_score += min(2, examples_found * 0.8)
        
        score += min(20, depth_score)
        
        # ============================================================
        # 5. COMPLETENESS SIGNALS (0-10 points)
        # ============================================================
        completeness_score = 0.0
        
        # Check if response seems truncated (ends mid-sentence)
        last_chars = response[-5:].strip() if len(response) >= 5 else response
        ends_properly = last_chars[-1] in '.!?")\']' if last_chars else False
        
        if ends_properly:
            completeness_score += 3
        else:
            # Penalize truncation
            completeness_score -= 2
        
        # Check for conclusion/summary indicators
        conclusion_phrases = [
            'in conclusion', 'in summary', 'to summarize', 'overall',
            'in short', 'to sum up', 'hope this helps', 'good luck',
            'feel free', 'let me know', 'happy to help'
        ]
        has_conclusion = any(cp in response_lower for cp in conclusion_phrases)
        if has_conclusion:
            completeness_score += 3
        
        # Check for introduction (contextual framing)
        intro_phrases = [
            'great question', 'that\'s a', 'certainly', 'absolutely',
            'here are', 'here\'s', 'let me', 'i\'d be happy',
            'there are several', 'there are many', 'to answer'
        ]
        has_intro = any(ip in response_lower[:200] for ip in intro_phrases)
        if has_intro:
            completeness_score += 2
        
        # Check for caveats/nuance (indicates thorough treatment)
        nuance_phrases = [
            'however', 'although', 'while', 'on the other hand',
            'it depends', 'keep in mind', 'note that', 'be aware',
            'it\'s important to', 'it is important to', 'worth noting'
        ]
        nuances_found = sum(1 for np in nuance_phrases if np in response_lower)
        completeness_score += min(2, nuances_found * 0.6)
        
        score += max(0, min(10, completeness_score))
        
        # ============================================================
        # 6. EDGE CASE AND QUALIFICATION HANDLING (0-10 points)
        # ============================================================
        edge_score = 0.0
        
        # Check if query asks about edge cases, exceptions, or qualifications
        query_asks_edge = any(w in query_lower for w in [
            'what if', 'edge case', 'exception', 'always', 'never',
            'any', 'all', 'every', 'different', 'various', 'multiple'
        ])
        
        # Check if response handles qualifications
        qualification_phrases = [
            'depending on', 'it depends', 'in some cases', 'in most cases',
            'typically', 'usually', 'generally', 'often', 'sometimes',
            'may vary', 'can vary', 'not always', 'exception',
            'unless', 'provided that', 'assuming', 'if you'
        ]
        qualifications_found = sum(1 for qp in qualification_phrases if qp in response_lower)
        edge_score += min(5, qualifications_found * 1.0)
        
        # Check for multiple options/alternatives presented
        alternative_phrases = [
            'alternatively', 'another option', 'you could also',
            'another way', 'option', 'method', 'approach',
            'on the other hand', 'or you can', 'you might also'
        ]
        alternatives_found = sum(1 for ap in alternative_phrases if ap in response_lower)
        edge_score += min(5, alternatives_found * 1.2)
        
        score += min(10, edge_score)
        
        # ============================================================
        # FINAL ADJUSTMENTS
        # ============================================================
        
        # Penalty for very short responses relative to query complexity
        query_word_count = len(query.split())
        if query_word_count > 15 and word_count < 50:
            score *= 0.7
        
        # Penalty for responses that are clearly cut off mid-word
        if response[-1].isalpha() and not response.endswith(('etc', 'etc.')):
            last_word = response.split()[-1] if response.split() else ''
            # Check if it seems like a real ending word
            if len(last_word) > 1 and not last_word[-1] in '.!?,;:)]\'"':
                score *= 0.85
        
        # Bonus for responses that directly address the query type
        if any(w in query_lower for w in ['how', 'steps', 'process', 'procedure']):
            # How-to questions benefit from step-by-step format
            if num_list_count >= 3:
                score += 3
        
        if any(w in query_lower for w in ['why', 'reason', 'explain']):
            # Why questions benefit from causal language
            causal_words = ['because', 'since', 'due to', 'as a result', 'therefore', 'reason']
            causal_found = sum(1 for cw in causal_words if cw in response_lower)
            if causal_found >= 2:
                score += 2
        
        if any(w in query_lower for w in ['what are', 'list', 'ideas', 'suggestions', 'examples']):
            # List questions benefit from enumeration
            if num_list_count >= 3 or bullet_count >= 3:
                score += 3
        
        # Cap the score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a basic score based on response length
        try:
            if response and len(response) > 0:
                return min(30.0, len(response.split()) * 0.15)
            return 0.0
        except:
            return 0.0