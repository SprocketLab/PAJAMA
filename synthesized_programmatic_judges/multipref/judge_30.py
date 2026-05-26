def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant focuses on:
    - Citation/reference patterns (specific names, dates, numbers as verifiable facts)
    - Hallucination red-flags (overly precise unsourced stats, absolute claims)
    - Appropriate hedging vs. overconfidence
    - Sensationalism and conspiracy-style language detection
    - Structural credibility signals (organized reasoning, step-by-step)
    - Specificity-to-query relevance ratio
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        response_lower = response.lower()
        query_lower = query.lower()
        words = response_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        score = 50.0  # Start at midpoint
        
        # === 1. VERIFIABLE FACT INDICATORS ===
        # Specific numbers (dates, measurements, quantities)
        number_patterns = re.findall(r'\b\d+[\.,]?\d*\b', response)
        num_count = len(number_patterns)
        # Moderate numbers are good; too many unsourced might be hallucination
        if 1 <= num_count <= 8:
            score += num_count * 1.2
        elif num_count > 8:
            score += 8 * 1.2 - (num_count - 8) * 0.3  # diminishing, slight penalty for excess
        
        # Year mentions (specific dates suggest factual grounding)
        year_mentions = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', response)
        score += min(len(year_mentions), 4) * 1.5
        
        # Named entities heuristic: capitalized multi-word sequences not at sentence start
        sentences = re.split(r'[.!?]+', response)
        named_entity_count = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 3:
                # Find capitalized words not at the start of sentence
                inner_words = sent.split()[1:]  # skip first word
                for w in inner_words:
                    if w and w[0].isupper() and len(w) > 1 and not w.isupper():
                        named_entity_count += 1
        score += min(named_entity_count, 10) * 0.8
        
        # === 2. HALLUCINATION RED FLAGS ===
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d+\.\d{2,}\s*%', response)
        score -= len(precise_stats) * 2.0
        
        # Absolute/universal claims
        absolute_phrases = [
            'always', 'never', 'everyone knows', 'it is a fact that',
            'undeniably', 'without question', 'there is no doubt',
            'proven beyond', 'absolutely certain', 'guaranteed to',
            'every single', 'no one can deny', '100% certain',
            'the truth is that', 'the fact is'
        ]
        absolute_count = sum(1 for phrase in absolute_phrases if phrase in response_lower)
        score -= absolute_count * 2.5
        
        # === 3. SENSATIONALISM & CONSPIRACY DETECTION ===
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'mind-blowing',
            'unbelievable', 'insane', 'crazy', 'terrifying',
            'they don\'t want you to know', 'wake up', 'sheeple',
            'mainstream media', 'cover-up', 'coverup', 'big pharma',
            'deep state', 'conspiracy', 'hoax', 'scam',
            'secret agenda', 'hidden truth', 'what they don\'t tell you',
            'exposed', 'bombshell', 'scandal'
        ]
        sensational_count = sum(1 for w in sensational_words if w in response_lower)
        score -= sensational_count * 4.0
        
        # Excessive exclamation marks (sensationalism indicator)
        exclamation_count = response.count('!')
        if exclamation_count > 3:
            score -= (exclamation_count - 3) * 1.0
        
        # ALL CAPS words (shouting/sensationalism) - excluding acronyms
        caps_words = [w for w in response.split() if w.isupper() and len(w) > 3]
        score -= min(len(caps_words), 5) * 1.5
        
        # === 4. APPROPRIATE HEDGING ===
        hedging_phrases = [
            'may', 'might', 'could', 'possibly', 'perhaps',
            'it appears', 'it seems', 'likely', 'unlikely',
            'generally', 'typically', 'often', 'sometimes',
            'in many cases', 'tend to', 'can vary',
            'depending on', 'it is possible', 'suggests that',
            'according to', 'research suggests', 'studies suggest',
            'it\'s worth noting', 'keep in mind', 'note that',
            'however', 'although', 'while', 'on the other hand'
        ]
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
        # Good hedging is positive, but too much can indicate lack of knowledge
        if hedge_count <= 8:
            score += hedge_count * 1.5
        else:
            score += 8 * 1.5 - (hedge_count - 8) * 0.5
        
        # === 5. STRUCTURAL CREDIBILITY ===
        # Numbered steps or lists (organized thinking)
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        has_numbered_list = len(numbered_items) >= 2
        if has_numbered_list:
            score += 3.0
        
        # Bold/formatted headers (markdown formatting shows structure)
        bold_headers = re.findall(r'\*\*[^*]+\*\*', response)
        has_formatting = len(bold_headers) >= 1
        if has_formatting:
            score += 2.5
        
        # Presence of ### headers
        md_headers = re.findall(r'#{1,3}\s+\S+', response)
        if md_headers:
            score += 2.0
        
        # Multi-sentence coherence: average sentence length variety
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sent_lengths) >= 3:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            sent_std = math.sqrt(sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths))
            # Some variety in sentence length is good (not robotic)
            if 2 < sent_std < 15:
                score += 2.0
            # Very short average sentences might indicate low substance
            if avg_sent_len < 5:
                score -= 2.0
            elif avg_sent_len > 10:
                score += 1.5
        
        # === 6. RESPONSE COMPLETENESS & ENGAGEMENT ===
        # Starts with direct acknowledgment of the query
        opening_engagement = [
            'certainly', 'great question', 'that\'s a', 'here',
            'yes', 'no,', 'absolutely', 'of course', 'sure',
            'let\'s', 'to answer', 'the answer'
        ]
        first_30_chars = response_lower[:60]
        engaged_opening = any(phrase in first_30_chars for phrase in opening_engagement)
        if engaged_opening:
            score += 2.0
        
        # Check if response seems cut off (ends mid-sentence without punctuation)
        stripped_response = response.rstrip()
        if stripped_response and stripped_response[-1] not in '.!?:)"\']':
            # Might be truncated - slight penalty but not huge
            score -= 1.0
        
        # === 7. QUERY-RESPONSE RELEVANCE (keyword overlap) ===
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        response_words_set = set(re.findall(r'\b[a-z]{3,}\b', response_lower))
        # Remove very common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'been', 'some', 'them', 'than', 'its', 'over',
                      'that', 'with', 'this', 'from', 'they', 'will', 'what',
                      'when', 'make', 'like', 'how', 'each', 'she', 'which',
                      'their', 'there', 'would', 'about', 'could', 'other'}
        query_content = query_words - stop_words
        response_content = response_words_set - stop_words
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
            score += overlap * 6.0
        
        # === 8. CITATION/REFERENCE PATTERNS ===
        citation_indicators = [
            'according to', 'research shows', 'studies have',
            'a study', 'researchers', 'published in',
            'data shows', 'evidence suggests', 'source:',
            'referenced', 'cited', 'per the', 'based on',
            'as noted', 'as described', 'documentation',
            'official', 'report'
        ]
        citation_count = sum(1 for c in citation_indicators if c in response_lower)
        score += min(citation_count, 5) * 1.8
        
        # === 9. EXPLANATORY DEPTH ===
        # Causal/explanatory connectors suggest deeper reasoning
        explanatory_words = [
            'because', 'therefore', 'thus', 'consequently',
            'as a result', 'this means', 'which means',
            'in other words', 'for example', 'for instance',
            'specifically', 'in particular', 'such as',
            'this is because', 'the reason', 'due to'
        ]
        explanation_count = sum(1 for e in explanatory_words if e in response_lower)
        score += min(explanation_count, 6) * 1.5
        
        # === 10. LEXICAL SOPHISTICATION (without being obscure) ===
        # Ratio of unique words to total (moderate diversity is good)
        if word_count > 10:
            unique_ratio = len(set(words)) / word_count
            # Sweet spot: 0.4-0.7 unique ratio
            if 0.4 <= unique_ratio <= 0.75:
                score += 3.0
            elif unique_ratio > 0.75:
                score += 1.5  # Very high might mean short or disjointed
            else:
                score += 0.5  # Very repetitive
        
        # Average word length (longer words can indicate more technical/precise language)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if 4.5 <= avg_word_len <= 6.5:
                score += 2.0
            elif avg_word_len > 6.5:
                score += 1.0  # Might be overly complex
        
        # === 11. PROPORTIONAL LENGTH BONUS ===
        # Longer responses tend to be more informative, but with diminishing returns
        length_bonus = math.log(max(word_count, 1) + 1) * 1.5
        score += min(length_bonus, 10.0)
        
        # === 12. PARAGRAPH STRUCTURE ===
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += min(len(paragraphs), 5) * 0.8
        
        # === 13. CONDITIONAL/NUANCED LANGUAGE ===
        conditional_phrases = [
            'if you', 'in case', 'whether', 'it depends',
            'on one hand', 'alternatively', 'another option',
            'pros and cons', 'trade-off', 'consider'
        ]
        conditional_count = sum(1 for c in conditional_phrases if c in response_lower)
        score += min(conditional_count, 4) * 1.5
        
        # Clamp score to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0