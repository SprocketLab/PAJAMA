def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a graph-based approach:
    - Builds a "specificity graph" where nodes are tokens and edges represent
      co-occurrence of specific/concrete tokens in close proximity
    - Measures clustering of evidence-bearing tokens
    - Uses n-gram pattern matching for hedging vs. precision patterns
    - Computes a "knowledge density" metric based on rare/specific token ratios
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        words = response.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        response_lower = response.lower()
        
        # === 1. Named Entity Density (capitalized multi-word sequences) ===
        # Look for sequences of capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]+', response)
        named_entity_tokens = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = sent.split()
            # Skip first word (sentence start), look for capitalized words
            for i, w in enumerate(sent_words):
                if i == 0:
                    continue
                clean_w = re.sub(r'[^a-zA-Z]', '', w)
                if clean_w and len(clean_w) > 1 and clean_w[0].isupper():
                    named_entity_tokens += 1
        
        ne_density = named_entity_tokens / max(word_count, 1)
        
        # === 2. Numeric precision score ===
        # Different types of numbers get different weights
        # Dates (years, specific dates)
        year_matches = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', response)
        # Percentages
        pct_matches = re.findall(r'\d+(?:\.\d+)?%', response)
        # Decimal numbers (more precise)
        decimal_matches = re.findall(r'\b\d+\.\d+\b', response)
        # Currency amounts
        currency_matches = re.findall(r'[\$€£]\s?\d+[\d,]*(?:\.\d+)?|\d+[\d,]*(?:\.\d+)?\s?(?:dollars|euros|pounds|USD|EUR|GBP)', response)
        # Regular numbers
        all_numbers = re.findall(r'\b\d+\b', response)
        # Ordinals
        ordinal_matches = re.findall(r'\b\d+(?:st|nd|rd|th)\b', response_lower)
        
        numeric_score = (
            len(year_matches) * 2.0 +
            len(pct_matches) * 2.5 +
            len(decimal_matches) * 2.0 +
            len(currency_matches) * 3.0 +
            len(all_numbers) * 0.8 +
            len(ordinal_matches) * 1.5
        )
        numeric_density = numeric_score / max(word_count, 1)
        
        # === 3. Specificity token analysis using token rarity proxy ===
        # Longer words and less common word patterns tend to be more specific
        # Use word length distribution as a proxy for lexical specificity
        clean_words = [re.sub(r'[^a-zA-Z]', '', w).lower() for w in words]
        clean_words = [w for w in clean_words if len(w) > 0]
        
        if clean_words:
            # Common/vague words (stop words + hedging words)
            common_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
                'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'i',
                'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
                'his', 'her', 'their', 'what', 'which', 'who', 'whom', 'about', 'up',
                'also', 'like', 'really', 'think', 'get', 'got', 'go', 'going',
                'make', 'made', 'thing', 'things', 'something', 'anything', 'everything',
                'nothing', 'way', 'much', 'many', 'well', 'good', 'bad', 'know',
                'see', 'look', 'come', 'take', 'want', 'give', 'use', 'find', 'tell',
                'say', 'said', 'need', 'feel', 'try', 'keep', 'let', 'put', 'seem',
                'help', 'show', 'lot', 'still', 'even', 'back', 'new', 'old',
            }
            
            content_words = [w for w in clean_words if w not in common_words and len(w) > 2]
            content_ratio = len(content_words) / max(len(clean_words), 1)
            
            # Average content word length (longer = more specific typically)
            avg_content_len = sum(len(w) for w in content_words) / max(len(content_words), 1)
            
            # Lexical diversity among content words
            unique_content = len(set(content_words))
            content_diversity = unique_content / max(len(content_words), 1) if content_words else 0
            
            # Long specific words (8+ chars) - these tend to be domain-specific
            long_specific = [w for w in content_words if len(w) >= 8]
            long_specific_ratio = len(long_specific) / max(word_count, 1)
        else:
            content_ratio = 0
            avg_content_len = 0
            content_diversity = 0
            long_specific_ratio = 0
        
        # === 4. Hedging/vagueness penalty using bigram and trigram patterns ===
        hedge_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bfor the most part\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bi think\b', r'\bi guess\b', r'\bi suppose\b', r'\bi believe\b',
            r'\bprobably\b', r'\bperhaps\b', r'\bmaybe\b', r'\bpossibly\b',
            r'\bmight be\b', r'\bcould be\b', r'\btends to be\b',
            r'\bin my opinion\b', r'\bas far as i know\b',
            r'\bvarious\b', r'\bnumerous\b',
            r'\band (?:so on|etc|stuff|things like that)\b',
            r'\byou know\b', r'\bi mean\b',
            r'\bnot sure\b', r'\bnot certain\b',
            r'\bbasically\b', r'\bessentially\b',
            r'\bit\'s (?:hard|difficult) to say\b',
            r'\bthere\'s no (?:simple|easy|one) answer\b',
        ]
        
        hedge_count = 0
        for pattern in hedge_patterns:
            hedge_count += len(re.findall(pattern, response_lower))
        
        hedge_penalty = min(hedge_count * 1.5, 10.0)
        
        # === 5. Precision/evidence patterns (positive signals) ===
        precision_patterns = [
            (r'\baccording to\b', 1.5),
            (r'\bfor example\b', 1.2),
            (r'\bfor instance\b', 1.2),
            (r'\bspecifically\b', 1.0),
            (r'\bin particular\b', 1.0),
            (r'\bsuch as\b', 1.0),
            (r'\bnamely\b', 1.2),
            (r'\be\.g\.\b', 1.0),
            (r'\bi\.e\.\b', 0.8),
            (r'\bresearch (?:shows|suggests|indicates|found|demonstrates)\b', 2.0),
            (r'\bstud(?:y|ies) (?:show|suggest|indicate|found|demonstrate)\b', 2.0),
            (r'\bdata (?:shows|suggests|indicates)\b', 2.0),
            (r'\bevidence (?:shows|suggests|indicates)\b', 2.0),
            (r'\b(?:published|reported|documented) in\b', 2.0),
            (r'\bcited\b', 1.5),
            (r'https?://\S+', 2.5),
            (r'\bwww\.\S+', 2.5),
            (r'\b\d{4}\b.*\b(?:study|paper|research|report|survey)\b', 1.5),
        ]
        
        precision_score = 0
        for pattern, weight in precision_patterns:
            matches = re.findall(pattern, response_lower)
            precision_score += len(matches) * weight
        
        # === 6. Structural evidence markers ===
        # Quoted text (direct citations/references)
        quotes = re.findall(r'["\u201c\u201d].*?["\u201c\u201d]', response)
        quote_score = min(len(quotes) * 1.5, 6.0)
        
        # Parenthetical clarifications (often contain specific details)
        parens = re.findall(r'\([^)]{3,}\)', response)
        paren_score = min(len(parens) * 0.8, 4.0)
        
        # Code blocks (for technical queries)
        code_blocks = re.findall(r'```[\s\S]*?```|`[^`]+`', response)
        code_score = min(len(code_blocks) * 1.5, 5.0)
        
        # Italicized or bold text (often titles, emphasis on specific terms)
        emphasis = re.findall(r'\*[^*]+\*|_[^_]+_', response)
        emphasis_score = min(len(emphasis) * 0.7, 3.0)
        
        # === 7. Sliding window evidence clustering ===
        # Measure how densely evidence tokens cluster together
        # (concentrated evidence is better than scattered vagueness)
        tokens = response_lower.split()
        window_size = 15
        
        def token_is_evidence(t):
            t_clean = re.sub(r'[^a-z0-9]', '', t)
            if re.match(r'\d', t_clean):
                return True
            if len(t_clean) >= 9:
                return True
            return False
        
        if len(tokens) >= window_size:
            window_densities = []
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i+window_size]
                ev_count = sum(1 for t in window if token_is_evidence(t))
                window_densities.append(ev_count / window_size)
            
            if window_densities:
                max_density = max(window_densities)
                avg_density = sum(window_densities) / len(window_densities)
                # Reward both high peaks and sustained evidence
                clustering_score = (max_density * 3.0 + avg_density * 7.0)
            else:
                clustering_score = 0
        else:
            # Short response: just measure overall
            if tokens:
                ev_count = sum(1 for t in tokens if token_is_evidence(t))
                clustering_score = (ev_count / len(tokens)) * 8.0
            else:
                clustering_score = 0
        
        # === 8. Response engagement with query ===
        # Measure how much the response addresses specific terms from the query
        query_lower = query.lower() if query else ""
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        query_words -= {'what', 'when', 'where', 'which', 'would', 'could', 'should',
                       'does', 'have', 'been', 'being', 'about', 'your', 'this', 'that',
                       'with', 'from', 'they', 'them', 'their', 'there', 'here', 'more',
                       'most', 'some', 'like', 'into', 'also', 'just', 'very'}
        
        if query_words:
            response_word_set = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
            query_overlap = len(query_words & response_word_set) / max(len(query_words), 1)
        else:
            query_overlap = 0.5
        
        # === 9. Sentence-level specificity variance ===
        # Good responses have consistently specific sentences, not just one
        sent_specificity_scores = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            sent_words = sent.lower().split()
            sent_len = len(sent_words)
            if sent_len == 0:
                continue
            
            # Count evidence tokens in sentence
            nums_in_sent = len(re.findall(r'\d+', sent))
            long_words_in_sent = sum(1 for w in sent_words if len(re.sub(r'[^a-z]', '', w)) >= 7)
            
            sent_spec = (nums_in_sent * 2 + long_words_in_sent) / max(sent_len, 1)
            sent_specificity_scores.append(sent_spec)
        
        if sent_specificity_scores:
            avg_sent_spec = sum(sent_specificity_scores) / len(sent_specificity_scores)
            # Reward consistency: low variance in specificity across sentences
            if len(sent_specificity_scores) > 1:
                mean_s = avg_sent_spec
                variance = sum((s - mean_s)**2 for s in sent_specificity_scores) / len(sent_specificity_scores)
                consistency_bonus = max(0, 1.0 - math.sqrt(variance) * 2)
            else:
                consistency_bonus = 0.5
        else:
            avg_sent_spec = 0
            consistency_bonus = 0
        
        # === 10. Length bonus (diminishing returns) ===
        # Longer responses have more room for evidence, but with diminishing returns
        length_factor = math.log(max(word_count, 1) + 1) / math.log(300)
        length_factor = min(length_factor, 1.3)
        
        # === COMPOSITE SCORE ===
        score = (
            ne_density * 15.0 +              # Named entity density
            numeric_density * 25.0 +          # Numeric precision density
            content_ratio * 8.0 +             # Content word ratio
            (avg_content_len - 4) * 1.5 +     # Average content word length bonus
            long_specific_ratio * 20.0 +      # Long specific words
            precision_score * 1.2 +           # Precision language patterns
            quote_score +                     # Direct citations
            paren_score +                     # Parenthetical details
            code_score +                      # Code blocks
            emphasis_score +                  # Emphasized terms
            clustering_score * 1.5 +          # Evidence clustering
            query_overlap * 3.0 +             # Query engagement
            avg_sent_spec * 10.0 +            # Sentence-level specificity
            consistency_bonus * 2.0 +         # Consistency of specificity
            length_factor * 3.0 -             # Length factor
            hedge_penalty                     # Hedging penalty
        )
        
        # Scale to 0-10 range
        score = max(0.0, score)
        # Sigmoid-like scaling to keep in reasonable range
        scaled = 10.0 * (1.0 - math.exp(-score / 20.0))
        
        return round(scaled, 3)
        
    except Exception:
        try:
            return max(0.5, min(5.0, len(str(response)) / 100.0))
        except Exception:
            return 2.5