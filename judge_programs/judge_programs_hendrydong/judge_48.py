def judging_function(query, response):
    """
    Evaluate clarity and conciseness using information density, sentence structure analysis,
    syntactic complexity estimation, and signal-to-noise ratio.
    
    This variant focuses on:
    1. Information density (unique content words per total words)
    2. Sentence length variance (consistent sentence lengths = clearer)
    3. Discourse marker quality (good vs bad connectives)
    4. Filler/weasel word ratio
    5. Clause depth estimation via punctuation patterns
    6. Direct address / engagement with query
    7. Specificity scoring (numbers, proper nouns, technical terms)
    8. Compression ratio (how much info per character)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            return 0.5
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response)
        words_lower = [w.lower() for w in words]
        total_words = len(words_lower)
        
        if total_words < 3:
            return 1.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Information Density ----
        # Function words (low information content)
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'must', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'over', 'about', 'against', 'without', 'within',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'and', 'but', 'or', 'nor', 'not', 'no', 'so', 'if', 'then', 'than',
            'too', 'very', 'just', 'also', 'there', 'here', 'what', 'which',
            'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'any',
            'only', 'own', 'same', 'while', 'because', 'until', 'although',
            'though', 'since', 'unless', 'whether', 'yet', 'still', 'already',
            'even', 'much', 'many', 'well', 'back', 'up', 'out', 'off',
            'am', 'get', 'got', 'goes', 'going', 'go', 'went', 'come', 'came',
            'make', 'made', 'take', 'took', 'give', 'gave', 'say', 'said',
            'know', 'knew', 'think', 'thought', 'see', 'saw', 'want', 'like',
            'one', 'two', 'first', 'new', 'way', 'thing', 'things', 'really',
            'actually', 'basically', 'simply', 'however', 'therefore', 'thus',
        }
        
        content_words = [w for w in words_lower if w not in function_words and len(w) > 2]
        num_content = len(content_words)
        info_density = num_content / total_words if total_words > 0 else 0
        
        # Unique content word ratio (penalize repetition)
        unique_content = len(set(content_words))
        content_uniqueness = unique_content / max(num_content, 1)
        
        # ---- 2. Sentence Length Consistency ----
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 2:
            mean_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sent_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sent_len = math.sqrt(variance)
            # Coefficient of variation - lower is more consistent
            cv = std_sent_len / max(mean_sent_len, 1)
            # Ideal sentence length: 10-25 words
            sent_len_score = 1.0 - min(cv, 1.0) * 0.5
            # Penalize very long or very short average sentences
            if mean_sent_len < 5:
                sent_len_score *= 0.7
            elif mean_sent_len > 35:
                sent_len_score *= 0.75
            elif 10 <= mean_sent_len <= 25:
                sent_len_score *= 1.1
        else:
            mean_sent_len = total_words
            sent_len_score = 0.6 if 10 <= mean_sent_len <= 30 else 0.4
        
        # ---- 3. Filler / Weasel / Vague Words ----
        filler_phrases = [
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bto be honest\b', r'\bto be fair\b', r'\bin my opinion\b',
            r'\bi think that\b', r'\bi believe that\b', r'\bi feel like\b',
            r'\bit seems like\b', r'\bit appears that\b', r'\bas far as i know\b',
            r'\bif you will\b', r'\bso to speak\b', r'\bin a sense\b',
            r'\bin some ways\b', r'\bfor what it\'s worth\b',
            r'\byou know\b', r'\bi mean\b', r'\blike i said\b',
            r'\bas i mentioned\b', r'\bas i said\b', r'\bneedless to say\b',
            r'\bit goes without saying\b', r'\bat the end of the day\b',
            r'\ball things considered\b', r'\bwhen all is said and done\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / max(num_sentences, 1)
        filler_penalty = max(0, 1.0 - filler_ratio * 0.3)
        
        # ---- 4. Specificity Score ----
        # Count numbers, capitalized words (proper nouns), quoted terms, code blocks
        num_numbers = len(re.findall(r'\b\d+\.?\d*\b', response))
        num_proper = len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', response))
        num_code_blocks = len(re.findall(r'```', response))
        num_quotes = len(re.findall(r'["\*]', response))
        has_urls = len(re.findall(r'https?://', response))
        
        # Specificity per 100 words
        specificity_raw = (num_numbers * 2 + num_proper * 1.0 + num_code_blocks * 3 + 
                          num_quotes * 0.3 + has_urls * 2)
        specificity_per_100 = specificity_raw / max(total_words, 1) * 100
        specificity_score = min(specificity_per_100 / 15, 1.0)  # Cap at 1.0
        
        # ---- 5. Clause Complexity Estimation ----
        # Count subordinate clause indicators
        subordinators = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhose\b',
            r'\bwhere\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bwhereas\b', r'\bbecause\b', r'\bsince\b', r'\bunless\b',
        ]
        clause_count = 0
        for pattern in subordinators:
            clause_count += len(re.findall(pattern, response_lower))
        
        # Commas as clause separator proxy
        num_commas = response.count(',')
        num_semicolons = response.count(';')
        num_colons = response.count(':')
        
        # Clauses per sentence - moderate complexity is ideal (1-3 clauses)
        clause_per_sent = (clause_count + num_commas * 0.5 + num_semicolons + num_colons * 0.5) / max(num_sentences, 1)
        if clause_per_sent < 0.5:
            clause_score = 0.6  # Too simple
        elif clause_per_sent <= 3.5:
            clause_score = 1.0  # Good complexity
        elif clause_per_sent <= 5:
            clause_score = 0.8  # Getting complex
        else:
            clause_score = 0.5  # Overly complex
        
        # ---- 6. Query Engagement ----
        # Extract key terms from query
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_content = set(w for w in query_words if w not in function_words and len(w) > 2)
        response_content_set = set(content_words)
        
        if query_content:
            query_coverage = len(query_content & response_content_set) / len(query_content)
        else:
            query_coverage = 0.5
        
        # ---- 7. Structural Organization ----
        # Check for structured elements (lists, formatting)
        has_bullets = bool(re.search(r'(?m)^[\s]*[-*•]\s', response))
        has_numbered = bool(re.search(r'(?m)^[\s]*\d+[.)]\s', response))
        has_headers = bool(re.search(r'(?m)^#+\s', response))
        has_paragraphs = response.count('\n\n') >= 1
        has_code = bool(re.search(r'```', response))
        
        structure_score = 0.5
        if has_bullets or has_numbered:
            structure_score += 0.2
        if has_headers:
            structure_score += 0.1
        if has_paragraphs:
            structure_score += 0.1
        if has_code:
            structure_score += 0.1
        structure_score = min(structure_score, 1.0)
        
        # ---- 8. Compression / Efficiency ----
        # Characters per content word (lower = more efficient encoding)
        total_chars = len(response_stripped)
        chars_per_content_word = total_chars / max(num_content, 1)
        # Ideal: 6-12 chars per content word
        if chars_per_content_word < 6:
            efficiency_score = 0.7
        elif chars_per_content_word <= 14:
            efficiency_score = 1.0
        elif chars_per_content_word <= 20:
            efficiency_score = 0.8
        else:
            efficiency_score = 0.5
        
        # ---- 9. Repetition Detection (beyond simple word counting) ----
        # Check for repeated 3-word sequences (trigrams of content words)
        if len(content_words) >= 6:
            trigrams = [' '.join(content_words[i:i+3]) for i in range(len(content_words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            repetition_penalty = max(0, 1.0 - repetition_ratio * 2.0)
        else:
            repetition_penalty = 1.0
        
        # ---- 10. Response Length Appropriateness ----
        # Not too short, not too long relative to query complexity
        query_len = len(query.split())
        
        # Base expectation: responses should have substance
        if total_words < 15:
            length_score = 0.4
        elif total_words < 30:
            length_score = 0.65
        elif total_words < 50:
            length_score = 0.8
        elif total_words <= 300:
            length_score = 1.0
        elif total_words <= 500:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ---- 11. Opening Directness ----
        # Penalize responses that start with meta-commentary instead of content
        opening_penalties = [
            r'^(sure|okay|ok|well|so|alright|great question|good question)',
            r'^(that\'s a great|that\'s an interesting|that\'s a good)',
            r'^(welcome to|please read|this is a)',
            r'^(i would say|i\'d say|i guess)',
        ]
        
        first_50 = response_lower[:100]
        opening_score = 1.0
        for pattern in opening_penalties:
            if re.search(pattern, first_50):
                opening_score = 0.85
                break
        
        # Check if response is mostly a redirect/meta rather than substantive
        redirect_patterns = [
            r'you might be interested in', r'check out', r'see this link',
            r'welcome to', r'please read our rules',
        ]
        is_redirect = False
        for pattern in redirect_patterns:
            if re.search(pattern, response_lower):
                is_redirect = True
                break
        
        if is_redirect and total_words < 40:
            opening_score *= 0.6
        
        # ---- 12. Assertiveness / Confidence ----
        # Count hedging vs assertive language
        hedge_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems',
                       'apparently', 'arguably', 'presumably', 'supposedly']
        hedge_count = sum(1 for w in words_lower if w in hedge_words)
        hedge_ratio = hedge_count / max(total_words, 1)
        assertiveness = max(0.5, 1.0 - hedge_ratio * 5)
        
        # ---- COMBINE SCORES ----
        # Weighted combination
        weights = {
            'info_density': 12,
            'content_uniqueness': 8,
            'sent_len': 8,
            'filler': 7,
            'specificity': 10,
            'clause': 5,
            'query_coverage': 10,
            'structure': 6,
            'efficiency': 5,
            'repetition': 8,
            'length': 10,
            'opening': 5,
            'assertiveness': 6,
        }
        
        scores = {
            'info_density': min(info_density / 0.5, 1.0),  # Normalize: 0.5 = perfect
            'content_uniqueness': content_uniqueness,
            'sent_len': sent_len_score,
            'filler': filler_penalty,
            'specificity': specificity_score,
            'clause': clause_score,
            'query_coverage': query_coverage,
            'structure': structure_score,
            'efficiency': efficiency_score,
            'repetition': repetition_penalty,
            'length': length_score,
            'opening': opening_score,
            'assertiveness': assertiveness,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        final_score = (weighted_sum / total_weight) * 10  # Scale to 0-10
        
        # Clamp
        final_score = max(0.5, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0