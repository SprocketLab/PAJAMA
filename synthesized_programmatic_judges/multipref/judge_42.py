def judging_function(query, response):
    """
    Evaluates clarity and conciseness using a signal-to-noise ratio approach.
    
    This variant focuses on:
    1. Information density (ratio of content words to total words)
    2. Filler/hedge phrase detection and penalization
    3. Sentence-level clarity scoring (parse complexity proxies)
    4. Transition word usage (signals logical flow)
    5. Direct engagement with the query
    6. Redundancy detection via n-gram repetition across sentences
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
            return 0.5
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Filler and hedge phrase penalty
        # ============================================================
        filler_phrases = [
            r'\bbasically\b', r'\bactually\b', r'\bessentially\b',
            r'\bin other words\b', r'\bthat being said\b', r'\bit is important to note\b',
            r'\bit should be noted\b', r'\bit is worth mentioning\b',
            r'\bneedless to say\b', r'\bas a matter of fact\b',
            r'\bin terms of\b', r'\bat the end of the day\b',
            r'\bthe fact that\b', r'\bdue to the fact that\b',
            r'\bin order to\b', r'\bfor the purpose of\b',
            r'\bit goes without saying\b', r'\bas you may know\b',
            r'\bas we all know\b', r'\bquite\b', r'\breally\b',
            r'\bvery\b', r'\bjust\b', r'\bsort of\b', r'\bkind of\b',
            r'\bmore or less\b', r'\bto be honest\b',
            r'\bi think that\b', r'\bi believe that\b',
            r'\bit is clear that\b', r'\bit is obvious that\b',
            r'\bthere is no doubt\b', r'\bwithout a doubt\b',
            r'\bcan be a\b', r'\bcan really\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / total_words
        filler_score = max(0, 1.0 - filler_ratio * 15)  # penalize heavily
        
        # ============================================================
        # FEATURE 2: Information density via content word ratio
        # ============================================================
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'which', 'who', 'whom', 'also', 'like', 'get', 'got', 'much',
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        # Ideal content ratio is around 0.45-0.60
        info_density_score = min(1.0, content_ratio / 0.55)
        
        # ============================================================
        # FEATURE 3: Transition/connective word usage (signals logical flow)
        # ============================================================
        transitions = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bhowever\b', r'\btherefore\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bconsequently\b', r'\bthus\b', r'\bspecifically\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bin contrast\b',
            r'\bon the other hand\b', r'\bmeanwhile\b', r'\binstead\b',
            r'\balternatively\b', r'\bsimilarly\b', r'\blikewise\b',
            r'\bhere\b', r'\bstep\b',
        ]
        
        transition_count = 0
        for pattern in transitions:
            transition_count += len(re.findall(pattern, response_lower))
        
        # Ideal: roughly 1 transition per 3-5 sentences
        transition_per_sentence = transition_count / num_sentences
        if transition_per_sentence < 0.1:
            transition_score = 0.4
        elif transition_per_sentence <= 0.8:
            transition_score = 0.6 + 0.5 * transition_per_sentence
        else:
            transition_score = min(1.0, 1.0 - (transition_per_sentence - 0.8) * 0.2)
        
        # ============================================================
        # FEATURE 4: Redundancy detection via trigram overlap across sentences
        # ============================================================
        def get_ngrams(text, n):
            tokens = re.findall(r'[a-z]+', text.lower())
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        all_trigrams = []
        sentence_trigram_sets = []
        for sent in sentences:
            tris = get_ngrams(sent, 3)
            sentence_trigram_sets.append(set(tris))
            all_trigrams.extend(tris)
        
        # Count cross-sentence trigram repetitions
        redundant_pairs = 0
        total_pairs = 0
        for i in range(len(sentence_trigram_sets)):
            for j in range(i + 1, len(sentence_trigram_sets)):
                if sentence_trigram_sets[i] and sentence_trigram_sets[j]:
                    overlap = len(sentence_trigram_sets[i] & sentence_trigram_sets[j])
                    min_size = min(len(sentence_trigram_sets[i]), len(sentence_trigram_sets[j]))
                    if min_size > 0:
                        redundant_pairs += overlap / min_size
                        total_pairs += 1
        
        avg_redundancy = redundant_pairs / total_pairs if total_pairs > 0 else 0
        redundancy_score = max(0, 1.0 - avg_redundancy * 3)
        
        # ============================================================
        # FEATURE 5: Directness - how quickly the response addresses the query
        # ============================================================
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) - stop_words
        query_words = {w for w in query_words if len(w) > 2}
        
        if query_words and len(sentences) > 0:
            first_sent_words = set(re.findall(r'[a-z]+', sentences[0].lower()))
            first_overlap = len(query_words & first_sent_words) / max(len(query_words), 1)
            directness_score = 0.5 + 0.5 * min(1.0, first_overlap * 2)
        else:
            directness_score = 0.5
        
        # ============================================================
        # FEATURE 6: Sentence complexity variance (clarity proxy)
        # Lower variance in sentence word count = more consistent readability
        # ============================================================
        sent_word_counts = []
        for sent in sentences:
            wc = len(re.findall(r'[a-zA-Z]+', sent))
            if wc > 0:
                sent_word_counts.append(wc)
        
        if len(sent_word_counts) > 1:
            mean_wc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_wc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            cv = math.sqrt(variance) / mean_wc if mean_wc > 0 else 0
            # Some variation is good (0.3-0.5), too much or too little is bad
            if cv < 0.2:
                consistency_score = 0.7  # too monotonous
            elif cv < 0.6:
                consistency_score = 1.0  # good variety
            else:
                consistency_score = max(0.3, 1.0 - (cv - 0.6) * 1.5)
        else:
            consistency_score = 0.6
        
        # Penalize very long average sentences (harder to parse)
        if sent_word_counts:
            avg_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            if avg_sent_len > 30:
                sent_len_penalty = max(0.5, 1.0 - (avg_sent_len - 30) * 0.02)
            elif avg_sent_len < 5:
                sent_len_penalty = 0.6
            else:
                sent_len_penalty = 1.0
        else:
            sent_len_penalty = 0.7
        
        # ============================================================
        # FEATURE 7: Formatting signals (bold, numbered lists, headers)
        # These enhance clarity for instructional/explanatory content
        # ============================================================
        has_bold = len(re.findall(r'\*\*[^*]+\*\*', response)) > 0
        has_numbering = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)) > 0
        has_headers = len(re.findall(r'#{1,4}\s', response)) > 0
        
        # Check if query seems to ask for structured info
        structural_queries = ['how', 'steps', 'ways', 'ideas', 'tips', 'list', 'suggest', 'recipe', 'help']
        query_lower = query.lower()
        wants_structure = any(kw in query_lower for kw in structural_queries)
        
        formatting_score = 0.5
        if wants_structure:
            if has_numbering or has_bold or has_headers:
                formatting_score = 1.0
            else:
                formatting_score = 0.35
        else:
            if has_bold or has_headers:
                formatting_score = 0.8
            elif has_numbering:
                formatting_score = 0.7
        
        # ============================================================
        # FEATURE 8: Conciseness penalty for excessive preamble
        # ============================================================
        preamble_phrases = [
            r'^(that\'s a great|great question|what a|awesome|oh)',
            r'^(certainly|of course|sure|absolutely|definitely)',
            r'(let\'s dive|let me explain|i\'d be happy)',
            r'(glad you asked|interesting question)',
        ]
        
        first_50_chars = response_lower[:80]
        has_preamble = any(re.search(p, first_50_chars) for p in preamble_phrases)
        
        # Light penalty - preamble is ok if rest is good, but direct answers are better
        # Actually from examples, some good responses have preamble, so very light penalty
        preamble_score = 0.85 if has_preamble else 1.0
        
        # ============================================================
        # FEATURE 9: Passive voice ratio (active voice = clearer)
        # ============================================================
        passive_patterns = [
            r'\b(?:is|are|was|were|been|be|being)\s+(?:\w+ly\s+)?(?:\w+ed|made|done|given|taken|known|seen|found|shown)\b',
        ]
        passive_count = 0
        for p in passive_patterns:
            passive_count += len(re.findall(p, response_lower))
        
        passive_ratio = passive_count / num_sentences
        passive_score = max(0.5, 1.0 - passive_ratio * 0.3)
        
        # ============================================================
        # Combine all features with weights
        # ============================================================
        score = (
            filler_score * 0.15 +
            info_density_score * 0.15 +
            transition_score * 0.10 +
            redundancy_score * 0.15 +
            directness_score * 0.10 +
            consistency_score * 0.08 +
            sent_len_penalty * 0.07 +
            formatting_score * 0.10 +
            preamble_score * 0.05 +
            passive_score * 0.05
        )
        
        # Scale to 0-10
        final_score = score * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 5.0