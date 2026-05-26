def judging_function(query, response):
    """
    Evaluates language quality and readability using a different approach:
    - Sentence structure variety (std dev of sentence lengths)
    - Punctuation diversity and density
    - Transition/cohesion word usage
    - Type-token ratio with hapax legomena ratio
    - Paragraph structure analysis
    - Formality and tone consistency measures
    - Avoidance of repetitive starts
    - Conjunction variety and complex sentence detection
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
        
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", response)
        if len(words) < 3:
            return 0.5
        
        # --- 1. Sentence structure variety ---
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        num_sentences = max(len(sentences), 1)
        
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sent_word_counts.append(len(sw))
        
        # Variety in sentence length (std dev) - moderate variety is good
        if len(sent_word_counts) > 1:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance)
            # Normalize: ideal std dev around 5-10
            variety_score = min(std_sl / 8.0, 1.0) * 10
        else:
            mean_sl = sent_word_counts[0] if sent_word_counts else 10
            variety_score = 2.0
        
        # --- 2. Sentence start diversity ---
        sentence_starts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sentence_starts.append(sw[0].lower())
        
        if len(sentence_starts) > 1:
            unique_starts = len(set(sentence_starts))
            start_diversity = unique_starts / len(sentence_starts)
            start_score = start_diversity * 10
        else:
            start_score = 5.0
        
        # --- 3. Punctuation diversity and density ---
        punct_chars = re.findall(r'[,;:\-\(\)\"\'!?\.]', response)
        punct_types = set(punct_chars)
        punct_diversity = len(punct_types)
        # Density: punctuation per 100 words
        punct_density = (len(punct_chars) / max(len(words), 1)) * 100
        
        # Good writing has moderate punctuation density (5-20 per 100 words)
        # and diverse punctuation types
        punct_div_score = min(punct_diversity / 5.0, 1.0) * 5
        if punct_density < 2:
            punct_dens_score = 2.0
        elif punct_density > 30:
            punct_dens_score = 3.0
        else:
            punct_dens_score = min(punct_density / 15.0, 1.0) * 5
        punct_score = punct_div_score + punct_dens_score
        
        # --- 4. Transition and cohesion words ---
        transition_words = {
            'however', 'moreover', 'furthermore', 'additionally', 'therefore',
            'consequently', 'nevertheless', 'meanwhile', 'although', 'though',
            'while', 'whereas', 'instead', 'otherwise', 'hence', 'thus',
            'likewise', 'similarly', 'conversely', 'accordingly', 'indeed',
            'certainly', 'notably', 'specifically', 'particularly', 'essentially',
            'importantly', 'significantly', 'ultimately', 'finally', 'firstly',
            'secondly', 'thirdly', 'also', 'besides', 'next', 'then',
            'afterwards', 'subsequently', 'naturally', 'clearly', 'obviously',
            'evidently', 'undoubtedly', 'perhaps', 'possibly', 'presumably'
        }
        
        words_lower = [w.lower() for w in words]
        transition_count = sum(1 for w in words_lower if w in transition_words)
        transition_ratio = transition_count / max(len(words), 1)
        # Ideal: 2-5% transition words
        transition_score = min(transition_ratio * 100 / 3.0, 1.0) * 10
        
        # --- 5. Hapax legomena ratio (words appearing exactly once) ---
        word_freq = Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        total_unique = len(word_freq)
        
        if total_unique > 0:
            hapax_ratio = hapax / total_unique
            # High hapax ratio = rich vocabulary
            hapax_score = hapax_ratio * 8
        else:
            hapax_score = 0
        
        # --- 6. Complex sentence detection (subordinate clauses) ---
        subordinators = {
            'because', 'since', 'although', 'though', 'while', 'whereas',
            'unless', 'until', 'whenever', 'wherever', 'whether', 'after',
            'before', 'once', 'if', 'when', 'that', 'which', 'who', 'whom',
            'whose', 'where'
        }
        
        subordinator_count = sum(1 for w in words_lower if w in subordinators)
        complex_ratio = subordinator_count / max(num_sentences, 1)
        # Ideal: 0.5-2 subordinators per sentence
        complex_score = min(complex_ratio / 1.0, 1.0) * 8
        
        # --- 7. Paragraph structure ---
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Longer responses benefit from paragraph breaks
        if len(words) > 80:
            if num_paragraphs >= 2:
                para_score = min(num_paragraphs / 3.0, 1.0) * 6
            else:
                para_score = 2.0
        else:
            para_score = 4.0  # Short responses don't need paragraphs
        
        # --- 8. Conjunction variety ---
        conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
        conj_used = set()
        for w in words_lower:
            if w in conjunctions:
                conj_used.add(w)
        conj_variety_score = min(len(conj_used) / 4.0, 1.0) * 5
        
        # --- 9. Empathy/engagement markers (relevant for conversational quality) ---
        empathy_phrases = [
            r"\bi (?:understand|see|hear|can see|can hear|can imagine)\b",
            r"\bit'?s (?:completely|totally|absolutely|perfectly|entirely) (?:understandable|okay|normal|fine|natural)\b",
            r"\byou(?:'re| are) (?:right|not alone|valued)\b",
            r"\blet'?s\b",
            r"\btogether\b",
            r"\bwe (?:can|will|should|could)\b",
            r"\bdon'?t (?:worry|hesitate)\b",
            r"\bfeel free\b",
            r"\bhere (?:to help|for you)\b",
        ]
        
        response_lower = response.lower()
        empathy_count = 0
        for pattern in empathy_phrases:
            empathy_count += len(re.findall(pattern, response_lower))
        empathy_score = min(empathy_count / 2.0, 1.0) * 5
        
        # --- 10. Avoidance of overly short or choppy sentences ---
        if sent_word_counts:
            very_short = sum(1 for c in sent_word_counts if c <= 4)
            choppy_ratio = very_short / len(sent_word_counts)
            choppiness_penalty = choppy_ratio * 5
        else:
            choppiness_penalty = 0
        
        # --- 11. Average word length (proxy for vocabulary sophistication) ---
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        # Ideal: 4.5-6.5
        if avg_word_len < 3.5:
            word_len_score = 2.0
        elif avg_word_len > 7.0:
            word_len_score = 4.0
        else:
            word_len_score = min((avg_word_len - 3.0) / 3.0, 1.0) * 6
        
        # --- 12. Response length adequacy ---
        word_count = len(words)
        if word_count < 20:
            length_score = 2.0
        elif word_count < 50:
            length_score = 4.0
        elif word_count < 150:
            length_score = 6.0
        elif word_count < 300:
            length_score = 5.5
        else:
            length_score = 4.5
        
        # --- 13. Repetition penalty (bigram repetition) ---
        if len(words_lower) > 2:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            bigram_freq = Counter(bigrams)
            repeated_bigrams = sum(1 for b, c in bigram_freq.items() if c > 2)
            repetition_penalty = min(repeated_bigrams / 3.0, 1.0) * 5
        else:
            repetition_penalty = 0
        
        # --- 14. Enumeration / structured content bonus ---
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        has_structure = numbered_items >= 2
        structure_bonus = 3.0 if has_structure else 0.0
        
        # --- Combine all scores ---
        raw_score = (
            variety_score * 1.0 +       # max 10
            start_score * 1.0 +          # max 10
            punct_score * 1.0 +          # max 10
            transition_score * 0.8 +     # max 8
            hapax_score * 0.8 +          # max ~6.4
            complex_score * 0.7 +        # max 5.6
            para_score * 0.6 +           # max 3.6
            conj_variety_score * 0.6 +   # max 3
            empathy_score * 0.7 +        # max 3.5
            word_len_score * 0.8 +       # max 4.8
            length_score * 0.8 +         # max 4.8
            structure_bonus * 0.5 +      # max 1.5
            - choppiness_penalty * 0.8 + # max penalty 4
            - repetition_penalty * 0.8   # max penalty 4
        )
        
        # Theoretical max is roughly ~70, map to 1-5 scale
        # Normalize
        normalized = (raw_score - 5) / 55.0  # roughly 0-1
        normalized = max(0.0, min(1.0, normalized))
        
        final_score = 1.0 + normalized * 4.0  # 1-5 scale
        
        # Round to 1 decimal
        final_score = round(final_score, 2)
        
        return final_score
        
    except Exception:
        return 2.5