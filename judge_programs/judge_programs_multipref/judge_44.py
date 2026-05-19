def judging_function(query, response):
    """
    Evaluates clarity and conciseness using information density, syntactic complexity,
    and directness metrics. Uses compression ratio concepts, syllable-based readability,
    and signal-to-noise ratio analysis.
    
    Different from other variants by focusing on:
    - Syllable-based readability (Flesch-like)
    - Information density via unique content words ratio
    - Sentence structure variance (penalize monotony AND excessive complexity)
    - Direct answer detection (does it address the query quickly?)
    - Filler/fluff phrase detection (specific multi-word patterns)
    - Parenthetical/qualifier density
    - Average clause length estimation
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        # === Helper: count syllables ===
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing silent e
            if word.endswith('e') and len(word) > 2:
                word = word[:-1]
            vowels = 'aeiou'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)
        
        # === Tokenization ===
        words = re.findall(r"[a-zA-Z']+", response)
        word_count = len(words)
        if word_count < 3:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        # Sentences (split on .!?;: followed by space or end)
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        sentence_count = max(1, len(sentences))
        
        # === 1. Syllable-based readability score ===
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / word_count
        avg_words_per_sentence = word_count / sentence_count
        
        # Flesch Reading Ease inspired (higher = easier to read = clearer)
        # Standard: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        # Normalize to 0-10 range (typical flesch: 0-100, we want moderate ~60-70 to score high)
        # Penalize both too simple (< 30) and too complex (> 90 might be too choppy)
        if flesch > 70:
            readability_score = 10.0 - (flesch - 70) * 0.05  # slight penalty for being too simple
        elif flesch >= 40:
            readability_score = 10.0 * (flesch - 40) / 30.0  # scale 40-70 to 0-10
        else:
            readability_score = max(0, flesch / 4.0)  # heavily penalize very complex
        readability_score = max(0, min(10, readability_score))
        
        # === 2. Information density: unique content words / total words ===
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'well', 'really', 'quite', 'much', 'many', 'any',
            'get', 'got', 'make', 'made', 'like', 'even', 'still', 'back',
        }
        
        content_words = [w for w in words_lower if w not in stop_words and len(w) > 2]
        content_word_count = len(content_words)
        unique_content = set(content_words)
        
        if content_word_count > 0:
            # Ratio of unique content words to total words
            info_density = len(unique_content) / word_count
            # Content word ratio (how much of text is actual content vs filler)
            content_ratio = content_word_count / word_count
        else:
            info_density = 0.0
            content_ratio = 0.0
        
        # Scale: higher info_density is better (more diverse vocabulary, less repetition)
        # Typical range: 0.1 - 0.5
        density_score = min(10, info_density * 25)
        content_score = min(10, content_ratio * 15)
        
        # === 3. Sentence length variance (penalize monotonous AND chaotic) ===
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) > 1:
            mean_swc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_swc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(1, mean_swc)
            # Ideal CV is moderate (0.3-0.6): varied but not chaotic
            if 0.25 <= cv <= 0.65:
                variance_score = 10.0
            elif cv < 0.25:
                variance_score = max(3, 10.0 * cv / 0.25)
            else:
                variance_score = max(2, 10.0 * (1.0 - (cv - 0.65) / 0.5))
        else:
            variance_score = 5.0  # single sentence, neutral
        
        variance_score = max(0, min(10, variance_score))
        
        # === 4. Directness: how quickly does the response address the query? ===
        # Check if key query terms appear in first sentence/paragraph
        query_words = set(re.findall(r"[a-zA-Z']+", query.lower()))
        query_content = query_words - stop_words
        query_content = {w for w in query_content if len(w) > 2}
        
        if query_content and sentences:
            first_chunk = ' '.join(sentences[:2]).lower() if len(sentences) > 1 else sentences[0].lower()
            first_chunk_words = set(re.findall(r"[a-zA-Z']+", first_chunk))
            overlap = len(query_content & first_chunk_words) / max(1, len(query_content))
            directness_score = min(10, overlap * 12)
        else:
            directness_score = 5.0
        
        # === 5. Fluff/filler multi-word phrase detection ===
        fluff_patterns = [
            r'\bit is important to (note|mention|remember|understand|realize)\b',
            r'\bit is worth (noting|mentioning)\b',
            r'\bas (mentioned|stated|noted) (earlier|above|before|previously)\b',
            r'\bin (other words|summary|conclusion|essence)\b',
            r'\bto be (honest|frank|fair)\b',
            r'\bat the end of the day\b',
            r'\bfor what it\'?s worth\b',
            r'\bneedless to say\b',
            r'\bit goes without saying\b',
            r'\ball things considered\b',
            r'\bwhen it comes to\b',
            r'\bin terms of\b',
            r'\bthe fact that\b',
            r'\bdue to the fact that\b',
            r'\bin order to\b',
            r'\bfor the purpose of\b',
            r'\bwith regard to\b',
            r'\bwith respect to\b',
            r'\bthat being said\b',
            r'\bhaving said that\b',
            r'\bit should be noted that\b',
            r'\bit can be said that\b',
            r'\bas a matter of fact\b',
            r'\bby and large\b',
            r'\bfirst and foremost\b',
            r'\blast but not least\b',
            r'\beach and every\b',
            r'\bany and all\b',
            r'\bbasically\b',
            r'\bessentially\b',
            r'\bfundamentally\b',
            r'\bgenerally speaking\b',
            r'\bin general\b',
            r'\boverall\b',
        ]
        
        response_lower = response.lower()
        fluff_count = 0
        for pattern in fluff_patterns:
            fluff_count += len(re.findall(pattern, response_lower))
        
        # Normalize by word count
        fluff_density = fluff_count / max(1, word_count / 100)
        fluff_penalty = min(5, fluff_density * 1.5)
        
        # === 6. Parenthetical/qualifier density ===
        # Count parentheses, em-dashes used as asides, excessive commas
        paren_count = response.count('(') + response.count(')')
        em_dash_count = response.count('—') + response.count(' - ') + response.count(' -- ')
        
        qualifier_density = (paren_count + em_dash_count) / max(1, sentence_count)
        qualifier_penalty = min(3, qualifier_density * 0.5)
        
        # === 7. Repetition detection (repeated content word sequences) ===
        # Check for repeated bigrams of content words
        if len(content_words) > 4:
            content_bigrams = [(content_words[i], content_words[i+1]) for i in range(len(content_words)-1)]
            bigram_counts = Counter(content_bigrams)
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
            repetition_penalty = min(4, repeated_bigrams * 0.8)
        else:
            repetition_penalty = 0
        
        # === 8. Structural clarity bonus ===
        # Reward for having clear structure (numbered lists, bold markers, etc.)
        has_numbering = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_bullets = bool(re.search(r'^\s*[-•*]\s', response, re.MULTILINE))
        has_headers = bool(re.search(r'^#{1,4}\s', response, re.MULTILINE))
        
        structure_bonus = 0
        if has_numbering:
            structure_bonus += 1.5
        if has_bold:
            structure_bonus += 1.0
        if has_bullets:
            structure_bonus += 0.8
        if has_headers:
            structure_bonus += 0.7
        structure_bonus = min(3, structure_bonus)
        
        # === 9. Opening quality: penalize generic/cliché openings ===
        generic_openings = [
            r"^(that'?s a )?great (question|idea|choice)",
            r"^(absolutely|certainly|definitely|sure)!?\s",
            r"^(oh,?\s)?(wow|awesome|amazing|wonderful|fantastic)",
            r"^(well,?\s)?(so,?\s)?let me",
            r"^a classic (question|problem|topic)",
            r"^i'?m (glad|happy|delighted) (you|to)",
        ]
        
        first_50 = response_lower[:80]
        opening_penalty = 0
        for pattern in generic_openings:
            if re.search(pattern, first_50):
                opening_penalty = 1.0
                break
        
        # === 10. Conciseness: penalize overly long responses relative to query complexity ===
        query_word_count = len(re.findall(r"[a-zA-Z']+", query))
        # Simple heuristic: very long responses to short queries might be bloated
        length_ratio = word_count / max(1, query_word_count)
        if length_ratio > 30:
            length_penalty = min(3, (length_ratio - 30) * 0.1)
        elif length_ratio < 2:
            length_penalty = 1.0  # too short might lack substance
        else:
            length_penalty = 0
        
        # === Combine scores ===
        # Weights chosen to emphasize clarity dimensions
        score = (
            readability_score * 0.15 +       # readability
            density_score * 0.18 +            # information density
            content_score * 0.12 +            # content word ratio
            variance_score * 0.10 +           # sentence variety
            directness_score * 0.15 +         # directness
            structure_bonus * 0.10 +          # structural clarity (0-3 scaled)
            5.0 * 0.20                        # base score component
            - fluff_penalty * 0.5             # fluff penalty
            - qualifier_penalty * 0.3         # qualifier penalty
            - repetition_penalty * 0.4        # repetition penalty
            - opening_penalty * 0.5           # generic opening penalty
            - length_penalty * 0.3            # length penalty
        )
        
        # Normalize to 0-10
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
        
    except Exception:
        return 5.0