def judging_function(query, response):
    """
    Evaluate language quality and readability of an LLM response.
    Uses a combination of:
    - Flesch-like readability scoring
    - Type-token ratio (vocabulary richness)
    - Grammar/punctuation heuristics
    - Sentence variety analysis
    - Penalty for repetition, artifacts, and incoherence
    
    Returns a score from 0-10 where higher is better.
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        # Edge cases
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        # Very short responses (< 5 chars) are almost always bad
        if len(response) < 5:
            return 0.5
        
        # ============================================================
        # 1. SYLLABLE COUNTING
        # ============================================================
        def count_syllables(word):
            word = word.lower().strip()
            if len(word) <= 2:
                return 1
            # Remove trailing 'e'
            if word.endswith('e') and len(word) > 3:
                word = word[:-1]
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)
        
        # ============================================================
        # 2. BASIC TEXT STATISTICS
        # ============================================================
        # Split into sentences
        sentence_endings = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 1]
        num_sentences = max(1, len(sentences))
        
        # Extract words (alphabetic tokens)
        words = re.findall(r"[a-zA-Z']+", response)
        num_words = len(words)
        
        if num_words == 0:
            return 0.5
        
        # Very short responses
        if num_words < 3:
            return 1.0
        
        words_lower = [w.lower() for w in words]
        
        total_syllables = sum(count_syllables(w) for w in words_lower)
        avg_syllables_per_word = total_syllables / num_words
        avg_words_per_sentence = num_words / num_sentences
        
        # ============================================================
        # 3. FLESCH READING EASE (modified)
        # ============================================================
        # Standard: 206.835 - 1.015 * ASL - 84.6 * ASW
        # We want moderate readability (score ~60-70 is ideal)
        flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        # Normalize: ideal range 40-80, map to 0-1
        if flesch > 100:
            flesch_score = 0.7  # very simple, slightly penalize
        elif flesch >= 30:
            flesch_score = 1.0 - abs(flesch - 60) / 100.0
        else:
            flesch_score = max(0.2, flesch / 100.0)
        flesch_score = max(0.0, min(1.0, flesch_score))
        
        # ============================================================
        # 4. TYPE-TOKEN RATIO (vocabulary richness)
        # ============================================================
        unique_words = set(words_lower)
        # Use root TTR to handle length dependency
        if num_words > 0:
            ttr = len(unique_words) / math.sqrt(num_words)
        else:
            ttr = 0
        # Normalize: typical good TTR (root) is around 4-8
        ttr_score = min(1.0, ttr / 7.0)
        
        # ============================================================
        # 5. SENTENCE LENGTH VARIETY
        # ============================================================
        if num_sentences >= 2:
            sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
            sent_lengths = [sl for sl in sent_lengths if sl > 0]
            if len(sent_lengths) >= 2:
                mean_sl = sum(sent_lengths) / len(sent_lengths)
                variance = sum((sl - mean_sl) ** 2 for sl in sent_lengths) / len(sent_lengths)
                std_sl = math.sqrt(variance)
                # Coefficient of variation
                cv = std_sl / max(1, mean_sl)
                variety_score = min(1.0, cv / 0.5)  # cv of 0.5 is good variety
            else:
                variety_score = 0.3
        else:
            # Single sentence - penalize slightly for lack of variety but don't kill score
            variety_score = 0.3
        
        # ============================================================
        # 6. PUNCTUATION QUALITY
        # ============================================================
        # Check for proper punctuation usage
        punct_chars = [c for c in response if c in '.,;:!?']
        punct_ratio = len(punct_chars) / max(1, num_words)
        # Good range: 0.05 - 0.25 punctuation per word
        if 0.03 <= punct_ratio <= 0.30:
            punct_score = 1.0
        elif punct_ratio < 0.03:
            punct_score = 0.4
        else:
            punct_score = max(0.2, 1.0 - (punct_ratio - 0.30) * 3)
        
        # Check if sentences start with capital letters
        capital_starts = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isupper():
                capital_starts += 1
        capital_ratio = capital_starts / max(1, num_sentences)
        capitalization_score = capital_ratio
        
        # ============================================================
        # 7. REPETITION PENALTY
        # ============================================================
        # Word-level repetition (bigram repetition)
        if num_words >= 4:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
            repetition_ratio = repeated_bigrams / max(1, total_bigrams)
            repetition_penalty = min(1.0, repetition_ratio * 3)
        else:
            repetition_penalty = 0.0
        
        # Sentence-level repetition
        sent_texts = [s.lower().strip() for s in sentences]
        sent_counter = Counter(sent_texts)
        duplicate_sents = sum(c - 1 for c in sent_counter.values() if c > 1)
        sent_rep_penalty = min(1.0, duplicate_sents / max(1, num_sentences))
        
        # Line/phrase repetition check
        lines = response.split('\n')
        lines_stripped = [l.strip().lower() for l in lines if l.strip()]
        if len(lines_stripped) > 2:
            line_counter = Counter(lines_stripped)
            dup_lines = sum(c - 1 for c in line_counter.values() if c > 1)
            line_rep_penalty = min(1.0, dup_lines / max(1, len(lines_stripped)))
        else:
            line_rep_penalty = 0.0
        
        total_rep_penalty = min(1.0, repetition_penalty + sent_rep_penalty * 0.5 + line_rep_penalty * 0.5)
        
        # ============================================================
        # 8. ARTIFACT / NOISE PENALTY
        # ============================================================
        artifact_penalty = 0.0
        
        # HTML tags
        html_tags = re.findall(r'<[^>]+>', response)
        if html_tags:
            artifact_penalty += min(0.5, len(html_tags) * 0.08)
        
        # Code artifacts (unless query asks for code)
        query_lower = query.lower() if query else ""
        is_code_query = any(kw in query_lower for kw in ['code', 'python', 'html', 'program', 'function', 'script', 'tag'])
        
        if not is_code_query:
            code_patterns = re.findall(r'(import |def |class |print\(|#include|```)', response)
            if code_patterns:
                artifact_penalty += min(0.5, len(code_patterns) * 0.1)
        
        # Excessive special characters
        special_chars = sum(1 for c in response if c in '#$%^&*{}[]|\\~`')
        special_ratio = special_chars / max(1, len(response))
        if special_ratio > 0.03:
            artifact_penalty += min(0.3, special_ratio * 5)
        
        # Random/broken text patterns
        # Consecutive same characters
        triple_chars = re.findall(r'(.)\1{2,}', response)
        if triple_chars:
            artifact_penalty += min(0.2, len(triple_chars) * 0.05)
        
        # "Input:" "Output:" patterns suggesting template artifacts
        template_patterns = re.findall(r'(?:Input:|Output:|Question:|Answer:)', response)
        if len(template_patterns) > 2:
            artifact_penalty += min(0.4, (len(template_patterns) - 2) * 0.1)
        
        artifact_penalty = min(1.0, artifact_penalty)
        
        # ============================================================
        # 9. COHERENCE HEURISTIC (word length distribution)
        # ============================================================
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / num_words
        # Good average word length: 4-7
        if 3.5 <= avg_word_length <= 7.5:
            word_len_score = 1.0
        else:
            word_len_score = max(0.3, 1.0 - abs(avg_word_length - 5.5) / 5.0)
        
        # ============================================================
        # 10. LENGTH ADEQUACY
        # ============================================================
        # Responses that are extremely short tend to be low quality
        # But we shouldn't overly penalize concise correct answers
        if num_words < 5:
            length_score = 0.3
        elif num_words < 10:
            length_score = 0.5
        elif num_words < 20:
            length_score = 0.7
        elif num_words <= 300:
            length_score = 1.0
        else:
            # Very long responses might have issues but aren't necessarily bad
            length_score = 0.9
        
        # ============================================================
        # 11. COMPLETENESS CHECK
        # ============================================================
        # Check if response appears truncated (ends mid-word or mid-sentence)
        truncation_penalty = 0.0
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?"\')]:;':
            # Might be truncated
            # Check if the last word seems complete
            last_words = words[-3:] if len(words) >= 3 else words
            if stripped[-1].isalpha():
                truncation_penalty = 0.1
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        # Weighted combination
        base_score = (
            flesch_score * 1.5 +          # Readability
            ttr_score * 1.5 +              # Vocabulary richness
            variety_score * 1.0 +          # Sentence variety
            punct_score * 1.0 +            # Punctuation quality
            capitalization_score * 0.8 +   # Proper capitalization
            word_len_score * 0.7 +         # Word length distribution
            length_score * 1.5             # Adequate length
        )
        
        max_possible = 1.5 + 1.5 + 1.0 + 1.0 + 0.8 + 0.7 + 1.5  # = 8.0
        
        # Normalize to 0-10
        normalized = (base_score / max_possible) * 10.0
        
        # Apply penalties
        penalty_multiplier = max(0.1, 1.0 - total_rep_penalty * 0.5 - artifact_penalty * 0.5 - truncation_penalty)
        
        final_score = normalized * penalty_multiplier
        
        # Clamp to 0-10
        final_score = max(0.5, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except Exception:
            return 2.0