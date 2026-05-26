def judging_function(query, response):
    """
    Evaluate language quality using a combination of:
    - Punctuation correctness and density
    - Capitalization patterns (proper sentence starts)
    - Repetition detection (duplicate phrases/sentences)
    - Lexical sophistication (word length distribution, rare long words)
    - Coherence via sentence-to-sentence word overlap (cohesion)
    - Penalty for code/markup artifacts
    - Sentence completeness heuristics
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        text = response.strip()
        
        # Very short responses get low scores
        if len(text) < 5:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+", text)
        if len(words) == 0:
            return 0.5
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # ===== 1. Punctuation Density & Correctness =====
        # Good prose has reasonable punctuation density
        punctuation_chars = re.findall(r'[.,;:!?\-\'"()\[\]]', text)
        punct_density = len(punctuation_chars) / max(num_words, 1)
        # Ideal punct density around 0.1-0.3
        if punct_density < 0.02:
            punct_score = 2.0
        elif punct_density < 0.05:
            punct_score = 5.0
        elif punct_density <= 0.35:
            punct_score = 10.0
        elif punct_density <= 0.5:
            punct_score = 7.0
        else:
            punct_score = 3.0
        
        # ===== 2. Capitalization Patterns =====
        # Check if sentences start with capital letters
        cap_correct = 0
        raw_sentences = re.split(r'[.!?]+', text)
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
        for s in raw_sentences:
            if s and s[0].isupper():
                cap_correct += 1
        cap_ratio = cap_correct / max(len(raw_sentences), 1)
        cap_score = cap_ratio * 10.0
        
        # Also check for excessive ALL CAPS words (shouting)
        all_caps_words = [w for w in words if w.isupper() and len(w) > 2]
        caps_penalty = min(len(all_caps_words) / max(num_words, 1) * 50, 5.0)
        cap_score = max(cap_score - caps_penalty, 0)
        
        # ===== 3. Repetition Detection =====
        # Detect repeated sentences
        sentence_lower = [s.lower().strip() for s in sentences]
        sentence_counts = Counter(sentence_lower)
        repeated_sentences = sum(c - 1 for c in sentence_counts.values() if c > 1)
        sent_repeat_ratio = repeated_sentences / max(num_sentences, 1)
        
        # Detect repeated trigrams
        lower_words = [w.lower() for w in words]
        trigrams = [tuple(lower_words[i:i+3]) for i in range(len(lower_words) - 2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_repeat_ratio = repeated_trigrams / max(len(trigrams), 1)
        else:
            trigram_repeat_ratio = 0
        
        # Detect repeated bigrams
        bigrams = [tuple(lower_words[i:i+2]) for i in range(len(lower_words) - 1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
            bigram_repeat_ratio = repeated_bigrams / max(len(bigrams), 1)
        else:
            bigram_repeat_ratio = 0
        
        repetition_penalty = (sent_repeat_ratio * 5.0 + 
                              trigram_repeat_ratio * 3.0 + 
                              bigram_repeat_ratio * 1.5)
        repetition_score = max(10.0 - repetition_penalty * 10.0, 0)
        
        # ===== 4. Lexical Sophistication =====
        word_lengths = [len(w) for w in words]
        avg_word_len = sum(word_lengths) / max(num_words, 1)
        
        # Distribution of word lengths - good writing has variety
        length_distribution = Counter(word_lengths)
        # Proportion of "sophisticated" words (6+ letters)
        sophisticated_ratio = sum(1 for wl in word_lengths if wl >= 6) / max(num_words, 1)
        
        # Ideal avg word length is around 4.5-5.5
        if 4.0 <= avg_word_len <= 6.0:
            word_len_score = 10.0
        elif 3.0 <= avg_word_len < 4.0 or 6.0 < avg_word_len <= 7.0:
            word_len_score = 7.0
        else:
            word_len_score = 4.0
        
        # Vocabulary diversity via hapax legomena ratio
        word_freq = Counter(lower_words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(len(word_freq), 1)
        # Higher hapax ratio = more diverse vocabulary
        vocab_diversity_score = min(hapax_ratio * 12.0, 10.0)
        
        lexical_score = (word_len_score * 0.4 + 
                         sophisticated_ratio * 15.0 * 0.3 + 
                         vocab_diversity_score * 0.3)
        lexical_score = min(lexical_score, 10.0)
        
        # ===== 5. Sentence-to-Sentence Cohesion =====
        if len(sentences) >= 2:
            cohesion_scores = []
            for i in range(1, len(sentences)):
                words_prev = set(re.findall(r"[a-z']+", sentences[i-1].lower()))
                words_curr = set(re.findall(r"[a-z']+", sentences[i].lower()))
                # Remove very common words for more meaningful overlap
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                            'it', 'this', 'that', 'and', 'or', 'but', 'not', 'no', 'if'}
                words_prev_content = words_prev - stopwords
                words_curr_content = words_curr - stopwords
                if words_prev_content and words_curr_content:
                    overlap = len(words_prev_content & words_curr_content)
                    union = len(words_prev_content | words_curr_content)
                    cohesion_scores.append(overlap / max(union, 1))
                else:
                    cohesion_scores.append(0.1)
            avg_cohesion = sum(cohesion_scores) / len(cohesion_scores)
            # Some cohesion is good, too much means repetitive
            if 0.05 <= avg_cohesion <= 0.4:
                cohesion_score = 10.0
            elif avg_cohesion < 0.05:
                cohesion_score = 5.0
            else:
                cohesion_score = max(10.0 - (avg_cohesion - 0.4) * 15, 3.0)
        else:
            cohesion_score = 5.0  # Single sentence, neutral
        
        # ===== 6. Code/Markup Artifact Penalty =====
        code_patterns = [
            r'<[a-zA-Z/][^>]*>',  # HTML tags
            r'\{[^}]*\}',          # Curly braces (code blocks)
            r'import\s+\w+',       # Python imports
            r'def\s+\w+\s*\(',    # Function definitions
            r'class\s+\w+',        # Class definitions
            r'```',                 # Code fences
            r'#include',            # C includes
            r'return\s+\w+',       # Return statements
        ]
        code_matches = 0
        for pattern in code_patterns:
            code_matches += len(re.findall(pattern, text))
        
        # Ratio of code artifacts to total content
        code_ratio = code_matches / max(num_words / 10, 1)
        code_penalty = min(code_ratio * 3.0, 8.0)
        
        # ===== 7. Sentence Completeness =====
        # Check if the last sentence seems complete (ends with punctuation)
        text_stripped = text.rstrip()
        ends_with_punct = text_stripped[-1] in '.!?"\')' if text_stripped else False
        completeness_score = 8.0 if ends_with_punct else 4.0
        
        # Check for truncated words (ending with -)
        if text_stripped.endswith('-') or text_stripped.endswith('...'):
            completeness_score = max(completeness_score - 2.0, 1.0)
        
        # ===== 8. Response Substance =====
        # Penalize very short responses relative to query
        query_words = len(re.findall(r"[a-zA-Z']+", query)) if query else 1
        response_to_query_ratio = num_words / max(query_words, 1)
        
        if num_words < 3:
            substance_score = 1.0
        elif num_words < 8:
            substance_score = 3.0
        elif num_words < 15:
            substance_score = 5.0
        elif num_words < 30:
            substance_score = 7.0
        else:
            substance_score = 9.0
        
        # But also penalize extremely long responses that might be rambling
        if num_words > 300 and repetition_score < 5:
            substance_score = max(substance_score - 2.0, 3.0)
        
        # ===== 9. Character-level entropy (writing variety) =====
        char_freq = Counter(text.lower())
        total_chars = len(text)
        char_entropy = 0
        for ch, count in char_freq.items():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Normalize: good English text has entropy around 4.0-4.5
        if 3.5 <= char_entropy <= 5.0:
            entropy_score = 10.0
        elif 3.0 <= char_entropy < 3.5 or 5.0 < char_entropy <= 5.5:
            entropy_score = 7.0
        else:
            entropy_score = 4.0
        
        # ===== 10. Sentence Length Variance =====
        if len(sentences) >= 2:
            sent_word_counts = []
            for s in sentences:
                sw = re.findall(r"[a-zA-Z']+", s)
                sent_word_counts.append(len(sw))
            if sent_word_counts:
                mean_sl = sum(sent_word_counts) / len(sent_word_counts)
                variance_sl = sum((x - mean_sl)**2 for x in sent_word_counts) / len(sent_word_counts)
                std_sl = math.sqrt(variance_sl)
                cv_sl = std_sl / max(mean_sl, 1)
                # Some variation is good (0.2-0.6 CV)
                if 0.15 <= cv_sl <= 0.7:
                    variety_score = 10.0
                elif cv_sl < 0.15:
                    variety_score = 5.0  # Too uniform
                else:
                    variety_score = 6.0  # Too varied
            else:
                variety_score = 5.0
        else:
            variety_score = 5.0
        
        # ===== FINAL WEIGHTED COMBINATION =====
        final_score = (
            punct_score * 0.08 +
            cap_score * 0.10 +
            repetition_score * 0.15 +
            lexical_score * 0.10 +
            cohesion_score * 0.07 +
            completeness_score * 0.10 +
            substance_score * 0.18 +
            entropy_score * 0.07 +
            variety_score * 0.07 +
            (10.0 - code_penalty) * 0.08
        )
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middling score based on length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 2.0