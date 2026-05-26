def judging_function(query, response):
    """
    Evaluate language quality and readability using a combination of:
    - Punctuation correctness and variety
    - Sentence structure analysis (clause detection, sentence length variance)
    - Repetition penalty (n-gram repetition detection)
    - Character-level entropy for information density
    - Capitalization correctness
    - Response completeness heuristics
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not response.strip():
            return 0.0
        
        text = response.strip()
        
        # ---- Feature 1: Response length adequacy ----
        char_count = len(text)
        word_tokens = text.split()
        num_words = len(word_tokens)
        
        if num_words == 0:
            return 0.0
        
        # Very short responses are likely low quality
        if num_words <= 2:
            length_score = 0.5
        elif num_words <= 5:
            length_score = 2.0
        elif num_words <= 15:
            length_score = 4.0
        elif num_words <= 50:
            length_score = 6.0
        elif num_words <= 200:
            length_score = 7.0
        else:
            length_score = 6.5  # Very long can be rambling
        
        # ---- Feature 2: Sentence structure and variance ----
        # Split into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Sentence length variance (good writing has varied sentence lengths)
        sent_word_counts = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_word_counts) / len(sent_word_counts) if sent_word_counts else 0
        
        if len(sent_word_counts) > 1:
            mean_sl = avg_sent_len
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            sent_len_std = math.sqrt(variance)
            # Normalize: some variance is good, too much is bad
            variance_score = min(sent_len_std / 5.0, 1.0) * 5.0  # 0-5 range
        else:
            variance_score = 1.0  # Single sentence gets low variance score
        
        # Penalize very long or very short average sentence length
        if avg_sent_len < 3:
            avg_len_score = 1.0
        elif avg_sent_len < 8:
            avg_len_score = 3.0
        elif avg_sent_len <= 25:
            avg_len_score = 5.0
        elif avg_sent_len <= 40:
            avg_len_score = 3.5
        else:
            avg_len_score = 2.0
        
        # ---- Feature 3: Capitalization correctness ----
        cap_correct = 0
        cap_total = 0
        for s in sentences:
            s_stripped = s.lstrip()
            if s_stripped:
                cap_total += 1
                # Check if first character is uppercase or a special char
                first_char = s_stripped[0]
                if first_char.isupper() or not first_char.isalpha():
                    cap_correct += 1
        
        cap_score = (cap_correct / cap_total * 5.0) if cap_total > 0 else 2.5
        
        # ---- Feature 4: Punctuation variety and correctness ----
        punct_chars = re.findall(r'[.,;:!?\-\'"()\[\]]', text)
        num_punct = len(punct_chars)
        punct_types = len(set(punct_chars))
        
        # Punctuation density (per word)
        punct_density = num_punct / num_words if num_words > 0 else 0
        
        # Good range: 0.05 to 0.3 punctuation per word
        if 0.05 <= punct_density <= 0.3:
            punct_density_score = 5.0
        elif punct_density < 0.05:
            punct_density_score = 2.0
        else:
            punct_density_score = 3.0
        
        # Variety of punctuation
        punct_variety_score = min(punct_types / 4.0, 1.0) * 5.0
        
        # ---- Feature 5: N-gram repetition penalty ----
        # Detect repetitive content using bigrams and trigrams
        words_lower = [w.lower().strip('.,!?;:"\'-()[]') for w in word_tokens]
        words_lower = [w for w in words_lower if w]
        
        def ngram_repetition_ratio(words, n):
            if len(words) < n:
                return 0.0
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                return 0.0
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            return repeated / len(ngrams)
        
        bigram_rep = ngram_repetition_ratio(words_lower, 2)
        trigram_rep = ngram_repetition_ratio(words_lower, 3)
        
        # Also check for repeated sentences
        sent_lower = [s.lower().strip() for s in sentences]
        sent_counts = Counter(sent_lower)
        repeated_sents = sum(c - 1 for c in sent_counts.values() if c > 1)
        sent_rep_ratio = repeated_sents / num_sentences if num_sentences > 0 else 0
        
        # Repetition penalty: higher repetition = lower score
        rep_penalty = (bigram_rep * 3.0 + trigram_rep * 5.0 + sent_rep_ratio * 7.0)
        rep_score = max(5.0 - rep_penalty, 0.0)
        
        # ---- Feature 6: Character-level entropy (information density) ----
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        
        # Good English text typically has entropy around 4.0-4.5
        # Very low entropy = repetitive, very high = random/code
        if 3.5 <= char_entropy <= 5.0:
            entropy_score = 5.0
        elif 3.0 <= char_entropy < 3.5:
            entropy_score = 3.5
        elif char_entropy < 3.0:
            entropy_score = 2.0
        else:
            entropy_score = 3.0
        
        # ---- Feature 7: Detect garbage/code/HTML content ----
        # Check for excessive HTML tags, code patterns
        html_tags = len(re.findall(r'<[^>]+>', text))
        code_patterns = len(re.findall(r'(import |def |class |print\(|return |if __name__|#include)', text))
        
        garbage_penalty = 0.0
        if html_tags > 2:
            garbage_penalty += min(html_tags * 0.5, 3.0)
        if code_patterns > 1:
            garbage_penalty += min(code_patterns * 0.5, 3.0)
        
        # Check for excessive special characters
        special_chars = len(re.findall(r'[{}|\\<>@#$%^&*_~`]', text))
        special_ratio = special_chars / char_count if char_count > 0 else 0
        if special_ratio > 0.05:
            garbage_penalty += min(special_ratio * 20, 3.0)
        
        # ---- Feature 8: Truncation detection ----
        truncation_penalty = 0.0
        # Check if response ends mid-sentence (no terminal punctuation)
        last_char = text.rstrip()[-1] if text.rstrip() else ''
        if last_char not in '.!?"\')]:' and num_words > 10:
            truncation_penalty = 1.5
        
        # ---- Feature 9: Coherent word patterns ----
        # Check for words that are actually words (have vowels)
        vowel_pattern = re.compile(r'[aeiouAEIOU]')
        real_word_count = sum(1 for w in words_lower if vowel_pattern.search(w) or len(w) <= 2)
        real_word_ratio = real_word_count / len(words_lower) if words_lower else 0
        word_quality_score = real_word_ratio * 5.0
        
        # ---- Feature 10: Clause and connector detection ----
        # Different from transition words - look for subordinate clauses, relative pronouns
        clause_markers = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when', 
                         'while', 'although', 'because', 'since', 'unless', 'if',
                         'whether', 'after', 'before', 'until', 'as', 'though']
        
        words_set = set(words_lower)
        clause_count = sum(1 for m in clause_markers if m in words_set)
        clause_score = min(clause_count / 3.0, 1.0) * 3.0  # 0-3
        
        # ---- Combine scores ----
        # Weighted combination
        raw_score = (
            length_score * 0.20 +
            variance_score * 0.05 +
            avg_len_score * 0.10 +
            cap_score * 0.08 +
            punct_density_score * 0.05 +
            punct_variety_score * 0.05 +
            rep_score * 0.15 +
            entropy_score * 0.10 +
            word_quality_score * 0.07 +
            clause_score * 0.05
        )
        
        # Apply penalties
        raw_score -= garbage_penalty * 0.3
        raw_score -= truncation_penalty * 0.15
        
        # Scale to 0-10 range
        # Current max theoretical: ~5.5 (weighted sum of max components)
        # Scale up
        final_score = (raw_score / 5.5) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, final_score))
        
        # Extra boost for responses that are well-formed and substantial
        if num_words >= 20 and cap_score >= 4.0 and rep_score >= 3.0 and garbage_penalty == 0:
            final_score = min(10.0, final_score + 0.5)
        
        # Extra penalty for single-word or very terse responses
        if num_words <= 3:
            final_score = min(final_score, 3.0)
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            if not response or not response.strip():
                return 0.0
            words = response.strip().split()
            if len(words) <= 2:
                return 1.0
            elif len(words) <= 10:
                return 3.0
            elif len(words) <= 50:
                return 5.0
            else:
                return 6.0
        except Exception:
            return 2.0