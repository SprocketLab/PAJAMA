def judging_function(query, response):
    """
    Evaluate language quality and readability using a combination of:
    - Punctuation correctness and variety
    - Sentence structure analysis (clause detection, sentence type variety)
    - Repetition penalty (repeated phrases/lines)
    - Character-level noise detection (HTML tags, code artifacts, special chars)
    - Coherence via connective density and paragraph structure
    - Hapax legomena ratio (words appearing only once, indicating vocabulary richness)
    - Word length distribution variance (indicating vocabulary variety)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 2:
            return 0.0
        
        # ---- Feature 1: Content substance ----
        # Penalize extremely short responses
        word_tokens = re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text)
        num_words = len(word_tokens)
        
        if num_words == 0:
            return 0.0
        
        # Length score: logarithmic scaling, sweet spot around 30-200 words
        if num_words < 3:
            length_score = 0.5
        elif num_words <= 200:
            length_score = min(10.0, 2.0 + 4.0 * math.log(num_words, 2) / math.log(200, 2))
        else:
            # Slight penalty for very long responses (might be rambling)
            length_score = max(4.0, 6.0 - 0.5 * math.log(num_words / 200, 2))
        
        # ---- Feature 2: Noise / garbage detection ----
        noise_score = 10.0
        
        # HTML tags
        html_tags = re.findall(r'<[^>]+>', text)
        html_ratio = len(''.join(html_tags)) / max(len(text), 1)
        if html_ratio > 0.1:
            noise_score -= min(4.0, html_ratio * 20)
        
        # Code-like patterns
        code_patterns = re.findall(r'(?:import |def |class |#include|function\s|var\s|let\s|const\s)', text)
        # Only penalize if query doesn't seem to ask for code
        query_lower = (query or "").lower()
        asks_for_code = any(w in query_lower for w in ['code', 'program', 'function', 'script', 'html', 'python', 'javascript'])
        if not asks_for_code and len(code_patterns) > 2:
            noise_score -= min(3.0, len(code_patterns) * 0.5)
        
        # Excessive special characters
        special_chars = sum(1 for c in text if c in '{}[]|\\<>@#$%^&*~`')
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio > 0.05:
            noise_score -= min(3.0, special_ratio * 30)
        
        noise_score = max(0.0, noise_score)
        
        # ---- Feature 3: Repetition penalty ----
        repetition_score = 10.0
        
        # Line-level repetition
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counter = Counter(lines)
            repeated_lines = sum(c - 1 for c in line_counter.values() if c > 1)
            line_rep_ratio = repeated_lines / max(len(lines), 1)
            repetition_score -= min(5.0, line_rep_ratio * 15)
        
        # Trigram repetition
        lower_words = [w.lower() for w in word_tokens]
        if len(lower_words) >= 3:
            trigrams = [tuple(lower_words[i:i+3]) for i in range(len(lower_words) - 2)]
            trigram_counter = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counter.values() if c > 1)
            trigram_rep_ratio = repeated_trigrams / max(total_trigrams, 1)
            repetition_score -= min(4.0, trigram_rep_ratio * 12)
        
        # "Output:" or "Input:" or "Question:" repetition (common in bad responses)
        prompt_labels = re.findall(r'(?:Output|Input|Question|Answer)\s*:', text)
        if len(prompt_labels) > 3:
            repetition_score -= min(3.0, (len(prompt_labels) - 2) * 0.7)
        
        repetition_score = max(0.0, repetition_score)
        
        # ---- Feature 4: Punctuation quality and sentence structure ----
        punct_score = 5.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        num_sentences = max(len(sentences), 1)
        
        # Check for proper capitalization at sentence starts
        if num_sentences > 1:
            capitalized = sum(1 for s in sentences if s and s[0].isupper())
            cap_ratio = capitalized / num_sentences
            punct_score += cap_ratio * 2.0
        elif num_sentences == 1 and sentences[0][0].isupper():
            punct_score += 1.5
        
        # Punctuation variety (using . , ! ? ; : -)
        punct_types = set()
        for c in text:
            if c in '.!?,;:-':
                punct_types.add(c)
        punct_score += min(2.0, len(punct_types) * 0.4)
        
        # Sentence length variance (good writing has varied sentence lengths)
        if num_sentences >= 3:
            sent_lengths = [len(re.findall(r'\S+', s)) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            if mean_sl > 0:
                variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
                cv = math.sqrt(variance) / mean_sl  # coefficient of variation
                # Good variety: cv around 0.3-0.7
                if 0.2 <= cv <= 0.8:
                    punct_score += 1.5
                elif cv > 0.1:
                    punct_score += 0.5
        
        punct_score = min(10.0, max(0.0, punct_score))
        
        # ---- Feature 5: Hapax legomena ratio & word length distribution ----
        vocab_score = 5.0
        
        if num_words >= 5:
            word_freq = Counter(w.lower() for w in word_tokens)
            hapax = sum(1 for w, c in word_freq.items() if c == 1)
            hapax_ratio = hapax / max(len(word_freq), 1)
            # Higher hapax ratio = richer vocabulary (for moderate-length texts)
            vocab_score += hapax_ratio * 3.0
            
            # Word length distribution: variance indicates diverse vocabulary
            word_lengths = [len(w) for w in word_tokens]
            mean_wl = sum(word_lengths) / len(word_lengths)
            wl_var = sum((l - mean_wl) ** 2 for l in word_lengths) / len(word_lengths)
            # Good variety: variance around 3-8
            if wl_var >= 2:
                vocab_score += min(2.0, wl_var * 0.3)
        
        vocab_score = min(10.0, max(0.0, vocab_score))
        
        # ---- Feature 6: Connective/coherence density ----
        coherence_score = 5.0
        
        connectives = [
            r'\bhowever\b', r'\btherefore\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bon the other hand\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bnevertheless\b', r'\balthough\b', r'\bwhile\b', r'\bsince\b',
            r'\bbecause\b', r'\bthus\b', r'\bhence\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bincluding\b',
            r'\bthis\b', r'\bthese\b', r'\bthat\b', r'\bwhich\b',
            r'\balso\b', r'\bbut\b', r'\byet\b', r'\bso\b',
        ]
        
        text_lower = text.lower()
        connective_count = 0
        unique_connectives = set()
        for pattern in connectives:
            matches = re.findall(pattern, text_lower)
            if matches:
                connective_count += len(matches)
                unique_connectives.add(pattern)
        
        if num_words > 10:
            conn_density = connective_count / num_words
            # Good density: 0.03-0.10
            if 0.02 <= conn_density <= 0.12:
                coherence_score += min(3.0, len(unique_connectives) * 0.4)
            elif conn_density > 0.0:
                coherence_score += min(1.5, len(unique_connectives) * 0.2)
        
        # Paragraph structure (having multiple paragraphs is good for longer texts)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if num_words > 50 and len(paragraphs) > 1:
            coherence_score += min(1.5, len(paragraphs) * 0.3)
        
        coherence_score = min(10.0, max(0.0, coherence_score))
        
        # ---- Feature 7: Completeness heuristic ----
        completeness_score = 8.0
        
        # Check if response ends mid-sentence (truncated)
        stripped = text.rstrip()
        if stripped and stripped[-1] not in '.!?"\')]:':
            # Might be truncated
            last_sentence = sentences[-1] if sentences else ""
            last_words = re.findall(r'\S+', last_sentence)
            if len(last_words) > 3:
                completeness_score -= 2.0
        
        # Check if starts properly (not mid-sentence)
        if text and text[0].islower() and not text.startswith('http'):
            completeness_score -= 1.0
        
        # Very short responses for non-trivial queries
        query_words = len(re.findall(r'\S+', query or ""))
        if query_words > 10 and num_words < 5:
            completeness_score -= 3.0
        
        completeness_score = max(0.0, min(10.0, completeness_score))
        
        # ---- Combine scores with weights ----
        # Different weighting than other variants
        final_score = (
            0.10 * length_score +
            0.20 * noise_score +
            0.20 * repetition_score +
            0.15 * punct_score +
            0.10 * vocab_score +
            0.10 * coherence_score +
            0.15 * completeness_score
        )
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        # Never crash
        try:
            # Minimal fallback: score based on length
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except Exception:
            return 1.0