def judging_function(query, response):
    """
    Evaluates language quality and readability of an LLM response.
    Uses a combination of:
    - Flesch-like readability scoring
    - Type-token ratio (vocabulary richness)
    - Sentence variety (length variance)
    - Grammar/punctuation heuristics
    - Structural quality indicators
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        import re
        import math
        import string
        from collections import Counter
        
        # ---- Helper functions ----
        
        def count_syllables(word):
            """Estimate syllable count for a word."""
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
        
        def split_sentences(text):
            """Split text into sentences."""
            # Split on sentence-ending punctuation followed by space or end
            sents = re.split(r'(?<=[.!?])\s+', text)
            # Filter out very short fragments
            return [s.strip() for s in sents if len(s.strip()) > 2]
        
        def get_words(text):
            """Extract words from text."""
            return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
        
        # ---- Extract basic components ----
        
        sentences = split_sentences(response)
        num_sentences = max(1, len(sentences))
        
        words = get_words(response)
        num_words = max(1, len(words))
        
        words_lower = [w.lower() for w in words]
        
        # ---- 1. Flesch Reading Ease (modified) ----
        # Standard: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        # We want moderate readability (not too simple, not too complex)
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / num_words
        avg_words_per_sentence = num_words / num_sentences
        
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        # Flesch ranges roughly 0-100; ideal for general audience is 60-70
        # We want to reward scores in the 30-70 range (informative but accessible)
        # Map to 0-15 points
        if flesch_score < 0:
            flesch_score = 0
        if flesch_score > 100:
            flesch_score = 100
        
        # Optimal readability around 40-65 for informative content
        if 35 <= flesch_score <= 70:
            readability_points = 15.0
        elif 20 <= flesch_score < 35 or 70 < flesch_score <= 80:
            readability_points = 12.0
        elif 10 <= flesch_score < 20 or 80 < flesch_score <= 90:
            readability_points = 9.0
        else:
            readability_points = 5.0
        
        # ---- 2. Type-Token Ratio (vocabulary richness) ----
        # Higher TTR = more diverse vocabulary
        
        unique_words = set(words_lower)
        num_unique = len(unique_words)
        
        # Use root TTR to normalize for text length: unique / sqrt(total)
        if num_words > 0:
            root_ttr = num_unique / math.sqrt(num_words)
        else:
            root_ttr = 0
        
        # Root TTR typically ranges 3-10 for normal text
        # Map to 0-15 points
        ttr_points = min(15.0, max(0.0, (root_ttr - 2.0) * 2.5))
        
        # ---- 3. Sentence variety (length variance) ----
        # Good writing has varied sentence lengths
        
        sent_lengths = [len(get_words(s)) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(1, mean_len)
            # Good variety: CV around 0.3-0.7
            if 0.25 <= cv <= 0.8:
                variety_points = 12.0
            elif 0.15 <= cv < 0.25 or 0.8 < cv <= 1.0:
                variety_points = 9.0
            elif cv < 0.15:
                variety_points = 5.0  # Too uniform
            else:
                variety_points = 6.0  # Too erratic
        else:
            variety_points = 3.0  # Only one sentence
        
        # ---- 4. Structural quality indicators ----
        
        structure_points = 0.0
        
        # Check for markdown formatting (headers, bold, lists)
        has_headers = bool(re.search(r'#{1,4}\s+\S', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_numbered_list = bool(re.search(r'\n\s*\d+[\.\)]\s+', response))
        has_bullet_list = bool(re.search(r'\n\s*[-*]\s+', response))
        
        if has_headers:
            structure_points += 3.0
        if has_bold:
            structure_points += 2.0
        if has_numbered_list:
            structure_points += 2.5
        if has_bullet_list:
            structure_points += 2.0
        
        # Paragraphing: multiple paragraphs indicate good structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 3:
            structure_points += 3.0
        elif len(paragraphs) >= 2:
            structure_points += 1.5
        
        structure_points = min(12.0, structure_points)
        
        # ---- 5. Grammar and punctuation heuristics ----
        
        grammar_points = 10.0  # Start with full points, deduct for issues
        
        # Check for proper capitalization at sentence starts
        cap_violations = 0
        for s in sentences:
            s_stripped = s.lstrip('#* \t-')
            if s_stripped and s_stripped[0].isalpha() and not s_stripped[0].isupper():
                cap_violations += 1
        if num_sentences > 0:
            cap_ratio = cap_violations / num_sentences
            grammar_points -= cap_ratio * 4.0
        
        # Check for repeated spaces
        double_spaces = len(re.findall(r'  +', response))
        if double_spaces > 3:
            grammar_points -= min(2.0, double_spaces * 0.3)
        
        # Check for common grammar issues
        # Double periods
        if '..' in response and '...' not in response:
            grammar_points -= 1.0
        
        # Missing space after punctuation
        missing_space = len(re.findall(r'[.!?,;:][a-zA-Z]', response))
        # Exclude common patterns like URLs, abbreviations
        if missing_space > 2:
            grammar_points -= min(2.0, missing_space * 0.3)
        
        # Reward proper use of punctuation variety
        punct_types = set()
        for ch in response:
            if ch in '.!?,;:-':
                punct_types.add(ch)
        if len(punct_types) >= 4:
            grammar_points += 1.5
        elif len(punct_types) >= 3:
            grammar_points += 0.5
        
        grammar_points = max(0.0, min(12.0, grammar_points))
        
        # ---- 6. Engagement and tone quality ----
        
        engagement_points = 0.0
        
        # Opening quality: starts with a direct, engaging statement
        first_line = response.split('\n')[0].strip()
        first_words = get_words(first_line)
        
        # Conversational/engaging openers
        engaging_starters = ['certainly', 'great', 'awesome', 'absolutely', 'excellent',
                            'interesting', 'wonderful', 'fantastic', 'good', 'sure',
                            'the art', 'organizing', 'traveling', 'that']
        if first_words:
            first_word_lower = first_words[0].lower()
            for starter in engaging_starters:
                if first_word_lower == starter or first_line.lower().startswith(starter):
                    engagement_points += 2.0
                    break
        
        # Check for transitional words/phrases (indicates flow)
        transitions = ['however', 'furthermore', 'moreover', 'additionally', 'in contrast',
                       'therefore', 'consequently', 'meanwhile', 'nevertheless', 'specifically',
                       'for example', 'in addition', 'on the other hand', 'as a result',
                       'first', 'second', 'third', 'finally', 'also', 'next', 'then',
                       'here are', 'let\'s', 'this means', 'in other words']
        
        response_lower = response.lower()
        transition_count = sum(1 for t in transitions if t in response_lower)
        engagement_points += min(4.0, transition_count * 0.6)
        
        # Reward appropriate response length relative to query
        query_words = len(get_words(query)) if query else 1
        response_to_query_ratio = num_words / max(1, query_words)
        if 3.0 <= response_to_query_ratio <= 20.0:
            engagement_points += 2.0
        elif 1.5 <= response_to_query_ratio < 3.0 or 20.0 < response_to_query_ratio <= 30.0:
            engagement_points += 1.0
        
        engagement_points = min(10.0, engagement_points)
        
        # ---- 7. Word sophistication ----
        
        sophistication_points = 0.0
        
        # Average word length (proxy for vocabulary sophistication)
        avg_word_len = sum(len(w) for w in words) / num_words if num_words > 0 else 0
        # Good informative text: avg word length 4.5-6.5
        if 4.5 <= avg_word_len <= 6.5:
            sophistication_points += 5.0
        elif 4.0 <= avg_word_len < 4.5 or 6.5 < avg_word_len <= 7.5:
            sophistication_points += 3.5
        elif 3.5 <= avg_word_len < 4.0:
            sophistication_points += 2.0
        else:
            sophistication_points += 1.0
        
        # Proportion of longer words (6+ chars)
        long_words = [w for w in words if len(w) >= 6]
        long_word_ratio = len(long_words) / num_words if num_words > 0 else 0
        if 0.2 <= long_word_ratio <= 0.45:
            sophistication_points += 3.0
        elif 0.15 <= long_word_ratio < 0.2 or 0.45 < long_word_ratio <= 0.55:
            sophistication_points += 2.0
        else:
            sophistication_points += 1.0
        
        sophistication_points = min(8.0, sophistication_points)
        
        # ---- 8. Coherence heuristic ----
        # Check if response relates to the query
        
        coherence_points = 0.0
        if query:
            query_words_set = set(w.lower() for w in get_words(query) if len(w) > 3)
            response_words_set = set(words_lower)
            if query_words_set:
                overlap = len(query_words_set & response_words_set) / len(query_words_set)
                coherence_points = min(6.0, overlap * 8.0)
            else:
                coherence_points = 3.0
        else:
            coherence_points = 3.0
        
        # ---- 9. Penalize truncation ----
        truncation_penalty = 0.0
        # Check if response appears cut off (no ending punctuation, ends mid-sentence)
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        if last_char not in '.!?:"\')]}':
            truncation_penalty = 3.0
        # Check for incomplete last sentence (very short last segment)
        if sentences:
            last_sent = sentences[-1]
            last_sent_words = get_words(last_sent)
            if len(last_sent_words) < 3 and last_char not in '.!?':
                truncation_penalty += 1.5
        
        # ---- Combine all scores ----
        
        total = (
            readability_points +      # max 15
            ttr_points +              # max 15
            variety_points +          # max 12
            structure_points +        # max 12
            grammar_points +          # max 12
            engagement_points +       # max 10
            sophistication_points +   # max 8
            coherence_points -        # max 6
            truncation_penalty        # max -4.5
        )
        
        # Normalize to 0-100 scale (theoretical max ~90)
        total = max(0.0, min(100.0, total * 1.11))
        
        return round(total, 2)
    
    except Exception:
        # Fallback: return a minimal score based on length
        try:
            return min(20.0, len(str(response)) / 50.0)
        except Exception:
            return 0.0