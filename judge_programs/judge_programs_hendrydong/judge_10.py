def judging_function(query, response):
    """
    Evaluates language quality and readability using:
    - Type-Token Ratio (vocabulary richness)
    - Punctuation diversity and correctness
    - Sentence variety (std dev of sentence lengths)
    - Connective/transition word usage
    - Capitalization correctness
    - Word sophistication (longer words ratio)
    - Balanced paragraph structure
    - Avoidance of repetition (bigram uniqueness)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        # --- Tokenization ---
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", text)
        if len(words) < 2:
            return 1.0
        
        word_count = len(words)
        words_lower = [w.lower() for w in words]
        
        # --- Sentences ---
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        num_sentences = max(len(sentences), 1)
        
        # =============================================
        # FEATURE 1: Type-Token Ratio (vocabulary richness)
        # Using root TTR to normalize for length: unique / sqrt(total)
        # =============================================
        unique_words = set(words_lower)
        root_ttr = len(unique_words) / math.sqrt(word_count)
        # Typical range: 3-10, normalize to 0-10
        ttr_score = min(root_ttr / 8.0, 1.0) * 10
        
        # =============================================
        # FEATURE 2: Sentence length variety (coefficient of variation)
        # =============================================
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 2:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance)
            cv = std_sl / max(mean_sl, 1)
            # Good variety: cv around 0.3-0.7
            variety_score = min(cv / 0.5, 1.0) * 10
        else:
            variety_score = 3.0  # Single sentence gets moderate score
        
        # Penalize very short average sentence length or very long
        avg_sent_len = sum(sent_word_counts) / max(len(sent_word_counts), 1) if sent_word_counts else word_count
        if avg_sent_len < 5:
            sent_len_penalty = -2.0
        elif avg_sent_len > 50:
            sent_len_penalty = -1.5
        elif 10 <= avg_sent_len <= 25:
            sent_len_penalty = 1.5
        else:
            sent_len_penalty = 0.0
        
        # =============================================
        # FEATURE 3: Connective/transition words (discourse coherence)
        # =============================================
        connectives = {
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'nevertheless', 'meanwhile', 'although', 'whereas',
            'specifically', 'essentially', 'particularly', 'notably', 'importantly',
            'similarly', 'conversely', 'alternatively', 'accordingly', 'thus',
            'hence', 'indeed', 'overall', 'ultimately', 'typically',
            'for example', 'in addition', 'on the other hand', 'in contrast',
            'as a result', 'in fact', 'that said', 'in other words',
            'first', 'second', 'third', 'finally', 'also', 'while',
            'though', 'yet', 'still', 'besides', 'likewise', 'instead'
        }
        
        text_lower = text.lower()
        connective_count = 0
        for conn in connectives:
            connective_count += len(re.findall(r'\b' + re.escape(conn) + r'\b', text_lower))
        
        conn_density = connective_count / max(num_sentences, 1)
        # Ideal: ~0.3-1.0 connectives per sentence
        conn_score = min(conn_density / 0.6, 1.0) * 8
        
        # =============================================
        # FEATURE 4: Punctuation diversity and usage
        # =============================================
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-—()[]"\'':
                punct_types.add(ch)
        
        # Count specific punctuation
        commas = text.count(',')
        semicolons = text.count(';')
        colons = text.count(':')
        parens = text.count('(') + text.count(')')
        dashes = text.count('—') + text.count('--') + text.count(' - ')
        
        punct_diversity = len(punct_types)
        # Normalize: 0-8 types mapped to 0-10
        punct_div_score = min(punct_diversity / 6.0, 1.0) * 7
        
        # Comma usage rate (good writing uses commas appropriately)
        comma_rate = commas / max(num_sentences, 1)
        comma_score = min(comma_rate / 1.5, 1.0) * 3
        
        punct_score = punct_div_score + comma_score
        
        # =============================================
        # FEATURE 5: Word sophistication (proportion of longer words)
        # Using character length >= 7 as "sophisticated"
        # =============================================
        long_words = [w for w in words_lower if len(w) >= 7]
        long_word_ratio = len(long_words) / max(word_count, 1)
        # Typical good range: 0.15-0.35
        sophistication_score = min(long_word_ratio / 0.25, 1.0) * 8
        
        # Also measure average word length
        avg_word_len = sum(len(w) for w in words_lower) / max(word_count, 1)
        # Good range: 4.5-6.0
        if 4.5 <= avg_word_len <= 6.0:
            awl_bonus = 2.0
        elif 4.0 <= avg_word_len <= 6.5:
            awl_bonus = 1.0
        else:
            awl_bonus = 0.0
        
        sophistication_score += awl_bonus
        
        # =============================================
        # FEATURE 6: Bigram repetition (lower = more varied expression)
        # =============================================
        if len(words_lower) >= 3:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            unique_bigrams = len(bigram_counts)
            bigram_uniqueness = unique_bigrams / max(total_bigrams, 1)
            # Higher uniqueness = less repetition = better
            repetition_score = bigram_uniqueness * 8
        else:
            repetition_score = 4.0
        
        # =============================================
        # FEATURE 7: Capitalization correctness
        # =============================================
        cap_errors = 0
        # Check if sentences start with capital letters
        for s in sentences:
            s_stripped = s.lstrip()
            if s_stripped and s_stripped[0].isalpha() and s_stripped[0].islower():
                cap_errors += 1
        
        # Check for random mid-sentence capitalizations (rough heuristic)
        # Words that are capitalized but not at sentence start, not "I", not acronyms
        all_caps_words = [w for w in words if w.isupper() and len(w) > 2]
        
        cap_error_rate = cap_errors / max(num_sentences, 1)
        cap_score = max(0, (1 - cap_error_rate)) * 5
        
        # =============================================
        # FEATURE 8: Spelling heuristic - unusual character patterns
        # =============================================
        # Look for double-letter anomalies and unusual patterns
        misspell_patterns = [
            r'[a-z]{4,}[A-Z]',  # camelCase mid-word
            r'([a-z])\1{2,}',    # triple+ letters like "soooo"
            r'\b[a-z]{1}\b',     # single letter words (except 'a', 'i')
        ]
        
        spelling_deductions = 0
        for pattern in misspell_patterns:
            matches = re.findall(pattern, text)
            spelling_deductions += len(matches) * 0.3
        
        # Check for common misspelling indicators
        # Words with unusual consonant clusters
        spelling_score = max(0, 5 - spelling_deductions)
        
        # =============================================
        # FEATURE 9: Response completeness and structure
        # =============================================
        # Penalize truncated responses (ending mid-word or mid-sentence)
        truncation_penalty = 0
        if text[-1] not in '.!?"\')]}:' and not text.endswith('...'):
            truncation_penalty = -2.0
        
        # Bonus for having multiple well-formed sentences
        if num_sentences >= 3:
            multi_sent_bonus = 2.0
        elif num_sentences >= 2:
            multi_sent_bonus = 1.0
        else:
            multi_sent_bonus = 0.0
        
        # =============================================
        # FEATURE 10: Sentence starter diversity
        # =============================================
        if len(sentences) >= 2:
            starters = []
            for s in sentences:
                s_words = re.findall(r"[a-zA-Z']+", s)
                if s_words:
                    starters.append(s_words[0].lower())
            if starters:
                unique_starters = len(set(starters))
                starter_diversity = unique_starters / len(starters)
                starter_score = starter_diversity * 5
            else:
                starter_score = 2.5
        else:
            starter_score = 2.5
        
        # =============================================
        # FEATURE 11: Content density - ratio of content words to function words
        # =============================================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'so', 'if', 'that', 'this', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom'
        }
        
        content_words = [w for w in words_lower if w not in function_words]
        content_ratio = len(content_words) / max(word_count, 1)
        # Good range: 0.45-0.65
        content_score = min(content_ratio / 0.55, 1.0) * 5
        
        # =============================================
        # AGGREGATE SCORE
        # =============================================
        raw_score = (
            ttr_score * 1.2 +          # Vocabulary richness (max ~12)
            variety_score * 0.8 +       # Sentence variety (max ~8)
            conn_score * 1.0 +          # Discourse coherence (max ~8)
            punct_score * 0.7 +         # Punctuation (max ~7)
            sophistication_score * 1.0 + # Word sophistication (max ~10)
            repetition_score * 0.6 +    # Low repetition (max ~4.8)
            cap_score * 0.5 +           # Capitalization (max ~2.5)
            spelling_score * 0.4 +      # Spelling (max ~2)
            starter_score * 0.6 +       # Sentence starter diversity (max ~3)
            content_score * 0.5 +       # Content density (max ~2.5)
            sent_len_penalty +          # Sentence length bonus/penalty
            truncation_penalty +        # Truncation penalty
            multi_sent_bonus            # Multi-sentence bonus
        )
        
        # Length bonus: longer, well-formed responses tend to be better
        # But diminishing returns
        length_factor = math.log(max(word_count, 1) + 1) / math.log(300)
        length_bonus = min(length_factor, 1.0) * 5
        raw_score += length_bonus
        
        # Normalize to 0-100 range
        # Theoretical max is roughly: 12+8+8+7+10+4.8+2.5+2+3+2.5+1.5+0+2+5 = ~68.3
        # Practical range: 10-60
        final_score = max(0, min(100, raw_score * 1.4))
        
        return round(final_score, 2)
    
    except Exception:
        return 5.0