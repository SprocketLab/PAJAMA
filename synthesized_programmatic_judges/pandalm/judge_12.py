def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique approach focused on:
    - Sentence structure variety (std dev of sentence lengths)
    - Punctuation sophistication (use of commas, semicolons, colons, parentheses)
    - Word repetition penalty (penalize excessive repeated words/phrases)
    - Hapax legomena ratio (words appearing exactly once - indicates vocabulary richness)
    - Clause complexity (approximated by subordinating conjunctions and relative pronouns)
    - Capitalization correctness
    - Response completeness (not cut off mid-sentence)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        # Tokenize
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", response)
        word_count = len(words)
        
        if word_count == 0:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # 1. Sentence Length Variety Score (std dev of word counts per sentence)
        # ============================================================
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) > 1:
            mean_swc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_swc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_swc = math.sqrt(variance)
            # Ideal std dev is around 5-10 words variation
            variety_score = min(std_swc / 8.0, 1.0) * 10
        else:
            variety_score = 2.0  # Single sentence gets low variety score
        
        # ============================================================
        # 2. Punctuation Sophistication Score
        # ============================================================
        comma_count = response.count(',')
        semicolon_count = response.count(';')
        colon_count = response.count(':')
        paren_count = response.count('(') + response.count(')')
        dash_count = response.count('—') + response.count('–') + response.count(' - ')
        
        # Normalize by word count
        punct_density = (comma_count * 1.0 + semicolon_count * 2.0 + colon_count * 1.5 + 
                        paren_count * 1.5 + dash_count * 1.5) / max(word_count, 1)
        # Ideal punct density around 0.05-0.15
        if punct_density < 0.01:
            punct_score = 2.0
        elif punct_density <= 0.20:
            punct_score = 5.0 + 5.0 * min(punct_density / 0.10, 1.0)
        else:
            punct_score = max(5.0, 10.0 - (punct_density - 0.20) * 20)
        
        # ============================================================
        # 3. Word Repetition Penalty (bigram and unigram repetition)
        # ============================================================
        lower_words = [w.lower() for w in words]
        word_freq = Counter(lower_words)
        
        # Exclude common function words from repetition penalty
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them',
            'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
            'i', 'me', 'my', 'also', 'which', 'who', 'whom', 'whose', 'what',
            'if', 'then', 'because', 'while', 'although', 'though', 'when',
            'where', 'how', 'about', 'up', 'out', 'just', 'like'
        }
        
        content_words = [w for w in lower_words if w not in function_words and len(w) > 2]
        content_freq = Counter(content_words)
        
        if content_words:
            # Calculate max repetition ratio for any content word
            max_rep = max(content_freq.values()) if content_freq else 1
            total_content = len(content_words)
            rep_ratio = max_rep / max(total_content, 1)
            
            # Also check bigram repetition
            bigrams = [(lower_words[i], lower_words[i+1]) for i in range(len(lower_words)-1)]
            bigram_freq = Counter(bigrams)
            # Filter out bigrams that are all function words
            content_bigrams = {k: v for k, v in bigram_freq.items() 
                             if k[0] not in function_words or k[1] not in function_words}
            max_bigram_rep = max(content_bigrams.values()) if content_bigrams else 1
            bigram_rep_ratio = max_bigram_rep / max(len(bigrams), 1)
            
            # Severe penalty for high repetition
            if rep_ratio > 0.3 or max_rep > 5:
                repetition_penalty = -15.0
            elif rep_ratio > 0.15 or max_rep > 3:
                repetition_penalty = -8.0
            elif bigram_rep_ratio > 0.1:
                repetition_penalty = -6.0
            else:
                repetition_penalty = 0.0
        else:
            repetition_penalty = 0.0
        
        # ============================================================
        # 4. Hapax Legomena Ratio (words appearing exactly once)
        # ============================================================
        if word_count >= 5:
            hapax = sum(1 for w, c in word_freq.items() if c == 1 and w not in function_words)
            unique_content = len([w for w in word_freq if w not in function_words])
            hapax_ratio = hapax / max(unique_content, 1)
            # Higher hapax ratio = richer vocabulary
            hapax_score = hapax_ratio * 10.0
        else:
            hapax_score = 3.0
        
        # ============================================================
        # 5. Clause Complexity (subordinating conjunctions, relative pronouns)
        # ============================================================
        complexity_markers = [
            'although', 'because', 'since', 'while', 'whereas', 'unless',
            'whenever', 'wherever', 'whether', 'though', 'even though',
            'in order to', 'so that', 'provided that', 'as long as',
            'which', 'whom', 'whose', 'whereby', 'furthermore', 'moreover',
            'nevertheless', 'consequently', 'therefore', 'additionally',
            'specifically', 'particularly', 'essentially', 'fundamentally',
            'however', 'nonetheless', 'accordingly', 'subsequently'
        ]
        
        response_lower = response.lower()
        complexity_count = sum(1 for marker in complexity_markers if marker in response_lower)
        complexity_density = complexity_count / max(num_sentences, 1)
        complexity_score = min(complexity_density * 5.0, 10.0)
        
        # ============================================================
        # 6. Capitalization and Basic Grammar Correctness
        # ============================================================
        cap_score = 10.0
        
        # Check sentence-initial capitalization
        for s in sentences:
            s = s.strip()
            if s and s[0].isalpha() and not s[0].isupper():
                cap_score -= 2.0
        
        # Check for ALL CAPS words (excluding acronyms of 2-4 letters)
        all_caps_words = [w for w in words if w.isupper() and len(w) > 4]
        if all_caps_words:
            cap_score -= min(len(all_caps_words) * 1.5, 5.0)
        
        cap_score = max(cap_score, 0.0)
        
        # ============================================================
        # 7. Completeness Score (does response end properly?)
        # ============================================================
        completeness_score = 8.0
        
        # Check if response is cut off (ends mid-word or mid-sentence)
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?"\')':
            completeness_score -= 4.0
        
        # Check for very abrupt endings
        if stripped and len(stripped) > 20:
            last_20 = stripped[-20:]
            if last_20.count(' ') < 2:
                completeness_score -= 2.0
        
        # Empty or near-empty response
        if word_count < 3:
            completeness_score = 1.0
        
        # ============================================================
        # 8. Response Length Appropriateness
        # ============================================================
        # Longer, more detailed responses tend to be better (up to a point)
        if word_count < 5:
            length_score = 1.0
        elif word_count < 15:
            length_score = 3.0
        elif word_count < 30:
            length_score = 5.0
        elif word_count < 60:
            length_score = 7.0
        elif word_count < 150:
            length_score = 9.0
        elif word_count < 300:
            length_score = 8.0
        else:
            length_score = 7.0
        
        # ============================================================
        # 9. Average Word Length (proxy for vocabulary sophistication)
        # ============================================================
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        # Ideal average word length is around 5-6 characters
        if avg_word_len < 3.5:
            word_len_score = 3.0
        elif avg_word_len < 4.5:
            word_len_score = 6.0
        elif avg_word_len <= 6.5:
            word_len_score = 9.0
        elif avg_word_len <= 8.0:
            word_len_score = 7.0
        else:
            word_len_score = 5.0
        
        # ============================================================
        # 10. Unique trigram ratio (penalize formulaic/repetitive text)
        # ============================================================
        if len(lower_words) >= 3:
            trigrams = [(lower_words[i], lower_words[i+1], lower_words[i+2]) 
                       for i in range(len(lower_words)-2)]
            unique_trigrams = len(set(trigrams))
            total_trigrams = len(trigrams)
            trigram_uniqueness = unique_trigrams / max(total_trigrams, 1)
            trigram_score = trigram_uniqueness * 10.0
        else:
            trigram_score = 5.0
        
        # ============================================================
        # Combine scores with weights
        # ============================================================
        final_score = (
            variety_score * 0.10 +        # Sentence variety
            punct_score * 0.08 +           # Punctuation sophistication
            repetition_penalty * 0.15 +    # Repetition penalty (can be negative)
            hapax_score * 0.12 +           # Vocabulary richness
            complexity_score * 0.10 +      # Clause complexity
            cap_score * 0.08 +             # Capitalization correctness
            completeness_score * 0.12 +    # Completeness
            length_score * 0.10 +          # Length appropriateness
            word_len_score * 0.08 +        # Word sophistication
            trigram_score * 0.07           # Trigram uniqueness
        )
        
        # Normalize to 0-100 range
        final_score = max(0.0, min(final_score * 10.0, 100.0))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a minimal score based on response length
        try:
            if response and len(response.strip()) > 0:
                return min(len(response.strip()) / 10.0, 30.0)
            return 0.0
        except Exception:
            return 0.0