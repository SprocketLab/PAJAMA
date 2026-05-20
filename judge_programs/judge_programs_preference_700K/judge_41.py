def judging_function(query, response):
    """
    Evaluate clarity and conciseness of an LLM response.
    Higher scores = better clarity and conciseness.
    
    This variant focuses on:
    - Sentence-level clarity metrics (avg sentence length, variation)
    - Information density (content words vs filler)
    - Structural organization signals
    - Redundancy detection via n-gram repetition
    - Readability approximation
    - Directness and engagement with the query
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 5:
            return 0.5
        
        # ============================================================
        # 1. RESPONSE LENGTH SCORING (moderate length preferred)
        # ============================================================
        resp_len = len(response)
        word_tokens = response.split()
        num_words = len(word_tokens)
        
        if num_words < 3:
            return 1.0
        
        # Ideal length relative to query
        query_words = len(query.split()) if query else 10
        
        # Prefer responses that are substantive but not bloated
        # Sweet spot: roughly 50-300 words for most queries
        if num_words < 15:
            length_score = 2.0
        elif num_words < 30:
            length_score = 4.0
        elif num_words < 80:
            length_score = 7.0
        elif num_words < 200:
            length_score = 8.0
        elif num_words < 400:
            length_score = 7.0
        elif num_words < 600:
            length_score = 6.0
        else:
            length_score = 5.0
        
        # ============================================================
        # 2. SENTENCE STRUCTURE ANALYSIS
        # ============================================================
        # Split into sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length in words
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        # Ideal average sentence length: 10-20 words
        if 8 <= avg_sent_len <= 22:
            sent_len_score = 8.0
        elif 5 <= avg_sent_len <= 30:
            sent_len_score = 6.0
        elif avg_sent_len < 5:
            sent_len_score = 3.0
        else:
            sent_len_score = 4.0
        
        # Sentence length variation (some variation is good, too much is bad)
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            coeff_var = std_dev / mean_sl if mean_sl > 0 else 0
            
            if 0.2 <= coeff_var <= 0.7:
                variation_score = 8.0
            elif coeff_var < 0.2:
                variation_score = 5.0  # Too monotonous
            else:
                variation_score = 4.0  # Too erratic
        else:
            variation_score = 5.0
        
        # ============================================================
        # 3. FILLER AND HEDGE WORD DETECTION
        # ============================================================
        lower_resp = response.lower()
        words_lower = lower_resp.split()
        
        filler_phrases = [
            'basically', 'essentially', 'actually', 'literally', 'honestly',
            'obviously', 'clearly', 'of course', 'needless to say',
            'it goes without saying', 'as you know', 'you know',
            'i mean', 'sort of', 'kind of', 'more or less',
            'in my opinion', 'i think that', 'i believe that',
            'it is important to note that', 'it should be noted that',
            'it is worth mentioning that', 'it is interesting to note',
            'as a matter of fact', 'the fact of the matter is',
            'at the end of the day', 'when all is said and done',
            'in terms of', 'with respect to', 'with regard to',
            'in order to',  # could just be "to"
            'due to the fact that',  # could be "because"
            'in the event that',  # could be "if"
            'for the purpose of',  # could be "to"
            'a lot of', 'very', 'really', 'quite', 'rather',
            'just', 'simply', 'merely',
        ]
        
        filler_count = 0
        for phrase in filler_phrases:
            filler_count += lower_resp.count(phrase)
        
        filler_ratio = filler_count / num_words if num_words > 0 else 0
        filler_score = max(0, 8.0 - filler_ratio * 80)
        
        # ============================================================
        # 4. REDUNDANCY / REPETITION DETECTION
        # ============================================================
        # Check for repeated trigrams
        if num_words >= 6:
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            
            # Some repetition of common phrases is normal
            redundancy_score = max(0, 8.0 - trigram_repetition_ratio * 40)
        else:
            redundancy_score = 5.0
        
        # Check for repeated bigrams more aggressively
        if num_words >= 4:
            bigrams = [' '.join(words_lower[i:i+2]) for i in range(len(words_lower) - 1)]
            # Filter out very common bigrams
            common_bigrams = {'of the', 'in the', 'to the', 'and the', 'on the', 'is a',
                            'it is', 'to be', 'for the', 'with the', 'that the', 'from the',
                            'at the', 'by the', 'as a', 'this is', 'if you', 'you can'}
            meaningful_bigrams = [b for b in bigrams if b not in common_bigrams]
            bigram_counts = Counter(meaningful_bigrams)
            high_repeat_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
            bigram_penalty = min(high_repeat_bigrams * 0.5, 4.0)
        else:
            bigram_penalty = 0
        
        redundancy_score = max(0, redundancy_score - bigram_penalty)
        
        # ============================================================
        # 5. VOCABULARY RICHNESS (Type-Token Ratio)
        # ============================================================
        # Clean words for vocabulary analysis
        clean_words = [w.strip(string.punctuation).lower() for w in word_tokens if w.strip(string.punctuation)]
        clean_words = [w for w in clean_words if w]
        
        if len(clean_words) > 5:
            # Use root TTR to normalize for length
            unique_words = len(set(clean_words))
            ttr = unique_words / math.sqrt(len(clean_words))
            
            # Normalize: typical values range from ~3 to ~10
            vocab_score = min(8.0, max(2.0, (ttr - 2) * 1.5))
        else:
            vocab_score = 4.0
        
        # ============================================================
        # 6. STRUCTURAL ORGANIZATION
        # ============================================================
        structure_score = 5.0
        
        # Check for organizational markers
        has_lists = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response) or 
                        re.search(r'(?:^|\n)\s*\d+[.)]\s', response))
        has_paragraphs = '\n\n' in response or response.count('\n') >= 2
        has_headers = bool(re.search(r'(?:^|\n)#+\s', response) or 
                          re.search(r'(?:^|\n)\*\*[^*]+\*\*', response))
        has_code = '```' in response
        
        if has_lists:
            structure_score += 1.0
        if has_paragraphs and num_words > 50:
            structure_score += 0.5
        if has_headers and num_words > 100:
            structure_score += 0.5
        if has_code:
            structure_score += 0.5
        
        structure_score = min(8.0, structure_score)
        
        # ============================================================
        # 7. QUERY RELEVANCE / ENGAGEMENT
        # ============================================================
        # Check if response engages with key terms from query
        query_content_words = set()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
                      'if', 'then', 'than', 'that', 'this', 'these', 'those',
                      'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
                      'it', 'they', 'them', 'their', 'what', 'which', 'who',
                      'whom', 'how', 'when', 'where', 'why', 'am', 'im', 'its'}
        
        if query:
            q_words = [w.strip(string.punctuation).lower() for w in query.split()]
            query_content_words = {w for w in q_words if w and len(w) > 2 and w not in stop_words}
        
        if query_content_words:
            resp_word_set = set(clean_words)
            overlap = query_content_words & resp_word_set
            relevance_ratio = len(overlap) / len(query_content_words) if query_content_words else 0
            relevance_score = 3.0 + relevance_ratio * 5.0  # 3-8 range
        else:
            relevance_score = 5.0
        
        # ============================================================
        # 8. DIRECTNESS / CONFIDENCE
        # ============================================================
        directness_score = 6.0
        
        # Penalize overly hedged language
        hedge_patterns = [
            r'\bi think\b', r'\bi believe\b', r'\bperhaps\b', r'\bmaybe\b',
            r'\bpossibly\b', r'\bmight be\b', r'\bcould be\b',
            r'\bnot sure\b', r'\bdon\'t know\b', r'\bnot certain\b',
        ]
        hedge_count = sum(1 for p in hedge_patterns if re.search(p, lower_resp))
        directness_score -= min(hedge_count * 0.5, 3.0)
        
        # Reward direct opening (gets to the point)
        first_sentence = sentences[0] if sentences else ""
        first_words = first_sentence.lower().split()[:5]
        
        # Penalize meta-responses that don't engage with content
        meta_openings = ['welcome to', 'please read', 'your question', 'great question',
                        'that\'s a great', 'thank you for', 'thanks for asking',
                        'i appreciate', 'do not fear']
        for opening in meta_openings:
            if lower_resp.startswith(opening):
                directness_score -= 1.5
                break
        
        directness_score = max(2.0, min(8.0, directness_score))
        
        # ============================================================
        # 9. CONTENT DENSITY
        # ============================================================
        # Ratio of content words to total words
        content_words = [w for w in clean_words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / len(clean_words) if clean_words else 0
        
        # Good content density is around 0.5-0.7
        if 0.4 <= content_ratio <= 0.75:
            density_score = 8.0
        elif 0.3 <= content_ratio <= 0.8:
            density_score = 6.0
        else:
            density_score = 4.0
        
        # ============================================================
        # 10. READABILITY (Simplified Flesch-like metric)
        # ============================================================
        # Count syllables approximately
        def count_syllables(word):
            word = word.lower().strip(string.punctuation)
            if not word:
                return 0
            if len(word) <= 3:
                return 1
            count = 0
            vowels = 'aeiouy'
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)
        
        total_syllables = sum(count_syllables(w) for w in clean_words)
        avg_syllables_per_word = total_syllables / len(clean_words) if clean_words else 0
        
        # Prefer moderate complexity (not too simple, not too complex)
        if 1.3 <= avg_syllables_per_word <= 1.8:
            readability_score = 8.0
        elif 1.1 <= avg_syllables_per_word <= 2.2:
            readability_score = 6.0
        else:
            readability_score = 4.0
        
        # ============================================================
        # 11. SPECIFIC DETAIL INDICATORS
        # ============================================================
        detail_score = 5.0
        
        # Check for specific examples, names, numbers, citations
        has_numbers = bool(re.search(r'\d+', response))
        has_quotes = bool(re.search(r'["""]', response) or re.search(r"'[^']{10,}'", response))
        has_names = bool(re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', response))
        has_specific_refs = bool(re.search(r'\b(?:for example|such as|e\.g\.|i\.e\.|specifically|in particular)\b', lower_resp))
        has_italics_or_bold = bool(re.search(r'\*[^*]+\*', response))
        
        detail_indicators = sum([has_numbers, has_quotes, has_names, has_specific_refs, has_italics_or_bold])
        detail_score += min(detail_indicators * 0.6, 3.0)
        detail_score = min(8.0, detail_score)
        
        # ============================================================
        # COMBINE SCORES WITH WEIGHTS
        # ============================================================
        weights = {
            'length': 0.08,
            'sent_len': 0.10,
            'variation': 0.06,
            'filler': 0.12,
            'redundancy': 0.12,
            'vocab': 0.08,
            'structure': 0.06,
            'relevance': 0.12,
            'directness': 0.08,
            'density': 0.06,
            'readability': 0.06,
            'detail': 0.06,
        }
        
        scores = {
            'length': length_score,
            'sent_len': sent_len_score,
            'variation': variation_score,
            'filler': filler_score,
            'redundancy': redundancy_score,
            'vocab': vocab_score,
            'structure': structure_score,
            'relevance': relevance_score,
            'directness': directness_score,
            'density': density_score,
            'readability': readability_score,
            'detail': detail_score,
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        # Apply a small bonus for responses that seem complete (end with punctuation)
        if response.rstrip()[-1] in '.!?)":':
            final_score += 0.2
        
        # Penalize very short responses that lack substance
        if num_words < 20:
            final_score *= 0.7
        elif num_words < 10:
            final_score *= 0.4
        
        # Penalize responses that are just meta-commentary (e.g., bot messages)
        meta_patterns = ['welcome to', 'please read our rules', 'your comments will be removed',
                        'while we do not require']
        meta_count = sum(1 for p in meta_patterns if p in lower_resp)
        if meta_count >= 2:
            final_score *= 0.5
        
        return round(max(0.0, min(10.0, final_score)), 3)
        
    except Exception:
        # Fallback: return a neutral score based on response length
        try:
            words = len(response.split()) if response else 0
            if words < 5:
                return 1.0
            elif words < 20:
                return 3.0
            elif words < 100:
                return 5.0
            else:
                return 4.5
        except Exception:
            return 3.0