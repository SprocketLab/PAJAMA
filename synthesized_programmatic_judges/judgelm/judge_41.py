def judging_function(query, response):
    """
    Evaluate clarity and conciseness of an LLM response.
    Uses a multi-signal approach focusing on:
    - Substantive content presence
    - Repetition detection
    - Filler/bloat detection
    - Sentence structure quality
    - Response length appropriateness relative to query
    - Formatting cleanliness
    """
    try:
        import re
        import math
        import collections
        import string

        # Edge cases
        if not response or not response.strip():
            return 0.5
        
        if not query or not query.strip():
            query = ""
        
        resp = response.strip()
        q = query.strip()
        
        # Very short responses - could be terse/unhelpful or perfectly concise
        if len(resp) < 3:
            return 1.0
        
        # ============================================================
        # SIGNAL 1: Substantive Content Score (0-10)
        # Does the response contain meaningful words?
        # ============================================================
        words = re.findall(r'[a-zA-Z]+', resp.lower())
        word_count = len(words)
        
        if word_count == 0:
            return 1.0
        
        # Count unique meaningful words (excluding very short ones)
        meaningful_words = [w for w in words if len(w) > 2]
        meaningful_count = len(meaningful_words)
        unique_meaningful = set(meaningful_words)
        
        if meaningful_count == 0:
            substantive_score = 2.0
        else:
            vocab_richness = len(unique_meaningful) / max(meaningful_count, 1)
            # Scale: 0.3+ is decent, 0.6+ is very rich
            substantive_score = min(10.0, vocab_richness * 14)
        
        # ============================================================
        # SIGNAL 2: Repetition Penalty (0 to -5)
        # Detect repeated phrases, sentences, and n-grams
        # ============================================================
        repetition_penalty = 0.0
        
        # Sentence-level repetition
        sentences = re.split(r'[.!?\n]+', resp)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) > 1:
            sentence_counter = collections.Counter(sentences)
            repeated_sentences = sum(count - 1 for count in sentence_counter.values() if count > 1)
            repetition_penalty -= min(4.0, repeated_sentences * 1.5)
        
        # Trigram repetition
        if len(words) >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counter = collections.Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(count - 1 for count in trigram_counter.values() if count > 1)
            trigram_rep_ratio = repeated_trigrams / max(total_trigrams, 1)
            repetition_penalty -= min(3.0, trigram_rep_ratio * 10)
        
        # Check for repeated lines (common in bad outputs)
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counter = collections.Counter(lines)
            repeated_lines = sum(count - 1 for count in line_counter.values() if count > 1)
            repetition_penalty -= min(3.0, repeated_lines * 1.0)
        
        repetition_penalty = max(-5.0, repetition_penalty)
        
        # ============================================================
        # SIGNAL 3: Filler and Bloat Detection (0 to -3)
        # ============================================================
        bloat_penalty = 0.0
        
        filler_phrases = [
            r'\bit is important to note that\b',
            r'\bit should be noted that\b',
            r'\bin order to\b',
            r'\bas a matter of fact\b',
            r'\bfor what it\'s worth\b',
            r'\bneedless to say\b',
            r'\bit goes without saying\b',
            r'\bat the end of the day\b',
            r'\bwhen all is said and done\b',
            r'\bthe fact of the matter is\b',
            r'\bin terms of\b',
            r'\bwith that being said\b',
            r'\bhaving said that\b',
            r'\ball things considered\b',
            r'\bas previously mentioned\b',
            r'\bas i mentioned\b',
        ]
        
        resp_lower = resp.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, resp_lower))
        
        bloat_penalty -= min(2.0, filler_count * 0.5)
        
        # Detect excessive hedging
        hedge_words = ['perhaps', 'maybe', 'possibly', 'somewhat', 'arguably', 'relatively', 'basically', 'essentially', 'actually', 'literally']
        hedge_count = sum(1 for w in words if w in hedge_words)
        hedge_ratio = hedge_count / max(word_count, 1)
        bloat_penalty -= min(1.0, hedge_ratio * 20)
        
        bloat_penalty = max(-3.0, bloat_penalty)
        
        # ============================================================
        # SIGNAL 4: Formatting and Structure Quality (-3 to +2)
        # ============================================================
        structure_score = 0.0
        
        # Check for garbled/broken formatting
        # Excessive HTML tags in non-HTML queries
        html_tags = re.findall(r'<[^>]+>', resp)
        query_asks_html = bool(re.search(r'\bhtml\b|\btag\b|\bweb\b', q.lower()))
        
        if not query_asks_html and len(html_tags) > 3:
            structure_score -= 2.0
        
        # Check for code blocks when not asked for code
        query_asks_code = bool(re.search(r'\bcode\b|\bprogram\b|\bfunction\b|\bscript\b|\bpython\b|\bjavascript\b', q.lower()))
        has_code_indicators = bool(re.search(r'(import |def |class |function |var |let |const )', resp))
        
        if not query_asks_code and has_code_indicators:
            structure_score -= 1.5
        
        # Reward clean paragraph structure
        if len(sentences) >= 2 and len(sentences) <= 8:
            structure_score += 1.0
        
        # Penalize responses that look like they're generating new questions/prompts
        prompt_patterns = [
            r'\b(input|output)\s*:',
            r'\bquestion\s*:',
            r'\banswer\s*:',
        ]
        prompt_leakage = 0
        for pat in prompt_patterns:
            prompt_leakage += len(re.findall(pat, resp_lower))
        
        if prompt_leakage > 2:
            structure_score -= min(2.0, prompt_leakage * 0.5)
        
        structure_score = max(-3.0, min(2.0, structure_score))
        
        # ============================================================
        # SIGNAL 5: Length Appropriateness (-2 to +2)
        # ============================================================
        length_score = 0.0
        
        query_words = len(re.findall(r'[a-zA-Z]+', q.lower()))
        
        # Very short response for a substantive query
        if query_words > 5 and word_count < 3:
            length_score -= 2.0
        # Extremely long response (likely bloated)
        elif word_count > 300:
            length_score -= min(1.5, (word_count - 300) / 400)
        # Moderate length is generally good
        elif 10 <= word_count <= 200:
            length_score += 1.0
        elif 3 <= word_count < 10:
            # Short but potentially concise - neutral to slightly positive
            length_score += 0.5
        
        length_score = max(-2.0, min(2.0, length_score))
        
        # ============================================================
        # SIGNAL 6: Relevance Signal (-2 to +2)
        # Do response words overlap with query words?
        # ============================================================
        relevance_score = 0.0
        
        if query_words > 0:
            q_words_set = set(re.findall(r'[a-zA-Z]+', q.lower()))
            # Remove very common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                        'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                        'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                        'my', 'your', 'his', 'its', 'our', 'their', 'and', 'or',
                        'but', 'not', 'no', 'if', 'what', 'which', 'who', 'how',
                        'when', 'where', 'why', 'there', 'here', 'all', 'each',
                        'also', 'about', 'up', 'out', 'so', 'than', 'too', 'very',
                        'just', 'any', 'some', 'other', 'into', 'more', 'make'}
            
            q_content = q_words_set - stopwords
            r_content = set(words) - stopwords
            
            if len(q_content) > 0 and len(r_content) > 0:
                overlap = len(q_content & r_content) / max(len(q_content), 1)
                relevance_score = min(2.0, overlap * 3.0)
            elif len(q_content) > 0 and len(r_content) == 0:
                relevance_score = -1.0
        
        # ============================================================
        # SIGNAL 7: Clarity indicators (-1 to +1)
        # ============================================================
        clarity_score = 0.0
        
        # Average sentence length (too long = unclear, too short = choppy)
        if len(sentences) > 0:
            avg_sentence_words = word_count / len(sentences)
            if 8 <= avg_sentence_words <= 25:
                clarity_score += 0.5
            elif avg_sentence_words > 40:
                clarity_score -= 0.5
        
        # Check for complete-looking sentences (starts with capital, has punctuation)
        if resp[0].isupper():
            clarity_score += 0.25
        
        if resp[-1] in '.!?)':
            clarity_score += 0.25
        elif resp[-1] == '\n':
            pass  # neutral
        
        clarity_score = max(-1.0, min(1.0, clarity_score))
        
        # ============================================================
        # SIGNAL 8: Truncation detection (-1 to 0)
        # ============================================================
        truncation_penalty = 0.0
        
        # Response seems cut off mid-sentence
        if len(resp) > 50 and resp[-1] not in '.!?)\n"\'':
            # Check if last word is complete-ish
            last_word = words[-1] if words else ''
            if len(last_word) <= 2 and resp[-1].isalpha():
                truncation_penalty = -0.5
        
        # ============================================================
        # COMBINE SIGNALS
        # ============================================================
        # Base score
        base = 5.0
        
        # Weight and combine
        final_score = (
            base
            + substantive_score * 0.25       # 0 to 2.5
            + repetition_penalty * 0.8       # -4 to 0
            + bloat_penalty * 0.6            # -1.8 to 0
            + structure_score * 0.7          # -2.1 to 1.4
            + length_score * 0.6             # -1.2 to 1.2
            + relevance_score * 0.5          # -1.0 to 1.0
            + clarity_score * 0.5            # -0.5 to 0.5
            + truncation_penalty * 0.5       # -0.25 to 0
        )
        
        # Clamp to 0-10
        final_score = max(0.5, min(10.0, final_score))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception:
        # Fallback: return a middle-of-the-road score
        try:
            if not response or len(response.strip()) < 3:
                return 1.0
            return 5.0
        except Exception:
            return 5.0