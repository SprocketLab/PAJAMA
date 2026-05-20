def judging_function(query, response):
    """
    Evaluate language quality and readability using a substantially different approach:
    - Punctuation correctness and variety
    - Sentence structure analysis (clause detection, subordination)
    - Cohesion markers and discourse connectives
    - Spelling error heuristics (unusual character patterns)
    - Token sophistication via word frequency rank approximation
    - Rhythm/cadence via sentence length variance patterns
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", text)
        if len(words) < 2:
            return 1.0
        
        # Split sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if not sentences:
            sentences = [text]
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        num_chars = len(text)
        
        score = 0.0
        
        # ============================================================
        # 1. PUNCTUATION QUALITY & VARIETY (0-15 points)
        # ============================================================
        punct_chars = [c for c in text if c in string.punctuation]
        punct_ratio = len(punct_chars) / max(num_chars, 1)
        
        # Good punctuation ratio is roughly 0.03-0.10
        if 0.03 <= punct_ratio <= 0.10:
            punct_score = 8.0
        elif 0.01 <= punct_ratio < 0.03:
            punct_score = 5.0
        elif 0.10 < punct_ratio <= 0.15:
            punct_score = 5.0
        else:
            punct_score = 2.0
        
        # Variety of punctuation used
        unique_punct = set(punct_chars)
        punct_variety = min(len(unique_punct), 6) / 6.0 * 7.0
        
        score += punct_score + punct_variety
        
        # ============================================================
        # 2. CLAUSE COMPLEXITY & SUBORDINATION (0-15 points)
        # ============================================================
        subordinators = {
            'because', 'although', 'though', 'while', 'whereas', 'since',
            'unless', 'until', 'after', 'before', 'when', 'whenever',
            'where', 'wherever', 'if', 'whether', 'that', 'which', 'who',
            'whom', 'whose', 'how', 'what', 'why', 'even', 'provided',
            'assuming', 'given', 'once', 'so'
        }
        
        lower_words = [w.lower() for w in words]
        sub_count = sum(1 for w in lower_words if w in subordinators)
        sub_ratio = sub_count / max(num_sentences, 1)
        
        # Comma usage per sentence (indicates clause separation)
        comma_count = text.count(',')
        commas_per_sent = comma_count / max(num_sentences, 1)
        
        # Good subordination: 0.5-3 subordinators per sentence
        clause_score = min(sub_ratio / 2.0, 1.0) * 7.0
        
        # Good comma usage: 1-3 commas per sentence
        if 0.5 <= commas_per_sent <= 3.5:
            clause_score += 5.0
        elif 0.2 <= commas_per_sent < 0.5 or 3.5 < commas_per_sent <= 5:
            clause_score += 3.0
        else:
            clause_score += 1.0
        
        # Semicolons, colons, dashes indicate sophisticated structure
        advanced_punct = text.count(';') + text.count(':') + text.count('—') + text.count('--')
        clause_score += min(advanced_punct, 3) * 1.0
        
        score += min(clause_score, 15.0)
        
        # ============================================================
        # 3. COHESION & DISCOURSE CONNECTIVES (0-15 points)
        # ============================================================
        connectives = {
            'however', 'moreover', 'furthermore', 'nevertheless', 'therefore',
            'consequently', 'additionally', 'alternatively', 'specifically',
            'particularly', 'essentially', 'ultimately', 'similarly',
            'conversely', 'meanwhile', 'subsequently', 'accordingly',
            'indeed', 'certainly', 'naturally', 'obviously', 'clearly',
            'importantly', 'significantly', 'notably', 'interestingly',
            'surprisingly', 'unfortunately', 'fortunately', 'admittedly',
            'arguably', 'presumably', 'apparently', 'evidently',
            'thus', 'hence', 'nonetheless', 'regardless', 'overall',
            'likewise', 'instead', 'otherwise', 'still', 'yet'
        }
        
        # Multi-word connectives
        multi_connectives = [
            'in addition', 'on the other hand', 'for example', 'for instance',
            'in contrast', 'as a result', 'in other words', 'that is',
            'in fact', 'of course', 'at the same time', 'in particular',
            'to be sure', 'in any case', 'as such', 'in this case',
            'more importantly', 'on top of that', 'having said that',
            'that said', 'to put it', 'in short', 'to summarize',
            'first of all', 'to begin with', 'last but not least'
        ]
        
        lower_text = text.lower()
        conn_count = sum(1 for w in lower_words if w in connectives)
        multi_conn_count = sum(1 for mc in multi_connectives if mc in lower_text)
        
        total_conn = conn_count + multi_conn_count
        conn_per_sent = total_conn / max(num_sentences, 1)
        
        # Good cohesion: 0.2-1.0 connectives per sentence
        cohesion_score = min(conn_per_sent / 0.6, 1.0) * 10.0
        
        # Referential cohesion: pronouns that refer back
        referential_words = {'this', 'that', 'these', 'those', 'such', 'it', 'they', 'them', 'its'}
        ref_count = sum(1 for w in lower_words if w in referential_words)
        ref_ratio = ref_count / max(num_words, 1)
        if 0.02 <= ref_ratio <= 0.08:
            cohesion_score += 5.0
        elif ref_ratio > 0:
            cohesion_score += 2.0
        
        score += min(cohesion_score, 15.0)
        
        # ============================================================
        # 4. SPELLING ERROR HEURISTICS (0-10 points)
        # ============================================================
        # Detect likely misspellings via unusual letter patterns
        # Common English doesn't have: triple letters, many rare bigrams, etc.
        
        spelling_penalty = 0
        
        # Triple letter detection
        for w in words:
            if re.search(r'(.)\1\1', w.lower()):
                spelling_penalty += 1
        
        # Unusual bigrams that rarely appear in English
        rare_bigrams = {'xz', 'zx', 'qx', 'xq', 'jx', 'xj', 'zj', 'jz',
                        'qk', 'kq', 'vx', 'xv', 'bx', 'xb', 'wx', 'xw',
                        'fq', 'qf', 'pz', 'zp', 'vj', 'jv', 'kz', 'zk'}
        
        for w in words:
            wl = w.lower()
            for i in range(len(wl) - 1):
                if wl[i:i+2] in rare_bigrams:
                    spelling_penalty += 1
        
        # Words with no vowels (length > 2) are likely errors
        for w in words:
            if len(w) > 2 and not re.search(r'[aeiouyAEIOUY]', w):
                spelling_penalty += 1
        
        # Repeated word detection (stuttering)
        repeated_words = 0
        for i in range(len(lower_words) - 1):
            if lower_words[i] == lower_words[i+1] and lower_words[i] not in {'that', 'had', 'very', 'really', 'so'}:
                repeated_words += 1
        spelling_penalty += repeated_words
        
        spell_ratio = spelling_penalty / max(num_words, 1)
        spelling_score = max(0, 10.0 - spell_ratio * 200.0)
        
        score += spelling_score
        
        # ============================================================
        # 5. SENTENCE LENGTH RHYTHM / CADENCE (0-10 points)
        # ============================================================
        # Good writing varies sentence length. Measure coefficient of variation.
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) >= 3:
            mean_len = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_len = math.sqrt(variance)
            cv = std_len / max(mean_len, 1)
            
            # Good CV is 0.3-0.7 (varied but not chaotic)
            if 0.25 <= cv <= 0.75:
                rhythm_score = 8.0
            elif 0.15 <= cv < 0.25 or 0.75 < cv <= 1.0:
                rhythm_score = 5.0
            elif cv < 0.15:
                rhythm_score = 3.0  # Too monotonous
            else:
                rhythm_score = 2.0  # Too chaotic
            
            # Check for a mix of short and long sentences
            short_sents = sum(1 for c in sent_word_counts if c <= 8)
            long_sents = sum(1 for c in sent_word_counts if c >= 20)
            if short_sents > 0 and long_sents > 0:
                rhythm_score += 2.0
        elif len(sent_word_counts) >= 2:
            rhythm_score = 5.0
        else:
            rhythm_score = 3.0
        
        score += min(rhythm_score, 10.0)
        
        # ============================================================
        # 6. WORD SOPHISTICATION via morphological complexity (0-12 points)
        # ============================================================
        # Count words with common sophisticated suffixes
        sophisticated_suffixes = [
            'tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance',
            'ious', 'eous', 'ible', 'able', 'ful', 'less', 'ive',
            'ical', 'ular', 'ously', 'ively', 'ially', 'ally',
            'ism', 'ist', 'ize', 'ise', 'ify', 'ology', 'ographic'
        ]
        
        sophisticated_count = 0
        for w in lower_words:
            if len(w) >= 6:
                for suffix in sophisticated_suffixes:
                    if w.endswith(suffix):
                        sophisticated_count += 1
                        break
        
        soph_ratio = sophisticated_count / max(num_words, 1)
        
        # Good sophistication: 5-20% of words
        if 0.05 <= soph_ratio <= 0.25:
            soph_score = 8.0
        elif 0.02 <= soph_ratio < 0.05:
            soph_score = 5.0
        elif soph_ratio > 0.25:
            soph_score = 6.0  # Slightly overwritten
        else:
            soph_score = 2.0
        
        # Proportion of longer words (7+ chars) as proxy for vocabulary level
        long_word_count = sum(1 for w in words if len(w) >= 7)
        long_ratio = long_word_count / max(num_words, 1)
        if 0.15 <= long_ratio <= 0.40:
            soph_score += 4.0
        elif 0.08 <= long_ratio < 0.15 or 0.40 < long_ratio <= 0.50:
            soph_score += 2.0
        else:
            soph_score += 1.0
        
        score += min(soph_score, 12.0)
        
        # ============================================================
        # 7. RESPONSE COMPLETENESS & STRUCTURE (0-13 points)
        # ============================================================
        # Proper capitalization at sentence starts
        cap_starts = 0
        for s in sentences:
            stripped = s.strip()
            if stripped and stripped[0].isupper():
                cap_starts += 1
        cap_ratio = cap_starts / max(num_sentences, 1)
        struct_score = cap_ratio * 4.0
        
        # Ending punctuation
        if text.rstrip()[-1] in '.!?)':
            struct_score += 2.0
        elif text.rstrip()[-1] in ':;,':
            struct_score += 0.5
        
        # Response length relative to query (longer, more detailed answers tend to be better)
        query_words = len(re.findall(r"[a-zA-Z']+", query)) if query else 10
        length_ratio = num_words / max(query_words, 1)
        if length_ratio >= 2.0:
            struct_score += 4.0
        elif length_ratio >= 1.0:
            struct_score += 2.5
        elif length_ratio >= 0.5:
            struct_score += 1.0
        
        # Presence of formatting (markdown, lists, etc.) — indicates effort
        has_formatting = bool(re.search(r'(\*\*|__|```|^\s*[-*]\s|\d+\.)', text, re.MULTILINE))
        if has_formatting:
            struct_score += 2.0
        
        # Multiple sentences indicate thoroughness
        if num_sentences >= 4:
            struct_score += 1.0
        
        score += min(struct_score, 13.0)
        
        # ============================================================
        # 8. OPENING QUALITY (0-5 points)
        # ============================================================
        # Good responses often start with engaging, relevant openings
        first_sent = sentences[0] if sentences else ""
        first_words_list = re.findall(r"[a-zA-Z']+", first_sent)
        
        # Penalize very short openers that are just filler
        filler_openers = {'yes', 'no', 'well', 'ok', 'okay', 'sure', 'hi', 'hello'}
        if first_words_list and first_words_list[0].lower() in filler_openers and len(first_words_list) < 5:
            opening_score = 1.0
        elif len(first_words_list) >= 5:
            opening_score = 4.0
        else:
            opening_score = 2.0
        
        # Check if first sentence contains query-relevant words
        query_lower = query.lower() if query else ""
        query_words_set = set(re.findall(r"[a-zA-Z']+", query_lower))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'or', 'but', 'not', 'this', 'that', 'it', 'i', 'you',
                     'we', 'they', 'my', 'your', 'our', 'what', 'how', 'which'}
        query_content = query_words_set - stopwords
        first_sent_lower = first_sent.lower()
        relevance_hits = sum(1 for qw in query_content if qw in first_sent_lower)
        if relevance_hits >= 2:
            opening_score += 1.0
        
        score += min(opening_score, 5.0)
        
        # ============================================================
        # 9. HEDGING & NUANCE LANGUAGE (0-5 points)
        # ============================================================
        # Sophisticated responses often use hedging/qualification
        hedge_words = {
            'perhaps', 'possibly', 'likely', 'unlikely', 'arguably',
            'typically', 'generally', 'usually', 'often', 'sometimes',
            'tends', 'might', 'could', 'may', 'seem', 'seems',
            'appear', 'appears', 'suggest', 'suggests', 'indicate',
            'indicates', 'relatively', 'somewhat', 'fairly', 'rather',
            'approximately', 'roughly', 'essentially', 'primarily',
            'largely', 'mostly', 'partly', 'potentially'
        }
        
        hedge_count = sum(1 for w in lower_words if w in hedge_words)
        hedge_ratio = hedge_count / max(num_words, 1)
        
        if 0.01 <= hedge_ratio <= 0.06:
            hedge_score = 5.0
        elif hedge_ratio > 0.06:
            hedge_score = 3.0  # Over-hedging
        elif hedge_count >= 1:
            hedge_score = 2.0
        else:
            hedge_score = 0.5
        
        score += hedge_score
        
        # ============================================================
        # Final normalization to 0-100 range
        # ============================================================
        # Max possible: 15 + 15 + 15 + 10 + 10 + 12 + 13 + 5 + 5 = 100
        final_score = max(0.0, min(100.0, score))
        
        return round(final_score, 2)
        
    except Exception:
        return 5.0