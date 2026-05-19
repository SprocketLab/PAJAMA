def judging_function(query, response):
    """
    Evaluate clarity and conciseness using a compression/signal-to-noise approach.
    
    This variant focuses on:
    1. Information density via unique content words ratio (signal-to-noise)
    2. Sentence structure variance (penalize monotonous or overly complex structures)
    3. Discourse coherence via pronoun/referent balance
    4. Filler/weasel word density
    5. Directive clarity (action-oriented language)
    6. Clause complexity estimation
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Tokenize
        words = re.findall(r"[a-zA-Z']+", response_clean.lower())
        if len(words) < 3:
            return 0.5
        
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        if not sentences:
            sentences = [response_clean]
        
        # ---- Feature 1: Compression Ratio (unique meaningful words / total words) ----
        # High ratio = dense, low redundancy
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'up',
            'about', 'also', 'like', 'get', 'got', 'much', 'even', 'still',
            'well', 'back', 'down', 'way', 'thing', 'things', 'really', 'quite'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        if len(words) > 0:
            compression_ratio = len(unique_content) / len(words)
        else:
            compression_ratio = 0
        
        # ---- Feature 2: Sentence Length Coefficient of Variation ----
        # Some variance is good (not monotonous), but extreme variance is bad
        sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
        sent_lengths = [sl for sl in sent_lengths if sl > 0]
        
        if len(sent_lengths) >= 2:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            if mean_sl > 0:
                var_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
                cv_sl = math.sqrt(var_sl) / mean_sl
            else:
                cv_sl = 0
        else:
            mean_sl = sum(sent_lengths) / max(len(sent_lengths), 1)
            cv_sl = 0.3  # neutral
        
        # Optimal CV is around 0.3-0.5 (some variety but not chaotic)
        cv_score = 1.0 - min(abs(cv_sl - 0.4) / 0.6, 1.0)
        
        # ---- Feature 3: Filler / Weasel / Hedge word density ----
        filler_patterns = [
            r'\bbasically\b', r'\bactually\b', r'\bliterally\b', r'\bprobably\b',
            r'\bmaybe\b', r'\bperhaps\b', r'\bkind of\b', r'\bsort of\b',
            r'\byou know\b', r'\bi mean\b', r'\bi think\b', r'\bi guess\b',
            r'\blike\b', r'\bjust\b', r'\breally\b', r'\bvery\b', r'\bquite\b',
            r'\bsomewhat\b', r'\bsomehow\b', r'\banyway\b', r'\banyways\b',
            r'\bwhatever\b', r'\bstuff\b', r'\bthings\b', r'\ba lot\b',
            r'\bpretty much\b', r'\bmore or less\b', r'\bat the end of the day\b',
            r'\bin my opinion\b', r'\bto be honest\b', r'\bto be fair\b',
            r'\bneedless to say\b', r'\bit goes without saying\b',
            r'\bas a matter of fact\b', r'\bit should be noted\b',
            r'\bit is worth mentioning\b', r'\bit is important to note\b',
            r'\bmight\b', r'\bcould be\b', r'\bseems like\b', r'\bappears to\b'
        ]
        
        response_lower = response_clean.lower()
        filler_count = 0
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_density = filler_count / max(len(words), 1)
        filler_score = max(0, 1.0 - filler_density * 8)
        
        # ---- Feature 4: Clause complexity via comma and conjunction density ----
        # Moderate use is fine; excessive suggests convoluted sentences
        comma_count = response_clean.count(',')
        semicolon_count = response_clean.count(';')
        conjunction_words = ['and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'while', 'although', 'because', 'since', 'unless', 'whereas', 'however', 'moreover', 'furthermore', 'nevertheless']
        conj_count = sum(1 for w in words if w in conjunction_words)
        
        clause_markers = comma_count + semicolon_count * 2 + conj_count
        clause_density = clause_markers / max(len(sentences), 1)
        
        # Optimal clause density around 1.5-3 per sentence
        if clause_density < 1.0:
            clause_score = 0.7 + 0.3 * clause_density  # simple is mostly ok
        elif clause_density <= 3.0:
            clause_score = 1.0  # sweet spot
        else:
            clause_score = max(0.2, 1.0 - (clause_density - 3.0) * 0.15)
        
        # ---- Feature 5: Directness / Action orientation ----
        # Count imperative-like starts and direct address
        direct_patterns = [
            r'(?:^|\. )(?:try|start|begin|make|take|keep|let|use|consider|remember|don\'t|avoid|ensure|check|look|think|ask|set|break|stay|give|get)\b',
            r'\byou (?:can|should|could|will|need|might want)\b',
            r'\bhere (?:are|is)\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b', r'\bthen\b',
            r'\bstep \d\b',
        ]
        
        direct_count = 0
        for pattern in direct_patterns:
            direct_count += len(re.findall(pattern, response_lower))
        
        directness_score = min(1.0, direct_count / max(len(sentences), 1) * 0.5)
        
        # ---- Feature 6: Repetition detection via trigram reuse ----
        if len(content_words) >= 3:
            trigrams = [' '.join(content_words[i:i+3]) for i in range(len(content_words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition = repeated_trigrams / max(len(trigrams), 1)
        else:
            trigram_repetition = 0
        
        repetition_score = max(0, 1.0 - trigram_repetition * 10)
        
        # ---- Feature 7: Structural organization ----
        # Check for numbered lists, bullet points, paragraph breaks
        has_numbering = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-•*]\s', response_clean))
        has_paragraphs = '\n\n' in response_clean or '\n \n' in response_clean
        has_colons = bool(re.search(r':\s*\n', response_clean))  # headers/labels
        
        structure_score = 0.4  # baseline
        if has_numbering:
            structure_score += 0.25
        if has_bullets:
            structure_score += 0.2
        if has_paragraphs:
            structure_score += 0.1
        if has_colons:
            structure_score += 0.05
        structure_score = min(1.0, structure_score)
        
        # ---- Feature 8: Mean sentence length penalty ----
        # Very short or very long average sentences are suboptimal
        if mean_sl > 0:
            if mean_sl < 5:
                length_score = 0.4 + 0.12 * mean_sl
            elif mean_sl <= 20:
                length_score = 1.0
            elif mean_sl <= 35:
                length_score = 1.0 - (mean_sl - 20) * 0.03
            else:
                length_score = max(0.2, 0.55 - (mean_sl - 35) * 0.02)
        else:
            length_score = 0.3
        
        # ---- Feature 9: Query relevance via keyword overlap ----
        query_words = set(re.findall(r"[a-zA-Z']+", query.lower())) - stop_words
        query_content = {w for w in query_words if len(w) > 2}
        
        if query_content:
            overlap = unique_content & query_content
            relevance_score = min(1.0, len(overlap) / max(min(len(query_content), 8), 1))
        else:
            relevance_score = 0.5
        
        # ---- Feature 10: Empathy/Engagement markers (context-dependent) ----
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'s (?:completely |totally |absolutely )?(?:understandable|okay|fine|normal|natural)\b',
            r'\bi\'m sorry\b', r'\bsorry to hear\b', r'\bcompletely understandable\b',
            r'\babsolutely okay\b', r'\bit\'s (?:okay|fine|natural|normal|perfectly)\b',
            r'\bdon\'t worry\b', r'\bwe\'re here\b', r'\bwe value\b',
        ]
        
        # Check if query seems emotional
        emotional_keywords = {'feel', 'feeling', 'frustrated', 'sad', 'angry', 'stress', 'stressed',
                            'lonely', 'loneliness', 'heartbroken', 'devastated', 'exhausted',
                            'struggle', 'struggling', 'difficult', 'tough', 'hard', 'regret',
                            'worried', 'anxious', 'upset', 'disappointed', 'down', 'depressed'}
        query_lower = query.lower()
        is_emotional = any(kw in query_lower for kw in emotional_keywords)
        
        empathy_count = 0
        for pattern in empathy_patterns:
            empathy_count += len(re.findall(pattern, response_lower))
        
        if is_emotional:
            empathy_score = min(1.0, empathy_count * 0.3)
        else:
            empathy_score = 0.5  # neutral when not needed
        
        # ---- Feature 11: Negative tone / dismissive language ----
        dismissive_patterns = [
            r'\bjust\s+(?:do|try|get|move|deal|handle)\b',
            r'\bit\'s (?:just|only) a\b',
            r'\bget over it\b', r'\bmove on\b',
            r'\bnothing wrong\b', r'\bdon\'t let it\b',
            r'\byou need to get\b', r'\bget yourself together\b',
            r'\bstop (?:being|feeling)\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        if is_emotional:
            dismissive_penalty = min(0.5, dismissive_count * 0.15)
        else:
            dismissive_penalty = min(0.2, dismissive_count * 0.05)
        
        # ---- Feature 12: Response adequacy (not too short, not bloated) ----
        response_len = len(words)
        if response_len < 15:
            adequacy = 0.3
        elif response_len < 30:
            adequacy = 0.5 + (response_len - 15) * 0.033
        elif response_len <= 200:
            adequacy = 1.0
        elif response_len <= 350:
            adequacy = 1.0 - (response_len - 200) * 0.002
        else:
            adequacy = max(0.5, 0.7 - (response_len - 350) * 0.001)
        
        # ---- Feature 13: Specificity via concrete word detection ----
        # Words that suggest specific, actionable advice
        specific_patterns = [
            r'\b\d+\b',  # numbers
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b',
        ]
        specific_count = sum(len(re.findall(p, response_lower)) for p in specific_patterns)
        
        # Also count words > 7 chars as potentially more specific
        long_words = [w for w in content_words if len(w) > 7]
        long_word_ratio = len(long_words) / max(len(content_words), 1)
        
        specificity_score = min(1.0, 0.3 + specific_count * 0.1 + long_word_ratio * 0.8)
        
        # ---- Feature 14: Negative capability indicators ----
        # "might not", "probably won't", "may not be able" - hedging about inability
        inability_patterns = [
            r'\bmight not\b', r'\bmay not\b', r'\bprobably won\'t\b',
            r'\bcan\'t\b', r'\bcannot\b', r'\bnot able\b', r'\bwon\'t be able\b',
            r'\bit might not\b', r'\bit may not\b', r'\bit probably\b',
        ]
        inability_count = sum(len(re.findall(p, response_lower)) for p in inability_patterns)
        inability_penalty = min(0.3, inability_count * 0.08)
        
        # ============ COMBINE SCORES ============
        # Weighted combination
        score = (
            compression_ratio * 12.0 +      # 0-1 range, weight ~12
            cv_score * 4.0 +                  # sentence variety
            filler_score * 8.0 +              # penalize filler heavily
            clause_score * 4.0 +              # clause complexity
            directness_score * 6.0 +          # action orientation
            repetition_score * 6.0 +          # penalize repetition
            structure_score * 5.0 +           # structural organization
            length_score * 5.0 +              # sentence length
            relevance_score * 8.0 +           # query relevance
            empathy_score * 5.0 +             # empathy when needed
            adequacy * 5.0 +                  # response length adequacy
            specificity_score * 4.0 -         # specificity
            dismissive_penalty * 10.0 -       # dismissive penalty
            inability_penalty * 8.0           # inability/hedging penalty
        )
        
        # Normalize to roughly 1-5 scale
        # Max theoretical: ~12 + 4 + 8 + 4 + 6 + 6 + 5 + 5 + 8 + 5 + 5 + 4 = 72
        # Min theoretical: ~0 - 5 - 2.4 = negative
        # Typical good: ~50-60, typical bad: ~25-35
        
        # Map to 1-5 scale
        raw_max = 68.0
        raw_min = 15.0
        normalized = (score - raw_min) / (raw_max - raw_min)
        normalized = max(0, min(1, normalized))
        
        final_score = 1.0 + normalized * 4.0  # 1-5 range
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5