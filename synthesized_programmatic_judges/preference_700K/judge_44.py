def judging_function(query, response):
    """
    Evaluates clarity and conciseness using sentence-level analysis,
    information density metrics, and structural coherence signals.
    
    Algorithm focuses on:
    1. Sentence length variance and optimal sentence length targeting
    2. Information density (unique content words per total words ratio)
    3. Passive voice / weak construction detection
    4. Filler phrase and weasel word detection
    5. Average word length as proxy for precision vocabulary
    6. Sentence-to-sentence coherence via shared topic words
    7. Response completeness relative to query
    8. Redundancy detection via sentence-level semantic overlap (cosine on word sets)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_text = response.strip()
        if len(response_text) < 5:
            return 0.5
        
        # ---- Tokenization helpers ----
        def get_words(text):
            return re.findall(r"[a-zA-Z']+", text.lower())
        
        def get_sentences(text):
            # Split on sentence-ending punctuation, filter empties
            sents = re.split(r'(?<=[.!?])\s+', text)
            sents = [s.strip() for s in sents if len(s.strip()) > 2]
            if not sents:
                sents = [text]
            return sents
        
        words = get_words(response_text)
        total_words = len(words)
        if total_words < 2:
            return 1.0
        
        sentences = get_sentences(response_text)
        num_sentences = max(len(sentences), 1)
        
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'that', 'this', 'it', 'its', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'these',
            'those', 'am', 'also', 'don', 't', 's', 're', 've', 'll', 'd', 'doesn',
            'didn', 'won', 'wouldn', 'couldn', 'shouldn', 'isn', 'aren', 'wasn',
            'weren', 'hasn', 'haven', 'hadn'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        # ---- METRIC 1: Sentence length optimality (0-10) ----
        # Optimal avg sentence length is ~15-20 words for clarity
        sent_word_counts = []
        for s in sentences:
            sw = get_words(s)
            sent_word_counts.append(len(sw))
        
        avg_sent_len = sum(sent_word_counts) / max(len(sent_word_counts), 1)
        
        # Penalize too short (<8) or too long (>30) average sentence length
        if avg_sent_len < 1:
            sent_len_score = 2.0
        elif avg_sent_len <= 8:
            sent_len_score = 4.0 + (avg_sent_len / 8.0) * 3.0
        elif avg_sent_len <= 22:
            sent_len_score = 7.0 + 3.0 * (1.0 - abs(avg_sent_len - 15.0) / 7.0)
        elif avg_sent_len <= 35:
            sent_len_score = 7.0 - (avg_sent_len - 22.0) / 13.0 * 4.0
        else:
            sent_len_score = max(1.0, 3.0 - (avg_sent_len - 35.0) / 20.0)
        
        # Sentence length variance - some variety is good, too much is bad
        if len(sent_word_counts) > 1:
            mean_sl = avg_sent_len
            variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_dev = math.sqrt(variance)
            cv = std_dev / max(mean_sl, 1)  # coefficient of variation
            # Ideal CV around 0.3-0.5
            if cv < 0.1:
                variety_bonus = 0.0
            elif cv < 0.5:
                variety_bonus = 1.0
            elif cv < 0.8:
                variety_bonus = 0.5
            else:
                variety_bonus = -0.5
        else:
            variety_bonus = 0.0
        
        sent_len_score = min(10, max(0, sent_len_score + variety_bonus))
        
        # ---- METRIC 2: Information density (0-10) ----
        # Ratio of unique content words to total words
        if total_words > 0:
            content_ratio = len(content_words) / total_words
            unique_content_ratio = len(unique_content) / total_words
        else:
            content_ratio = 0
            unique_content_ratio = 0
        
        # Higher content ratio = more informative, less filler
        density_score = min(10, content_ratio * 15 + unique_content_ratio * 8)
        
        # ---- METRIC 3: Filler and weasel phrase detection (0-10, start at 10, subtract) ----
        filler_phrases = [
            r'\bbasically\b', r'\bessentially\b', r'\bactually\b', r'\bliterally\b',
            r'\bobviously\b', r'\bclearly\b', r'\bof course\b', r'\bneedless to say\b',
            r'\bit is worth noting that\b', r'\bit should be noted that\b',
            r'\bit is important to note\b', r'\bas a matter of fact\b',
            r'\bin terms of\b', r'\bat the end of the day\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bfor what it\'s worth\b', r'\bin my opinion\b',
            r'\bi think that\b', r'\bi believe that\b', r'\bi feel like\b',
            r'\bto be honest\b', r'\bfrankly\b', r'\bhonestly\b',
            r'\bkind of\b', r'\bsort of\b', r'\ba little bit\b',
            r'\bmore or less\b', r'\bso to speak\b',
            r'\bas you know\b', r'\bas we all know\b',
            r'\bthe thing is\b', r'\bthe fact of the matter\b',
            r'\bwith that being said\b', r'\bthat being said\b',
            r'\bhaving said that\b', r'\ball things considered\b',
            r'\bquite\b', r'\brather\b', r'\bsomewhat\b', r'\bperhaps\b',
            r'\bpossibly\b', r'\bpresumably\b', r'\bseemingly\b',
        ]
        
        response_lower = response_text.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_rate = filler_count / max(num_sentences, 1)
        filler_score = max(0, 10.0 - filler_rate * 3.0)
        
        # ---- METRIC 4: Passive voice detection (0-10) ----
        passive_patterns = [
            r'\b(?:is|are|was|were|been|be|being)\s+(?:\w+ly\s+)?(?:\w+ed|written|done|made|given|taken|seen|known|found|told|shown)\b',
        ]
        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, response_lower))
        
        passive_rate = passive_count / max(num_sentences, 1)
        passive_score = max(0, 10.0 - passive_rate * 4.0)
        
        # ---- METRIC 5: Average word length (precision vocabulary proxy) (0-10) ----
        if content_words:
            avg_word_len = sum(len(w) for w in content_words) / len(content_words)
        else:
            avg_word_len = 3.0
        
        # Ideal average content word length: 5-7 characters
        if avg_word_len < 3:
            word_len_score = 3.0
        elif avg_word_len < 5:
            word_len_score = 3.0 + (avg_word_len - 3.0) * 2.5
        elif avg_word_len <= 7:
            word_len_score = 8.0 + (avg_word_len - 5.0) * 0.5
        elif avg_word_len <= 9:
            word_len_score = 9.0 - (avg_word_len - 7.0) * 1.0
        else:
            word_len_score = max(3.0, 7.0 - (avg_word_len - 9.0) * 0.5)
        
        word_len_score = min(10, max(0, word_len_score))
        
        # ---- METRIC 6: Sentence-level redundancy detection (0-10) ----
        # Compare each pair of sentences using word-set overlap (cosine-like)
        if len(sentences) > 1:
            sent_word_sets = []
            for s in sentences:
                sw = set(get_words(s)) - stop_words
                sent_word_sets.append(sw)
            
            overlaps = []
            for i in range(len(sent_word_sets)):
                for j in range(i + 1, len(sent_word_sets)):
                    s1, s2 = sent_word_sets[i], sent_word_sets[j]
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    intersection = len(s1 & s2)
                    # Cosine-like: intersection / sqrt(|s1| * |s2|)
                    denom = math.sqrt(len(s1) * len(s2))
                    if denom > 0:
                        sim = intersection / denom
                        overlaps.append(sim)
            
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                max_overlap = max(overlaps)
                # High overlap = redundancy
                redundancy_penalty = avg_overlap * 5 + (max_overlap > 0.8) * 3
                redundancy_score = max(0, 10.0 - redundancy_penalty)
            else:
                redundancy_score = 7.0
        else:
            redundancy_score = 7.0  # Single sentence - neutral
        
        # ---- METRIC 7: Response substantiveness / completeness (0-10) ----
        # Check if response addresses query terms
        query_words = set(get_words(query)) - stop_words
        if query_words and content_words:
            query_coverage = len(unique_content & query_words) / max(len(query_words), 1)
            # Also measure how much content beyond query terms
            novel_content = len(unique_content - query_words)
            novelty_ratio = novel_content / max(len(unique_content), 1)
            
            substantive_score = min(10, query_coverage * 5 + novelty_ratio * 6)
        else:
            substantive_score = 5.0
        
        # Bonus for appropriate length (not too short, not bloated)
        # For clarity+conciseness, moderate length is ideal
        if total_words < 10:
            length_factor = 0.6
        elif total_words < 25:
            length_factor = 0.75
        elif total_words < 50:
            length_factor = 0.85
        elif total_words < 150:
            length_factor = 1.0
        elif total_words < 300:
            length_factor = 0.95
        elif total_words < 500:
            length_factor = 0.9
        else:
            length_factor = max(0.7, 0.9 - (total_words - 500) / 2000)
        
        # ---- METRIC 8: Structural signals (0-10) ----
        struct_score = 5.0
        
        # Code blocks - neutral to positive for technical queries
        code_blocks = len(re.findall(r'```', response_text))
        has_code = code_blocks >= 2
        
        # Presence of examples or specifics (numbers, proper nouns, etc.)
        specific_patterns = [
            r'\b\d+(?:\.\d+)?%?\b',  # numbers
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # proper nouns / names
            r'"[^"]+?"',  # quoted text
            r'\*[^*]+?\*',  # emphasized text
        ]
        specificity_count = 0
        for pat in specific_patterns:
            specificity_count += min(3, len(re.findall(pat, response_text)))
        
        struct_score += min(3.0, specificity_count * 0.5)
        
        # Penalize responses that start with overly generic openings
        generic_starts = [
            r'^(?:sure|of course|great question|that\'s a great|certainly|absolutely)',
            r'^(?:well,?\s)',
            r'^(?:so,?\s)',
            r'^(?:i\'m glad you asked)',
        ]
        for pat in generic_starts:
            if re.match(pat, response_lower.strip()):
                struct_score -= 1.0
                break
        
        # Penalize trailing incomplete sentences (truncation)
        if response_text and response_text[-1] not in '.!?)"\':;':
            # Might be truncated
            struct_score -= 0.5
        
        struct_score = min(10, max(0, struct_score))
        
        # ---- METRIC 9: Directness - does response get to the point quickly? (0-10) ----
        # Check first sentence for relevance to query
        first_sent_words = set(get_words(sentences[0])) - stop_words if sentences else set()
        if query_words and first_sent_words:
            first_sent_relevance = len(first_sent_words & query_words) / max(len(query_words), 1)
            directness_score = min(10, 4 + first_sent_relevance * 8)
        else:
            directness_score = 5.0
        
        # Penalize if first sentence is a meta-comment about the question
        meta_patterns = [
            r'^(?:that\'s? (?:a |an )?(?:great|good|interesting|excellent|wonderful))',
            r'^(?:thank you for)',
            r'^(?:i\'d be happy to)',
            r'^(?:let me )',
            r'^(?:welcome to)',
        ]
        for pat in meta_patterns:
            if re.match(pat, response_lower.strip()):
                directness_score -= 2.0
                break
        
        directness_score = min(10, max(0, directness_score))
        
        # ---- Combine all metrics with weights ----
        weights = {
            'sent_len': 0.10,
            'density': 0.15,
            'filler': 0.15,
            'passive': 0.05,
            'word_len': 0.05,
            'redundancy': 0.15,
            'substantive': 0.15,
            'structure': 0.10,
            'directness': 0.10,
        }
        
        raw_score = (
            weights['sent_len'] * sent_len_score +
            weights['density'] * density_score +
            weights['filler'] * filler_score +
            weights['passive'] * passive_score +
            weights['word_len'] * word_len_score +
            weights['redundancy'] * redundancy_score +
            weights['structure'] * struct_score +
            weights['substantive'] * substantive_score +
            weights['directness'] * directness_score
        )
        
        # Apply length factor
        final_score = raw_score * length_factor
        
        # Clamp to 0-10
        final_score = min(10.0, max(0.0, final_score))
        
        return round(final_score, 3)
    
    except Exception:
        # Never crash - return neutral score
        return 5.0