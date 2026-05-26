def judging_function(query, response):
    """
    Evaluates clarity and conciseness of an LLM response.
    
    This variant focuses on:
    - Information density (ratio of meaningful content words to total words)
    - Transition/connector word usage (indicates logical flow)
    - Filler/hedge word penalty
    - Redundancy detection via n-gram repetition analysis
    - Directness of opening (does it get to the point quickly?)
    - Formatting effectiveness (markdown structure as signal of organization)
    - Word specificity (longer, more specific words vs short vague ones)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words)
        if total_words < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Filler and hedge word density (PENALTY)
        # ============================================================
        filler_words = {
            'really', 'very', 'quite', 'rather', 'somewhat', 'basically',
            'essentially', 'actually', 'literally', 'honestly', 'frankly',
            'simply', 'just', 'perhaps', 'maybe', 'possibly', 'probably',
            'kind', 'sort', 'stuff', 'things', 'thing', 'like',
            'obviously', 'clearly', 'definitely', 'certainly', 'absolutely',
            'totally', 'completely', 'entirely', 'virtually', 'practically'
        }
        
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / total_words
        filler_score = max(0, 10 - filler_ratio * 120)  # 0-10 scale
        
        # ============================================================
        # FEATURE 2: Redundancy via trigram repetition
        # ============================================================
        if total_words >= 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition_ratio = repeated_trigrams / max(len(trigrams), 1)
            redundancy_score = max(0, 10 - trigram_repetition_ratio * 80)
        else:
            redundancy_score = 7.0
        
        # ============================================================
        # FEATURE 3: Directness of opening
        # How quickly does the response address the query?
        # Penalize excessive preamble / pleasantries before substance
        # ============================================================
        opening_fluff_patterns = [
            r"^that'?s?\s+(a\s+)?(great|good|excellent|wonderful|fantastic|awesome|interesting|nice)",
            r"^(oh|wow|ah|well|so|hey)\b",
            r"^(of course|sure thing|no problem|happy to help|glad you asked)",
            r"^i'?d be happy to",
            r"^what a (great|good|wonderful|fantastic)",
            r"^a classic",
        ]
        
        first_line = response.split('\n')[0].strip().lower()
        opening_fluff = 0
        for pat in opening_fluff_patterns:
            if re.search(pat, first_line):
                opening_fluff += 1
        
        # Check if first sentence is exclamatory filler
        first_sent = sentences[0].lower() if sentences else ""
        first_sent_words = re.findall(r'[a-zA-Z]+', first_sent)
        if len(first_sent_words) > 0 and len(first_sent_words) < 5:
            # Very short first sentence might be filler like "Great question!"
            fluff_starters = {'great', 'good', 'awesome', 'nice', 'wonderful', 'excellent', 'classic', 'sure'}
            if any(w in fluff_starters for w in first_sent_words):
                opening_fluff += 1
        
        directness_score = max(0, 10 - opening_fluff * 2.5)
        
        # ============================================================
        # FEATURE 4: Structural organization (markdown formatting)
        # ============================================================
        has_headers = len(re.findall(r'^#{1,4}\s+', response, re.MULTILINE))
        has_bold = len(re.findall(r'\*\*[^*]+\*\*', response))
        has_numbered_list = len(re.findall(r'^\s*\d+[\.\)]\s+', response, re.MULTILINE))
        has_bullet_list = len(re.findall(r'^\s*[-*•]\s+', response, re.MULTILINE))
        
        # Structure score: organized responses score higher, but don't over-reward
        structure_elements = min(has_headers, 3) * 0.8 + min(has_bold, 5) * 0.4 + \
                           min(has_numbered_list, 5) * 0.5 + min(has_bullet_list, 5) * 0.3
        structure_score = min(10, 4 + structure_elements)
        
        # ============================================================
        # FEATURE 5: Information density via content word ratio
        # ============================================================
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'while',
            'because', 'until', 'about', 'this', 'that', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'we', 'our', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom'
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / total_words if total_words > 0 else 0
        info_density_score = min(10, content_ratio * 20)  # ~0.5 ratio -> 10
        
        # ============================================================
        # FEATURE 6: Sentence-level clarity
        # Measure variance in sentence length (moderate variance is good)
        # Very uniform = robotic; very high variance = disorganized
        # ============================================================
        sent_lengths = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Optimal CV around 0.3-0.5 (some variety but not chaotic)
            if cv < 0.15:
                sent_variety_score = 6.0  # too uniform
            elif cv < 0.6:
                sent_variety_score = 8.0 + (cv - 0.15) * 4  # reward moderate variety
            else:
                sent_variety_score = max(4, 10 - (cv - 0.6) * 8)  # penalize chaos
            sent_variety_score = min(10, max(0, sent_variety_score))
        else:
            sent_variety_score = 5.0
        
        # ============================================================
        # FEATURE 7: Wordy phrase detection (PENALTY)
        # ============================================================
        wordy_phrases = [
            r'\bin order to\b', r'\bdue to the fact that\b', r'\bat this point in time\b',
            r'\bfor the purpose of\b', r'\bin the event that\b', r'\bit is important to note that\b',
            r'\bit should be noted that\b', r'\bas a matter of fact\b', r'\bin terms of\b',
            r'\bwith regard to\b', r'\bwith respect to\b', r'\bin spite of the fact that\b',
            r'\bthere are several\b', r'\bthere are many\b', r'\bthere are a number of\b',
            r'\bthe fact that\b', r'\bit is worth noting\b', r'\bit is important to\b',
            r'\bas mentioned (earlier|above|before|previously)\b',
            r'\bin conclusion\b', r'\ball in all\b', r'\bat the end of the day\b',
            r'\bneedless to say\b', r'\blast but not least\b',
            r'\bcan be a fun and rewarding\b', r'\bcan really make a difference\b',
        ]
        
        response_lower = response.lower()
        wordy_count = 0
        for pattern in wordy_phrases:
            wordy_count += len(re.findall(pattern, response_lower))
        
        wordy_penalty = min(5, wordy_count * 1.2)
        wordiness_score = max(0, 10 - wordy_penalty)
        
        # ============================================================
        # FEATURE 8: Specificity - ratio of specific/technical words
        # Words with 7+ chars tend to be more specific/technical
        # ============================================================
        specific_words = [w for w in content_words if len(w) >= 7]
        specificity_ratio = len(specific_words) / max(len(content_words), 1)
        specificity_score = min(10, 3 + specificity_ratio * 18)
        
        # ============================================================
        # FEATURE 9: Paragraph structure
        # ============================================================
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Single giant block of text is less clear
        if num_paragraphs == 1 and total_words > 80:
            paragraph_score = 5.0
        elif num_paragraphs >= 2:
            paragraph_score = min(10, 6 + num_paragraphs * 0.5)
        else:
            paragraph_score = 6.0
        
        # ============================================================
        # FEATURE 10: Repetitive phrase patterns at sentence starts
        # ============================================================
        sentence_starts = []
        for s in sentences:
            s_words = re.findall(r'[a-zA-Z]+', s.lower())
            if len(s_words) >= 2:
                sentence_starts.append(tuple(s_words[:2]))
        
        if len(sentence_starts) > 2:
            start_counts = Counter(sentence_starts)
            repeated_starts = sum(c - 1 for c in start_counts.values() if c > 1)
            start_repetition_ratio = repeated_starts / len(sentence_starts)
            start_variety_score = max(0, 10 - start_repetition_ratio * 30)
        else:
            start_variety_score = 7.0
        
        # ============================================================
        # FEATURE 11: Query relevance - check if key query terms appear
        # ============================================================
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        query_content = query_words - stop_words
        if query_content:
            response_word_set = set(words)
            overlap = len(query_content & response_word_set)
            relevance_ratio = overlap / len(query_content)
            relevance_score = min(10, 4 + relevance_ratio * 8)
        else:
            relevance_score = 7.0
        
        # ============================================================
        # COMBINE SCORES with weights
        # ============================================================
        weights = {
            'filler': 0.10,
            'redundancy': 0.12,
            'directness': 0.08,
            'structure': 0.15,
            'info_density': 0.10,
            'sent_variety': 0.08,
            'wordiness': 0.10,
            'specificity': 0.07,
            'paragraph': 0.08,
            'start_variety': 0.05,
            'relevance': 0.07,
        }
        
        scores = {
            'filler': filler_score,
            'redundancy': redundancy_score,
            'directness': directness_score,
            'structure': structure_score,
            'info_density': info_density_score,
            'sent_variety': sent_variety_score,
            'wordiness': wordiness_score,
            'specificity': specificity_score,
            'paragraph': paragraph_score,
            'start_variety': start_variety_score,
            'relevance': relevance_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-100 range
        final_score = final_score * 10
        
        # Clamp
        final_score = max(0, min(100, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        return 50.0