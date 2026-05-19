def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    focused on detecting concrete information tokens: numbers, proper nouns,
    technical terms, specific references, and structured information delivery.
    
    Algorithm: Token-level classification into "evidence tokens" vs "filler tokens",
    computing an evidence density ratio, combined with anti-pattern detection for
    vagueness and a structural coherence bonus.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        resp = response.strip()
        query_text = (query or "").strip()
        
        # Tokenize response
        words = re.findall(r'\b\w+\b', resp)
        if len(words) == 0:
            return 0.0
        
        resp_lower = resp.lower()
        words_lower = [w.lower() for w in words]
        
        # ============================================================
        # FEATURE 1: Named Entity / Proper Noun density
        # Detect capitalized words that aren't sentence starters
        # ============================================================
        sentences = re.split(r'[.!?]\s+', resp)
        sentence_starters = set()
        for s in sentences:
            s = s.strip()
            if s:
                first_word = re.match(r'\b(\w+)\b', s)
                if first_word:
                    sentence_starters.add(first_word.group(1))
        
        # Words that are capitalized but not sentence starters and not all-caps short words
        proper_noun_count = 0
        for i, w in enumerate(words):
            if len(w) > 1 and w[0].isupper() and not w.isupper():
                # Not a sentence starter (approximate: check if preceded by period-like)
                if i > 0 and w not in sentence_starters:
                    proper_noun_count += 1
                elif i == 0:
                    pass  # skip first word
                else:
                    # Could still be a proper noun even if it starts a sentence
                    # Give partial credit if it looks like a real name
                    if len(w) > 2 and w not in {'The', 'This', 'That', 'These', 'Those', 
                                                  'Here', 'There', 'What', 'When', 'Where',
                                                  'How', 'Why', 'Who', 'Which', 'Some',
                                                  'Many', 'Most', 'All', 'Any', 'Each',
                                                  'Every', 'Other', 'Another', 'Such',
                                                  'However', 'Therefore', 'Moreover',
                                                  'Furthermore', 'Additionally', 'Also',
                                                  'But', 'And', 'Or', 'Not', 'Yes', 'No',
                                                  'Sure', 'Comment', 'Output', 'Input',
                                                  'Question', 'Answer', 'Once', 'You',
                                                  'We', 'They', 'He', 'She', 'It', 'If',
                                                  'For', 'From', 'With', 'About', 'Into',
                                                  'Over', 'After', 'Before', 'Between',
                                                  'Under', 'During', 'Without', 'Through'}:
                        proper_noun_count += 0.5
        
        proper_noun_density = proper_noun_count / max(len(words), 1)
        
        # ============================================================
        # FEATURE 2: Numeric / quantitative information
        # ============================================================
        # Count numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', resp)
        percentages = re.findall(r'\d+\.?\d*\s*%', resp)
        dates = re.findall(r'\b(?:19|20)\d{2}\b', resp)
        measurements = re.findall(r'\b\d+\.?\d*\s*(?:km|mi|lb|kg|ft|m|cm|mm|inch|inches|feet|miles|hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|cents?|USD|EUR|GBP)\b', resp, re.IGNORECASE)
        
        numeric_count = len(numbers) + len(percentages) * 1.5 + len(dates) * 1.5 + len(measurements) * 2
        numeric_density = numeric_count / max(len(words), 1)
        
        # ============================================================
        # FEATURE 3: Specificity vocabulary detection
        # Words/phrases that signal concrete information
        # ============================================================
        specific_markers = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bincluding\b', r'\bnamely\b',
            r'\baccording to\b', r'\bbased on\b', r'\bresearch shows\b',
            r'\bstudies show\b', r'\bdata shows\b', r'\bevidence suggests\b',
            r'\bin particular\b', r'\bexactly\b', r'\bprecisely\b',
            r'\bapproximately\b', r'\broughly\b', r'\babout \d',
            r'\bknown as\b', r'\bcalled\b', r'\breferred to as\b',
            r'\blocated (?:in|at|near)\b', r'\bfounded (?:in|by)\b',
            r'\bcreated (?:in|by)\b', r'\bpublished (?:in|by)\b',
            r'\bwritten by\b', r'\bauthored by\b', r'\binvented by\b',
            r'\bdiscovered by\b',
        ]
        
        specificity_hits = 0
        for pattern in specific_markers:
            specificity_hits += len(re.findall(pattern, resp_lower))
        
        specificity_score = min(specificity_hits / max(len(words) / 20, 1), 3.0)
        
        # ============================================================
        # FEATURE 4: Vagueness / hedging anti-patterns
        # ============================================================
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things|aspects)\b',
            r'\bin general\b', r'\bgenerally speaking\b',
            r'\bit\'?s? (?:hard|difficult) to say\b',
            r'\bthere (?:is|are) no (?:easy|simple|clear|definitive) answer\b',
            r'\beveryone (?:knows|thinks|believes)\b',
            r'\bas (?:we all|everyone) knows?\b',
            r'\bthings like that\b', r'\band so on\b', r'\betc\.?\b',
            r'\bvarious\b', r'\bnumerous\b(?! characters)',
            r'\ba lot of\b', r'\bquite a (?:few|bit|lot)\b',
            r'\bkind of\b', r'\bsort of\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bmight be\b', r'\bcould be\b',
            r'\bsomewhat\b', r'\bsomehow\b',
        ]
        
        vague_count = 0
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, resp_lower))
        
        vagueness_penalty = min(vague_count * 0.4, 3.0)
        
        # ============================================================
        # FEATURE 5: Information-bearing unique word ratio
        # Ratio of unique content words (excluding stopwords) to total words
        # ============================================================
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
            'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he',
            'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
            'their', 'theirs', 'here', 'there', 'up', 'about', 'also',
        }
        
        content_words = [w for w in words_lower if w not in stopwords and len(w) > 2]
        unique_content = set(content_words)
        
        if len(content_words) > 0:
            content_richness = len(unique_content) / max(len(content_words), 1)
        else:
            content_richness = 0
        
        # ============================================================
        # FEATURE 6: Average word length of content words (longer = more technical/specific)
        # ============================================================
        if content_words:
            avg_content_word_len = sum(len(w) for w in content_words) / len(content_words)
        else:
            avg_content_word_len = 0
        
        word_complexity_score = max(0, (avg_content_word_len - 4.0)) / 4.0  # normalized 0-1ish
        
        # ============================================================
        # FEATURE 7: Parenthetical / explanatory content
        # Parentheses, quotes, colons often signal specific details
        # ============================================================
        parens = len(re.findall(r'\([^)]+\)', resp))
        quotes = len(re.findall(r'["\u201c][^"\u201d]+["\u201d]', resp))
        colons_info = len(re.findall(r':\s*\w', resp))
        
        explanatory_score = min((parens * 0.5 + quotes * 0.5 + colons_info * 0.3), 2.0)
        
        # ============================================================
        # FEATURE 8: Repetition penalty
        # Detect repeated sentences or phrases (sign of low quality)
        # ============================================================
        sentence_list = [s.strip().lower() for s in re.split(r'[.!?\n]', resp) if s.strip()]
        if len(sentence_list) > 1:
            sentence_counter = Counter(sentence_list)
            repeated = sum(c - 1 for c in sentence_counter.values() if c > 1)
            repetition_penalty = min(repeated * 0.8, 3.0)
        else:
            repetition_penalty = 0
        
        # ============================================================
        # FEATURE 9: Response relevance (query term coverage)
        # ============================================================
        query_words = set(re.findall(r'\b\w+\b', query_text.lower())) - stopwords
        if query_words:
            resp_word_set = set(words_lower)
            query_coverage = len(query_words & resp_word_set) / max(len(query_words), 1)
        else:
            query_coverage = 0.5  # neutral if no meaningful query words
        
        # ============================================================
        # FEATURE 10: Substantive length (not too short, diminishing returns for long)
        # ============================================================
        word_count = len(words)
        if word_count < 3:
            length_score = 0.5
        elif word_count < 10:
            length_score = 1.5
        elif word_count < 30:
            length_score = 2.5
        elif word_count < 80:
            length_score = 3.0
        elif word_count < 200:
            length_score = 2.8
        else:
            length_score = 2.5  # very long may have padding
        
        # ============================================================
        # FEATURE 11: Off-topic / garbage detection
        # ============================================================
        # Detect HTML tags, code blocks, random formatting
        html_tags = len(re.findall(r'<[a-zA-Z/][^>]*>', resp))
        code_indicators = len(re.findall(r'\b(?:import|def |class |return |print\(|function\b)', resp))
        
        # Check if response seems like code when query doesn't ask for code
        query_asks_code = bool(re.search(r'\b(?:code|program|script|function|html|css|python|java)\b', query_text.lower()))
        
        if not query_asks_code and (code_indicators > 3 or html_tags > 5):
            offtopic_penalty = min((code_indicators + html_tags) * 0.3, 3.0)
        else:
            offtopic_penalty = 0
        
        # ============================================================
        # FEATURE 12: Factual assertion density
        # Sentences that make declarative claims (subject-verb patterns)
        # ============================================================
        declarative_patterns = [
            r'\b(?:is|are|was|were|has|have|had)\s+(?:a|an|the|one|two|three|\d)',
            r'\b(?:founded|created|built|established|invented|discovered|published|written|born|died)\b',
            r'\b(?:contains|includes|consists|comprises|features|provides|offers|produces)\b',
            r'\b(?:measures|weighs|costs|lasts|takes|requires|needs)\b',
            r'\b(?:located|situated|found|based|headquartered)\b',
        ]
        
        factual_hits = 0
        for pattern in declarative_patterns:
            factual_hits += len(re.findall(pattern, resp_lower))
        
        factual_density = min(factual_hits / max(len(words) / 15, 1), 2.5)
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        score = 2.0  # base score
        
        # Positive contributions
        score += proper_noun_density * 12.0       # up to ~2-3 points
        score += numeric_density * 25.0            # up to ~2 points
        score += specificity_score * 1.0           # up to 3 points
        score += word_complexity_score * 1.0       # up to ~1 point
        score += explanatory_score * 0.5           # up to 1 point
        score += content_richness * 1.5            # up to 1.5 points
        score += factual_density * 1.0             # up to 2.5 points
        score += length_score * 0.8                # up to 2.4 points
        score += query_coverage * 1.5              # up to 1.5 points
        
        # Negative contributions
        score -= vagueness_penalty                 # up to -3 points
        score -= repetition_penalty                # up to -3 points
        score -= offtopic_penalty                  # up to -3 points
        
        # Very short responses that are just a word or two
        if word_count <= 2:
            score = min(score, 2.0)
        
        # Responses that are just punctuation or near-empty
        if len(resp.replace('.', '').replace(' ', '').replace('\n', '')) < 3:
            return 0.5
        
        # Clamp to 0-10
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: return a middling score based on length
        try:
            words = response.strip().split()
            if len(words) < 3:
                return 1.0
            elif len(words) < 20:
                return 3.0
            else:
                return 5.0
        except Exception:
            return 3.0