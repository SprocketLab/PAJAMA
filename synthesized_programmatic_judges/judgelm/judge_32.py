def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a sentence-level structural analysis approach.
    
    This variant focuses on:
    1. Sentence-level quality scoring (analyzing each sentence independently)
    2. Information density via named entity proxies (capitalized words, numbers)
    3. Discourse coherence markers
    4. Red flag detection at sentence level
    5. Response completeness signals
    
    Different from other variants by using sentence-level decomposition and 
    aggregating per-sentence quality signals rather than document-level word overlap.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # === SENTENCE DECOMPOSITION ===
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        # === 1. SENTENCE-LEVEL QUALITY SCORING ===
        sentence_scores = []
        for sent in sentences:
            score = 0.0
            words = sent.split()
            if not words:
                sentence_scores.append(0.0)
                continue
            
            # Reasonable sentence length (5-40 words is good)
            wlen = len(words)
            if 5 <= wlen <= 40:
                score += 2.0
            elif 3 <= wlen < 5:
                score += 1.0
            elif wlen > 40:
                score += 0.5  # run-on
            else:
                score += 0.3
            
            # Contains a verb-like word (ends in common verb suffixes)
            verb_patterns = re.findall(r'\b\w+(ed|ing|es|tion|ment|ize|ise|ify|ate)\b', sent.lower())
            if verb_patterns:
                score += 0.5
            
            # Contains specific factual indicators
            # Numbers (dates, quantities, measurements)
            numbers = re.findall(r'\b\d+[\d,.-]*\b', sent)
            if numbers:
                score += 1.5 * min(len(numbers), 3) / 3.0
            
            # Capitalized proper nouns (not sentence-start)
            words_after_first = words[1:] if len(words) > 1 else []
            proper_nouns = [w for w in words_after_first if w[0].isupper() and w.isalpha() and len(w) > 1]
            if proper_nouns:
                score += 1.0 * min(len(proper_nouns), 4) / 4.0
            
            # Parenthetical information (often citations or clarifications)
            if '(' in sent and ')' in sent:
                score += 0.8
            
            # Quotation marks (citing sources)
            if '"' in sent or "'" in sent or '\u201c' in sent:
                score += 0.3
            
            sentence_scores.append(score)
        
        avg_sentence_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0
        
        # === 2. INFORMATION DENSITY ANALYSIS ===
        all_words = response_stripped.split()
        total_words = len(all_words)
        
        if total_words == 0:
            return 0.5
        
        # Unique content words (not stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'i',
            'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
            'his', 'her', 'their', 'what', 'which', 'who', 'whom'
        }
        
        content_words = [w.lower().strip('.,!?;:()[]"\'') for w in all_words 
                        if w.lower().strip('.,!?;:()[]"\'') not in stopwords and len(w) > 1]
        
        unique_content = set(content_words)
        content_ratio = len(unique_content) / max(total_words, 1)
        
        # Information density: unique content words per sentence
        info_density = len(unique_content) / max(len(sentences), 1)
        info_density_score = min(info_density / 5.0, 1.0) * 3.0  # normalize to 0-3
        
        # === 3. DISCOURSE COHERENCE MARKERS ===
        coherence_score = 0.0
        
        # Transition words and connectives
        transitions = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'although', 'whereas', 'meanwhile',
            'specifically', 'particularly', 'notably', 'importantly', 'similarly',
            'conversely', 'alternatively', 'subsequently', 'accordingly', 'thus',
            'hence', 'indeed', 'in fact', 'for example', 'for instance',
            'in addition', 'on the other hand', 'as a result', 'in contrast',
            'in particular', 'such as', 'including', 'according to'
        ]
        
        response_lower = response_stripped.lower()
        transition_count = sum(1 for t in transitions if t in response_lower)
        coherence_score += min(transition_count * 0.5, 2.0)
        
        # Appropriate hedging (shows epistemic awareness)
        hedging_phrases = [
            'it is possible', 'it appears', 'it seems', 'likely', 'unlikely',
            'may be', 'might be', 'could be', 'generally', 'typically',
            'often', 'usually', 'tend to', 'in most cases', 'approximately',
            'roughly', 'about', 'estimated', 'believed to', 'considered to',
            'widely regarded', 'commonly', 'it is thought', 'suggests that',
            'indicates that', 'depending on', 'varies', 'can vary',
            'it is difficult to', 'not always', 'in some cases'
        ]
        
        hedge_count = sum(1 for h in hedging_phrases if h in response_lower)
        hedging_score = min(hedge_count * 0.6, 2.0)
        
        # === 4. RED FLAG DETECTION (per-sentence) ===
        red_flag_penalty = 0.0
        
        # Sensationalism / conspiracy markers
        sensational_words = [
            'shocking', 'unbelievable', 'mind-blowing', 'conspiracy', 'cover-up',
            'they don\'t want you to know', 'the truth is', 'wake up', 'sheeple',
            'big pharma', 'mainstream media lies', 'exposed', 'bombshell',
            'you won\'t believe', 'secret agenda', 'deep state', 'hoax',
            'fake news', 'brainwash', 'propaganda'
        ]
        for sw in sensational_words:
            if sw in response_lower:
                red_flag_penalty += 1.5
        
        # Overly precise unsourced statistics
        precise_stats = re.findall(r'\b\d{1,3}\.\d{2,}%\b', response_stripped)
        red_flag_penalty += len(precise_stats) * 0.5
        
        # Absolute claims without qualification
        absolute_phrases = [
            'always', 'never', 'everyone knows', 'it is certain',
            'without a doubt', 'undeniably', 'proven fact', 'absolutely',
            'no one can deny', 'the only', 'guaranteed'
        ]
        abs_count = sum(1 for ap in absolute_phrases if ap in response_lower)
        red_flag_penalty += abs_count * 0.3
        
        # Repetitive content (hallucination signal)
        if len(sentences) >= 3:
            sent_texts = [s.lower().strip() for s in sentences]
            repeated = len(sent_texts) - len(set(sent_texts))
            red_flag_penalty += repeated * 1.0
        
        # Repeated phrases/fragments
        if total_words > 10:
            trigrams = [' '.join(all_words[i:i+3]).lower() for i in range(len(all_words)-2)]
            trigram_counts = Counter(trigrams)
            high_repeat_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            red_flag_penalty += high_repeat_trigrams * 0.4
        
        # === 5. RESPONSE COMPLETENESS & RELEVANCE ===
        completeness_score = 0.0
        
        # Length adequacy (not too short, not excessively long)
        if total_words < 3:
            completeness_score -= 3.0
        elif total_words < 10:
            completeness_score -= 1.0
        elif 10 <= total_words <= 300:
            completeness_score += 2.0
        elif total_words > 300:
            completeness_score += 1.0  # might be verbose
        
        # Ends with proper punctuation (not cut off)
        last_char = response_stripped[-1] if response_stripped else ''
        if last_char in '.!?)':
            completeness_score += 0.5
        elif last_char == ':' or response_stripped.endswith('...'):
            completeness_score -= 0.3
        
        # Check if response seems cut off mid-sentence
        if response_stripped.endswith((',', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'of', 'to')):
            completeness_score -= 1.0
        
        # Query-response relevance via shared content words
        query_words = set(w.lower().strip('.,!?;:()[]"\'') for w in query.split() 
                         if w.lower().strip('.,!?;:()[]"\'') not in stopwords and len(w) > 1)
        response_content_set = unique_content
        
        if query_words:
            overlap = len(query_words & response_content_set)
            relevance_ratio = overlap / len(query_words)
            completeness_score += relevance_ratio * 2.0
        
        # === 6. STRUCTURAL QUALITY ===
        structural_score = 0.0
        
        # Multiple meaningful sentences
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 3]
        if len(meaningful_sentences) >= 3:
            structural_score += 1.5
        elif len(meaningful_sentences) >= 2:
            structural_score += 1.0
        elif len(meaningful_sentences) == 1:
            structural_score += 0.5
        
        # Contains structured elements (lists, colons for definitions)
        if re.search(r'^\s*[-•*]\s', response_stripped, re.MULTILINE):
            structural_score += 0.5
        if re.search(r':\s', response_stripped):
            structural_score += 0.3
        
        # HTML/code contamination penalty
        html_tags = re.findall(r'<[^>]+>', response_stripped)
        # Allow some tags but penalize excessive code
        if len(html_tags) > 3:
            structural_score -= 1.5
        
        # Random code/programming content when not asked for
        query_lower = query.lower()
        is_code_query = any(kw in query_lower for kw in ['code', 'program', 'function', 'html', 'python', 'script', 'tag'])
        if not is_code_query:
            code_indicators = ['import ', 'def ', 'class ', 'return ', 'print(', '#!/', 'var ', 'function(']
            code_count = sum(1 for ci in code_indicators if ci in response_stripped)
            if code_count >= 2:
                structural_score -= 2.0
        
        # === 7. GIBBERISH / LOW-EFFORT DETECTION ===
        low_effort_penalty = 0.0
        
        # Very short responses to complex queries
        query_complexity = len(query.split())
        if total_words < 5 and query_complexity > 5:
            low_effort_penalty += 3.0
        
        # Single word/character responses
        if total_words <= 1:
            low_effort_penalty += 4.0
        
        # Response is just a period or minimal
        if response_stripped in ['.', '..', '...', '-', '--', 'no', 'yes', 'ok', 'n/a']:
            low_effort_penalty += 5.0
        
        # Excessive special characters ratio
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        total_chars = len(response_stripped)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.3:
                low_effort_penalty += 2.0
        
        # === AGGREGATE SCORE ===
        # Weighted combination
        raw_score = (
            avg_sentence_score * 1.2 +      # sentence quality (0-~5 range)
            info_density_score * 1.0 +        # information density (0-3)
            coherence_score * 1.0 +           # discourse coherence (0-2)
            hedging_score * 0.8 +             # appropriate hedging (0-2)
            completeness_score * 0.9 +        # completeness (~ -3 to 4.5)
            structural_score * 0.7 +          # structure (~ -2 to 2.3)
            content_ratio * 2.0 -             # vocabulary richness
            red_flag_penalty * 0.8 -          # red flags
            low_effort_penalty * 1.0          # low effort
        )
        
        # Normalize to 0-10 scale using sigmoid-like mapping
        # Expected raw range: roughly -8 to 15
        # Shift and scale
        normalized = (raw_score + 3.0) / 18.0 * 10.0
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except Exception:
            return 3.0