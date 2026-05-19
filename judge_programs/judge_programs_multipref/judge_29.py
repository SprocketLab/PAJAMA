def judging_function(query, response):
    """
    Evaluates factual accuracy indicators using a citation/specificity analysis approach.
    
    This variant focuses on:
    1. Named entity density (proper nouns, specific references)
    2. Numeric/quantitative claim analysis (dates, measurements, statistics)
    3. Hedging vs absolute claim ratio
    4. Sensationalism/conspiracy language detection
    5. Source/citation indicators
    6. Structural credibility signals (organized reasoning, step-by-step)
    7. Qualifier-to-assertion ratio
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        words = resp_lower.split()
        
        if len(words) < 3:
            return 0.5
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        score = 50.0  # Start at midpoint
        
        # ========== 1. Named Entity / Proper Noun Density ==========
        # Count capitalized words that aren't sentence starters
        proper_nouns = 0
        response_words = response.split()
        for i, w in enumerate(response_words):
            if i == 0:
                continue
            # Check if previous character was not a sentence ender
            clean_w = re.sub(r'[^a-zA-Z]', '', w)
            if len(clean_w) > 1 and clean_w[0].isupper() and not clean_w.isupper():
                # Check it's not after a period
                prev_text = ' '.join(response_words[:i])
                if prev_text and prev_text[-1] not in '.!?:':
                    proper_nouns += 1
        
        proper_noun_ratio = proper_nouns / max(len(response_words), 1)
        # Moderate proper noun density is good (specific but not excessive)
        if 0.02 <= proper_noun_ratio <= 0.15:
            score += 3.0
        elif proper_noun_ratio > 0.15:
            score += 1.0  # Might be overly name-heavy
        
        # ========== 2. Numeric/Quantitative Claims Analysis ==========
        # Find all numbers in the response
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|°|degrees|kg|m/s|mph|km|miles|feet|meters|lbs|oz|cm|mm|inches)?\b', response)
        num_count = len(numbers)
        
        # Dates (years)
        dates = re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b', response)
        date_count = len(dates)
        
        # Overly precise unsourced statistics are a red flag
        suspicious_stats = re.findall(r'\b\d{2,}\.\d{2,}%', response)  # e.g., "73.47%"
        overly_precise = len(suspicious_stats)
        
        # Reasonable use of numbers is good
        num_density = num_count / max(num_sentences, 1)
        if 0.1 <= num_density <= 2.0:
            score += 3.0
        elif num_density > 2.0:
            score += 1.0  # Could be good or could be hallucinated
        
        if date_count > 0:
            score += 2.0  # Temporal specificity
        
        # Penalize overly precise unsourced stats
        score -= overly_precise * 2.0
        
        # ========== 3. Hedging and Uncertainty Language ==========
        hedging_phrases = [
            'may ', 'might ', 'could ', 'possibly', 'perhaps', 'likely',
            'it is possible', 'it seems', 'appears to', 'suggests that',
            'generally', 'typically', 'usually', 'often', 'tends to',
            'in most cases', 'it depends', 'not necessarily',
            'approximately', 'roughly', 'around ', 'about ',
            'as far as i know', 'to my knowledge', 'i believe',
            'it\'s worth noting', 'keep in mind', 'note that',
            'however', 'although', 'on the other hand', 'that said',
            'depending on', 'can vary', 'varies'
        ]
        
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in resp_lower)
        
        # Absolute/overconfident language
        absolute_phrases = [
            'always ', 'never ', 'definitely ', 'certainly ',
            'without a doubt', 'absolutely ', 'guaranteed',
            'there is no way', 'impossible', 'everyone knows',
            'it is a fact that', 'undeniably', 'unquestionably',
            'the truth is', 'the fact is', 'obviously ',
            'clearly ', 'no question', '100%', 'proven fact'
        ]
        
        absolute_count = sum(1 for phrase in absolute_phrases if phrase in resp_lower)
        
        # Good ratio: more hedging relative to absolutes
        hedge_ratio = hedge_count / max(hedge_count + absolute_count, 1)
        if hedge_ratio >= 0.6:
            score += 4.0
        elif hedge_ratio >= 0.4:
            score += 2.0
        elif absolute_count > 3:
            score -= 3.0
        
        # ========== 4. Sensationalism and Conspiracy Detection ==========
        sensational_words = [
            'shocking', 'bombshell', 'explosive', 'mind-blowing',
            'they don\'t want you to know', 'wake up', 'sheeple',
            'mainstream media', 'cover-up', 'coverup', 'conspiracy',
            'big pharma', 'deep state', 'hoax', 'scam ',
            'unbelievable', 'jaw-dropping', 'insane ', 'crazy ',
            'you won\'t believe', 'secret ', 'hidden truth',
            'exposed', 'breaking:', 'urgent:', 'warning!',
            'miracle ', 'cure-all', 'guaranteed to'
        ]
        
        sensational_count = sum(1 for w in sensational_words if w in resp_lower)
        score -= sensational_count * 3.0
        
        # ========== 5. Source/Citation Indicators ==========
        citation_patterns = [
            r'according to', r'research shows', r'studies suggest',
            r'a study', r'researchers', r'experts',
            r'published in', r'journal of', r'university of',
            r'report(?:ed|s)?', r'data from', r'survey',
            r'source:', r'reference', r'cited',
            r'for (?:more|further) information', r'see also',
            r'based on', r'evidence suggests'
        ]
        
        citation_count = sum(1 for pat in citation_patterns if re.search(pat, resp_lower))
        score += min(citation_count * 1.5, 6.0)
        
        # ========== 6. Structural Credibility ==========
        # Numbered/ordered steps indicate organized thinking
        numbered_steps = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*\*(?:Step|step)\s+\d+)', response)
        has_structure = len(numbered_steps) >= 2
        
        # Markdown formatting (bold, headers) indicates organized presentation
        bold_items = re.findall(r'\*\*[^*]+\*\*', response)
        has_bold = len(bold_items) >= 1
        
        # Check for ### headers
        headers = re.findall(r'#{1,4}\s+\S+', response)
        has_headers = len(headers) >= 1
        
        structural_score = 0
        if has_structure:
            structural_score += 2.5
        if has_bold:
            structural_score += 1.5
        if has_headers:
            structural_score += 1.5
        score += min(structural_score, 5.0)
        
        # ========== 7. Response Completeness and Engagement ==========
        # Check if response appears cut off (ends mid-sentence without punctuation)
        stripped = response.rstrip()
        if stripped and stripped[-1] not in '.!?"\')':
            score -= 2.0  # Likely truncated
        
        # Check for engagement with the query
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'from', 'this', 'that', 'with', 'what', 'how',
                      'why', 'when', 'where', 'who', 'which', 'will', 'would',
                      'could', 'should', 'been', 'were', 'they', 'them',
                      'some', 'than', 'other', 'into', 'more', 'about'}
        query_content = query_words - stop_words
        
        if query_content:
            resp_words_set = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
            overlap = len(query_content & resp_words_set) / len(query_content)
            score += overlap * 4.0  # Up to 4 points for relevance
        
        # ========== 8. Explanatory Depth ==========
        # Causal/explanatory connectors indicate reasoning
        explanatory_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'this means', 'in other words',
            'for example', 'for instance', 'such as',
            'specifically', 'in particular', 'namely',
            'the reason', 'due to', 'caused by', 'leads to',
            'which means', 'this is because', 'this allows'
        ]
        
        explanation_count = sum(1 for m in explanatory_markers if m in resp_lower)
        score += min(explanation_count * 1.0, 5.0)
        
        # ========== 9. Vocabulary Sophistication (without being jargon-heavy) ==========
        # Average word length as proxy for vocabulary level
        word_lengths = [len(w) for w in words if w.isalpha()]
        if word_lengths:
            avg_word_len = sum(word_lengths) / len(word_lengths)
            # Sweet spot: 4.5-6.5 character average
            if 4.5 <= avg_word_len <= 6.5:
                score += 2.0
            elif avg_word_len > 6.5:
                score += 0.5  # Might be overly technical
        
        # ========== 10. Tone Appropriateness ==========
        # Exclamation marks (too many = sensational)
        exclamation_count = response.count('!')
        if exclamation_count > 3:
            score -= (exclamation_count - 3) * 0.5
        
        # ALL CAPS words (shouting/sensational)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', response)
        # Filter out common abbreviations
        common_abbrevs = {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'YOU', 'ALL',
                          'CAN', 'HAS', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT',
                          'USA', 'FBI', 'CIA', 'NASA', 'HTML', 'CSS', 'API', 'SQL',
                          'HTTP', 'URL', 'PDF', 'CEO', 'CFO', 'CTO', 'PhD', 'DNA',
                          'RNA', 'GPS', 'BBQ', 'DIY', 'FAQ', 'LED', 'RAM', 'ROM'}
        non_abbrev_caps = [w for w in caps_words if w not in common_abbrevs]
        if len(non_abbrev_caps) > 2:
            score -= len(non_abbrev_caps) * 0.5
        
        # ========== 11. Introductory Framing ==========
        # Good responses often start with context-setting
        intro_patterns = [
            r'^(?:certainly|great question|that\'s a)',
            r'^(?:here|let me|i\'d be happy)',
            r'^(?:to answer|in response|regarding)',
        ]
        has_good_intro = any(re.match(pat, resp_lower) for pat in intro_patterns)
        if has_good_intro:
            score += 1.5
        
        # ========== 12. Balanced Perspective ==========
        # Check for acknowledgment of multiple viewpoints
        balance_markers = [
            'on the other hand', 'however', 'alternatively',
            'pros and cons', 'advantages and disadvantages',
            'while ', 'although ', 'despite ', 'conversely',
            'some argue', 'others believe', 'different perspectives'
        ]
        balance_count = sum(1 for m in balance_markers if m in resp_lower)
        score += min(balance_count * 1.0, 3.0)
        
        # ========== 13. Response Length Quality ==========
        # Not too short, not excessively long relative to query complexity
        query_word_count = len(query_lower.split())
        resp_word_count = len(words)
        
        # Reasonable length bonus
        if resp_word_count >= 30:
            score += 2.0
        if resp_word_count >= 80:
            score += 1.5
        if resp_word_count >= 150:
            score += 1.0
        
        # ========== 14. Sentence Variety ==========
        # Good writing has varied sentence lengths
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len  # Coefficient of variation
                if 0.3 <= cv <= 1.0:
                    score += 2.0  # Good variety
                elif cv < 0.3:
                    score -= 0.5  # Too uniform (robotic)
        
        # Clamp score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 25.0