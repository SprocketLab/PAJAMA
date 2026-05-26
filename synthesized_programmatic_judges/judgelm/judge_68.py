def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    
    This variant focuses on:
    - Information density and signal-to-noise ratio
    - Sentence structure variety and rhythm
    - Repetition detection (both content and structural)
    - Proportional formatting (formatting appropriate to content length)
    - Coherent flow via sentence-level analysis
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
        
        query = query.strip() if query else ""
        
        score = 5.0  # Start at midpoint
        
        # ============================================================
        # 1. RESPONSE LENGTH ADEQUACY (relative to query complexity)
        # ============================================================
        query_words = len(query.split())
        resp_words = len(response.split())
        resp_chars = len(response)
        
        # Very short responses are usually bad
        if resp_words <= 2:
            return max(0.5, 1.0 + (resp_words * 0.25))
        
        if resp_words < 5:
            score -= 2.0
        elif resp_words < 10:
            score -= 1.0
        elif resp_words >= 20:
            score += 0.5
        
        # ============================================================
        # 2. SENTENCE STRUCTURE VARIETY AND RHYTHM
        # ============================================================
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        if num_sentences > 1:
            # Measure sentence length variety (std dev of word counts)
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                # Coefficient of variation - some variety is good
                cv = std_dev / mean_len if mean_len > 0 else 0
                
                # Moderate variety (cv between 0.2 and 0.8) is ideal
                if 0.15 <= cv <= 0.9:
                    score += 0.8
                elif cv < 0.15 and num_sentences > 3:
                    # All sentences same length = monotonous
                    score -= 0.5
            
            # Check sentence opening variety
            openers = []
            for s in sentences:
                words = s.split()
                if words:
                    openers.append(words[0].lower())
            
            if len(openers) > 2:
                opener_counts = Counter(openers)
                most_common_freq = opener_counts.most_common(1)[0][1]
                repetition_ratio = most_common_freq / len(openers)
                if repetition_ratio > 0.6:
                    score -= 0.7  # Repetitive sentence starts
                elif repetition_ratio < 0.35:
                    score += 0.4  # Good variety
        
        # ============================================================
        # 3. REPETITION DETECTION (content-level)
        # ============================================================
        # Check for repeated phrases (n-grams)
        words_lower = response.lower().split()
        
        if len(words_lower) >= 6:
            # Check trigram repetition
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower) - 2)]
            trigram_counts = Counter(trigrams)
            
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                trigram_rep_ratio = repeated_trigrams / len(trigram_counts) if trigram_counts else 0
                
                if trigram_rep_ratio > 0.15:
                    score -= 2.0  # Heavy repetition
                elif trigram_rep_ratio > 0.08:
                    score -= 1.0
        
        # Check for repeated lines/paragraphs
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if len(lines) > 1:
            line_counts = Counter(lines)
            duplicate_lines = sum(c - 1 for c in line_counts.values() if c > 1)
            dup_ratio = duplicate_lines / len(lines) if lines else 0
            if dup_ratio > 0.3:
                score -= 2.5
            elif dup_ratio > 0.1:
                score -= 1.0
        
        # ============================================================
        # 4. STRUCTURAL ELEMENT DETECTION (different approach: regex patterns for structure types)
        # ============================================================
        
        # Detect enumeration patterns (numbered items, lettered items)
        numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|[a-zA-Z][\.\)])\s+\S', response)
        has_enumeration = len(numbered_pattern) >= 2
        
        # Detect bullet-like patterns (various bullet chars)
        bullet_pattern = re.findall(r'(?:^|\n)\s*[-•*▪►→✓✗◦‣⁃]\s+\S', response)
        has_bullets = len(bullet_pattern) >= 2
        
        # Detect key-value / label patterns like "Label: value"
        kv_pattern = re.findall(r'(?:^|\n)\s*[A-Z][A-Za-z\s]{1,25}:\s+\S', response)
        has_kv_structure = len(kv_pattern) >= 2
        
        # Detect markdown-style headers
        md_headers = re.findall(r'(?:^|\n)\s*#{1,6}\s+\S', response)
        
        # Detect emphasis patterns (bold, italic in markdown)
        emphasis = re.findall(r'\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_', response)
        
        # Detect section-like breaks (blank lines separating content)
        double_newlines = len(re.findall(r'\n\s*\n', response))
        
        # Score structural elements proportionally
        structural_elements = 0
        if has_enumeration:
            structural_elements += 1
            score += 1.0
        if has_bullets:
            structural_elements += 1
            score += 0.8
        if has_kv_structure:
            structural_elements += 1
            score += 0.7
        if md_headers:
            structural_elements += 1
            score += 0.6
        if emphasis:
            structural_elements += 0.5
            score += 0.3
        
        # ============================================================
        # 5. PARAGRAPH QUALITY (different from line/paragraph counting)
        # ============================================================
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        if resp_words > 50:
            # Long responses should have some paragraph breaks
            if num_paragraphs == 1 and not has_enumeration and not has_bullets:
                # Wall of text penalty
                score -= 1.5
            elif num_paragraphs >= 2:
                score += 0.5
                
                # Check paragraph size balance
                para_sizes = [len(p.split()) for p in paragraphs]
                if para_sizes:
                    max_para = max(para_sizes)
                    min_para = min(para_sizes)
                    if max_para > 0 and min_para / max_para < 0.05 and num_paragraphs > 2:
                        score -= 0.3  # Very unbalanced paragraphs
        
        # ============================================================
        # 6. SIGNAL-TO-NOISE RATIO
        # ============================================================
        # Detect filler/noise patterns
        noise_patterns = [
            r'(?:^|\n)\s*(?:Input|Output)\s*:?\s*$',  # Empty Input/Output labels
            r'(?:^|\n)\s*```\s*$',  # Orphan code fences
            r'(?:Question|Answer)\s*:\s*(?:Question|Answer)',  # Recursive Q&A
            r'(?:^|\n)\s*\.\s*$',  # Lines that are just periods
        ]
        
        noise_count = 0
        for pat in noise_patterns:
            noise_count += len(re.findall(pat, response))
        
        if noise_count > 0:
            score -= min(noise_count * 0.5, 2.0)
        
        # Detect runaway/garbled output
        # Check for excessive special characters relative to alphanumeric
        alpha_chars = sum(1 for c in response if c.isalpha())
        special_chars = sum(1 for c in response if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]{}')
        
        if resp_chars > 0:
            alpha_ratio = alpha_chars / resp_chars
            if alpha_ratio < 0.3:
                score -= 1.5  # Too few alphabetic characters
            
            if special_chars > 0 and alpha_chars > 0:
                special_ratio = special_chars / alpha_chars
                if special_ratio > 0.3:
                    score -= 1.0
        
        # ============================================================
        # 7. COMPLETENESS INDICATORS
        # ============================================================
        # Check if response appears truncated
        truncated = response.rstrip()
        if truncated and truncated[-1] not in '.!?:;"\')]}…' and resp_words > 20:
            # Might be truncated, mild penalty
            score -= 0.3
        
        # Check for proper sentence ending
        has_proper_ending = bool(re.search(r'[.!?]\s*$', response.strip()))
        if has_proper_ending and resp_words > 5:
            score += 0.3
        
        # ============================================================
        # 8. CONTEXTUAL APPROPRIATENESS OF FORMATTING
        # ============================================================
        # Short factual queries might be best answered concisely
        # Check if query asks for lists/steps/multiple items
        list_query_signals = re.findall(
            r'\b(?:list|steps|ways|reasons|examples|types|kinds|identify|compare|differences|similarities|pros|cons|advantages|disadvantages)\b',
            query.lower()
        )
        
        if list_query_signals and (has_enumeration or has_bullets or has_kv_structure):
            score += 1.0  # Good: query asks for list, response uses list format
        elif list_query_signals and resp_words > 30 and not has_enumeration and not has_bullets:
            score -= 0.5  # Query asks for list but response is prose
        
        # Short queries with short correct answers shouldn't be penalized for brevity
        if query_words <= 10 and 5 <= resp_words <= 30 and num_sentences <= 3:
            # Concise answer - don't penalize
            score = max(score, 5.0)
        
        # ============================================================
        # 9. INTERNAL COHERENCE MARKERS
        # ============================================================
        # Check for logical connectors and discourse markers
        coherence_markers = re.findall(
            r'\b(?:however|therefore|furthermore|moreover|additionally|consequently|'
            r'in addition|as a result|for example|for instance|in contrast|'
            r'on the other hand|first|second|third|finally|in conclusion|'
            r'to summarize|specifically|notably|importantly)\b',
            response.lower()
        )
        
        if len(coherence_markers) >= 2 and resp_words > 30:
            score += 0.6
        elif len(coherence_markers) >= 1:
            score += 0.3
        
        # ============================================================
        # 10. ANTI-PATTERNS (strong negative signals)
        # ============================================================
        # Detect responses that seem to be confused/wrong format
        code_dump = bool(re.search(r'(?:import\s+\w+|def\s+\w+\(|class\s+\w+)', response))
        query_asks_code = bool(re.search(r'\b(?:code|program|function|script|implement|python|java|html|css)\b', query.lower()))
        
        if code_dump and not query_asks_code:
            score -= 2.0  # Code dump when not asked for code
        
        # Detect echo/repetition of the query without adding value
        if query and len(query) > 10:
            query_in_response = query.lower() in response.lower()
            if query_in_response and resp_words < query_words * 2:
                score -= 0.5
        
        # Detect HTML dumps when not asked
        html_tags = re.findall(r'<(?:h[1-6]|p|div|blockquote|span|br|hr)\b[^>]*>', response, re.IGNORECASE)
        query_asks_html = bool(re.search(r'\b(?:html|tag|webpage|web page)\b', query.lower()))
        if len(html_tags) > 3 and not query_asks_html:
            score -= 1.0
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Clamp score to [0, 10]
        score = max(0.0, min(10.0, score))
        
        return round(score, 2)
    
    except Exception:
        return 3.0