def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality using a 
    hierarchical structure analysis approach based on:
    - Visual structure depth (nesting levels)
    - Content-to-chrome ratio (substance vs formatting artifacts)
    - Structural rhythm analysis (variation in line/paragraph lengths)
    - Coherence flow scoring (sentence-to-sentence connectivity)
    - Repetition/degeneration detection
    """
    try:
        if not response or not response.strip():
            return 0.0
        
        text = response.strip()
        
        # If extremely short (< 10 chars), minimal structure possible
        if len(text) < 10:
            # Very short can be fine if query demands it
            query_len = len(query.strip()) if query else 0
            if query_len > 50 and len(text) < 10:
                return 1.0
            return 2.0
        
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # ============================================================
        # FEATURE 1: Structural Depth Score
        # Measure how many "levels" of structure exist
        # ============================================================
        depth_indicators = set()
        
        # Check for indentation levels
        indent_levels = set()
        for line in non_empty_lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                indent_levels.add(indent // 2)  # normalize to 2-space levels
        
        if indent_levels:
            depth_indicators.add('indentation')
        
        # Check for markdown-style headers (# ## ###)
        import re
        md_headers = [l for l in non_empty_lines if re.match(r'^#{1,6}\s+\S', l.strip())]
        if md_headers:
            depth_indicators.add('md_headers')
        
        # Check for HTML-like tags
        html_tags = re.findall(r'<(\w+)[^>]*>', text)
        structural_html = [t for t in html_tags if t.lower() in 
                          ('h1','h2','h3','h4','h5','h6','p','div','ul','ol','li','blockquote','table','tr','td')]
        if structural_html:
            depth_indicators.add('html_structure')
        
        # Check for numbered items (1. 2. or 1) 2) or (a) (b))
        numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]\s|[a-zA-Z][\.\)]\s|\([a-zA-Z0-9]+\)\s)', text)
        has_numbered = len(numbered_pattern) >= 2
        if has_numbered:
            depth_indicators.add('numbered_list')
        
        # Check for bullet points (-, *, •, ▪, ►)
        bullet_pattern = re.findall(r'(?:^|\n)\s*[\-\*\•\▪\►]\s+\S', text)
        has_bullets = len(bullet_pattern) >= 2
        if has_bullets:
            depth_indicators.add('bullets')
        
        # Check for colon-based definitions (Term: definition)
        colon_defs = re.findall(r'(?:^|\n)\s*[A-Z][^:]{1,40}:\s+\S', text)
        if len(colon_defs) >= 2:
            depth_indicators.add('definitions')
        
        # Check for bold/italic markers
        bold_markers = re.findall(r'\*\*[^*]+\*\*|__[^_]+__', text)
        if bold_markers:
            depth_indicators.add('emphasis')
        
        depth_score = min(len(depth_indicators) * 1.5, 6.0)
        
        # ============================================================
        # FEATURE 2: Structural Rhythm Analysis
        # Good structure has varied but purposeful line/paragraph lengths
        # ============================================================
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Calculate paragraph length variation
        if len(paragraphs) >= 2:
            para_lengths = [len(p) for p in paragraphs]
            mean_len = sum(para_lengths) / len(para_lengths)
            
            if mean_len > 0:
                # Coefficient of variation - some variation is good
                variance = sum((l - mean_len) ** 2 for l in para_lengths) / len(para_lengths)
                std_dev = variance ** 0.5
                cv = std_dev / mean_len
                
                # Sweet spot: CV between 0.2 and 0.8 suggests purposeful variation
                if 0.15 <= cv <= 0.9:
                    rhythm_score = 2.5
                elif cv < 0.15:
                    # Too uniform - might be okay for lists
                    rhythm_score = 1.5
                else:
                    # Too chaotic
                    rhythm_score = 0.8
            else:
                rhythm_score = 0.5
            
            # Bonus for having multiple paragraphs (appropriate chunking)
            para_count_bonus = min(len(paragraphs) * 0.3, 1.5)
            rhythm_score += para_count_bonus
        elif len(paragraphs) == 1:
            # Single block - check if it's appropriately short
            if len(text) < 150:
                rhythm_score = 2.0  # Short single paragraph is fine
            elif len(text) < 400:
                rhythm_score = 1.2  # Medium - could use some breaking up
            else:
                rhythm_score = 0.3  # Wall of text
        else:
            rhythm_score = 0.5
        
        rhythm_score = min(rhythm_score, 4.0)
        
        # ============================================================
        # FEATURE 3: Content-to-Chrome Ratio
        # Measures meaningful content vs noise/artifacts/repetition
        # ============================================================
        
        # Detect repetition (degeneration)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        repetition_penalty = 0.0
        if len(sentences) >= 2:
            seen = {}
            for s in sentences:
                # Normalize whitespace
                normalized = ' '.join(s.split())
                if normalized in seen:
                    seen[normalized] += 1
                else:
                    seen[normalized] = 1
            
            repeated = sum(v - 1 for v in seen.values() if v > 1)
            rep_ratio = repeated / len(sentences) if sentences else 0
            repetition_penalty = min(rep_ratio * 5.0, 4.0)
        
        # Detect line-level repetition
        if len(non_empty_lines) >= 3:
            line_set = set()
            dup_lines = 0
            for line in non_empty_lines:
                norm_line = ' '.join(line.strip().lower().split())
                if norm_line in line_set:
                    dup_lines += 1
                line_set.add(norm_line)
            line_rep_ratio = dup_lines / len(non_empty_lines)
            repetition_penalty += min(line_rep_ratio * 4.0, 3.0)
        
        # Detect garbage/noise patterns
        noise_patterns = [
            r'(?:Input:|Output:){3,}',  # Repeated Input/Output tags
            r'(?:Question:.*?Answer:){3,}',  # Repeated Q&A that wasn't asked for
            r'(.{20,})\1{2,}',  # Long repeated substrings
        ]
        noise_penalty = 0.0
        for pat in noise_patterns:
            if re.search(pat, text, re.DOTALL):
                noise_penalty += 2.0
        
        # Detect if response trails off (truncation indicator)
        truncation_penalty = 0.0
        if text.rstrip()[-1] not in '.!?"\')>}:;' and len(text) > 200:
            # Might be truncated - minor penalty
            truncation_penalty = 0.5
        
        chrome_score = max(0, 4.0 - repetition_penalty - noise_penalty - truncation_penalty)
        
        # ============================================================
        # FEATURE 4: Sentence Flow & Connectivity
        # Measures how well sentences connect to each other
        # ============================================================
        
        # Transition/connective words at sentence starts
        connective_starters = {
            'however', 'moreover', 'furthermore', 'additionally', 'therefore',
            'consequently', 'nevertheless', 'meanwhile', 'similarly', 'conversely',
            'in addition', 'on the other hand', 'as a result', 'for example',
            'for instance', 'in contrast', 'in particular', 'specifically',
            'first', 'second', 'third', 'finally', 'next', 'then', 'also',
            'thus', 'hence', 'indeed', 'notably', 'importantly',
            'that said', 'in summary', 'to summarize', 'in conclusion',
            'overall', 'this means', 'this suggests', 'this indicates',
            'the', 'it', 'they', 'he', 'she', 'we', 'these', 'those', 'such'
        }
        
        flow_score = 0.0
        if len(sentences) >= 2:
            connected = 0
            for s in sentences[1:]:
                words = s.split()
                if words:
                    first_word = words[0].lower().strip(',')
                    first_two = ' '.join(words[:2]).lower() if len(words) >= 2 else ''
                    first_three = ' '.join(words[:3]).lower() if len(words) >= 3 else ''
                    
                    if (first_word in connective_starters or 
                        first_two in connective_starters or
                        first_three in connective_starters):
                        connected += 1
            
            connectivity_ratio = connected / (len(sentences) - 1) if len(sentences) > 1 else 0
            # Some connectivity is good, but not every sentence needs a connector
            if 0.1 <= connectivity_ratio <= 0.6:
                flow_score = 2.5
            elif connectivity_ratio > 0.6:
                flow_score = 2.0
            elif connectivity_ratio > 0:
                flow_score = 1.5
            else:
                flow_score = 0.8
        else:
            flow_score = 1.5  # Single sentence - neutral
        
        flow_score = min(flow_score, 3.0)
        
        # ============================================================
        # FEATURE 5: Proportionality Score
        # Is the response length proportional to the query complexity?
        # ============================================================
        query_text = query.strip() if query else ""
        query_words = len(query_text.split())
        response_words = len(text.split())
        
        # Estimate query complexity
        question_indicators = len(re.findall(r'\?', query_text))
        multi_part = max(1, question_indicators)
        
        # Complex queries with multiple sub-questions
        list_indicators = len(re.findall(r'(?:and|,|\d\.)', query_text))
        complexity = min(query_words / 5.0, 10) + multi_part + min(list_indicators * 0.3, 2)
        
        # Expected response length range
        if complexity < 3:
            # Simple query
            ideal_min, ideal_max = 5, 150
        elif complexity < 6:
            ideal_min, ideal_max = 20, 400
        else:
            ideal_min, ideal_max = 50, 800
        
        if ideal_min <= response_words <= ideal_max:
            proportion_score = 2.0
        elif response_words < ideal_min:
            proportion_score = max(0.3, 2.0 * response_words / ideal_min)
        else:
            # Over-verbose
            excess = response_words / ideal_max
            proportion_score = max(0.5, 2.0 / excess)
        
        proportion_score = min(proportion_score, 2.0)
        
        # ============================================================
        # FEATURE 6: Clean Start and End
        # Good responses start cleanly and end properly
        # ============================================================
        framing_score = 0.0
        
        # Check for clean start (not starting with lowercase, random punctuation, etc.)
        first_char = text[0]
        if first_char.isupper() or first_char in '#*-•1':
            framing_score += 0.5
        elif first_char.islower():
            framing_score += 0.1
        
        # Check for clean ending
        last_char = text.rstrip()[-1] if text.rstrip() else ''
        if last_char in '.!?"\')}':
            framing_score += 0.5
        elif last_char in ':;,':
            framing_score += 0.1
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        # depth_score: 0-6
        # rhythm_score: 0-4  
        # chrome_score: 0-4
        # flow_score: 0-3
        # proportion_score: 0-2
        # framing_score: 0-1
        # Total max ~20, normalize to 0-10
        
        raw_score = (
            depth_score * 0.8 +      # Structure depth matters
            rhythm_score * 1.2 +      # Rhythm is important for readability
            chrome_score * 1.5 +      # Clean content is crucial
            flow_score * 0.8 +        # Flow connectivity
            proportion_score * 1.0 +  # Proportionality
            framing_score * 0.7       # Clean framing
        )
        
        # Normalize: theoretical max is about 6*0.8 + 4*1.2 + 4*1.5 + 3*0.8 + 2*1.0 + 1*0.7 = 4.8+4.8+6+2.4+2+0.7 = 20.7
        normalized = (raw_score / 20.7) * 10.0
        
        # Apply floor and ceiling
        final_score = max(0.5, min(10.0, normalized))
        
        # Round to 1 decimal
        return round(final_score, 1)
        
    except Exception:
        # Fallback: return a middle-ground score
        try:
            if not response or len(response.strip()) < 5:
                return 1.0
            return 4.0
        except Exception:
            return 3.0