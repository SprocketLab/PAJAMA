def judging_function(query, response):
    """
    Evaluate evidence density and specificity using a pattern-matching approach
    focused on detecting concrete knowledge signals vs. vague filler.
    
    This variant uses a fundamentally different approach: it builds a "specificity profile"
    by scanning for multiple categories of concrete evidence markers using regex patterns,
    then computes a ratio-based score against detected vagueness/filler patterns.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 0.5
        
        words = response_text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # Lowercase for pattern matching
        resp_lower = response_text.lower()
        
        # ========== CATEGORY 1: Named Entity Signals ==========
        # Detect capitalized multi-word sequences (likely proper nouns/named entities)
        named_entity_pattern = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response_text)
        # Single capitalized words not at sentence start
        mid_sentence_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response_text)
        
        named_entity_score = len(named_entity_pattern) * 2.5 + len(mid_sentence_caps) * 1.0
        
        # ========== CATEGORY 2: Numeric Precision ==========
        # Percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', response_text)
        # Years (4-digit numbers starting with 1 or 2)
        years = re.findall(r'\b(?:1[0-9]{3}|2[0-9]{3})\b', response_text)
        # Decimal numbers
        decimals = re.findall(r'\b\d+\.\d+\b', response_text)
        # Numbers with units
        num_with_units = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|lb|lbs|mg|g|km|miles?|hours?|minutes?|seconds?|days?|years?|months?|weeks?|feet|ft|inches|cm|mm|m|°[CF]|degrees?|watts?|volts?|amps?|Hz|MHz|GHz|TB|GB|MB|KB|mph|kph|psi|rpm)\b', resp_lower)
        # Currency amounts
        currency = re.findall(r'[\$£€¥]\s?\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars?|euros?|pounds?|cents?)', resp_lower)
        # Ordinal numbers
        ordinals = re.findall(r'\b\d+(?:st|nd|rd|th)\b', resp_lower)
        # Plain significant numbers (3+ digits)
        big_numbers = re.findall(r'\b\d{3,}\b', response_text)
        
        numeric_score = (len(percentages) * 3.0 + len(years) * 2.0 + len(decimals) * 2.5 +
                        len(num_with_units) * 3.0 + len(currency) * 3.0 + 
                        len(ordinals) * 1.5 + len(big_numbers) * 1.0)
        
        # ========== CATEGORY 3: Citation/Reference Signals ==========
        # Book/work titles (italicized with * or in quotes)
        italic_titles = re.findall(r'\*[A-Z][^*]{2,50}\*', response_text)
        quoted_titles = re.findall(r'"[A-Z][^"]{2,80}"', response_text)
        # URLs
        urls = re.findall(r'https?://\S+|www\.\S+', response_text)
        # Academic-style references
        academic_refs = re.findall(r'\b(?:et al\.?|ibid|op\.?\s*cit|cf\.|see also|according to)\b', resp_lower)
        # User references (Reddit style)
        user_refs = re.findall(r'u/\w+|r/\w+|@\w+', response_text)
        # Parenthetical citations
        paren_citations = re.findall(r'\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}\)', response_text)
        
        citation_score = (len(italic_titles) * 4.0 + len(quoted_titles) * 3.0 + 
                         len(urls) * 2.0 + len(academic_refs) * 2.0 +
                         len(user_refs) * 1.5 + len(paren_citations) * 4.0)
        
        # ========== CATEGORY 4: Technical/Domain Vocabulary ==========
        # Detect technical patterns: acronyms, compound technical terms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', response_text)
        # Filter out common non-technical acronyms
        common_acronyms = {'I', 'A', 'THE', 'AND', 'OR', 'BUT', 'NOT', 'IF', 'SO', 'AM', 'PM', 'OK', 'IT', 'IS', 'AS', 'AT', 'IN', 'ON', 'TO', 'DO', 'NO', 'OF'}
        tech_acronyms = [a for a in acronyms if a not in common_acronyms and len(a) >= 2]
        
        # Hyphenated compound terms (often technical)
        hyphenated = re.findall(r'\b\w+-\w+(?:-\w+)*\b', response_text)
        
        # Code-like content (backticks, code blocks, SQL, etc.)
        code_blocks = re.findall(r'```[\s\S]*?```', response_text)
        inline_code = re.findall(r'`[^`]+`', response_text)
        code_keywords = re.findall(r'\b(?:SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|def |class |import |return |function |var |let |const )\b', response_text)
        
        technical_score = (len(tech_acronyms) * 1.5 + len(hyphenated) * 0.8 +
                          len(code_blocks) * 5.0 + len(inline_code) * 2.0 +
                          len(code_keywords) * 1.5)
        
        # ========== CATEGORY 5: Structural Specificity ==========
        # Causal/explanatory connectors showing reasoning depth
        causal_connectors = re.findall(r'\b(?:because|therefore|consequently|as a result|due to|since|thus|hence|this means|which leads to|resulting in|caused by|the reason)\b', resp_lower)
        
        # Conditional/nuanced reasoning
        conditional = re.findall(r'\b(?:if you|when you|in the case|for example|for instance|specifically|in particular|such as|namely|e\.g\.|i\.e\.)\b', resp_lower)
        
        # Contrast/distinction markers (shows nuanced understanding)
        contrast = re.findall(r'\b(?:however|whereas|unlike|in contrast|on the other hand|but rather|instead of|the difference|distinction between|while .+ is)\b', resp_lower)
        
        structural_score = (len(causal_connectors) * 1.5 + len(conditional) * 2.0 + 
                           len(contrast) * 1.5)
        
        # ========== CATEGORY 6: Experiential/Concrete Detail ==========
        # First-person experience markers (with specifics)
        experience_markers = re.findall(r'\b(?:I (?:worked|used|built|designed|implemented|tested|found|discovered|experienced|noticed|observed|measured|calculated)|in my experience|when I was|I have (?:been|seen|done|had))\b', resp_lower)
        
        # Temporal specificity
        temporal = re.findall(r'\b(?:in \d{4}|last (?:year|month|week)|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}|\d+ years? ago|recently)\b', resp_lower)
        
        # Location specificity
        location_preps = re.findall(r'\b(?:in|at|from|near)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', response_text)
        
        experiential_score = (len(experience_markers) * 2.0 + len(temporal) * 2.5 + 
                             len(location_preps) * 1.5)
        
        # ========== VAGUENESS PENALTIES ==========
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b', r'\bvarious factors\b',
            r'\bthere are (?:many|various|several|some) (?:ways|reasons|factors|things)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bmore or less\b',
            r'\bkind of\b', r'\bsort of\b', r'\bpretty much\b', r'\bbasically\b',
            r'\bi think\b', r'\bi guess\b', r'\bi suppose\b', r'\bmaybe\b',
            r'\bprobably\b', r'\bperhaps\b', r'\bmight be\b', r'\bcould be\b',
            r'\bi\'m not sure\b', r'\bi don\'t know\b', r'\bnot really sure\b',
            r'\band stuff\b', r'\band things\b', r'\bwhatever\b', r'\byou know\b',
            r'\band so on\b', r'\betc\.?\b', r'\bblah\b', r'\bwhatever works\b',
            r'\bit\'s complicated\b', r'\bit\'s complex\b', r'\bthere\'s no simple\b',
            r'\bsome say\b', r'\bpeople say\b', r'\bthey say\b',
        ]
        
        vague_count = 0
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, resp_lower))
        
        # Empty filler sentences (very short sentences that add nothing)
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        filler_sentences = sum(1 for s in sentences if len(s.split()) <= 3 and len(sentences) > 1)
        
        vagueness_penalty = vague_count * 1.5 + filler_sentences * 0.5
        
        # ========== CATEGORY 7: Information Density (unique content words ratio) ==========
        # Strip common stop words and measure unique meaningful word ratio
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
            'its', 'they', 'them', 'their', 'up', 'about', 'there', 'here',
        }
        
        content_words = [w.lower().strip('.,!?;:"\'-()[]{}') for w in words 
                        if w.lower().strip('.,!?;:"\'-()[]{}') not in stop_words 
                        and len(w.strip('.,!?;:"\'-()[]{}')) > 2]
        
        if content_words:
            unique_content = len(set(content_words))
            total_content = len(content_words)
            # Vocabulary richness - but adjusted for length
            vocab_richness = unique_content / (total_content ** 0.7) if total_content > 0 else 0
        else:
            vocab_richness = 0
        
        # ========== CATEGORY 8: Response Engagement with Query ==========
        # Check if response addresses query-specific terms
        query_lower = query.lower() if query else ""
        query_words = set(w.strip('.,!?;:"\'-()[]{}') for w in query_lower.split() 
                         if len(w.strip('.,!?;:"\'-()[]{}')) > 3 and 
                         w.strip('.,!?;:"\'-()[]{}') not in stop_words)
        
        resp_words_set = set(w.lower().strip('.,!?;:"\'-()[]{}') for w in words)
        
        if query_words:
            query_coverage = len(query_words & resp_words_set) / len(query_words)
        else:
            query_coverage = 0.5
        
        # ========== CATEGORY 9: Sentence Complexity and Elaboration ==========
        avg_sentence_len = word_count / max(len(sentences), 1)
        # Reward moderate-to-long sentences (indicates elaboration), penalize too short
        if avg_sentence_len < 5:
            sentence_complexity = 0.3
        elif avg_sentence_len < 10:
            sentence_complexity = 0.6
        elif avg_sentence_len < 25:
            sentence_complexity = 1.0
        else:
            sentence_complexity = 0.85
        
        # ========== CATEGORY 10: Parenthetical/Clarifying Details ==========
        parentheticals = re.findall(r'\([^)]{5,}\)', response_text)
        em_dashes = re.findall(r'—[^—]{5,}—|--[^-]{5,}--', response_text)
        
        clarification_score = len(parentheticals) * 1.5 + len(em_dashes) * 1.5
        
        # ========== COMBINE SCORES ==========
        # Raw evidence score (sum of all positive categories)
        raw_evidence = (named_entity_score + numeric_score + citation_score + 
                       technical_score + structural_score + experiential_score +
                       clarification_score)
        
        # Normalize by word count to get density (per 100 words)
        evidence_density = (raw_evidence / word_count) * 100 if word_count > 0 else 0
        
        # Vagueness ratio (per 100 words)
        vagueness_density = (vagueness_penalty / word_count) * 100 if word_count > 0 else 0
        
        # Length bonus: longer responses that maintain evidence density get a bonus
        # Use log scale to avoid domination by length alone
        length_factor = min(math.log(word_count + 1, 10) / 2.5, 1.2)  # caps at ~1.2
        
        # Minimum length threshold
        if word_count < 15:
            length_factor *= 0.5
        
        # Compute final composite score
        # Base: evidence density weighted by vocabulary richness and sentence quality
        composite = (
            evidence_density * 1.0 +           # Core evidence density
            vocab_richness * 3.0 +             # Vocabulary richness
            query_coverage * 4.0 +             # Query relevance
            sentence_complexity * 2.0 -         # Sentence quality
            vagueness_density * 2.0             # Vagueness penalty
        )
        
        # Apply length factor
        composite *= length_factor
        
        # Bonus for having evidence across multiple categories (breadth of evidence)
        categories_with_evidence = sum([
            named_entity_score > 0,
            numeric_score > 0,
            citation_score > 0,
            technical_score > 0,
            structural_score > 0,
            experiential_score > 0,
            clarification_score > 0,
        ])
        
        breadth_bonus = categories_with_evidence * 0.8
        composite += breadth_bonus
        
        # Scale to 0-10 range with sigmoid-like mapping
        # Use tanh to smoothly map to bounded range
        scaled = 5.0 * (1.0 + math.tanh((composite - 8.0) / 6.0))
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, scaled))
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.split()) > 20:
                return 4.0
            return 2.0
        except:
            return 2.0