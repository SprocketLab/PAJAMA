def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    focused on detecting named entities, precise references, technical terms,
    and information-rich sentence structures.
    
    This variant uses a sentence-level analysis approach, scoring each sentence
    for its "information payload" and aggregating across the response.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        # === FEATURE 1: Capitalized Named Entities ===
        # Detect words that look like proper nouns / named entities
        # (capitalized words not at sentence start)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        named_entity_count = 0
        for sent in sentences:
            words = sent.split()
            if len(words) < 2:
                continue
            # Skip first word (sentence start), check for capitalized words
            for w in words[1:]:
                clean = re.sub(r'[^a-zA-Z]', '', w)
                if clean and clean[0].isupper() and len(clean) > 1:
                    # Filter out common non-entity capitalized words
                    common_caps = {'I', 'The', 'This', 'That', 'These', 'Those', 'It', 
                                   'He', 'She', 'They', 'We', 'You', 'My', 'Your',
                                   'His', 'Her', 'Its', 'Our', 'Their', 'If', 'But',
                                   'And', 'Or', 'So', 'Yet', 'For', 'Not', 'Also',
                                   'However', 'Moreover', 'Furthermore', 'Additionally',
                                   'Therefore', 'Thus', 'Hence', 'Meanwhile', 'Although',
                                   'Because', 'Since', 'While', 'When', 'Where', 'What',
                                   'Which', 'Who', 'How', 'Why', 'Some', 'Many', 'Most',
                                   'All', 'Any', 'Each', 'Every', 'No', 'Just', 'Even',
                                   'Still', 'Here', 'There', 'Now', 'Then', 'Well',
                                   'Being', 'Having', 'Getting', 'Going', 'Coming',
                                   'Making', 'Taking', 'Doing'}
                    if clean not in common_caps:
                        named_entity_count += 1
        
        # === FEATURE 2: Numeric precision ===
        # Count specific numbers, dates, percentages, measurements
        number_patterns = [
            r'\b\d{1,3}(?:,\d{3})+\b',  # Large numbers with commas: 1,000,000
            r'\b\d+\.\d+\b',              # Decimal numbers: 3.14
            r'\b\d+%\b',                   # Percentages: 80%
            r'\b\d{4}\b',                  # Years: 1250, 2023
            r'\b\d+(?:st|nd|rd|th)\b',     # Ordinals: 1st, 2nd
            r'\$\d+',                       # Dollar amounts
            r'\b\d+\s*(?:mg|kg|lb|oz|ml|cm|mm|km|mi|ft|in|hr|min|sec|mph|kph|GB|MB|TB|GHz|MHz)\b',  # Measurements
        ]
        
        numeric_count = 0
        for pat in number_patterns:
            numeric_count += len(re.findall(pat, response, re.IGNORECASE))
        
        # Also count standalone meaningful numbers (not just "1" or "2" alone)
        all_numbers = re.findall(r'\b\d+\b', response)
        meaningful_numbers = [n for n in all_numbers if len(n) >= 2 or int(n) > 2]
        numeric_count += len(meaningful_numbers) * 0.3
        
        # === FEATURE 3: Quoted / Referenced works ===
        # Books, papers, titles in quotes or italics
        quoted_refs = len(re.findall(r'["\u201c\u201d](.+?)["\u201c\u201d]', response))
        italic_refs = len(re.findall(r'\*(.+?)\*', response))
        backtick_refs = len(re.findall(r'`(.+?)`', response))
        reference_count = quoted_refs + italic_refs + backtick_refs
        
        # === FEATURE 4: Technical / domain-specific vocabulary density ===
        # Longer, less common words tend to be more technical/specific
        words = re.findall(r'[a-zA-Z]+', response.lower())
        total_words = len(words) if words else 1
        
        # Common filler/vague words to penalize
        vague_phrases = [
            'many people', 'some people', 'it depends', 'various factors',
            'there are many', 'there are various', 'in general', 'generally speaking',
            'it is important', 'it is worth', 'keep in mind', 'at the end of the day',
            'a lot of', 'lots of', 'kind of', 'sort of', 'more or less',
            'pretty much', 'basically', 'essentially', 'obviously', 'clearly',
            'needless to say', 'goes without saying', 'as we all know',
            'in my opinion', 'i think that', 'i believe that', 'i would say',
            'you should consider', 'you might want to', 'it really depends',
            'there are pros and cons', 'on the other hand', 'having said that',
            'to be honest', 'to be fair', 'at the same time', 'in some cases',
            'in many cases', 'for the most part', 'by and large',
        ]
        
        response_lower = response.lower()
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += response_lower.count(phrase)
        
        # === FEATURE 5: Sentence information density ===
        # Score each sentence by its "payload" - ratio of content words to total
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'she', 'they', 'them', 'his', 'her',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        }
        
        sentence_scores = []
        for sent in sentences:
            sent_words = re.findall(r'[a-zA-Z]+', sent.lower())
            if len(sent_words) < 3:
                continue
            content_words = [w for w in sent_words if w not in stop_words]
            # Information density = content words / total words
            density = len(content_words) / len(sent_words) if sent_words else 0
            
            # Bonus for longer content words (more specific/technical)
            avg_content_len = (sum(len(w) for w in content_words) / len(content_words)) if content_words else 0
            specificity_bonus = max(0, (avg_content_len - 5) * 0.1)
            
            sentence_scores.append(density + specificity_bonus)
        
        avg_sentence_info = (sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.3
        
        # === FEATURE 6: Structural evidence markers ===
        # Detect patterns that indicate concrete examples/evidence
        evidence_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies show\b',
            r'\bevidence suggests\b', r'\bdata shows\b', r'\bdata indicates\b',
            r'\be\.g\.\b', r'\bi\.e\.\b',
        ]
        evidence_marker_count = 0
        for pat in evidence_markers:
            evidence_marker_count += len(re.findall(pat, response_lower))
        
        # === FEATURE 7: Code blocks and structured content ===
        code_blocks = len(re.findall(r'```', response)) // 2
        inline_code = len(re.findall(r'`[^`]+`', response))
        
        # === FEATURE 8: URL/link references ===
        url_count = len(re.findall(r'https?://\S+', response))
        path_refs = len(re.findall(r'/[a-zA-Z_]+/[a-zA-Z_]+', response))
        
        # === FEATURE 9: Parenthetical clarifications ===
        # Parentheses often contain specific clarifying details
        paren_count = len(re.findall(r'\([^)]{3,}\)', response))
        
        # === FEATURE 10: Unique word richness (vocabulary diversity) ===
        if total_words > 10:
            unique_ratio = len(set(words)) / total_words
        else:
            unique_ratio = 0.5
        
        # === FEATURE 11: Response length (moderate bonus, diminishing returns) ===
        length_score = math.log(max(total_words, 1) + 1) / math.log(500)
        length_score = min(length_score, 1.5)
        
        # === FEATURE 12: Causal/explanatory chains ===
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bleading to\b', r'\bresulting in\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bthe reason\b',
        ]
        causal_count = 0
        for pat in causal_markers:
            causal_count += len(re.findall(pat, response_lower))
        
        # === SCORING AGGREGATION ===
        # Normalize features and combine
        
        # Named entities: cap at ~15 for full score
        ne_score = min(named_entity_count / 8.0, 2.0)
        
        # Numeric precision: cap at ~8 for full score
        num_score = min(numeric_count / 4.0, 2.0)
        
        # References: each one is valuable
        ref_score = min(reference_count / 2.0, 1.5)
        
        # Vagueness penalty
        vague_penalty = min(vague_count * 0.4, 3.0)
        
        # Sentence information density (0 to ~0.7 typically)
        info_density_score = avg_sentence_info * 2.0
        
        # Evidence markers
        evidence_score = min(evidence_marker_count * 0.5, 1.5)
        
        # Code/structured content
        code_score = min((code_blocks * 0.8 + inline_code * 0.2), 1.5)
        
        # URL references
        url_score = min((url_count + path_refs * 0.3) * 0.5, 1.0)
        
        # Parenthetical details
        paren_score = min(paren_count * 0.3, 1.0)
        
        # Vocabulary richness
        vocab_score = unique_ratio * 1.5
        
        # Causal chains
        causal_score = min(causal_count * 0.25, 1.0)
        
        # Length factor (multiplier, not additive)
        length_factor = min(max(length_score, 0.3), 1.3)
        
        # Combine all scores
        raw_score = (
            ne_score * 1.5 +          # Named entities (high weight)
            num_score * 1.3 +          # Numeric precision (high weight)
            ref_score * 1.2 +          # Referenced works
            info_density_score * 1.0 + # Sentence-level info density
            evidence_score * 1.0 +     # Evidence markers
            code_score * 0.8 +         # Code/structure
            url_score * 0.5 +          # URLs
            paren_score * 0.7 +        # Parenthetical details
            vocab_score * 0.8 +        # Vocabulary richness
            causal_score * 0.6 -       # Causal reasoning
            vague_penalty * 1.2        # Vagueness penalty
        )
        
        # Apply length factor
        raw_score = raw_score * length_factor
        
        # Bonus for responses that have MULTIPLE types of evidence
        evidence_types_present = sum([
            named_entity_count > 2,
            numeric_count > 1,
            reference_count > 0,
            evidence_marker_count > 0,
            code_blocks > 0,
            paren_count > 0,
            causal_count > 1,
        ])
        diversity_bonus = evidence_types_present * 0.3
        
        raw_score += diversity_bonus
        
        # Scale to 0-10 range
        # Typical raw scores range from about -1 to 12
        final_score = max(0.0, min(10.0, raw_score * 0.8 + 1.0))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a neutral score based on response length
        try:
            return min(max(len(str(response)) / 200.0, 0.5), 5.0)
        except:
            return 2.5