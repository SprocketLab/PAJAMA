def judging_function(query, response):
    """
    Evaluate evidence density and specificity using a pattern-matching approach
    that identifies and counts specific evidence markers: proper nouns, numbers,
    technical terms, quoted references, parenthetical details, and penalizes
    vague hedging language.
    
    This variant focuses on a "specificity ratio" approach - measuring the proportion
    of sentences that contain at least one concrete evidence marker, combined with
    a diversity score for different types of evidence used.
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
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 1.0
        
        # === CATEGORY 1: Numeric specificity ===
        # Specific numbers (years, percentages, quantities, money)
        year_pattern = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
        percentage_pattern = re.findall(r'\b\d+\.?\d*\s*%', response)
        money_pattern = re.findall(r'[\$€£]\s*\d+[\d,]*\.?\d*|\b\d+[\d,]*\s*(?:dollars|euros|pounds|USD|EUR|GBP)\b', response, re.IGNORECASE)
        quantity_pattern = re.findall(r'\b\d+\.?\d*\s*(?:lb|kg|oz|mg|ml|mm|cm|km|miles?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|feet|inches|gallons?|liters?|watts?|volts?|amps?|degrees?|mph|kph)\b', response, re.IGNORECASE)
        plain_numbers = re.findall(r'\b\d{2,}\b', response)
        
        numeric_score = (
            len(year_pattern) * 2.0 +
            len(percentage_pattern) * 2.5 +
            len(money_pattern) * 2.5 +
            len(quantity_pattern) * 2.0 +
            len(plain_numbers) * 0.5
        )
        
        # === CATEGORY 2: Named entities / proper nouns ===
        # Words starting with capital letters that aren't sentence starters
        words = response.split()
        proper_nouns = []
        for i, w in enumerate(words):
            if i == 0:
                continue
            cleaned = re.sub(r'[^a-zA-Z]', '', w)
            if not cleaned:
                continue
            # Check if previous char was a sentence ender
            prev_text = ' '.join(words[:i])
            if prev_text and prev_text[-1] in '.!?:':
                continue
            if cleaned[0].isupper() and len(cleaned) > 1 and not cleaned.isupper():
                proper_nouns.append(cleaned)
        
        # Also look for multi-word proper nouns
        multi_proper = re.findall(r'(?<![.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response)
        
        # Look for quoted titles or references
        quoted_refs = re.findall(r'["\*]([^"\*]{3,60})["\*]', response)
        italic_refs = re.findall(r'\*([^*]{3,60})\*', response)
        
        # Username/author references
        author_refs = re.findall(r'u/\w+|@\w+|(?:by|from|according to)\s+[A-Z][a-z]+', response, re.IGNORECASE)
        
        entity_score = (
            min(len(proper_nouns), 15) * 1.0 +
            len(multi_proper) * 2.0 +
            len(quoted_refs) * 2.5 +
            len(italic_refs) * 2.5 +
            len(author_refs) * 2.0
        )
        
        # === CATEGORY 3: Technical/domain vocabulary density ===
        # Measure words that are longer and likely domain-specific
        all_words = re.findall(r'[a-zA-Z]+', response.lower())
        if all_words:
            long_technical = [w for w in all_words if len(w) >= 8]
            unique_long = set(long_technical)
            technical_ratio = len(unique_long) / max(len(set(all_words)), 1)
            technical_score = min(technical_ratio * 50, 10)
        else:
            technical_score = 0
        
        # === CATEGORY 4: Structural evidence markers ===
        # Code blocks
        code_blocks = len(re.findall(r'```', response)) // 2
        inline_code = len(re.findall(r'`[^`]+`', response))
        
        # Parenthetical details (often contain specifics)
        parentheticals = re.findall(r'\([^)]{5,}\)', response)
        
        # Enumerated/structured content
        enum_items = re.findall(r'(?:^|\n)\s*(?:\d+[.):]|\*|-|•)\s+\S', response)
        
        # Conditional/causal specifics
        causal_markers = len(re.findall(r'\b(?:because|since|therefore|thus|hence|specifically|in particular|for example|for instance|such as|e\.g\.|i\.e\.)\b', response, re.IGNORECASE))
        
        structural_score = (
            code_blocks * 3.0 +
            inline_code * 1.5 +
            len(parentheticals) * 1.5 +
            min(len(enum_items), 10) * 1.0 +
            causal_markers * 1.0
        )
        
        # === CATEGORY 5: Sentence-level specificity ratio ===
        # What fraction of sentences contain at least one concrete marker?
        specific_sentence_count = 0
        for sent in sentences:
            has_number = bool(re.search(r'\d', sent))
            has_proper = bool(re.search(r'(?<!\. )[A-Z][a-z]{2,}', sent[1:] if len(sent) > 1 else ''))
            has_example = bool(re.search(r'\b(?:for example|such as|e\.g\.|like|specifically|including)\b', sent, re.IGNORECASE))
            has_quote = bool(re.search(r'["\*`]', sent))
            
            if has_number or has_proper or has_example or has_quote:
                specific_sentence_count += 1
        
        specificity_ratio = specific_sentence_count / max(len(sentences), 1)
        specificity_score = specificity_ratio * 15
        
        # === CATEGORY 6: Vagueness penalties ===
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things)\b',
            r'\bgenerally speaking\b', r'\bin general\b',
            r'\bI think\b', r'\bI believe\b', r'\bI guess\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bkind of\b', r'\bsort of\b',
            r'\bbasically\b', r'\bessentially\b',
            r'\bvarious\b', r'\bnumerous\b',
            r'\bthere are many\b', r'\bthere are several\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b',
            r'\bcould be\b', r'\bmight be\b',
            r'\bsomewhat\b', r'\brelatively\b',
        ]
        
        vague_count = 0
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        vague_penalty = min(vague_count * 1.5, 12)
        
        # === CATEGORY 7: Response engagement/depth ===
        # Longer responses with maintained specificity are better
        word_count = len(all_words)
        length_bonus = math.log(max(word_count, 1) + 1, 2) * 0.8  # logarithmic length bonus
        length_bonus = min(length_bonus, 8)
        
        # === CATEGORY 8: Evidence diversity bonus ===
        # Reward using multiple TYPES of evidence
        evidence_types_present = 0
        if len(year_pattern) + len(percentage_pattern) + len(money_pattern) + len(quantity_pattern) > 0:
            evidence_types_present += 1
        if len(proper_nouns) + len(multi_proper) > 0:
            evidence_types_present += 1
        if len(quoted_refs) + len(italic_refs) > 0:
            evidence_types_present += 1
        if code_blocks + inline_code > 0:
            evidence_types_present += 1
        if len(parentheticals) > 0:
            evidence_types_present += 1
        if len(enum_items) > 2:
            evidence_types_present += 1
        if causal_markers > 0:
            evidence_types_present += 1
        if len(author_refs) > 0:
            evidence_types_present += 1
        
        diversity_bonus = evidence_types_present * 1.5
        
        # === CATEGORY 9: Unique information density ===
        # Ratio of unique words to total words (higher = more information-dense, less repetitive)
        if all_words:
            # Filter out very common words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                         'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                         'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
                         'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
                         'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                         'than', 'too', 'very', 'just', 'because', 'if', 'when', 'where',
                         'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                         'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                         'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}
            content_words = [w for w in all_words if w not in stop_words and len(w) > 2]
            if content_words:
                unique_content = set(content_words)
                vocab_richness = len(unique_content) / max(len(content_words), 1)
                info_density_score = vocab_richness * 8
            else:
                info_density_score = 0
        else:
            info_density_score = 0
        
        # === COMBINE SCORES ===
        raw_score = (
            numeric_score * 1.0 +
            entity_score * 1.0 +
            technical_score * 1.0 +
            structural_score * 1.0 +
            specificity_score * 1.0 +
            length_bonus * 1.0 +
            diversity_bonus * 1.0 +
            info_density_score * 0.8 -
            vague_penalty * 1.0
        )
        
        # Normalize to 0-100 range with sigmoid-like scaling
        # Use a tuned sigmoid to spread scores nicely
        normalized = 100 / (1 + math.exp(-0.08 * (raw_score - 20)))
        
        # Ensure within bounds
        final_score = max(0.0, min(100.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 30.0
            return 10.0
        except Exception:
            return 10.0