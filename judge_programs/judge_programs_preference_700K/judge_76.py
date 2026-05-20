def judging_function(query, response):
    """
    Evaluates evidence density and specificity using an information-theoretic approach.
    
    This variant focuses on:
    1. Named entity density (capitalized multi-word phrases, proper nouns)
    2. Numeric/quantitative information density
    3. Specificity markers vs vagueness markers (ratio-based)
    4. Information entropy of vocabulary (diverse, specific vocab = higher entropy)
    5. Clause-level evidence scoring (sentences with concrete referents)
    6. Technical/domain term density
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        words = response.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # Normalize response for some checks
        resp_lower = response.lower()
        
        # ---- 1. Named Entity Density ----
        # Look for capitalized words that aren't sentence starters
        # Split into sentences, then check for capitalized words mid-sentence
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        named_entity_count = 0
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) <= 1:
                continue
            # Skip first word (sentence start), check rest for capitalization
            for w in sent_words[1:]:
                clean_w = re.sub(r'[^a-zA-Z]', '', w)
                if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                    named_entity_count += 1
        
        # Also detect patterns like "Dr. X", "Mr. Y", titles, book/work names in italics/quotes
        quoted_refs = re.findall(r'["\*\'](.*?)["\*\']', response)
        named_entity_count += len(quoted_refs)
        
        # Detect "u/" reddit usernames, URLs, specific references
        specific_refs = re.findall(r'u/\w+|https?://\S+|www\.\S+', response)
        named_entity_count += len(specific_refs) * 2
        
        ne_density = named_entity_count / max(word_count, 1)
        ne_score = min(ne_density * 40, 10)  # cap at 10
        
        # ---- 2. Numeric/Quantitative Information ----
        # Find numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d+[\d,]*\.?\d*\b', response)
        percentages = re.findall(r'\d+\s*%', response)
        dates = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', response)
        measurements = re.findall(r'\b\d+\s*(?:lb|kg|oz|mg|ml|km|miles?|hours?|minutes?|seconds?|feet|ft|inches?|in|cm|mm|degrees?|°|F|C|psi|rpm|mph|kph|watts?|volts?|amps?|GB|MB|TB)\b', resp_lower)
        money = re.findall(r'\$[\d,]+\.?\d*|\b\d+\s*(?:dollars?|euros?|pounds?|cents?)\b', resp_lower)
        
        numeric_items = len(numbers) + len(percentages) * 2 + len(dates) * 1.5 + len(measurements) * 2 + len(money) * 2
        numeric_density = numeric_items / max(word_count, 1)
        numeric_score = min(numeric_density * 50, 12)
        
        # ---- 3. Specificity vs Vagueness Ratio ----
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b', 
            r'\bthere are (?:many|various|several|different) (?:factors|reasons|ways|things)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bfor the most part\b',
            r'\btypically\b', r'\busually\b', r'\boften\b', r'\bsometimes\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b', r'\bmight\b',
            r'\bcould be\b', r'\btend to\b', r'\bkind of\b', r'\bsort of\b',
            r'\ba lot of\b', r'\ba number of\b', r'\bquite a few\b',
            r'\bvarious\b', r'\bnumerous\b', r'\bcountless\b',
            r'\bI think\b', r'\bI believe\b', r'\bI guess\b',
            r'\bmore or less\b', r'\bto some extent\b', r'\bin a way\b',
            r'\byou know\b', r'\bbasically\b', r'\bessentially\b',
        ]
        
        specificity_phrases = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies show\b',
            r'\bevidence suggests\b', r'\bdata shows\b',
            r'\bthe reason is\b', r'\bthis is because\b',
            r'\bin fact\b', r'\bin practice\b', r'\bin reality\b',
            r'\bconcretely\b', r'\bprecisely\b', r'\bexactly\b',
        ]
        
        vague_count = sum(len(re.findall(p, resp_lower)) for p in vague_phrases)
        specific_count = sum(len(re.findall(p, resp_lower)) for p in specificity_phrases)
        
        # Ratio-based scoring
        if vague_count + specific_count > 0:
            specificity_ratio = (specific_count - vague_count * 0.5) / (vague_count + specific_count + 1)
        else:
            specificity_ratio = 0
        
        vagueness_penalty = vague_count * 0.3
        specificity_bonus = specific_count * 0.8
        spec_score = max(min(specificity_bonus - vagueness_penalty + 2, 8), -3)
        
        # ---- 4. Vocabulary Entropy (information-theoretic) ----
        # Higher entropy = more diverse, specific vocabulary
        clean_words = [re.sub(r'[^a-z]', '', w.lower()) for w in words]
        clean_words = [w for w in clean_words if len(w) > 2]
        
        if clean_words:
            word_freq = Counter(clean_words)
            total = len(clean_words)
            entropy = -sum((c/total) * math.log2(c/total) for c in word_freq.values())
            # Normalize: max entropy for N unique words from total T
            max_possible_entropy = math.log2(max(total, 2))
            normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        entropy_score = normalized_entropy * 6  # 0-6 range
        
        # ---- 5. Sentence-level Evidence Scoring ----
        # Score each sentence for containing concrete evidence
        evidence_sentence_count = 0
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if not sent_lower:
                continue
            
            has_evidence = False
            # Contains a number
            if re.search(r'\d', sent):
                has_evidence = True
            # Contains a quoted/referenced work
            if re.search(r'["\*\']\w', sent):
                has_evidence = True
            # Contains a proper noun (capitalized word mid-sentence)
            s_words = sent.split()
            if len(s_words) > 2:
                for w in s_words[1:]:
                    cw = re.sub(r'[^a-zA-Z]', '', w)
                    if cw and cw[0].isupper() and len(cw) > 1:
                        has_evidence = True
                        break
            # Contains specific technical terms or domain language
            if re.search(r'\b(?:algorithm|protocol|theorem|equation|mechanism|process|procedure|method|technique|framework|model|theory|principle|law|regulation|statute|provision|clause)\b', sent_lower):
                has_evidence = True
            
            if has_evidence:
                evidence_sentence_count += 1
        
        total_sentences = max(len(sentences), 1)
        evidence_ratio = evidence_sentence_count / total_sentences
        evidence_sentence_score = evidence_ratio * 10  # 0-10
        
        # ---- 6. Technical/Domain Term Density ----
        # Words that are longer, less common, more domain-specific
        long_specific_words = [w for w in clean_words if len(w) >= 8]
        tech_density = len(long_specific_words) / max(len(clean_words), 1)
        tech_score = min(tech_density * 20, 8)
        
        # ---- 7. Structural Depth ----
        # Parenthetical clarifications, em-dashes, colons for elaboration
        parentheticals = len(re.findall(r'\(.*?\)', response))
        colons = response.count(':')
        em_dashes = response.count('--') + response.count('—')
        semicolons = response.count(';')
        
        structural_items = parentheticals + colons * 0.5 + em_dashes * 0.7 + semicolons * 0.5
        structural_score = min(structural_items * 0.8, 5)
        
        # ---- 8. Code/SQL/Formal Content Detection ----
        code_blocks = len(re.findall(r'```', response))
        has_code = code_blocks > 0 or re.search(r'\b(?:SELECT|FROM|WHERE|JOIN|INSERT|CREATE|def |class |import |function)\b', response) is not None
        code_bonus = 3 if has_code else 0
        
        # ---- 9. Response Length Factor ----
        # Longer responses have more opportunity for evidence, but normalize
        # Use log to prevent pure length from dominating
        length_factor = min(math.log(max(word_count, 1) + 1) / math.log(500), 1.5)
        
        # ---- 10. Causal/Explanatory Depth ----
        causal_markers = re.findall(r'\b(?:because|therefore|thus|hence|consequently|as a result|due to|caused by|leads to|results in|implies|means that|so that|in order to)\b', resp_lower)
        causal_score = min(len(causal_markers) * 0.6, 4)
        
        # ---- 11. Contrast/Comparison (shows nuanced understanding) ----
        contrast_markers = re.findall(r'\b(?:however|whereas|while|although|on the other hand|in contrast|unlike|compared to|rather than|instead of|but|yet|nevertheless)\b', resp_lower)
        contrast_score = min(len(contrast_markers) * 0.4, 3)
        
        # ---- COMBINE SCORES ----
        raw_score = (
            ne_score * 1.2 +           # Named entities: up to 12
            numeric_score * 1.3 +       # Numeric info: up to 15.6
            spec_score * 1.0 +          # Specificity: -3 to 8
            entropy_score * 0.8 +       # Vocabulary diversity: up to 4.8
            evidence_sentence_score * 1.5 +  # Evidence sentences: up to 15
            tech_score * 0.7 +          # Technical terms: up to 5.6
            structural_score * 0.6 +    # Structural depth: up to 3
            code_bonus * 1.0 +          # Code: 0 or 3
            causal_score * 0.8 +        # Causal depth: up to 3.2
            contrast_score * 0.6        # Contrast: up to 1.8
        )
        
        # Apply length factor as a mild multiplier
        raw_score *= (0.5 + 0.5 * length_factor)
        
        # Normalize to 0-100 range
        # Theoretical max is around 72, practical high around 50
        final_score = max(0.0, min(100.0, raw_score * 1.8))
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a basic length-based score
        try:
            return min(len(response.split()) * 0.1, 20.0)
        except Exception:
            return 0.0