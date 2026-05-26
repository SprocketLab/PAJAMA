def judging_function(query, response):
    """
    Evaluate evidence density and specificity using an information-theoretic approach.
    
    Strategy: Measure the "information entropy" of the response by analyzing:
    1. Lexical diversity and rare word usage (Shannon entropy of word distribution)
    2. Named entity density via pattern matching (capitalized multi-word phrases, numbers, etc.)
    3. Specificity ratio: concrete/specific words vs vague/hedge words
    4. Information uniqueness: ratio of unique n-grams to total n-grams (penalizes repetition)
    5. Detail clause density: subordinate clauses that add qualifying information
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        words = re.findall(r'[a-zA-Z]+', response.lower())
        raw_words = re.findall(r'\S+', response)
        
        if len(words) < 2:
            return 1.0
        
        # === 1. Shannon entropy of word distribution (higher = more diverse vocabulary) ===
        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize: max entropy for N unique words is log2(N)
        max_entropy = math.log2(total_words) if total_words > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        entropy_score = normalized_entropy * 15  # 0-15 range
        
        # === 2. Repetition penalty using unique n-gram ratios ===
        # Bigram uniqueness
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)] if len(words) > 1 else []
        trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)] if len(words) > 2 else []
        
        bigram_unique_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0
        trigram_unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 0
        
        repetition_score = (bigram_unique_ratio * 0.4 + trigram_unique_ratio * 0.6) * 10  # 0-10
        
        # === 3. Numeric and quantitative evidence density ===
        numbers = re.findall(r'\b\d+[\.,]?\d*%?\b', response)
        numeric_density = min(len(numbers) / max(total_words / 20, 1), 3.0)
        numeric_score = numeric_density * 5  # 0-15
        
        # === 4. Capitalized entity detection (not sentence-start) ===
        sentences = re.split(r'[.!?]\s+', response)
        entity_count = 0
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) > 1:
                # Words after the first that are capitalized (potential named entities)
                for w in sent_words[1:]:
                    clean = re.sub(r'[^a-zA-Z]', '', w)
                    if clean and clean[0].isupper() and len(clean) > 1:
                        entity_count += 1
        
        entity_density = min(entity_count / max(total_words / 15, 1), 3.0)
        entity_score = entity_density * 5  # 0-15
        
        # === 5. Vagueness penalty ===
        vague_patterns = [
            r'\bmany people\b', r'\bit depends\b', r'\bvarious factors\b',
            r'\bin general\b', r'\bsome people\b', r'\bthere are many\b',
            r'\bcan be\b', r'\bmay be\b', r'\bmight be\b', r'\bcould be\b',
            r'\boften\b', r'\bsometimes\b', r'\busually\b', r'\btypically\b',
            r'\bvarious\b', r'\bmany ways\b', r'\bin many ways\b',
            r'\bsome\b', r'\bperhaps\b', r'\bgenerally\b',
            r'\ba lot of\b', r'\bkind of\b', r'\bsort of\b',
            r'\betc\.?\b', r'\band so on\b', r'\band more\b',
            r'\bthings\b', r'\bstuff\b', r'\bwhatever\b',
        ]
        vague_count = 0
        response_lower = response.lower()
        for pat in vague_patterns:
            vague_count += len(re.findall(pat, response_lower))
        
        vague_density = vague_count / max(total_words / 10, 1)
        vague_penalty = min(vague_density * 5, 15)  # 0-15 penalty
        
        # === 6. Specificity indicators (concrete language) ===
        specific_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bincluding\b', r'\be\.g\.\b', r'\bi\.e\.\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies\b',
            r'\bpercent\b', r'\b%\b', r'\bdollars?\b', r'\b\$\b',
            r'\bin \d{4}\b',  # years
            r'\bfounded\b', r'\bcreated\b', r'\bdeveloped\b',
            r'\bmeasured\b', r'\bcalculated\b',
        ]
        specific_count = 0
        for pat in specific_patterns:
            specific_count += len(re.findall(pat, response_lower))
        
        specificity_score = min(specific_count * 2, 10)  # 0-10
        
        # === 7. Detail clause density (commas, semicolons, parentheticals = more qualifying info) ===
        commas = response.count(',')
        semicolons = response.count(';')
        parens = response.count('(')
        colons = response.count(':')
        dashes = response.count('—') + response.count('–') + response.count(' - ')
        
        clause_markers = commas + semicolons * 2 + parens * 2 + colons + dashes
        clause_density = clause_markers / max(total_words / 10, 1)
        clause_score = min(clause_density * 3, 10)  # 0-10
        
        # === 8. Unique word ratio (type-token ratio) ===
        unique_words = len(set(words))
        ttr = unique_words / total_words if total_words > 0 else 0
        # Penalize very low TTR (heavy repetition)
        ttr_score = ttr * 10  # 0-10
        
        # === 9. Response length bonus (longer responses tend to have more evidence, but diminishing returns) ===
        length_score = min(math.log2(max(total_words, 1)) * 1.5, 12)  # 0-12 roughly
        
        # === 10. Adjective/adverb specificity: penalize superlatives without evidence ===
        superlative_vague = len(re.findall(
            r'\b(very|really|extremely|incredibly|absolutely|totally|completely|quite|rather)\b',
            response_lower
        ))
        intensifier_penalty = min(superlative_vague * 0.5, 5)
        
        # === Combine scores ===
        total_score = (
            entropy_score          # 0-15: vocabulary diversity
            + repetition_score     # 0-10: non-repetitive content
            + numeric_score        # 0-15: quantitative evidence
            + entity_score         # 0-15: named entities
            + specificity_score    # 0-10: specificity markers
            + clause_score         # 0-10: detail clause density
            + ttr_score            # 0-10: type-token ratio
            + length_score         # 0-12: response length
            - vague_penalty        # 0-15: vagueness penalty
            - intensifier_penalty  # 0-5: intensifier penalty
        )
        
        # Clamp to 0-100 range
        total_score = max(0.0, min(100.0, total_score))
        
        return round(total_score, 2)
        
    except Exception:
        try:
            # Fallback: simple length-based score
            return min(len(response.split()) * 0.3, 50.0)
        except Exception:
            return 5.0