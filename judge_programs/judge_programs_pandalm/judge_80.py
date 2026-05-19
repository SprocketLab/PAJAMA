def judging_function(query, response):
    """
    Evaluate evidence density and specificity using an information-theoretic approach
    based on token surprisal/rarity, named entity pattern detection, and 
    structural information packaging density.
    
    This variant focuses on:
    1. Rare/specific word ratio using zipf-like frequency estimation
    2. Named entity and specific reference pattern matching
    3. Information packaging density (info per sentence)
    4. Repetition penalty (inverse of compression ratio)
    5. Specificity markers vs vagueness markers ratio
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        import re
        import math
        from collections import Counter
        
        # ---- 1. Token rarity score ----
        # Common/stop words get low score; unusual/specific words get high score
        # Instead of using concreteness ratings, we use a frequency-based approach
        # where we estimate word commonality from a hand-curated set of the most
        # common English words (top ~200), and reward words NOT in this set.
        
        ultra_common = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
            'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
            'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
            'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
            'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'being',
            'has', 'had', 'does', 'did', 'doing', 'am', 'more', 'very', 'much',
            'many', 'such', 'those', 'may', 'might', 'should', 'must', 'shall',
            'need', 'here', 'still', 'own', 'too', 'each', 'where', 'while',
            'both', 'between', 'under', 'same', 'through', 'during', 'before',
            'after', 'above', 'below', 'again', 'further', 'once', 'why', 'how',
            'every', 'never', 'always', 'often', 'sometimes', 'usually',
            'really', 'quite', 'rather', 'already', 'yet', 'perhaps', 'maybe',
            'however', 'although', 'though', 'whether', 'since', 'until',
            'unless', 'also', 'just', 'only', 'still', 'already', 'another',
            'something', 'anything', 'nothing', 'everything', 'someone',
            'anyone', 'everyone', 'thing', 'things', 'part', 'place',
            'case', 'point', 'fact', 'kind', 'lot', 'number', 'set',
            'different', 'important', 'large', 'small', 'long', 'great',
            'high', 'old', 'right', 'big', 'little', 'last', 'few', 'able',
            'possible', 'likely', 'certain', 'sure', 'real', 'main',
            'various', 'several', 'general', 'specific', 'particular',
        }
        
        words = re.findall(r'[a-zA-Z]+', response.lower())
        if len(words) < 3:
            return 1.0
        
        # Proportion of non-ultra-common words (content-rich words)
        content_words = [w for w in words if w not in ultra_common and len(w) > 2]
        content_ratio = len(content_words) / max(len(words), 1)
        
        # Reward longer content words (they tend to be more specific)
        avg_content_word_len = (sum(len(w) for w in content_words) / max(len(content_words), 1)) if content_words else 0
        word_len_score = min(avg_content_word_len / 10.0, 1.0)
        
        # ---- 2. Specific reference patterns (regex-based entity detection) ----
        # Count patterns that indicate specific/concrete information
        
        # Numbers and quantities
        number_patterns = re.findall(r'\b\d+[\.,]?\d*\s*(%|percent|million|billion|thousand|kg|lb|km|miles?|meters?|feet|inches|hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|euros?|pounds?|cents?|USD|EUR|GBP)?\b', response)
        num_count = len(number_patterns)
        
        # Standalone numbers
        standalone_nums = re.findall(r'\b\d{1,}\b', response)
        num_count += len(standalone_nums) * 0.5
        
        # Capitalized multi-word names (likely proper nouns / named entities)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response)
        proper_count = len(proper_nouns)
        
        # Single capitalized words not at sentence start
        sentences_raw = re.split(r'[.!?]+', response)
        mid_caps = 0
        for sent in sentences_raw:
            sent = sent.strip()
            if len(sent) < 3:
                continue
            # Find capitalized words that aren't the first word
            sent_words = sent.split()
            for w in sent_words[1:]:
                if w and w[0].isupper() and w.isalpha() and len(w) > 1:
                    mid_caps += 1
        
        # Dates and years
        dates = re.findall(r'\b(19|20)\d{2}\b', response)
        date_count = len(dates)
        
        # Quoted terms or technical terms
        quoted = re.findall(r'"[^"]{2,}"', response) + re.findall(r"'[^']{2,}'", response)
        quoted_count = len(quoted)
        
        # URLs or references
        urls = re.findall(r'https?://\S+|www\.\S+', response)
        url_count = len(urls)
        
        # Parenthetical info (often contains specifics like abbreviations, dates)
        parens = re.findall(r'\([^)]{2,}\)', response)
        paren_count = len(parens)
        
        # Technical/domain terms: words with mixed case, hyphens, or unusual patterns
        technical = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', response)  # camelCase
        tech_count = len(technical)
        
        # Compound/hyphenated terms
        hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', response)
        hyphen_count = len(hyphenated)
        
        # Total specificity markers
        specificity_raw = (num_count * 2.0 + proper_count * 2.5 + mid_caps * 1.5 + 
                          date_count * 3.0 + quoted_count * 1.5 + url_count * 2.0 + 
                          paren_count * 1.5 + tech_count * 1.5 + hyphen_count * 1.0)
        
        # Normalize by response length (per 100 words)
        specificity_density = (specificity_raw / max(len(words), 1)) * 100
        specificity_score = min(specificity_density / 15.0, 1.0)  # cap at 1.0
        
        # ---- 3. Vagueness penalty ----
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different|numerous) (?:factors|reasons|ways|things|aspects)\b',
            r'\bin many ways\b', r'\bin various ways\b', r'\bin some ways\b',
            r'\bgenerally speaking\b', r'\bfor the most part\b',
            r'\bit is (?:important|interesting|worth noting)\b',
            r'\bas we (?:all )?know\b', r'\bneedless to say\b',
            r'\bit goes without saying\b', r'\bat the end of the day\b',
            r'\ball in all\b', r'\bin conclusion\b', r'\boverall\b',
            r'\band so on\b', r'\band so forth\b', r'\betc\.?\b',
            r'\bsort of\b', r'\bkind of\b', r'\bmore or less\b',
            r'\bto some (?:extent|degree)\b', r'\bin general\b',
            r'\ba (?:wide )?(?:range|variety) of\b',
            r'\bnumerous\b', r'\bcountless\b',
        ]
        
        vague_count = 0
        response_lower = response.lower()
        for pattern in vague_phrases:
            vague_count += len(re.findall(pattern, response_lower))
        
        vague_density = (vague_count / max(len(words), 1)) * 100
        vague_penalty = min(vague_density * 3.0, 3.0)  # max penalty of 3.0
        
        # ---- 4. Repetition penalty (compression-like metric) ----
        # High repetition = low information density
        word_counts = Counter(words)
        if len(words) > 0:
            # Calculate type-token ratio variant: unique bigrams / total bigrams
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            if bigrams:
                unique_bigrams = len(set(bigrams))
                total_bigrams = len(bigrams)
                bigram_diversity = unique_bigrams / total_bigrams
            else:
                bigram_diversity = 0.5
            
            # Also check for repeated phrases (3+ word sequences)
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            if trigrams:
                trigram_counts = Counter(trigrams)
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
                trigram_repetition = repeated_trigrams / max(len(set(trigrams)), 1)
            else:
                trigram_repetition = 0
            
            repetition_penalty = max(0, (1.0 - bigram_diversity) * 3.0 + trigram_repetition * 2.0)
        else:
            repetition_penalty = 0
            bigram_diversity = 0.5
        
        # ---- 5. Information packaging density ----
        # How much specific content per sentence?
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Average sentence length (too short = not enough info, too long might be filler)
        avg_sent_len = len(words) / num_sentences
        # Sweet spot around 15-25 words per sentence
        if avg_sent_len < 5:
            sent_len_score = 0.3
        elif avg_sent_len < 10:
            sent_len_score = 0.6
        elif avg_sent_len <= 30:
            sent_len_score = 1.0
        else:
            sent_len_score = 0.8
        
        # ---- 6. Action/process verbs (indicate concrete descriptions) ----
        action_patterns = [
            r'\b(?:sends?|receives?|processes?|creates?|generates?|builds?|computes?|calculates?|measures?|tracks?|monitors?|displays?|stores?|retrieves?|converts?|transforms?|analyzes?|detects?|identifies?|classifies?|extracts?|filters?|sorts?|maps?|connects?|transmits?|encrypts?|decrypts?|validates?|verifies?|authenticates?|compiles?|executes?|deploys?|configures?|installs?|downloads?|uploads?|imports?|exports?)\b',
        ]
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, response_lower))
        
        action_density = (action_count / max(len(words), 1)) * 100
        action_score = min(action_density / 5.0, 0.5)  # smaller contribution
        
        # ---- 7. Enumeration and structure (lists, steps, categories) ----
        # Numbered items
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[.)]\s', response)
        # Lettered items
        lettered_items = re.findall(r'(?:^|\n)\s*[a-z][.)]\s', response)
        # Bullet points
        bullet_items = re.findall(r'(?:^|\n)\s*[-•*]\s', response)
        # "First, ... Second, ... Third, ..."
        ordinal_items = re.findall(r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b', response_lower)
        
        enum_count = len(numbered_items) + len(lettered_items) + len(bullet_items) + len(ordinal_items) * 0.5
        enum_score = min(enum_count / 5.0, 0.5)
        
        # ---- 8. Response length adequacy ----
        # Very short responses are rarely evidence-dense in absolute terms
        length_factor = min(len(words) / 30.0, 1.5)  # ramp up to 30 words, slight bonus beyond
        length_factor = max(length_factor, 0.2)  # floor
        
        # ---- 9. Unique information tokens ----
        # Count unique content words as a proxy for breadth of information
        unique_content = len(set(content_words))
        info_breadth = min(unique_content / 20.0, 1.0)
        
        # ---- COMPOSITE SCORE ----
        # Weight the components
        raw_score = (
            content_ratio * 2.5 +          # Non-common word density (0-2.5)
            word_len_score * 1.0 +          # Average content word length (0-1.0)
            specificity_score * 3.0 +       # Named entities, numbers, etc (0-3.0)
            action_score * 1.0 +            # Action/process verbs (0-0.5 -> *1.0)
            enum_score * 0.8 +              # Structured enumeration (0-0.5 -> *0.8)
            sent_len_score * 0.7 +          # Sentence length quality (0-1.0 -> *0.7)
            info_breadth * 1.5 +            # Breadth of unique content (0-1.5)
            bigram_diversity * 0.5 -         # Diversity bonus (0-0.5)
            vague_penalty -                 # Vagueness penalty (0-3.0)
            repetition_penalty              # Repetition penalty (0+)
        )
        
        # Apply length factor as a multiplier
        raw_score = raw_score * length_factor
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            # Minimal fallback: just use response length and basic content
            words = response.split() if response else []
            return min(len(words) / 10.0, 5.0)
        except Exception:
            return 0.0