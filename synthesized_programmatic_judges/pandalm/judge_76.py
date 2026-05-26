def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    that identifies and counts specific types of concrete evidence markers.
    
    Algorithm: Named Entity / Concrete Detail Pattern Recognition
    - Detects capitalized named entities (proper nouns)
    - Counts numeric/quantitative expressions
    - Identifies technical/domain-specific vocabulary
    - Measures specificity through detail-rich syntactic patterns
    - Penalizes vague hedging and filler patterns
    - Rewards actionable, precise language
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.0
        
        words = response.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        score = 0.0
        
        # === 1. Named Entity Detection (capitalized multi-word sequences) ===
        # Find sequences of capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]+', response)
        named_entity_count = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = sent.split()
            if len(sent_words) < 2:
                continue
            # Skip first word (sentence starter), look for capitalized words
            for w in sent_words[1:]:
                clean_w = re.sub(r'[^a-zA-Z]', '', w)
                if clean_w and len(clean_w) > 1 and clean_w[0].isupper() and not clean_w.isupper():
                    named_entity_count += 1
        
        # Capitalized multi-word phrases (proper nouns like "United States", "New York")
        proper_noun_phrases = re.findall(r'(?<!\. )(?<!\.\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response)
        named_entity_count += len(proper_noun_phrases) * 2
        
        entity_density = named_entity_count / max(word_count, 1)
        score += min(entity_density * 80, 15)
        
        # === 2. Numeric and Quantitative Expressions ===
        # Actual numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', response)
        num_count = len(numbers)
        
        # Quantitative words
        quant_words = re.findall(
            r'\b(?:percent|percentage|ratio|rate|average|median|total|approximately|'
            r'roughly|exactly|precisely|million|billion|thousand|hundred|dozen|'
            r'twice|triple|half|quarter|third|double|increment|decrease|increase)\b',
            response, re.IGNORECASE
        )
        num_count += len(quant_words)
        
        # Date/time patterns
        dates = re.findall(r'\b(?:\d{4}|\d{1,2}/\d{1,2}|\d{1,2}:\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b', response)
        num_count += len(dates)
        
        numeric_density = num_count / max(word_count, 1)
        score += min(numeric_density * 120, 15)
        
        # === 3. Specificity Markers (precise language patterns) ===
        specificity_patterns = [
            r'\bsuch as\b',
            r'\bfor example\b',
            r'\bfor instance\b',
            r'\bincluding\b',
            r'\bspecifically\b',
            r'\bin particular\b',
            r'\bnamely\b',
            r'\bi\.e\.',
            r'\be\.g\.',
            r'\bknown as\b',
            r'\breferred to as\b',
            r'\bcalled\b',
        ]
        specificity_count = 0
        for pat in specificity_patterns:
            specificity_count += len(re.findall(pat, response, re.IGNORECASE))
        
        score += min(specificity_count * 3.0, 12)
        
        # === 4. Technical/Domain Vocabulary Density ===
        # Words that are longer and likely domain-specific (not common English)
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'what', 'which', 'who', 'whom', 'also', 'about', 'up', 'many', 'much',
            'well', 'back', 'even', 'still', 'way', 'take', 'come', 'make', 'like',
            'long', 'great', 'little', 'right', 'good', 'bad', 'new', 'old', 'big',
            'small', 'different', 'important', 'things', 'thing', 'people', 'person',
            'time', 'year', 'day', 'world', 'life', 'hand', 'part', 'place', 'case',
            'work', 'fact', 'group', 'number', 'point', 'said', 'says', 'say',
            'get', 'got', 'go', 'goes', 'went', 'see', 'saw', 'know', 'knew',
            'think', 'thought', 'want', 'give', 'use', 'find', 'tell', 'ask',
            'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin', 'show',
            'hear', 'play', 'run', 'move', 'live', 'believe', 'bring', 'happen',
            'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
            'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch',
            'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend',
            'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider',
            'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build',
            'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'form', 'various',
            'several', 'often', 'however', 'therefore', 'thus', 'hence', 'moreover',
            'furthermore', 'additionally', 'meanwhile', 'nevertheless', 'nonetheless',
            'indeed', 'certainly', 'perhaps', 'maybe', 'probably', 'possibly',
            'actually', 'really', 'quite', 'rather', 'almost', 'enough', 'always',
            'never', 'sometimes', 'usually', 'generally', 'typically', 'simply',
            'already', 'whether', 'without', 'within', 'along', 'across', 'around',
            'among', 'towards', 'upon', 'since', 'until', 'unless', 'while',
            'another', 'first', 'second', 'last', 'next', 'high', 'low', 'able',
        }
        
        lower_words = [re.sub(r'[^a-z]', '', w.lower()) for w in words]
        lower_words = [w for w in lower_words if w]
        
        # Technical words: uncommon, longer words
        technical_count = 0
        for w in lower_words:
            if len(w) >= 7 and w not in common_words:
                technical_count += 1
        
        tech_density = technical_count / max(len(lower_words), 1)
        score += min(tech_density * 50, 12)
        
        # === 5. Unique Content Words Ratio (information richness) ===
        content_words = [w for w in lower_words if w not in common_words and len(w) > 2]
        if content_words:
            unique_content = len(set(content_words))
            total_content = len(content_words)
            # Penalize repetition
            uniqueness_ratio = unique_content / total_content
            score += uniqueness_ratio * 8
        
        # === 6. Vagueness / Hedging Penalty ===
        vague_patterns = [
            r'\bmany people\b',
            r'\bsome people\b',
            r'\bit depends\b',
            r'\bthere are (?:many|various|several|different|a number of) (?:factors|reasons|ways|things|aspects)\b',
            r'\bin many ways\b',
            r'\bin various ways\b',
            r'\bgenerally speaking\b',
            r'\bit is (?:important|essential|crucial|necessary) to\b',
            r'\bthere are (?:many|several|various) (?:types|kinds|forms)\b',
            r'\bcan be (?:very|quite|rather) (?:difficult|challenging|complex|complicated)\b',
            r'\bdiffer in many ways\b',
            r'\bin today\'s (?:world|society|age)\b',
            r'\bas we (?:all )?know\b',
            r'\bit is worth (?:noting|mentioning)\b',
            r'\bneedless to say\b',
            r'\boverall\b',
            r'\bin general\b',
            r'\bin conclusion\b',
        ]
        
        vague_count = 0
        for pat in vague_patterns:
            vague_count += len(re.findall(pat, response, re.IGNORECASE))
        
        vague_density = vague_count / max(word_count / 50, 1)
        score -= min(vague_density * 4, 10)
        
        # === 7. Hedging Words Penalty ===
        hedge_words = [
            'maybe', 'perhaps', 'possibly', 'somewhat', 'somehow',
            'arguably', 'supposedly', 'allegedly', 'seemingly',
            'kind of', 'sort of', 'more or less'
        ]
        hedge_count = 0
        response_lower = response.lower()
        for hw in hedge_words:
            hedge_count += response_lower.count(hw)
        
        hedge_density = hedge_count / max(word_count / 30, 1)
        score -= min(hedge_density * 3, 6)
        
        # === 8. Structural Detail Markers ===
        # Parenthetical details (e.g., "(also known as...)", "(approximately 50%)")
        parentheticals = re.findall(r'\([^)]{3,}\)', response)
        score += min(len(parentheticals) * 1.5, 5)
        
        # Colon-based elaborations
        colon_elaborations = re.findall(r':\s*\w', response)
        score += min(len(colon_elaborations) * 1.0, 4)
        
        # Quoted terms or phrases
        quoted = re.findall(r'["\u201c][^"\u201d]+["\u201d]', response)
        score += min(len(quoted) * 1.0, 4)
        
        # === 9. Actionable Language (verbs with specific objects) ===
        action_patterns = [
            r'\b(?:track|monitor|analyze|categorize|organize|manage|configure|implement|deploy|execute|optimize|calculate|measure|evaluate|assess|diagnose)\b',
            r'\b(?:download|upload|install|submit|register|subscribe|navigate|select|click|enter|input|specify|define|customize)\b',
        ]
        action_count = 0
        for pat in action_patterns:
            action_count += len(re.findall(pat, response, re.IGNORECASE))
        
        action_density = action_count / max(word_count, 1)
        score += min(action_density * 80, 8)
        
        # === 10. Comparative/Contrastive Specificity ===
        comparison_patterns = [
            r'\bwhile\b.*\b(?:instead|rather|whereas)\b',
            r'\bunlike\b',
            r'\bin contrast\b',
            r'\bon the other hand\b',
            r'\bwhereas\b',
            r'\bcompared to\b',
        ]
        comparison_count = 0
        for pat in comparison_patterns:
            comparison_count += len(re.findall(pat, response, re.IGNORECASE))
        score += min(comparison_count * 1.5, 5)
        
        # === 11. Response Length Bonus (with diminishing returns) ===
        # Longer responses tend to have more evidence, but with diminishing returns
        length_bonus = math.log(max(word_count, 1) + 1) * 1.5
        score += min(length_bonus, 8)
        
        # === 12. Sentence Complexity / Information per Sentence ===
        sentence_list = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        num_sentences = max(len(sentence_list), 1)
        
        # Average words per sentence (moderate length = more detailed)
        avg_words_per_sent = word_count / num_sentences
        if 10 <= avg_words_per_sent <= 25:
            score += 3
        elif 8 <= avg_words_per_sent < 10 or 25 < avg_words_per_sent <= 35:
            score += 1.5
        
        # === 13. Comma Density (indicates embedded details/lists) ===
        comma_count = response.count(',')
        comma_density = comma_count / max(num_sentences, 1)
        if 1 <= comma_density <= 4:
            score += min(comma_density * 1.5, 5)
        
        # === 14. Repetition Penalty ===
        # Detect repeated phrases (3-grams)
        if len(lower_words) >= 3:
            trigrams = [' '.join(lower_words[i:i+3]) for i in range(len(lower_words)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
            score -= min(repeated_trigrams * 3, 15)
        
        # === 15. Empty/Nonsense Detection ===
        if re.match(r'^<noinput>$', response.strip(), re.IGNORECASE):
            return 0.5
        
        # Detect if response is mostly repetitive gibberish
        if word_count > 10:
            unique_words = len(set(lower_words))
            if unique_words / len(lower_words) < 0.2:
                score -= 10
        
        # Normalize to 0-100 range
        score = max(0, min(100, score))
        
        return round(score, 2)
        
    except Exception:
        return 5.0