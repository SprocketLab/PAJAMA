def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a pattern-matching approach
    focused on detecting concrete linguistic markers: proper nouns, numbers,
    technical terms, specific actions, and structural completeness.
    
    This variant uses a "specificity token classification" approach where each
    token/phrase is classified into specificity categories and scored accordingly,
    combined with sentence-level analysis of information packaging density.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        words = response.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # Normalize words for analysis
        lower_response = response.lower()
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Numeric/Quantitative Evidence Detection
        # Look for numbers, percentages, dates, measurements, ranges
        # ============================================================
        number_patterns = [
            r'\b\d+\.?\d*\s*%',           # percentages
            r'\b\d{4}\b',                   # years
            r'\$\s*\d+',                    # dollar amounts
            r'\b\d+\.?\d*\s*(million|billion|trillion|thousand)',  # large numbers
            r'\b\d+\.?\d*\s*(kg|lb|km|mi|cm|mm|gb|mb|tb|hz|mhz|ghz)',  # measurements
            r'\b\d+\s*[-–]\s*\d+\b',       # ranges like 5-10
            r'\b\d+(?:st|nd|rd|th)\b',      # ordinals
            r'\b\d+:\d+\b',                # time
            r'\b\d+/\d+\b',                # fractions/dates
        ]
        
        numeric_count = 0
        for pattern in number_patterns:
            numeric_count += len(re.findall(pattern, lower_response))
        
        # Also count standalone numbers
        standalone_numbers = re.findall(r'\b\d+\.?\d*\b', response)
        numeric_count += len(standalone_numbers) * 0.5
        
        numeric_density = numeric_count / max(word_count, 1)
        score += min(numeric_density * 80, 15)
        
        # ============================================================
        # FEATURE 2: Named Entity Proxy Detection
        # Detect capitalized multi-word phrases (likely proper nouns/names)
        # ============================================================
        # Find capitalized words not at sentence start
        sentences = re.split(r'[.!?]+', response)
        named_entity_count = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = sent.split()
            if len(sent_words) < 2:
                continue
            # Skip first word (sentence start), check for capitalized words
            for w in sent_words[1:]:
                cleaned = re.sub(r'[^a-zA-Z]', '', w)
                if cleaned and len(cleaned) > 1 and cleaned[0].isupper() and not cleaned.isupper():
                    named_entity_count += 1
        
        # Capitalized sequences (multi-word proper nouns)
        cap_sequences = re.findall(r'(?<!\. )(?<!\.\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', response)
        named_entity_count += len(cap_sequences) * 2
        
        ne_density = named_entity_count / max(word_count, 1)
        score += min(ne_density * 60, 12)
        
        # ============================================================
        # FEATURE 3: Specificity Lexicon Scoring
        # Words that indicate concrete/specific content vs vague content
        # ============================================================
        
        # Specific/concrete indicator words (weighted)
        specific_markers = {
            'specifically': 2, 'particular': 1.5, 'exactly': 1.5, 'precisely': 2,
            'namely': 2, 'including': 1.5, 'such as': 2, 'for example': 2,
            'for instance': 2, 'e.g.': 2, 'i.e.': 1.5, 'in particular': 2,
            'defined as': 1.5, 'known as': 1.5, 'referred to as': 1.5,
            'consists of': 1.5, 'comprised of': 1.5, 'characterized by': 1.5,
            'resulting in': 1, 'leading to': 1, 'because': 0.8,
            'therefore': 0.8, 'consequently': 1, 'due to': 1,
            'according to': 2, 'research shows': 2, 'studies show': 2,
            'data suggests': 2, 'evidence indicates': 2,
        }
        
        specificity_score = 0
        for marker, weight in specific_markers.items():
            count = lower_response.count(marker)
            specificity_score += count * weight
        
        score += min(specificity_score * 2, 15)
        
        # ============================================================
        # FEATURE 4: Vagueness / Hedging Penalty
        # Penalize vague, non-committal language
        # ============================================================
        vague_phrases = [
            'many people', 'some people', 'it depends', 'various factors',
            'there are many', 'there are various', 'in many ways',
            'a lot of', 'lots of', 'kind of', 'sort of',
            'more or less', 'to some extent', 'in some cases',
            'it is important to', 'it is essential to',
            'it can be said', 'one could argue', 'it is worth noting',
            'needless to say', 'goes without saying',
            'and so on', 'and so forth', 'etc etc',
            'things like that', 'stuff like that',
            'you know', 'basically', 'essentially',
            'in general', 'generally speaking', 'broadly speaking',
            'as we all know', 'everyone knows',
        ]
        
        vague_count = 0
        for phrase in vague_phrases:
            vague_count += lower_response.count(phrase)
        
        vague_density = vague_count / max(word_count / 20, 1)
        score -= min(vague_density * 4, 10)
        
        # ============================================================
        # FEATURE 5: Information Density per Sentence
        # Measure how much unique information each sentence carries
        # ============================================================
        
        sentence_splits = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentence_splits if len(s.strip()) > 10]
        num_sentences = max(len(valid_sentences), 1)
        
        # Count unique content words per sentence
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
            'he', 'she', 'him', 'her', 'his', 'we', 'us', 'our', 'you', 'your',
            'i', 'me', 'my', 'mine', 'and', 'or', 'but', 'nor', 'not', 'no',
            'if', 'then', 'than', 'so', 'as', 'of', 'in', 'on', 'at', 'to',
            'for', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'also', 'very', 'just',
            'while', 'which', 'who', 'whom', 'what', 'whose',
        }
        
        all_content_words = []
        sentence_content_counts = []
        
        for sent in valid_sentences:
            sent_words = re.findall(r'[a-z]+', sent.lower())
            content_words = [w for w in sent_words if w not in stop_words and len(w) > 2]
            sentence_content_counts.append(len(content_words))
            all_content_words.extend(content_words)
        
        # Average content words per sentence
        avg_content_per_sentence = sum(sentence_content_counts) / max(num_sentences, 1)
        score += min(avg_content_per_sentence * 1.2, 10)
        
        # ============================================================
        # FEATURE 6: Unique Information Ratio
        # Penalize repetition heavily - repeated content is not new evidence
        # ============================================================
        
        content_word_counter = Counter(all_content_words)
        total_content = len(all_content_words)
        unique_content = len(content_word_counter)
        
        if total_content > 0:
            uniqueness_ratio = unique_content / total_content
            # Penalize very low uniqueness (high repetition)
            if uniqueness_ratio < 0.3:
                score -= 8
            elif uniqueness_ratio < 0.5:
                score -= 3
            else:
                score += uniqueness_ratio * 8
        
        # Check for repeated phrases (3-grams)
        if word_count >= 3:
            trigrams = []
            lower_words = lower_response.split()
            for i in range(len(lower_words) - 2):
                trigrams.append(' '.join(lower_words[i:i+3]))
            trigram_counter = Counter(trigrams)
            repeated_trigrams = sum(1 for _, c in trigram_counter.items() if c > 1)
            total_trigrams = max(len(trigrams), 1)
            repetition_ratio = repeated_trigrams / total_trigrams
            score -= repetition_ratio * 15
        
        # ============================================================
        # FEATURE 7: Technical/Domain Vocabulary Density
        # Longer, more complex words often indicate technical specificity
        # ============================================================
        
        word_lengths = [len(w) for w in re.findall(r'[a-zA-Z]+', response)]
        if word_lengths:
            avg_word_length = sum(word_lengths) / len(word_lengths)
            long_words = sum(1 for l in word_lengths if l >= 8)
            long_word_ratio = long_words / len(word_lengths)
            
            # Reward moderate-to-high average word length
            if avg_word_length > 5.5:
                score += min((avg_word_length - 5.5) * 3, 6)
            
            score += min(long_word_ratio * 25, 8)
        
        # ============================================================
        # FEATURE 8: Causal/Explanatory Chain Detection
        # Specific explanations with cause-effect relationships
        # ============================================================
        
        causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'resulting in', 'leads to', 'caused by',
            'due to', 'owing to', 'this means', 'which means',
            'in order to', 'so that', 'if.*then', 'when.*will',
        ]
        
        causal_count = 0
        for marker in causal_markers:
            causal_count += len(re.findall(marker, lower_response))
        
        score += min(causal_count * 1.5, 8)
        
        # ============================================================
        # FEATURE 9: Structural Completeness & Detail Layering
        # Responses with multiple distinct points/aspects score higher
        # ============================================================
        
        # Count distinct clauses (approximated by commas, semicolons, conjunctions)
        clause_separators = len(re.findall(r'[,;]|\band\b|\bwhile\b|\bwhereas\b|\balthough\b', lower_response))
        clause_density = clause_separators / max(num_sentences, 1)
        score += min(clause_density * 1.5, 6)
        
        # ============================================================
        # FEATURE 10: Response Length Scaling (diminishing returns)
        # Longer responses have more room for evidence, but with diminishing returns
        # ============================================================
        
        if word_count < 10:
            length_factor = 0.3
        elif word_count < 25:
            length_factor = 0.6
        elif word_count < 50:
            length_factor = 0.85
        elif word_count < 100:
            length_factor = 1.0
        elif word_count < 200:
            length_factor = 1.05
        else:
            length_factor = 1.08
        
        score *= length_factor
        
        # ============================================================
        # FEATURE 11: Action Verb Density
        # Specific action verbs indicate concrete descriptions
        # ============================================================
        
        action_verbs = [
            'create', 'build', 'design', 'implement', 'develop', 'generate',
            'analyze', 'calculate', 'measure', 'track', 'monitor', 'detect',
            'send', 'receive', 'process', 'transform', 'convert', 'extract',
            'display', 'render', 'compile', 'execute', 'deploy', 'configure',
            'categorize', 'classify', 'organize', 'prioritize', 'optimize',
            'reduce', 'increase', 'improve', 'enhance', 'modify', 'adjust',
            'connect', 'integrate', 'combine', 'separate', 'filter', 'sort',
            'validate', 'verify', 'authenticate', 'encrypt', 'decrypt',
            'allows', 'enables', 'provides', 'includes', 'contains', 'requires',
            'involves', 'utilizes', 'employs', 'applies', 'demonstrates',
        ]
        
        action_count = 0
        for verb in action_verbs:
            # Match word boundaries
            action_count += len(re.findall(r'\b' + verb + r'(?:s|ed|ing|es)?\b', lower_response))
        
        action_density = action_count / max(word_count, 1)
        score += min(action_density * 60, 10)
        
        # ============================================================
        # FEATURE 12: Parenthetical/Qualifying Detail
        # Parenthetical additions often add specific detail
        # ============================================================
        
        parens = len(re.findall(r'\([^)]+\)', response))
        quotes = len(re.findall(r'"[^"]+"|\'[^\']+\'', response))
        score += min((parens + quotes) * 1.5, 5)
        
        # ============================================================
        # FEATURE 13: Contrast/Comparison Markers
        # Comparing specific things indicates analytical depth
        # ============================================================
        
        contrast_markers = [
            'however', 'whereas', 'in contrast', 'on the other hand',
            'unlike', 'compared to', 'rather than', 'instead of',
            'while', 'although', 'but', 'yet', 'nevertheless',
            'differ', 'different', 'distinction', 'distinguish',
        ]
        
        contrast_count = 0
        for marker in contrast_markers:
            contrast_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', lower_response))
        
        score += min(contrast_count * 0.8, 5)
        
        # Ensure score is in reasonable range [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        # Fallback: return a minimal score based on length
        try:
            return min(len(str(response).split()) * 0.1, 10)
        except Exception:
            return 0.0