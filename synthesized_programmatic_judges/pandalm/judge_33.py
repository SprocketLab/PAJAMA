def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    Uses analysis of logical flow, structure, transitions, contradictions,
    and argument completeness. Higher scores = better quality.
    Returns a score roughly in range 0-100.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        # Handle edge cases
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 5.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # ============================================================
        # FEATURE 1: Structural completeness (sentences, paragraphs)
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        # Reward having multiple sentences (shows developed argument)
        if num_sentences == 0:
            sentence_score = 0
        elif num_sentences == 1:
            sentence_score = 15
        elif num_sentences == 2:
            sentence_score = 35
        elif num_sentences <= 5:
            sentence_score = 50 + (num_sentences - 2) * 5
        elif num_sentences <= 10:
            sentence_score = 65
        else:
            sentence_score = 60  # Very long might be rambling
        
        # ============================================================
        # FEATURE 2: Logical connectors and transition words
        # ============================================================
        response_lower = response.lower()
        
        # Causal/logical connectors
        causal_connectors = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'in order to',
            'leads to', 'causes', 'results in', 'owing to'
        ]
        
        # Contrast/comparison connectors
        contrast_connectors = [
            'however', 'but', 'although', 'whereas', 'while',
            'on the other hand', 'in contrast', 'nevertheless',
            'despite', 'yet', 'conversely', 'unlike', 'rather than'
        ]
        
        # Additive/elaboration connectors
        additive_connectors = [
            'also', 'furthermore', 'moreover', 'in addition',
            'additionally', 'as well', 'not only', 'besides',
            'along with', 'coupled with'
        ]
        
        # Sequence/structure connectors
        sequence_connectors = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'initially', 'subsequently', 'afterward', 'lastly',
            'to begin', 'in conclusion', 'to summarize'
        ]
        
        # Clarification connectors
        clarification_connectors = [
            'for example', 'for instance', 'such as', 'specifically',
            'in particular', 'namely', 'that is', 'in other words',
            'this means', 'meaning that'
        ]
        
        def count_connectors(connector_list, text):
            count = 0
            for c in connector_list:
                count += len(re.findall(r'\b' + re.escape(c) + r'\b', text))
            return count
        
        causal_count = count_connectors(causal_connectors, response_lower)
        contrast_count = count_connectors(contrast_connectors, response_lower)
        additive_count = count_connectors(additive_connectors, response_lower)
        sequence_count = count_connectors(sequence_connectors, response_lower)
        clarification_count = count_connectors(clarification_connectors, response_lower)
        
        total_connectors = (causal_count + contrast_count + additive_count + 
                           sequence_count + clarification_count)
        
        # Variety of connector types used
        connector_types_used = sum([
            causal_count > 0,
            contrast_count > 0,
            additive_count > 0,
            sequence_count > 0,
            clarification_count > 0
        ])
        
        # Score connectors: both quantity and variety matter
        connector_quantity_score = min(total_connectors * 4, 30)
        connector_variety_score = connector_types_used * 6  # max 30
        connector_score = min(connector_quantity_score + connector_variety_score, 50)
        
        # ============================================================
        # FEATURE 3: Repetition detection (sign of poor logic/padding)
        # ============================================================
        words = re.findall(r'\b\w+\b', response_lower)
        num_words = len(words)
        
        if num_words == 0:
            return 1.0
        
        # Check for repeated phrases (3-grams)
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i:i+3]))
        
        trigram_counts = Counter(trigrams)
        if trigrams:
            max_trigram_repeat = max(trigram_counts.values())
            total_trigrams = len(trigrams)
            # Ratio of the most repeated trigram
            repetition_ratio = max_trigram_repeat / max(total_trigrams, 1)
        else:
            max_trigram_repeat = 0
            repetition_ratio = 0
        
        # Heavy penalty for excessive repetition
        if max_trigram_repeat > 3 and repetition_ratio > 0.15:
            repetition_penalty = -30
        elif max_trigram_repeat > 2 and repetition_ratio > 0.1:
            repetition_penalty = -15
        elif repetition_ratio > 0.05 and max_trigram_repeat > 2:
            repetition_penalty = -8
        else:
            repetition_penalty = 0
        
        # Check for word-level repetition (same word repeated consecutively)
        consecutive_repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1] and words[i] not in {'the', 'a', 'an', 'is', 'and', 'or', 'to', 'of'}:
                consecutive_repeats += 1
        
        if consecutive_repeats > 2:
            repetition_penalty -= 10
        
        # ============================================================
        # FEATURE 4: Response completeness (not truncated)
        # ============================================================
        completeness_score = 0
        
        # Check if response ends with proper punctuation
        if response.rstrip()[-1] in '.!?':
            completeness_score += 10
        else:
            completeness_score -= 5
        
        # Check for truncation indicators
        if response.rstrip().endswith('...') or response.rstrip().endswith('…'):
            completeness_score -= 5
        
        # Check if the last sentence seems complete (has a verb-like structure)
        last_sentence = sentences[-1] if sentences else ""
        last_words = re.findall(r'\b\w+\b', last_sentence.lower())
        if len(last_words) >= 3:
            completeness_score += 5
        elif len(last_words) == 1:
            completeness_score -= 3
        
        # ============================================================
        # FEATURE 5: Vocabulary richness (lexical diversity)
        # ============================================================
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or',
                     'but', 'not', 'no', 'if', 'then', 'than', 'so', 'as'}
        
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        if content_words:
            unique_content = set(content_words)
            lexical_diversity = len(unique_content) / len(content_words)
        else:
            lexical_diversity = 0
        
        diversity_score = lexical_diversity * 15  # max ~15
        
        # ============================================================
        # FEATURE 6: Query relevance (topic alignment)
        # ============================================================
        query_words = set(re.findall(r'\b\w+\b', query.lower())) - stopwords
        response_content_set = set(content_words)
        
        if query_words:
            overlap = len(query_words & response_content_set)
            relevance_ratio = overlap / len(query_words)
        else:
            relevance_ratio = 0.5
        
        relevance_score = relevance_ratio * 15  # max 15
        
        # ============================================================
        # FEATURE 7: Argument depth - premise-conclusion structure
        # ============================================================
        depth_score = 0
        
        # Check for explanatory patterns
        explanatory_patterns = [
            r'\bthis means\b', r'\bthis suggests\b', r'\bthis implies\b',
            r'\bwhich means\b', r'\bwhich suggests\b', r'\bwhich indicates\b',
            r'\bin other words\b', r'\bmore specifically\b',
            r'\bthe reason\b', r'\bone reason\b', r'\banother reason\b',
            r'\bimportant(?:ly)?\b', r'\bsignificant(?:ly)?\b',
            r'\bit is important\b', r'\bit suggests\b'
        ]
        
        for pattern in explanatory_patterns:
            if re.search(pattern, response_lower):
                depth_score += 3
        
        depth_score = min(depth_score, 15)
        
        # Check for multi-point arguments
        bullet_or_number_pattern = r'(?:^|\n)\s*(?:\d+[.)\-]|[-•*])\s+'
        numbered_points = len(re.findall(bullet_or_number_pattern, response))
        if numbered_points >= 2:
            depth_score += min(numbered_points * 2, 8)
        
        depth_score = min(depth_score, 20)
        
        # ============================================================
        # FEATURE 8: Internal contradiction detection
        # ============================================================
        contradiction_penalty = 0
        
        # Simple contradiction patterns
        contradiction_patterns = [
            (r'\bis\b.*\bbut is not\b', -5),
            (r'\balways\b.*\bnever\b', -3),
            (r'\beverything\b.*\bnothing\b', -2),
        ]
        
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, response_lower):
                contradiction_penalty += penalty
        
        # ============================================================
        # FEATURE 9: Response length appropriateness
        # ============================================================
        length_score = 0
        
        # Very short responses are usually low quality
        if num_words < 5:
            length_score = -15
        elif num_words < 10:
            length_score = -5
        elif num_words < 20:
            length_score = 5
        elif num_words < 50:
            length_score = 10
        elif num_words < 150:
            length_score = 12
        elif num_words < 300:
            length_score = 10
        else:
            length_score = 7  # Very long can be rambling
        
        # ============================================================
        # FEATURE 10: Coherent opening (addresses the query)
        # ============================================================
        opening_score = 0
        
        first_sentence = sentences[0].lower() if sentences else ""
        
        # Check if response begins by addressing the topic
        query_content_words = [w for w in query_words if len(w) > 3]
        first_sent_words = set(re.findall(r'\b\w+\b', first_sentence))
        
        if query_content_words:
            first_sent_overlap = len(set(query_content_words) & first_sent_words)
            if first_sent_overlap > 0:
                opening_score += 5
            if first_sent_overlap >= 2:
                opening_score += 3
        
        # ============================================================
        # FEATURE 11: Sentence-to-sentence coherence
        # ============================================================
        coherence_score = 0
        
        if num_sentences >= 2:
            # Check if consecutive sentences share content words (topic continuity)
            sentence_word_sets = []
            for s in sentences:
                s_words = set(re.findall(r'\b\w+\b', s.lower())) - stopwords
                sentence_word_sets.append(s_words)
            
            transitions_with_overlap = 0
            total_transitions = len(sentence_word_sets) - 1
            
            for i in range(total_transitions):
                if sentence_word_sets[i] and sentence_word_sets[i+1]:
                    overlap = len(sentence_word_sets[i] & sentence_word_sets[i+1])
                    if overlap > 0:
                        transitions_with_overlap += 1
            
            if total_transitions > 0:
                coherence_ratio = transitions_with_overlap / total_transitions
                coherence_score = coherence_ratio * 15
            else:
                coherence_score = 5
        else:
            coherence_score = 3  # Single sentence - can't evaluate transitions well
        
        # ============================================================
        # FEATURE 12: Check for gibberish / nonsensical content
        # ============================================================
        gibberish_penalty = 0
        
        # Check for excessive special characters
        alpha_chars = sum(1 for c in response if c.isalpha())
        total_chars = len(response)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.5:
                gibberish_penalty = -15
            elif alpha_ratio < 0.65:
                gibberish_penalty = -5
        
        # Check for very long words (likely gibberish)
        long_words = [w for w in words if len(w) > 20]
        if len(long_words) > 2:
            gibberish_penalty -= 10
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        total_score = (
            sentence_score * 0.20 +       # Structure: 0-13
            connector_score * 0.25 +       # Logical connectors: 0-12.5
            completeness_score * 0.10 +    # Completeness: -1 to 1.5
            diversity_score * 0.10 +       # Vocabulary: 0-1.5
            relevance_score * 0.10 +       # Relevance: 0-1.5
            depth_score * 0.15 +           # Argument depth: 0-3
            coherence_score * 0.15 +       # Sentence coherence: 0-2.25
            opening_score * 0.10 +         # Opening quality: 0-0.8
            length_score * 0.15 +          # Length appropriateness: -2.25 to 1.8
            repetition_penalty +           # Repetition: -30 to 0
            contradiction_penalty +        # Contradictions: -10 to 0
            gibberish_penalty              # Gibberish: -25 to 0
        )
        
        # Normalize to 0-100 range
        # Theoretical rough range: about -65 to +40
        # Map to 0-100
        normalized_score = (total_score + 10) * 2.0  # Shift and scale
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, normalized_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middling score based on length
        try:
            if response and len(response.strip()) > 0:
                return min(max(len(response.strip()) / 10.0, 5.0), 50.0)
            return 0.0
        except:
            return 25.0