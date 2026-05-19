def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question decomposition (detecting sub-questions/aspects in query)
    - Information density (unique information units per sentence)
    - Structural depth analysis (nested explanations, elaborations)
    - Repetition penalty
    - Topic coverage via keyword extraction from query
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 5.0
        
        query = query.strip()
        response = response.strip()
        
        # === 1. Query Complexity Analysis ===
        # Estimate how many aspects/sub-questions the query contains
        
        # Count explicit question markers
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'describe',
                         'explain', 'compare', 'contrast', 'list', 'provide', 'generate',
                         'create', 'write', 'give', 'name', 'identify', 'discuss', 'analyze']
        
        query_lower = query.lower()
        query_words = re.findall(r'\b[a-z]+\b', query_lower)
        
        # Count conjunctions and commas suggesting multiple aspects
        multi_aspect_markers = query_lower.count(' and ') + query_lower.count(',') + query_lower.count(';')
        question_mark_count = query.count('?')
        
        # Estimate expected complexity
        query_complexity = 1.0 + multi_aspect_markers * 0.5 + max(0, question_mark_count - 1) * 0.5
        
        # Check for comparison queries
        is_comparison = any(w in query_lower for w in ['compare', 'contrast', 'difference', 'similar', 'versus', 'vs'])
        if is_comparison:
            query_complexity += 1.0
        
        # Check for enumeration queries
        is_enumeration = any(w in query_lower for w in ['list', 'examples', 'provide', 'name several', 'give me'])
        
        # === 2. Response Sentence Analysis ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        response_lower = response.lower()
        response_words = re.findall(r'\b[a-z]+\b', response_lower)
        total_words = len(response_words)
        
        if total_words == 0:
            return 0.5
        
        # === 3. Unique Information Density ===
        # Measure how much unique information each sentence contributes
        unique_words = set(response_words)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'that', 'this', 'these',
                     'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
                     'your', 'he', 'she', 'him', 'her', 'his', 'and', 'but', 'or', 'nor',
                     'not', 'so', 'if', 'than', 'too', 'very', 'just', 'also', 'more',
                     'about', 'up', 'which', 'who', 'whom', 'what', 'when', 'where', 'how',
                     'all', 'each', 'every', 'both', 'few', 'some', 'such', 'no', 'only',
                     'own', 'same', 'other', 'while'}
        
        content_words = [w for w in response_words if w not in stopwords and len(w) > 2]
        unique_content_words = set(content_words)
        content_word_count = len(content_words)
        unique_content_count = len(unique_content_words)
        
        # Information density: ratio of unique content words to total content words
        if content_word_count > 0:
            info_density = unique_content_count / content_word_count
        else:
            info_density = 0.0
        
        # === 4. Repetition Penalty ===
        # Detect excessive repetition at word level and phrase level
        word_counts = Counter(content_words)
        if content_word_count > 0:
            max_word_freq = max(word_counts.values()) if word_counts else 0
            repetition_ratio = max_word_freq / content_word_count
        else:
            repetition_ratio = 0.0
        
        # Detect repeated phrases (3-grams)
        trigrams = []
        for i in range(len(response_words) - 2):
            trigrams.append(' '.join(response_words[i:i+3]))
        trigram_counts = Counter(trigrams)
        if trigrams:
            max_trigram_freq = max(trigram_counts.values())
            trigram_repetition = max_trigram_freq / max(len(trigrams), 1)
        else:
            trigram_repetition = 0.0
        
        # Heavy repetition penalty
        repetition_penalty = 0.0
        if repetition_ratio > 0.15:
            repetition_penalty += (repetition_ratio - 0.15) * 15
        if trigram_repetition > 0.1:
            repetition_penalty += (trigram_repetition - 0.1) * 20
        
        # Check for degenerate repetition (same word/phrase repeated many times)
        if max(word_counts.values(), default=0) > 5 and unique_content_count < 10:
            repetition_penalty += 5.0
        
        repetition_penalty = min(repetition_penalty, 10.0)
        
        # === 5. Elaboration Depth ===
        # Detect explanatory connectors and elaboration patterns
        elaboration_markers = ['because', 'since', 'therefore', 'thus', 'consequently',
                              'as a result', 'for example', 'for instance', 'such as',
                              'in other words', 'specifically', 'in particular',
                              'furthermore', 'moreover', 'additionally', 'in addition',
                              'however', 'on the other hand', 'nevertheless', 'although',
                              'whereas', 'while', 'this means', 'this suggests',
                              'in contrast', 'similarly', 'likewise', 'unlike']
        
        elaboration_count = sum(1 for marker in elaboration_markers if marker in response_lower)
        elaboration_score = min(elaboration_count / max(query_complexity, 1.0), 3.0)
        
        # === 6. Query Aspect Coverage ===
        # Extract key content words from query and check coverage in response
        query_content_words = [w for w in query_words if w not in stopwords and len(w) > 2]
        query_content_set = set(query_content_words)
        
        if query_content_set:
            covered = sum(1 for w in query_content_set if w in response_lower)
            query_coverage = covered / len(query_content_set)
        else:
            query_coverage = 0.5
        
        # === 7. Comparison Coverage (if applicable) ===
        comparison_score = 0.0
        if is_comparison:
            # Check if response discusses both sides
            contrast_words = ['while', 'whereas', 'but', 'however', 'on the other hand',
                            'in contrast', 'unlike', 'differ', 'different', 'difference',
                            'similar', 'similarly', 'both', 'although']
            contrast_count = sum(1 for w in contrast_words if w in response_lower)
            comparison_score = min(contrast_count * 0.5, 2.0)
        
        # === 8. Enumeration Richness (if applicable) ===
        enumeration_score = 0.0
        if is_enumeration:
            # Count distinct items (approximate by counting commas + sentence boundaries)
            comma_items = response.count(',')
            enumeration_score = min(comma_items * 0.3, 2.0)
        
        # === 9. Structural Completeness ===
        # Check for introduction + body + conclusion-like structure
        has_intro = num_sentences >= 1
        has_body = num_sentences >= 2
        has_depth = num_sentences >= 3
        
        structural_score = (0.5 * has_intro + 1.0 * has_body + 1.0 * has_depth)
        
        # === 10. Response Length Adequacy ===
        # Score based on response length relative to query complexity
        # Use logarithmic scaling to avoid rewarding verbosity too much
        if total_words > 0:
            length_score = math.log(total_words + 1) / math.log(500)  # normalized
            length_score = min(length_score, 1.5)
        else:
            length_score = 0.0
        
        # Penalize very short responses for complex queries
        shortness_penalty = 0.0
        if total_words < 15 and query_complexity > 1.5:
            shortness_penalty = 2.0
        elif total_words < 8:
            shortness_penalty = 3.0
        
        # === 11. Specificity Score ===
        # Count specific/concrete indicators: numbers, proper nouns (capitalized words),
        # specific technical terms, adjectives suggesting detail
        numbers = len(re.findall(r'\b\d+\b', response))
        # Words with 7+ chars often more specific
        long_content_words = [w for w in content_words if len(w) >= 7]
        specificity = (len(long_content_words) / max(content_word_count, 1)) * 2.0 + min(numbers * 0.3, 1.0)
        specificity = min(specificity, 2.5)
        
        # === 12. Sentence-level Information Progression ===
        # Check that each sentence adds new information (not just rephrasing)
        sentence_word_sets = []
        for s in sentences:
            s_words = set(re.findall(r'\b[a-z]+\b', s.lower())) - stopwords
            sentence_word_sets.append(s_words)
        
        if len(sentence_word_sets) > 1:
            new_info_per_sentence = []
            cumulative = set()
            for sw in sentence_word_sets:
                new_words = sw - cumulative
                if sw:
                    new_info_per_sentence.append(len(new_words) / max(len(sw), 1))
                cumulative |= sw
            
            avg_new_info = sum(new_info_per_sentence) / len(new_info_per_sentence) if new_info_per_sentence else 0
            progression_score = avg_new_info * 2.0
        else:
            progression_score = 0.5
        
        # === 13. Truncation Detection ===
        truncation_penalty = 0.0
        if response.rstrip()[-1:] not in '.!?")\']' and total_words > 10:
            truncation_penalty = 1.5
        # Check if response ends mid-sentence
        last_chars = response.strip()[-5:]
        if last_chars and last_chars[-1] in 'abcdefghijklmnopqrstuvwxyz,':
            truncation_penalty = max(truncation_penalty, 1.0)
        
        # === FINAL SCORE COMPOSITION ===
        score = 0.0
        
        # Base: length adequacy (0-1.5) weighted by 10
        score += length_score * 10.0  # max ~15
        
        # Information density (0-1) weighted by 8
        score += info_density * 8.0  # max ~8
        
        # Query coverage (0-1) weighted by 12
        score += query_coverage * 12.0  # max ~12
        
        # Elaboration depth (0-3) weighted by 3
        score += elaboration_score * 3.0  # max ~9
        
        # Structural completeness (0-2.5) weighted by 3
        score += structural_score * 3.0  # max ~7.5
        
        # Comparison handling (0-2) weighted by 2
        score += comparison_score * 2.0  # max ~4
        
        # Enumeration handling (0-2) weighted by 2
        score += enumeration_score * 2.0  # max ~4
        
        # Specificity (0-2.5) weighted by 3
        score += specificity * 3.0  # max ~7.5
        
        # Progression (0-2) weighted by 4
        score += progression_score * 4.0  # max ~8
        
        # Penalties
        score -= repetition_penalty
        score -= shortness_penalty
        score -= truncation_penalty
        
        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 3)
    
    except Exception:
        # Fallback: return a minimal score based on response length
        try:
            words = len(response.split()) if response else 0
            return min(words * 0.3, 30.0)
        except Exception:
            return 0.0