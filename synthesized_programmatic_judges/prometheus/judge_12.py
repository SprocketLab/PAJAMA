def judging_function(query, response):
    """
    Evaluate language quality and readability using:
    - Punctuation diversity and correctness
    - Sentence structure variance (std dev of sentence lengths)
    - Hapax legomena ratio (words appearing exactly once) as sophistication measure
    - Connective/discourse marker density
    - Comma-to-sentence ratio (clause complexity)
    - Character-level entropy as a proxy for linguistic richness
    - Proportion of sentences starting with different words
    """
    import re
    import math
    import collections
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+", text)
        if len(words) < 3:
            return 0.5
        
        # --- 1. Character-level entropy (linguistic richness) ---
        char_counts = collections.Counter(text.lower())
        total_chars = len(text)
        char_entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                char_entropy -= p * math.log2(p)
        # Normalize: English text typically has entropy ~4.0-4.5
        # Scale to 0-10
        entropy_score = min(max((char_entropy - 3.0) / 1.5 * 10, 0), 10)
        
        # --- 2. Sentence structure variance ---
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        
        if len(sentences) < 2:
            sent_variance_score = 3.0
        else:
            sent_lengths = [len(re.findall(r'\S+', s)) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Some variance is good (variety), but too much is chaotic
            # Optimal std_dev around 4-8
            if std_dev < 1:
                sent_variance_score = 3.0
            elif std_dev <= 8:
                sent_variance_score = 3.0 + (std_dev - 1) / 7 * 7  # 3 to 10
            else:
                sent_variance_score = max(10 - (std_dev - 8) * 0.5, 3)
        
        # --- 3. Sentence opener diversity ---
        if len(sentences) >= 2:
            openers = []
            for s in sentences:
                w = re.findall(r'[a-zA-Z]+', s)
                if w:
                    openers.append(w[0].lower())
            if openers:
                unique_openers = len(set(openers))
                opener_diversity = unique_openers / len(openers)
            else:
                opener_diversity = 0.5
        else:
            opener_diversity = 0.5
        opener_score = opener_diversity * 10
        
        # --- 4. Hapax legomena ratio (vocabulary sophistication) ---
        word_lower = [w.lower() for w in words]
        word_freq = collections.Counter(word_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / len(word_freq) if word_freq else 0
        # Typical range 0.4-0.8; higher = more diverse vocabulary
        hapax_score = min(hapax_ratio * 12, 10)
        
        # --- 5. Punctuation diversity ---
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-()"\'"':
                punct_types.add(ch)
        # Count specific punctuation
        commas = text.count(',')
        semicolons = text.count(';')
        colons = text.count(':')
        dashes = text.count('-') + text.count('—') + text.count('–')
        
        punct_diversity = len(punct_types)
        punct_score = min(punct_diversity * 1.5, 10)
        
        # --- 6. Comma-to-sentence ratio (clause complexity) ---
        num_sentences = max(len(sentences), 1)
        comma_ratio = commas / num_sentences
        # Optimal: 1-3 commas per sentence
        if comma_ratio < 0.5:
            comma_score = 3.0
        elif comma_ratio <= 3.0:
            comma_score = 3.0 + (comma_ratio - 0.5) / 2.5 * 7
        else:
            comma_score = max(10 - (comma_ratio - 3) * 1.5, 3)
        
        # --- 7. Discourse/connective markers density ---
        discourse_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\btherefore\b',
            r'\bnevertheless\b', r'\bconsequently\b', r'\bmeanwhile\b',
            r'\bin addition\b', r'\bon the other hand\b', r'\bfor instance\b',
            r'\bfor example\b', r'\bin fact\b', r'\bas a result\b',
            r'\bin contrast\b', r'\bsimilarly\b', r'\bspecifically\b',
            r'\bultimately\b', r'\bindeed\b', r'\bthus\b', r'\bhence\b',
            r'\bnonetheless\b', r'\baccordingly\b', r'\blet\'s\b',
            r'\bremember\b', r'\bfirst\b', r'\bsecond\b', r'\bnext\b',
            r'\bfinally\b', r'\balso\b', r'\badditionally\b',
            r'\bimportantly\b', r'\bnotably\b'
        ]
        text_lower = text.lower()
        marker_count = 0
        for pattern in discourse_markers:
            marker_count += len(re.findall(pattern, text_lower))
        
        marker_density = marker_count / num_sentences
        # Optimal: 0.3-1.0 markers per sentence
        if marker_density < 0.1:
            marker_score = 2.0
        elif marker_density <= 1.0:
            marker_score = 2.0 + (marker_density - 0.1) / 0.9 * 8
        else:
            marker_score = max(10 - (marker_density - 1.0) * 2, 4)
        
        # --- 8. Empathy/engagement markers (contextual quality) ---
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'s\b',
            r'\bcompletely\b', r'\babsolutely\b', r'\bperfectly\b',
            r'\bunderstandable\b', r'\bnatural\b', r'\bi\'m sorry\b',
            r'\bdon\'t hesitate\b', r'\bfeel free\b', r'\bhere are\b',
            r'\blet me\b', r'\bplease\b', r'\bthank\b'
        ]
        empathy_count = 0
        for pattern in empathy_patterns:
            empathy_count += len(re.findall(pattern, text_lower))
        
        empathy_score = min(empathy_count * 2.0, 8)
        
        # --- 9. Paragraph structure ---
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        if num_paragraphs >= 2:
            para_score = min(num_paragraphs * 2, 8)
        else:
            para_score = 3.0
        
        # --- 10. Numbered/structured list detection ---
        list_items = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
        list_items += len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
        structure_bonus = min(list_items * 1.0, 4)
        
        # --- 11. Average word length (moderate is best) ---
        avg_word_len = sum(len(w) for w in words) / len(words)
        # Optimal: 4.5-5.5
        if 4.0 <= avg_word_len <= 6.0:
            word_len_score = 8.0
        elif 3.5 <= avg_word_len <= 6.5:
            word_len_score = 6.0
        else:
            word_len_score = 3.0
        
        # --- 12. Response length adequacy ---
        word_count = len(words)
        if word_count < 20:
            length_score = 2.0
        elif word_count < 50:
            length_score = 5.0
        elif word_count <= 200:
            length_score = 8.0
        else:
            length_score = 7.0
        
        # --- Weighted combination ---
        final_score = (
            entropy_score * 0.08 +
            sent_variance_score * 0.10 +
            opener_score * 0.10 +
            hapax_score * 0.08 +
            punct_score * 0.07 +
            comma_score * 0.08 +
            marker_score * 0.12 +
            empathy_score * 0.10 +
            para_score * 0.07 +
            structure_bonus * 0.05 +
            word_len_score * 0.07 +
            length_score * 0.08
        )
        
        # Scale to 1-5 range to match the examples
        scaled = 1.0 + (final_score / 10.0) * 4.0
        scaled = max(1.0, min(5.0, scaled))
        
        return round(scaled, 3)
    
    except Exception:
        return 2.5