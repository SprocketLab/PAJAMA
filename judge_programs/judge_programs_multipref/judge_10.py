def judging_function(query, response):
    """
    Evaluates language quality and readability using a combination of:
    - Coleman-Liau Index (character-based readability)
    - Gunning Fog Index approximation
    - Type-Token Ratio with hapax legomena ratio
    - Sentence structure variety (std dev of sentence lengths)
    - Punctuation diversity and correctness
    - Transition word usage
    - Paragraph/formatting structure
    """
    import re
    import math
    import collections
    import string
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.0
        
        # --- Tokenization ---
        # Split into sentences using multiple delimiters
        sentence_endings = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        # Words
        words = re.findall(r"[a-zA-Z']+(?:-[a-zA-Z']+)*", text)
        num_words = max(len(words), 1)
        words_lower = [w.lower() for w in words]
        
        # Characters (letters only)
        num_chars = sum(1 for c in text if c.isalpha())
        
        # --- 1. Coleman-Liau Index ---
        # CLI = 0.0588 * L - 0.296 * S - 15.8
        # L = avg letters per 100 words, S = avg sentences per 100 words
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        
        # Ideal Coleman-Liau for general audience: 8-12
        # Score peaks around 9-11, drops off for too simple or too complex
        cli_score = max(0, 10 - abs(coleman_liau - 10) * 0.8)
        
        # --- 2. Gunning Fog approximation ---
        # Count "complex words" (3+ syllables approximated by word length >= 7)
        complex_words = sum(1 for w in words_lower if len(w) >= 7)
        complex_ratio = complex_words / num_words
        avg_sentence_len = num_words / num_sentences
        fog_index = 0.4 * (avg_sentence_len + 100 * complex_ratio)
        
        # Ideal fog: 8-14 for clear, educated writing
        fog_score = max(0, 10 - abs(fog_index - 11) * 0.5)
        
        # --- 3. Vocabulary richness: hapax legomena ratio + TTR variant ---
        word_freq = collections.Counter(words_lower)
        unique_words = len(word_freq)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        
        # Yule's K approximation - use log TTR (Herdan's C)
        if num_words > 1:
            herdan_c = math.log(unique_words) / math.log(num_words) if num_words > 1 else 0
        else:
            herdan_c = 0
        
        # Hapax ratio (proportion of words used only once)
        hapax_ratio = hapax / max(unique_words, 1)
        
        # Good writing: herdan_c around 0.7-0.9, hapax_ratio around 0.5-0.7
        vocab_score = (min(herdan_c, 0.95) / 0.95) * 5 + (min(hapax_ratio, 0.8) / 0.8) * 5
        
        # --- 4. Sentence variety (std dev of sentence word counts) ---
        sent_word_counts = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                sent_word_counts.append(len(s_words))
        
        if len(sent_word_counts) >= 2:
            mean_swc = sum(sent_word_counts) / len(sent_word_counts)
            variance = sum((x - mean_swc) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_swc = math.sqrt(variance)
            # Coefficient of variation
            cv = std_swc / max(mean_swc, 1)
            # Good variety: CV around 0.3-0.6
            variety_score = min(cv / 0.5, 1.0) * 10
        else:
            variety_score = 3.0  # Single sentence gets moderate score
        
        # --- 5. Transition/discourse words ---
        transition_words = {
            'however', 'moreover', 'furthermore', 'additionally', 'therefore',
            'consequently', 'nevertheless', 'nonetheless', 'meanwhile', 'alternatively',
            'specifically', 'particularly', 'notably', 'importantly', 'essentially',
            'first', 'second', 'third', 'finally', 'lastly', 'next',
            'although', 'whereas', 'while', 'since', 'because',
            'thus', 'hence', 'accordingly', 'indeed', 'certainly',
            'overall', 'ultimately', 'essentially', 'basically',
            'in addition', 'for example', 'for instance', 'in contrast',
            'on the other hand', 'as a result', 'in particular'
        }
        
        text_lower = text.lower()
        transition_count = 0
        for tw in transition_words:
            transition_count += len(re.findall(r'\b' + re.escape(tw) + r'\b', text_lower))
        
        # Normalize by number of sentences
        transition_density = transition_count / num_sentences
        transition_score = min(transition_density / 0.4, 1.0) * 10
        
        # --- 6. Punctuation diversity and usage ---
        punct_types_used = set()
        for ch in text:
            if ch in '.,;:!?-—()[]"\'':
                punct_types_used.add(ch)
        
        # Count specific punctuation
        comma_count = text.count(',')
        colon_count = text.count(':')
        semicolon_count = text.count(';')
        
        # Comma rate per sentence (good: 1-3 per sentence)
        comma_rate = comma_count / num_sentences
        comma_quality = max(0, 10 - abs(comma_rate - 1.5) * 3)
        
        punct_diversity = min(len(punct_types_used) / 6, 1.0) * 5
        punct_score = (comma_quality + punct_diversity) / 1.5
        
        # --- 7. Formatting and structure ---
        has_headers = bool(re.search(r'#{1,3}\s+\S|^\*\*[^*]+\*\*', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*•]\s|\d+\.\s', text, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', text))
        has_paragraphs = '\n\n' in text or '\n' in text
        
        structure_score = 0
        if has_headers:
            structure_score += 2.5
        if has_lists:
            structure_score += 2.5
        if has_bold:
            structure_score += 2.0
        if has_paragraphs:
            structure_score += 1.5
        # Bonus for longer, well-structured responses
        if num_words > 50:
            structure_score += 1.0
        if num_words > 100:
            structure_score += 0.5
        structure_score = min(structure_score, 10)
        
        # --- 8. Sentence opener variety ---
        openers = []
        for s in sentences:
            s_words = re.findall(r"[a-zA-Z']+", s)
            if s_words:
                openers.append(s_words[0].lower())
        
        if len(openers) >= 2:
            unique_openers = len(set(openers))
            opener_ratio = unique_openers / len(openers)
            opener_score = opener_ratio * 10
        else:
            opener_score = 5.0
        
        # --- 9. Average word length (proxy for sophistication) ---
        avg_word_len = sum(len(w) for w in words) / num_words
        # Ideal: 4.5-6.0
        word_len_score = max(0, 10 - abs(avg_word_len - 5.2) * 3)
        
        # --- 10. Completeness penalty ---
        # Check if response seems cut off
        completeness_penalty = 0
        last_chars = text[-5:].strip() if len(text) >= 5 else text
        if not last_chars[-1:] in '.!?:)"\']' and not text.rstrip().endswith('```'):
            completeness_penalty = 3
        
        # --- Combine scores with weights ---
        # Different weights emphasize different aspects
        final_score = (
            cli_score * 0.10 +          # Coleman-Liau readability
            fog_score * 0.10 +           # Gunning Fog readability  
            vocab_score * 0.15 +         # Vocabulary richness
            variety_score * 0.12 +       # Sentence variety
            transition_score * 0.12 +    # Discourse coherence
            punct_score * 0.08 +         # Punctuation quality
            structure_score * 0.15 +     # Formatting/structure
            opener_score * 0.08 +        # Sentence opener variety
            word_len_score * 0.10        # Word sophistication
        )
        
        final_score -= completeness_penalty
        
        # Normalize to 0-100 range
        final_score = max(0, min(100, final_score * 10))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Fallback: simple length + basic quality
            return min(len(response.split()) / 5, 50)
        except Exception:
            return 0.0