def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique approach focused on:
    - Discourse markers and cohesion signals
    - Sentence-level complexity variation (coefficient of variation)
    - Punctuation sophistication
    - Lexical formality estimation
    - Bigram/trigram repetition penalty
    - Clause density estimation
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+", text)
        if len(words) < 3:
            return 0.5
        
        # Tokenize sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if not sentences:
            sentences = [text]
        
        word_count = len(words)
        words_lower = [w.lower() for w in words]
        
        # ============================================================
        # 1. DISCOURSE MARKERS & COHESION (unique feature)
        # ============================================================
        # Discourse markers signal well-structured, cohesive writing
        discourse_markers = {
            'however', 'moreover', 'furthermore', 'additionally', 'nevertheless',
            'consequently', 'therefore', 'meanwhile', 'nonetheless', 'similarly',
            'likewise', 'conversely', 'specifically', 'notably', 'importantly',
            'indeed', 'certainly', 'ultimately', 'essentially', 'particularly',
            'accordingly', 'hence', 'thus', 'alternatively', 'subsequently',
            'incidentally', 'admittedly', 'granted', 'undoubtedly'
        }
        
        # Connective phrases (check bigrams)
        connective_starters = {
            'in addition', 'on the other hand', 'as a result', 'for example',
            'for instance', 'in contrast', 'in fact', 'of course', 'at the same time',
            'in other words', 'that said', 'to summarize', 'in summary',
            'first of all', 'on top of that', 'having said that', 'with that',
            'to begin with', 'to put it', 'it is', "it's important", "it's crucial",
            'keep in mind', 'bear in mind', 'let me', "let's"
        }
        
        text_lower = text.lower()
        
        discourse_count = sum(1 for w in words_lower if w in discourse_markers)
        connective_count = sum(1 for phrase in connective_starters if phrase in text_lower)
        
        cohesion_density = (discourse_count + connective_count) / max(word_count, 1) * 100
        # Cap at a reasonable level
        cohesion_score = min(cohesion_density * 8, 10.0)
        
        # ============================================================
        # 2. SENTENCE LENGTH VARIATION (coefficient of variation)
        # ============================================================
        sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
        sent_lengths = [sl for sl in sent_lengths if sl > 0]
        
        if len(sent_lengths) >= 2:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance_sl)
            cv = std_sl / max(mean_sl, 1)
            
            # Ideal CV is around 0.3-0.6 (some variety but not chaotic)
            if cv < 0.1:
                variation_score = 3.0  # Too uniform / monotonous
            elif cv < 0.25:
                variation_score = 6.0
            elif cv < 0.55:
                variation_score = 9.0  # Good variety
            elif cv < 0.8:
                variation_score = 7.0
            else:
                variation_score = 4.0  # Too erratic
            
            # Penalize very short average sentence length or very long
            if mean_sl < 6:
                variation_score *= 0.7
            elif mean_sl > 35:
                variation_score *= 0.8
            elif 12 <= mean_sl <= 25:
                variation_score *= 1.05
        else:
            # Single sentence
            variation_score = 4.0
        
        variation_score = min(variation_score, 10.0)
        
        # ============================================================
        # 3. PUNCTUATION SOPHISTICATION (unique angle)
        # ============================================================
        # Count different punctuation types used
        punct_types = {
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'dash': text.count('—') + text.count('–') + text.count(' - '),
            'question': text.count('?'),
            'exclamation': text.count('!'),
            'parentheses': text.count('(') + text.count(')'),
            'quotes': text.count('"') + text.count("'") + text.count('"') + text.count('"'),
            'ellipsis': text.count('...') + text.count('…'),
        }
        
        # Variety of punctuation used
        punct_variety = sum(1 for v in punct_types.values() if v > 0)
        
        # Comma usage rate per sentence (good writing uses commas for clause separation)
        comma_rate = punct_types['comma'] / max(len(sent_lengths), 1)
        
        punct_score = min(punct_variety * 1.3, 7.0)
        if 0.5 <= comma_rate <= 3.0:
            punct_score += 2.0
        elif comma_rate > 0:
            punct_score += 1.0
        
        punct_score = min(punct_score, 10.0)
        
        # ============================================================
        # 4. LEXICAL FORMALITY & SOPHISTICATION
        # ============================================================
        # Estimate formality by looking at word characteristics
        # Longer words tend to be more formal/sophisticated in English
        
        char_lengths = [len(w) for w in words]
        avg_char_len = sum(char_lengths) / max(len(char_lengths), 1)
        
        # Words with 3+ syllables (rough estimate: every 3 chars ~ 1 syllable)
        complex_words = sum(1 for w in words if len(w) >= 8)
        complex_ratio = complex_words / max(word_count, 1)
        
        # Contractions indicate casual but natural writing
        contractions = sum(1 for w in words if "'" in w and len(w) > 2)
        contraction_rate = contractions / max(word_count, 1)
        
        # Unique trigrams (character-level) for vocabulary sophistication
        unique_words = set(words_lower)
        type_token = len(unique_words) / max(word_count, 1)
        
        # Hapax legomena ratio (words appearing exactly once)
        word_freq = Counter(words_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(len(unique_words), 1)
        
        formality_score = 0.0
        # Average word length contribution
        if 4.0 <= avg_char_len <= 6.0:
            formality_score += 4.0
        elif avg_char_len > 6.0:
            formality_score += 3.0
        elif avg_char_len >= 3.5:
            formality_score += 2.5
        else:
            formality_score += 1.0
        
        # Complex word contribution
        formality_score += min(complex_ratio * 30, 3.0)
        
        # Type-token ratio contribution
        if type_token > 0.7:
            formality_score += 2.5
        elif type_token > 0.5:
            formality_score += 2.0
        elif type_token > 0.3:
            formality_score += 1.0
        
        formality_score = min(formality_score, 10.0)
        
        # ============================================================
        # 5. BIGRAM REPETITION PENALTY (unique feature)
        # ============================================================
        # Excessive repetition of word pairs indicates poor writing
        if len(words_lower) >= 4:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            bigram_freq = Counter(bigrams)
            total_bigrams = len(bigrams)
            
            # Count repeated bigrams (appearing 3+ times)
            repeated_bigrams = sum(count - 2 for count in bigram_freq.values() if count > 2)
            repetition_ratio = repeated_bigrams / max(total_bigrams, 1)
            
            repetition_penalty = min(repetition_ratio * 50, 5.0)
        else:
            repetition_penalty = 0.0
        
        # ============================================================
        # 6. CLAUSE DENSITY (commas, conjunctions per sentence)
        # ============================================================
        subordinating_conj = {'because', 'although', 'though', 'while', 'whereas',
                              'since', 'unless', 'until', 'whenever', 'wherever',
                              'whether', 'if', 'when', 'after', 'before', 'once'}
        coordinating_conj = {'and', 'but', 'or', 'nor', 'yet', 'so'}
        relative_pronouns = {'which', 'that', 'who', 'whom', 'whose', 'where'}
        
        sub_count = sum(1 for w in words_lower if w in subordinating_conj)
        coord_count = sum(1 for w in words_lower if w in coordinating_conj)
        rel_count = sum(1 for w in words_lower if w in relative_pronouns)
        
        clause_markers = sub_count + coord_count + rel_count
        clause_density = clause_markers / max(len(sent_lengths), 1)
        
        # Ideal clause density: 1.5-3.5 per sentence
        if 1.5 <= clause_density <= 3.5:
            clause_score = 9.0
        elif 1.0 <= clause_density < 1.5:
            clause_score = 7.0
        elif 3.5 < clause_density <= 5.0:
            clause_score = 6.5
        elif 0.5 <= clause_density < 1.0:
            clause_score = 5.5
        else:
            clause_score = 4.0
        
        # ============================================================
        # 7. OPENING & CLOSING QUALITY
        # ============================================================
        # Good responses often have strong openings
        opening_quality = 5.0
        first_sent = sentences[0].lower() if sentences else ""
        
        # Empathetic/engaging openings
        empathetic_openers = ['i can', "i'm", 'i understand', 'i hear', 'i see',
                              'it sounds', "it's completely", "it's understandable",
                              'imagine', 'hey', 'absolutely', 'great question',
                              'thank you', 'welcome']
        if any(first_sent.startswith(op) for op in empathetic_openers):
            opening_quality = 8.0
        elif any(first_sent.startswith(op) for op in ['to ', 'the ', 'a ', 'an ']):
            opening_quality = 5.5
        
        # Structured responses (numbered lists, bullet points)
        has_structure = bool(re.search(r'(?:^|\n)\s*(?:\d+[\.\):]|[-•*])\s', text))
        if has_structure:
            opening_quality = min(opening_quality + 1.5, 10.0)
        
        # ============================================================
        # 8. NEGATIVE LANGUAGE / DISMISSIVENESS PENALTY
        # ============================================================
        dismissive_phrases = [
            'just do', 'just try', 'just get', 'just keep', 'just remember',
            'you should be able', 'you need to get', 'nothing wrong',
            'maybe you\'re just', 'it\'s just', "that's a bummer"
        ]
        dismissive_count = sum(1 for phrase in dismissive_phrases if phrase in text_lower)
        dismissive_penalty = min(dismissive_count * 1.5, 4.0)
        
        # ============================================================
        # AGGREGATE SCORE
        # ============================================================
        # Weighted combination
        raw_score = (
            cohesion_score * 0.15 +
            variation_score * 0.15 +
            punct_score * 0.10 +
            formality_score * 0.20 +
            clause_score * 0.12 +
            opening_quality * 0.15 +
            # Length adequacy bonus
            min(word_count / 30, 1.0) * 1.3  # 13% weight for adequate length
        )
        
        # Apply penalties
        raw_score -= repetition_penalty
        raw_score -= dismissive_penalty
        
        # Bonus for response length being substantial but not excessive
        if 50 <= word_count <= 300:
            raw_score += 0.5
        elif word_count > 300:
            raw_score += 0.3
        
        # Scale to 1-5 range to match examples
        # raw_score typically ranges from about 2 to 9
        final_score = max(1.0, min(5.0, raw_score * 0.55 + 0.5))
        
        # Round to 1 decimal
        final_score = round(final_score, 2)
        
        return final_score
        
    except Exception:
        return 2.5