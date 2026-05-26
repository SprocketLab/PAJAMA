def judging_function(query, response):
    """
    Evaluate language quality and readability using a unique approach focused on:
    - Punctuation variety and correctness
    - Discourse markers and transitional phrases
    - Sentence structure variety (measured by length distribution entropy)
    - Paragraph structure and formatting
    - Lexical sophistication (longer words ratio, not just type-token)
    - Cohesion indicators
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
        
        # ============================================================
        # 1. PUNCTUATION VARIETY AND DENSITY (unique focus)
        # ============================================================
        punctuation_marks = {'.', ',', ';', ':', '!', '?', '-', '(', ')', '"', "'", '—', '–'}
        punct_counts = Counter()
        for ch in text:
            if ch in punctuation_marks:
                punct_counts[ch] += 1
        
        total_punct = sum(punct_counts.values())
        punct_variety = len(punct_counts)  # how many different punctuation types used
        
        # Ideal punctuation density: roughly 1 punct per 8-15 chars
        punct_density = total_punct / max(len(text), 1)
        # Sweet spot around 0.06-0.10
        if punct_density < 0.02:
            punct_density_score = punct_density / 0.02 * 3
        elif punct_density <= 0.12:
            punct_density_score = 5 + (min(punct_density, 0.08) / 0.08) * 5
        else:
            punct_density_score = max(3, 10 - (punct_density - 0.12) * 50)
        
        # Variety score: using more types of punctuation is better
        punct_variety_score = min(10, punct_variety * 1.5)
        
        punctuation_score = punct_density_score * 0.4 + punct_variety_score * 0.6
        
        # ============================================================
        # 2. DISCOURSE MARKERS AND TRANSITIONS (unique focus)
        # ============================================================
        discourse_markers = [
            r'\bhowever\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bconsequently\b', r'\btherefore\b', r'\bnevertheless\b', r'\bmeanwhile\b',
            r'\bin contrast\b', r'\bon the other hand\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bas a result\b', r'\bin fact\b',
            r'\bnotably\b', r'\bimportantly\b', r'\badditionally\b', r'\balternatively\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b', r'\blastly\b',
            r'\bnext\b', r'\bthen\b', r'\boverall\b', r'\bin summary\b', r'\bto summarize\b',
            r'\bin conclusion\b', r'\bthat said\b', r'\bwhile\b', r'\balthough\b',
            r'\bdespite\b', r'\bregardless\b', r'\bsimilarly\b', r'\blikewise\b',
            r'\bhere\'s\b', r'\bhere are\b', r'\blet\'s\b', r'\bthis means\b',
            r'\bas such\b', r'\bthat is\b', r'\bin other words\b'
        ]
        
        text_lower = text.lower()
        marker_count = 0
        unique_markers = 0
        for pattern in discourse_markers:
            matches = re.findall(pattern, text_lower)
            if matches:
                unique_markers += 1
                marker_count += len(matches)
        
        # Sentences for normalization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # Discourse density: markers per sentence
        discourse_density = marker_count / num_sentences
        # Ideal: 0.15 - 0.5 markers per sentence
        if discourse_density < 0.05:
            discourse_score = 2
        elif discourse_density < 0.15:
            discourse_score = 2 + (discourse_density - 0.05) / 0.10 * 3
        elif discourse_density <= 0.6:
            discourse_score = 5 + min(5, unique_markers * 0.8)
        else:
            discourse_score = max(5, 10 - (discourse_density - 0.6) * 5)
        
        discourse_score = min(10, discourse_score)
        
        # ============================================================
        # 3. SENTENCE LENGTH DISTRIBUTION ENTROPY (unique metric)
        # ============================================================
        words_in_text = text.split()
        total_words = len(words_in_text)
        
        if num_sentences >= 2:
            sent_lengths = []
            for s in sentences:
                wc = len(s.split())
                if wc > 0:
                    sent_lengths.append(wc)
            
            if len(sent_lengths) >= 2:
                # Bin sentence lengths into categories
                bins = Counter()
                for sl in sent_lengths:
                    if sl <= 5:
                        bins['very_short'] += 1
                    elif sl <= 12:
                        bins['short'] += 1
                    elif sl <= 20:
                        bins['medium'] += 1
                    elif sl <= 30:
                        bins['long'] += 1
                    else:
                        bins['very_long'] += 1
                
                # Calculate entropy of sentence length distribution
                total_sents = sum(bins.values())
                entropy = 0
                for count in bins.values():
                    if count > 0:
                        p = count / total_sents
                        entropy -= p * math.log2(p)
                
                max_entropy = math.log2(min(len(bins), 5))
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = 0
                
                # Also compute coefficient of variation of sentence lengths
                mean_sl = sum(sent_lengths) / len(sent_lengths)
                variance = sum((x - mean_sl) ** 2 for x in sent_lengths) / len(sent_lengths)
                std_sl = math.sqrt(variance)
                cv = std_sl / max(mean_sl, 1)
                
                # Good variety: entropy > 0.5, CV between 0.3 and 0.8
                entropy_score = normalized_entropy * 7
                cv_score = 0
                if 0.2 <= cv <= 0.9:
                    cv_score = 5
                elif cv < 0.2:
                    cv_score = cv / 0.2 * 3
                else:
                    cv_score = max(2, 5 - (cv - 0.9) * 3)
                
                sentence_variety_score = entropy_score * 0.6 + cv_score * 0.4
            else:
                sentence_variety_score = 3
        else:
            sentence_variety_score = 2
        
        sentence_variety_score = min(10, sentence_variety_score)
        
        # ============================================================
        # 4. FORMATTING AND STRUCTURE QUALITY (unique focus)
        # ============================================================
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Check for structured elements
        has_headers = bool(re.search(r'#{1,4}\s+\S|^\*\*[^*]+\*\*$', text, re.MULTILINE))
        has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s+\S', text, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^\s*[-*•]\s+\S', text, re.MULTILINE))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', text))
        
        # Paragraph count (blocks separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        structure_elements = sum([has_headers, has_numbered_list, has_bullet_list, has_bold])
        
        # Structure score: reward organized content
        if total_words < 30:
            structure_score = 5  # neutral for short responses
        else:
            base_struct = 3
            base_struct += structure_elements * 1.2
            # Multiple paragraphs are good for longer text
            if num_paragraphs >= 2:
                base_struct += min(2, (num_paragraphs - 1) * 0.5)
            structure_score = min(10, base_struct)
        
        # ============================================================
        # 5. LEXICAL SOPHISTICATION (different from type-token ratio)
        # ============================================================
        # Focus on proportion of "sophisticated" words (3+ syllables, or 7+ chars)
        # and hapax legomena ratio (words appearing exactly once)
        
        word_tokens = re.findall(r'[a-zA-Z]+', text_lower)
        num_word_tokens = len(word_tokens) if word_tokens else 1
        
        # Sophisticated words: 7+ characters
        sophisticated = [w for w in word_tokens if len(w) >= 7]
        sophistication_ratio = len(sophisticated) / num_word_tokens
        
        # Hapax legomena ratio (words appearing exactly once / total unique words)
        word_freq = Counter(word_tokens)
        unique_words = len(word_freq)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        hapax_ratio = hapax / max(unique_words, 1)
        
        # Yule's K measure of vocabulary richness (different from TTR)
        # K = 10^4 * (M2 - N) / N^2 where M2 = sum(i^2 * Vi) and Vi = number of words with freq i
        N = num_word_tokens
        freq_spectrum = Counter(word_freq.values())
        M2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        if N > 1:
            yules_k = 10000 * (M2 - N) / (N * N)
        else:
            yules_k = 0
        
        # Lower Yule's K = richer vocabulary
        # Typical range: 20-200 for normal text
        if yules_k < 50:
            vocab_richness_score = 9
        elif yules_k < 100:
            vocab_richness_score = 7
        elif yules_k < 150:
            vocab_richness_score = 5
        elif yules_k < 250:
            vocab_richness_score = 3
        else:
            vocab_richness_score = 2
        
        # Sophistication ratio: ideal around 0.15-0.30
        if sophistication_ratio < 0.05:
            soph_score = 2
        elif sophistication_ratio < 0.15:
            soph_score = 2 + (sophistication_ratio - 0.05) / 0.10 * 4
        elif sophistication_ratio <= 0.35:
            soph_score = 6 + (sophistication_ratio - 0.15) / 0.20 * 4
        else:
            soph_score = max(5, 10 - (sophistication_ratio - 0.35) * 10)
        
        lexical_score = vocab_richness_score * 0.4 + soph_score * 0.4 + (hapax_ratio * 8) * 0.2
        lexical_score = min(10, lexical_score)
        
        # ============================================================
        # 6. COHESION: Pronoun and reference usage (unique focus)
        # ============================================================
        pronouns = re.findall(r'\b(this|that|these|those|it|they|them|its|their|which|who|whom)\b', text_lower)
        pronoun_density = len(pronouns) / max(num_sentences, 1)
        
        # Sentence-initial variety (don't start every sentence with the same word)
        if num_sentences >= 3:
            starters = []
            for s in sentences:
                words = s.strip().split()
                if words:
                    starters.append(words[0].lower().strip('*#'))
            
            starter_freq = Counter(starters)
            if starters:
                most_common_starter_ratio = max(starter_freq.values()) / len(starters)
                # Lower ratio = more variety in sentence starters
                starter_variety_score = (1 - most_common_starter_ratio) * 10
            else:
                starter_variety_score = 5
        else:
            starter_variety_score = 5
        
        # Pronoun density: some is good (shows cohesion), too much is bad
        if pronoun_density < 0.3:
            pronoun_score = 3 + pronoun_density / 0.3 * 3
        elif pronoun_density <= 2.0:
            pronoun_score = 6 + min(4, (pronoun_density - 0.3) / 1.7 * 4)
        else:
            pronoun_score = max(4, 10 - (pronoun_density - 2.0) * 2)
        
        cohesion_score = starter_variety_score * 0.5 + pronoun_score * 0.5
        cohesion_score = min(10, cohesion_score)
        
        # ============================================================
        # 7. GRAMMAR HEURISTICS (unique approach: error pattern detection)
        # ============================================================
        grammar_penalty = 0
        
        # Double spaces (minor)
        double_spaces = len(re.findall(r'  +', text))
        grammar_penalty += min(1, double_spaces * 0.1)
        
        # Missing capitalization after sentence-ending punctuation
        missing_caps = len(re.findall(r'[.!?]\s+[a-z]', text))
        grammar_penalty += min(2, missing_caps * 0.3)
        
        # Repeated words (e.g., "the the")
        repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', text_lower))
        grammar_penalty += min(1.5, repeated_words * 0.5)
        
        # Sentences not ending with punctuation (last char of each sentence block)
        if non_empty_lines:
            last_line = non_empty_lines[-1].strip()
            if last_line and last_line[-1] not in '.!?:;,)"\'>*':
                grammar_penalty += 0.3
        
        # Very long sentences without commas (run-on detection)
        for s in sentences:
            wc = len(s.split())
            commas = s.count(',')
            if wc > 25 and commas == 0:
                grammar_penalty += 0.5
        
        grammar_score = max(0, 10 - grammar_penalty)
        
        # ============================================================
        # 8. OPENING QUALITY (unique: how well the response begins)
        # ============================================================
        first_sentence = sentences[0] if sentences else ""
        first_words = first_sentence.split()
        
        opening_score = 5  # baseline
        
        # Engaging openings
        engaging_patterns = [
            r'^(certainly|absolutely|great|excellent|that\'s|here|let)',
            r'^(the art|a classic|organizing|there)',
            r'^(no,|yes,)',  # direct answers
        ]
        for pat in engaging_patterns:
            if re.match(pat, text_lower.strip()):
                opening_score += 1
                break
        
        # Penalize very abrupt/short openings
        if len(first_words) < 3:
            opening_score -= 1
        elif len(first_words) > 8:
            opening_score += 1
        
        # Direct address to query
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower())) if query else set()
        response_first_sent_words = set(re.findall(r'[a-zA-Z]+', first_sentence.lower()))
        overlap = len(query_words & response_first_sent_words)
        if overlap >= 2:
            opening_score += 1
        
        opening_score = min(10, max(0, opening_score))
        
        # ============================================================
        # FINAL WEIGHTED COMBINATION
        # ============================================================
        final_score = (
            punctuation_score * 0.10 +
            discourse_score * 0.15 +
            sentence_variety_score * 0.15 +
            structure_score * 0.15 +
            lexical_score * 0.15 +
            cohesion_score * 0.10 +
            grammar_score * 0.10 +
            opening_score * 0.10
        )
        
        # Scale to 0-10 range
        final_score = max(0, min(10, final_score))
        
        return round(final_score, 4)
    
    except Exception:
        return 3.0