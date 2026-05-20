def judging_function(query, response):
    """
    Evaluates clarity and conciseness using:
    - Information density (content words vs total words ratio)
    - Sentence structure analysis (avg sentence length, variance)
    - Redundancy detection via sentence-level semantic similarity (word overlap)
    - Filler/weasel word penalization
    - Directness score (how quickly the response gets to the point)
    - Compression ratio as a proxy for information density
    - Specificity scoring (concrete details, numbers, proper nouns)
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            return 0.0
        
        # Tokenize into words
        words = re.findall(r'[a-zA-Z]+', response.lower())
        if len(words) < 3:
            return 1.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        if not sentences:
            sentences = [response_stripped]
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # FEATURE 1: Information Density
        # Ratio of content words (not function words) to total words
        # ============================================================
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
            'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'about', 'up',
            'there', 'here', 'also', 'much', 'many', 'well', 'back', 'even',
            'still', 'already', 'since', 'while', 'although', 'though',
            'however', 'whether', 'unless', 'until', 'upon', 'within', 'without',
            'along', 'among', 'around', 'against', 'down', 'like', 'near',
            'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'going',
            'gone', 'went', 'come', 'came', 'take', 'took', 'taken',
        }
        
        content_words = [w for w in words if w not in function_words and len(w) > 2]
        info_density = len(content_words) / max(num_words, 1)
        # Ideal range: 0.4-0.6; penalize extremes
        info_density_score = 1.0 - abs(info_density - 0.50) * 2.0
        info_density_score = max(0.0, min(1.0, info_density_score))
        
        # ============================================================
        # FEATURE 2: Sentence Length Consistency & Appropriateness
        # Good writing has moderate sentence lengths with some variation
        # ============================================================
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r'[a-zA-Z]+', s)
            if sw:
                sent_word_counts.append(len(sw))
        
        if sent_word_counts:
            avg_sent_len = sum(sent_word_counts) / len(sent_word_counts)
            # Ideal average sentence length: 12-22 words
            if avg_sent_len < 5:
                sent_len_score = 0.3
            elif avg_sent_len < 12:
                sent_len_score = 0.3 + 0.7 * (avg_sent_len - 5) / 7
            elif avg_sent_len <= 25:
                sent_len_score = 1.0
            elif avg_sent_len <= 40:
                sent_len_score = 1.0 - 0.5 * (avg_sent_len - 25) / 15
            else:
                sent_len_score = 0.3
            
            # Variance in sentence length (some variation is good, too much is bad)
            if len(sent_word_counts) > 1:
                mean_sl = sum(sent_word_counts) / len(sent_word_counts)
                variance = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
                std_dev = math.sqrt(variance)
                cv = std_dev / max(mean_sl, 1)
                # Coefficient of variation: ideal ~0.3-0.5
                if cv < 0.1:
                    var_score = 0.6  # too uniform
                elif cv < 0.5:
                    var_score = 0.8 + 0.2 * (cv - 0.1) / 0.4
                elif cv < 0.8:
                    var_score = 1.0 - 0.3 * (cv - 0.5) / 0.3
                else:
                    var_score = 0.5
            else:
                var_score = 0.7
        else:
            sent_len_score = 0.5
            var_score = 0.5
        
        # ============================================================
        # FEATURE 3: Redundancy Detection
        # Pairwise sentence similarity using Jaccard on word sets
        # ============================================================
        sentence_word_sets = []
        for s in sentences:
            sw = set(re.findall(r'[a-zA-Z]+', s.lower()))
            sw -= function_words
            if sw:
                sentence_word_sets.append(sw)
        
        redundancy_penalties = 0.0
        pair_count = 0
        if len(sentence_word_sets) > 1:
            for i in range(len(sentence_word_sets)):
                for j in range(i + 1, min(len(sentence_word_sets), i + 4)):  # check nearby sentences
                    intersection = len(sentence_word_sets[i] & sentence_word_sets[j])
                    union = len(sentence_word_sets[i] | sentence_word_sets[j])
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.5:
                            redundancy_penalties += (jaccard - 0.5) * 2
                    pair_count += 1
        
        if pair_count > 0:
            avg_redundancy = redundancy_penalties / pair_count
        else:
            avg_redundancy = 0
        redundancy_score = max(0.0, 1.0 - avg_redundancy * 3)
        
        # ============================================================
        # FEATURE 4: Filler / Weasel / Vague Language Penalization
        # ============================================================
        filler_phrases = [
            r'\bit is worth noting that\b', r'\bit should be noted that\b',
            r'\bin order to\b', r'\bdue to the fact that\b',
            r'\bat the end of the day\b', r'\ball things considered\b',
            r'\bneedless to say\b', r'\bas a matter of fact\b',
            r'\bfor what it\'s worth\b', r'\bto be honest\b',
            r'\bbasically\b', r'\bactually\b', r'\bliterally\b',
            r'\bobviously\b', r'\bclearly\b', r'\bsimply put\b',
            r'\bin my opinion\b', r'\bi think that\b',
            r'\bit goes without saying\b', r'\bwith that being said\b',
            r'\bhaving said that\b', r'\bthat being said\b',
            r'\bin terms of\b', r'\bwith respect to\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bto some extent\b', r'\bin a sense\b',
            r'\bquite\b', r'\brather\b', r'\bsomewhat\b',
            r'\bperhaps\b', r'\bpossibly\b', r'\bpresumably\b',
            r'\bsupposedly\b', r'\ballegedly\b',
        ]
        
        response_lower = response.lower()
        filler_count = 0
        for pattern in filler_phrases:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_ratio = filler_count / max(num_sentences, 1)
        filler_score = max(0.0, 1.0 - filler_ratio * 0.4)
        
        # ============================================================
        # FEATURE 5: Directness / Gets to the Point
        # Check if first sentence is substantive (not just pleasantries)
        # ============================================================
        pleasantry_patterns = [
            r'^(great|good|excellent|wonderful|nice|interesting)\s+(question|point|thought)',
            r'^(sure|of course|absolutely|certainly|definitely)',
            r'^(well|so|okay|ok|alright)\s*[,.]',
            r'^(thank you|thanks)\s+(for|so much)',
            r'^(hello|hi|hey|greetings)',
            r'^(i\'d be happy to|i\'m happy to|i would be happy to)',
            r'^(that\'s a great|that\'s an excellent|that\'s a good)',
            r'^(let me|allow me)',
        ]
        
        first_sent = sentences[0].strip().lower() if sentences else ""
        directness_penalty = 0
        for pattern in pleasantry_patterns:
            if re.search(pattern, first_sent):
                directness_penalty += 0.15
        
        directness_score = max(0.0, 1.0 - directness_penalty)
        
        # ============================================================
        # FEATURE 6: Specificity Score
        # Concrete details: numbers, proper nouns, technical terms, examples
        # ============================================================
        # Count numbers/quantities
        numbers = re.findall(r'\b\d+[\d,.]*\b', response)
        number_density = min(len(numbers) / max(num_sentences, 1), 1.0)
        
        # Count capitalized words (potential proper nouns, excluding sentence starts)
        all_words_raw = re.findall(r'\b[A-Za-z]+\b', response)
        proper_nouns = 0
        for i, w in enumerate(all_words_raw):
            if i > 0 and w[0].isupper() and w.lower() not in function_words:
                # Check it's not after a sentence-ending punctuation
                proper_nouns += 1
        
        proper_noun_density = min(proper_nouns / max(num_words, 1) * 10, 1.0)
        
        # Example indicators
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b',
        ]
        example_count = sum(len(re.findall(p, response_lower)) for p in example_patterns)
        example_density = min(example_count / max(num_sentences, 1), 1.0)
        
        specificity_score = 0.3 + 0.3 * number_density + 0.2 * proper_noun_density + 0.2 * example_density
        specificity_score = min(1.0, specificity_score)
        
        # ============================================================
        # FEATURE 7: Compression Ratio (unique chars / total chars)
        # Proxy for information density at character level
        # ============================================================
        chars = response_stripped.lower()
        if len(chars) > 0:
            char_counter = Counter(chars)
            # Shannon entropy
            entropy = 0.0
            for count in char_counter.values():
                p = count / len(chars)
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalize: English text typically ~4.0-4.5 bits per char
            # Higher entropy = more diverse character usage = potentially more information
            entropy_score = min(entropy / 4.5, 1.0)
        else:
            entropy_score = 0.5
        
        # ============================================================
        # FEATURE 8: Response Length Appropriateness
        # Not too short (lacks substance), not too long (bloated)
        # Relative to query complexity
        # ============================================================
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_len = len(query_words)
        
        # Estimate query complexity
        query_complexity = min(query_len / 50.0, 1.0)  # 0 to 1
        
        # Ideal response length scales with query complexity
        ideal_min = 20 + query_complexity * 40   # 20-60 words min
        ideal_max = 100 + query_complexity * 200  # 100-300 words max
        
        if num_words < ideal_min:
            length_score = 0.3 + 0.7 * (num_words / max(ideal_min, 1))
            length_score = min(1.0, max(0.0, length_score))
        elif num_words <= ideal_max:
            length_score = 1.0
        else:
            overshoot = (num_words - ideal_max) / max(ideal_max, 1)
            length_score = max(0.3, 1.0 - overshoot * 0.5)
        
        # ============================================================
        # FEATURE 9: Unique Word Ratio (type-token at content level)
        # Measures vocabulary richness without being TTR on all words
        # ============================================================
        if content_words:
            unique_content = len(set(content_words))
            ttr_content = unique_content / len(content_words)
            # Adjust for length (longer texts naturally have lower TTR)
            adjusted_ttr = ttr_content * math.log(len(content_words) + 1) / math.log(50)
            vocab_score = min(1.0, max(0.0, adjusted_ttr))
        else:
            vocab_score = 0.3
        
        # ============================================================
        # FEATURE 10: Structural Organization
        # Check for lists, bullet points, code blocks, structured formatting
        # ============================================================
        has_list = bool(re.search(r'(?m)^[\s]*[-*•]\s', response) or re.search(r'(?m)^[\s]*\d+[.)]\s', response))
        has_code = bool(re.search(r'```', response))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response))
        has_headers = bool(re.search(r'(?m)^#+\s', response))
        
        structure_score = 0.5
        if has_list:
            structure_score += 0.15
        if has_code:
            structure_score += 0.1
        if has_bold:
            structure_score += 0.1
        if has_headers:
            structure_score += 0.1
        structure_score = min(1.0, structure_score)
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        weights = {
            'info_density': 1.5,
            'sent_len': 1.0,
            'variance': 0.5,
            'redundancy': 2.0,
            'filler': 1.5,
            'directness': 1.0,
            'specificity': 2.0,
            'entropy': 0.5,
            'length': 1.5,
            'vocab': 1.0,
            'structure': 0.5,
        }
        
        scores = {
            'info_density': info_density_score,
            'sent_len': sent_len_score,
            'variance': var_score,
            'redundancy': redundancy_score,
            'filler': filler_score,
            'directness': directness_score,
            'specificity': specificity_score,
            'entropy': entropy_score,
            'length': length_score,
            'vocab': vocab_score,
            'structure': structure_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        final_score = (weighted_sum / total_weight) * 10.0
        
        # Clamp to 0-10
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0