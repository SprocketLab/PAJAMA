def judging_function(query, response):
    """
    Evaluates clarity and conciseness using a structure-based approach:
    - Signal-to-noise ratio (content words vs filler/function words)
    - Response coherence via sentence-level topic drift detection
    - Repetition detection via sliding window character-level similarity
    - Formatting noise detection (HTML tags, code artifacts, random symbols)
    - Completion and substance checks
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.5
        
        resp = response.strip()
        query_clean = query.strip() if query else ""
        
        # === 1. Basic substance check ===
        # Very short responses: could be good (concise) or bad (empty/useless)
        word_tokens = resp.split()
        num_words = len(word_tokens)
        
        if num_words <= 2:
            # Check if it's a meaningful short answer
            # Very short non-answers get penalized
            non_answers = {'no', 'yes', '.', '..', '...', 'n/a', 'na', 'none'}
            if resp.lower().strip('.,!? ') in non_answers:
                return 1.0
            # Could be a valid concise answer - give moderate score
            return 4.0
        
        # === 2. Formatting noise ratio ===
        # Detect HTML tags, code blocks, markdown artifacts
        html_tags = re.findall(r'<[^>]+>', resp)
        html_char_count = sum(len(t) for t in html_tags)
        
        code_patterns = re.findall(r'(?:import |def |class |if |for |while |return |print\()', resp)
        code_noise = len(code_patterns)
        
        # Random repeated punctuation or symbols
        symbol_noise = len(re.findall(r'[#*=_]{3,}', resp))
        
        total_chars = len(resp)
        noise_ratio = (html_char_count + code_noise * 10 + symbol_noise * 5) / max(total_chars, 1)
        noise_penalty = max(0, min(3.0, noise_ratio * 15))
        
        # === 3. Repetition detection using sliding window of sentences ===
        sentences = re.split(r'[.!?\n]+', resp)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        def normalize_sent(s):
            return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
        
        repetition_score = 0.0
        if num_sentences >= 2:
            norm_sents = [normalize_sent(s) for s in sentences]
            
            # Compare each pair of sentences using character trigram overlap
            pair_similarities = []
            for i in range(len(norm_sents)):
                for j in range(i + 1, min(i + 5, len(norm_sents))):
                    s1, s2 = norm_sents[i], norm_sents[j]
                    if not s1 or not s2:
                        continue
                    # Character trigram Jaccard
                    tri1 = set(s1[k:k+3] for k in range(len(s1)-2))
                    tri2 = set(s2[k:k+3] for k in range(len(s2)-2))
                    if tri1 and tri2:
                        jaccard = len(tri1 & tri2) / len(tri1 | tri2)
                        pair_similarities.append(jaccard)
            
            if pair_similarities:
                avg_sim = sum(pair_similarities) / len(pair_similarities)
                high_sim_count = sum(1 for s in pair_similarities if s > 0.6)
                repetition_score = avg_sim * 2.5 + (high_sim_count / max(len(pair_similarities), 1)) * 2.0
        
        # Also check for exact duplicate lines
        line_counts = Counter(normalize_sent(s) for s in sentences if normalize_sent(s))
        duplicate_lines = sum(c - 1 for c in line_counts.values() if c > 1)
        repetition_score += min(3.0, duplicate_lines * 0.8)
        
        repetition_penalty = min(4.0, repetition_score)
        
        # === 4. Signal-to-noise word ratio ===
        # Instead of just function words, measure "content density"
        filler_words = {
            'very', 'really', 'actually', 'basically', 'essentially', 'literally',
            'just', 'quite', 'rather', 'somewhat', 'perhaps', 'maybe', 'possibly',
            'kind', 'sort', 'like', 'stuff', 'things', 'thing', 'etc', 'well',
            'anyway', 'anyways', 'obviously', 'clearly', 'definitely', 'certainly',
            'honestly', 'frankly', 'simply', 'merely', 'virtually'
        }
        
        hedge_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'as you may know', 'as we all know', 'needless to say',
            'in other words', 'that is to say', 'to put it another way',
            'at the end of the day', 'when all is said and done',
            'it goes without saying', 'for what it is worth'
        ]
        
        resp_lower = resp.lower()
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in resp_lower)
        
        words_lower = [w.lower().strip('.,!?;:"\'-()[]{}') for w in word_tokens]
        words_lower = [w for w in words_lower if w]
        
        filler_count = sum(1 for w in words_lower if w in filler_words)
        filler_ratio = filler_count / max(len(words_lower), 1)
        
        filler_penalty = filler_ratio * 3.0 + hedge_count * 0.5
        filler_penalty = min(2.5, filler_penalty)
        
        # === 5. Sentence structure variety and clarity ===
        # Good writing has varied but not extreme sentence lengths
        sent_lengths = [len(s.split()) for s in sentences if s]
        
        structure_score = 0.0
        if sent_lengths:
            avg_len = sum(sent_lengths) / len(sent_lengths)
            
            # Ideal average sentence length: 10-20 words
            if 8 <= avg_len <= 22:
                structure_score += 1.0
            elif avg_len < 5:
                structure_score -= 0.5
            elif avg_len > 35:
                structure_score -= 1.0
            
            # Variance in sentence length (some variety is good)
            if len(sent_lengths) > 1:
                variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / max(avg_len, 1)
                # Moderate variation is good (cv 0.2-0.6)
                if 0.15 <= cv <= 0.7:
                    structure_score += 0.5
        
        # === 6. Relevance to query ===
        query_words = set(re.sub(r'[^a-z0-9 ]', '', query_clean.lower()).split())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
            'or', 'if', 'while', 'about', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
            'it', 'its', 'they', 'them', 'their', 'he', 'she', 'him', 'her'
        }
        
        query_content = query_words - stop_words
        resp_words_set = set(re.sub(r'[^a-z0-9 ]', '', resp_lower).split())
        resp_content = resp_words_set - stop_words
        
        if query_content:
            relevance = len(query_content & resp_content) / len(query_content)
        else:
            relevance = 0.5
        
        relevance_bonus = relevance * 1.5
        
        # === 7. Off-topic / derailment detection ===
        # Check if response goes off on tangents (e.g., starts answering then adds unrelated content)
        # Split response into halves and check if second half diverges
        derailment_penalty = 0.0
        if num_sentences >= 4:
            mid = len(sentences) // 2
            first_half_words = set()
            second_half_words = set()
            for s in sentences[:mid]:
                first_half_words.update(re.sub(r'[^a-z0-9 ]', '', s.lower()).split())
            for s in sentences[mid:]:
                second_half_words.update(re.sub(r'[^a-z0-9 ]', '', s.lower()).split())
            
            first_content = first_half_words - stop_words
            second_content = second_half_words - stop_words
            
            if first_content and second_content:
                coherence = len(first_content & second_content) / max(len(first_content | second_content), 1)
                if coherence < 0.1:
                    derailment_penalty = 1.5
                elif coherence < 0.2:
                    derailment_penalty = 0.8
        
        # === 8. Prompt echo detection ===
        # Penalize if response just repeats the query
        echo_penalty = 0.0
        if query_content and resp_content:
            query_norm = re.sub(r'[^a-z0-9 ]', '', query_clean.lower()).strip()
            resp_norm = re.sub(r'[^a-z0-9 ]', '', resp_lower).strip()
            if query_norm and resp_norm.startswith(query_norm):
                echo_penalty = 1.5
        
        # === 9. Completion check ===
        # Does the response appear truncated or complete?
        completion_penalty = 0.0
        last_char = resp.rstrip()[-1] if resp.rstrip() else ''
        if last_char not in '.!?)"\':;' and num_words > 20:
            # Likely truncated - mild penalty since it might still be clear
            completion_penalty = 0.3
        
        # === 10. Bloat detection ===
        # Check if response is much longer than needed relative to query complexity
        query_words_count = len(query_clean.split())
        bloat_penalty = 0.0
        
        # For simple queries, very long responses suggest bloat
        if query_words_count < 15 and num_words > 200:
            bloat_penalty = min(1.5, (num_words - 200) / 200)
        
        # Check for "Output:" prefix repetition (seen in examples)
        output_prefix_count = resp.count('Output:')
        if output_prefix_count > 3:
            repetition_penalty += min(1.0, (output_prefix_count - 3) * 0.3)
        
        # Check for "Question:" / "Answer:" pattern spam
        qa_pattern_count = len(re.findall(r'(?:Question|Answer|Input|Output)\s*:', resp))
        if qa_pattern_count > 4:
            noise_penalty += min(2.0, (qa_pattern_count - 4) * 0.4)
        
        # === Combine scores ===
        base_score = 7.0  # Start with a decent base
        
        # Bonuses
        score = base_score
        score += relevance_bonus          # up to +1.5
        score += structure_score           # up to +1.5
        
        # Penalties
        score -= noise_penalty             # up to -3.0
        score -= repetition_penalty        # up to -4.0
        score -= filler_penalty            # up to -2.5
        score -= derailment_penalty        # up to -1.5
        score -= echo_penalty              # up to -1.5
        score -= completion_penalty        # up to -0.3
        score -= bloat_penalty             # up to -1.5
        
        # Bonus for very concise, on-topic responses (sweet spot: 5-80 words)
        if 3 <= num_words <= 80 and repetition_penalty < 0.5 and noise_penalty < 0.5:
            conciseness_bonus = 0.8
            score += conciseness_bonus
        
        # Penalty for extremely short non-substantive responses
        if num_words < 5 and relevance < 0.3:
            score -= 3.0
        
        # Clamp to range
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 4.0