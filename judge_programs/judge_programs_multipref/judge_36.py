def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectives)
    - Sentence-level coherence via topic continuity (entity/noun tracking across sentences)
    - Argument depth detection (nested reasoning, conditional logic)
    - Structural balance and proportion
    - Absence of contradiction signals
    
    Returns a score where higher = better quality.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.0
        
        query_clean = query.strip().lower()
        
        # ============================================================
        # 1. DISCOURSE MARKER DENSITY AND VARIETY
        # Logical coherence is signaled by discourse connectives
        # ============================================================
        
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bfor this reason\b', r'\bit follows\b', r'\baccordingly\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bas such\b', r'\bgiven that\b', r'\bin light of\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\byet\b', r'\bnonetheless\b',
            r'\bconversely\b', r'\binstead\b', r'\brather than\b',
            r'\bon the contrary\b', r'\beven though\b', r'\bthat said\b'
        ]
        
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\bequally\b', r'\blikewise\b', r'\bsimilarly\b',
            r'\bnot only\b', r'\bas well as\b', r'\bcoupled with\b',
            r'\balong with\b', r'\bwhat\'s more\b'
        ]
        
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\binitially\b', r'\bpreviously\b',
            r'\bmeanwhile\b', r'\bat this point\b', r'\bonce\b',
            r'\bbefore\b', r'\bafter\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b'
        ]
        
        summary_markers = [
            r'\bin summary\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto summarize\b', r'\bin short\b', r'\ball in all\b',
            r'\btaken together\b', r'\bto sum up\b', r'\bin essence\b',
            r'\bultimately\b', r'\bthe key point\b', r'\bto conclude\b'
        ]
        
        response_lower = response_clean.lower()
        
        def count_markers(patterns, text):
            total = 0
            unique = 0
            for p in patterns:
                matches = re.findall(p, text)
                if matches:
                    total += len(matches)
                    unique += 1
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers, response_lower)
        contrastive_count, contrastive_unique = count_markers(contrastive_markers, response_lower)
        additive_count, additive_unique = count_markers(additive_markers, response_lower)
        temporal_count, temporal_unique = count_markers(temporal_markers, response_lower)
        summary_count, summary_unique = count_markers(summary_markers, response_lower)
        
        total_markers = causal_count + contrastive_count + additive_count + temporal_count + summary_count
        total_unique = causal_unique + contrastive_unique + additive_unique + temporal_unique + summary_unique
        
        # Categories present (out of 5)
        categories_present = sum(1 for c in [causal_count, contrastive_count, additive_count, temporal_count, summary_count] if c > 0)
        
        words = response_lower.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # Marker density per 100 words
        marker_density = (total_markers / word_count) * 100
        
        # Score: reward density up to a point, reward variety
        # Optimal density around 3-8 per 100 words
        density_score = min(marker_density / 5.0, 2.0)  # max 2.0
        variety_score = min(total_unique / 8.0, 2.0)  # max 2.0
        category_score = categories_present * 0.4  # max 2.0
        
        discourse_score = density_score + variety_score + category_score  # max ~6.0
        
        # ============================================================
        # 2. TOPIC CONTINUITY (Entity tracking across sentences)
        # Good logical flow means adjacent sentences share entities
        # ============================================================
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        def extract_content_words(text):
            """Extract meaningful content words (nouns, verbs approximation)."""
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'under', 'over',
                'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'if', 'then',
                'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them',
                'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
                'i', 'me', 'my', 'what', 'which', 'who', 'whom', 'how', 'when',
                'where', 'why', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than',
                'too', 'very', 'just', 'about', 'also', 'here', 'there'
            }
            text_words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return set(w for w in text_words if w not in stopwords)
        
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, num_sentences):
                prev_words = extract_content_words(sentences[i-1])
                curr_words = extract_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    # Use a modified overlap coefficient (not Jaccard - different from variant 4)
                    # Overlap coefficient = |intersection| / min(|A|, |B|)
                    min_size = min(len(prev_words), len(curr_words))
                    if min_size > 0:
                        overlap_coeff = overlap / min_size
                        continuity_scores.append(overlap_coeff)
            
            if continuity_scores:
                avg_continuity = sum(continuity_scores) / len(continuity_scores)
                # Also measure consistency (low variance = smooth flow)
                if len(continuity_scores) > 1:
                    mean_c = avg_continuity
                    variance = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                    consistency = 1.0 / (1.0 + math.sqrt(variance) * 3)
                else:
                    consistency = 0.7
                
                continuity_score = avg_continuity * 4.0 + consistency * 2.0  # max ~6.0
            else:
                continuity_score = 2.0
        else:
            continuity_score = 2.0 if word_count > 30 else 1.0
        
        # ============================================================
        # 3. ARGUMENT DEPTH: Conditional/nested reasoning
        # ============================================================
        
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bwhen\b.*\b(?:will|would|should|can|could)\b',
            r'\bassuming\b', r'\bprovided that\b', r'\bin the case\b',
            r'\bsuppose\b', r'\beven if\b', r'\bunless\b',
            r'\bwhether\b.*\bor\b', r'\bdepending on\b'
        ]
        
        reasoning_patterns = [
            r'\bthis (?:means|implies|suggests|indicates)\b',
            r'\bwe can (?:see|conclude|determine|infer)\b',
            r'\bit is (?:important|crucial|essential|worth noting)\b',
            r'\bthe reason\b', r'\bone reason\b', r'\banother reason\b',
            r'\bfor (?:example|instance)\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bin other words\b', r'\bthat is\b',
            r'\bmore importantly\b', r'\bsignificantly\b'
        ]
        
        conditional_count = sum(len(re.findall(p, response_lower)) for p in conditional_patterns)
        reasoning_count = sum(len(re.findall(p, response_lower)) for p in reasoning_patterns)
        
        depth_raw = conditional_count * 0.5 + reasoning_count * 0.4
        depth_score = min(depth_raw, 4.0)  # max 4.0
        
        # ============================================================
        # 4. STRUCTURAL BALANCE AND PROPORTION
        # Well-structured arguments have balanced paragraph/section lengths
        # ============================================================
        
        # Split by double newlines or numbered items
        paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.|\*\s|[-•])', response_clean)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]
        num_paragraphs = len(paragraphs)
        
        if num_paragraphs >= 2:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_len = sum(para_lengths) / len(para_lengths)
            
            if avg_para_len > 0:
                # Coefficient of variation - lower means more balanced
                para_std = math.sqrt(sum((l - avg_para_len) ** 2 for l in para_lengths) / len(para_lengths))
                cv = para_std / avg_para_len
                balance = 1.0 / (1.0 + cv)
            else:
                balance = 0.3
            
            # Reward having multiple well-sized paragraphs
            structure_quality = min(num_paragraphs / 4.0, 1.0)
            
            balance_score = balance * 2.0 + structure_quality * 2.0  # max ~4.0
        else:
            # Single block of text - check if it's at least well-sized
            balance_score = 1.5 if word_count > 50 else 0.5
        
        # ============================================================
        # 5. OPENING QUALITY: Does the response start with a clear thesis/position?
        # ============================================================
        
        first_sentence = sentences[0].lower() if sentences else ""
        
        opening_score = 0.0
        
        # Direct address of the query
        query_words = extract_content_words(query_clean)
        first_sent_words = extract_content_words(first_sentence)
        if query_words and first_sent_words:
            query_address = len(query_words & first_sent_words) / max(len(query_words), 1)
            opening_score += min(query_address * 2.0, 1.5)
        
        # Clear stance/thesis indicators
        thesis_patterns = [
            r'^(?:yes|no|certainly|absolutely|definitely)',
            r'\bi (?:believe|think|recommend|suggest|would)\b',
            r'\bthe (?:answer|solution|key|main|best|most)\b',
            r'\bthere are (?:several|many|multiple|a few)\b',
            r'\bhere(?:\'s| is| are)\b',
            r'\blet(?:\'s| me| us)\b'
        ]
        
        for p in thesis_patterns:
            if re.search(p, first_sentence):
                opening_score += 0.5
                break
        
        opening_score = min(opening_score, 2.0)  # max 2.0
        
        # ============================================================
        # 6. CONTRADICTION AND INCOHERENCE DETECTION (penalty)
        # ============================================================
        
        contradiction_penalty = 0.0
        
        # Check for self-contradicting patterns
        contradiction_patterns = [
            (r'\bis\b.*\bbut\b.*\bis not\b', 0.3),
            (r'\balways\b.*\bnever\b', 0.3),
            (r'\bshould\b.*\bshould not\b', 0.2),
            (r'\bcannot\b.*\bcan\b', 0.1),  # mild - could be valid contrast
        ]
        
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, response_lower):
                contradiction_penalty += penalty
        
        # Excessive hedging can signal incoherent reasoning
        hedge_words = [
            r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bmight\b',
            r'\bcould be\b', r'\bsort of\b', r'\bkind of\b', r'\bsomewhat\b',
            r'\bi guess\b', r'\bi\'m not sure\b'
        ]
        hedge_count = sum(len(re.findall(p, response_lower)) for p in hedge_words)
        hedge_density = (hedge_count / word_count) * 100
        if hedge_density > 3.0:
            contradiction_penalty += min((hedge_density - 3.0) * 0.3, 1.5)
        
        # Repetition detection (same phrase repeated = less coherent)
        # Extract 4-grams
        if word_count >= 4:
            four_grams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
            four_gram_counts = Counter(four_grams)
            repeated = sum(1 for c in four_gram_counts.values() if c > 2)
            if repeated > 3:
                contradiction_penalty += min(repeated * 0.2, 1.5)
        
        contradiction_penalty = min(contradiction_penalty, 3.0)
        
        # ============================================================
        # 7. RESPONSE COMPLETENESS AND LENGTH ADEQUACY
        # ============================================================
        
        # Reasonable length for the query type
        query_complexity = len(query_clean.split())
        
        # Check if response seems truncated
        truncation_penalty = 0.0
        if response_clean[-1] not in '.!?"\')' and word_count > 30:
            # Likely truncated
            truncation_penalty = 0.5
        
        # Length adequacy - not too short, not excessively repetitive
        if word_count < 20:
            length_score = 0.5
        elif word_count < 50:
            length_score = 1.0
        elif word_count < 200:
            length_score = 2.0
        else:
            length_score = 2.0
        
        length_score -= truncation_penalty
        length_score = max(length_score, 0.0)
        
        # ============================================================
        # 8. ENUMERATION AND PROGRESSION QUALITY
        # ============================================================
        
        # Check for numbered lists or lettered lists
        numbered_items = re.findall(r'(?:^|\n)\s*(\d+)[.\)]\s', response_clean)
        if numbered_items:
            numbers = [int(n) for n in numbered_items]
            # Check if numbers are sequential
            is_sequential = all(numbers[i] <= numbers[i+1] for i in range(len(numbers)-1))
            starts_at_one = numbers[0] == 1 if numbers else False
            
            enum_score = 0.0
            if is_sequential:
                enum_score += 0.5
            if starts_at_one:
                enum_score += 0.3
            enum_score += min(len(numbers) * 0.15, 0.7)
        else:
            # Check for markdown headers or bold markers for structure
            headers = re.findall(r'(?:^|\n)\s*#{1,4}\s+\w+', response_clean)
            bold_markers = re.findall(r'\*\*[^*]+\*\*', response_clean)
            
            enum_score = 0.0
            if headers:
                enum_score += min(len(headers) * 0.2, 0.8)
            if bold_markers:
                enum_score += min(len(bold_markers) * 0.1, 0.5)
        
        enum_score = min(enum_score, 1.5)
        
        # ============================================================
        # 9. SEMANTIC PROGRESSION (are new ideas introduced progressively?)
        # ============================================================
        
        if num_sentences >= 3:
            # Track cumulative vocabulary - new words should appear throughout
            cumulative_words = set()
            new_word_rates = []
            
            for sent in sentences:
                sent_words = extract_content_words(sent)
                if sent_words:
                    new_words = sent_words - cumulative_words
                    rate = len(new_words) / len(sent_words) if sent_words else 0
                    new_word_rates.append(rate)
                    cumulative_words |= sent_words
            
            if new_word_rates:
                # Good responses introduce new content throughout (not just at start)
                # Check that later sentences still introduce some new content
                if len(new_word_rates) >= 4:
                    first_half = new_word_rates[:len(new_word_rates)//2]
                    second_half = new_word_rates[len(new_word_rates)//2:]
                    avg_first = sum(first_half) / len(first_half)
                    avg_second = sum(second_half) / len(second_half)
                    
                    # Second half should still introduce ~30%+ new content
                    if avg_second > 0.2:
                        progression_score = 1.5
                    elif avg_second > 0.1:
                        progression_score = 1.0
                    else:
                        progression_score = 0.5  # Getting repetitive
                else:
                    avg_rate = sum(new_word_rates) / len(new_word_rates)
                    progression_score = min(avg_rate * 2.5, 1.5)
            else:
                progression_score = 0.5
        else:
            progression_score = 0.8
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        # Weights reflect importance to logical coherence
        raw_score = (
            discourse_score * 1.5 +      # max ~9.0  - discourse markers are key
            continuity_score * 1.3 +      # max ~7.8  - topic continuity matters
            depth_score * 1.0 +           # max ~4.0  - argument depth
            balance_score * 0.8 +         # max ~3.2  - structural balance
            opening_score * 1.0 +         # max ~2.0  - clear opening
            length_score * 0.7 +          # max ~1.4  - appropriate length
            enum_score * 0.8 +            # max ~1.2  - enumeration quality
            progression_score * 1.0 -     # max ~1.5  - semantic progression
            contradiction_penalty * 1.5   # penalty for incoherence
        )
        
        # Normalize to 0-100 scale
        # Theoretical max around 30, typical good response around 15-25
        normalized_score = (raw_score / 30.0) * 100.0
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, normalized_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 30.0
            return 10.0
        except:
            return 10.0