def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    a sentence-level coherence flow analysis approach.
    
    This variant focuses on:
    1. Sentence-to-sentence logical flow (do sentences build on each other?)
    2. Causal/logical connective density and variety
    3. Explanation depth via clause complexity
    4. Progressive information disclosure patterns
    5. Ratio of substantive vs. vacuous sentences
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses get low scores for reasoning transparency
        if len(response_stripped) < 10:
            return 0.5
        
        # ---- 1. Sentence segmentation ----
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        if len(sentences) == 0:
            return 0.5
        
        # ---- 2. Causal/logical connective analysis ----
        # These indicate reasoning steps and logical flow
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bleads to\b', r'\bresults in\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b',
        ]
        
        elaboration_connectives = [
            r'\bspecifically\b', r'\bin particular\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bto illustrate\b', r'\bto clarify\b',
        ]
        
        sequential_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bafter that\b', r'\bfinally\b', r'\blast\b',
            r'\bsubsequently\b', r'\bfollowing this\b', r'\bto begin\b',
            r'\bstep \d+\b', r'\b\d+\)\b', r'\b\d+\.\s',
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bdespite\b', r'\byet\b', r'\bconversely\b',
        ]
        
        conditional_connectives = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b', r'\bwhenever\b',
        ]
        
        response_lower = response_stripped.lower()
        
        def count_patterns(patterns, text):
            total = 0
            unique = 0
            for p in patterns:
                matches = len(re.findall(p, text))
                if matches > 0:
                    unique += 1
                    total += matches
            return total, unique
        
        causal_count, causal_unique = count_patterns(causal_connectives, response_lower)
        elab_count, elab_unique = count_patterns(elaboration_connectives, response_lower)
        seq_count, seq_unique = count_patterns(sequential_connectives, response_lower)
        contrast_count, contrast_unique = count_patterns(contrastive_connectives, response_lower)
        cond_count, cond_unique = count_patterns(conditional_connectives, response_lower)
        
        total_connectives = causal_count + elab_count + seq_count + contrast_count + cond_count
        total_unique_types = causal_unique + elab_unique + seq_unique + contrast_unique + cond_unique
        
        # Connective variety score (0-1): how many different types of connectives used
        connective_variety = min(1.0, total_unique_types / 12.0)
        
        # Connective density: connectives per sentence
        connective_density = total_connectives / max(len(sentences), 1)
        connective_density_score = min(1.0, connective_density / 1.5)
        
        # ---- 3. Clause complexity analysis ----
        # Count subordinate clauses and compound structures per sentence
        subordinators = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhere\b',
            r'\bwhen\b', r'\bwhile\b', r'\balthough\b', r'\bbecause\b',
            r'\bsince\b', r'\bunless\b', r'\bif\b', r'\bwhereas\b',
        ]
        
        clause_counts = []
        for sent in sentences:
            sent_lower = sent.lower()
            clause_count = 1  # base clause
            for sub in subordinators:
                clause_count += len(re.findall(sub, sent_lower))
            # Also count commas as potential clause boundaries
            comma_count = sent.count(',')
            clause_count += comma_count * 0.3
            clause_counts.append(clause_count)
        
        avg_clause_complexity = sum(clause_counts) / len(clause_counts) if clause_counts else 1
        clause_complexity_score = min(1.0, (avg_clause_complexity - 1) / 3.0)
        clause_complexity_score = max(0.0, clause_complexity_score)
        
        # ---- 4. Sentence-to-sentence lexical cohesion (flow) ----
        # Measure how much consecutive sentences share content words (indicating building on previous points)
        def get_content_words(text):
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
                'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 'just', 'and', 'but', 'or', 'if', 'it', 'its',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
                'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                'our', 'their', 'what', 'which', 'who', 'whom', 'how', 'when',
                'where', 'why', 'also', 'about',
            }
            words = re.findall(r'\b[a-z]+\b', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        cohesion_scores = []
        if len(sentences) >= 2:
            for i in range(1, len(sentences)):
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    if union > 0:
                        cohesion_scores.append(overlap / union)
        
        avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0
        # Moderate cohesion is good (too high = repetitive, too low = disjointed)
        # Optimal around 0.15-0.35
        if avg_cohesion < 0.05:
            cohesion_quality = 0.1
        elif avg_cohesion < 0.15:
            cohesion_quality = 0.4
        elif avg_cohesion <= 0.40:
            cohesion_quality = 1.0
        elif avg_cohesion <= 0.60:
            cohesion_quality = 0.6
        else:
            cohesion_quality = 0.3  # too repetitive
        
        # ---- 5. Progressive information disclosure ----
        # Check if new content words are introduced across sentences
        seen_words = set()
        new_word_ratios = []
        for sent in sentences:
            words = get_content_words(sent)
            if words:
                new_words = words - seen_words
                ratio = len(new_words) / len(words)
                new_word_ratios.append(ratio)
                seen_words.update(words)
        
        # Good responses introduce new info progressively (not all at once, not none)
        avg_new_ratio = sum(new_word_ratios) / len(new_word_ratios) if new_word_ratios else 0
        # Optimal: 0.4-0.8 (some new info each sentence, but also connecting to previous)
        if avg_new_ratio < 0.2:
            progressive_score = 0.2
        elif avg_new_ratio < 0.4:
            progressive_score = 0.6
        elif avg_new_ratio <= 0.85:
            progressive_score = 1.0
        else:
            progressive_score = 0.7
        
        # ---- 6. Substantive sentence ratio ----
        # Sentences that contain actual information vs. filler
        substantive_count = 0
        for sent in sentences:
            words = re.findall(r'\b[a-z]+\b', sent.lower())
            content_words = get_content_words(sent)
            if len(words) >= 4 and len(content_words) >= 2:
                substantive_count += 1
        
        substantive_ratio = substantive_count / len(sentences) if sentences else 0
        
        # ---- 7. Explanation markers (showing the 'why') ----
        why_markers = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis means\b',
            r'\bin this case\b', r'\bthis suggests\b', r'\bthis indicates\b',
            r'\bthis implies\b', r'\bwe can see\b', r'\bwe can conclude\b',
            r'\bnote that\b', r'\bit follows\b', r'\bthis works\b',
            r'\bthe idea\b', r'\bthe point\b', r'\bthe key\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bessentially\b',
            r'\bfundamentally\b', r'\bin essence\b',
            r'\blet me\b', r'\blet\'s\b', r'\bconsider\b',
            r'\bto understand\b', r'\bto see why\b',
        ]
        
        why_count = 0
        why_unique = 0
        for p in why_markers:
            matches = len(re.findall(p, response_lower))
            if matches > 0:
                why_unique += 1
                why_count += matches
        
        why_score = min(1.0, why_count / 4.0)
        
        # ---- 8. Detect garbage/repetition ----
        # Check for excessive repetition which indicates low quality
        words_all = re.findall(r'\b\w+\b', response_lower)
        if len(words_all) > 10:
            word_freq = Counter(words_all)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            repetition_ratio = most_common_freq / len(words_all)
            # Also check for repeated phrases (3-grams)
            trigrams = [' '.join(words_all[i:i+3]) for i in range(len(words_all)-2)]
            if trigrams:
                trigram_freq = Counter(trigrams)
                most_common_trigram = trigram_freq.most_common(1)[0][1]
                trigram_rep = most_common_trigram / len(trigrams)
            else:
                trigram_rep = 0
        else:
            repetition_ratio = 0
            trigram_rep = 0
        
        # Penalty for excessive repetition
        repetition_penalty = 0.0
        if trigram_rep > 0.15:
            repetition_penalty = min(3.0, (trigram_rep - 0.15) * 15)
        
        # ---- 9. Response length adequacy ----
        word_count = len(words_all)
        # Minimum viable response for showing reasoning
        if word_count < 5:
            length_score = 0.1
        elif word_count < 15:
            length_score = 0.3
        elif word_count < 30:
            length_score = 0.5
        elif word_count < 60:
            length_score = 0.7
        elif word_count <= 300:
            length_score = 1.0
        else:
            length_score = 0.9  # very long may be rambling
        
        # ---- 10. Query relevance via content word overlap ----
        query_content = get_content_words(query)
        response_content = get_content_words(response_stripped)
        if query_content and response_content:
            relevance = len(query_content & response_content) / len(query_content)
            relevance_score = min(1.0, relevance * 1.5)
        else:
            relevance_score = 0.3
        
        # ---- 11. Detect if response is mostly code/HTML/noise ----
        code_chars = sum(1 for c in response_stripped if c in '{}[]<>=/\\;')
        code_ratio = code_chars / len(response_stripped) if response_stripped else 0
        code_penalty = 0.0
        if code_ratio > 0.1:
            code_penalty = min(2.0, (code_ratio - 0.1) * 10)
        
        # ---- 12. Multi-sentence structure bonus ----
        # Having 2+ sentences is important for showing reasoning
        sentence_count = len(sentences)
        if sentence_count == 1:
            multi_sent_score = 0.2
        elif sentence_count == 2:
            multi_sent_score = 0.5
        elif sentence_count <= 8:
            multi_sent_score = 1.0
        elif sentence_count <= 15:
            multi_sent_score = 0.9
        else:
            multi_sent_score = 0.7
        
        # ---- COMPOSITE SCORE ----
        # Weight the components
        score = (
            connective_variety * 1.5 +          # variety of logical connectives (0-1.5)
            connective_density_score * 1.2 +     # density of connectives (0-1.2)
            clause_complexity_score * 0.8 +      # clause complexity (0-0.8)
            cohesion_quality * 1.0 +             # sentence-to-sentence flow (0-1.0)
            progressive_score * 0.7 +            # progressive info disclosure (0-0.7)
            substantive_ratio * 1.0 +            # substantive content (0-1.0)
            why_score * 1.3 +                    # explanation markers (0-1.3)
            length_score * 1.0 +                 # adequate length (0-1.0)
            relevance_score * 0.8 +              # query relevance (0-0.8)
            multi_sent_score * 0.7               # multi-sentence structure (0-0.7)
        )
        # Max possible ~10.0
        
        # Apply penalties
        score -= repetition_penalty
        score -= code_penalty
        
        # Clamp to 0-10
        score = max(0.0, min(10.0, score))
        
        # Scale to 0-10 range more discriminatively
        # The raw max is ~10 but typical good responses get ~5-7
        # Rescale: multiply by 1.3 and cap
        score = min(10.0, score * 1.3)
        
        return round(score, 2)
    
    except Exception:
        return 2.0