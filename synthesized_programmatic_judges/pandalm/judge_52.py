def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant focuses on:
    1. Causal/logical connective density (because, therefore, since, thus, etc.)
    2. Sequential/process markers (first, then, next, finally, etc.)
    3. Explanation depth via clause complexity (subordinate clauses)
    4. Elaboration ratio (how much the response expands beyond restating the query)
    5. Hedging and qualification markers (suggests, indicates, likely, etc.)
    6. Contrast/comparison markers for nuanced reasoning
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        from collections import Counter
        
        resp_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        # Edge case: very short or empty response
        if len(resp_lower) < 5:
            return 0.0
        
        resp_words = resp_lower.split()
        word_count = len(resp_words)
        
        if word_count < 2:
            return 0.5
        
        # --- Feature 1: Causal/logical connectives ---
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bwhich means\b',
            r'\bwhich leads to\b', r'\bin order to\b', r'\bso that\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bcaused by\b',
            r'\bresulting in\b', r'\bleading to\b', r'\bit follows\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        # Normalize by word count (per 100 words)
        causal_density = (causal_count / word_count) * 100 if word_count > 0 else 0
        causal_score = min(causal_density * 5, 15)  # max 15 points
        
        # --- Feature 2: Sequential/process markers ---
        sequential_markers = [
            r'\bfirst\b', r'\bfirstly\b', r'\bsecond\b', r'\bsecondly\b',
            r'\bthird\b', r'\bthirdly\b', r'\bthen\b', r'\bnext\b',
            r'\bafter that\b', r'\bfinally\b', r'\blastly\b',
            r'\bsubsequently\b', r'\bfollowing this\b', r'\bin the end\b',
            r'\bto begin\b', r'\bto start\b', r'\bstep \d\b',
            r'\bonce\b', r'\bafter\b', r'\bbefore\b',
            r'\binitially\b', r'\beventually\b', r'\bat this point\b',
        ]
        seq_count = 0
        for pattern in sequential_markers:
            seq_count += len(re.findall(pattern, resp_lower))
        
        seq_density = (seq_count / word_count) * 100 if word_count > 0 else 0
        seq_score = min(seq_density * 4, 12)  # max 12 points
        
        # --- Feature 3: Clause complexity via subordinating conjunctions and relative pronouns ---
        subordinating = [
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\beven though\b',
            r'\bif\b', r'\bunless\b', r'\bwhich\b', r'\bthat\b',
            r'\bwho\b', r'\bwhere\b', r'\bwhen\b', r'\bwherever\b',
            r'\bwhenever\b', r'\bhowever\b', r'\bnevertheless\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\bon the other hand\b', r'\bin contrast\b',
        ]
        sub_count = 0
        for pattern in subordinating:
            sub_count += len(re.findall(pattern, resp_lower))
        
        sub_density = (sub_count / word_count) * 100 if word_count > 0 else 0
        clause_score = min(sub_density * 2.5, 12)  # max 12 points
        
        # --- Feature 4: Elaboration ratio ---
        # How much new content vs just restating the query
        query_words_set = set(query_lower.split())
        resp_words_set = set(resp_words)
        
        if len(resp_words_set) > 0:
            novel_words = resp_words_set - query_words_set
            # Remove very common words from novel count
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                        'and', 'or', 'but', 'not', 'no', 'it', 'its', 'this', 'that',
                        'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
                        'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him',
                        'us', 'them', 'as', 'if', 'so'}
            meaningful_novel = novel_words - stopwords
            meaningful_resp = resp_words_set - stopwords
            
            if len(meaningful_resp) > 0:
                elaboration_ratio = len(meaningful_novel) / len(meaningful_resp)
            else:
                elaboration_ratio = 0
        else:
            elaboration_ratio = 0
        
        elaboration_score = elaboration_ratio * 10  # max ~10 points
        
        # --- Feature 5: Explanation markers ---
        explanation_markers = [
            r'\bthis means\b', r'\bin other words\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bto illustrate\b', r'\bto clarify\b', r'\bput differently\b',
            r'\bmore specifically\b', r'\bin fact\b', r'\bindeed\b',
            r'\bnotably\b', r'\bimportantly\b', r'\bsignificantly\b',
            r'\bessentially\b', r'\bfundamentally\b',
        ]
        expl_count = 0
        for pattern in explanation_markers:
            expl_count += len(re.findall(pattern, resp_lower))
        
        expl_density = (expl_count / word_count) * 100 if word_count > 0 else 0
        expl_score = min(expl_density * 6, 10)  # max 10 points
        
        # --- Feature 6: Sentence count and average sentence length ---
        sentences = re.split(r'[.!?]+', resp_lower)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        # Multiple sentences suggest step-by-step exposition
        sentence_score = min(num_sentences * 1.5, 10)  # max 10 points
        
        # --- Feature 7: Comma density as proxy for complex sentence structure ---
        comma_count = response.count(',')
        comma_density = (comma_count / word_count) * 100 if word_count > 0 else 0
        comma_score = min(comma_density * 1.5, 8)  # max 8 points
        
        # --- Feature 8: Response length (moderate bonus, with diminishing returns) ---
        import math
        length_score = min(math.log(word_count + 1) * 2, 10)  # max ~10 points
        
        # --- Feature 9: Contrast/comparison markers (shows nuanced reasoning) ---
        contrast_markers = [
            r'\bbut\b', r'\bhowever\b', r'\bon the other hand\b',
            r'\bin contrast\b', r'\bwhereas\b', r'\bwhile\b',
            r'\balthough\b', r'\bdespite\b', r'\bnevertheless\b',
            r'\byet\b', r'\binstead\b', r'\brather\b',
            r'\bcompared to\b', r'\bunlike\b', r'\bdiffers?\b',
            r'\bsimilarly\b', r'\blikewise\b', r'\bjust as\b',
            r'\bboth\b', r'\beither\b', r'\bneither\b',
        ]
        contrast_count = 0
        for pattern in contrast_markers:
            contrast_count += len(re.findall(pattern, resp_lower))
        
        contrast_density = (contrast_count / word_count) * 100 if word_count > 0 else 0
        contrast_score = min(contrast_density * 3, 8)  # max 8 points
        
        # --- Feature 10: Repetition penalty ---
        # Detect excessive repetition which indicates low quality
        word_freq = Counter(resp_words)
        if word_count > 10:
            # Remove stopwords for repetition check
            content_words = [w for w in resp_words if w not in stopwords and len(w) > 3]
            if content_words:
                content_freq = Counter(content_words)
                most_common_count = content_freq.most_common(1)[0][1] if content_freq else 0
                repetition_ratio = most_common_count / len(content_words) if content_words else 0
                if repetition_ratio > 0.3:
                    repetition_penalty = (repetition_ratio - 0.3) * 30
                else:
                    repetition_penalty = 0
            else:
                repetition_penalty = 0
        else:
            repetition_penalty = 0
        
        # --- Feature 11: Unique vocabulary richness ---
        if word_count > 0:
            vocab_richness = len(set(resp_words)) / word_count
        else:
            vocab_richness = 0
        vocab_score = vocab_richness * 5  # max ~5 points
        
        # --- Aggregate score ---
        total = (
            causal_score +       # max 15
            seq_score +          # max 12
            clause_score +       # max 12
            elaboration_score +  # max 10
            expl_score +         # max 10
            sentence_score +     # max 10
            comma_score +        # max 8
            length_score +       # max 10
            contrast_score +     # max 8
            vocab_score          # max 5
        )
        
        total -= repetition_penalty
        
        # Clamp to 0-100
        total = max(0.0, min(100.0, total))
        
        return round(total, 2)
        
    except Exception:
        return 0.0