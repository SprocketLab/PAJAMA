def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical chain analysis approach.
    
    This variant focuses on:
    1. Causal connective density (because, therefore, since, thus, so, hence, as a result)
    2. Reasoning chain depth - detecting multi-step logical progressions
    3. Elaboration patterns - detecting claim+explanation pairs
    4. Conditional reasoning (if/then patterns)
    5. Perspective/consideration markers (on one hand, alternatively, however)
    6. Self-referential reasoning markers (let me, consider, note that, importantly)
    7. Sentence-to-sentence coherence via shared entity tracking
    8. Ratio of explanatory vs declarative sentences
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        import re
        from collections import Counter
        
        # Tokenize into sentences
        sentences = re.split(r'[.!?]+(?:\s|$)', response_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # ---- Feature 1: Causal connective density ----
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bwhich means\b',
            r'\bwhich leads to\b', r'\bthis means\b', r'\bthis implies\b',
            r'\bit follows\b', r'\baccordingly\b', r'\bfor this reason\b',
            r'\bthat\'s why\b', r'\bthats why\b', r'\bthe reason\b',
            r'\bcaused by\b', r'\bleads to\b', r'\bresults in\b',
        ]
        causal_count = 0
        resp_lower = response_clean.lower()
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, resp_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 3.5, 1.0)  # cap at 1.0
        
        # ---- Feature 2: Reasoning chain depth ----
        # Look for sequences of sentences connected by logical flow
        chain_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bafter that\b',
            r'\bto begin\b', r'\bto start\b', r'\bfollowing\b',
            r'\bstep\b', r'\bstage\b', r'\bphase\b',
            r'\bsubsequently\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\badditionally\b', r'\balso\b',
            r'\bbeyond that\b', r'\bon top of\b',
        ]
        chain_count = 0
        for pattern in chain_markers:
            chain_count += len(re.findall(pattern, resp_lower))
        
        chain_score = min(chain_count / max(num_sentences, 1) * 2.0, 1.0)
        
        # ---- Feature 3: Elaboration patterns ----
        # Detect sentences that elaborate on previous claims
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bput differently\b',
            r'\bto clarify\b', r'\bto put it\b', r'\bwhat this means\b',
            r'\bto be more\b', r'\bmore specifically\b', r'\blike\b',
        ]
        elab_count = 0
        for pattern in elaboration_markers:
            elab_count += len(re.findall(pattern, resp_lower))
        
        elab_score = min(elab_count / num_sentences * 3.0, 1.0)
        
        # ---- Feature 4: Conditional reasoning ----
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\bwould\b', r'\bif\b.*\bcould\b',
            r'\bif\b.*\bmight\b', r'\bif\b.*\bshould\b',
            r'\bwhen\b.*\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bsuppose\b', r'\bprovided that\b', r'\bin case\b',
            r'\bwhereas\b', r'\bwhile\b.*\bon the other\b',
            r'\bunless\b', r'\beven if\b', r'\bwhether\b',
        ]
        cond_count = 0
        for pattern in conditional_patterns:
            # Check per-sentence to avoid cross-sentence matching
            for sent in sentences:
                sent_lower = sent.lower()
                if re.search(pattern, sent_lower):
                    cond_count += 1
                    break  # count pattern once
            # Actually let's count total occurrences across all text
        cond_count2 = 0
        for pattern in conditional_patterns:
            cond_count2 += len(re.findall(pattern, resp_lower))
        
        cond_score = min(cond_count2 / num_sentences * 2.5, 1.0)
        
        # ---- Feature 5: Perspective/consideration markers ----
        perspective_markers = [
            r'\bon one hand\b', r'\bon the other hand\b', r'\bhowever\b',
            r'\balternatively\b', r'\bconversely\b', r'\bin contrast\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bdespite\b',
            r'\balthough\b', r'\beven though\b', r'\bthat said\b',
            r'\bat the same time\b', r'\byet\b', r'\bbut\b',
            r'\bstill\b.*\bthough\b', r'\bwhile it\b',
        ]
        persp_count = 0
        for pattern in perspective_markers:
            persp_count += len(re.findall(pattern, resp_lower))
        
        persp_score = min(persp_count / num_sentences * 2.0, 1.0)
        
        # ---- Feature 6: Metacognitive / self-referential reasoning ----
        meta_markers = [
            r'\blet me\b', r'\bconsider\b', r'\bnote that\b',
            r'\bimportantly\b', r'\bnotice\b', r'\bkeep in mind\b',
            r'\bremember\b', r'\bthe key\b', r'\bcrucially\b',
            r'\bthe point\b', r'\bthe idea\b', r'\bthe question\b',
            r'\bthink about\b', r'\bthink of\b', r'\breflect\b',
            r'\bwe can see\b', r'\bwe need to\b', r'\bwe should\b',
            r'\bit\'s worth\b', r'\bits worth\b', r'\bworth noting\b',
            r'\bin essence\b', r'\bessentially\b', r'\bfundamentally\b',
        ]
        meta_count = 0
        for pattern in meta_markers:
            meta_count += len(re.findall(pattern, resp_lower))
        
        meta_score = min(meta_count / num_sentences * 2.5, 1.0)
        
        # ---- Feature 7: Entity coherence between consecutive sentences ----
        # Track how many content words are shared between consecutive sentences
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'out', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'not',
                'so', 'yet', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too',
                'very', 'just', 'it', 'its', 'this', 'that', 'these', 'those',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
                'who', 'whom', 'how', 'all', 'any', 'there', 'here', 'when',
                'where', 'why', 'if', 'about', 'up', 'also',
            }
            w = re.findall(r'\b[a-z]+\b', text.lower())
            return set(w2 for w2 in w if w2 not in stop_words and len(w2) > 2)
        
        coherence_scores = []
        for i in range(1, len(sentences)):
            prev_words = get_content_words(sentences[i-1])
            curr_words = get_content_words(sentences[i])
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words)
                union = len(prev_words | curr_words)
                coherence_scores.append(overlap / union if union > 0 else 0)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        coherence_score = min(avg_coherence * 5.0, 1.0)
        
        # ---- Feature 8: Explanatory vs declarative sentence ratio ----
        explanatory_starts = [
            r'^(because|since|this is|the reason|this means|what happens|'
            r'essentially|in other words|so |that\'s|the idea|the key|'
            r'to understand|the way|how this|why this)',
        ]
        explanatory_count = 0
        for sent in sentences:
            sent_stripped = sent.strip().lower()
            for pattern in explanatory_starts:
                if re.match(pattern, sent_stripped):
                    explanatory_count += 1
                    break
        
        explanatory_ratio = explanatory_count / num_sentences
        explanatory_score = min(explanatory_ratio * 4.0, 1.0)
        
        # ---- Feature 9: Response substantiveness ----
        # Longer, more developed responses tend to show more reasoning
        # But we use a logarithmic scale to avoid over-rewarding length
        import math
        length_factor = min(math.log(num_words + 1) / math.log(300), 1.0)
        
        # ---- Feature 10: Clause density ----
        # More clauses per sentence = more complex reasoning
        clause_markers = [',', ';', ':', ' - ', ' -- ', '—']
        clause_count = 0
        for marker in clause_markers:
            clause_count += response_clean.count(marker)
        
        clause_density = clause_count / num_sentences
        clause_score = min(clause_density / 3.0, 1.0)
        
        # ---- Feature 11: Question engagement ----
        # Does the response reference or engage with the query?
        query_content = get_content_words(query)
        response_content = get_content_words(response_clean)
        if query_content and response_content:
            query_engagement = len(query_content & response_content) / len(query_content)
        else:
            query_engagement = 0
        engagement_score = min(query_engagement * 2.0, 1.0)
        
        # ---- Feature 12: Parenthetical explanations ----
        paren_count = len(re.findall(r'\(.*?\)', response_clean))
        paren_score = min(paren_count / num_sentences * 3.0, 1.0)
        
        # ---- Combine features with weights ----
        score = (
            causal_score * 1.8 +        # Causal reasoning is central
            chain_score * 1.5 +          # Step-wise progression
            elab_score * 1.4 +           # Elaboration/examples
            cond_score * 1.2 +           # Conditional reasoning
            persp_score * 1.0 +          # Multiple perspectives
            meta_score * 1.3 +           # Metacognitive markers
            coherence_score * 0.8 +      # Sentence coherence
            explanatory_score * 1.0 +    # Explanatory sentences
            length_factor * 1.5 +        # Substantiveness
            clause_score * 0.8 +         # Clause complexity
            engagement_score * 0.9 +     # Query engagement
            paren_score * 0.4            # Parenthetical explanations
        )
        
        # Normalize to 0-10 scale
        max_possible = (1.0 * 1.8 + 1.0 * 1.5 + 1.0 * 1.4 + 1.0 * 1.2 +
                       1.0 * 1.0 + 1.0 * 1.3 + 1.0 * 0.8 + 1.0 * 1.0 +
                       1.0 * 1.5 + 1.0 * 0.8 + 1.0 * 0.9 + 1.0 * 0.4)
        
        normalized = (score / max_possible) * 10.0
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This makes mid-range scores more discriminative
        x = normalized - 3.0  # center around 3
        transformed = 10.0 / (1.0 + math.exp(-0.5 * x))
        
        return round(transformed, 3)
        
    except Exception:
        return 2.0