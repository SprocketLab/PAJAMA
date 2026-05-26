def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant uses a DEPENDENCY CHAIN analysis approach:
    - Tracks logical connective density and variety
    - Measures causal/explanatory chain depth
    - Analyzes sentence-to-sentence coherence via shared concept threading
    - Evaluates progressive elaboration (information gain per sentence)
    - Penalizes repetition and information stagnation
    
    Different from other variants: focuses on inter-sentence dependency graphs
    and information flow analysis rather than simple word overlap, bullet detection,
    or regex pattern matching.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 5:
            return 0.5
        
        import re
        from collections import Counter
        import math
        
        # === Helper: tokenize into lowercased words ===
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
        # === Helper: split into sentences ===
        def split_sentences(text):
            # Split on sentence-ending punctuation
            sents = re.split(r'(?<=[.!?])\s+', text)
            # Also split on semicolons and colons that introduce new clauses
            expanded = []
            for s in sents:
                parts = re.split(r';\s*', s)
                expanded.extend(parts)
            return [s.strip() for s in expanded if len(s.strip()) > 3]
        
        sentences = split_sentences(response)
        num_sentences = len(sentences)
        words = tokenize(response)
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # ============================================================
        # FEATURE 1: Logical Connective Density & Variety
        # Measures how many different logical/causal connectors are used
        # ============================================================
        
        # Categories of connectives with weights
        causal_connectives = [
            'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'owing to', 'caused by', 'leads to',
            'resulting in', 'so that', 'in order to', 'for this reason'
        ]
        
        sequential_connectives = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'subsequently', 'afterward', 'initially', 'lastly',
            'to begin', 'following this', 'at this point', 'step'
        ]
        
        contrastive_connectives = [
            'however', 'but', 'although', 'whereas', 'while', 'on the other hand',
            'in contrast', 'nevertheless', 'yet', 'despite', 'conversely',
            'on the contrary', 'rather than', 'instead'
        ]
        
        elaborative_connectives = [
            'specifically', 'in particular', 'for example', 'for instance',
            'such as', 'namely', 'that is', 'in other words', 'to illustrate',
            'more specifically', 'to clarify', 'meaning that', 'this means'
        ]
        
        conditional_connectives = [
            'if', 'when', 'unless', 'provided that', 'assuming',
            'in the case', 'given that', 'suppose', 'whether'
        ]
        
        additive_connectives = [
            'furthermore', 'moreover', 'additionally', 'also', 'in addition',
            'not only', 'besides', 'likewise', 'similarly'
        ]
        
        response_lower = response.lower()
        
        def count_connective_category(connectives):
            found = set()
            count = 0
            for c in connectives:
                occurrences = response_lower.count(c)
                if occurrences > 0:
                    found.add(c)
                    count += occurrences
            return count, len(found)
        
        causal_count, causal_variety = count_connective_category(causal_connectives)
        seq_count, seq_variety = count_connective_category(sequential_connectives)
        contrast_count, contrast_variety = count_connective_category(contrastive_connectives)
        elab_count, elab_variety = count_connective_category(elaborative_connectives)
        cond_count, cond_variety = count_connective_category(conditional_connectives)
        add_count, add_variety = count_connective_category(additive_connectives)
        
        total_connective_count = causal_count + seq_count + contrast_count + elab_count + cond_count + add_count
        total_variety = causal_variety + seq_variety + contrast_variety + elab_variety + cond_variety + add_variety
        categories_used = sum(1 for v in [causal_variety, seq_variety, contrast_variety, 
                                           elab_variety, cond_variety, add_variety] if v > 0)
        
        # Normalize connective density per 100 words
        connective_density = (total_connective_count / max(num_words, 1)) * 100
        
        # Score: reward density up to a point, and reward variety of categories
        connective_score = min(connective_density * 1.5, 10) + min(total_variety * 0.6, 8) + min(categories_used * 1.2, 7)
        # Max ~25
        
        # ============================================================
        # FEATURE 2: Sentence-to-Sentence Concept Threading
        # Measures how well each sentence connects to previous ones
        # via shared concepts (not just word overlap but concept chains)
        # ============================================================
        
        if num_sentences >= 2:
            sentence_word_sets = []
            # Use content words (filter out very common words)
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                        'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or',
                        'not', 'no', 'as', 'if', 'but', 'so', 'than', 'too', 'very',
                        'just', 'about', 'up', 'out', 'into', 'over', 'after', 'i',
                        'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us',
                        'them', 'my', 'your', 'his', 'our', 'their', 'which', 'who',
                        'what', 'where', 'when', 'how', 'all', 'each', 'every', 'both',
                        'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
                        'any', 'few', 'many', 'much', 'one', 'two'}
            
            for s in sentences:
                s_words = set(tokenize(s)) - stopwords
                # Also add bigrams for better concept matching
                s_tokens = tokenize(s)
                bigrams = set()
                for i in range(len(s_tokens) - 1):
                    if s_tokens[i] not in stopwords or s_tokens[i+1] not in stopwords:
                        bigrams.add(s_tokens[i] + '_' + s_tokens[i+1])
                sentence_word_sets.append(s_words | bigrams)
            
            # Calculate threading: each sentence should share concepts with at least one prior sentence
            threading_scores = []
            for i in range(1, len(sentence_word_sets)):
                # Check overlap with ALL previous sentences (not just the immediate one)
                max_overlap = 0
                cumulative_prev = set()
                for j in range(i):
                    cumulative_prev |= sentence_word_sets[j]
                
                if len(sentence_word_sets[i]) > 0 and len(cumulative_prev) > 0:
                    overlap = len(sentence_word_sets[i] & cumulative_prev)
                    # Normalize by the smaller set
                    norm = min(len(sentence_word_sets[i]), len(cumulative_prev))
                    max_overlap = overlap / max(norm, 1)
                
                threading_scores.append(max_overlap)
            
            avg_threading = sum(threading_scores) / max(len(threading_scores), 1)
            # Also measure: does each sentence add NEW information?
            new_info_scores = []
            for i in range(1, len(sentence_word_sets)):
                cumulative_prev = set()
                for j in range(i):
                    cumulative_prev |= sentence_word_sets[j]
                if len(sentence_word_sets[i]) > 0:
                    new_concepts = len(sentence_word_sets[i] - cumulative_prev)
                    new_ratio = new_concepts / max(len(sentence_word_sets[i]), 1)
                    new_info_scores.append(new_ratio)
            
            avg_new_info = sum(new_info_scores) / max(len(new_info_scores), 1) if new_info_scores else 0
            
            # Good reasoning: moderate threading (connected) AND moderate new info (progressing)
            # Both should be > 0 for good reasoning
            threading_quality = avg_threading * avg_new_info * 4  # product rewards balance
            threading_score = min(threading_quality * 20, 15) + min(avg_threading * 5, 5)
        else:
            threading_score = 2.0  # Single sentence gets low score
        
        # ============================================================
        # FEATURE 3: Progressive Elaboration Depth
        # Measures whether the response builds up complexity
        # (later sentences should have richer vocabulary / longer constructs)
        # ============================================================
        
        if num_sentences >= 3:
            # Measure unique content words per sentence
            sent_complexities = []
            for s in sentences:
                s_words = tokenize(s)
                content_words = [w for w in s_words if w not in stopwords and len(w) > 2]
                # Complexity = unique content words + average word length
                if content_words:
                    avg_word_len = sum(len(w) for w in content_words) / len(content_words)
                    complexity = len(set(content_words)) * (avg_word_len / 4.0)
                else:
                    complexity = 0
                sent_complexities.append(complexity)
            
            # Check if complexity is maintained or grows (not declining)
            if len(sent_complexities) >= 2:
                # Compare first half vs second half average complexity
                mid = len(sent_complexities) // 2
                first_half = sum(sent_complexities[:mid]) / max(mid, 1)
                second_half = sum(sent_complexities[mid:]) / max(len(sent_complexities) - mid, 1)
                
                if first_half > 0:
                    elaboration_ratio = second_half / first_half
                else:
                    elaboration_ratio = 1.0
                
                # Reward maintained or increasing complexity
                if elaboration_ratio >= 0.8:
                    elaboration_score = min(elaboration_ratio * 5, 10)
                else:
                    elaboration_score = elaboration_ratio * 4
            else:
                elaboration_score = 3.0
        else:
            elaboration_score = 2.0
        
        # ============================================================
        # FEATURE 4: Explanation Depth Markers
        # Looks for patterns that indicate explanatory depth
        # ============================================================
        
        depth_patterns = [
            (r'\bthis\s+(means|implies|suggests|indicates|shows)\b', 2.0),
            (r'\bwhich\s+(means|allows|enables|causes|leads|results|provides)\b', 1.8),
            (r'\bin\s+other\s+words\b', 2.5),
            (r'\bthe\s+reason\s+(is|being|for|why)\b', 2.5),
            (r'\bthis\s+is\s+because\b', 2.5),
            (r'\bas\s+a\s+result\b', 2.0),
            (r'\bby\s+doing\s+(this|so)\b', 1.5),
            (r'\bwhat\s+this\s+means\b', 2.5),
            (r'\bto\s+understand\s+why\b', 3.0),
            (r'\blet\s+(me|us)\s+(explain|break|consider|look|examine)\b', 2.0),
            (r'\bhere\s*\'?s?\s+(why|how|what)\b', 2.0),
            (r'\bthe\s+key\s+(point|idea|concept|insight|difference)\b', 1.5),
            (r'\bnote\s+that\b', 1.5),
            (r'\bimportantly\b', 1.5),
            (r'\bput\s+(simply|differently|another\s+way)\b', 2.0),
            (r'\bto\s+put\s+it\b', 1.5),
            (r'\bthink\s+of\s+it\s+(as|like)\b', 2.0),
            (r'\bso\s+when\b', 1.0),
            (r'\bonce\s+.{5,30}\s*,\s*(then|the)\b', 1.5),
        ]
        
        depth_score = 0
        for pattern, weight in depth_patterns:
            matches = re.findall(pattern, response_lower)
            depth_score += len(matches) * weight
        
        depth_score = min(depth_score, 15)
        
        # ============================================================
        # FEATURE 5: Repetition Penalty
        # Penalize responses that repeat the same phrases/sentences
        # ============================================================
        
        # Check for repeated trigrams
        if num_words >= 6:
            trigrams = []
            for i in range(len(words) - 2):
                trigrams.append(tuple(words[i:i+3]))
            
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            repetition_ratio = repeated_trigrams / max(total_trigrams, 1)
            
            # Heavy penalty for high repetition
            repetition_penalty = min(repetition_ratio * 30, 20)
        else:
            repetition_penalty = 0
        
        # ============================================================
        # FEATURE 6: Structural Complexity
        # Measures clause depth via comma/subordinate clause usage
        # ============================================================
        
        # Count commas per sentence (indicates subordinate clauses)
        if num_sentences > 0:
            comma_counts = [s.count(',') for s in sentences]
            avg_commas = sum(comma_counts) / num_sentences
            
            # Parenthetical/aside usage (shows meta-reasoning)
            parens = response.count('(') + response.count(')')
            dashes = response.count(' - ') + response.count('—')
            
            structural_score = min(avg_commas * 1.5, 5) + min((parens + dashes) * 0.5, 3)
        else:
            structural_score = 0
        
        # ============================================================
        # FEATURE 7: Response Substantiveness relative to query
        # ============================================================
        
        query_words = set(tokenize(query)) - stopwords
        response_content_words = set(tokenize(response)) - stopwords
        
        # Response should address query concepts AND add new ones
        if query_words:
            query_coverage = len(query_words & response_content_words) / max(len(query_words), 1)
        else:
            query_coverage = 0.5
        
        # Vocabulary richness: unique words / total words
        if num_words > 0:
            vocab_richness = len(set(words)) / num_words
        else:
            vocab_richness = 0
        
        substantiveness_score = query_coverage * 3 + vocab_richness * 5 + min(num_sentences * 0.5, 4)
        substantiveness_score = min(substantiveness_score, 12)
        
        # ============================================================
        # FEATURE 8: Length appropriateness
        # Very short responses rarely show reasoning
        # ============================================================
        
        # Use log scale for length bonus
        length_score = min(math.log(max(num_words, 1) + 1) * 1.5, 8)
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        raw_score = (
            connective_score * 0.20 +      # Logical connectives
            threading_score * 0.18 +         # Concept threading
            elaboration_score * 0.10 +       # Progressive elaboration
            depth_score * 0.18 +             # Explanation depth markers
            structural_score * 0.08 +        # Structural complexity
            substantiveness_score * 0.14 +   # Substantiveness
            length_score * 0.12 -            # Length
            repetition_penalty * 0.20        # Repetition penalty
        )
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
    
    except Exception as e:
        # Never crash - return a neutral score
        try:
            # At minimum, give credit for non-empty responses
            if response and len(response.strip()) > 10:
                return 3.0
            return 1.0
        except:
            return 1.0