def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    discourse coherence analysis: tracking logical connectives, causal chains,
    progressive elaboration depth, and explanation density.
    
    This variant focuses on:
    1. Causal/logical connective density (tracking discourse markers)
    2. Progressive elaboration (each sentence building on previous)
    3. Explanation density (ratio of explanatory content to assertions)
    4. Referential coherence (pronouns/references linking back to prior content)
    5. Depth of unpacking (how much a response expands beyond minimal answer)
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
        
        # Tokenize into sentences (simple split)
        import re
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 3]
        words = response.lower().split()
        word_count = len(words)
        
        if word_count < 3:
            return 1.0
        
        score = 0.0
        
        # ============================================================
        # 1. CAUSAL/LOGICAL CONNECTIVE ANALYSIS (0-20 points)
        # Categorize connectives by their reasoning function
        # ============================================================
        
        causal_connectives = [
            'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'owing to', 'leads to', 'causes',
            'so that', 'in order to', 'for this reason', 'this means',
            'which means', 'meaning that', 'implies that', 'resulting in'
        ]
        
        elaboration_connectives = [
            'specifically', 'in particular', 'for example', 'for instance',
            'such as', 'namely', 'to illustrate', 'in other words',
            'that is', 'i.e.', 'e.g.', 'to clarify', 'more specifically',
            'to be specific', 'in detail'
        ]
        
        sequential_connectives = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'subsequently', 'afterward', 'following this', 'step',
            'to begin', 'initially', 'lastly', 'in the first place',
            'to start', 'moving on', 'after that', 'once'
        ]
        
        contrastive_connectives = [
            'however', 'but', 'although', 'whereas', 'while', 'on the other hand',
            'in contrast', 'nevertheless', 'nonetheless', 'yet', 'despite',
            'conversely', 'rather', 'instead', 'unlike', 'differ'
        ]
        
        additive_reasoning = [
            'furthermore', 'moreover', 'additionally', 'also', 'in addition',
            'besides', 'not only', 'as well', 'along with', 'coupled with'
        ]
        
        response_lower = response.lower()
        
        causal_count = sum(1 for c in causal_connectives if c in response_lower)
        elaboration_count = sum(1 for c in elaboration_connectives if c in response_lower)
        sequential_count = sum(1 for c in sequential_connectives if c in response_lower)
        contrastive_count = sum(1 for c in contrastive_connectives if c in response_lower)
        additive_count = sum(1 for c in additive_reasoning if c in response_lower)
        
        # Weight causal and elaboration connectives more heavily
        connective_score = (
            causal_count * 3.0 +
            elaboration_count * 2.5 +
            sequential_count * 2.0 +
            contrastive_count * 1.5 +
            additive_count * 1.0
        )
        
        # Normalize by word count to get density, then scale
        connective_density = connective_score / max(word_count, 1) * 100
        score += min(connective_density * 3.0, 20.0)
        
        # ============================================================
        # 2. PROGRESSIVE ELABORATION DEPTH (0-20 points)
        # Measure how sentences build upon each other through
        # lexical chains and referential links
        # ============================================================
        
        if len(sentences) >= 2:
            elaboration_links = 0
            total_pairs = 0
            
            for i in range(1, len(sentences)):
                prev_words = set(sentences[i-1].lower().split())
                curr_words = set(sentences[i].lower().split())
                
                # Remove very common words for meaningful overlap
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                           'been', 'being', 'have', 'has', 'had', 'do', 'does',
                           'did', 'will', 'would', 'could', 'should', 'may',
                           'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                           'on', 'with', 'at', 'by', 'from', 'it', 'its',
                           'this', 'that', 'and', 'or', 'not', 'as'}
                
                prev_content = prev_words - stopwords
                curr_content = curr_words - stopwords
                
                if prev_content and curr_content:
                    overlap = len(prev_content & curr_content)
                    new_content = len(curr_content - prev_content)
                    
                    # Good elaboration: some overlap (coherence) + new content (progression)
                    if overlap > 0 and new_content > 0:
                        elaboration_links += min(overlap, 3) + min(new_content / 3, 2)
                    elif new_content > 0:
                        elaboration_links += 0.5  # New content but no clear link
                
                # Check for referential pronouns linking back
                referential_words = {'this', 'these', 'that', 'those', 'it', 'its',
                                    'they', 'them', 'their', 'which', 'who'}
                curr_sentence_lower = sentences[i].lower()
                ref_count = sum(1 for w in curr_words if w in referential_words)
                # Referential pronouns at start of sentence are especially good for coherence
                first_few = curr_sentence_lower.split()[:3]
                if any(w in referential_words for w in first_few):
                    elaboration_links += 1.5
                
                total_pairs += 1
            
            if total_pairs > 0:
                avg_elaboration = elaboration_links / total_pairs
                score += min(avg_elaboration * 4.0, 20.0)
        else:
            # Single sentence gets minimal elaboration score
            score += 1.0
        
        # ============================================================
        # 3. EXPLANATION DENSITY (0-20 points)
        # Ratio of explanatory phrases vs bare assertions
        # ============================================================
        
        # Explanatory phrase indicators
        explanation_markers = [
            'this means', 'this is because', 'the reason', 'which allows',
            'which enables', 'which helps', 'which provides', 'which makes',
            'in this case', 'when this happens', 'as a consequence',
            'the idea is', 'the point is', 'what this does', 'how this works',
            'the way', 'by doing', 'through this', 'with this approach',
            'suggests that', 'indicates that', 'shows that', 'demonstrates',
            'explains', 'describes', 'involves', 'requires', 'allows',
            'enables', 'ensures', 'provides', 'includes'
        ]
        
        explanation_count = sum(1 for m in explanation_markers if m in response_lower)
        
        # Also count "why" explanations
        why_patterns = re.findall(r'\bwhy\b|\breason\b|\bpurpose\b|\bgoal\b|\baim\b', response_lower)
        explanation_count += len(why_patterns) * 0.5
        
        explanation_density = explanation_count / max(len(sentences), 1)
        score += min(explanation_density * 8.0, 20.0)
        
        # ============================================================
        # 4. STRUCTURAL COMPLEXITY & DEPTH (0-15 points)
        # Measure syntactic complexity as proxy for reasoning depth
        # ============================================================
        
        # Count subordinate clauses (approximated by subordinating conjunctions)
        subordinators = ['because', 'although', 'while', 'whereas', 'since',
                        'unless', 'until', 'when', 'where', 'if', 'though',
                        'even though', 'so that', 'in order that', 'provided that',
                        'as long as', 'whether']
        
        subordinate_count = 0
        for sub in subordinators:
            subordinate_count += len(re.findall(r'\b' + re.escape(sub) + r'\b', response_lower))
        
        # Count comma-separated clauses as indicator of complex sentences
        comma_count = response.count(',')
        
        # Average sentence length (longer sentences often contain more reasoning)
        avg_sent_len = word_count / max(len(sentences), 1)
        
        complexity_score = (
            subordinate_count * 2.0 +
            min(comma_count / max(len(sentences), 1), 3) * 1.5 +
            min(avg_sent_len / 5.0, 3.0)  # reward moderate-long sentences
        )
        
        score += min(complexity_score, 15.0)
        
        # ============================================================
        # 5. INFORMATION EXPANSION RATIO (0-15 points)
        # How much does the response expand beyond a minimal answer?
        # ============================================================
        
        query_words = set(query.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'of',
                                                      'to', 'in', 'for', 'on', 'with', 'what',
                                                      'how', 'why', 'when', 'where', 'which',
                                                      'do', 'does', 'did', 'can', 'could',
                                                      'would', 'should', 'will', 'may', 'might'}
        response_words = set(words) - {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'in',
                                        'for', 'on', 'with', 'it', 'its', 'this', 'that'}
        
        # New vocabulary introduced (not from query)
        new_vocab = response_words - query_words
        expansion_ratio = len(new_vocab) / max(len(query_words), 1)
        
        score += min(expansion_ratio * 2.0, 15.0)
        
        # ============================================================
        # 6. ANTI-PATTERNS / PENALTIES
        # ============================================================
        
        # Penalty for excessive repetition (indicates low-quality/broken response)
        if word_count > 10:
            from collections import Counter
            word_freq = Counter(words)
            most_common_freq = word_freq.most_common(1)[0][1]
            repetition_ratio = most_common_freq / word_count
            if repetition_ratio > 0.15:
                score *= max(0.3, 1.0 - (repetition_ratio - 0.15) * 3)
        
        # Penalty for very short responses (likely not showing reasoning)
        if word_count < 10:
            score *= 0.3
        elif word_count < 20:
            score *= 0.6
        elif word_count < 30:
            score *= 0.8
        
        # Penalty for responses that are just echoing the query
        if query_words and response_words:
            echo_ratio = len(response_words & query_words) / max(len(response_words), 1)
            if echo_ratio > 0.7:
                score *= 0.5
        
        # Penalty for empty/placeholder responses
        if response_lower in ['<noinput>', 'n/a', 'none', 'no response']:
            return 0.5
        
        # Bonus for multi-sentence responses that maintain coherence
        if len(sentences) >= 3:
            score += 3.0
        if len(sentences) >= 5:
            score += 2.0
        
        # ============================================================
        # 7. RESPONSE COMPLETENESS (0-5 points)
        # Check if response appears truncated
        # ============================================================
        
        if response[-1] not in '.!?")\']':
            # Likely truncated
            score *= 0.7
        
        # Normalize to 0-100 range
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)
        
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 20:
                return 5.0
            return 1.0
        except:
            return 1.0