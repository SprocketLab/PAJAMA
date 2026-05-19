def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a sentence-level analysis approach:
    - Analyzes each sentence for epistemic stance (certain, hedged, speculative)
    - Measures the ratio of qualified claims to total claims
    - Evaluates structural completeness and information density
    - Penalizes repetition and empty responses
    - Rewards appropriate epistemic markers matched to query type
    """
    try:
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 2.0
        
        response = response.strip()
        query = query.strip()
        
        import re
        from collections import Counter
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if not sentences:
            return 1.0
        
        # === QUERY ANALYSIS: Determine if topic demands uncertainty ===
        query_lower = query.lower()
        
        # Topics that inherently involve uncertainty/speculation
        speculative_query_signals = [
            'hypothetical', 'predict', 'future', 'might', 'could', 'would',
            'opinion', 'believe', 'think', 'controversial', 'debate',
            'compare and contrast', 'pros and cons', 'advantages', 'disadvantages',
            'should', 'ethical', 'moral', 'best way', 'worst'
        ]
        
        # Topics that are more factual/instructional
        factual_query_signals = [
            'describe', 'explain', 'what is', 'what are', 'how does', 'how do',
            'define', 'list', 'provide', 'give', 'name', 'rewrite', 'generate',
            'create', 'write', 'come up with', 'crop', 'convert'
        ]
        
        # Creative/generative tasks
        creative_query_signals = [
            'creative', 'generate', 'come up with', 'write a', 'create',
            'design', 'imagine', 'invent', 'compose', 'craft'
        ]
        
        speculative_score = sum(1 for s in speculative_query_signals if s in query_lower)
        factual_score = sum(1 for s in factual_query_signals if s in query_lower)
        creative_score = sum(1 for s in creative_query_signals if s in query_lower)
        
        query_type = 'neutral'
        if speculative_score > factual_score and speculative_score > creative_score:
            query_type = 'speculative'
        elif factual_score > speculative_score:
            query_type = 'factual'
        elif creative_score > 0:
            query_type = 'creative'
        
        # === SENTENCE-LEVEL EPISTEMIC ANALYSIS ===
        response_lower = response.lower()
        
        # Overconfidence markers - presenting things too definitively
        overconfidence_phrases = [
            'it is clear that', 'obviously', 'undoubtedly', 'without a doubt',
            'there is no question', 'certainly', 'absolutely', 'definitely',
            'everyone knows', 'it is a fact', 'always', 'never',
            'the only way', 'the best way', 'the worst', 'guaranteed',
            'proven beyond', 'unquestionably', 'indisputably'
        ]
        
        # Appropriate hedging/qualification markers
        hedging_phrases = [
            'likely', 'unlikely', 'probably', 'possibly', 'perhaps',
            'research suggests', 'studies suggest', 'evidence suggests',
            'it appears', 'it seems', 'may be', 'might be', 'could be',
            'tends to', 'in general', 'generally', 'typically',
            'one possible', 'one approach', 'some argue', 'some believe',
            'it is thought', 'it is believed', 'according to',
            'in some cases', 'often', 'sometimes', 'frequently',
            'can vary', 'depends on', 'it depends', 'arguably'
        ]
        
        # Epistemic source attribution
        source_markers = [
            'according to', 'research shows', 'studies show', 'experts',
            'scientists', 'researchers', 'data suggests', 'evidence',
            'findings', 'analysis', 'surveys', 'reports indicate'
        ]
        
        # Nuance/qualification markers
        nuance_markers = [
            'however', 'although', 'on the other hand', 'nevertheless',
            'while', 'whereas', 'but', 'yet', 'despite', 'in contrast',
            'conversely', 'alternatively', 'that said', 'nonetheless',
            'it is worth noting', 'importantly', 'notably'
        ]
        
        # Count occurrences
        overconfidence_count = sum(1 for p in overconfidence_phrases if p in response_lower)
        hedging_count = sum(1 for p in hedging_phrases if p in response_lower)
        source_count = sum(1 for p in source_markers if p in response_lower)
        nuance_count = sum(1 for p in nuance_markers if p in response_lower)
        
        # === SENTENCE STRUCTURE ANALYSIS ===
        # Count declarative sentences (potential claims)
        declarative_count = 0
        qualified_count = 0
        
        for sent in sentences:
            sent_lower = sent.lower().strip()
            # Check if sentence makes a claim
            is_claim = (len(sent_lower.split()) > 4 and 
                       not sent_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')))
            
            if is_claim:
                declarative_count += 1
                # Check if the claim is appropriately qualified
                has_hedge = any(h in sent_lower for h in hedging_phrases)
                has_nuance = any(n in sent_lower for n in nuance_markers)
                has_source = any(s in sent_lower for s in source_markers)
                if has_hedge or has_nuance or has_source:
                    qualified_count += 1
        
        # === REPETITION DETECTION (different from word overlap) ===
        # Use n-gram repetition at the phrase level
        words = response_lower.split()
        total_words = len(words)
        
        if total_words < 3:
            return 2.0
        
        # Trigram repetition
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i:i+3]))
        
        trigram_counter = Counter(trigrams)
        if trigrams:
            repeated_trigrams = sum(c - 1 for c in trigram_counter.values() if c > 1)
            trigram_repetition_ratio = repeated_trigrams / len(trigrams)
        else:
            trigram_repetition_ratio = 0
        
        # Sentence-level repetition
        sent_texts = [s.lower().strip() for s in sentences]
        unique_sents = len(set(sent_texts))
        sent_repetition_ratio = 1 - (unique_sents / max(len(sent_texts), 1))
        
        # === INFORMATION DENSITY ===
        # Unique content words (not stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'very', 'just',
            'than', 'too', 'also', 'that', 'this', 'these', 'those', 'it', 'its',
            'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she',
            'him', 'her', 'his', 'i', 'me', 'my', 'more', 'most', 'other',
            'some', 'such', 'no', 'each', 'every', 'all', 'both', 'few', 'any',
            'which', 'who', 'whom', 'what', 'when', 'where', 'how', 'while'
        }
        
        content_words = [w for w in re.findall(r'[a-z]+', response_lower) if w not in stopwords and len(w) > 2]
        unique_content = set(content_words)
        
        if content_words:
            content_diversity = len(unique_content) / max(len(content_words), 1)
        else:
            content_diversity = 0
        
        # === COMPLETENESS CHECK ===
        # Does the response appear truncated?
        is_truncated = (response[-1] not in '.!?"\')' and 
                       len(response) > 100 and 
                       not response.rstrip().endswith(('>', ']', '}')))
        
        # === SCORING ===
        score = 50.0  # Base score
        
        # 1. Epistemic calibration score (0-20 points)
        epistemic_score = 0.0
        
        if query_type == 'speculative':
            # For speculative queries, reward hedging and nuance
            epistemic_score += min(hedging_count * 3, 8)
            epistemic_score += min(nuance_count * 3, 6)
            epistemic_score += min(source_count * 2, 4)
            epistemic_score -= overconfidence_count * 2
        elif query_type == 'factual':
            # For factual queries, moderate hedging is fine, overconfidence less penalized
            epistemic_score += min(hedging_count * 1.5, 4)
            epistemic_score += min(nuance_count * 2, 4)
            epistemic_score += min(source_count * 2, 4)
            epistemic_score -= overconfidence_count * 1
            # Reward clear, direct statements for factual queries
            if declarative_count > 0 and overconfidence_count == 0:
                epistemic_score += 3
        elif query_type == 'creative':
            # For creative queries, epistemic markers less relevant
            epistemic_score += 2  # Small baseline
            epistemic_score -= overconfidence_count * 0.5
        else:
            epistemic_score += min(hedging_count * 2, 6)
            epistemic_score += min(nuance_count * 2, 5)
            epistemic_score += min(source_count * 2, 4)
            epistemic_score -= overconfidence_count * 1.5
        
        epistemic_score = max(min(epistemic_score, 20), -10)
        score += epistemic_score
        
        # 2. Information richness (0-15 points)
        info_score = 0.0
        info_score += min(len(unique_content) * 0.3, 8)
        info_score += min(content_diversity * 7, 7)
        score += info_score
        
        # 3. Structural quality (0-10 points)
        struct_score = 0.0
        # Reward multiple sentences (more complete response)
        struct_score += min(len(sentences) * 1.5, 6)
        # Reward appropriate length
        if 30 <= total_words <= 300:
            struct_score += 4
        elif 15 <= total_words < 30:
            struct_score += 2
        elif total_words > 300:
            struct_score += 2
        score += struct_score
        
        # 4. Repetition penalty (-20 to 0)
        repetition_penalty = 0.0
        repetition_penalty -= trigram_repetition_ratio * 15
        repetition_penalty -= sent_repetition_ratio * 10
        
        # Detect extreme repetition (like "miserable, miserable, miserable")
        if total_words > 0:
            most_common_word = Counter(content_words).most_common(1)
            if most_common_word:
                top_freq = most_common_word[0][1]
                if top_freq / max(len(content_words), 1) > 0.5 and len(content_words) > 3:
                    repetition_penalty -= 15
        
        score += max(repetition_penalty, -20)
        
        # 5. Truncation penalty
        if is_truncated:
            score -= 5
        
        # 6. Qualification ratio bonus for longer responses
        if declarative_count > 3:
            qual_ratio = qualified_count / declarative_count
            if query_type == 'speculative':
                score += qual_ratio * 5
            else:
                score += qual_ratio * 2
        
        # 7. Appropriate explanation depth
        # Check if response actually addresses the query
        query_words = set(re.findall(r'[a-z]+', query_lower)) - stopwords
        response_content = set(re.findall(r'[a-z]+', response_lower)) - stopwords
        
        if query_words:
            relevance = len(query_words & response_content) / len(query_words)
            score += relevance * 5
        
        # 8. Bonus for comparative/contrastive structure when appropriate
        if 'compare' in query_lower or 'contrast' in query_lower:
            comparative_words = ['while', 'whereas', 'unlike', 'similarly', 'both', 
                               'differ', 'different', 'same', 'contrast', 'compare',
                               'on the other hand', 'in contrast']
            comp_count = sum(1 for c in comparative_words if c in response_lower)
            score += min(comp_count * 2, 6)
        
        # Clamp final score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        try:
            if response and response.strip():
                return 25.0
            return 0.0
        except Exception:
            return 0.0