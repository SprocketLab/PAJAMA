def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis,
    causal/logical connective chains, argument density, and structural progression.
    
    This variant focuses on:
    1. Discourse marker chains and logical connective analysis
    2. Argument progression (claim -> evidence -> conclusion patterns)
    3. Sentence-level coherence via topic threading
    4. Structural scaffolding quality (not just presence of headers/bullets)
    5. Contradiction/negation pattern detection
    6. Rhetorical move analysis
    """
    import re
    import math
    from collections import Counter, defaultdict
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.0
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 1.0
        
        words = re.findall(r'\b[a-z]+\b', response.lower())
        word_count = len(words)
        if word_count < 5:
            return 1.0
        
        # ========== 1. DISCOURSE CONNECTIVE CHAIN ANALYSIS ==========
        # Categorize connectives by their rhetorical function
        additive_connectives = [
            'also', 'furthermore', 'moreover', 'additionally', 'in addition',
            'besides', 'likewise', 'similarly', 'as well', 'plus', 'and also'
        ]
        causal_connectives = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'owing to',
            'for this reason', 'accordingly', 'thereby'
        ]
        contrastive_connectives = [
            'however', 'but', 'although', 'nevertheless', 'on the other hand',
            'in contrast', 'whereas', 'yet', 'despite', 'conversely',
            'nonetheless', 'while', 'even though', 'on the contrary'
        ]
        temporal_connectives = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'subsequently', 'afterwards', 'before', 'after', 'meanwhile',
            'initially', 'eventually', 'lastly', 'to begin', 'to start'
        ]
        conclusive_connectives = [
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'to conclude', 'in short', 'all in all', 'ultimately',
            'in the end', 'to sum up'
        ]
        elaboration_connectives = [
            'for example', 'for instance', 'specifically', 'in particular',
            'namely', 'that is', 'in other words', 'to illustrate',
            'such as', 'to clarify'
        ]
        
        response_lower = response.lower()
        
        def count_connectives(connective_list):
            count = 0
            for c in connective_list:
                count += len(re.findall(r'\b' + re.escape(c) + r'\b', response_lower))
            return count
        
        additive_count = count_connectives(additive_connectives)
        causal_count = count_connectives(causal_connectives)
        contrastive_count = count_connectives(contrastive_connectives)
        temporal_count = count_connectives(temporal_connectives)
        conclusive_count = count_connectives(conclusive_connectives)
        elaboration_count = count_connectives(elaboration_connectives)
        
        total_connectives = (additive_count + causal_count + contrastive_count + 
                           temporal_count + conclusive_count + elaboration_count)
        
        # Connective diversity: how many different categories are used
        categories_used = sum(1 for c in [additive_count, causal_count, contrastive_count,
                                           temporal_count, conclusive_count, elaboration_count] if c > 0)
        
        # Connective density (per 100 words)
        connective_density = (total_connectives / max(word_count, 1)) * 100
        
        # Score: reward diversity and moderate density
        connective_diversity_score = min(categories_used / 4.0, 1.5)  # max ~1.5
        connective_density_score = min(connective_density / 5.0, 1.5)  # aim for ~5 per 100 words
        
        # ========== 2. ARGUMENT PROGRESSION ANALYSIS ==========
        # Detect rhetorical moves: claim, evidence, reasoning, conclusion
        
        claim_markers = [
            r'\bi (?:believe|think|argue|suggest|propose|contend)\b',
            r'\bit is (?:important|essential|crucial|clear|evident)\b',
            r'\bthe (?:key|main|primary|central) (?:point|idea|argument|reason)\b',
            r'\bshould\b', r'\bmust\b', r'\bneed to\b',
            r'\bis (?:better|worse|superior|inferior)\b'
        ]
        
        evidence_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\baccording to\b', r'\bresearch (?:shows|suggests|indicates)\b',
            r'\bstudies (?:show|suggest|indicate)\b', r'\bevidence\b',
            r'\bdata\b', r'\bstatistics\b', r'\bin fact\b'
        ]
        
        reasoning_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bthis (?:means|implies|suggests|indicates|shows)\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bit follows\b',
            r'\bif .+ then\b', r'\bgiven that\b'
        ]
        
        def count_markers(marker_list):
            count = 0
            for pattern in marker_list:
                count += len(re.findall(pattern, response_lower))
            return count
        
        claim_count = count_markers(claim_markers)
        evidence_count = count_markers(evidence_markers)
        reasoning_count = count_markers(reasoning_markers)
        
        # Argument completeness: having claims supported by evidence and reasoning
        argument_types_present = sum(1 for c in [claim_count, evidence_count, reasoning_count] if c > 0)
        argument_completeness = argument_types_present / 3.0
        
        # Argument density
        argument_density = (claim_count + evidence_count + reasoning_count) / max(len(sentences), 1)
        argument_density_score = min(argument_density * 2.0, 1.5)
        
        # ========== 3. SENTENCE-LEVEL TOPIC THREADING ==========
        # Measure how well consecutive sentences share content words (topic continuity)
        
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
                'if', 'while', 'this', 'that', 'these', 'those', 'it', 'its', 'i',
                'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'our', 'their', 'what', 'which', 'who', 'whom'
            }
            w = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return set(w) - stop_words
        
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if words_a and words_b:
                    # Dice coefficient for consecutive sentence overlap
                    overlap = len(words_a & words_b)
                    dice = (2 * overlap) / (len(words_a) + len(words_b))
                    coherence_scores.append(dice)
            
            if coherence_scores:
                avg_coherence = sum(coherence_scores) / len(coherence_scores)
                # Also measure variance - consistent coherence is better
                coherence_variance = sum((s - avg_coherence) ** 2 for s in coherence_scores) / len(coherence_scores)
                topic_threading_score = avg_coherence * 3.0 - coherence_variance * 2.0
                topic_threading_score = max(0, min(topic_threading_score, 1.5))
            else:
                topic_threading_score = 0.3
        else:
            topic_threading_score = 0.3
        
        # ========== 4. STRUCTURAL SCAFFOLDING QUALITY ==========
        # Beyond just detecting headers/bullets, analyze the quality of structure
        
        # Detect numbered/ordered sequences
        numbered_items = re.findall(r'(?:^|\n)\s*(\d+)[.)]\s', response)
        has_ordered_list = len(numbered_items) >= 2
        
        # Check if numbering is sequential
        numbering_sequential = False
        if has_ordered_list:
            nums = [int(n) for n in numbered_items]
            expected = list(range(nums[0], nums[0] + len(nums)))
            numbering_sequential = nums == expected
        
        # Detect markdown headers with hierarchy
        headers = re.findall(r'^(#{1,4})\s+(.+)$', response, re.MULTILINE)
        header_levels = [len(h[0]) for h in headers]
        
        # Check header hierarchy quality
        header_hierarchy_quality = 0.0
        if len(header_levels) >= 2:
            # Good hierarchy: levels should be non-decreasing or follow a pattern
            transitions = [header_levels[i+1] - header_levels[i] for i in range(len(header_levels)-1)]
            # Penalize wild jumps
            smooth_transitions = sum(1 for t in transitions if abs(t) <= 1)
            header_hierarchy_quality = smooth_transitions / len(transitions) if transitions else 0
        
        # Detect paragraph structure (blocks separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        
        # Score structural scaffolding
        scaffolding_score = 0.0
        if has_ordered_list:
            scaffolding_score += 0.4
            if numbering_sequential:
                scaffolding_score += 0.3
        if len(headers) >= 1:
            scaffolding_score += 0.3
            scaffolding_score += header_hierarchy_quality * 0.3
        if len(paragraphs) >= 2:
            scaffolding_score += 0.2
        
        scaffolding_score = min(scaffolding_score, 1.5)
        
        # ========== 5. CONTRADICTION/CONSISTENCY ANALYSIS ==========
        # Look for potential internal contradictions
        
        negation_patterns = [
            (r'\bis\b', r'\bis not\b'), (r'\bcan\b', r'\bcannot\b'),
            (r'\bwill\b', r'\bwill not\b'), (r'\bshould\b', r'\bshould not\b'),
            (r'\balways\b', r'\bnever\b'), (r'\beveryone\b', r'\bno one\b'),
            (r'\ball\b', r'\bnone\b')
        ]
        
        # Check for absolute statements followed by contradictions
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'no one', 
                         'absolutely', 'certainly', 'definitely', 'impossible']
        hedge_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be',
                      'sometimes', 'often', 'usually', 'generally', 'typically']
        
        absolute_count = sum(1 for w in absolute_words if w in response_lower)
        hedge_count = sum(1 for w in hedge_words if w in response_lower)
        
        # High mix of absolutes and hedges may indicate inconsistency
        if absolute_count > 0 and hedge_count > 0:
            consistency_penalty = min((absolute_count * hedge_count) * 0.05, 0.5)
        else:
            consistency_penalty = 0.0
        
        # ========== 6. RHETORICAL MOVE SEQUENCE ANALYSIS ==========
        # Analyze the sequence of rhetorical moves through the response
        
        # Classify each sentence by its rhetorical function
        def classify_sentence(sent):
            s = sent.lower()
            if any(re.search(p, s) for p in [r'^yes\b', r'^no\b', r'^i (?:think|believe|feel)',
                                               r'\bshould\b', r'\bmust\b']):
                return 'claim'
            if any(w in s for w in ['for example', 'for instance', 'such as', 'like']):
                return 'evidence'
            if any(w in s for w in ['because', 'therefore', 'thus', 'since', 'so ']):
                return 'reasoning'
            if any(w in s for w in ['in conclusion', 'overall', 'in summary', 'to summarize']):
                return 'conclusion'
            if any(w in s for w in ['however', 'but', 'although', 'on the other hand']):
                return 'contrast'
            if s.endswith('?'):
                return 'question'
            if any(w in s for w in ['here are', 'let me', "let's", 'the following']):
                return 'introduction'
            return 'exposition'
        
        move_sequence = [classify_sentence(s) for s in sentences]
        
        # Reward good rhetorical patterns
        rhetorical_score = 0.0
        
        # Starting with introduction/claim is good
        if move_sequence and move_sequence[0] in ('introduction', 'claim'):
            rhetorical_score += 0.3
        
        # Having variety in moves is good
        unique_moves = len(set(move_sequence))
        rhetorical_score += min(unique_moves / 5.0, 0.4)
        
        # Check for claim-reasoning pairs (good argument structure)
        for i in range(len(move_sequence) - 1):
            if move_sequence[i] == 'claim' and move_sequence[i+1] in ('reasoning', 'evidence'):
                rhetorical_score += 0.15
            if move_sequence[i] == 'evidence' and move_sequence[i+1] == 'reasoning':
                rhetorical_score += 0.1
        
        # Ending with conclusion is good
        if move_sequence and move_sequence[-1] in ('conclusion', 'claim'):
            rhetorical_score += 0.2
        
        rhetorical_score = min(rhetorical_score, 1.5)
        
        # ========== 7. QUERY-RESPONSE ALIGNMENT ==========
        # Check if the response addresses the query
        
        query_content = get_content_words(query)
        response_content = get_content_words(response)
        
        if query_content and response_content:
            query_coverage = len(query_content & response_content) / max(len(query_content), 1)
        else:
            query_coverage = 0.0
        
        alignment_score = min(query_coverage * 1.5, 1.0)
        
        # ========== 8. SENTENCE LENGTH VARIATION ==========
        # Good writing has varied sentence lengths
        
        sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        if len(sent_lengths) >= 2:
            avg_len = sum(sent_lengths) / len(sent_lengths)
            length_variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            # Moderate variance is good (not too uniform, not too wild)
            cv = math.sqrt(length_variance) / max(avg_len, 1)  # coefficient of variation
            # Optimal CV around 0.3-0.6
            if 0.2 <= cv <= 0.7:
                variation_score = 0.5
            elif cv < 0.2:
                variation_score = cv * 2.5  # too uniform
            else:
                variation_score = max(0, 0.5 - (cv - 0.7) * 0.5)  # too wild
        else:
            variation_score = 0.2
        
        # ========== 9. RESPONSE COMPLETENESS ==========
        # Check if response seems truncated
        
        truncation_penalty = 0.0
        last_chars = response[-20:] if len(response) >= 20 else response
        
        # Check for mid-sentence truncation
        if not re.search(r'[.!?:]\s*$', response.rstrip()):
            truncation_penalty += 0.3
        
        # Check for incomplete list items
        if has_ordered_list:
            last_num = int(numbered_items[-1]) if numbered_items else 0
            if last_num >= 3 and not re.search(r'[.!?]\s*$', response.rstrip()):
                truncation_penalty += 0.2
        
        # ========== 10. OPENING QUALITY ==========
        # Good responses often start with a clear framing
        
        opening_score = 0.0
        first_sent = sentences[0].lower() if sentences else ""
        
        # Direct answer to question
        if first_sent.startswith(('yes', 'no', 'certainly', 'absolutely')):
            opening_score += 0.3
        
        # Framing/context setting
        if any(w in first_sent for w in ['great question', 'good question', "let's", 'here are',
                                          'there are several', 'this is']):
            opening_score += 0.2
        
        # Acknowledgment + pivot
        if re.search(r'^(that\'s|what) a (great|good|interesting|fun)', first_sent):
            opening_score += 0.2
        
        opening_score = min(opening_score, 0.5)
        
        # ========== FINAL SCORE COMPOSITION ==========
        
        # Weighted combination
        raw_score = (
            connective_diversity_score * 1.2 +    # Diversity of logical connectives
            connective_density_score * 0.8 +       # Density of connectives
            argument_completeness * 1.5 +           # Claim-evidence-reasoning completeness
            argument_density_score * 0.8 +          # Density of argument markers
            topic_threading_score * 1.5 +           # Sentence-to-sentence coherence
            scaffolding_score * 1.0 +               # Structural quality
            rhetorical_score * 1.3 +                # Rhetorical move quality
            alignment_score * 0.8 +                 # Query-response alignment
            variation_score * 0.5 +                 # Sentence length variation
            opening_score * 0.6 -                   # Opening quality
            consistency_penalty * 1.0 -             # Contradiction penalty
            truncation_penalty * 1.5                # Truncation penalty
        )
        
        # Length bonus: longer well-structured responses tend to be better
        # But diminishing returns
        length_factor = math.log(max(word_count, 10)) / math.log(500)
        length_factor = min(max(length_factor, 0.5), 1.2)
        
        raw_score *= length_factor
        
        # Normalize to 0-10 range
        # Typical raw scores range from ~0.5 to ~8
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a minimal score based on response length
        try:
            return min(len(response.split()) / 50.0, 3.0)
        except:
            return 0.0