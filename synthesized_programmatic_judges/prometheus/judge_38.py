def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis,
    causal/logical connective density, contradiction detection, and progressive
    elaboration patterns. 
    
    This variant focuses on:
    1. Discourse connective analysis (causal, contrastive, additive, temporal)
    2. Progressive elaboration detection (ideas building on each other)
    3. Contradiction/inconsistency signals
    4. Sentence-level coherence via entity threading (topic continuity)
    5. Structural scaffolding detection (framing, development, resolution)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        if len(response_clean) < 10:
            return 0.5
        
        sentences = re.split(r'(?<=[.!?])\s+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # ============================================================
        # 1. DISCOURSE CONNECTIVE ANALYSIS
        # Classify connectives by their discourse function
        # ============================================================
        
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bfor this reason\b',
            r'\bthat\'s why\b', r'\bthis is why\b', r'\baccordingly\b',
            r'\bit follows\b', r'\bgiven that\b', r'\bin light of\b'
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bdespite\b',
            r'\byet\b', r'\bwhile\b', r'\bwhereas\b', r'\bnonetheless\b',
            r'\beven though\b', r'\binstead\b', r'\brather than\b',
            r'\bon the contrary\b', r'\bstill\b'
        ]
        
        additive_connectives = [
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bnot only\b', r'\bequally\b',
            r'\blikewise\b', r'\bsimilarly\b', r'\bcoupled with\b'
        ]
        
        temporal_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bafterward\b', r'\bsubsequently\b',
            r'\bmeanwhile\b', r'\bpreviously\b', r'\bbefore\b', r'\bafter\b',
            r'\binitially\b', r'\beventually\b', r'\bat this point\b',
            r'\bonce\b', r'\blast\b', r'\blastly\b'
        ]
        
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bto illustrate\b',
            r'\bto clarify\b', r'\bput simply\b', r'\bimagine\b',
            r'\bconsider\b', r'\bthink of\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_patterns(patterns, text):
            count = 0
            for p in patterns:
                count += len(re.findall(p, text))
            return count
        
        causal_count = count_patterns(causal_connectives, resp_lower)
        contrastive_count = count_patterns(contrastive_connectives, resp_lower)
        additive_count = count_patterns(additive_connectives, resp_lower)
        temporal_count = count_patterns(temporal_connectives, resp_lower)
        elaboration_count = count_patterns(elaboration_connectives, resp_lower)
        
        total_connectives = causal_count + contrastive_count + additive_count + temporal_count + elaboration_count
        
        # Connective density (per sentence)
        connective_density = total_connectives / num_sentences
        
        # Variety of connective types used (out of 5 categories)
        connective_types_used = sum([
            causal_count > 0,
            contrastive_count > 0,
            additive_count > 0,
            temporal_count > 0,
            elaboration_count > 0
        ])
        
        # Score: reward density up to a point, and variety
        connective_score = min(connective_density * 2.5, 3.0) + min(connective_types_used * 0.6, 3.0)
        # Max ~6.0
        
        # ============================================================
        # 2. ENTITY THREADING / TOPIC CONTINUITY
        # Measure how well consecutive sentences share content words
        # (not just overlap, but specifically tracking entity/topic chains)
        # ============================================================
        
        def get_content_words(text):
            """Extract content words (nouns, verbs, adjectives heuristic)"""
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
                'or', 'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
                'whom', 'about', 'up', 'also', 'like', 'get', 'got', 'make', 'made',
                'take', 'come', 'go', 'know', 'think', 'see', 'want', 'use', 'find',
                'give', 'tell', 'say', 'said', 'try', 'need', 'feel', 'become',
                'keep', 'let', 'begin', 'seem', 'help', 'show', 'hear', 'play',
                'run', 'move', 'live', 'believe', 'bring', 'happen', 'must', 'right',
                'still', 'well', 'back', 'even', 'much', 'many', 'really', 'don',
                't', 's', 're', 've', 'll', 'd', 'doesn', 'didn', 'won', 'wouldn',
                'couldn', 'shouldn', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven',
                'hadn', 'don', 'now', 'new', 'way', 'thing', 'things'
            }
            w = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(w) - stop_words
        
        if len(sentences) >= 2:
            continuity_scores = []
            for i in range(1, len(sentences)):
                prev_content = get_content_words(sentences[i-1])
                curr_content = get_content_words(sentences[i])
                if prev_content and curr_content:
                    # Proportion of current sentence's content words that appeared in previous
                    forward_link = len(prev_content & curr_content) / max(len(curr_content), 1)
                    continuity_scores.append(forward_link)
            
            if continuity_scores:
                avg_continuity = sum(continuity_scores) / len(continuity_scores)
                # Also check consistency: low variance means steady threading
                if len(continuity_scores) > 1:
                    mean_c = avg_continuity
                    var_c = sum((x - mean_c)**2 for x in continuity_scores) / len(continuity_scores)
                    consistency_bonus = max(0, 0.5 - var_c)  # Lower variance = better
                else:
                    consistency_bonus = 0.25
            else:
                avg_continuity = 0.1
                consistency_bonus = 0
        else:
            avg_continuity = 0.15
            consistency_bonus = 0
        
        # Score entity threading: reward moderate continuity (not too low, not too repetitive)
        # Ideal continuity is around 0.15-0.4
        if avg_continuity < 0.05:
            threading_score = 0.5
        elif avg_continuity < 0.15:
            threading_score = 1.5
        elif avg_continuity <= 0.45:
            threading_score = 3.0
        else:
            # Too repetitive
            threading_score = 2.0
        threading_score += consistency_bonus
        # Max ~3.5
        
        # ============================================================
        # 3. PROGRESSIVE ELABORATION DETECTION
        # Check if the response builds ideas progressively 
        # (increasing specificity, examples after claims, etc.)
        # ============================================================
        
        # Detect claim-then-support patterns
        claim_markers = [
            r'\bit\'s\s+\w+\s+(?:important|crucial|essential|necessary|vital)\b',
            r'\byou\s+(?:should|need|must|can|could)\b',
            r'\bremember\s+(?:to|that)\b',
            r'\bthe\s+key\b', r'\bthe\s+main\b', r'\bthe\s+goal\b',
            r'\bthis\s+(?:is|means|helps|allows|ensures)\b',
        ]
        
        support_markers = [
            r'\bfor\s+(?:example|instance)\b', r'\bsuch\s+as\b',
            r'\bthis\s+(?:includes|involves|means)\b',
            r'\blike\s+(?:when|how|a)\b', r'\bimagine\b',
            r'\bconsider\b', r'\bthink\s+of\b', r'\bsay\b',
            r'\bspecifically\b', r'\bin\s+practice\b',
        ]
        
        claim_count = count_patterns(claim_markers, resp_lower)
        support_count = count_patterns(support_markers, resp_lower)
        
        # Check for claim-support adjacency
        claim_support_pairs = 0
        for i in range(len(sentences) - 1):
            s_lower = sentences[i].lower()
            next_lower = sentences[i+1].lower() if i+1 < len(sentences) else ""
            has_claim = any(re.search(p, s_lower) for p in claim_markers)
            has_support = any(re.search(p, next_lower) for p in support_markers)
            if has_claim and has_support:
                claim_support_pairs += 1
        
        elaboration_score = min(claim_count * 0.3, 1.0) + min(support_count * 0.4, 1.0) + min(claim_support_pairs * 0.8, 1.5)
        # Max ~3.5
        
        # ============================================================
        # 4. CONTRADICTION / INCONSISTENCY SIGNALS
        # Detect potential contradictions or confused reasoning
        # ============================================================
        
        contradiction_patterns = [
            r'\bbut\s+(?:then\s+again|wait|actually|no)\b',
            r'\bI\s+(?:mean|guess)\b',
            r'\bor\s+(?:maybe|perhaps|not)\b',
            r'\bactually,?\s+(?:no|never\s+mind)\b',
            r'\bwell,?\s+(?:sort\s+of|kind\s+of|not\s+really)\b',
        ]
        
        # Dismissive / undermining patterns (indicating weak logical structure)
        dismissive_patterns = [
            r'\bjust\s+(?:do|try|get|go|keep|remember)\b',
            r'\bit\'s\s+(?:just|simply|only)\s+a\b',
            r'\byou\s+(?:just|simply)\s+need\b',
            r'\bwhatever\b', r'\banyway\b',
            r'\bi\s+don\'t\s+know\b', r'\bwho\s+knows\b',
            r'\bprobably\s+won\'t\b', r'\bmight\s+not\b',
        ]
        
        contradiction_count = count_patterns(contradiction_patterns, resp_lower)
        dismissive_count = count_patterns(dismissive_patterns, resp_lower)
        
        # Penalty for contradictions and dismissiveness
        incoherence_penalty = min(contradiction_count * 0.5 + dismissive_count * 0.3, 3.0)
        
        # ============================================================
        # 5. STRUCTURAL SCAFFOLDING
        # Detect framing (intro), development (body), and resolution (conclusion)
        # ============================================================
        
        # Opening framing signals
        opening_patterns = [
            r'^(?:I\s+(?:understand|can\s+see|hear|appreciate|\'m\s+sorry))',
            r'^(?:It\'s\s+(?:completely|totally|absolutely|perfectly))',
            r'^(?:Let\s+me|Let\'s|Here\'s|Imagine)',
            r'^(?:Great\s+question|Good\s+point|That\'s\s+(?:a\s+great|understandable))',
        ]
        
        has_framing = 0
        if sentences:
            first_sent = sentences[0].strip()
            for p in opening_patterns:
                if re.search(p, first_sent, re.IGNORECASE):
                    has_framing = 1
                    break
        
        # Closing/resolution signals
        closing_patterns = [
            r'\bremember\b.*\.$', r'\bin\s+(?:summary|conclusion)\b',
            r'\boverall\b', r'\bultimately\b', r'\bmost\s+importantly\b',
            r'\bdon\'t\s+(?:hesitate|forget|be\s+afraid)\b',
            r'\bfeel\s+free\b', r'\bwe\'re\s+here\b', r'\bwe\s+(?:value|appreciate)\b',
            r'\byou\'ve\s+got\s+this\b', r'\btake\s+(?:care|your\s+time)\b',
        ]
        
        has_resolution = 0
        if len(sentences) >= 2:
            last_two = ' '.join(sentences[-2:]).lower()
            for p in closing_patterns:
                if re.search(p, last_two):
                    has_resolution = 1
                    break
        
        # Detect numbered or explicitly structured lists
        has_numbered_structure = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean)) >= 2
        has_labeled_structure = len(re.findall(r'(?:^|\n)\s*(?:[A-Z][a-z]+\s*:)', response_clean)) >= 2
        
        scaffolding_score = (
            has_framing * 1.0 +
            has_resolution * 1.0 +
            (has_numbered_structure or has_labeled_structure) * 1.0 +
            min(num_sentences / 5.0, 1.0) * 0.5  # Reward sufficient length
        )
        # Max ~3.5
        
        # ============================================================
        # 6. QUERY RESPONSIVENESS
        # Check if the response actually addresses the query's topic
        # ============================================================
        
        query_content = get_content_words(query_clean)
        response_content = get_content_words(response_clean)
        
        if query_content and response_content:
            relevance = len(query_content & response_content) / max(len(query_content), 1)
        else:
            relevance = 0.1
        
        relevance_score = min(relevance * 4.0, 2.5)
        
        # ============================================================
        # 7. EMPATHY / ACKNOWLEDGMENT QUALITY (for emotional queries)
        # ============================================================
        
        emotional_query_signals = [
            r'\bfeel\w*\b', r'\bfrustrat\w*\b', r'\bstress\w*\b', r'\bsad\b',
            r'\bupset\b', r'\bangr\w*\b', r'\bworr\w*\b', r'\banxi\w*\b',
            r'\blonely\b', r'\bdespair\b', r'\bheartbr\w*\b', r'\bdevast\w*\b',
            r'\bexhaust\w*\b', r'\bstruggl\w*\b', r'\bdifficult\w*\b',
        ]
        
        is_emotional_query = sum(1 for p in emotional_query_signals if re.search(p, query_clean.lower())) >= 2
        
        empathy_patterns = [
            r'\bi\s+(?:understand|can\s+see|hear|appreciate|\'m\s+sorry)\b',
            r'\bit\'s\s+(?:completely|totally|absolutely|perfectly)\s+(?:okay|normal|understandable|natural|fine)\b',
            r'\byour\s+feelings\b', r'\bthat\'s\s+(?:tough|hard|difficult|understandable)\b',
            r'\bgive\s+yourself\b', r'\bit\'s\s+okay\b', r'\bit\'s\s+natural\b',
            r'\bpermission\s+to\b',
        ]
        
        empathy_count = count_patterns(empathy_patterns, resp_lower)
        
        empathy_score = 0
        if is_emotional_query:
            empathy_score = min(empathy_count * 0.8, 2.0)
        
        # ============================================================
        # 8. SENTENCE QUALITY DISTRIBUTION
        # Check that sentences aren't all trivially short or all rambling
        # ============================================================
        
        sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        if sent_lengths:
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            if len(sent_lengths) > 1:
                sent_len_var = sum((x - avg_sent_len)**2 for x in sent_lengths) / len(sent_lengths)
                sent_len_std = math.sqrt(sent_len_var)
            else:
                sent_len_std = 0
            
            # Good responses have moderate sentence length (10-25 words) with some variety
            if 8 <= avg_sent_len <= 28:
                length_quality = 1.0
            elif 5 <= avg_sent_len < 8 or 28 < avg_sent_len <= 40:
                length_quality = 0.5
            else:
                length_quality = 0.2
            
            # Some variety is good (std 3-10), too much or too little is bad
            if 2 <= sent_len_std <= 12:
                variety_quality = 0.5
            else:
                variety_quality = 0.2
            
            sentence_quality_score = length_quality + variety_quality
        else:
            sentence_quality_score = 0.3
        # Max ~1.5
        
        # ============================================================
        # COMBINE ALL SCORES
        # ============================================================
        
        raw_score = (
            connective_score * 1.0 +        # Max ~6.0
            threading_score * 0.8 +          # Max ~2.8
            elaboration_score * 0.8 +        # Max ~2.8
            scaffolding_score * 1.0 +        # Max ~3.5
            relevance_score * 0.7 +          # Max ~1.75
            empathy_score * 0.5 +            # Max ~1.0
            sentence_quality_score * 0.6 -   # Max ~0.9
            incoherence_penalty * 0.8        # Max penalty ~2.4
        )
        
        # Normalize to 1-5 scale
        # Theoretical max ~18.75, practical max ~14
        # Theoretical min ~-2.4, practical min ~0
        
        normalized = 1.0 + (raw_score / 14.0) * 4.0
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        return 2