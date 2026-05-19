def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses sentence-level analysis with a state machine approach:
    - Classifies each sentence by its epistemic stance (assertive, hedged, speculative, uncertain)
    - Evaluates appropriateness of epistemic stance given the query type
    - Measures the diversity and sophistication of uncertainty language
    - Penalizes structural/formatting issues that indicate low quality
    
    Different from Variant 1 (word overlap, confidence markers) by using:
    - Sentence-level classification rather than word-level counting
    - Query-type detection to calibrate expected uncertainty levels
    - Information density and coherence metrics
    - Sophistication scoring of epistemic language patterns
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # ============================================================
        # STEP 1: Query type classification
        # ============================================================
        query_lower = query.lower().strip()
        
        # Factual queries expect more certain answers (but appropriate hedging on ambiguous ones)
        factual_indicators = [
            r'\bhow many\b', r'\bwhat is\b', r'\bwhat was\b', r'\bwhat are\b',
            r'\bwho is\b', r'\bwho was\b', r'\bwhere is\b', r'\bwhere did\b',
            r'\bwhen did\b', r'\bwhen was\b', r'\bidentify\b', r'\bname\b',
            r'\blist\b', r'\bcreate\b', r'\brewrite\b', r'\bwrite\b'
        ]
        
        opinion_indicators = [
            r'\bis it ok\b', r'\bshould i\b', r'\bwhat do you think\b',
            r'\bopinion\b', r'\badvice\b', r'\brecommend\b', r'\bdo you believe\b',
            r'\bis it possible\b', r'\bcan you suggest\b'
        ]
        
        ambiguous_indicators = [
            r'\bwhy\b', r'\bhow does\b', r'\bexplain\b', r'\bwhat causes\b',
            r'\bhistory of\b', r'\bmore about\b', r'\bcan you\b.*\bmore\b'
        ]
        
        task_indicators = [
            r'\brewrite\b', r'\bcreate\b', r'\bgenerate\b', r'\bmake\b',
            r'\bwrite\b', r'\btranslate\b', r'\bconvert\b', r'\bsummarize\b',
            r'\bregenerate\b', r'\bremove\b', r'\bshorten\b'
        ]
        
        query_type_scores = {'factual': 0, 'opinion': 0, 'ambiguous': 0, 'task': 0}
        for pat in factual_indicators:
            if re.search(pat, query_lower):
                query_type_scores['factual'] += 1
        for pat in opinion_indicators:
            if re.search(pat, query_lower):
                query_type_scores['opinion'] += 1
        for pat in ambiguous_indicators:
            if re.search(pat, query_lower):
                query_type_scores['ambiguous'] += 1
        for pat in task_indicators:
            if re.search(pat, query_lower):
                query_type_scores['task'] += 1
        
        query_type = max(query_type_scores, key=query_type_scores.get)
        if all(v == 0 for v in query_type_scores.values()):
            query_type = 'general'
        
        # ============================================================
        # STEP 2: Sentence segmentation and classification
        # ============================================================
        # Split into sentences more carefully
        sentence_endings = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 2]
        
        if not sentences:
            sentences = [response_stripped]
        
        # Epistemic stance patterns for sentence-level classification
        strong_hedge_patterns = [
            r'\bit is difficult to\b', r'\bit\'s hard to\b', r'\bnot entirely clear\b',
            r'\bremains uncertain\b', r'\bno definitive\b', r'\bsubject to debate\b',
            r'\bcan be subjective\b', r'\bvary depending\b', r'\bdepends on\b',
            r'\bnot without controversy\b', r'\bhas been criticized\b',
            r'\bopen question\b', r'\bunclear whether\b'
        ]
        
        moderate_hedge_patterns = [
            r'\blikely\b', r'\bprobably\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bappears to\b',
            r'\bseems to\b', r'\btends to\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\bsometimes\b', r'\bin some cases\b',
            r'\bto some extent\b', r'\brelatively\b', r'\bapproximately\b'
        ]
        
        evidence_patterns = [
            r'\bresearch suggests\b', r'\bstudies show\b', r'\baccording to\b',
            r'\bevidence indicates\b', r'\bdata suggests\b', r'\bexperts\b',
            r'\bscholars\b', r'\bhistorians\b', r'\bis considered\b',
            r'\bis known as\b', r'\bis also known\b', r'\bis often\b',
            r'\bwidely regarded\b', r'\bgenerally accepted\b'
        ]
        
        overconfident_patterns = [
            r'\bdefinitely\b', r'\babsolutely\b', r'\bwithout a doubt\b',
            r'\bundeniably\b', r'\bunquestionably\b', r'\bclearly\b.*\bobvious\b',
            r'\beveryone knows\b', r'\bit is a fact that\b', r'\bthe truth is\b',
            r'\bobviously\b', r'\bof course\b'
        ]
        
        # Classify each sentence
        sentence_classes = []
        for sent in sentences:
            sent_lower = sent.lower()
            cls = {
                'strong_hedge': 0,
                'moderate_hedge': 0,
                'evidence_based': 0,
                'overconfident': 0,
                'assertive': 0,
                'length': len(sent)
            }
            
            for pat in strong_hedge_patterns:
                if re.search(pat, sent_lower):
                    cls['strong_hedge'] += 1
            for pat in moderate_hedge_patterns:
                if re.search(pat, sent_lower):
                    cls['moderate_hedge'] += 1
            for pat in evidence_patterns:
                if re.search(pat, sent_lower):
                    cls['evidence_based'] += 1
            for pat in overconfident_patterns:
                if re.search(pat, sent_lower):
                    cls['overconfident'] += 1
            
            # If no epistemic markers, it's assertive
            total_markers = sum([cls['strong_hedge'], cls['moderate_hedge'], 
                               cls['evidence_based'], cls['overconfident']])
            if total_markers == 0:
                cls['assertive'] = 1
            
            sentence_classes.append(cls)
        
        # ============================================================
        # STEP 3: Compute epistemic calibration score
        # ============================================================
        num_sentences = len(sentence_classes)
        
        total_strong_hedge = sum(s['strong_hedge'] for s in sentence_classes)
        total_moderate_hedge = sum(s['moderate_hedge'] for s in sentence_classes)
        total_evidence = sum(s['evidence_based'] for s in sentence_classes)
        total_overconfident = sum(s['overconfident'] for s in sentence_classes)
        total_assertive = sum(s['assertive'] for s in sentence_classes)
        
        # Epistemic diversity: how many different types of epistemic stances are used
        stance_types_used = sum([
            1 if total_strong_hedge > 0 else 0,
            1 if total_moderate_hedge > 0 else 0,
            1 if total_evidence > 0 else 0,
            1 if total_assertive > 0 else 0
        ])
        
        epistemic_diversity_score = stance_types_used / 4.0  # 0 to 1
        
        # Hedging ratio
        total_epistemic = (total_strong_hedge + total_moderate_hedge + 
                          total_evidence + total_overconfident + total_assertive)
        if total_epistemic > 0:
            hedge_ratio = (total_strong_hedge + total_moderate_hedge + total_evidence) / total_epistemic
            overconfidence_ratio = total_overconfident / total_epistemic
        else:
            hedge_ratio = 0
            overconfidence_ratio = 0
        
        # ============================================================
        # STEP 4: Response quality fundamentals (needed for baseline)
        # ============================================================
        response_lower = response_stripped.lower()
        words = response_lower.split()
        word_count = len(words)
        
        # Minimum viable response check
        if word_count <= 1:
            return 0.5
        if word_count <= 3:
            return 1.5
        
        # Unique word ratio (vocabulary richness)
        unique_words = len(set(words))
        vocab_richness = unique_words / word_count if word_count > 0 else 0
        
        # Repetition detection at sentence level
        sentence_texts = [s.lower().strip() for s in sentences]
        unique_sentences = len(set(sentence_texts))
        sentence_repetition = 1.0 - (unique_sentences / len(sentence_texts)) if len(sentence_texts) > 0 else 0
        
        # Detect garbage/irrelevant content
        garbage_indicators = 0
        # HTML in non-HTML queries
        if '<' in response and '>' in response and 'html' not in query_lower and 'tag' not in query_lower:
            html_tags = re.findall(r'<[^>]+>', response)
            if len(html_tags) > 3:
                garbage_indicators += 1
        
        # Code blocks in non-code queries
        if 'import ' in response and 'def ' in response and 'code' not in query_lower and 'python' not in query_lower and 'program' not in query_lower:
            garbage_indicators += 1
        
        # Excessive repetition of query
        if query.strip().lower() in response_lower and response_lower.count(query.strip().lower()) > 2:
            garbage_indicators += 1
        
        # "Input:" / "Output:" spam
        io_pattern_count = len(re.findall(r'\b(?:input|output)\s*:', response_lower))
        if io_pattern_count > 4:
            garbage_indicators += 1
        
        # Question-answer spam pattern
        qa_spam = len(re.findall(r'\bquestion\s*:', response_lower))
        if qa_spam > 2:
            garbage_indicators += 1
        
        # ============================================================
        # STEP 5: Information density score
        # ============================================================
        # Count substantive content words (not stopwords, not filler)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'it',
            'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom'
        }
        
        content_words = [w for w in words if w.strip('.,!?;:()[]{}"\'-') not in stopwords and len(w) > 2]
        content_ratio = len(content_words) / word_count if word_count > 0 else 0
        
        # ============================================================
        # STEP 6: Coherence and relevance estimation
        # ============================================================
        # Simple relevance: overlap of content words with query
        query_words = set(query_lower.split())
        query_content = {w.strip('.,!?;:()[]{}"\'-') for w in query_words if w.strip('.,!?;:()[]{}"\'-') not in stopwords and len(w) > 2}
        response_content = set(content_words)
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5  # neutral if query has no content words
        
        # ============================================================
        # STEP 7: Compute composite score
        # ============================================================
        
        # Base score from response length (logarithmic, diminishing returns)
        if word_count <= 5:
            length_score = word_count / 5.0 * 2.0  # 0-2
        elif word_count <= 50:
            length_score = 2.0 + math.log(word_count / 5.0) * 1.5  # ~2-5.5
        elif word_count <= 200:
            length_score = 5.5 + math.log(word_count / 50.0) * 0.8  # ~5.5-6.6
        else:
            length_score = 6.6 + min(math.log(word_count / 200.0) * 0.3, 0.4)  # cap ~7
        
        # Clamp length score
        length_score = min(length_score, 7.0)
        
        # Epistemic calibration bonus/penalty
        epistemic_score = 0.0
        
        # Reward appropriate hedging
        if query_type in ('ambiguous', 'opinion', 'general'):
            # For ambiguous/opinion queries, hedging is very valuable
            epistemic_score += hedge_ratio * 1.5
            epistemic_score += epistemic_diversity_score * 0.8
            epistemic_score -= overconfidence_ratio * 1.5
        elif query_type == 'factual':
            # For factual queries, some hedging is fine, too much is wishy-washy
            if hedge_ratio < 0.5:
                epistemic_score += hedge_ratio * 0.5
            else:
                epistemic_score += 0.25 - (hedge_ratio - 0.5) * 0.3
            epistemic_score += total_evidence * 0.2
            epistemic_score -= overconfidence_ratio * 1.0
        elif query_type == 'task':
            # For task queries, hedging matters less; execution matters more
            epistemic_score += 0.1  # small baseline
            epistemic_score -= overconfidence_ratio * 0.5
        
        # Clamp epistemic score
        epistemic_score = max(-2.0, min(epistemic_score, 2.5))
        
        # Quality penalties
        penalty = 0.0
        
        # Garbage content penalty
        penalty += garbage_indicators * 2.0
        
        # Repetition penalty
        penalty += sentence_repetition * 3.0
        
        # Very low vocabulary richness penalty
        if vocab_richness < 0.3 and word_count > 20:
            penalty += (0.3 - vocab_richness) * 5.0
        
        # Low content ratio penalty
        if content_ratio < 0.25:
            penalty += (0.25 - content_ratio) * 4.0
        
        # Relevance bonus
        relevance_bonus = relevance * 1.5
        
        # Content richness bonus
        content_bonus = min(content_ratio * 1.5, 1.0)
        
        # Final composite
        score = length_score + epistemic_score + relevance_bonus + content_bonus - penalty
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        # Additional edge case: single word or very terse non-task responses
        if word_count <= 2 and query_type != 'task':
            score = min(score, 2.0)
        
        # Responses that are just "no" or "yes" with nothing else
        if response_stripped.lower().rstrip('.!') in ('no', 'yes', 'n/a', 'none'):
            score = min(score, 1.0)
        
        # Single period or empty-ish
        if response_stripped in ('.', '..', '...', '-', '--'):
            return 0.0
        
        return round(score, 2)
        
    except Exception as e:
        # Fallback: return a middling score based on length
        try:
            if response and len(response.strip()) > 0:
                wc = len(response.strip().split())
                return min(5.0, max(1.0, math.log(wc + 1) * 1.5))
            return 0.0
        except:
            return 0.0