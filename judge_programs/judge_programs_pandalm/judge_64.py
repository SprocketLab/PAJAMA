def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a SENTENCE-LEVEL analysis approach:
    - Analyzes each sentence individually for epistemic stance
    - Scores based on sentence-level claim type classification (fact, opinion, speculation, hedged)
    - Uses query complexity/ambiguity detection to determine expected calibration level
    - Penalizes monotonic epistemic tone (all sentences same confidence level)
    - Rewards appropriate epistemic variety matching query demands
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        if len(response_clean) < 5:
            return 1.0
        
        # Split response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 1.0
        
        # ---- STEP 1: Query ambiguity/complexity detection ----
        # Determine how much epistemic calibration we should expect
        
        ambiguity_indicators = [
            r'\bwhy\b', r'\bexplain\b', r'\bcompare\b', r'\bcontrast\b',
            r'\bopinion\b', r'\bthink\b', r'\bbelieve\b', r'\bshould\b',
            r'\bcould\b', r'\bwould\b', r'\bmight\b', r'\bhypothetical\b',
            r'\bpredict\b', r'\bfuture\b', r'\bimpact\b', r'\beffect\b',
            r'\badvantage\b', r'\bdisadvantage\b', r'\bpros?\b', r'\bcons?\b',
            r'\bbest\b', r'\bworst\b', r'\bmeaning\b', r'\binterpret\b',
            r'\bmetaphor\b', r'\bsymbol\b', r'\bimpl(?:y|ication)\b',
        ]
        
        factual_indicators = [
            r'\bdefine\b', r'\blist\b', r'\bname\b', r'\bdescribe\b',
            r'\bwhat is\b', r'\brewrite\b', r'\bgenerate\b', r'\bcreate\b',
            r'\bprovide\b', r'\bgive\b', r'\bwrite\b', r'\bcome up with\b',
            r'\bcrop\b', r'\bconvert\b', r'\btranslate\b',
        ]
        
        query_lower = query_clean.lower()
        ambiguity_count = sum(1 for p in ambiguity_indicators if re.search(p, query_lower))
        factual_count = sum(1 for p in factual_indicators if re.search(p, query_lower))
        
        # Query ambiguity score: 0 (very factual) to 1 (very ambiguous)
        if ambiguity_count + factual_count == 0:
            query_ambiguity = 0.3
        else:
            query_ambiguity = ambiguity_count / (ambiguity_count + factual_count + 1)
        
        # ---- STEP 2: Classify each sentence by epistemic stance ----
        
        # Definitive/absolute markers
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bcertainly\b', r'\bdefinitely\b',
            r'\bundoubtedly\b', r'\bwithout (?:a )?doubt\b', r'\babsolutely\b',
            r'\bclearly\b', r'\bobviously\b', r'\bof course\b', r'\bno question\b',
            r'\bunquestionably\b', r'\bwithout exception\b', r'\bevery\b',
            r'\bnone\b', r'\ball\b(?! )', r'\bthe fact is\b', r'\bit is certain\b',
            r'\bguaranteed\b', r'\bproven\b', r'\bundeniable\b',
        ]
        
        # Hedging/uncertainty markers  
        hedge_patterns = [
            r'\bperhaps\b', r'\bmaybe\b', r'\bpossibly\b', r'\bprobably\b',
            r'\blikely\b', r'\bunlikely\b', r'\bmight\b', r'\bcould\b',
            r'\bmay\b', r'\btend(?:s)? to\b', r'\bgenerally\b', r'\btypically\b',
            r'\busually\b', r'\boften\b', r'\bsometimes\b', r'\bin some cases\b',
            r'\bit (?:seems?|appears?)\b', r'\bsuggests?\b', r'\bimplies?\b',
            r'\bresearch suggests\b', r'\bstudies (?:show|suggest|indicate)\b',
            r'\baccording to\b', r'\bin general\b', r'\bon average\b',
            r'\bto some (?:extent|degree)\b', r'\barguably\b',
            r'\bit is (?:thought|believed|considered)\b',
            r'\bone (?:could|might|may) argue\b',
        ]
        
        # Evidential markers (citing sources/reasoning)
        evidential_patterns = [
            r'\bbecause\b', r'\bdue to\b', r'\bas a result\b', r'\btherefore\b',
            r'\bconsequently\b', r'\bfor (?:example|instance)\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bevidence\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bdata\b', r'\bexpert\b', r'\bscientist\b',
        ]
        
        # Opinion/subjective markers
        opinion_patterns = [
            r'\bi think\b', r'\bi believe\b', r'\bin my (?:opinion|view)\b',
            r'\bpersonally\b', r'\bfrom my perspective\b', r'\bi feel\b',
            r'\bone (?:perspective|view|interpretation)\b',
            r'\bsome (?:people|experts|researchers) (?:think|believe|argue)\b',
            r'\bdebat(?:e|able|ed)\b', r'\bcontrovers(?:y|ial)\b',
        ]
        
        sentence_classifications = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            abs_count = sum(1 for p in absolute_patterns if re.search(p, sent_lower))
            hedge_count = sum(1 for p in hedge_patterns if re.search(p, sent_lower))
            evid_count = sum(1 for p in evidential_patterns if re.search(p, sent_lower))
            opinion_count = sum(1 for p in opinion_patterns if re.search(p, sent_lower))
            
            # Classify sentence
            total = abs_count + hedge_count + evid_count + opinion_count
            if total == 0:
                classification = 'neutral'
            elif abs_count > hedge_count and abs_count >= evid_count:
                classification = 'absolute'
            elif hedge_count > 0 and hedge_count >= abs_count:
                classification = 'hedged'
            elif opinion_count > 0:
                classification = 'opinion'
            elif evid_count > 0:
                classification = 'evidential'
            else:
                classification = 'neutral'
            
            sentence_classifications.append({
                'class': classification,
                'abs': abs_count,
                'hedge': hedge_count,
                'evid': evid_count,
                'opinion': opinion_count,
                'length': len(sent.split())
            })
        
        # ---- STEP 3: Compute epistemic profile scores ----
        
        n_sent = len(sentence_classifications)
        class_counts = Counter(s['class'] for s in sentence_classifications)
        
        # 3a. Epistemic diversity score (variety of stances)
        n_classes_used = len([c for c in class_counts if class_counts[c] > 0])
        max_possible_classes = min(5, n_sent)
        if max_possible_classes > 0:
            diversity_score = n_classes_used / max_possible_classes
        else:
            diversity_score = 0
        
        # 3b. Hedging appropriateness
        hedge_ratio = class_counts.get('hedged', 0) / n_sent if n_sent > 0 else 0
        absolute_ratio = class_counts.get('absolute', 0) / n_sent if n_sent > 0 else 0
        evidential_ratio = class_counts.get('evidential', 0) / n_sent if n_sent > 0 else 0
        
        # For ambiguous queries, we want more hedging; for factual, less is ok
        if query_ambiguity > 0.4:
            # Ambiguous query: reward hedging, penalize absolutes
            hedge_appropriateness = hedge_ratio * 2.0 - absolute_ratio * 1.5
        else:
            # Factual query: slight penalty for excessive hedging, mild penalty for absolutes
            hedge_appropriateness = min(hedge_ratio, 0.3) - absolute_ratio * 0.5
        
        hedge_appropriateness = max(-1.0, min(1.0, hedge_appropriateness))
        
        # 3c. Overconfidence penalty
        # Count sentences that make broad claims with absolute language
        overconfident_sentences = 0
        for s in sentence_classifications:
            if s['abs'] > 0 and s['hedge'] == 0 and s['evid'] == 0:
                overconfident_sentences += 1
        
        overconfidence_penalty = overconfident_sentences / n_sent if n_sent > 0 else 0
        
        # 3d. Evidence-based reasoning bonus
        evidence_bonus = min(evidential_ratio * 1.5, 0.5)
        
        # ---- STEP 4: Response quality basics ----
        
        response_words = response_clean.split()
        n_words = len(response_words)
        
        # Length adequacy (not too short, not excessively long)
        if n_words < 10:
            length_score = n_words / 10.0
        elif n_words < 20:
            length_score = 0.6 + 0.4 * (n_words - 10) / 10
        elif n_words <= 200:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (n_words - 200) / 500)
        
        # Repetition detection at word level (bigrams)
        if n_words >= 4:
            bigrams = [f"{response_words[i].lower()} {response_words[i+1].lower()}" for i in range(n_words - 1)]
            bigram_counts = Counter(bigrams)
            if len(bigrams) > 0:
                max_bigram_freq = max(bigram_counts.values())
                repetition_ratio = max_bigram_freq / len(bigrams)
                repetition_penalty = max(0, repetition_ratio - 0.1) * 3
            else:
                repetition_penalty = 0
        else:
            repetition_penalty = 0
        
        # Sentence-level repetition
        sent_texts = [s.lower().strip() for s in sentences]
        unique_sents = len(set(sent_texts))
        if len(sent_texts) > 0:
            sent_repetition_penalty = max(0, 1.0 - unique_sents / len(sent_texts)) * 2
        else:
            sent_repetition_penalty = 0
        
        # ---- STEP 5: Structural coherence ----
        
        # Check for logical connectors between sentences
        connector_patterns = [
            r'^(?:however|moreover|furthermore|additionally|in addition|'
            r'on the other hand|conversely|nevertheless|nonetheless|'
            r'in contrast|similarly|likewise|consequently|therefore|'
            r'as a result|for this reason|meanwhile|subsequently|'
            r'first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|also|'
            r'while|although|though|yet|but|and|or)\b'
        ]
        
        connector_count = 0
        for sent in sentences[1:]:  # Skip first sentence
            sent_lower = sent.lower().strip()
            for p in connector_patterns:
                if re.search(p, sent_lower):
                    connector_count += 1
                    break
        
        if n_sent > 1:
            coherence_score = min(connector_count / (n_sent - 1), 1.0)
        else:
            coherence_score = 0.5
        
        # ---- STEP 6: Information density ----
        # Unique content words ratio
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
            'or', 'if', 'while', 'that', 'this', 'it', 'its', 'they', 'them',
            'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
        }
        
        content_words = [w.lower().strip('.,!?;:"\'-()[]') for w in response_words 
                        if w.lower().strip('.,!?;:"\'-()[]') not in stop_words 
                        and len(w.strip('.,!?;:"\'-()[]')) > 1]
        
        if n_words > 0 and len(content_words) > 0:
            unique_content = len(set(content_words))
            info_density = unique_content / n_words
            content_diversity = unique_content / len(content_words) if content_words else 0
        else:
            info_density = 0
            content_diversity = 0
        
        # ---- STEP 7: Query-response relevance via shared content words ----
        query_words = query_clean.split()
        query_content = set(w.lower().strip('.,!?;:"\'-()[]') for w in query_words 
                          if w.lower().strip('.,!?;:"\'-()[]') not in stop_words
                          and len(w.strip('.,!?;:"\'-()[]')) > 1)
        response_content = set(content_words)
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        # ---- STEP 8: Composite scoring ----
        
        # Base quality score (0-4)
        base_quality = (
            length_score * 1.5 +
            coherence_score * 0.8 +
            min(info_density * 4, 1.0) * 0.8 +
            content_diversity * 0.9
        )
        
        # Epistemic calibration score (0-3)
        epistemic_score = (
            diversity_score * 0.6 +
            hedge_appropriateness * 1.0 +
            evidence_bonus * 0.8 +
            (1.0 - overconfidence_penalty) * 0.6
        )
        
        # Relevance component (0-2)
        relevance_score = relevance * 2.0
        
        # Penalties
        total_penalty = repetition_penalty + sent_repetition_penalty
        
        # Final score
        raw_score = base_quality + epistemic_score + relevance_score - total_penalty
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        return round(final_score, 3)
        
    except Exception:
        try:
            if response and len(response.strip()) > 10:
                return 3.0
            return 1.0
        except Exception:
            return 1.0