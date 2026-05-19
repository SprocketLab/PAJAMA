def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a DIFFERENT approach: sentence-level analysis of claim types,
    modal verb usage patterns, evidential markers, and a novel "assertion density" metric
    that measures the ratio of unsupported definitive claims to total claims.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            return 1.0
        
        query = query.strip() if query else ""
        
        # Split into sentences for sentence-level analysis
        sentences = re.split(r'(?<=[.!?])\s+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response.lower())
        num_words = max(len(words), 1)
        
        # ============================================================
        # FEATURE 1: Modal verb spectrum analysis
        # Categorize modals by strength and compute a "modal balance" score
        # ============================================================
        
        weak_modals = ['might', 'could', 'may', 'would']
        medium_modals = ['should', 'can', 'ought']
        strong_modals = ['must', 'shall', 'will', 'need']
        
        weak_count = sum(1 for w in words if w in weak_modals)
        medium_count = sum(1 for w in words if w in medium_modals)
        strong_count = sum(1 for w in words if w in strong_modals)
        
        total_modals = weak_count + medium_count + strong_count
        
        # Modal diversity: having a mix of modal strengths is good
        modal_types_present = sum([weak_count > 0, medium_count > 0, strong_count > 0])
        modal_diversity_score = modal_types_present / 3.0 if total_modals > 0 else 0.0
        
        # ============================================================
        # FEATURE 2: Evidential markers - sources of knowledge
        # These indicate WHERE knowledge comes from
        # ============================================================
        
        evidential_patterns = [
            r'\baccording to\b', r'\bresearch\s+(suggests?|shows?|indicates?|finds?)\b',
            r'\bstudies\s+(suggest|show|indicate|find|have\s+found)\b',
            r'\bevidence\s+(suggests?|shows?|indicates?)\b',
            r'\bdata\s+(suggests?|shows?|indicates?)\b',
            r'\bexperts?\s+(say|believe|suggest|argue|note)\b',
            r'\bit\s+is\s+(widely|generally|commonly)\s+(accepted|believed|understood|known)\b',
            r'\bhistorically\b', r'\btraditionally\b',
            r'\bin\s+(theory|practice|principle)\b',
            r'\bfrom\s+(a|the)\s+\w+\s+perspective\b',
            r'\bscientific(ally)?\b', r'\bempirical(ly)?\b',
        ]
        
        response_lower = response.lower()
        evidential_count = 0
        for pattern in evidential_patterns:
            evidential_count += len(re.findall(pattern, response_lower))
        
        evidential_density = evidential_count / num_sentences
        
        # ============================================================
        # FEATURE 3: Assertion density - ratio of "bare assertions" 
        # (sentences with no qualification) to total sentences
        # This is the NOVEL metric for this variant
        # ============================================================
        
        qualification_indicators = [
            r'\b(might|could|may|possibly|perhaps|likely|unlikely|probably|arguably)\b',
            r'\b(sometimes|often|usually|generally|typically|tend[s]?\s+to)\b',
            r'\b(some|many|most|few|certain|several)\b',
            r'\b(appear[s]?|seem[s]?)\b',
            r'\b(in\s+some\s+cases|in\s+many\s+cases|in\s+general)\b',
            r'\b(it\s+depends|depending\s+on)\b',
            r'\b(one\s+(could|might|may)\s+argue)\b',
            r'\b(to\s+some\s+(extent|degree))\b',
            r'\b(not\s+necessarily|not\s+always)\b',
            r'\b(can\s+be|could\s+be|may\s+be|might\s+be)\b',
        ]
        
        # Definitive/absolute indicators in a sentence
        absolute_indicators = [
            r'\b(always|never|every|all|none|no\s+one|everyone|everything|nothing)\b',
            r'\b(certainly|definitely|absolutely|undoubtedly|unquestionably)\b',
            r'\b(clearly|obviously|evidently|plainly)\b',
            r'\b(the\s+fact\s+(is|that))\b',
            r'\b(without\s+(a\s+)?doubt)\b',
            r'\b(it\s+is\s+(clear|obvious|evident|certain))\b',
            r'\b(there\s+is\s+no\s+(question|doubt))\b',
            r'\b(proven|guaranteed|impossible)\b',
        ]
        
        bare_assertion_count = 0
        qualified_count = 0
        overconfident_count = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            has_qualification = any(re.search(p, sent_lower) for p in qualification_indicators)
            has_absolute = any(re.search(p, sent_lower) for p in absolute_indicators)
            
            if has_absolute:
                overconfident_count += 1
            
            if has_qualification:
                qualified_count += 1
            elif not has_absolute and len(sent.split()) > 5:
                # It's a bare assertion (no qualification, no absolute, substantive length)
                bare_assertion_count += 1
        
        bare_assertion_ratio = bare_assertion_count / num_sentences
        overconfident_ratio = overconfident_count / num_sentences
        qualified_ratio = qualified_count / num_sentences
        
        # ============================================================
        # FEATURE 4: Discourse-level epistemic structure
        # Does the response acknowledge complexity/nuance?
        # ============================================================
        
        complexity_markers = [
            r'\bhowever\b', r'\bon\s+the\s+other\s+hand\b', r'\balthough\b',
            r'\bwhile\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bthat\s+said\b', r'\bconversely\b', r'\bin\s+contrast\b',
            r'\bdepends\s+on\b', r'\bvaries\b', r'\bcomplex\b',
            r'\bnuance[ds]?\b', r'\bmultifaceted\b', r'\bdiverse\b',
            r'\bboth\b.*\band\b', r'\bnot\s+only\b.*\bbut\s+also\b',
        ]
        
        complexity_count = sum(1 for p in complexity_markers if re.search(p, response_lower))
        complexity_density = complexity_count / num_sentences
        
        # ============================================================
        # FEATURE 5: Query-type sensitivity
        # Detect if the query is about uncertain/debatable topics
        # ============================================================
        
        query_lower = query.lower()
        
        uncertain_topic_signals = [
            r'\bwhy\s+(do|does|did|is|are|was|were)\b',
            r'\bwhat\s+(do\s+you\s+think|is\s+your\s+opinion|are\s+the\s+pros)\b',
            r'\bshould\b', r'\bbest\b', r'\bworst\b',
            r'\bcompare\b', r'\bcontrast\b', r'\bdebate\b',
            r'\bcontroversial\b', r'\bopinion\b', r'\bbelieve\b',
            r'\bpredict\b', r'\bfuture\b', r'\bhypothetical\b',
            r'\bexplain\b', r'\bdescribe\b', r'\banalyze\b',
        ]
        
        factual_topic_signals = [
            r'\bdefine\b', r'\blist\b', r'\bname\b',
            r'\bwhat\s+is\b', r'\bwho\s+(is|was)\b',
            r'\bwhen\s+(did|was|is)\b', r'\bwhere\s+(is|was|did)\b',
            r'\bhow\s+many\b', r'\bhow\s+much\b',
            r'\brewrite\b', r'\bgenerate\b', r'\bcreate\b',
            r'\bprovide\b', r'\bgive\b',
        ]
        
        uncertain_query = sum(1 for p in uncertain_topic_signals if re.search(p, query_lower))
        factual_query = sum(1 for p in factual_topic_signals if re.search(p, query_lower))
        
        is_uncertain_topic = uncertain_query > factual_query
        
        # ============================================================
        # FEATURE 6: Response completeness and structure
        # (since examples show longer, more detailed responses winning)
        # ============================================================
        
        # Sentence count score (more sentences = more thorough, up to a point)
        sentence_score = min(num_sentences / 4.0, 1.0)  # Plateau at 4 sentences
        
        # Word count score
        word_score = min(num_words / 40.0, 1.0)  # Plateau at 40 words
        
        # Unique word ratio (vocabulary richness)
        unique_words = len(set(words))
        vocab_richness = unique_words / num_words if num_words > 0 else 0
        
        # ============================================================
        # FEATURE 7: Repetition penalty
        # Detect degenerate repetitive outputs
        # ============================================================
        
        # Check for repeated phrases (3-grams)
        if num_words >= 3:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            max_trigram_freq = max(trigram_counts.values()) if trigram_counts else 0
            repetition_ratio = max_trigram_freq / max(len(trigrams), 1)
        else:
            repetition_ratio = 0
        
        repetition_penalty = max(0, (repetition_ratio - 0.1) * 5)  # Penalty kicks in above 10%
        
        # ============================================================
        # FEATURE 8: Explanatory depth
        # Does the response explain WHY or just state WHAT?
        # ============================================================
        
        explanatory_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bas\s+a\s+result\b', r'\bdue\s+to\b', r'\bthis\s+(means|implies|suggests)\b',
            r'\bin\s+order\s+to\b', r'\bso\s+that\b', r'\bwhich\s+(means|allows|enables)\b',
            r'\bfor\s+(example|instance)\b', r'\bsuch\s+as\b', r'\blike\b',
            r'\bspecifically\b', r'\bin\s+particular\b',
        ]
        
        explanatory_count = sum(1 for p in explanatory_markers if re.search(p, response_lower))
        explanatory_density = explanatory_count / num_sentences
        
        # ============================================================
        # SCORING FORMULA
        # ============================================================
        
        score = 50.0  # Base score
        
        # Epistemic calibration components
        # Reward appropriate hedging/qualification (scaled by topic uncertainty)
        if is_uncertain_topic:
            # For uncertain topics, qualification is more important
            score += qualified_ratio * 12.0
            score -= overconfident_ratio * 15.0
            score += evidential_density * 8.0
            score += complexity_density * 8.0
            score += modal_diversity_score * 5.0
        else:
            # For factual topics, some confidence is appropriate
            score += qualified_ratio * 5.0
            score -= overconfident_ratio * 8.0
            score += evidential_density * 4.0
            score += complexity_density * 4.0
            score += modal_diversity_score * 3.0
        
        # Penalize high bare assertion ratio (especially for complex topics)
        if is_uncertain_topic:
            score -= bare_assertion_ratio * 6.0
        else:
            score -= bare_assertion_ratio * 2.0
        
        # Content quality components
        score += sentence_score * 10.0       # Reward sufficient length
        score += word_score * 8.0            # Reward sufficient detail
        score += vocab_richness * 8.0        # Reward vocabulary diversity
        score += explanatory_density * 6.0   # Reward explanatory depth
        
        # Penalties
        score -= repetition_penalty * 15.0   # Heavy penalty for repetition
        
        # Bonus for having both hedging AND definitive claims (shows discrimination)
        if qualified_count > 0 and bare_assertion_count > 0 and overconfident_count == 0:
            score += 3.0  # Balanced epistemic stance
        
        # Penalty for very short responses (likely low quality)
        if num_words < 10:
            score -= 10.0
        
        # Penalty for empty/near-empty
        if num_words < 3:
            score -= 20.0
        
        # Clamp to reasonable range
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception:
        return 25.0