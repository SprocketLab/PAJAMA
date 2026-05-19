def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a novel approach based on:
    1. Claim density analysis (ratio of declarative statements to total content)
    2. Evidential reasoning markers (references to sources, reasoning chains)
    3. Proportional uncertainty matching (does uncertainty level match topic ambiguity?)
    4. Assertion-to-qualification ratio
    5. Structural completeness and coherence signals
    
    This differs from previous variants by focusing on sentence-level claim analysis,
    topic ambiguity detection, and proportional calibration scoring rather than
    simple keyword counting.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response_stripped = response.strip()
        if len(response_stripped) < 2:
            return 0.0
        
        query_lower = query.lower().strip()
        response_lower = response.lower()
        
        # === 1. TOPIC AMBIGUITY DETECTION ===
        # Determine if the query is about something ambiguous/uncertain vs factual/concrete
        ambiguity_signals = [
            r'\bhow many\b', r'\bwhat is the (exact|precise)\b', r'\bis it (ok|okay|true|possible)\b',
            r'\bshould\b', r'\bcould\b', r'\bwhy\b', r'\bbelieve\b', r'\bopinion\b',
            r'\bwhat do you think\b', r'\bhistory of\b', r'\bexplain\b', r'\bdescribe\b',
            r'\bwhat was\b', r'\bwhere did\b', r'\brecent\b', r'\bcontrovers\b',
            r'\bdebat\b', r'\bcomplex\b', r'\bdifficult\b'
        ]
        factual_query_signals = [
            r'\bcreate\b', r'\bwrite\b', r'\brewrite\b', r'\bidentify\b', r'\blist\b',
            r'\bgenerate\b', r'\bmake\b', r'\btranslate\b', r'\bconvert\b',
            r'\bhtml\b', r'\bcode\b', r'\bformat\b'
        ]
        
        ambiguity_score = sum(1 for p in ambiguity_signals if re.search(p, query_lower))
        factual_score = sum(1 for p in factual_query_signals if re.search(p, query_lower))
        
        topic_ambiguity = min(1.0, ambiguity_score * 0.2) - min(0.5, factual_score * 0.15)
        topic_ambiguity = max(0.0, min(1.0, topic_ambiguity + 0.3))  # baseline ambiguity
        
        # === 2. SENTENCE-LEVEL CLAIM ANALYSIS ===
        # Split into sentences and analyze each one
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(1, len(sentences))
        
        # Absolute/definitive claim patterns (per sentence)
        absolute_patterns = [
            r'\b(always|never|certainly|definitely|absolutely|undoubtedly|without doubt)\b',
            r'\b(clearly|obviously|of course|everyone knows|it is certain)\b',
            r'\b(the fact is|the truth is|there is no doubt|unquestionably)\b',
            r'\b(proven|guaranteed|100%|impossible|must be)\b',
            r'\b(no one|everyone|all people|nobody)\b',
        ]
        
        # Qualified/hedged claim patterns (per sentence)
        qualified_patterns = [
            r'\b(may|might|could|possibly|perhaps|arguably)\b',
            r'\b(likely|unlikely|probably|presumably|conceivably)\b',
            r'\b(suggests?|indicates?|implies?|appears?|seems?)\b',
            r'\b(research suggests|studies show|evidence indicates|some believe)\b',
            r'\b(it is (thought|believed|estimated|considered))\b',
            r'\b(in (some|many|most) cases)\b',
            r'\b(tend to|generally|typically|often|sometimes|usually)\b',
            r'\b(approximately|roughly|around|about|estimated)\b',
            r'\b(according to|based on|from what)\b',
            r'\b(one (possibility|interpretation|view|perspective))\b',
            r'\b(it depends|varies|subjective|debat)\b',
            r'\b(however|although|nevertheless|on the other hand|that said)\b',
            r'\b(not (entirely |completely |fully )?(clear|certain|known|established))\b',
            r'\b(difficult to (say|determine|know|measure|quantify|provide))\b',
        ]
        
        # Evidential/reasoning patterns
        evidential_patterns = [
            r'\b(because|since|therefore|thus|hence|consequently)\b',
            r'\b(for (example|instance))\b',
            r'\b(such as|including|like)\b',
            r'\b(this (means|suggests|indicates|implies))\b',
            r'\b(reason|evidence|data|study|research|source)\b',
            r'\b(according to|cited|reference|documented)\b',
            r'\b(known (as|for))\b',
            r'\b(considered|regarded|recognized)\b',
        ]
        
        absolute_count = 0
        qualified_count = 0
        evidential_count = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            for p in absolute_patterns:
                if re.search(p, sent_lower):
                    absolute_count += 1
                    break
            for p in qualified_patterns:
                if re.search(p, sent_lower):
                    qualified_count += 1
                    break
            for p in evidential_patterns:
                if re.search(p, sent_lower):
                    evidential_count += 1
                    break
        
        # Rates
        absolute_rate = absolute_count / num_sentences
        qualified_rate = qualified_count / num_sentences
        evidential_rate = evidential_count / num_sentences
        
        # === 3. ASSERTION-TO-QUALIFICATION RATIO ===
        # Count raw assertion markers vs qualification markers across full text
        assertion_words = re.findall(
            r'\b(is|are|was|were|will|has|have|had|does|do|did)\b', response_lower
        )
        qualification_words = re.findall(
            r'\b(may|might|could|possibly|perhaps|likely|probably|suggests?|seems?|appears?|'
            r'generally|typically|often|sometimes|arguably|approximately|roughly|tend)\b',
            response_lower
        )
        
        num_assertion = len(assertion_words)
        num_qualification = len(qualification_words)
        
        if num_assertion + num_qualification > 0:
            qualification_ratio = num_qualification / (num_assertion + num_qualification)
        else:
            qualification_ratio = 0.0
        
        # === 4. STRUCTURAL COMPLETENESS ===
        words = response_stripped.split()
        num_words = len(words)
        
        # Very short responses are usually low quality
        if num_words < 3:
            length_score = 0.5
        elif num_words < 10:
            length_score = 2.0
        elif num_words < 30:
            length_score = 4.0
        elif num_words < 80:
            length_score = 5.5
        elif num_words < 200:
            length_score = 6.0
        else:
            length_score = 5.5  # very long can be rambly
        
        # Check for truncation
        truncation_penalty = 0.0
        if response_stripped[-1] not in '.!?"\')]}' and num_words > 20:
            truncation_penalty = 0.5
        
        # Check for repetition (sign of low quality)
        if num_words > 10:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                max_bigram_freq = max(bigram_counts.values())
                repetition_rate = max_bigram_freq / len(bigrams)
            else:
                repetition_rate = 0
        else:
            repetition_rate = 0
        
        repetition_penalty = min(3.0, repetition_rate * 10)
        
        # === 5. COHERENCE: Does response address the query? ===
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        response_words_set = set(re.findall(r'\b\w{3,}\b', response_lower))
        
        # Remove very common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                     'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                     'some', 'them', 'than', 'its', 'over', 'such', 'that', 'this',
                     'with', 'will', 'each', 'from', 'they', 'what', 'which', 'their',
                     'said', 'how', 'about', 'many', 'then', 'would', 'make', 'like',
                     'could', 'into', 'more', 'other', 'very', 'after', 'also', 'did'}
        
        query_content = query_words - stopwords
        response_content = response_words_set - stopwords
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        # === 6. GARBAGE/NOISE DETECTION ===
        # Check for HTML, code, or off-topic content
        html_tags = len(re.findall(r'<[^>]+>', response))
        code_markers = len(re.findall(r'(import |def |class |function |var |let |const )', response))
        
        noise_penalty = 0.0
        # If query doesn't ask for code/HTML but response has lots of it
        is_code_query = bool(re.search(r'\b(code|html|program|script|function|tag)\b', query_lower))
        if not is_code_query:
            noise_penalty += min(2.0, html_tags * 0.3)
            noise_penalty += min(2.0, code_markers * 0.5)
        
        # Check for excessive "Output:" or "Input:" or "Question:" patterns (copy-paste artifacts)
        artifact_count = len(re.findall(r'(Output:|Input:|Question:|Answer:)', response))
        if artifact_count > 2:
            noise_penalty += min(2.0, (artifact_count - 2) * 0.5)
        
        # === 7. PROPORTIONAL CALIBRATION SCORING ===
        # The key insight: for ambiguous topics, we want MORE hedging/qualification
        # For factual/concrete tasks, hedging is less important but overconfidence is still bad
        
        calibration_score = 0.0
        
        if topic_ambiguity > 0.5:
            # Ambiguous topic: reward qualification, penalize absolutism
            calibration_score += qualified_rate * 3.0  # up to 3 points for hedging
            calibration_score += evidential_rate * 2.0  # up to 2 points for evidence
            calibration_score -= absolute_rate * 2.0    # penalize absolutism
            calibration_score += qualification_ratio * 2.0  # reward qualification words
        else:
            # Concrete/factual topic: slight reward for appropriate confidence
            calibration_score += qualified_rate * 1.0
            calibration_score += evidential_rate * 1.5
            calibration_score -= absolute_rate * 0.5
            calibration_score += 0.5  # baseline for being direct on factual topics
        
        # === 8. MULTI-PERSPECTIVE CHECK ===
        # Does the response acknowledge multiple viewpoints or nuance?
        perspective_patterns = [
            r'\b(on the other hand|alternatively|conversely|in contrast)\b',
            r'\b(some (people|experts|researchers|scholars) (think|believe|argue|suggest))\b',
            r'\b(one (view|perspective|interpretation|approach))\b',
            r'\b(pros and cons|advantages and disadvantages|trade.?offs?)\b',
            r'\b(it depends|varies|context)\b',
            r'\b(not without|has been criticized|controversial)\b',
            r'\b(however|although|while|whereas|despite)\b',
        ]
        
        perspective_count = sum(1 for p in perspective_patterns if re.search(p, response_lower))
        perspective_bonus = min(1.5, perspective_count * 0.4)
        
        # === 9. FALSE PRECISION DETECTION ===
        # Penalize suspiciously precise numbers without sourcing
        precise_numbers = re.findall(r'\b\d{4,}\b', response)  # 4+ digit numbers
        sourced = bool(re.search(r'\b(according|source|census|data|report|statistic)\b', response_lower))
        
        false_precision_penalty = 0.0
        if precise_numbers and not sourced and topic_ambiguity > 0.3:
            false_precision_penalty = min(1.0, len(precise_numbers) * 0.3)
        
        # === FINAL SCORE COMPUTATION ===
        score = 0.0
        
        # Base: length/structure (0-6)
        score += length_score
        
        # Calibration component (-2 to +5)
        score += calibration_score
        
        # Relevance (0-2)
        score += relevance * 2.0
        
        # Perspective bonus (0-1.5)
        score += perspective_bonus
        
        # Penalties
        score -= truncation_penalty
        score -= repetition_penalty
        score -= noise_penalty
        score -= false_precision_penalty
        
        # Clamp to 0-10
        score = max(0.0, min(10.0, score))
        
        # Final adjustment: extremely short responses (< 5 words) cap at 3
        if num_words < 5:
            score = min(3.0, score)
        
        # Single word or very minimal responses
        if num_words <= 1:
            score = min(1.0, score)
        
        return round(score, 2)
        
    except Exception:
        return 2.0