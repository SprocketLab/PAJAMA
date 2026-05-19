def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a STRUCTURAL approach: analyzing sentence-level epistemic patterns,
    claim density, source attribution, conditional reasoning structures, and the ratio
    of assertive vs. tentative grammatical constructions.
    
    Different from other variants by focusing on:
    1. Sentence-level claim classification (assertive vs. qualified vs. evidential)
    2. Conditional/causal reasoning structures
    3. Source/evidence attribution patterns
    4. Epistemic verb analysis (know/think/believe/suggest spectrum)
    5. Quantified vs. vague claims ratio
    6. Response completeness and coherence signals
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        # Very short responses are almost always low quality
        if len(resp_lower) < 5:
            return 0.5
        
        words = resp_lower.split()
        word_count = len(words)
        
        if word_count < 2:
            return 1.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', resp_lower)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # ============================================================
        # 1. EPISTEMIC VERB SPECTRUM ANALYSIS
        # Classify verbs on a spectrum from high-certainty to high-uncertainty
        # ============================================================
        
        # High certainty epistemic verbs (potentially overconfident)
        high_certainty_verbs = [
            'is definitely', 'is certainly', 'is absolutely', 'is undoubtedly',
            'is unquestionably', 'always is', 'never is', 'must be',
            'is obviously', 'clearly is', 'is guaranteed', 'is proven',
            'without doubt', 'without question', 'there is no doubt',
            'it is certain', 'everyone knows', 'everybody knows',
            'is indisputable', 'is undeniable', 'is incontrovertible'
        ]
        
        # Moderate certainty (well-calibrated for established facts)
        moderate_certainty = [
            'is generally', 'is typically', 'is commonly', 'is widely',
            'is usually', 'is often', 'tends to be', 'is considered',
            'is regarded as', 'is known as', 'is recognized',
            'according to', 'based on', 'is established',
            'evidence shows', 'research shows', 'data shows',
            'studies show', 'studies indicate', 'research indicates',
            'it is well established', 'it is well known',
            'is generally accepted', 'the consensus is'
        ]
        
        # Appropriately uncertain (good epistemic calibration)
        appropriate_uncertainty = [
            'it is possible', 'it is likely', 'it appears', 'it seems',
            'may be', 'might be', 'could be', 'can be',
            'suggests that', 'indicates that', 'implies that',
            'one interpretation', 'one possibility', 'one explanation',
            'it is thought', 'it is believed', 'it is estimated',
            'approximately', 'roughly', 'around', 'about',
            'in some cases', 'in many cases', 'depending on',
            'to some extent', 'to a degree', 'in part',
            'not entirely clear', 'remains uncertain', 'is debated',
            'is controversial', 'varies', 'can vary'
        ]
        
        # Explicit uncertainty acknowledgment (excellent calibration)
        explicit_uncertainty = [
            'i\'m not sure', 'i\'m not certain', 'i don\'t know',
            'it is unclear', 'it is uncertain', 'it is not known',
            'it is difficult to say', 'it is hard to say',
            'it is difficult to determine', 'it is hard to determine',
            'there is no consensus', 'opinions differ', 'views differ',
            'this is debated', 'this is contested', 'this is controversial',
            'the evidence is mixed', 'the evidence is inconclusive',
            'more research is needed', 'further study is needed',
            'it depends on', 'this depends', 'it varies',
            'however', 'on the other hand', 'that said',
            'it is worth noting', 'it should be noted',
            'subjective', 'vary depending'
        ]
        
        high_cert_count = sum(1 for p in high_certainty_verbs if p in resp_lower)
        mod_cert_count = sum(1 for p in moderate_certainty if p in resp_lower)
        approp_uncert_count = sum(1 for p in appropriate_uncertainty if p in resp_lower)
        explicit_uncert_count = sum(1 for p in explicit_uncertainty if p in resp_lower)
        
        # ============================================================
        # 2. SENTENCE-LEVEL CLAIM CLASSIFICATION
        # Classify each sentence as: bare assertion, qualified claim, evidential claim
        # ============================================================
        
        bare_assertions = 0
        qualified_claims = 0
        evidential_claims = 0
        
        qualifier_words = {'may', 'might', 'could', 'possibly', 'perhaps', 'likely',
                          'unlikely', 'probably', 'potentially', 'sometimes', 'often',
                          'usually', 'typically', 'generally', 'approximately', 'roughly',
                          'about', 'around', 'estimated', 'suggests', 'appears', 'seems'}
        
        evidence_words = {'according', 'research', 'study', 'studies', 'evidence',
                         'data', 'survey', 'analysis', 'report', 'source', 'found',
                         'showed', 'demonstrated', 'published', 'documented',
                         'historically', 'traditionally', 'experts', 'scholars'}
        
        for sent in sentences:
            sent_words = set(sent.split())
            has_qualifier = bool(sent_words & qualifier_words)
            has_evidence = bool(sent_words & evidence_words)
            
            if has_evidence:
                evidential_claims += 1
            elif has_qualifier:
                qualified_claims += 1
            else:
                bare_assertions += 1
        
        # Ratio of calibrated sentences
        calibrated_ratio = (qualified_claims + evidential_claims) / num_sentences if num_sentences > 0 else 0
        
        # ============================================================
        # 3. CONDITIONAL/CAUSAL REASONING STRUCTURES
        # Presence of if-then, cause-effect, and nuanced reasoning
        # ============================================================
        
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\b(would|could|might|may)\b',
            r'\bdepending on\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bprovided that\b', r'\bin the case\b', r'\bwhen\b.*\bthen\b',
            r'\bwhether\b.*\bor\b', r'\bwhile\b.*\b(also|however|but)\b',
            r'\balthough\b', r'\beven though\b', r'\bdespite\b',
            r'\bnevertheless\b', r'\bnonetheless\b', r'\bhowever\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bin contrast\b',
            r'\bthat said\b', r'\bwith that being said\b'
        ]
        
        conditional_count = sum(1 for p in conditional_patterns if re.search(p, resp_lower))
        
        # ============================================================
        # 4. OVERCONFIDENCE DETECTION
        # Absolute/universal claims that are likely overconfident
        # ============================================================
        
        overconfident_patterns = [
            r'\balways\b', r'\bnever\b', r'\beveryone\b', r'\bnobody\b',
            r'\bno one\b', r'\beverything\b', r'\bnothing\b',
            r'\bthe best\b', r'\bthe worst\b', r'\bthe only\b',
            r'\bwithout exception\b', r'\bin every case\b',
            r'\bguaranteed\b', r'\b100%\b', r'\bperfect\b',
            r'\bimpossible\b', r'\bcertain\b(?!ly)',
            r'\bundoubtedly\b', r'\bunquestionably\b',
            r'\bobviously\b', r'\bclearly\b'
        ]
        
        overconfident_count = sum(1 for p in overconfident_patterns if re.search(p, resp_lower))
        
        # ============================================================
        # 5. RESPONSE QUALITY FUNDAMENTALS
        # Basic quality signals that correlate with good responses
        # ============================================================
        
        # Length adequacy (not too short, not excessively long)
        length_score = 0
        if word_count < 5:
            length_score = 0.5
        elif word_count < 15:
            length_score = 2.0
        elif word_count < 30:
            length_score = 3.5
        elif word_count < 80:
            length_score = 4.5
        elif word_count < 200:
            length_score = 5.0
        elif word_count < 400:
            length_score = 4.5
        else:
            length_score = 3.5
        
        # Repetition detection (sign of low quality)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            most_common_freq = bigram_counts.most_common(1)[0][1] if bigram_counts else 1
            repetition_ratio = most_common_freq / len(bigrams)
        else:
            repetition_ratio = 0
        
        repetition_penalty = min(repetition_ratio * 15, 4.0)
        
        # Check for garbled/broken text
        garbled_signals = 0
        if re.search(r'(.)\1{4,}', resp_lower):  # repeated characters
            garbled_signals += 1
        if resp_lower.count('#') > 3:
            garbled_signals += 1
        if resp_lower.count('```') > 2 and 'code' not in query_lower:
            garbled_signals += 0.5
        if re.search(r'(output:|input:){2,}', resp_lower):
            garbled_signals += 1.5
        if re.search(r'(question:|answer:){2,}', resp_lower):
            garbled_signals += 1
        
        # Check if response is actually addressing the query
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
        resp_words_set = set(re.findall(r'\b\w{4,}\b', resp_lower))
        
        if query_words:
            relevance = len(query_words & resp_words_set) / len(query_words)
        else:
            relevance = 0.5
        
        # ============================================================
        # 6. INFORMATION DENSITY AND STRUCTURE
        # ============================================================
        
        # Unique word ratio (vocabulary richness)
        unique_words = set(words)
        vocab_richness = len(unique_words) / word_count if word_count > 0 else 0
        
        # Presence of structured information (lists, examples, etc.)
        has_structure = 0
        if re.search(r'\d+[.)\]]\s', response):
            has_structure += 0.5
        if re.search(r'[-•]\s', response):
            has_structure += 0.5
        if re.search(r'\bfor example\b|\bsuch as\b|\be\.g\.\b|\bi\.e\.\b', resp_lower):
            has_structure += 0.5
        if re.search(r'\bfirst\b.*\bsecond\b', resp_lower):
            has_structure += 0.5
        
        # ============================================================
        # 7. QUERY TYPE ANALYSIS
        # Different query types warrant different levels of uncertainty
        # ============================================================
        
        # Factual queries (expecting more certainty)
        factual_indicators = ['how many', 'what is', 'what was', 'who is', 'who was',
                            'when did', 'where is', 'where did', 'name the', 'identify',
                            'list', 'define']
        is_factual = any(fi in query_lower for fi in factual_indicators)
        
        # Opinion/subjective queries (expecting more uncertainty acknowledgment)
        opinion_indicators = ['is it ok', 'should i', 'is it good', 'what do you think',
                            'opinion', 'recommend', 'best way', 'better', 'worse',
                            'is it possible', 'can you', 'how can']
        is_opinion = any(oi in query_lower for oi in opinion_indicators)
        
        # Creative/task queries (less need for uncertainty)
        creative_indicators = ['write', 'create', 'rewrite', 'generate', 'make',
                             'compose', 'design', 'html', 'code', 'translate']
        is_creative = any(ci in query_lower for ci in creative_indicators)
        
        # ============================================================
        # SCORING ASSEMBLY
        # ============================================================
        
        score = 0.0
        
        # Base: length and completeness (0-5 points)
        score += length_score
        
        # Epistemic calibration bonus (0-2.5 points)
        # Reward appropriate uncertainty and evidence-based claims
        epistemic_score = 0.0
        epistemic_score += min(mod_cert_count * 0.3, 0.6)
        epistemic_score += min(approp_uncert_count * 0.35, 0.8)
        epistemic_score += min(explicit_uncert_count * 0.4, 0.7)
        epistemic_score += min(conditional_count * 0.2, 0.4)
        score += epistemic_score
        
        # Calibrated sentence ratio bonus (0-1 point)
        score += calibrated_ratio * 1.0
        
        # Overconfidence penalty (0-2 points penalty)
        # Scale penalty by how much overconfidence relative to response length
        if word_count > 10:
            overconf_density = overconfident_count / (word_count / 50)
            overconf_penalty = min(overconf_density * 0.3, 2.0)
        else:
            overconf_penalty = 0
        
        # High certainty penalty (only if excessive)
        high_cert_penalty = min(high_cert_count * 0.3, 1.0)
        
        score -= overconf_penalty
        score -= high_cert_penalty
        
        # Relevance bonus (0-1.5 points)
        score += relevance * 1.5
        
        # Structure bonus (0-1 point)
        score += min(has_structure, 1.0)
        
        # Vocabulary richness bonus (0-0.5 points)
        if word_count > 10:
            # Good vocab richness is around 0.5-0.8 for medium texts
            richness_bonus = min(vocab_richness, 0.8) * 0.625
            score += richness_bonus
        
        # Repetition penalty
        score -= repetition_penalty
        
        # Garbled text penalty
        score -= garbled_signals * 1.5
        
        # Query-type specific adjustments
        if is_opinion:
            # For opinion queries, reward nuance and qualification more
            nuance_bonus = min((approp_uncert_count + explicit_uncert_count + conditional_count) * 0.2, 1.0)
            score += nuance_bonus
        
        if is_creative:
            # For creative tasks, reduce epistemic requirements
            # but still reward completion and quality
            score += 0.5  # slight bonus since epistemic features are less relevant
        
        # Bonus for responses that acknowledge limitations or complexity
        complexity_phrases = [
            'it is difficult to', 'it is hard to', 'it is complex',
            'there are many', 'there are several', 'there are various',
            'multiple factors', 'various factors', 'many factors',
            'not straightforward', 'nuanced', 'multifaceted',
            'it\'s important to note', 'it is important to note',
            'keep in mind', 'bear in mind', 'worth considering'
        ]
        complexity_count = sum(1 for p in complexity_phrases if p in resp_lower)
        score += min(complexity_count * 0.3, 0.8)
        
        # Penalty for responses that are just fragments or single words
        if num_sentences <= 1 and word_count < 10:
            score -= 2.0
        
        # Penalty for responses that look like they're cut off badly
        if resp_lower.endswith(('the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'at', 'to', 'of')):
            score -= 0.3  # mild penalty for truncation
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This helps discrimination between good and bad responses
        normalized = score / 10.0
        # Soft sigmoid centered at 0.5
        transformed = 1.0 / (1.0 + math.exp(-6 * (normalized - 0.45)))
        final_score = transformed * 10.0
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except:
            return 2.0