def judging_function(query, response):
    """
    Evaluates epistemic calibration and uncertainty communication using
    a sentence-level analysis approach. Analyzes each sentence for its
    epistemic stance (claim type, hedging, source attribution, etc.)
    and computes a composite score based on the distribution of epistemic
    stances across the response.
    
    This variant focuses on:
    1. Sentence-level epistemic classification (not just keyword counting)
    2. Claim density and qualification ratio
    3. Source/evidence attribution patterns
    4. Epistemic verb analysis
    5. Conditional/hypothetical reasoning detection
    6. Response structure and depth as context
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower()
        query_lower = query.lower()
        resp_len = len(response)
        
        # Split into sentences more carefully
        def split_sentences(text):
            # Split on sentence-ending punctuation, but be careful with abbreviations
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
            # Also split on newlines
            sentences = []
            for p in parts:
                sub = p.strip().split('\n')
                for s in sub:
                    s = s.strip()
                    if len(s) > 5:
                        sentences.append(s)
            if not sentences and text.strip():
                sentences = [text.strip()]
            return sentences
        
        sentences = split_sentences(response)
        num_sentences = max(len(sentences), 1)
        
        # ---- 1. Sentence-level epistemic classification ----
        # Classify each sentence into epistemic categories
        
        # Epistemic verbs (verbs that signal belief/knowledge states)
        epistemic_verbs = [
            'believe', 'think', 'suspect', 'assume', 'suppose',
            'estimate', 'guess', 'reckon', 'imagine', 'consider',
            'wonder', 'doubt', 'speculate', 'hypothesize', 'conjecture',
            'presume', 'infer', 'deduce', 'gather', 'understand'
        ]
        
        # Evidential markers (signal source of information)
        evidential_markers = [
            'according to', 'studies show', 'research indicates',
            'evidence suggests', 'data shows', 'it has been shown',
            'it is known', 'it is documented', 'historically',
            'scholars argue', 'experts say', 'the literature',
            'findings suggest', 'observations suggest', 'reports indicate',
            'based on', 'in my experience', 'from what i', 'as far as i know',
            'to my knowledge', 'from what i understand', 'i have seen',
            'i have found', 'in practice', 'in theory', 'empirically',
            'anecdotally', 'traditionally', 'conventionally'
        ]
        
        # Conditional/hypothetical markers
        conditional_markers = [
            'if ', 'unless', 'assuming', 'provided that', 'in case',
            'were to', 'would be', 'could be', 'might be', 'may be',
            'it depends', 'depending on', 'contingent on', 'subject to',
            'hypothetically', 'in principle', 'theoretically',
            'under certain', 'in some cases', 'in certain circumstances'
        ]
        
        # Absolute/overconfident markers
        absolute_markers = [
            'always', 'never', 'definitely', 'absolutely', 'certainly',
            'undoubtedly', 'without question', 'without doubt',
            'there is no', 'it is impossible', 'guaranteed',
            'no way', 'everyone knows', 'obviously', 'clearly',
            'of course', 'needless to say', 'indisputably',
            'unquestionably', 'beyond doubt', 'no question',
            'the fact is', 'the truth is', 'plain and simple',
            'period.', 'end of story', 'hands down'
        ]
        
        # Nuance/qualification markers
        qualification_markers = [
            'however', 'although', 'though', 'but', 'nevertheless',
            'on the other hand', 'that said', 'having said that',
            'with that being said', 'at the same time', 'conversely',
            'alternatively', 'to some extent', 'in part', 'partially',
            'somewhat', 'relatively', 'comparatively', 'more or less',
            'to a degree', 'up to a point', 'with some exceptions',
            'generally speaking', 'broadly speaking', 'as a rule',
            'for the most part', 'in most cases', 'typically',
            'tend to', 'tends to'
        ]
        
        # Degree/probability markers  
        probability_markers = [
            'probably', 'possibly', 'perhaps', 'maybe', 'likely',
            'unlikely', 'plausibly', 'conceivably', 'presumably',
            'seemingly', 'apparently', 'ostensibly', 'potentially',
            'chances are', 'odds are', 'it seems', 'it appears',
            'it looks like', 'there is a chance', 'there is a possibility'
        ]
        
        # Classify sentences
        epistemic_verb_count = 0
        evidential_count = 0
        conditional_count = 0
        absolute_count = 0
        qualification_count = 0
        probability_count = 0
        
        # Track sentences with claims vs hedged statements
        claim_sentences = 0
        hedged_sentences = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            has_epistemic = any(v in sent_lower for v in epistemic_verbs)
            has_evidential = any(e in sent_lower for e in evidential_markers)
            has_conditional = any(c in sent_lower for c in conditional_markers)
            has_absolute = any(a in sent_lower for a in absolute_markers)
            has_qualification = any(q in sent_lower for q in qualification_markers)
            has_probability = any(p in sent_lower for p in probability_markers)
            
            if has_epistemic:
                epistemic_verb_count += 1
            if has_evidential:
                evidential_count += 1
            if has_conditional:
                conditional_count += 1
            if has_absolute:
                absolute_count += 1
            if has_qualification:
                qualification_count += 1
            if has_probability:
                probability_count += 1
            
            # Is this a hedged sentence?
            is_hedged = has_epistemic or has_evidential or has_conditional or has_probability or has_qualification
            # Is this a strong claim?
            # Sentences with "is", "are", declarative statements without hedging
            is_declarative = bool(re.search(r'\b(is|are|was|were|will|does|do|did|has|have|had)\b', sent_lower))
            
            if is_hedged:
                hedged_sentences += 1
            if is_declarative and not is_hedged:
                claim_sentences += 1
        
        # ---- 2. Compute epistemic calibration metrics ----
        
        # Hedging ratio: proportion of sentences that are hedged
        hedging_ratio = hedged_sentences / num_sentences
        
        # Absolute claim ratio: proportion with absolute markers (penalize)
        absolute_ratio = absolute_count / num_sentences
        
        # Qualification density
        qualification_density = qualification_count / num_sentences
        
        # Evidential density
        evidential_density = evidential_count / num_sentences
        
        # Probability language density
        probability_density = probability_count / num_sentences
        
        # Conditional reasoning density
        conditional_density = conditional_count / num_sentences
        
        # Epistemic verb density
        epistemic_density = epistemic_verb_count / num_sentences
        
        # ---- 3. Analyze response depth and engagement ----
        words = response_lower.split()
        word_count = len(words)
        
        # Depth score based on response length (diminishing returns)
        if word_count < 10:
            depth_score = 0.1
        elif word_count < 30:
            depth_score = 0.3
        elif word_count < 60:
            depth_score = 0.5
        elif word_count < 120:
            depth_score = 0.7
        elif word_count < 250:
            depth_score = 0.85
        else:
            depth_score = 1.0
        
        # ---- 4. Analyze reasoning structure ----
        # Look for causal reasoning, examples, explanations
        causal_markers = [
            'because', 'since', 'therefore', 'thus', 'hence',
            'as a result', 'consequently', 'due to', 'owing to',
            'this means', 'this suggests', 'this implies',
            'for this reason', 'the reason', 'explains why',
            'which leads to', 'resulting in'
        ]
        causal_count = sum(1 for c in causal_markers if c in response_lower)
        causal_density = min(causal_count / max(num_sentences, 1), 1.0)
        
        # Example/illustration markers
        example_markers = [
            'for example', 'for instance', 'such as', 'e.g.',
            'like ', 'consider ', 'imagine ', 'suppose ',
            'in the case of', 'to illustrate', 'specifically',
            'in particular', 'namely'
        ]
        example_count = sum(1 for e in example_markers if e in response_lower)
        example_density = min(example_count / max(num_sentences, 1), 0.5)
        
        # ---- 5. Detect meta-epistemic awareness ----
        # Does the response acknowledge limitations or complexity?
        meta_epistemic = [
            'it\'s complicated', 'it\'s complex', 'it\'s nuanced',
            'there are different', 'there are various', 'there are multiple',
            'perspectives', 'viewpoints', 'schools of thought',
            'debate', 'controversial', 'contested', 'disputed',
            'open question', 'not settled', 'ongoing',
            'i\'m not sure', 'i don\'t know', 'hard to say',
            'difficult to determine', 'unclear', 'uncertain',
            'ambiguous', 'it\'s hard to', 'it\'s difficult to',
            'depends on', 'varies', 'context', 'situation'
        ]
        meta_count = sum(1 for m in meta_epistemic if m in response_lower)
        meta_density = min(meta_count / max(num_sentences, 1), 0.8)
        
        # ---- 6. Personal experience vs universal claims ----
        personal_markers = [
            'i think', 'i believe', 'in my opinion', 'in my experience',
            'i feel', 'i would', 'i\'d', 'i have', 'i\'ve',
            'from my', 'personally', 'my own'
        ]
        personal_count = sum(1 for p in personal_markers if p in response_lower)
        personal_density = min(personal_count / max(num_sentences, 1), 0.6)
        
        # ---- 7. Query complexity assessment ----
        # More complex/ambiguous queries should reward more hedging
        query_words = query_lower.split()
        query_len = len(query_words)
        
        # Detect if query is about opinions, ethics, subjective topics
        subjective_query_markers = [
            'ethical', 'moral', 'opinion', 'think', 'feel',
            'should', 'better', 'best', 'worst', 'wrong', 'right',
            'argue', 'debate', 'controversial', 'philosophy',
            'political', 'believe', 'perspective', 'view'
        ]
        query_subjectivity = sum(1 for m in subjective_query_markers if m in query_lower)
        is_subjective_query = query_subjectivity >= 1
        
        # Detect factual/technical queries
        factual_query_markers = [
            'how to', 'how do', 'what is', 'what are', 'when did',
            'where is', 'who is', 'which', 'calculate', 'formula',
            'sql', 'code', 'program', 'create table', 'select'
        ]
        query_factuality = sum(1 for m in factual_query_markers if m in query_lower)
        is_factual_query = query_factuality >= 1
        
        # ---- 8. Composite scoring ----
        
        # Base score from depth
        score = depth_score * 3.0  # 0-3 points
        
        # Epistemic calibration components
        # Reward hedging (but not too much - optimal is moderate)
        optimal_hedging = 0.3 if is_factual_query else 0.5
        hedging_score = 1.0 - abs(hedging_ratio - optimal_hedging) / max(optimal_hedging, 0.3)
        hedging_score = max(hedging_score, 0.0)
        score += hedging_score * 1.5  # 0-1.5 points
        
        # Reward evidential markers
        score += min(evidential_density * 3.0, 1.5)  # 0-1.5 points
        
        # Reward qualification/nuance
        score += min(qualification_density * 2.0, 1.0)  # 0-1 points
        
        # Reward probability language
        score += min(probability_density * 2.0, 0.8)  # 0-0.8 points
        
        # Reward conditional reasoning
        score += min(conditional_density * 2.0, 0.8)  # 0-0.8 points
        
        # Reward causal reasoning
        score += min(causal_density * 1.5, 0.7)  # 0-0.7 points
        
        # Reward examples
        score += min(example_density * 2.0, 0.5)  # 0-0.5 points
        
        # Reward meta-epistemic awareness
        score += min(meta_density * 2.0, 0.8)  # 0-0.8 points
        
        # Reward epistemic verbs
        score += min(epistemic_density * 1.5, 0.6)  # 0-0.6 points
        
        # Penalize overconfidence (absolute markers without qualification)
        overconfidence_penalty = absolute_ratio * 2.0
        if qualification_density > 0 or hedging_ratio > 0.2:
            overconfidence_penalty *= 0.5  # Reduced penalty if also hedging
        score -= min(overconfidence_penalty, 2.0)
        
        # Context-sensitive adjustments
        if is_subjective_query:
            # For subjective queries, reward personal framing and meta-epistemic awareness
            score += personal_density * 0.5
            score += meta_density * 0.5
            # Extra penalty for being too absolute on subjective topics
            if absolute_ratio > 0.3 and hedging_ratio < 0.1:
                score -= 1.0
        
        # ---- 9. Structural engagement bonus ----
        # Multiple paragraphs or structured response
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.3
        if len(paragraphs) >= 3:
            score += 0.2
        
        # Contains references to specific things (names, titles, concepts)
        # Detect capitalized words that aren't sentence starters
        proper_nouns = re.findall(r'(?<!\. )(?<!\n)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        if len(proper_nouns) >= 2:
            score += 0.3
        
        # ---- 10. Variety in epistemic stance ----
        # Reward responses that use multiple types of epistemic markers
        epistemic_types_used = sum([
            epistemic_verb_count > 0,
            evidential_count > 0,
            conditional_count > 0,
            qualification_count > 0,
            probability_count > 0,
            meta_count > 0,
            personal_count > 0,
            causal_count > 0
        ])
        variety_bonus = min(epistemic_types_used * 0.15, 0.9)
        score += variety_bonus
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
        
    except Exception as e:
        # Fallback: return a middle score
        try:
            if response and len(response) > 50:
                return 4.0
            return 2.0
        except:
            return 2.0