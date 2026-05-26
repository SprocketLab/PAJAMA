def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker / connective analysis (causal, contrastive, additive, temporal)
    - Argument depth via clause nesting and subordination
    - Logical flow scoring via sequential connective placement
    - Contradiction detection via opposing sentiment signals
    - Explanation completeness (premise-conclusion patterns)
    - Information density and elaboration ratio
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_clean = response.strip()
        if len(response_clean) < 5:
            return 0.0
        
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # =====================================================
        # 1. DISCOURSE CONNECTIVE ANALYSIS
        # Categorize connectives by their logical function
        # =====================================================
        
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\bso\b(?=\s+\w)', r'\baccordingly\b',
            r'\bthat\'s why\b', r'\bthis means\b', r'\bwhich means\b',
            r'\bleading to\b', r'\bcaused by\b', r'\bresulting in\b',
            r'\bgiven that\b', r'\bin light of\b'
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\byet\b',
            r'\bdespite\b', r'\bin contrast\b', r'\bwhile\b(?=\s)',
            r'\bwhereas\b', r'\bnonetheless\b', r'\beven though\b',
            r'\bon the contrary\b', r'\bthat said\b', r'\bhaving said that\b',
            r'\bstill\b', r'\binstead\b', r'\brather\b'
        ]
        
        additive_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bwhat\'s more\b', r'\bnot only\b',
            r'\bbeyond that\b', r'\bon top of that\b', r'\bplus\b',
            r'\bas well\b', r'\bcoupled with\b'
        ]
        
        temporal_connectives = [
            r'\bthen\b', r'\bnext\b', r'\bafterward\b', r'\bfinally\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\binitially\b',
            r'\bsubsequently\b', r'\bpreviously\b', r'\bmeanwhile\b',
            r'\bat first\b', r'\bin the end\b', r'\bultimately\b',
            r'\beventually\b', r'\bto begin with\b'
        ]
        
        exemplification = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bto illustrate\b', r'\bnamely\b', r'\bspecifically\b',
            r'\bin particular\b', r'\be\.g\.\b', r'\bi\.e\.\b',
            r'\bconsider\b', r'\btake for example\b', r'\blike\b(?=\s+\w+ing|\s+when|\s+a\b)'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\ball in all\b', r'\boverall\b', r'\bin short\b',
            r'\bthe point is\b', r'\bessentially\b', r'\bin essence\b',
            r'\bto sum up\b', r'\bthe bottom line\b', r'\bultimately\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectives, resp_lower)
        contrastive_count = count_patterns(contrastive_connectives, resp_lower)
        additive_count = count_patterns(additive_connectives, resp_lower)
        temporal_count = count_patterns(temporal_connectives, resp_lower)
        exemplification_count = count_patterns(exemplification, resp_lower)
        conclusion_count = count_patterns(conclusion_markers, resp_lower)
        
        total_connectives = (causal_count + contrastive_count + additive_count + 
                           temporal_count + exemplification_count + conclusion_count)
        
        # Connective density (per 100 words) - sweet spot around 3-8 per 100 words
        connective_density = (total_connectives / num_words) * 100
        # Score peaks around 5 per 100 words
        density_score = max(0, 10 - abs(connective_density - 5) * 1.5)
        density_score = min(density_score, 10)
        
        # Connective variety - having multiple types is better
        type_counts = [causal_count, contrastive_count, additive_count, 
                      temporal_count, exemplification_count, conclusion_count]
        types_present = sum(1 for c in type_counts if c > 0)
        variety_score = min(types_present * 1.8, 10)
        
        # Causal reasoning bonus - strongest indicator of logical argument
        causal_score = min(causal_count * 2.5, 10)
        
        # =====================================================
        # 2. SUBORDINATION AND CLAUSE COMPLEXITY
        # Complex sentences with subordinate clauses indicate deeper reasoning
        # =====================================================
        
        subordinators = [
            r'\bif\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b', r'\bthat\b',
            r'\bwho\b', r'\bwhom\b', r'\bwhose\b', r'\bafter\b', r'\bbefore\b',
            r'\bunless\b', r'\buntil\b', r'\bprovided\b', r'\bassuming\b',
            r'\bsupposing\b', r'\bwhenever\b', r'\bwherever\b'
        ]
        
        subordination_count = count_patterns(subordinators, resp_lower)
        # Normalize by sentence count
        sub_per_sentence = subordination_count / num_sentences
        subordination_score = min(sub_per_sentence * 3.5, 10)
        
        # =====================================================
        # 3. LOGICAL FLOW - Connectives appearing at sentence starts
        # Indicates explicit logical transitions between ideas
        # =====================================================
        
        flow_markers = [
            r'^(however|but|therefore|thus|hence|moreover|furthermore|additionally|'
            r'consequently|nevertheless|in addition|on the other hand|as a result|'
            r'for this reason|that said|in contrast|similarly|likewise|'
            r'first|second|third|finally|also|yet|still|instead|'
            r'to begin|in summary|overall|essentially|the point is)',
        ]
        
        sentence_start_transitions = 0
        for sent in sentences:
            sent_stripped = sent.strip().lower()
            for pattern in flow_markers:
                if re.match(pattern, sent_stripped):
                    sentence_start_transitions += 1
                    break
        
        if num_sentences > 1:
            transition_ratio = sentence_start_transitions / (num_sentences - 1)
        else:
            transition_ratio = 0
        
        # Sweet spot: 20-50% of sentences start with transitions
        flow_score = min(transition_ratio * 15, 10)
        
        # =====================================================
        # 4. ARGUMENT STRUCTURE DETECTION
        # Look for premise-conclusion patterns
        # =====================================================
        
        # Premise indicators
        premise_markers = [
            r'\bgiven\b', r'\bassuming\b', r'\bsince\b', r'\bbecause\b',
            r'\bif we\b', r'\bconsidering\b', r'\bthe fact that\b',
            r'\bit is true that\b', r'\bwe know that\b', r'\bas\b(?=\s+\w+\s+\w+)',
            r'\bnote that\b', r'\bremember that\b'
        ]
        
        conclusion_indicators = [
            r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bso\b',
            r'\bconsequently\b', r'\bit follows\b', r'\bwe can conclude\b',
            r'\bthis means\b', r'\bthis suggests\b', r'\bthis implies\b',
            r'\bin other words\b', r'\bthe takeaway\b', r'\bthe upshot\b'
        ]
        
        premise_count = count_patterns(premise_markers, resp_lower)
        conclusion_indicator_count = count_patterns(conclusion_indicators, resp_lower)
        
        # Having both premises and conclusions indicates complete arguments
        if premise_count > 0 and conclusion_indicator_count > 0:
            argument_completeness = min((premise_count + conclusion_indicator_count) * 2, 10)
        elif premise_count > 0 or conclusion_indicator_count > 0:
            argument_completeness = min((premise_count + conclusion_indicator_count) * 1.2, 6)
        else:
            argument_completeness = 0
        
        # =====================================================
        # 5. ELABORATION AND EXPLANATION DEPTH
        # Parenthetical explanations, appositive phrases, clarifications
        # =====================================================
        
        parenthetical_count = len(re.findall(r'\([^)]+\)', response_clean))
        dash_elaboration = len(re.findall(r'--[^-]+--|—[^—]+—', response_clean))
        colon_elaboration = len(re.findall(r':\s+\w', response_clean))
        
        elaboration_signals = [
            r'\bin other words\b', r'\bthat is\b', r'\bmeaning\b',
            r'\bput differently\b', r'\bto clarify\b', r'\bto put it\b',
            r'\bwhat I mean\b', r'\bmore precisely\b', r'\bto be specific\b'
        ]
        elab_count = count_patterns(elaboration_signals, resp_lower)
        
        total_elaboration = parenthetical_count + dash_elaboration + colon_elaboration + elab_count
        elaboration_score = min(total_elaboration * 1.5, 8)
        
        # =====================================================
        # 6. INTERNAL CONSISTENCY CHECK
        # Look for potential contradictions (opposing claims in proximity)
        # =====================================================
        
        contradiction_penalty = 0
        negation_patterns = [r'\bnot\b', r'\bnever\b', r'\bno\b', r'\bnone\b', 
                           r'\bneither\b', r'\bnor\b', r'\bn\'t\b']
        
        # Check if adjacent sentences make contradictory claims about same subject
        for i in range(len(sentences) - 1):
            s1_words = set(re.findall(r'\b[a-z]+\b', sentences[i].lower()))
            s2_words = set(re.findall(r'\b[a-z]+\b', sentences[i+1].lower()))
            
            # High word overlap between adjacent sentences
            if len(s1_words) > 0 and len(s2_words) > 0:
                overlap = len(s1_words & s2_words)
                overlap_ratio = overlap / min(len(s1_words), len(s2_words))
                
                if overlap_ratio > 0.4:
                    # Check if one has negation and other doesn't for same key terms
                    s1_neg = count_patterns(negation_patterns, sentences[i].lower())
                    s2_neg = count_patterns(negation_patterns, sentences[i+1].lower())
                    
                    if (s1_neg > 0) != (s2_neg > 0):
                        # Potential contradiction but only penalize if no contrastive marker
                        has_contrast = count_patterns(contrastive_connectives, sentences[i+1].lower())
                        if has_contrast == 0:
                            contradiction_penalty += 2
        
        contradiction_penalty = min(contradiction_penalty, 6)
        
        # =====================================================
        # 7. RESPONSE SUBSTANTIVENESS
        # Longer, more developed responses tend to have better argument structure
        # =====================================================
        
        # Length score with diminishing returns
        length_score = min(math.log(num_words + 1) * 1.8, 10)
        
        # Sentence count bonus (multiple sentences = more room for logical flow)
        sentence_bonus = min(math.log(num_sentences + 1) * 2.5, 8)
        
        # =====================================================
        # 8. QUALIFICATION AND NUANCE
        # Good logical arguments acknowledge limitations and conditions
        # =====================================================
        
        qualification_markers = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends to\b', r'\bin most cases\b', r'\bit depends\b',
            r'\bto some extent\b', r'\bunder certain\b', r'\bcaveats?\b',
            r'\bwith the exception\b', r'\bunless\b', r'\bprovided that\b',
            r'\bin some cases\b', r'\bthe trade-off\b', r'\bon one hand\b',
            r'\bdepending on\b', r'\bvaries\b', r'\bnot always\b',
            r'\bnot necessarily\b', r'\bcan be\b', r'\bmay\b', r'\bmight\b'
        ]
        
        qualification_count = count_patterns(qualification_markers, resp_lower)
        qualification_score = min(qualification_count * 1.2, 7)
        
        # =====================================================
        # 9. TOPICAL COHERENCE VIA REPEATED KEY CONCEPTS
        # Good arguments return to and develop key concepts
        # =====================================================
        
        # Filter out stopwords for content word analysis
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or',
            'if', 'because', 'while', 'although', 'this', 'that', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'you', 'your', 'he', 'she', 'they',
            'we', 'them', 'their', 'his', 'her', 'what', 'which', 'who', 'whom',
            'up', 'about', 'also', 'well', 'back', 'even', 'still', 'way', 'take',
            'come', 'make', 'like', 'get', 'got', 'go', 'know', 'think', 'see',
            'one', 'two', 'much', 'many', 'any', 'really', 'thing', 'things',
            'don', 'doesn', 'didn', 'won', 'wouldn', 'couldn', 'shouldn', 've',
            'll', 're', 'am', 'going', 'been', 'say', 'said', 'new', 'now'
        }
        
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        content_counter = Counter(content_words)
        
        if len(content_words) > 3:
            # Check if key concepts from query appear and recur in response
            query_words = set(re.findall(r'\b[a-zA-Z]+\b', query.lower())) - stopwords
            query_words = {w for w in query_words if len(w) > 2}
            
            # Concept recurrence: key content words that appear multiple times
            recurring_concepts = sum(1 for w, c in content_counter.items() if c >= 2)
            concept_development = min(recurring_concepts * 0.8, 6)
            
            # Query relevance through shared concepts
            shared_concepts = sum(1 for w in query_words if w in content_counter)
            relevance_score = min(shared_concepts * 0.8, 5)
        else:
            concept_development = 0
            relevance_score = 0
        
        # =====================================================
        # 10. SENTENCE-TO-SENTENCE COHERENCE
        # Adjacent sentences should share some content words (topic continuity)
        # but not be repetitive
        # =====================================================
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            s1_content = set(re.findall(r'\b[a-z]+\b', sentences[i].lower())) - stopwords
            s2_content = set(re.findall(r'\b[a-z]+\b', sentences[i+1].lower())) - stopwords
            
            s1_content = {w for w in s1_content if len(w) > 2}
            s2_content = {w for w in s2_content if len(w) > 2}
            
            if len(s1_content) > 0 and len(s2_content) > 0:
                union = len(s1_content | s2_content)
                intersection = len(s1_content & s2_content)
                if union > 0:
                    sim = intersection / union
                    # Ideal similarity: 0.1 - 0.4 (some overlap but not repetitive)
                    if 0.05 <= sim <= 0.5:
                        coherence_scores.append(1.0)
                    elif sim > 0.5:
                        coherence_scores.append(0.5)  # Too repetitive
                    else:
                        coherence_scores.append(0.3)  # Too disconnected
        
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            sequential_coherence_score = avg_coherence * 8
        else:
            sequential_coherence_score = 3  # Single sentence - neutral
        
        # =====================================================
        # COMBINE ALL SCORES
        # =====================================================
        
        # Weighted combination
        final_score = (
            density_score * 0.08 +          # Connective density
            variety_score * 0.10 +           # Connective variety
            causal_score * 0.12 +            # Causal reasoning
            subordination_score * 0.06 +     # Clause complexity
            flow_score * 0.08 +              # Logical flow transitions
            argument_completeness * 0.10 +   # Premise-conclusion structure
            elaboration_score * 0.05 +       # Elaboration depth
            length_score * 0.10 +            # Substantiveness
            sentence_bonus * 0.06 +          # Multi-sentence development
            qualification_score * 0.05 +     # Nuance
            concept_development * 0.07 +     # Concept recurrence
            relevance_score * 0.05 +         # Query relevance
            sequential_coherence_score * 0.08 - # Sequential coherence
            contradiction_penalty * 0.05     # Contradiction penalty
        )
        
        # Scale to 0-10 range
        final_score = max(0, min(10, final_score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This makes the function more discriminative
        normalized = final_score / 10.0
        spread = 1 / (1 + math.exp(-6 * (normalized - 0.35)))
        final_score = spread * 10
        
        return round(final_score, 3)
        
    except Exception:
        return 2.0