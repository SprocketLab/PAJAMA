def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectors)
    - Argument depth via clause/subordination complexity
    - Logical flow scoring via sequential discourse progression
    - Contradiction detection via negation patterns near repeated claims
    - Evidence/example integration detection
    - Coherent conclusion detection
    
    This variant focuses on DISCOURSE MARKERS and ARGUMENT CHAIN ANALYSIS,
    distinct from previous variants that used sentence length, vocabulary diversity,
    paragraph analysis, word overlap, Jaccard similarity, hedging, confidence markers,
    bullet/list/header detection.
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
        if len(response_clean) < 10:
            return 0.5
        
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # =====================================================
        # 1. DISCOURSE MARKER DENSITY AND VARIETY
        # =====================================================
        
        # Causal connectors - indicate reasoning chains
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bso that\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bthis leads to\b', r'\bfor this reason\b', r'\bit follows\b',
            r'\bgiven that\b', r'\baccordingly\b', r'\bin turn\b'
        ]
        
        # Contrastive connectors - indicate nuanced thinking
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\byet\b',
            r'\bdespite\b', r'\bwhile\b', r'\bwhereas\b', r'\binstead\b',
            r'\brather\b', r'\bstill\b', r'\bnonetheless\b',
            r'\bthat said\b', r'\beven so\b', r'\bthen again\b'
        ]
        
        # Additive/elaboration connectors
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bnot only\b',
            r'\bbesides\b', r'\blikewise\b', r'\bsimilarly\b',
            r'\bequally\b', r'\bwhat\'s more\b', r'\bon top of\b'
        ]
        
        # Temporal/sequential connectors - indicate structured flow
        sequential_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bto begin\b', r'\bto start\b', r'\blast\b',
            r'\binitially\b', r'\bafterward\b', r'\bpreviously\b',
            r'\bbefore\b', r'\bafter\b', r'\bonce\b'
        ]
        
        # Exemplification markers - indicate evidence-based reasoning
        example_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bconsider\b', r'\btake\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\blike\b'
        ]
        
        # Conclusion/summary markers
        conclusion_markers = [
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
            r'\bto summarize\b', r'\bin short\b', r'\bultimately\b',
            r'\ball in all\b', r'\bthe point is\b', r'\bin essence\b',
            r'\bto sum up\b', r'\bthe bottom line\b'
        ]
        
        # Conditional/hypothetical markers - indicate analytical thinking
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b',
            r'\bassuming\b', r'\bsuppose\b', r'\bin case\b',
            r'\bwould\b', r'\bcould\b', r'\bmight\b',
            r'\bwhat if\b', r'\bhypothetically\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_markers(patterns):
            total = 0
            unique = 0
            for p in patterns:
                matches = len(re.findall(p, resp_lower))
                if matches > 0:
                    unique += 1
                    total += matches
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers)
        contrastive_count, contrastive_unique = count_markers(contrastive_markers)
        additive_count, additive_unique = count_markers(additive_markers)
        sequential_count, sequential_unique = count_markers(sequential_markers)
        example_count, example_unique = count_markers(example_markers)
        conclusion_count, conclusion_unique = count_markers(conclusion_markers)
        conditional_count, conditional_unique = count_markers(conditional_markers)
        
        total_discourse_markers = (causal_count + contrastive_count + additive_count +
                                   sequential_count + example_count + conclusion_count +
                                   conditional_count)
        total_unique_types = (causal_unique + contrastive_unique + additive_unique +
                              sequential_unique + example_unique + conclusion_unique +
                              conditional_unique)
        
        # Discourse marker density (per 100 words)
        marker_density = (total_discourse_markers / num_words) * 100
        # Cap and normalize: ideal is around 5-15 per 100 words
        marker_density_score = min(marker_density / 10.0, 1.5) * 10  # max ~15
        
        # Discourse marker variety (number of unique categories with markers)
        categories_present = sum(1 for c in [causal_unique, contrastive_unique, additive_unique,
                                              sequential_unique, example_unique, conclusion_unique,
                                              conditional_unique] if c > 0)
        variety_score = (categories_present / 7.0) * 12  # max 12
        
        # =====================================================
        # 2. ARGUMENT CHAIN DEPTH (subordination complexity)
        # =====================================================
        
        # Count subordinating conjunctions indicating nested reasoning
        subordinators = [
            r'\bbecause\b', r'\balthough\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bsince\b', r'\bunless\b', r'\bif\b', r'\bwhen\b',
            r'\bwhere\b', r'\bthat\b', r'\bwhich\b', r'\bwho\b',
            r'\bwhom\b', r'\bwhose\b'
        ]
        
        subordination_count = sum(len(re.findall(p, resp_lower)) for p in subordinators)
        # Subordination density per sentence
        sub_per_sentence = subordination_count / num_sentences
        # Ideal: 1-3 subordinations per sentence indicates complex but readable arguments
        sub_score = min(sub_per_sentence / 2.0, 1.0) * 8  # max 8
        
        # =====================================================
        # 3. LOGICAL FLOW: INTER-SENTENCE COHERENCE
        # =====================================================
        # Check if sentences that start with discourse markers connect to previous content
        # by sharing content words
        
        flow_score = 0.0
        if num_sentences > 1:
            sentence_word_sets = []
            for s in sentences:
                s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
                # Remove very common words
                stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                            'have', 'been', 'some', 'them', 'than', 'its', 'over',
                            'with', 'this', 'that', 'from', 'they', 'will', 'would',
                            'there', 'their', 'what', 'about', 'which', 'when',
                            'make', 'like', 'just', 'into', 'your', 'more', 'other',
                            'could', 'also', 'very', 'then', 'these', 'only', 'does'}
                s_words -= stopwords
                sentence_word_sets.append(s_words)
            
            coherent_transitions = 0
            total_transitions = num_sentences - 1
            
            for i in range(1, len(sentence_word_sets)):
                if sentence_word_sets[i] and sentence_word_sets[i-1]:
                    overlap = len(sentence_word_sets[i] & sentence_word_sets[i-1])
                    union = len(sentence_word_sets[i] | sentence_word_sets[i-1])
                    if union > 0:
                        # Even a small overlap indicates topical continuity
                        if overlap >= 1:
                            coherent_transitions += 1
                        # Bonus for higher overlap
                        if overlap >= 2:
                            coherent_transitions += 0.3
            
            if total_transitions > 0:
                flow_ratio = coherent_transitions / total_transitions
                flow_score = flow_ratio * 12  # max 12
        
        # =====================================================
        # 4. PROGRESSIVE ARGUMENT STRUCTURE
        # =====================================================
        # Check if the response has a beginning (setup), middle (development), end (conclusion)
        
        structure_score = 0.0
        
        # Opening: does it set context or acknowledge the query?
        first_sentence = sentences[0].lower() if sentences else ""
        opening_patterns = [
            r'\bessentially\b', r'\bthe\b.*\bis\b', r'\bthere are\b',
            r'\bit depends\b', r'\byes\b', r'\bno\b', r'\babsolutely\b',
            r'\bgreat question\b', r'\bwell\b', r'\bso\b',
            r'\bin general\b', r'\btypically\b', r'\busually\b',
            r'\bfirst\b', r'\bto answer\b', r'\bthe short answer\b',
            r'\bthe key\b', r'\bthe main\b', r'\bthe important\b'
        ]
        has_opening = any(re.search(p, first_sentence) for p in opening_patterns)
        if has_opening:
            structure_score += 2.0
        
        # Development: middle sentences elaborate
        if num_sentences >= 3:
            middle_sentences = sentences[1:-1]
            middle_text = ' '.join(middle_sentences).lower()
            development_indicators = (
                len(re.findall(r'\bfor example\b|\bsuch as\b|\bspecifically\b|\bin particular\b', middle_text)) +
                len(re.findall(r'\bbecause\b|\bsince\b|\bdue to\b', middle_text)) +
                len(re.findall(r'\bhowever\b|\bbut\b|\balthough\b|\bon the other hand\b', middle_text)) +
                len(re.findall(r'\bfurthermore\b|\bmoreover\b|\badditionally\b|\balso\b', middle_text))
            )
            structure_score += min(development_indicators / 3.0, 1.0) * 4.0
        
        # Conclusion: last sentence wraps up
        if num_sentences >= 2:
            last_sentence = sentences[-1].lower()
            conclusion_patterns = [
                r'\boverall\b', r'\bin short\b', r'\bso\b', r'\bultimately\b',
                r'\bthe point\b', r'\bto sum\b', r'\bin conclusion\b',
                r'\bhope\b', r'\btrust me\b', r'\bplease\b', r'\bgood luck\b',
                r'\bI recommend\b', r'\bI suggest\b', r'\bI would\b',
                r'\bbottom line\b', r'\ball in all\b'
            ]
            has_conclusion = any(re.search(p, last_sentence) for p in conclusion_patterns)
            if has_conclusion:
                structure_score += 2.0
        
        # max structure_score ~ 8
        
        # =====================================================
        # 5. CONTRADICTION / INCONSISTENCY DETECTION (penalty)
        # =====================================================
        
        contradiction_penalty = 0.0
        
        # Check for explicit contradictions: "X is Y" followed by "X is not Y"
        # Simplified: look for negation near repeated content
        negation_patterns = re.findall(r'\bnot\b|\bnever\b|\bno\b|\bnone\b|\bneither\b|\bnor\b', resp_lower)
        negation_density = len(negation_patterns) / num_words
        
        # High negation density without contrastive markers might indicate confusion
        if negation_density > 0.05 and contrastive_count == 0:
            contradiction_penalty += 3.0
        
        # Check for "but" or "however" that contradicts the immediately preceding claim
        # This is hard to do perfectly, but excessive self-contradiction is a signal
        contradiction_phrases = re.findall(
            r'(?:it is|it\'s|this is|they are|there is).*?(?:but|however).*?(?:it is not|it isn\'t|it\'s not|they are not|there is no)',
            resp_lower
        )
        contradiction_penalty += len(contradiction_phrases) * 2.0
        
        # =====================================================
        # 6. RESPONSE SUBSTANTIVENESS
        # =====================================================
        
        # Longer, more developed responses tend to have better argument structure
        # But we score this via information density, not raw length
        
        # Unique content words ratio
        content_words = [w for w in words if len(w) > 3]
        unique_content = set(content_words)
        content_richness = len(unique_content) / max(len(content_words), 1)
        
        # Sweet spot: 0.3-0.7 (too high = no repetition/development, too low = repetitive)
        if content_richness > 0.7:
            richness_score = (1.0 - (content_richness - 0.7) / 0.3) * 5
        elif content_richness < 0.3:
            richness_score = (content_richness / 0.3) * 5
        else:
            richness_score = 5.0
        richness_score = max(richness_score, 0)
        
        # Length bonus (diminishing returns)
        length_score = min(math.log(num_words + 1) / math.log(300), 1.0) * 8  # max 8
        
        # =====================================================
        # 7. CAUSAL REASONING CHAINS
        # =====================================================
        # Detect patterns like "X because Y", "if X then Y", "X therefore Y"
        
        causal_chains = 0
        causal_chain_patterns = [
            r'\b\w+\b.*?\bbecause\b.*?\b\w+\b',
            r'\bif\b.*?\bthen\b',
            r'\b\w+\b.*?\btherefore\b.*?\b\w+\b',
            r'\b\w+\b.*?\bthus\b.*?\b\w+\b',
            r'\bthis means\b.*?\b\w+\b',
            r'\bwhich leads to\b',
            r'\bas a result\b.*?\b\w+\b',
            r'\b\w+\b.*?\bso\b.*?\b\w+\b'  # "X so Y" pattern
        ]
        
        for pattern in causal_chain_patterns:
            causal_chains += len(re.findall(pattern, resp_lower))
        
        causal_chain_score = min(causal_chains / 4.0, 1.0) * 8  # max 8
        
        # =====================================================
        # 8. QUALIFICATION AND NUANCE
        # =====================================================
        # Good logical arguments often include qualifications
        
        qualification_markers = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin most cases\b', r'\btends to\b', r'\boften\b',
            r'\bsometimes\b', r'\bit depends\b', r'\bnot always\b',
            r'\bin some\b', r'\bcertain\b', r'\bparticular\b',
            r'\bto some extent\b', r'\brelatively\b', r'\bcomparatively\b',
            r'\broughly\b', r'\bapproximately\b'
        ]
        
        qual_count = sum(len(re.findall(p, resp_lower)) for p in qualification_markers)
        qualification_score = min(qual_count / 3.0, 1.0) * 5  # max 5
        
        # =====================================================
        # 9. EXPLANATION DEPTH INDICATORS
        # =====================================================
        
        explanation_patterns = [
            r'\bthe reason\b', r'\bthis is because\b', r'\bthe idea\b',
            r'\bthe point\b', r'\bwhat this means\b', r'\bin other words\b',
            r'\bto put it\b', r'\bto clarify\b', r'\bto be clear\b',
            r'\bessentially\b', r'\bfundamentally\b', r'\bat its core\b',
            r'\bthe trade-off\b', r'\bthe trade off\b', r'\bthe key\b',
            r'\bthe difference\b', r'\bthe distinction\b'
        ]
        
        explanation_count = sum(len(re.findall(p, resp_lower)) for p in explanation_patterns)
        explanation_score = min(explanation_count / 2.0, 1.0) * 6  # max 6
        
        # =====================================================
        # 10. MULTI-PERSPECTIVE / COMPARATIVE REASONING
        # =====================================================
        
        comparative_patterns = [
            r'\bon one hand\b', r'\bon the other\b', r'\bcompared to\b',
            r'\bin contrast\b', r'\bwhereas\b', r'\bwhile\b.*?\b(?:also|however)\b',
            r'\bboth\b', r'\beither\b', r'\bneither\b',
            r'\bnot only\b.*?\bbut also\b', r'\bmore\b.*?\bthan\b',
            r'\bless\b.*?\bthan\b'
        ]
        
        comparative_count = sum(len(re.findall(p, resp_lower)) for p in comparative_patterns)
        comparative_score = min(comparative_count / 2.0, 1.0) * 5  # max 5
        
        # =====================================================
        # AGGREGATE SCORING
        # =====================================================
        
        raw_score = (
            marker_density_score * 0.8 +    # up to 12
            variety_score * 0.9 +            # up to 10.8
            sub_score * 0.7 +                # up to 5.6
            flow_score * 1.0 +               # up to 12
            structure_score * 0.9 +          # up to 7.2
            causal_chain_score * 0.9 +       # up to 7.2
            richness_score * 0.5 +           # up to 2.5
            length_score * 0.8 +             # up to 6.4
            qualification_score * 0.6 +      # up to 3.0
            explanation_score * 0.8 +        # up to 4.8
            comparative_score * 0.6 -        # up to 3.0
            contradiction_penalty             # penalty
        )
        
        # Theoretical max ~ 74.5, but practical max is much lower
        # Normalize to 0-100 scale
        final_score = max(0.0, min(raw_score * 1.5, 100.0))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This helps discrimination
        midpoint = 35.0
        steepness = 0.08
        transformed = 100.0 / (1.0 + math.exp(-steepness * (final_score - midpoint)))
        
        # Scale to 0-10 for cleaner output
        result = transformed / 10.0
        
        return round(result, 3)
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 3.0
            return 1.0
        except:
            return 1.0