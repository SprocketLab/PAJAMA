def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    1. Discourse marker analysis (causal, contrastive, additive, temporal connectives)
    2. Argument depth via clause complexity and subordination
    3. Logical flow via sentence-to-sentence topic continuity (entity threading)
    4. Absence of contradiction signals
    5. Evidence/example integration patterns
    6. Conclusion/synthesis detection
    
    Returns a score where higher = better logical coherence.
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 1.0
        
        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation, keeping it simple but effective
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            # Also split on newlines that seem like sentence boundaries
            result = []
            for s in sents:
                parts = re.split(r'\n+', s)
                result.extend([p.strip() for p in parts if p.strip()])
            return result
        
        sentences = split_sentences(response_clean)
        num_sentences = len(sentences)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # ============================================================
        # FEATURE 1: Discourse Marker Density and Variety
        # Measures how well the response uses logical connectives
        # ============================================================
        
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\bthis means\b', r'\bthis implies\b',
            r'\bso\b(?=\s+\w)', r'\bcaused by\b', r'\bleads to\b',
            r'\bresulting in\b', r'\bgiven that\b', r'\bin light of\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bnonetheless\b', r'\bon the other hand\b', r'\bin contrast\b',
            r'\bconversely\b', r'\bwhile\b', r'\bwhereas\b', r'\bdespite\b',
            r'\byet\b', r'\binstead\b', r'\brather\b', r'\bstill\b',
            r'\bthat said\b', r'\beven so\b', r'\bon the contrary\b'
        ]
        
        additive_markers = [
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bnot only\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bequally\b', r'\bby the same token\b'
        ]
        
        temporal_sequential = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bpreviously\b', r'\binitially\b', r'\bultimately\b',
            r'\bto begin with\b', r'\bin the first place\b',
            r'\bafter that\b', r'\bbefore this\b', r'\blastly\b'
        ]
        
        exemplification = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bconsider\b', r'\btake\b.*\bfor example\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\blike\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bto sum up\b',
            r'\ball in all\b', r'\bultimately\b', r'\bthe point is\b',
            r'\bessentially\b', r'\bin essence\b', r'\bthe key\b',
            r'\bthe bottom line\b', r'\bthe takeaway\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_markers(patterns, text):
            total = 0
            unique = 0
            for p in patterns:
                matches = re.findall(p, text)
                if matches:
                    total += len(matches)
                    unique += 1
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers, resp_lower)
        contrastive_count, contrastive_unique = count_markers(contrastive_markers, resp_lower)
        additive_count, additive_unique = count_markers(additive_markers, resp_lower)
        temporal_count, temporal_unique = count_markers(temporal_sequential, resp_lower)
        exemplification_count, exemplification_unique = count_markers(exemplification, resp_lower)
        conclusion_count, conclusion_unique = count_markers(conclusion_markers, resp_lower)
        
        total_markers = (causal_count + contrastive_count + additive_count + 
                        temporal_count + exemplification_count + conclusion_count)
        total_unique_types = (causal_unique + contrastive_unique + additive_unique + 
                             temporal_unique + exemplification_unique + conclusion_unique)
        
        # Categories present (out of 6)
        categories_present = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                              temporal_count, exemplification_count, conclusion_count] if c > 0)
        
        # Normalize marker density per 100 words
        marker_density = (total_markers / max(num_words, 1)) * 100
        # Optimal range: 3-12 per 100 words
        if marker_density <= 12:
            marker_density_score = min(marker_density / 8.0, 1.0)
        else:
            marker_density_score = max(0.3, 1.0 - (marker_density - 12) / 20.0)
        
        # Variety score
        marker_variety_score = min(total_unique_types / 8.0, 1.0)
        
        # Category breadth score
        category_score = min(categories_present / 3.5, 1.0)
        
        discourse_score = (marker_density_score * 0.4 + marker_variety_score * 0.3 + category_score * 0.3)
        
        # ============================================================
        # FEATURE 2: Clause Complexity / Subordination Depth
        # Complex arguments use subordinate clauses
        # ============================================================
        
        subordinators = [
            r'\bif\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bwhere\b', r'\bthat\b', r'\bwhich\b', r'\bwho\b',
            r'\bwhom\b', r'\bwhose\b', r'\bbecause\b', r'\bunless\b',
            r'\buntil\b', r'\bafter\b', r'\bbefore\b', r'\bonce\b',
            r'\bwhenever\b', r'\bwherever\b', r'\bwhether\b',
            r'\beven though\b', r'\bso that\b', r'\bin order to\b',
            r'\bprovided that\b', r'\bas long as\b'
        ]
        
        subordination_count = 0
        for pattern in subordinators:
            subordination_count += len(re.findall(pattern, resp_lower))
        
        # Commas often indicate clause boundaries
        comma_count = response_clean.count(',')
        semicolon_count = response_clean.count(';')
        colon_count = response_clean.count(':')
        
        # Average subordinations per sentence
        avg_subordination = subordination_count / max(num_sentences, 1)
        subordination_score = min(avg_subordination / 2.5, 1.0)
        
        # Punctuation complexity per sentence
        punct_complexity = (comma_count + semicolon_count * 2 + colon_count * 1.5) / max(num_sentences, 1)
        punct_score = min(punct_complexity / 3.0, 1.0)
        
        complexity_score = subordination_score * 0.6 + punct_score * 0.4
        
        # ============================================================
        # FEATURE 3: Entity Threading / Topic Continuity
        # Good logical flow means consecutive sentences share entities/concepts
        # ============================================================
        
        def get_content_words(text):
            """Extract content words (nouns, verbs approximated by length and not being stopwords)."""
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'don', 'now', 'and',
                'but', 'or', 'if', 'it', 'its', 'this', 'that', 'these', 'those',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
                'about', 'up', 'down', 'get', 'got', 'like', 'also', 'much',
                'well', 'back', 'even', 'still', 'way', 'take', 'come', 'make',
                'know', 'think', 'see', 'go', 'thing', 'things', 'really', 'one',
                'two', 'first', 'new', 'good', 'time', 'long', 'great', 'little',
                'right', 'old', 'big', 'high', 'small', 'large', 'say', 'said',
                'going', 'something', 'anything', 'nothing', 'everything', 'been',
                'being', 'having', 'doing', 'made', 'put', 'use', 'used', 'using'
            }
            w = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(w) - stopwords
        
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if words_a and words_b:
                    # Overlap ratio
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    # We want moderate overlap (not zero, not identical)
                    overlap_ratio = overlap / max(union, 1)
                    # Optimal: 0.1 - 0.5
                    if overlap_ratio < 0.05:
                        cont = 0.1  # No continuity - topic jump
                    elif overlap_ratio > 0.7:
                        cont = 0.5  # Too repetitive
                    else:
                        cont = min(overlap_ratio / 0.3, 1.0)
                    continuity_scores.append(cont)
                else:
                    continuity_scores.append(0.3)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.5
            # Consistency of continuity (low variance = smooth flow)
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                variance_c = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                consistency = max(0, 1.0 - math.sqrt(variance_c) * 2)
            else:
                consistency = 0.7
            
            continuity_score = avg_continuity * 0.6 + consistency * 0.4
        else:
            continuity_score = 0.4  # Single sentence - limited flow assessment
        
        # ============================================================
        # FEATURE 4: Argument Structure Detection
        # Look for premise-conclusion patterns, conditional reasoning
        # ============================================================
        
        # Conditional reasoning patterns
        conditional_patterns = [
            r'\bif\b.*\bthen\b', r'\bif\b.*\bwould\b', r'\bif\b.*\bcould\b',
            r'\bif\b.*\bwill\b', r'\bif\b.*\bmight\b',
            r'\bwhen\b.*\bthen\b', r'\bgiven\b.*\bthen\b',
            r'\bassuming\b.*\bthen\b', r'\bsuppose\b.*\bthen\b'
        ]
        
        conditional_count = 0
        for p in conditional_patterns:
            conditional_count += len(re.findall(p, resp_lower))
        
        # Premise-conclusion patterns
        premise_conclusion_patterns = [
            r'\bsince\b.*\b(therefore|thus|so|hence)\b',
            r'\bbecause\b.*\b(therefore|thus|so|hence|this means)\b',
            r'\bgiven that\b.*\b(it follows|therefore|thus|we can)\b',
            r'\bthe fact that\b.*\b(means|implies|suggests|shows)\b'
        ]
        
        pc_count = 0
        for p in premise_conclusion_patterns:
            pc_count += len(re.findall(p, resp_lower, re.DOTALL))
        
        # Qualification/nuance patterns (sign of sophisticated reasoning)
        qualification_patterns = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin most cases\b', r'\bit depends\b', r'\bnot always\b',
            r'\bthere are exceptions\b', r'\bto some extent\b',
            r'\bto a degree\b', r'\bin some cases\b', r'\boften\b',
            r'\btends to\b', r'\bcan be\b', r'\bmay be\b',
            r'\bnot necessarily\b', r'\bthe trade-off\b'
        ]
        
        qualification_count = 0
        for p in qualification_patterns:
            qualification_count += len(re.findall(p, resp_lower))
        
        argument_score = min(1.0, (
            min(conditional_count, 3) * 0.15 +
            min(pc_count, 2) * 0.2 +
            min(qualification_count, 5) * 0.08 +
            min(causal_count, 4) * 0.1
        ))
        
        # ============================================================
        # FEATURE 5: Contradiction / Incoherence Detection (penalty)
        # ============================================================
        
        contradiction_patterns = [
            r'\bbut\s+(?:actually|wait|no)\b',
            r'\bi mean\b.*\bactually\b',
            r'\bnevermind\b', r'\bscratch that\b',
            r'\bforget what i said\b',
        ]
        
        contradiction_count = 0
        for p in contradiction_patterns:
            contradiction_count += len(re.findall(p, resp_lower))
        
        # Repetition detection (saying the same thing multiple times = weak structure)
        if num_sentences >= 3:
            sent_word_sets = [get_content_words(s) for s in sentences]
            high_overlap_pairs = 0
            total_pairs = 0
            for i in range(len(sent_word_sets)):
                for j in range(i + 2, len(sent_word_sets)):  # Skip adjacent (those are continuity)
                    if sent_word_sets[i] and sent_word_sets[j]:
                        overlap = len(sent_word_sets[i] & sent_word_sets[j])
                        smaller = min(len(sent_word_sets[i]), len(sent_word_sets[j]))
                        if smaller > 0 and overlap / smaller > 0.7:
                            high_overlap_pairs += 1
                        total_pairs += 1
            
            repetition_ratio = high_overlap_pairs / max(total_pairs, 1)
        else:
            repetition_ratio = 0
        
        incoherence_penalty = min(0.3, contradiction_count * 0.1 + repetition_ratio * 0.3)
        
        # ============================================================
        # FEATURE 6: Response Substance and Engagement with Query
        # ============================================================
        
        query_content = get_content_words(query)
        response_content = get_content_words(response_clean)
        
        if query_content and response_content:
            query_engagement = len(query_content & response_content) / max(len(query_content), 1)
            query_engagement_score = min(query_engagement / 0.3, 1.0)
        else:
            query_engagement_score = 0.3
        
        # ============================================================
        # FEATURE 7: Response Length and Completeness
        # ============================================================
        
        # Longer responses (to a point) tend to have more developed arguments
        length_score = min(1.0, math.log(max(num_words, 1) + 1) / math.log(200))
        
        # Check if response seems truncated
        truncated = response_clean[-1] not in '.!?")\']' if response_clean else True
        truncation_penalty = 0.05 if truncated else 0
        
        # Check for multiple developed points
        # Approximate by looking at paragraph-like structures or multiple argument markers
        developed_points = max(1, causal_count + contrastive_count + exemplification_count)
        development_score = min(developed_points / 4.0, 1.0)
        
        # ============================================================
        # FEATURE 8: Explanatory depth - ratio of explanatory to assertive sentences
        # ============================================================
        
        explanatory_starters = [
            r'^(this|that|it|these|those)\s+(is|are|was|were|means|implies|suggests|shows|indicates)',
            r'^(the\s+reason|the\s+point|the\s+key|the\s+idea)',
            r'^(in\s+other\s+words|put\s+differently|to\s+clarify)',
            r'^(essentially|basically|fundamentally)',
        ]
        
        explanatory_count = 0
        for sent in sentences:
            sent_stripped = sent.strip().lower()
            for p in explanatory_starters:
                if re.match(p, sent_stripped):
                    explanatory_count += 1
                    break
        
        explanatory_ratio = explanatory_count / max(num_sentences, 1)
        explanatory_score = min(explanatory_ratio / 0.2, 1.0) * 0.5 + 0.3  # baseline
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        # Weighted combination
        raw_score = (
            discourse_score * 2.0 +          # Discourse markers: strong signal
            complexity_score * 1.5 +          # Clause complexity
            continuity_score * 1.8 +          # Topic continuity / flow
            argument_score * 2.2 +            # Argument structure
            query_engagement_score * 1.0 +    # Query relevance
            length_score * 1.5 +              # Length/completeness
            development_score * 1.5 +         # Multiple developed points
            explanatory_score * 0.5 -         # Explanatory depth
            incoherence_penalty * 3.0 -       # Contradiction penalty
            truncation_penalty * 1.0          # Truncation penalty
        )
        
        # Normalize to 0-10 range
        max_possible = 2.0 + 1.5 + 1.8 + 2.2 + 1.0 + 1.5 + 1.5 + 0.5  # = 12.0
        normalized = (raw_score / max_possible) * 10.0
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a middling score based on length
        try:
            word_count = len(response.split()) if response else 0
            return min(5.0, max(1.0, word_count / 20.0))
        except:
            return 3.0