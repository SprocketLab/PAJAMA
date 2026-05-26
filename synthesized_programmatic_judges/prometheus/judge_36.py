def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis,
    causal/logical connective density, argument chain detection, and contradiction signals.
    
    This variant focuses on:
    1. Discourse marker sophistication and variety
    2. Causal reasoning chain detection
    3. Structural progression (claim -> evidence -> conclusion patterns)
    4. Contradiction and incoherence detection
    5. Referential coherence (pronoun/demonstrative usage indicating connected ideas)
    6. Sentence-to-sentence semantic progression
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
            return 0.5
        
        sentences = re.split(r'(?<=[.!?])\s+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # 1. CAUSAL/LOGICAL CONNECTIVE ANALYSIS
        # These indicate reasoning chains - weighted by sophistication
        causal_connectives = {
            'because': 2.0, 'therefore': 2.5, 'consequently': 2.5, 'thus': 2.0,
            'hence': 2.0, 'since': 1.5, 'due': 1.5, 'owing': 1.5,
            'as a result': 2.5, 'for this reason': 2.5, 'that is why': 2.0,
            'which means': 2.0, 'which leads': 2.0, 'leading to': 1.5,
            'causes': 1.5, 'results in': 2.0, 'stems from': 2.0,
        }
        
        conditional_connectives = {
            'if': 1.0, 'unless': 1.5, 'provided that': 2.0, 'assuming': 1.5,
            'in case': 1.5, 'when': 0.8, 'whenever': 1.0,
            'otherwise': 1.5, 'alternatively': 1.5,
        }
        
        contrastive_connectives = {
            'however': 2.0, 'although': 2.0, 'nevertheless': 2.5, 'nonetheless': 2.5,
            'on the other hand': 2.5, 'in contrast': 2.5, 'conversely': 2.5,
            'despite': 2.0, 'yet': 1.0, 'but': 0.8, 'while': 1.0,
            'even though': 2.0, 'rather': 1.5, 'instead': 1.5,
        }
        
        additive_connectives = {
            'moreover': 2.0, 'furthermore': 2.0, 'additionally': 2.0,
            'in addition': 2.0, 'also': 0.8, 'besides': 1.5,
            'not only': 1.5, 'as well': 1.0, 'equally': 1.5,
        }
        
        elaboration_connectives = {
            'for example': 2.0, 'for instance': 2.0, 'specifically': 2.0,
            'in particular': 2.0, 'such as': 1.5, 'namely': 2.0,
            'to illustrate': 2.5, 'in other words': 2.0, 'that is': 1.5,
            'meaning': 1.0, 'this means': 1.5,
        }
        
        conclusion_markers = {
            'in conclusion': 2.5, 'to summarize': 2.5, 'in summary': 2.5,
            'overall': 2.0, 'ultimately': 2.0, 'in short': 2.0,
            'to sum up': 2.0, 'all in all': 2.0, 'the key point': 2.0,
        }
        
        response_lower = response_clean.lower()
        
        def count_connective_score(connective_dict):
            total = 0.0
            count = 0
            for phrase, weight in connective_dict.items():
                occurrences = len(re.findall(r'\b' + re.escape(phrase) + r'\b', response_lower))
                total += occurrences * weight
                count += occurrences
            return total, count
        
        causal_score, causal_count = count_connective_score(causal_connectives)
        conditional_score, conditional_count = count_connective_score(conditional_connectives)
        contrastive_score, contrastive_count = count_connective_score(contrastive_connectives)
        additive_score, additive_count = count_connective_score(additive_connectives)
        elaboration_score, elaboration_count = count_connective_score(elaboration_connectives)
        conclusion_score, conclusion_count = count_connective_score(conclusion_markers)
        
        total_connective_score = (causal_score + conditional_score + contrastive_score + 
                                   additive_score + elaboration_score + conclusion_score)
        total_connective_count = (causal_count + conditional_count + contrastive_count + 
                                   additive_count + elaboration_count + conclusion_count)
        
        # Connective density normalized by sentence count
        connective_density = total_connective_score / num_sentences
        # Cap at reasonable level
        connective_density_score = min(connective_density, 3.0) / 3.0  # 0-1 scale
        
        # 2. CONNECTIVE VARIETY - using different types indicates sophisticated argumentation
        type_counts = [causal_count, conditional_count, contrastive_count, 
                       additive_count, elaboration_count, conclusion_count]
        types_used = sum(1 for c in type_counts if c > 0)
        variety_score = min(types_used / 4.0, 1.0)  # 0-1, using 4+ types is max
        
        # 3. ARGUMENT CHAIN DETECTION
        # Look for sequences where one sentence's conclusion feeds into next sentence's premise
        chain_score = 0.0
        referential_words = {'this', 'that', 'these', 'those', 'such', 'it', 'they', 'them',
                            'their', 'its', 'here', 'there', 'above', 'below', 'said', 'mentioned'}
        
        demonstrative_starts = 0
        for sent in sentences:
            sent_words = sent.lower().split()
            if sent_words:
                first_word = re.sub(r'[^a-z]', '', sent_words[0])
                first_two = ' '.join(sent_words[:2]).lower() if len(sent_words) > 1 else ''
                if first_word in {'this', 'that', 'these', 'those', 'such', 'here'}:
                    demonstrative_starts += 1
                elif first_two.startswith('by doing') or first_two.startswith('in doing'):
                    demonstrative_starts += 1
        
        # Referential coherence ratio
        ref_count = 0
        for w in words:
            if w in referential_words:
                ref_count += 1
        referential_density = ref_count / num_words
        referential_score = min(referential_density / 0.08, 1.0)  # 0-1
        
        # Demonstrative sentence starts indicate logical chaining
        demo_ratio = demonstrative_starts / num_sentences
        chain_score = min(demo_ratio / 0.3, 1.0)  # 0-1
        
        # 4. STRUCTURAL PROGRESSION DETECTION
        # Check for claim-evidence-conclusion pattern
        
        # Opening quality: does the response start with acknowledgment or clear thesis?
        opening_patterns = [
            r'^(i understand|i can see|it\'s|that\'s|you\'re|i\'m sorry|i hear|let me|imagine|'
            r'consider|think of|here|the|to|when|a |an )',
        ]
        opening_quality = 0.5  # default
        first_sent_lower = sentences[0].lower() if sentences else ""
        
        # Strong openings that establish context
        strong_openers = [
            r'^(i understand|i can see|i hear|i\'m sorry|it\'s completely|that\'s)',
            r'^(imagine|consider|let\'s|here\'s|the key|to understand)',
            r'^(when|in order|before|first)',
        ]
        for pattern in strong_openers:
            if re.match(pattern, first_sent_lower):
                opening_quality = 1.0
                break
        
        # Weak/dismissive openings
        weak_openers = [
            r'^(just|well|so,|hmm|oh|um|uh|yeah|ok so)',
            r'^(i guess|i don\'t know|maybe)',
        ]
        for pattern in weak_openers:
            if re.match(pattern, first_sent_lower):
                opening_quality = 0.2
                break
        
        # 5. CONTRADICTION DETECTION
        contradiction_signals = 0
        
        # Look for self-contradicting patterns
        negation_pairs = [
            (r'\bcan\b', r'\bcannot\b'), (r'\bcan\b', r'\bcan\'t\b'),
            (r'\bwill\b', r'\bwon\'t\b'), (r'\bshould\b', r'\bshouldn\'t\b'),
            (r'\bis\b', r'\bisn\'t\b'), (r'\bdo\b', r'\bdon\'t\b'),
        ]
        
        # Check for hedging that undermines assertions
        strong_assertions = len(re.findall(r'\b(definitely|certainly|always|never|absolutely|must)\b', response_lower))
        hedges_after = len(re.findall(r'\b(maybe|perhaps|might|possibly|probably|not sure|i guess)\b', response_lower))
        
        if strong_assertions > 0 and hedges_after > 0:
            contradiction_signals += min(hedges_after, strong_assertions) * 0.3
        
        # Detect "but" contradictions within same sentence
        for sent in sentences:
            sent_l = sent.lower()
            if ' but ' in sent_l:
                parts = sent_l.split(' but ')
                if len(parts) == 2:
                    # Check if the but-clause negates the first clause significantly
                    p1_words = set(re.findall(r'\b[a-z]+\b', parts[0]))
                    p2_words = set(re.findall(r'\b[a-z]+\b', parts[1]))
                    neg_words = {'not', 'no', 'never', 'don\'t', 'can\'t', 'won\'t', 'shouldn\'t'}
                    if p2_words & neg_words and p1_words & p2_words - neg_words - {'but', 'the', 'a', 'an', 'is', 'are'}:
                        contradiction_signals += 0.5
        
        contradiction_penalty = min(contradiction_signals * 0.3, 1.0)  # 0-1 penalty
        
        # 6. DISMISSIVE/INCOHERENT LANGUAGE DETECTION
        dismissive_patterns = [
            r'\bjust\s+(do|get|try|go|buy|make)\b',
            r'\byou\s+should\s+be\s+able\b',
            r'\bit\'s\s+not\s+that\s+(hard|difficult|bad)\b',
            r'\bjust\s+remember\b',
            r'\bget\s+over\s+it\b',
            r'\bmove\s+on\b',
            r'\bget\s+yourself\s+together\b',
        ]
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        dismissive_penalty = min(dismissive_count * 0.15, 0.6)
        
        # 7. SENTENCE-LEVEL PROGRESSION ANALYSIS
        # Analyze whether sentences build on each other vs. being disconnected
        progression_score = 0.0
        if num_sentences >= 2:
            word_sets = []
            for sent in sentences:
                sw = set(re.findall(r'\b[a-z]{3,}\b', sent.lower()))
                # Remove very common words
                stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                            'would', 'could', 'should', 'will', 'with', 'this', 'that', 'from',
                            'they', 'been', 'said', 'each', 'which', 'their', 'there', 'what',
                            'about', 'than', 'them', 'then', 'some', 'into', 'just', 'your',
                            'also', 'more', 'other', 'when', 'very'}
                sw = sw - stopwords
                word_sets.append(sw)
            
            # Measure progressive overlap between consecutive sentences
            overlaps = []
            for i in range(len(word_sets) - 1):
                if word_sets[i] and word_sets[i + 1]:
                    intersection = word_sets[i] & word_sets[i + 1]
                    union = word_sets[i] | word_sets[i + 1]
                    if union:
                        overlap = len(intersection) / len(union)
                        overlaps.append(overlap)
            
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                # Sweet spot: some overlap (coherent) but not too much (repetitive)
                # Best around 0.1-0.3
                if avg_overlap < 0.05:
                    progression_score = 0.3  # Too disconnected
                elif avg_overlap < 0.1:
                    progression_score = 0.6
                elif avg_overlap <= 0.35:
                    progression_score = 1.0  # Good progressive overlap
                elif avg_overlap <= 0.5:
                    progression_score = 0.7  # Somewhat repetitive
                else:
                    progression_score = 0.4  # Very repetitive
            else:
                progression_score = 0.5
        else:
            progression_score = 0.4  # Single sentence - limited structure
        
        # 8. NUMBERED/STRUCTURED ARGUMENT DETECTION
        # Look for explicit structuring (numbered points, clear sections)
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean))
        has_structure = 1.0 if numbered_items >= 2 else (0.5 if numbered_items == 1 else 0.0)
        
        # Also check for paragraph breaks indicating organized thought
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if p.strip()]
        paragraph_score = min(len(paragraphs) / 3.0, 1.0) if len(paragraphs) > 1 else 0.3
        
        structure_score = max(has_structure, paragraph_score)
        
        # 9. RESPONSE COMPLETENESS AND DEPTH
        # Longer, more developed responses tend to have better argument structure
        # But normalize - very long isn't always better
        length_score = 0.0
        if num_words < 20:
            length_score = 0.2
        elif num_words < 50:
            length_score = 0.5
        elif num_words < 100:
            length_score = 0.7
        elif num_words < 200:
            length_score = 0.9
        else:
            length_score = 1.0
        
        # 10. QUERY RELEVANCE CHECK
        # Arguments should address the query
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower())) if query else set()
        query_stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                          'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                          'would', 'could', 'should', 'will', 'with', 'this', 'that', 'from',
                          'how', 'what', 'where', 'when', 'who', 'why', 'which', 'there',
                          'their', 'about', 'need', 'want', 'know', 'like'}
        query_content = query_words - query_stopwords
        response_words_set = set(words)
        
        if query_content:
            relevance = len(query_content & response_words_set) / len(query_content)
            relevance_score = min(relevance / 0.3, 1.0)
        else:
            relevance_score = 0.5
        
        # 11. EMPATHETIC/ACKNOWLEDGING FRAMING (important for advice-type queries)
        empathy_patterns = [
            r'\b(understand|understandable|completely|totally|absolutely|genuinely)\b',
            r'\b(feel|feeling|feelings|emotions|emotional)\b',
            r'\b(sorry|hear|see|acknowledge|recognize|appreciate)\b',
            r'\bit\'s\s+(okay|ok|fine|natural|normal|perfectly|completely)\b',
        ]
        empathy_count = 0
        for pattern in empathy_patterns:
            empathy_count += len(re.findall(pattern, response_lower))
        
        # Detect if query is emotional/advice-seeking
        emotional_query = bool(re.search(
            r'\b(feeling|feel|stress|sad|frustrated|lonely|heartbroken|upset|worried|anxious|'
            r'struggling|difficulty|comfort|advice|help|cope)\b', query.lower()
        )) if query else False
        
        if emotional_query:
            empathy_score = min(empathy_count / 4.0, 1.0)
        else:
            empathy_score = 0.5  # neutral for non-emotional queries
        
        # 12. IMPERATIVE/COMMANDING vs COLLABORATIVE TONE
        # Collaborative tone indicates better argument structure
        commanding_phrases = len(re.findall(
            r'\b(you need to|you must|you should|you have to|just do|just get|just try)\b', 
            response_lower
        ))
        collaborative_phrases = len(re.findall(
            r'\b(let\'s|we can|you might|you could|consider|perhaps try|one approach|'
            r'it may help|it might be|you may want)\b',
            response_lower
        ))
        
        tone_score = 0.5  # neutral default
        if commanding_phrases + collaborative_phrases > 0:
            tone_ratio = collaborative_phrases / (commanding_phrases + collaborative_phrases)
            tone_score = 0.3 + 0.7 * tone_ratio
        
        # === COMBINE ALL SCORES ===
        # Weighted combination
        weights = {
            'connective_density': 1.5,
            'variety': 1.2,
            'referential': 0.8,
            'chain': 0.7,
            'opening': 0.8,
            'progression': 1.3,
            'structure': 1.0,
            'length': 0.6,
            'relevance': 0.8,
            'empathy': 0.7,
            'tone': 0.6,
        }
        
        scores = {
            'connective_density': connective_density_score,
            'variety': variety_score,
            'referential': referential_score,
            'chain': chain_score,
            'opening': opening_quality,
            'progression': progression_score,
            'structure': structure_score,
            'length': length_score,
            'relevance': relevance_score,
            'empathy': empathy_score,
            'tone': tone_score,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        base_score = weighted_sum / total_weight  # 0-1
        
        # Apply penalties
        final_01 = base_score - contradiction_penalty * 0.15 - dismissive_penalty * 0.2
        final_01 = max(0.0, min(1.0, final_01))
        
        # Map to 1-5 scale
        final_score = 1.0 + final_01 * 4.0
        
        # Round to one decimal
        final_score = round(final_score, 1)
        
        return final_score
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5