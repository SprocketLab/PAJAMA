def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectors)
    - Sentence-to-sentence semantic progression (topic continuity via shared noun phrases)
    - Argument depth detection (nested reasoning chains)
    - Contradiction/inconsistency detection via negation patterns
    - Structural completeness (intro-body-conclusion pattern)
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
        
        # Split into sentences more carefully
        def split_sentences(text):
            # Handle common abbreviations to avoid false splits
            temp = text
            abbrevs = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.', 'St.']
            for ab in abbrevs:
                temp = temp.replace(ab, ab.replace('.', '<DOT>'))
            # Split on sentence-ending punctuation
            parts = re.split(r'(?<=[.!?])\s+', temp)
            parts = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]
            # Also split on newlines if they seem to separate ideas
            final = []
            for p in parts:
                subparts = re.split(r'\n+', p)
                final.extend([s.strip() for s in subparts if s.strip()])
            return final
        
        sentences = split_sentences(response_clean)
        num_sentences = len(sentences)
        
        # ---- 1. DISCOURSE MARKER DENSITY AND VARIETY ----
        # Causal connectors indicate reasoning chains
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b', r'\bowing to\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bit follows\b', r'\bso that\b',
            r'\bfor this reason\b', r'\bgiven that\b', r'\bin order to\b',
            r'\bleads to\b', r'\bresults in\b', r'\bcaused by\b'
        ]
        
        # Contrastive markers indicate nuanced thinking
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b', r'\bwhereas\b',
            r'\bdespite\b', r'\byet\b', r'\binstead\b', r'\brather\b',
            r'\bconversely\b', r'\bnonetheless\b', r'\bthat said\b', r'\beven so\b'
        ]
        
        # Additive/elaboration markers indicate building arguments
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', r'\bin addition\b',
            r'\balso\b', r'\bsimilarly\b', r'\blikewise\b', r'\bnot only\b',
            r'\bwhat\'s more\b', r'\bon top of\b', r'\bbesides\b', r'\bequally\b'
        ]
        
        # Exemplification markers indicate grounding arguments
        exemplification_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b', r'\blike\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bto illustrate\b',
            r'\bconsider\b', r'\btake\b', r'\bnamely\b', r'\be\.g\.\b'
        ]
        
        # Conclusion/summary markers
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b', r'\boverall\b',
            r'\bin short\b', r'\bultimately\b', r'\ball in all\b', r'\bthe point is\b',
            r'\bin essence\b', r'\bthe key\b', r'\bto sum up\b'
        ]
        
        # Conditional/hypothetical reasoning
        conditional_markers = [
            r'\bif\b', r'\bassuming\b', r'\bsuppose\b', r'\bwould\b', r'\bcould\b',
            r'\bin the case\b', r'\bprovided that\b', r'\bunless\b', r'\bwhen\b'
        ]
        
        response_lower = response_clean.lower()
        
        def count_markers(markers):
            total = 0
            unique = 0
            for m in markers:
                found = len(re.findall(m, response_lower))
                if found > 0:
                    unique += 1
                    total += found
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers)
        contrastive_count, contrastive_unique = count_markers(contrastive_markers)
        additive_count, additive_unique = count_markers(additive_markers)
        exemplification_count, exemplification_unique = count_markers(exemplification_markers)
        conclusion_count, conclusion_unique = count_markers(conclusion_markers)
        conditional_count, conditional_unique = count_markers(conditional_markers)
        
        total_markers = (causal_count + contrastive_count + additive_count + 
                        exemplification_count + conclusion_count + conditional_count)
        total_unique = (causal_unique + contrastive_unique + additive_unique + 
                       exemplification_unique + conclusion_unique + conditional_unique)
        
        word_count = len(response_clean.split())
        if word_count == 0:
            return 0.5
        
        # Marker density per 100 words
        marker_density = (total_markers / max(word_count, 1)) * 100
        # Cap and normalize: ideal is around 3-8 markers per 100 words
        marker_density_score = min(marker_density / 6.0, 1.5) * 10  # max ~15
        
        # Variety of marker types used (out of 6 categories)
        categories_used = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                           exemplification_count, conclusion_count, conditional_count] if c > 0)
        marker_variety_score = (categories_used / 6.0) * 12  # max 12
        
        # Unique marker diversity
        unique_diversity_score = min(total_unique / 8.0, 1.0) * 8  # max 8
        
        # ---- 2. SENTENCE-TO-SENTENCE TOPIC CONTINUITY ----
        # Extract content words (nouns, verbs approximated by longer words)
        def get_content_words(text):
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # Remove very common words
            stopwords = {
                'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
                'their', 'them', 'will', 'would', 'could', 'should', 'about',
                'which', 'when', 'where', 'what', 'there', 'these', 'those',
                'than', 'then', 'into', 'some', 'more', 'very', 'just', 'also',
                'your', 'does', 'doing', 'being', 'having', 'other', 'each',
                'make', 'like', 'over', 'such', 'only', 'most', 'same', 'after',
                'before', 'between', 'under', 'here', 'much', 'many', 'well',
                'back', 'even', 'still', 'good', 'know', 'need', 'want', 'look',
                'think', 'come', 'made', 'find', 'going', 'long', 'down', 'take'
            }
            return set(words) - stopwords
        
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, min(num_sentences, 30)):  # limit to avoid slowness
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union_size = len(prev_words | curr_words)
                    # Modified overlap: we want some continuity but not pure repetition
                    if union_size > 0:
                        ratio = overlap / union_size
                        # Ideal continuity is moderate (0.1-0.4)
                        if 0.05 <= ratio <= 0.5:
                            continuity_scores.append(1.0)
                        elif ratio > 0.5:
                            continuity_scores.append(0.6)  # too repetitive
                        else:
                            continuity_scores.append(0.3)  # disconnected
                    else:
                        continuity_scores.append(0.3)
                else:
                    continuity_scores.append(0.4)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.5
            continuity_score = avg_continuity * 12  # max 12
        else:
            continuity_score = 5.0  # single sentence gets moderate score
        
        # ---- 3. ARGUMENT DEPTH: NESTED REASONING CHAINS ----
        # Detect chains like "because X, therefore Y" or "if X then Y, which means Z"
        
        # Count reasoning chain indicators
        chain_patterns = [
            r'\bbecause\b.*?\b(therefore|thus|so|hence|this means)\b',
            r'\bif\b.*?\b(then|would|could|means)\b',
            r'\bsince\b.*?\b(therefore|thus|it follows|consequently)\b',
            r'\bnot only\b.*?\bbut also\b',
            r'\bon one hand\b.*?\bon the other\b',
            r'\bfirst\b.*?\b(second|then|next|finally)\b',
            r'\bwhile\b.*?\b(however|but|nevertheless)\b',
        ]
        
        chain_count = 0
        for pattern in chain_patterns:
            chain_count += len(re.findall(pattern, response_lower, re.DOTALL))
        
        chain_score = min(chain_count * 2.5, 10)  # max 10
        
        # ---- 4. STRUCTURAL PROGRESSION ----
        # Check if response has a discernible structure: opening, development, conclusion
        
        structure_score = 0
        
        if num_sentences >= 3:
            # Opening: first sentence/paragraph addresses the query
            query_words = get_content_words(query)
            first_sent_words = get_content_words(sentences[0])
            if query_words and first_sent_words:
                opening_relevance = len(query_words & first_sent_words) / max(len(query_words), 1)
                structure_score += min(opening_relevance * 4, 3)
            
            # Development: middle sentences elaborate
            mid_start = max(1, num_sentences // 4)
            mid_end = min(num_sentences - 1, 3 * num_sentences // 4 + 1)
            mid_sentences = sentences[mid_start:mid_end]
            if mid_sentences:
                mid_text = ' '.join(mid_sentences).lower()
                # Check for elaboration signals
                elaboration_signals = [
                    r'\bfor example\b', r'\bspecifically\b', r'\bin particular\b',
                    r'\bthis\b', r'\bthat\b', r'\bwhich\b', r'\bsuch\b',
                    r'\bmore\b', r'\bfurther\b', r'\banother\b'
                ]
                elab_count = sum(1 for p in elaboration_signals if re.search(p, mid_text))
                structure_score += min(elab_count * 0.6, 3)
            
            # Conclusion or wrapping up in last portion
            last_sentences = sentences[-min(2, num_sentences):]
            last_text = ' '.join(last_sentences).lower()
            conclusion_signals = [
                r'\boverall\b', r'\bin summary\b', r'\bultimately\b', r'\bso\b',
                r'\bthe key\b', r'\bin short\b', r'\btherefore\b', r'\bhope\b',
                r'\bgood luck\b', r'\bin conclusion\b', r'\ball in all\b'
            ]
            concl_count = sum(1 for p in conclusion_signals if re.search(p, last_text))
            structure_score += min(concl_count * 1.5, 3)
        
        structure_score = min(structure_score, 9)  # max 9
        
        # ---- 5. INFORMATION DENSITY AND SPECIFICITY ----
        # Specific claims, numbers, names, references indicate substantive arguments
        
        # Count specific entities: numbers, proper nouns (capitalized words mid-sentence), 
        # technical terms, quoted phrases
        
        number_count = len(re.findall(r'\b\d+[\d,.]*\b', response_clean))
        # Proper nouns (capitalized words not at start of sentence)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response_clean)
        proper_noun_count = len(proper_nouns)
        # Quoted or referenced material
        quote_count = len(re.findall(r'["\*].*?["\*]', response_clean))
        # Parenthetical explanations (indicate precision)
        paren_count = len(re.findall(r'\(.*?\)', response_clean))
        
        specificity_raw = (number_count * 0.5 + proper_noun_count * 0.4 + 
                          quote_count * 0.6 + paren_count * 0.5)
        specificity_score = min(specificity_raw, 8)  # max 8
        
        # ---- 6. ABSENCE OF LOGICAL FLAWS ----
        # Detect potential issues: self-contradiction, vagueness, non-sequiturs
        
        flaw_penalty = 0
        
        # Excessive vagueness / empty filler
        vague_phrases = [
            r'\bkind of\b', r'\bsort of\b', r'\bmaybe\b', r'\bprobably\b',
            r'\bi guess\b', r'\bi think maybe\b', r'\bnot sure\b',
            r'\bi don\'t know\b', r'\bit depends\b'
        ]
        vague_count = sum(len(re.findall(p, response_lower)) for p in vague_phrases)
        vague_ratio = vague_count / max(num_sentences, 1)
        if vague_ratio > 0.5:
            flaw_penalty += 3
        elif vague_ratio > 0.25:
            flaw_penalty += 1.5
        
        # Self-contradiction detection: sentence says X, later says not X
        # Simple approach: find negation patterns that contradict earlier assertions
        negation_patterns = re.findall(r'\b(not|never|no|don\'t|doesn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|cannot|can\'t|won\'t)\s+(\w+)', response_lower)
        affirmation_words = set()
        for sent in sentences:
            words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower()))
            affirmation_words.update(words)
        
        # Check for potential contradictions (simplified)
        contradiction_signals = 0
        for neg_match in negation_patterns:
            negated_word = neg_match[1]
            if negated_word in affirmation_words and len(negated_word) > 4:
                # This is very rough - just a signal, not definitive
                contradiction_signals += 0.3
        
        if contradiction_signals > 2:
            flaw_penalty += 2
        
        # Repetition penalty: same idea stated multiple times
        if num_sentences >= 4:
            sent_word_sets = [get_content_words(s) for s in sentences[:20]]
            high_overlap_count = 0
            for i in range(len(sent_word_sets)):
                for j in range(i + 2, min(i + 5, len(sent_word_sets))):
                    if sent_word_sets[i] and sent_word_sets[j]:
                        overlap = len(sent_word_sets[i] & sent_word_sets[j])
                        min_size = min(len(sent_word_sets[i]), len(sent_word_sets[j]))
                        if min_size > 0 and overlap / min_size > 0.7:
                            high_overlap_count += 1
            if high_overlap_count > 3:
                flaw_penalty += 2
        
        flaw_penalty = min(flaw_penalty, 7)
        
        # ---- 7. RESPONSE LENGTH APPROPRIATENESS ----
        # Longer responses tend to have more room for logical structure
        # But we don't want to just reward length—we want length that's used well
        
        length_factor = 1.0
        if word_count < 20:
            length_factor = 0.5
        elif word_count < 40:
            length_factor = 0.7
        elif word_count < 80:
            length_factor = 0.85
        elif word_count > 500:
            length_factor = 1.05
        else:
            length_factor = 1.0
        
        # ---- 8. QUERY-RESPONSE ALIGNMENT ----
        # Does the response actually address the logical structure the query demands?
        
        query_lower = query.lower()
        alignment_score = 0
        
        # If query asks "why", response should have causal reasoning
        if re.search(r'\bwhy\b', query_lower):
            if causal_count >= 1:
                alignment_score += 2
        
        # If query asks "how", response should have procedural/explanatory structure
        if re.search(r'\bhow\b', query_lower):
            procedural_markers = len(re.findall(r'\b(first|then|next|step|by|through|using|start|begin)\b', response_lower))
            if procedural_markers >= 2:
                alignment_score += 2
        
        # If query asks for comparison, response should have contrastive markers
        if re.search(r'\b(compare|versus|vs|difference|better|worse)\b', query_lower):
            if contrastive_count >= 1:
                alignment_score += 2
        
        # If query asks "is there" or yes/no, response should take a clear position
        if re.search(r'\b(is there|are there|can|should|does|do you)\b', query_lower):
            position_markers = len(re.findall(r'\b(yes|no|absolutely|definitely|certainly|indeed|essentially)\b', response_lower))
            if position_markers >= 1:
                alignment_score += 1.5
        
        # General query-response content overlap
        query_content = get_content_words(query)
        response_content = get_content_words(response_clean)
        if query_content and response_content:
            relevance = len(query_content & response_content) / max(len(query_content), 1)
            alignment_score += min(relevance * 3, 3)
        
        alignment_score = min(alignment_score, 7)  # max 7
        
        # ---- COMBINE ALL SCORES ----
        raw_score = (
            marker_density_score * 1.0 +      # max ~15
            marker_variety_score * 1.0 +       # max 12
            unique_diversity_score * 0.8 +     # max 6.4
            continuity_score * 1.0 +           # max 12
            chain_score * 1.2 +                # max 12
            structure_score * 1.0 +            # max 9
            specificity_score * 0.8 +          # max 6.4
            alignment_score * 1.0 -            # max 7
            flaw_penalty * 1.5                 # max -10.5
        )
        
        # Apply length factor
        raw_score *= length_factor
        
        # Normalize to 0-10 range
        # Theoretical max is roughly 80, typical good response ~40-60
        normalized = (raw_score / 55.0) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 2)
    
    except Exception as e:
        # Fallback: return a middle score
        try:
            if response and len(response.strip()) > 50:
                return 4.0
            return 2.0
        except:
            return 2.0