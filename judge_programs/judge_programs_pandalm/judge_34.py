def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    analysis: causal/logical connectors, sentence-to-sentence semantic continuity,
    contradiction detection, and argument depth/progression.
    
    This variant focuses on:
    1. Sentence-level logical connector analysis (causal, conditional, comparative)
    2. Topic continuity via shared noun/content word chains between adjacent sentences
    3. Argument depth (claims supported by reasoning/evidence patterns)
    4. Contradiction/repetition detection
    5. Structural progression (intro -> body -> conclusion patterns)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) == 0:
            return 0.0
        
        # Tokenize into sentences
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space or end
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            # Also split on semicolons as logical breaks
            expanded = []
            for s in sents:
                parts = re.split(r';\s*', s)
                expanded.extend([p.strip() for p in parts if p.strip()])
            return expanded
        
        sentences = split_sentences(response)
        num_sentences = len(sentences)
        
        # Extract content words (nouns, verbs, adjectives approximation - longer words)
        def get_content_words(text):
            words = re.findall(r'[a-zA-Z]{3,}', text.lower())
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
                'would', 'could', 'should', 'will', 'their', 'there', 'what',
                'which', 'when', 'where', 'who', 'how', 'that', 'this', 'these',
                'those', 'with', 'from', 'they', 'were', 'also', 'more', 'than',
                'other', 'into', 'some', 'such', 'them', 'then', 'its', 'over',
                'about', 'being', 'does', 'did', 'while'
            }
            return [w for w in words if w not in stop_words]
        
        # ---- SCORE 1: Logical Connector Density and Variety ----
        # Categorize connectors by logical function
        causal_connectors = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bleading to\b', r'\bcaused by\b', r'\bhence\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b'
        ]
        
        conditional_connectors = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b.*\bthen\b', r'\bgiven that\b',
            r'\bwhether\b'
        ]
        
        comparative_connectors = [
            r'\bwhile\b', r'\bwhereas\b', r'\bin contrast\b', r'\bon the other hand\b',
            r'\bhowever\b', r'\bnevertheless\b', r'\balthough\b', r'\bdespite\b',
            r'\bconversely\b', r'\byet\b', r'\bbut\b', r'\binstead\b',
            r'\bunlike\b', r'\bsimilarly\b', r'\blikewise\b', r'\bjust as\b'
        ]
        
        additive_connectors = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bnot only\b',
            r'\bbeyond that\b', r'\bequally\b'
        ]
        
        concluding_connectors = [
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
            r'\bto summarize\b', r'\bin short\b', r'\bultimately\b',
            r'\ball in all\b', r'\btaken together\b'
        ]
        
        elaboration_connectors = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bin other words\b', r'\bto illustrate\b'
        ]
        
        resp_lower = response.lower()
        
        def count_connector_category(patterns, text):
            count = 0
            matched_types = set()
            for p in patterns:
                finds = re.findall(p, text)
                if finds:
                    count += len(finds)
                    matched_types.add(p)
            return count, len(matched_types)
        
        causal_count, causal_variety = count_connector_category(causal_connectors, resp_lower)
        conditional_count, cond_variety = count_connector_category(conditional_connectors, resp_lower)
        comparative_count, comp_variety = count_connector_category(comparative_connectors, resp_lower)
        additive_count, add_variety = count_connector_category(additive_connectors, resp_lower)
        concluding_count, conc_variety = count_connector_category(concluding_connectors, resp_lower)
        elaboration_count, elab_variety = count_connector_category(elaboration_connectors, resp_lower)
        
        total_connectors = (causal_count + conditional_count + comparative_count + 
                           additive_count + concluding_count + elaboration_count)
        
        # Categories present (diversity of logical operations)
        categories_present = sum(1 for c in [causal_count, conditional_count, comparative_count,
                                              additive_count, concluding_count, elaboration_count] if c > 0)
        
        # Normalize connector density per sentence
        connector_density = total_connectors / max(num_sentences, 1)
        # Score: optimal density around 0.5-1.5 connectors per sentence
        connector_score = min(connector_density / 1.0, 1.5) * 10  # max ~15
        
        # Variety bonus
        variety_score = categories_present * 2.5  # max 15
        
        # ---- SCORE 2: Sentence-to-Sentence Topic Continuity (Cohesion Chain) ----
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, num_sentences):
                prev_words = set(get_content_words(sentences[i-1]))
                curr_words = set(get_content_words(sentences[i]))
                if len(prev_words) == 0 or len(curr_words) == 0:
                    continuity_scores.append(0.0)
                else:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    jaccard = overlap / union if union > 0 else 0
                    # Also check if new words are introduced (progression)
                    new_words = len(curr_words - prev_words)
                    progression = new_words / len(curr_words) if len(curr_words) > 0 else 0
                    # Good coherence = some overlap + some new content
                    # Optimal: overlap ~0.2-0.5, progression ~0.5-0.8
                    coherence = jaccard * 0.6 + progression * 0.4
                    continuity_scores.append(coherence)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0
            # Penalize very low continuity (topic jumping) and very high (pure repetition)
            if avg_continuity > 0.8:
                continuity_final = 0.5  # too repetitive
            elif avg_continuity < 0.05:
                continuity_final = 0.1  # no connection
            else:
                continuity_final = avg_continuity / 0.5  # normalize, optimal around 0.3-0.5
            
            continuity_score = min(continuity_final, 1.0) * 15
        else:
            continuity_score = 5.0  # single sentence, neutral
        
        # ---- SCORE 3: Argument Depth and Claim-Evidence Patterns ----
        # Detect claim-evidence patterns: statement followed by elaboration/example
        claim_evidence_patterns = 0
        for i in range(num_sentences - 1):
            curr = sentences[i].lower()
            nxt = sentences[i+1].lower() if i+1 < num_sentences else ""
            
            # Check if next sentence provides evidence/elaboration for current
            evidence_markers = [
                r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
                r'\bthis means\b', r'\bthis is because\b', r'\bin particular\b',
                r'\bspecifically\b', r'\bthis includes\b', r'\bto illustrate\b'
            ]
            for marker in evidence_markers:
                if re.search(marker, nxt):
                    claim_evidence_patterns += 1
                    break
            
            # Check if current makes a claim and next provides reasoning
            reasoning_markers = [
                r'\bbecause\b', r'\bsince\b', r'\bthis is due to\b',
                r'\bthe reason\b', r'\bas a result\b'
            ]
            for marker in reasoning_markers:
                if re.search(marker, nxt):
                    claim_evidence_patterns += 1
                    break
        
        depth_score = min(claim_evidence_patterns * 3, 12)
        
        # ---- SCORE 4: Repetition and Contradiction Detection ----
        # Check for excessive repetition of phrases (3+ word sequences)
        def get_ngrams(text, n):
            words = re.findall(r'[a-zA-Z]+', text.lower())
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        trigrams = get_ngrams(response, 3)
        trigram_counts = Counter(trigrams)
        
        if trigrams:
            max_trigram_freq = max(trigram_counts.values())
            total_trigrams = len(trigrams)
            repetition_ratio = max_trigram_freq / total_trigrams if total_trigrams > 0 else 0
        else:
            repetition_ratio = 0
            max_trigram_freq = 0
        
        # Heavy penalty for extreme repetition
        if max_trigram_freq > 5:
            repetition_penalty = min((max_trigram_freq - 5) * 3, 25)
        elif repetition_ratio > 0.15:
            repetition_penalty = 10
        else:
            repetition_penalty = 0
        
        # Check for sentence-level repetition
        sentence_texts = [re.sub(r'[^a-z\s]', '', s.lower()).strip() for s in sentences]
        unique_sentences = len(set(sentence_texts))
        if num_sentences > 0:
            sentence_uniqueness = unique_sentences / num_sentences
        else:
            sentence_uniqueness = 1.0
        
        if sentence_uniqueness < 0.5:
            repetition_penalty += 15
        elif sentence_uniqueness < 0.8:
            repetition_penalty += 5
        
        # ---- SCORE 5: Structural Progression ----
        # Check for intro-body-conclusion structure
        structural_score = 0
        
        if num_sentences >= 3:
            first_sent = sentences[0].lower()
            last_sent = sentences[-1].lower()
            
            # Introduction patterns: definitions, topic statements
            intro_patterns = [
                r'\bis\s+(a|an|the)\b', r'\brefers to\b', r'\bcan be defined\b',
                r'\bis\s+when\b', r'\bmeans\b', r'\binvolves\b'
            ]
            has_intro = any(re.search(p, first_sent) for p in intro_patterns)
            
            # Conclusion patterns
            conclusion_patterns = [
                r'\bin conclusion\b', r'\boverall\b', r'\btherefore\b',
                r'\bin summary\b', r'\bthus\b', r'\bit is important\b',
                r'\bultimately\b'
            ]
            has_conclusion = any(re.search(p, last_sent) for p in conclusion_patterns)
            
            if has_intro:
                structural_score += 4
            if has_conclusion:
                structural_score += 4
            
            # Check for logical progression through body
            # Body sentences should build on each other
            if num_sentences >= 4:
                structural_score += 2
        elif num_sentences == 2:
            structural_score = 3
        else:
            structural_score = 1
        
        # ---- SCORE 6: Query Relevance via Logical Addressing ----
        query_content = set(get_content_words(query))
        response_content = set(get_content_words(response))
        
        if query_content:
            query_coverage = len(query_content & response_content) / len(query_content)
        else:
            query_coverage = 0.5
        
        relevance_score = query_coverage * 8  # max 8
        
        # ---- SCORE 7: Information Density ----
        # Unique content words per sentence (avoiding fluff)
        all_content_words = get_content_words(response)
        unique_content = len(set(all_content_words))
        
        if num_sentences > 0:
            info_density = unique_content / num_sentences
        else:
            info_density = 0
        
        # Optimal info density: 4-10 unique content words per sentence
        if info_density < 2:
            density_score = info_density * 2
        elif info_density <= 10:
            density_score = 6 + (info_density - 2) * 0.5
        else:
            density_score = 10
        
        # ---- SCORE 8: Sentence Complexity and Subordination ----
        # Complex sentences with subordinate clauses indicate structured reasoning
        subordination_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b', r'\bwhere\b',
            r'\bwhen\b', r'\balthough\b', r'\bwhile\b', r'\bif\b', r'\bunless\b',
            r'\bbecause\b', r'\bsince\b', r'\bafter\b', r'\bbefore\b'
        ]
        
        complex_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            sub_count = sum(1 for m in subordination_markers if re.search(m, sent_lower))
            # Also check for comma-separated clauses
            comma_count = sent.count(',')
            if sub_count >= 1 or comma_count >= 2:
                complex_sentences += 1
        
        if num_sentences > 0:
            complexity_ratio = complex_sentences / num_sentences
        else:
            complexity_ratio = 0
        
        complexity_score = complexity_ratio * 8  # max 8
        
        # ---- SCORE 9: Length Appropriateness ----
        # Not too short (lacks substance) or absurdly long
        word_count = len(response.split())
        if word_count < 5:
            length_score = 1
        elif word_count < 15:
            length_score = 3
        elif word_count < 30:
            length_score = 5
        elif word_count < 80:
            length_score = 7
        elif word_count < 200:
            length_score = 6
        else:
            length_score = 5
        
        # ---- COMBINE SCORES ----
        total = (
            connector_score +       # max ~15
            variety_score +          # max 15
            continuity_score +       # max 15
            depth_score +            # max 12
            structural_score +       # max 10
            relevance_score +        # max 8
            density_score +          # max 10
            complexity_score +       # max 8
            length_score             # max 7
            - repetition_penalty     # penalty up to 40
        )
        
        # Normalize to 0-100 range
        # Theoretical max: ~100, typical good: 50-70
        final_score = max(0.0, min(100.0, total))
        
        return round(final_score, 2)
    
    except Exception as e:
        # Fallback: return a basic score based on length
        try:
            if response and len(response.strip()) > 0:
                return min(len(response.strip().split()) * 0.5, 30.0)
            return 0.0
        except:
            return 0.0