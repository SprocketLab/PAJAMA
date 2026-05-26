def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    1. Discourse marker analysis (causal, contrastive, additive, temporal connectives)
    2. Sentence-level coherence via topic continuity (entity/noun tracking across sentences)
    3. Argument depth detection (claim-evidence-conclusion patterns)
    4. Logical fallacy/weakness indicators
    5. Paragraph-level structural progression
    
    Returns a score where higher = better logical coherence.
    """
    import re
    import math
    from collections import Counter, defaultdict
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # ============================================================
        # 1. DISCOURSE MARKER ANALYSIS
        # Evaluate the presence and variety of logical connectives
        # ============================================================
        
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\bthis means\b', r'\bthis leads to\b',
            r'\bcaused by\b', r'\bresulting in\b', r'\bwhich means\b',
            r'\baccordingly\b', r'\bthis is why\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\byet\b', r'\bconversely\b',
            r'\binstead\b', r'\bon the contrary\b', r'\beven though\b',
            r'\bnonetheless\b', r'\brather than\b'
        ]
        
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bnot only\b', r'\bas well as\b',
            r'\bwhat\'s more\b', r'\bbesides\b', r'\bon top of\b'
        ]
        
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bthen\b',
            r'\bnext\b', r'\bfinally\b', r'\bafter\b', r'\bbefore\b',
            r'\bsubsequently\b', r'\bpreviously\b', r'\binitially\b',
            r'\beventually\b', r'\bmeanwhile\b', r'\bonce\b',
            r'\bto begin\b', r'\bto start\b', r'\blast\b', r'\blastly\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\bto sum up\b', r'\bin short\b',
            r'\ball in all\b', r'\bultimately\b', r'\bin the end\b',
            r'\btaken together\b', r'\bthe bottom line\b'
        ]
        
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\be\.g\.\b', r'\bin other words\b', r'\bthis includes\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_markers(patterns):
            total = 0
            unique = 0
            for p in patterns:
                matches = re.findall(p, resp_lower)
                if matches:
                    unique += 1
                    total += len(matches)
            return total, unique
        
        causal_count, causal_unique = count_markers(causal_markers)
        contrastive_count, contrastive_unique = count_markers(contrastive_markers)
        additive_count, additive_unique = count_markers(additive_markers)
        temporal_count, temporal_unique = count_markers(temporal_markers)
        conclusion_count, conclusion_unique = count_markers(conclusion_markers)
        elaboration_count, elaboration_unique = count_markers(elaboration_markers)
        
        total_marker_count = (causal_count + contrastive_count + additive_count + 
                              temporal_count + conclusion_count + elaboration_count)
        total_marker_unique = (causal_unique + contrastive_unique + additive_unique + 
                               temporal_unique + conclusion_unique + elaboration_unique)
        
        # Categories present (out of 6)
        categories_present = sum(1 for c in [causal_count, contrastive_count, additive_count,
                                              temporal_count, conclusion_count, elaboration_count] if c > 0)
        
        # Normalized marker density (per 100 words)
        marker_density = (total_marker_count / num_words) * 100
        # Ideal density: around 3-8 per 100 words
        if marker_density <= 8:
            density_score = min(marker_density / 5.0, 1.0)
        else:
            density_score = max(0.3, 1.0 - (marker_density - 8) / 15.0)
        
        # Variety score
        variety_score = min(total_marker_unique / 8.0, 1.0)
        
        # Category breadth score
        category_score = min(categories_present / 4.0, 1.0)
        
        discourse_score = (density_score * 0.35 + variety_score * 0.35 + category_score * 0.3)
        
        # ============================================================
        # 2. TOPIC CONTINUITY / ENTITY COHERENCE
        # Track content words across consecutive sentences
        # ============================================================
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'although',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'whose',
            'also', 'however', 'therefore', 'thus', 'hence', 'about', 'up',
            'down', 'still', 'already', 'even', 'much', 'many', 'well', 'back',
            'get', 'got', 'make', 'made', 'like', 'know', 'think', 'see', 'come',
            'go', 'take', 'want', 'use', 'find', 'give', 'tell', 'say', 'said'
        }
        
        def get_content_words(text):
            w = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return set(w) - stopwords
        
        if num_sentences >= 2:
            continuity_scores = []
            for i in range(1, len(sentences)):
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    jaccard = overlap / union if union > 0 else 0
                    # Also check for partial word stem overlap
                    prev_stems = {w[:5] for w in prev_words if len(w) >= 5}
                    curr_stems = {w[:5] for w in curr_words if len(w) >= 5}
                    stem_overlap = len(prev_stems & curr_stems) / max(len(prev_stems | curr_stems), 1)
                    combined = max(jaccard, stem_overlap)
                    continuity_scores.append(combined)
                else:
                    continuity_scores.append(0.1)
            
            avg_continuity = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0
            # Check for consistency (low variance is better)
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                var_c = sum((x - mean_c)**2 for x in continuity_scores) / len(continuity_scores)
                consistency = 1.0 / (1.0 + var_c * 10)
            else:
                consistency = 0.5
            
            # Ideal continuity: 0.05-0.35 (some overlap but not repetitive)
            if avg_continuity < 0.02:
                continuity_quality = 0.2
            elif avg_continuity < 0.05:
                continuity_quality = 0.4 + (avg_continuity / 0.05) * 0.3
            elif avg_continuity <= 0.35:
                continuity_quality = 0.7 + (avg_continuity / 0.35) * 0.3
            else:
                continuity_quality = max(0.5, 1.0 - (avg_continuity - 0.35))
            
            coherence_score = continuity_quality * 0.6 + consistency * 0.4
        else:
            coherence_score = 0.4
        
        # ============================================================
        # 3. ARGUMENT DEPTH & STRUCTURE
        # Detect claim-evidence-conclusion patterns
        # ============================================================
        
        # Claim indicators
        claim_patterns = [
            r'\bi (?:believe|think|argue|contend|maintain|suggest)\b',
            r'\bit is (?:important|essential|crucial|clear|evident|necessary)\b',
            r'\bthe (?:key|main|primary|central|fundamental) (?:point|issue|reason|factor)\b',
            r'\bshould\b', r'\bmust\b', r'\bneed to\b', r'\bimportant to\b'
        ]
        
        # Evidence indicators
        evidence_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\baccording to\b', r'\bresearch\b', r'\bstudies?\b',
            r'\bdata\b', r'\bevidence\b', r'\bstatistics?\b',
            r'\bin fact\b', r'\bspecifically\b', r'\bdemonstrat\w+\b',
            r'\bshow[sn]?\b', r'\bprov\w+\b', r'\bindicate\w*\b'
        ]
        
        # Reasoning indicators (beyond simple causal)
        reasoning_patterns = [
            r'\bthis (?:means|implies|suggests|indicates|shows)\b',
            r'\bif\b.*\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bit follows\b', r'\bwe can (?:see|conclude|infer)\b',
            r'\bthe reason\b', r'\bthis is because\b',
            r'\bconsidering\b', r'\btaking into account\b'
        ]
        
        claim_count = sum(len(re.findall(p, resp_lower)) for p in claim_patterns)
        evidence_count = sum(len(re.findall(p, resp_lower)) for p in evidence_patterns)
        reasoning_count = sum(len(re.findall(p, resp_lower)) for p in reasoning_patterns)
        
        # Argument depth: presence of all three components
        has_claims = min(claim_count / 2.0, 1.0)
        has_evidence = min(evidence_count / 2.0, 1.0)
        has_reasoning = min(reasoning_count / 2.0, 1.0)
        
        # Bonus for having all three (synergy)
        components_present = sum(1 for x in [has_claims, has_evidence, has_reasoning] if x > 0.3)
        synergy_bonus = 0.15 * max(0, components_present - 1)
        
        argument_depth_score = (has_claims * 0.3 + has_evidence * 0.35 + has_reasoning * 0.35 + synergy_bonus)
        argument_depth_score = min(argument_depth_score, 1.0)
        
        # ============================================================
        # 4. LOGICAL WEAKNESS DETECTION
        # Penalize contradictions, vagueness, hedging excess
        # ============================================================
        
        # Contradiction indicators
        contradiction_patterns = [
            r'\bbut (?:actually|really|in fact)\b',
            r'\bcontradicts?\b',
            r'\bon one hand\b.*\bon the other\b',
        ]
        
        # Excessive hedging
        hedge_words = [
            r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bsomewhat\b',
            r'\bkind of\b', r'\bsort of\b', r'\bi guess\b', r'\bi suppose\b',
            r'\bmight be\b', r'\bcould be\b', r'\bprobably\b', r'\bseems?\b'
        ]
        
        # Vagueness indicators
        vague_patterns = [
            r'\bstuff\b', r'\bthings?\b', r'\bwhatever\b', r'\betc\.?\b',
            r'\band so on\b', r'\byou know\b', r'\bbasically\b',
            r'\blike\b(?= [a-z])', r'\bwhatever\b'
        ]
        
        hedge_count = sum(len(re.findall(p, resp_lower)) for p in hedge_words)
        vague_count = sum(len(re.findall(p, resp_lower)) for p in vague_patterns)
        
        hedge_density = hedge_count / num_words * 100
        vague_density = vague_count / num_words * 100
        
        # Penalty for excessive hedging/vagueness
        weakness_penalty = min(0.3, hedge_density * 0.03 + vague_density * 0.05)
        
        # ============================================================
        # 5. STRUCTURAL PROGRESSION
        # Check if response has intro -> body -> (conclusion) flow
        # ============================================================
        
        paragraphs = re.split(r'\n\s*\n|\n(?=#{1,3}\s)', response_clean)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 15]
        num_paragraphs = len(paragraphs)
        
        # Check for numbered/ordered structure
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|[a-z][\.\)]|\*\*(?:step|point|reason)\s*\d+)', resp_lower)
        has_numbered_structure = len(numbered_items) >= 2
        
        # Check for introductory sentence/paragraph
        first_sentence = sentences[0].lower() if sentences else ""
        intro_indicators = [
            r'\bhere (?:are|is)\b', r'\bthere are\b', r'\blet\'?s?\b',
            r'\bi (?:will|would|\'ll)\b', r'\bto (?:answer|address|help)\b',
            r'\bgreat\b', r'\bgood\b', r'\binteresting\b', r'\bexcellent\b',
            r'\byes\b', r'\bno\b', r'\bcertainly\b', r'\babsolutely\b',
            r'\bsure\b', r'\bof course\b'
        ]
        has_intro = any(re.search(p, first_sentence) for p in intro_indicators)
        
        # Check for concluding element
        last_portion = resp_lower[-200:] if len(resp_lower) > 200 else resp_lower
        conclusion_indicators = [
            r'\bin (?:conclusion|summary)\b', r'\boverall\b', r'\bto (?:sum|summarize)\b',
            r'\bremember\b', r'\bhappy\b', r'\bgood luck\b', r'\benjoy\b',
            r'\bhope this\b', r'\bhave fun\b', r'\bfeel free\b'
        ]
        has_conclusion = any(re.search(p, last_portion) for p in conclusion_indicators)
        
        structure_score = 0.3  # base
        if has_intro:
            structure_score += 0.2
        if has_numbered_structure:
            structure_score += 0.2
        if has_conclusion:
            structure_score += 0.15
        if num_paragraphs >= 2:
            structure_score += 0.1
        if num_paragraphs >= 3:
            structure_score += 0.05
        
        structure_score = min(structure_score, 1.0)
        
        # ============================================================
        # 6. SENTENCE COMPLEXITY & VARIATION
        # Good arguments use varied sentence structures
        # ============================================================
        
        sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        if len(sent_lengths) >= 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_len)**2 for x in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / mean_len if mean_len > 0 else 0
            # Ideal CV: 0.3-0.7 (some variation but not wild)
            if cv < 0.15:
                length_variation_score = 0.4  # too uniform
            elif cv < 0.3:
                length_variation_score = 0.6
            elif cv <= 0.7:
                length_variation_score = 0.9
            else:
                length_variation_score = 0.6  # too erratic
        else:
            length_variation_score = 0.4
        
        # ============================================================
        # 7. QUERY RELEVANCE CHECK
        # Ensure the response addresses the query topic
        # ============================================================
        
        query_content = get_content_words(query)
        response_content = get_content_words(response_clean)
        
        if query_content and response_content:
            relevance_overlap = len(query_content & response_content) / max(len(query_content), 1)
            # Also check stems
            query_stems = {w[:5] for w in query_content if len(w) >= 5}
            resp_stems = {w[:5] for w in response_content if len(w) >= 5}
            stem_relevance = len(query_stems & resp_stems) / max(len(query_stems), 1)
            relevance_score = max(relevance_overlap, stem_relevance)
            relevance_score = min(relevance_score, 1.0)
        else:
            relevance_score = 0.3
        
        # ============================================================
        # 8. RESPONSE COMPLETENESS
        # Longer, more developed responses tend to have better arguments
        # (but with diminishing returns)
        # ============================================================
        
        # Log-scaled length bonus
        if num_words < 20:
            completeness = 0.2
        elif num_words < 50:
            completeness = 0.4
        elif num_words < 100:
            completeness = 0.6
        elif num_words < 200:
            completeness = 0.75
        elif num_words < 400:
            completeness = 0.9
        else:
            completeness = 1.0
        
        # ============================================================
        # 9. OPENING QUALITY
        # Good responses often start with a clear framing statement
        # ============================================================
        
        opening_quality = 0.3
        if sentences:
            first = sentences[0]
            # Direct answer / position statement
            if re.search(r'\b(yes|no|certainly|absolutely|definitely)\b', first.lower()):
                opening_quality += 0.2
            # Acknowledges the question
            if re.search(r'\b(great|good|interesting|excellent|that\'s|awesome)\b', first.lower()):
                opening_quality += 0.1
            # States what will follow
            if re.search(r'\b(here|let|i\'ll|following|below|steps|tips|ways)\b', first.lower()):
                opening_quality += 0.15
            # Provides context/definition
            if len(first.split()) > 10:
                opening_quality += 0.1
        
        opening_quality = min(opening_quality, 1.0)
        
        # ============================================================
        # FINAL SCORE COMPOSITION
        # ============================================================
        
        raw_score = (
            discourse_score * 2.0 +          # Discourse markers (weight: 2.0)
            coherence_score * 1.8 +           # Topic continuity (weight: 1.8)
            argument_depth_score * 1.8 +      # Argument depth (weight: 1.8)
            structure_score * 1.5 +           # Structural progression (weight: 1.5)
            length_variation_score * 0.7 +    # Sentence variation (weight: 0.7)
            relevance_score * 1.0 +           # Query relevance (weight: 1.0)
            completeness * 0.8 +              # Completeness (weight: 0.8)
            opening_quality * 0.6             # Opening quality (weight: 0.6)
        )
        
        # Apply weakness penalty
        raw_score -= weakness_penalty * 2.0
        
        # Max possible ~10.2, normalize to 0-10 range
        max_possible = 2.0 + 1.8 + 1.8 + 1.5 + 0.7 + 1.0 + 0.8 + 0.6  # = 10.2
        normalized = (raw_score / max_possible) * 10.0
        
        # Clamp
        final_score = max(0.0, min(10.0, normalized))
        
        return round(final_score, 3)
    
    except Exception as e:
        # Fallback: return a middle-of-the-road score
        try:
            if response and len(response.strip()) > 50:
                return 4.0
            return 2.0
        except:
            return 2.0