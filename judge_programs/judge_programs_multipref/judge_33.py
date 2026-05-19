def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    - Discourse marker analysis (causal, contrastive, additive, temporal connectives)
    - Sentence-to-sentence semantic continuity (via shared concept chains)
    - Argument depth detection (premise-conclusion patterns)
    - Absence of contradictions (detecting negation flips)
    - Progressive elaboration (information density growth)
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
            return 0.0
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 0.0
        
        # ---- Feature 1: Discourse Marker Density and Variety ----
        # These markers indicate logical connections between ideas
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b', r'\bsince\b',
            r'\bso that\b', r'\bleading to\b', r'\bcaused by\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bfor this reason\b', r'\bit follows\b'
        ]
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\binstead\b', r'\brather\b',
            r'\byet\b', r'\beven though\b', r'\bstill\b'
        ]
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b', r'\blikewise\b',
            r'\bsimilarly\b', r'\bnot only\b', r'\bas well\b'
        ]
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bthen\b',
            r'\bnext\b', r'\bfinally\b', r'\bsubsequently\b', r'\bafterward\b',
            r'\bbefore\b', r'\bafter\b', r'\binitially\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b'
        ]
        summary_markers = [
            r'\bin summary\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto summarize\b', r'\bin short\b', r'\bto conclude\b',
            r'\ball in all\b', r'\bin essence\b'
        ]
        
        response_lower = response_clean.lower()
        
        def count_marker_types(markers):
            found = set()
            total = 0
            for i, pattern in enumerate(markers):
                matches = re.findall(pattern, response_lower)
                if matches:
                    found.add(i)
                    total += len(matches)
            return total, len(found)
        
        causal_count, causal_variety = count_marker_types(causal_markers)
        contrastive_count, contrastive_variety = count_marker_types(contrastive_markers)
        additive_count, additive_variety = count_marker_types(additive_markers)
        temporal_count, temporal_variety = count_marker_types(temporal_markers)
        summary_count, summary_variety = count_marker_types(summary_markers)
        
        total_markers = causal_count + contrastive_count + additive_count + temporal_count + summary_count
        total_variety = causal_variety + contrastive_variety + additive_variety + temporal_variety + summary_variety
        
        word_count = len(response_clean.split())
        if word_count == 0:
            return 0.0
        
        # Normalize marker density per 100 words
        marker_density = (total_markers / max(word_count, 1)) * 100
        # Score: reward density up to a point, diminishing returns
        marker_density_score = min(marker_density * 2.5, 15.0)
        
        # Variety score: reward using different types of connectives
        variety_score = min(total_variety * 0.8, 10.0)
        
        # Bonus for having multiple categories of markers (shows multi-dimensional reasoning)
        categories_used = sum(1 for c in [causal_count, contrastive_count, additive_count, temporal_count, summary_count] if c > 0)
        category_bonus = categories_used * 1.5
        
        # ---- Feature 2: Sentence-to-Sentence Concept Continuity (Chain Analysis) ----
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 'just', 'don', 'now', 'and', 'but', 'or',
                'if', 'while', 'that', 'this', 'these', 'those', 'it', 'its', 'i',
                'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
                'his', 'her', 'their', 'what', 'which', 'who', 'whom', 'up', 'about',
                'also', 'however', 'because', 'therefore', 'thus', 'although'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        # Measure concept chain continuity: how many content words carry over between consecutive sentences
        continuity_scores = []
        if len(sentences) > 1:
            prev_words = get_content_words(sentences[0])
            for i in range(1, len(sentences)):
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    if union > 0:
                        continuity_scores.append(overlap / union)
                    else:
                        continuity_scores.append(0)
                # Use a sliding window: also consider words from 2 sentences back
                prev_words = curr_words
        
        if continuity_scores:
            avg_continuity = sum(continuity_scores) / len(continuity_scores)
            # Also check consistency (low variance = consistent flow)
            if len(continuity_scores) > 1:
                mean_c = avg_continuity
                variance_c = sum((x - mean_c) ** 2 for x in continuity_scores) / len(continuity_scores)
                consistency = 1.0 / (1.0 + math.sqrt(variance_c) * 5)
            else:
                consistency = 0.5
        else:
            avg_continuity = 0.0
            consistency = 0.0
        
        continuity_score = avg_continuity * 12.0 + consistency * 5.0
        
        # ---- Feature 3: Argument Depth - Premise/Conclusion Patterns ----
        # Detect if response has premise-conclusion structure
        premise_indicators = [
            r'\bgiven that\b', r'\bsince\b', r'\bbecause\b', r'\bas\b.*\bshows?\b',
            r'\bthe reason\b', r'\bdue to\b', r'\bbased on\b', r'\bconsidering\b',
            r'\bif\b.*\bthen\b', r'\bassuming\b'
        ]
        conclusion_indicators = [
            r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
            r'\bso\b', r'\bit follows\b', r'\bwe can conclude\b', r'\bthis means\b',
            r'\bthis suggests\b', r'\bthis indicates\b', r'\bin conclusion\b',
            r'\bas a result\b', r'\bwhich means\b', r'\bimplying\b'
        ]
        
        premise_count = sum(len(re.findall(p, response_lower)) for p in premise_indicators)
        conclusion_count = sum(len(re.findall(p, response_lower)) for p in conclusion_indicators)
        
        # Having both premises and conclusions suggests structured argumentation
        if premise_count > 0 and conclusion_count > 0:
            argument_score = min((premise_count + conclusion_count) * 1.5, 10.0)
        elif premise_count > 0 or conclusion_count > 0:
            argument_score = min((premise_count + conclusion_count) * 0.8, 5.0)
        else:
            argument_score = 0.0
        
        # ---- Feature 4: Progressive Elaboration ----
        # Check if sentences get progressively more specific/detailed
        # Measured by information density (unique content words per sentence)
        sent_densities = []
        for s in sentences:
            cw = get_content_words(s)
            words_in_sent = len(s.split())
            if words_in_sent > 0:
                sent_densities.append(len(cw) / max(words_in_sent, 1))
        
        elaboration_score = 0.0
        if len(sent_densities) > 2:
            # Check if there's a reasonable information density throughout
            avg_density = sum(sent_densities) / len(sent_densities)
            elaboration_score = min(avg_density * 15, 8.0)
        elif sent_densities:
            avg_density = sum(sent_densities) / len(sent_densities)
            elaboration_score = min(avg_density * 10, 5.0)
        
        # ---- Feature 5: Structural Coherence Signals ----
        # Detect numbered/lettered sequences (shows organized thinking)
        numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*\*(?:step|point|reason)\s+\d+)', response_lower)
        has_numbered = len(numbered_pattern)
        
        # Detect markdown headers
        headers = re.findall(r'(?:^|\n)\s*#{1,4}\s+\S', response_clean)
        has_headers = len(headers)
        
        # Detect bold markers used for structure
        bold_markers = re.findall(r'\*\*[^*]+\*\*', response_clean)
        has_bold_structure = len(bold_markers)
        
        structural_score = 0.0
        if has_numbered > 1:
            structural_score += min(has_numbered * 1.0, 5.0)
        if has_headers > 0:
            structural_score += min(has_headers * 1.0, 3.0)
        if has_bold_structure > 1:
            structural_score += min(has_bold_structure * 0.4, 3.0)
        structural_score = min(structural_score, 8.0)
        
        # ---- Feature 6: Opening Framing and Closing ----
        # Good logical responses often frame the question first, then elaborate
        first_sentence = sentences[0].lower() if sentences else ""
        
        framing_score = 0.0
        # Check if response acknowledges/frames the question
        query_words = get_content_words(query)
        first_sent_words = get_content_words(first_sentence)
        if query_words and first_sent_words:
            query_overlap = len(query_words & first_sent_words) / max(len(query_words), 1)
            framing_score += min(query_overlap * 5, 3.0)
        
        # Check for a concluding/summarizing pattern in last portion
        last_portion = " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1] if sentences else ""
        last_lower = last_portion.lower()
        concluding_patterns = [
            r'\bin summary\b', r'\boverall\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\bkey takeaway\b', r'\bmost important\b',
            r'\bin short\b', r'\bremember\b', r'\bkeep in mind\b'
        ]
        has_conclusion = any(re.search(p, last_lower) for p in concluding_patterns)
        if has_conclusion:
            framing_score += 2.0
        
        framing_score = min(framing_score, 5.0)
        
        # ---- Feature 7: Absence of Incoherence Signals ----
        # Detect potential contradictions or incoherent patterns
        incoherence_penalty = 0.0
        
        # Repeated phrases (might indicate circular reasoning)
        trigrams = []
        words = response_lower.split()
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i:i+3]))
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for t, c in trigram_counts.items() if c > 2)
        incoherence_penalty += repeated_trigrams * 0.5
        
        # Detect abrupt topic shifts (very low continuity between consecutive sentences)
        if continuity_scores:
            abrupt_shifts = sum(1 for c in continuity_scores if c < 0.02)
            incoherence_penalty += abrupt_shifts * 0.8
        
        incoherence_penalty = min(incoherence_penalty, 8.0)
        
        # ---- Feature 8: Response Completeness Relative to Length ----
        # Longer, well-structured responses tend to be more logically developed
        length_score = 0.0
        if word_count >= 50:
            length_score = min(math.log(word_count / 50 + 1) * 3, 5.0)
        elif word_count >= 20:
            length_score = 1.0
        
        # ---- Feature 9: Explanatory Depth ----
        # Detect explanatory language patterns
        explanatory_patterns = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bin other words\b',
            r'\bspecifically\b', r'\bnamely\b', r'\bto illustrate\b',
            r'\bto clarify\b', r'\bto explain\b', r'\bput simply\b',
            r'\bwhat this means\b', r'\bin particular\b', r'\bnotably\b'
        ]
        explanatory_count = sum(len(re.findall(p, response_lower)) for p in explanatory_patterns)
        explanatory_score = min(explanatory_count * 1.5, 7.0)
        
        # ---- Feature 10: Hedging and Nuance (shows careful reasoning) ----
        nuance_patterns = [
            r'\bit depends\b', r'\bin some cases\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\bmay\b', r'\bmight\b',
            r'\bcould\b', r'\btend to\b', r'\boften\b', r'\bsometimes\b',
            r'\bnot always\b', r'\bnot necessarily\b', r'\brelatively\b'
        ]
        nuance_count = sum(len(re.findall(p, response_lower)) for p in nuance_patterns)
        nuance_score = min(nuance_count * 0.6, 4.0)
        
        # ---- Aggregate Score ----
        total_score = (
            marker_density_score +     # up to 15
            variety_score +            # up to 10
            category_bonus +           # up to 7.5
            continuity_score +         # up to 17
            argument_score +           # up to 10
            elaboration_score +        # up to 8
            structural_score +         # up to 8
            framing_score +            # up to 5
            length_score +             # up to 5
            explanatory_score +        # up to 7
            nuance_score -             # up to 4
            incoherence_penalty        # up to -8
        )
        
        # Normalize to 0-100 range
        # Theoretical max is about 96.5, practical max around 60-70
        final_score = max(0.0, min(total_score * 1.2, 100.0))
        
        return round(final_score, 2)
        
    except Exception:
        return 0.0