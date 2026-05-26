def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    Uses a different approach focused on discourse markers, argument flow,
    paragraph-level analysis, and structural consistency.
    Returns a score from 0 to 100 where higher = better.
    """
    try:
        if not response or not isinstance(response, str):
            return 0
        if not query or not isinstance(query, str):
            return 5

        import re
        import math
        from collections import Counter

        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 2

        score = 50.0  # Start at midpoint

        # === 1. DISCOURSE MARKER ANALYSIS (logical flow indicators) ===
        # Causal/logical connectors
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\baccordingly\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\bthis means\b', r'\bthis implies\b',
            r'\bleading to\b', r'\bwhich means\b', r'\bso that\b'
        ]
        
        # Contrastive/concessive markers
        contrast_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bdespite\b', r'\byet\b', r'\bnonetheless\b', r'\bwhereas\b',
            r'\beven though\b', r'\binstead\b', r'\brather\b'
        ]
        
        # Additive/elaboration markers
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bnot only\b', r'\bas well\b'
        ]
        
        # Sequential/organizational markers
        sequential_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blast(?:ly)?\b',
            r'\bto begin\b', r'\bin conclusion\b', r'\bto summarize\b',
            r'\bin summary\b', r'\boverall\b', r'\bstep \d+\b',
            r'\bto start\b', r'\bfollowing\b'
        ]
        
        # Exemplification markers
        example_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bincluding\b', r'\blike\b', r'\bnamely\b',
            r'\bto illustrate\b', r'\bspecifically\b', r'\bin particular\b'
        ]

        response_lower = response_stripped.lower()
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_markers)
        contrast_count = sum(len(re.findall(p, response_lower)) for p in contrast_markers)
        additive_count = sum(len(re.findall(p, response_lower)) for p in additive_markers)
        sequential_count = sum(len(re.findall(p, response_lower)) for p in sequential_markers)
        example_count = sum(len(re.findall(p, response_lower)) for p in example_markers)
        
        total_markers = causal_count + contrast_count + additive_count + sequential_count + example_count
        
        # Normalize by response length (per 100 words)
        words = response_stripped.split()
        word_count = len(words)
        if word_count == 0:
            return 2
        
        marker_density = (total_markers / word_count) * 100
        
        # Variety of marker types used
        marker_types_used = sum([
            causal_count > 0,
            contrast_count > 0,
            additive_count > 0,
            sequential_count > 0,
            example_count > 0
        ])
        
        # Reward moderate density (not too few, not excessive)
        if marker_density < 1.0:
            score += marker_density * 3
        elif marker_density < 5.0:
            score += 3 + (marker_density - 1.0) * 1.5
        elif marker_density < 10.0:
            score += 9
        else:
            score += 7  # Slightly penalize excessive markers
        
        # Reward variety of marker types
        score += marker_types_used * 1.5

        # === 2. SENTENCE-LEVEL COHERENCE ===
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences >= 2:
            # Check for topic continuity via word overlap between consecutive sentences
            overlap_scores = []
            for i in range(len(sentences) - 1):
                words_a = set(re.findall(r'\b\w{3,}\b', sentences[i].lower()))
                words_b = set(re.findall(r'\b\w{3,}\b', sentences[i + 1].lower()))
                # Remove very common words
                stopwords = {'the', 'and', 'for', 'are', 'was', 'were', 'has', 'have',
                            'had', 'been', 'will', 'can', 'could', 'would', 'should',
                            'may', 'might', 'this', 'that', 'these', 'those', 'with',
                            'from', 'into', 'not', 'you', 'your', 'they', 'their',
                            'its', 'also', 'more', 'some', 'than', 'when', 'what',
                            'which', 'who', 'how', 'all', 'each', 'any', 'there'}
                words_a -= stopwords
                words_b -= stopwords
                if words_a and words_b:
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    overlap_scores.append(overlap)
            
            if overlap_scores:
                avg_overlap = sum(overlap_scores) / len(overlap_scores)
                # Moderate overlap is good (0.1-0.5 range ideal)
                if avg_overlap < 0.05:
                    score -= 3  # Very low coherence
                elif avg_overlap < 0.15:
                    score += 2
                elif avg_overlap <= 0.5:
                    score += 5
                else:
                    score += 3  # Too repetitive

        # === 3. STRUCTURAL ORGANIZATION ===
        # Check for numbered lists, bullet points, headers
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s', response_stripped))
        has_bullets = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response_stripped))
        has_headers = bool(re.search(r'(?:^|\n)\s*#{1,4}\s', response_stripped))
        has_bold = bool(re.search(r'\*\*[^*]+\*\*', response_stripped))
        
        structural_elements = sum([has_numbered_list, has_bullets, has_headers, has_bold])
        
        # Reward structure for longer responses
        if word_count > 50:
            score += min(structural_elements * 2.5, 8)
        elif word_count > 20:
            score += min(structural_elements * 1.5, 5)

        # === 4. PARAGRAPH STRUCTURE ===
        paragraphs = [p.strip() for p in response_stripped.split('\n') if len(p.strip()) > 10]
        num_paragraphs = len(paragraphs)
        
        if word_count > 80:
            # Longer responses should have multiple paragraphs
            if num_paragraphs >= 3:
                score += 4
            elif num_paragraphs == 2:
                score += 2
            elif num_paragraphs == 1:
                score -= 2  # Wall of text penalty
        
        # Check paragraph length consistency (avoid wildly uneven paragraphs)
        if num_paragraphs >= 3:
            para_lengths = [len(p.split()) for p in paragraphs]
            # Filter out very short lines (headers, etc.)
            substantive_paras = [l for l in para_lengths if l > 5]
            if len(substantive_paras) >= 2:
                avg_len = sum(substantive_paras) / len(substantive_paras)
                if avg_len > 0:
                    variance = sum((l - avg_len) ** 2 for l in substantive_paras) / len(substantive_paras)
                    cv = math.sqrt(variance) / avg_len  # coefficient of variation
                    if cv < 0.5:
                        score += 2  # Well-balanced paragraphs
                    elif cv < 1.0:
                        score += 1

        # === 5. ARGUMENT COMPLETENESS ===
        # Check for introduction-body-conclusion pattern
        has_intro_signal = bool(re.search(
            r'(?:^|\n)(?:here|let|i\'ll|to answer|great question|certainly|absolutely|'
            r'that\'s|this is|there are|the answer|yes|no,?\s)',
            response_lower[:200]
        ))
        
        has_conclusion_signal = bool(re.search(
            r'(?:in conclusion|to summarize|overall|in summary|hope this helps|'
            r'remember|keep in mind|the key|in short|to wrap up|'
            r'good luck|happy|enjoy|have fun)',
            response_lower[-300:] if len(response_lower) > 300 else response_lower
        ))
        
        if has_intro_signal:
            score += 2
        if has_conclusion_signal:
            score += 2
        if has_intro_signal and has_conclusion_signal and num_sentences >= 5:
            score += 2  # Bonus for complete argument arc

        # === 6. CONTRADICTION DETECTION ===
        # Simple heuristic: look for patterns that might indicate contradictions
        contradiction_patterns = [
            r'(?:but|however)\s+(?:as (?:i|we) (?:said|mentioned)|earlier)',
            r'this (?:contradicts|conflicts with)',
            r'on one hand.*on the other hand',  # Not necessarily bad, but flag
        ]
        
        # Check for negation inconsistency (simple version)
        positive_claims = re.findall(r'\bis\s+(?:a|an|the)\s+\w+', response_lower)
        negative_claims = re.findall(r'\bis\s+not\s+(?:a|an|the)\s+\w+', response_lower)
        
        # If same subject appears in both positive and negative claims, might be contradiction
        # (Very rough heuristic)
        contradiction_found = False
        for pattern in contradiction_patterns:
            if re.search(pattern, response_lower):
                contradiction_found = True
                break
        
        if contradiction_found:
            score -= 3

        # === 7. RESPONSE COMPLETENESS (not truncated) ===
        last_char = response_stripped[-1] if response_stripped else ''
        last_50 = response_stripped[-50:] if len(response_stripped) > 50 else response_stripped
        
        # Check if response appears truncated
        appears_truncated = False
        if last_char in (',', ':', '-', '('):
            appears_truncated = True
        elif re.search(r'\b(?:and|or|the|a|an|to|in|of|for|with|is|are|was|but)\s*$', response_stripped):
            appears_truncated = True
        elif not re.search(r'[.!?:*\d\)]$', response_stripped.rstrip()):
            # Doesn't end with proper punctuation
            appears_truncated = True
        
        if appears_truncated:
            score -= 6
        else:
            score += 3

        # === 8. QUERY RELEVANCE CHECK ===
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
        query_words -= {'what', 'when', 'where', 'which', 'would', 'could', 'should',
                       'about', 'with', 'from', 'that', 'this', 'have', 'been',
                       'they', 'their', 'your', 'some', 'more', 'than', 'into'}
        
        response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
        
        if query_words:
            relevance = len(query_words & response_words) / len(query_words)
            if relevance > 0.5:
                score += 4
            elif relevance > 0.25:
                score += 2
            elif relevance < 0.1:
                score -= 5

        # === 9. SENTENCE VARIETY AND SOPHISTICATION ===
        if num_sentences >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            
            # Reward moderate sentence length (not all short, not all long)
            if 10 <= avg_sent_len <= 25:
                score += 2
            elif avg_sent_len < 5:
                score -= 2  # Too choppy
            
            # Reward sentence length variety
            if len(sent_lengths) >= 3:
                sent_variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                sent_std = math.sqrt(sent_variance)
                if avg_sent_len > 0:
                    sent_cv = sent_std / avg_sent_len
                    if 0.2 <= sent_cv <= 0.8:
                        score += 2  # Good variety

        # === 10. SPECIFICITY AND DEPTH ===
        # Check for specific details (numbers, proper nouns, technical terms)
        has_numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', response_stripped))
        has_quotes = response_stripped.count('"') + response_stripped.count("'")
        
        # Longer, more detailed responses tend to be more logically developed
        if word_count >= 100:
            score += 3
        elif word_count >= 60:
            score += 2
        elif word_count >= 30:
            score += 1
        elif word_count < 15:
            score -= 3
        
        # Numbers suggest specificity
        if has_numbers > 0:
            score += min(has_numbers * 0.5, 3)

        # === 11. OPENING QUALITY ===
        # Check if response starts with a clear, direct answer or framing
        first_100 = response_lower[:100]
        
        # Filler/weak openings
        weak_openings = [
            r'^(?:well,?\s)', r'^(?:um,?\s)', r'^(?:so,?\s+basically)',
            r'^(?:i think maybe)', r'^(?:it depends)'
        ]
        
        strong_openings = [
            r'^(?:yes|no)[,.]', r'^(?:certainly|absolutely|great question)',
            r'^(?:the |a |an |to |here|there|this|when|in order)',
        ]
        
        for pattern in weak_openings:
            if re.search(pattern, first_100):
                score -= 1
                break
        
        for pattern in strong_openings:
            if re.search(pattern, first_100):
                score += 1
                break

        # === 12. CONDITIONAL/NUANCED REASONING ===
        nuance_patterns = [
            r'\bif\b.*\bthen\b', r'\bdepending on\b', r'\bin (?:some|certain) cases\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bwhile\b.*\b(?:also|still|however)\b', r'\bnot necessarily\b',
            r'\bit\'s (?:important|worth|crucial) to (?:note|consider|remember)\b',
            r'\bkeep in mind\b', r'\bthat said\b', r'\bhaving said that\b'
        ]
        
        nuance_count = sum(1 for p in nuance_patterns if re.search(p, response_lower))
        score += min(nuance_count * 1.0, 4)

        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        return round(score, 2)

    except Exception:
        return 25.0  # Safe fallback