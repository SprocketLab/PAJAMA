def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using discourse analysis.
    
    This variant focuses on:
    1. Discourse markers and logical connectives (causal, contrastive, additive, temporal)
    2. Sentence-to-sentence semantic continuity (word overlap between adjacent sentences)
    3. Argument depth detection (claim-evidence-conclusion patterns)
    4. Absence of incoherence signals (contradictions, abrupt topic shifts)
    5. Response completeness and closure signals
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_text = response.strip()
        query_text = query.strip()
        
        if len(response_text) < 10:
            return 0.5
        
        # ============================================================
        # 1. DISCOURSE MARKER ANALYSIS
        # Detect logical connectives that signal structured reasoning
        # ============================================================
        
        response_lower = response_text.lower()
        
        # Causal connectives (showing reasoning chains)
        causal_markers = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bconsequently\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bleading to\b', r'\bcaused by\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\baccordingly\b', r'\bthis is why\b',
            r'\bgiven that\b', r'\bin order to\b'
        ]
        
        # Contrastive connectives (showing nuanced thinking)
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bwhereas\b', r'\bdespite\b', r'\byet\b', r'\binstead\b',
            r'\brather than\b', r'\bon the contrary\b', r'\beven though\b',
            r'\bnonetheless\b', r'\bthat said\b'
        ]
        
        # Additive/elaborative connectives (building arguments)
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bbesides\b',
            r'\bnot only\b', r'\bas well as\b', r'\bwhat\'s more\b',
            r'\bon top of that\b', r'\bequally important\b'
        ]
        
        # Sequential/temporal markers (showing ordered reasoning)
        sequential_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bto begin\b',
            r'\bto start\b', r'\bafter that\b', r'\bsubsequently\b',
            r'\bin the first place\b', r'\blast(?:ly)?\b', r'\bto conclude\b',
            r'\bin summary\b', r'\bin conclusion\b', r'\boverall\b',
            r'\bto summarize\b', r'\bstep \d+\b'
        ]
        
        # Exemplification markers
        exemplification_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bto illustrate\b',
            r'\bnamely\b', r'\blike\b', r'\bincluding\b', r'\be\.g\.\b'
        ]
        
        def count_markers(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_markers(causal_markers, response_lower)
        contrastive_count = count_markers(contrastive_markers, response_lower)
        additive_count = count_markers(additive_markers, response_lower)
        sequential_count = count_markers(sequential_markers, response_lower)
        exemplification_count = count_markers(exemplification_markers, response_lower)
        
        total_markers = causal_count + contrastive_count + additive_count + sequential_count + exemplification_count
        
        # Diversity of marker types used (0-5)
        marker_type_diversity = sum([
            1 for c in [causal_count, contrastive_count, additive_count, sequential_count, exemplification_count]
            if c > 0
        ])
        
        # ============================================================
        # 2. SENTENCE-LEVEL COHERENCE (adjacent sentence continuity)
        # ============================================================
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        def get_content_words(text):
            """Extract content words (remove common stop words)."""
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'and', 'or', 'but', 'if',
                'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she',
                'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                'our', 'their', 'what', 'which', 'who', 'whom'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        # Compute adjacent sentence overlap (coherence continuity)
        coherence_scores = []
        if num_sentences >= 2:
            for i in range(len(sentences) - 1):
                words_a = set(get_content_words(sentences[i]))
                words_b = set(get_content_words(sentences[i + 1]))
                if words_a and words_b:
                    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                    coherence_scores.append(overlap)
                else:
                    coherence_scores.append(0.0)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        # Also check for abrupt drops in coherence (incoherence signal)
        coherence_drops = 0
        if len(coherence_scores) >= 2:
            for i in range(len(coherence_scores)):
                if coherence_scores[i] == 0.0:
                    coherence_drops += 1
        
        # ============================================================
        # 3. ARGUMENT STRUCTURE DETECTION
        # ============================================================
        
        # Detect claim-evidence patterns
        claim_indicators = [
            r'\bi believe\b', r'\bi think\b', r'\bit is\b', r'\bwe should\b',
            r'\bthe (?:main|key|primary|important)\b', r'\bin my (?:opinion|view)\b',
            r'\bthe fact is\b', r'\bthe point is\b', r'\bargument\b',
            r'\bthe reason\b', r'\bschools should\b', r'\bit is important\b'
        ]
        
        evidence_indicators = [
            r'\bstudies show\b', r'\bresearch\b', r'\baccording to\b',
            r'\bevidence\b', r'\bdata\b', r'\bstatistics\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bin fact\b',
            r'\bhistorically\b', r'\bexperiment\b'
        ]
        
        conclusion_indicators = [
            r'\bin conclusion\b', r'\boverall\b', r'\btherefore\b',
            r'\bin summary\b', r'\bto sum up\b', r'\bultimately\b',
            r'\ball in all\b', r'\bin the end\b', r'\bto conclude\b',
            r'\btaken together\b', r'\bon balance\b'
        ]
        
        claim_count = count_markers(claim_indicators, response_lower)
        evidence_count = count_markers(evidence_indicators, response_lower)
        conclusion_count = count_markers(conclusion_indicators, response_lower)
        
        # Argument completeness: having claims + evidence/examples + conclusion
        argument_components = sum([
            1 if claim_count > 0 else 0,
            1 if evidence_count > 0 or exemplification_count > 0 else 0,
            1 if conclusion_count > 0 else 0
        ])
        
        # ============================================================
        # 4. STRUCTURAL FRAMING (intro + body + wrap-up)
        # ============================================================
        
        # Check for introductory framing
        intro_patterns = [
            r'^(?:certainly|absolutely|great question|that\'s a great|sure|of course|yes|no,)',
            r'^(?:here (?:are|is)|let me|i\'d be happy|let\'s)',
            r'^(?:the |a |an |this |there )',
            r'^(?:organizing|brewing|writing|traveling|when )',
        ]
        has_intro = any(re.match(p, response_lower.strip()) for p in intro_patterns)
        
        # Check for wrap-up / closure
        last_200 = response_lower[-200:] if len(response_lower) > 200 else response_lower
        closure_patterns = [
            r'\bhope this helps\b', r'\bgood luck\b', r'\bhappy\b',
            r'\bfeel free\b', r'\blet me know\b', r'\bin summary\b',
            r'\boverall\b', r'\bin conclusion\b', r'\bremember\b',
            r'\bmost importantly\b', r'\bkey takeaway\b', r'\bto wrap up\b',
            r'\bwith (?:these|this|that)\b', r'\bby following\b'
        ]
        has_closure = any(re.search(p, last_200) for p in closure_patterns)
        
        # ============================================================
        # 5. INCOHERENCE / CONTRADICTION DETECTION
        # ============================================================
        
        # Look for contradiction signals
        contradiction_patterns = [
            r'\bbut (?:earlier|previously|above) (?:i|we) (?:said|mentioned|stated)\b',
            r'\bcontradicts?\b',
            r'\bwait\b',
            r'\bactually,? (?:no|never mind)\b',
            r'\bignore (?:what|the above)\b',
        ]
        contradiction_count = count_markers(contradiction_patterns, response_lower)
        
        # Repetition detection (sign of circular reasoning)
        # Check if any sentence is very similar to another non-adjacent sentence
        repetition_score = 0.0
        if num_sentences >= 4:
            for i in range(len(sentences)):
                for j in range(i + 2, len(sentences)):
                    words_i = set(get_content_words(sentences[i]))
                    words_j = set(get_content_words(sentences[j]))
                    if words_i and words_j:
                        sim = len(words_i & words_j) / max(len(words_i | words_j), 1)
                        if sim > 0.7:
                            repetition_score += 1
        
        # ============================================================
        # 6. QUERY-RESPONSE ALIGNMENT
        # ============================================================
        
        query_content = set(get_content_words(query_text))
        response_content = set(get_content_words(response_text))
        
        if query_content:
            relevance = len(query_content & response_content) / len(query_content)
        else:
            relevance = 0.5
        
        # ============================================================
        # 7. RESPONSE RICHNESS AND DEPTH
        # ============================================================
        
        word_count = len(response_text.split())
        
        # Vocabulary richness (type-token ratio)
        all_words = re.findall(r'[a-z]+', response_lower)
        if all_words:
            ttr = len(set(all_words)) / len(all_words)
        else:
            ttr = 0.0
        
        # Sentence length variation (good writing has varied sentence lengths)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) >= 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            sent_length_variation = math.sqrt(variance)
        else:
            sent_length_variation = 0.0
        
        # ============================================================
        # 8. PROGRESSIVE DEVELOPMENT CHECK
        # ============================================================
        # Check if new content words are introduced progressively
        # (sign of developing argument vs repeating same ideas)
        
        new_word_ratios = []
        seen_words = set()
        for sent in sentences:
            content = set(get_content_words(sent))
            if content:
                new_words = content - seen_words
                ratio = len(new_words) / len(content) if content else 0
                new_word_ratios.append(ratio)
                seen_words.update(content)
        
        avg_new_word_ratio = sum(new_word_ratios) / len(new_word_ratios) if new_word_ratios else 0.0
        
        # ============================================================
        # SCORING
        # ============================================================
        
        score = 0.0
        
        # Discourse markers (up to 25 points)
        # Normalize by sentence count for density
        if num_sentences > 0:
            marker_density = total_markers / num_sentences
        else:
            marker_density = 0
        
        # Reward marker density (diminishing returns)
        marker_score = min(marker_density * 8, 10)
        # Reward diversity of marker types
        marker_score += marker_type_diversity * 3
        score += marker_score  # up to ~25
        
        # Sentence coherence (up to 15 points)
        coherence_score = avg_coherence * 15
        # Penalize coherence drops
        if num_sentences > 2:
            drop_ratio = coherence_drops / (num_sentences - 1)
            coherence_score *= (1 - drop_ratio * 0.3)
        score += coherence_score  # up to 15
        
        # Argument structure (up to 15 points)
        arg_score = argument_components * 5
        # Bonus for having causal reasoning
        arg_score += min(causal_count * 1.5, 5)
        score += arg_score  # up to 15
        
        # Structural framing (up to 10 points)
        frame_score = 0
        if has_intro:
            frame_score += 5
        if has_closure:
            frame_score += 5
        score += frame_score  # up to 10
        
        # Relevance (up to 10 points)
        score += relevance * 10  # up to 10
        
        # Progressive development (up to 10 points)
        score += avg_new_word_ratio * 10  # up to 10
        
        # Vocabulary richness (up to 5 points)
        # TTR around 0.4-0.7 is good for longer texts
        score += min(ttr * 7, 5)  # up to 5
        
        # Sentence length variation (up to 5 points)
        # Some variation is good
        score += min(sent_length_variation * 0.5, 5)  # up to 5
        
        # Response length bonus (up to 5 points) - not too short
        length_score = min(word_count / 40, 5)
        score += length_score  # up to 5
        
        # ============================================================
        # PENALTIES
        # ============================================================
        
        # Contradiction penalty
        score -= contradiction_count * 3
        
        # Excessive repetition penalty
        score -= repetition_score * 2
        
        # Truncation penalty (response cut off mid-sentence)
        last_char = response_text.rstrip()[-1] if response_text.rstrip() else ''
        if last_char not in '.!?"\')]}':
            score -= 5  # Significant penalty for incomplete responses
        
        # Very short response penalty
        if word_count < 20:
            score -= 5
        
        # Clamp score
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
        
    except Exception as e:
        # Never crash
        return 25.0