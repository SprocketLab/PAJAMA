def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation.
    
    This variant uses a DIFFERENT approach: analyzing causal/logical connective density,
    explanation depth via subordinate clause detection, question-answer self-dialogue patterns,
    progressive elaboration (information density growth), and meta-cognitive signaling.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        response_stripped = response.strip()
        
        if len(response_stripped) < 20:
            return 0.5
        
        # Split into sentences more carefully
        sentences = re.split(r'(?<=[.!?])\s+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = response_lower.split()
        num_words = max(len(words), 1)
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Causal & Logical Connective Density
        # Measures how much the response uses causal reasoning language
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich means\b', r'\bwhich leads to\b', r'\bso that\b',
            r'\bin order to\b', r'\bthe reason\b', r'\bthis is because\b',
            r'\bthis is why\b', r'\bthat\'s why\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\baccordingly\b', r'\bas such\b',
            r'\bgiven that\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bif\b.*\bthen\b', r'\bcauses\b', r'\bresults in\b',
            r'\bleads to\b', r'\bimplies\b', r'\bsuggests that\b',
        ]
        
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, response_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 3.5, 5.0)  # up to 5 points
        score += causal_score
        
        # ============================================================
        # FEATURE 2: Subordinate Clause Depth (explanation complexity)
        # Counts subordinating conjunctions and relative pronouns that
        # indicate nested explanations
        # ============================================================
        subordinators = [
            r'\bwhich\b', r'\bwhere\b', r'\bwhen\b', r'\bwhile\b',
            r'\balthough\b', r'\bwhereas\b', r'\beven though\b',
            r'\bso that\b', r'\bunless\b', r'\bwhether\b',
            r'\bthat\b(?=\s+(?:is|are|was|were|has|have|can|could|would|will|might|may|should))',
            r'\bwho\b', r'\bwhom\b', r'\bwhose\b',
        ]
        
        subord_count = 0
        for pattern in subordinators:
            subord_count += len(re.findall(pattern, response_lower))
        
        subord_density = subord_count / num_sentences
        subord_score = min(subord_density * 2.0, 4.0)  # up to 4 points
        score += subord_score
        
        # ============================================================
        # FEATURE 3: Meta-cognitive and Epistemic Signaling
        # Detects when the response signals its own reasoning process
        # ============================================================
        metacognitive_markers = [
            r'\blet\'s\b', r'\blet us\b', r'\bfirst\b.*\bthen\b',
            r'\bto understand\b', r'\bto put it\b', r'\bin other words\b',
            r'\bwhat this means\b', r'\bthink of it\b', r'\bimagine\b',
            r'\bconsider\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bnotice\b', r'\bobserve\b', r'\bremember\b',
            r'\bkeep in mind\b', r'\bthe key\b', r'\bimportantly\b',
            r'\bthe point is\b', r'\bthe idea is\b', r'\bessentially\b',
            r'\bfundamentally\b', r'\bat its core\b', r'\bput simply\b',
            r'\bto clarify\b', r'\bto elaborate\b', r'\bto illustrate\b',
            r'\bhere\'s\b', r'\bhere is\b', r'\bhere are\b',
            r'\bwhat happens is\b', r'\bthe way\b.*\bworks\b',
            r'\bstep\b', r'\bphase\b', r'\bstage\b',
            r'\bin summary\b', r'\bto summarize\b', r'\boverall\b',
            r'\blet me\b', r'\ballow me\b',
        ]
        
        meta_count = 0
        for pattern in metacognitive_markers:
            meta_count += len(re.findall(pattern, response_lower))
        
        meta_density = meta_count / num_sentences
        meta_score = min(meta_density * 3.0, 5.0)  # up to 5 points
        score += meta_score
        
        # ============================================================
        # FEATURE 4: Progressive Elaboration / Information Density Growth
        # Checks if the response builds up information progressively
        # (later sentences reference or build on earlier concepts)
        # ============================================================
        if num_sentences >= 3:
            # Measure unique content words per sentence
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
                'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
                'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'only', 'own', 'same', 'than', 'too', 'very', 'just', 'it',
                'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
                'him', 'his', 'she', 'her', 'they', 'them', 'their', 'this',
                'that', 'these', 'those', 'what', 'which', 'who', 'whom',
                'how', 'when', 'where', 'why', 'if', 'then', 'there', 'here',
            }
            
            sentence_content_words = []
            for s in sentences:
                s_words = re.findall(r'[a-z]+', s.lower())
                content = [w for w in s_words if w not in stopwords and len(w) > 2]
                sentence_content_words.append(set(content))
            
            # Measure cumulative vocabulary growth and back-referencing
            cumulative_vocab = set()
            back_references = 0
            new_introductions = 0
            
            for i, cw in enumerate(sentence_content_words):
                if i > 0:
                    overlap = len(cw & cumulative_vocab)
                    new = len(cw - cumulative_vocab)
                    if overlap > 0 and new > 0:
                        back_references += 1  # References old + introduces new = progressive
                cumulative_vocab |= cw
            
            if num_sentences > 1:
                progressive_ratio = back_references / (num_sentences - 1)
                progressive_score = progressive_ratio * 4.0  # up to 4 points
            else:
                progressive_score = 1.0
            
            score += min(progressive_score, 4.0)
        else:
            score += 0.5
        
        # ============================================================
        # FEATURE 5: Explicit Reasoning Chain Detection
        # Looks for "if...then", "when...you", premise-conclusion patterns
        # ============================================================
        reasoning_chains = [
            r'\bif\b[^.]{5,60}\bthen\b',
            r'\bwhen\b[^.]{5,60}\byou\b',
            r'\bonce\b[^.]{5,60}\byou\b',
            r'\bafter\b[^.]{5,60}\byou\b',
            r'\bbefore\b[^.]{5,60}\bmake sure\b',
            r'\bfirst\b[^.]{5,100}\bthen\b',
            r'\bnot only\b[^.]{5,60}\bbut also\b',
            r'\bthe more\b[^.]{5,60}\bthe more\b',
            r'\bon one hand\b[^.]{5,100}\bon the other\b',
        ]
        
        chain_count = 0
        for pattern in reasoning_chains:
            chain_count += len(re.findall(pattern, response_lower))
        
        chain_score = min(chain_count * 1.5, 4.0)  # up to 4 points
        score += chain_score
        
        # ============================================================
        # FEATURE 6: Analogical Reasoning Detection
        # Checks for analogies, comparisons, metaphors used to explain
        # ============================================================
        analogy_patterns = [
            r'\bjust like\b', r'\bsimilar to\b', r'\bthink of\b.*\bas\b',
            r'\bimagine\b.*\bas\b', r'\blike a\b', r'\bas if\b',
            r'\banalog\w*\b', r'\bcompar\w*\b', r'\bmetaphor\w*\b',
            r'\bin the same way\b', r'\bmuch like\b', r'\bpicture\b',
            r'\benvision\b', r'\bsuppose\b', r'\bpretend\b',
            r'\bequivalent\b', r'\bcorrespond\b',
        ]
        
        analogy_count = 0
        for pattern in analogy_patterns:
            analogy_count += len(re.findall(pattern, response_lower))
        
        analogy_score = min(analogy_count * 1.5, 3.0)  # up to 3 points
        score += analogy_score
        
        # ============================================================
        # FEATURE 7: Acknowledgment & Empathetic Framing before Reasoning
        # Good responses often acknowledge the situation before diving in
        # ============================================================
        first_two_sentences = ' '.join(sentences[:2]).lower() if len(sentences) >= 2 else response_lower[:200]
        
        acknowledgment_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b',
            r'\bthat\'s\b.*\b(?:understandable|normal|natural|okay|fine|valid)\b',
            r'\bit\'s\b.*\b(?:understandable|normal|natural|okay|fine|completely|perfectly)\b',
            r'\bi\'m sorry\b', r'\bi\'m genuinely\b',
            r'\bcompletely understandable\b', r'\bperfectly\b.*\b(?:fine|okay|normal|natural)\b',
            r'\bof course\b', r'\babsolutely\b',
        ]
        
        ack_found = 0
        for pattern in acknowledgment_patterns:
            if re.search(pattern, first_two_sentences):
                ack_found += 1
        
        ack_score = min(ack_found * 1.0, 2.0)  # up to 2 points
        score += ack_score
        
        # ============================================================
        # FEATURE 8: Absence of Opaque/Dismissive Language
        # Penalize responses that are dismissive or jump to conclusions
        # ============================================================
        dismissive_patterns = [
            r'\bjust\b.*\bdo\b', r'\bjust\b.*\bget\b',
            r'\byou should be able to\b', r'\bit\'s not that\b.*\bhard\b',
            r'\bsimply\b', r'\bobviously\b', r'\bclearly\b',
            r'\bjust\b.*\bremember\b', r'\byou need to get\b.*\btogether\b',
            r'\bmaybe you\'re\b.*\bnot\b', r'\byou\'re just\b',
            r'\bjust keep\b', r'\bjust\b.*\btry\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        dismissive_penalty = min(dismissive_count * 0.8, 4.0)
        score -= dismissive_penalty
        
        # ============================================================
        # FEATURE 9: Sentence Complexity Variance
        # Good reasoning shows varied sentence structure (mix of short
        # punchy statements and longer explanatory ones)
        # ============================================================
        if num_sentences >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            
            # Moderate variance is good (not all same length, not wildly different)
            # Optimal std_dev around 5-15
            if 3 <= std_dev <= 20:
                variance_score = 2.0
            elif 1 <= std_dev < 3:
                variance_score = 1.0
            elif 20 < std_dev <= 30:
                variance_score = 1.0
            else:
                variance_score = 0.5
            
            score += variance_score
        else:
            score += 0.5
        
        # ============================================================
        # FEATURE 10: Concrete Action/Instruction Specificity
        # Measures whether the response provides specific, actionable details
        # ============================================================
        specificity_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\blike\b',
            r'\bincluding\b', r'\b\d+\b',  # numbers indicate specificity
            r'\bapproximately\b', r'\babout\b.*\b\d+\b',
        ]
        
        spec_count = 0
        for pattern in specificity_markers:
            spec_count += len(re.findall(pattern, response_lower))
        
        spec_score = min(spec_count * 0.6, 3.0)  # up to 3 points
        score += spec_score
        
        # ============================================================
        # FEATURE 11: Coherent Structure Detection
        # Check for numbered items or sequential markers that show structure
        # ============================================================
        sequential_markers = re.findall(
            r'(?:^|\n)\s*(?:\d+[\.\):]|[a-z][\.\)]|\-\s|\*\s|•)', response
        )
        
        ordinal_words = re.findall(
            r'\b(?:first|firstly|second|secondly|third|thirdly|fourth|finally|lastly|next|additionally|furthermore|moreover)\b',
            response_lower
        )
        
        structure_count = len(sequential_markers) + len(ordinal_words)
        structure_score = min(structure_count * 0.7, 3.0)  # up to 3 points
        score += structure_score
        
        # ============================================================
        # FEATURE 12: Response Engagement with Query
        # Check how well the response addresses the specific query
        # ============================================================
        query_lower = query.lower()
        query_words = re.findall(r'[a-z]+', query_lower)
        query_content = {w for w in query_words if len(w) > 3}
        
        response_words_set = set(re.findall(r'[a-z]+', response_lower))
        
        if query_content:
            engagement = len(query_content & response_words_set) / len(query_content)
            engagement_score = engagement * 2.0  # up to 2 points
        else:
            engagement_score = 1.0
        
        score += engagement_score
        
        # ============================================================
        # FEATURE 13: Length adequacy
        # Very short responses can't show reasoning; very long isn't auto-good
        # ============================================================
        if num_words < 30:
            score *= 0.5
        elif num_words < 50:
            score *= 0.7
        elif num_words < 80:
            score *= 0.85
        elif num_words > 300:
            # Slight bonus for substantial responses but diminishing returns
            score *= 1.05
        
        # Normalize to 1-5 scale
        # Theoretical max is around 35-40, but practical max is ~25
        # Map roughly: 0-5 -> 1, 5-10 -> 2, 10-15 -> 3, 15-20 -> 4, 20+ -> 5
        
        normalized = 1.0 + (score / 5.5)  # scale factor
        
        # Clamp to 1-5 range
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception as e:
        return 2.5  # Safe fallback