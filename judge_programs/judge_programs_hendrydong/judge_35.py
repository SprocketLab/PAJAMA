def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    This variant focuses on:
    1. Discourse marker analysis (logical connectives and transition words)
    2. Argument depth via clause/sentence complexity
    3. Evidence of reasoning chains (premise -> conclusion patterns)
    4. Internal consistency signals (absence of contradictions)
    5. Paragraph-level coherence via topic continuity
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
        if len(response_clean) < 5:
            return 0.5
        
        query_clean = query.strip()
        
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_clean)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'\b[a-zA-Z]+\b', response_clean.lower())
        num_words = max(len(words), 1)
        
        # ====== FEATURE 1: Discourse markers and logical connectives ======
        # These indicate logical flow and argumentation structure
        
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b', 
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bcaused by\b', r'\bleads to\b', r'\bresults in\b', r'\bso that\b',
            r'\bfor this reason\b', r'\bit follows\b', r'\baccordingly\b'
        ]
        
        contrastive_markers = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bwhile\b', 
            r'\bwhereas\b', r'\bdespite\b', r'\byet\b', r'\bnonetheless\b',
            r'\binstead\b', r'\brather\b', r'\bconversely\b'
        ]
        
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', 
            r'\bin addition\b', r'\balso\b', r'\blikewise\b', r'\bsimilarly\b',
            r'\bnot only\b', r'\bbeyond that\b', r'\bwhat\'s more\b'
        ]
        
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bsuch as\b', r'\bnamely\b', r'\bthat is\b',
            r'\bin other words\b', r'\bto illustrate\b', r'\bi\.e\.\b', r'\be\.g\.\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bultimately\b', r'\ball in all\b',
            r'\bthe key point\b', r'\bto sum up\b', r'\bin essence\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b', r'\bgiven that\b', r'\bsuppose\b'
        ]
        
        resp_lower = response_clean.lower()
        
        def count_markers(patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, resp_lower))
            return total
        
        causal_count = count_markers(causal_markers)
        contrastive_count = count_markers(contrastive_markers)
        additive_count = count_markers(additive_markers)
        elaboration_count = count_markers(elaboration_markers)
        conclusion_count = count_markers(conclusion_markers)
        conditional_count = count_markers(conditional_markers)
        
        total_discourse = (causal_count + contrastive_count + additive_count + 
                          elaboration_count + conclusion_count + conditional_count)
        
        # Discourse density: markers per sentence (normalized)
        discourse_density = total_discourse / num_sentences
        # Cap and scale
        discourse_score = min(discourse_density, 2.0) / 2.0 * 10
        
        # Variety of discourse marker types used
        types_used = sum([
            1 if causal_count > 0 else 0,
            1 if contrastive_count > 0 else 0,
            1 if additive_count > 0 else 0,
            1 if elaboration_count > 0 else 0,
            1 if conclusion_count > 0 else 0,
            1 if conditional_count > 0 else 0
        ])
        discourse_variety_score = (types_used / 6.0) * 10
        
        # ====== FEATURE 2: Reasoning chain detection ======
        # Look for sequences that indicate logical reasoning
        
        reasoning_patterns = [
            r'\bif\b.{5,80}\bthen\b',
            r'\bnot only\b.{5,80}\bbut also\b',
            r'\bthe reason\b.{5,80}\bis\b',
            r'\bthis means\b',
            r'\bthis implies\b',
            r'\bthis suggests\b',
            r'\bin turn\b',
            r'\bwhich leads to\b',
            r'\bwhich means\b',
            r'\bas a consequence\b',
            r'\bfrom this\b',
            r'\bit follows that\b',
            r'\bwe can conclude\b',
            r'\bthis is because\b',
            r'\bthe implication\b',
            r'\bgiven that\b.{5,80}\bit\b',
            r'\bpremise\b', r'\bconclusion\b', r'\bargument\b',
        ]
        
        reasoning_count = 0
        for p in reasoning_patterns:
            reasoning_count += len(re.findall(p, resp_lower))
        
        reasoning_score = min(reasoning_count / max(num_sentences * 0.15, 1), 1.0) * 10
        
        # ====== FEATURE 3: Sentence-level coherence via word overlap between consecutive sentences ======
        # Measures topic continuity - adjacent sentences should share some vocabulary
        
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
                'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 'just', 'don', 'now', 'and', 'but', 'or', 'if',
                'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
                'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
                'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
                'up', 'about', 'get', 'got', 'like', 'also', 'well', 'much',
            }
            w = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(w) - stop_words
        
        coherence_scores = []
        if len(sentences) >= 2:
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if words_a and words_b:
                    overlap = len(words_a & words_b)
                    union = len(words_a | words_b)
                    coherence_scores.append(overlap / union if union > 0 else 0)
        
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            # Also check for very low coherence (non-sequiturs)
            low_coherence_count = sum(1 for c in coherence_scores if c < 0.02)
            non_sequitur_penalty = low_coherence_count / max(len(coherence_scores), 1)
        else:
            avg_coherence = 0.1
            non_sequitur_penalty = 0
        
        coherence_score = min(avg_coherence * 30, 10)  # Scale up since Jaccard is typically low
        non_sequitur_score = (1 - non_sequitur_penalty) * 5
        
        # ====== FEATURE 4: Argument complexity via clause depth ======
        # Subordinate clauses indicate more complex argumentation
        
        subordinators = [
            r'\bwho\b', r'\bwhich\b', r'\bthat\b', r'\bwhere\b', r'\bwhen\b',
            r'\bwhile\b', r'\balthough\b', r'\bbecause\b', r'\bsince\b',
            r'\bunless\b', r'\buntil\b', r'\bafter\b', r'\bbefore\b',
            r'\bwhereas\b', r'\beven though\b', r'\bso that\b'
        ]
        
        subordinate_count = 0
        for p in subordinators:
            subordinate_count += len(re.findall(p, resp_lower))
        
        clause_density = subordinate_count / num_sentences
        clause_score = min(clause_density / 1.5, 1.0) * 8
        
        # ====== FEATURE 5: Response substantiveness and completeness ======
        # Longer, more detailed responses tend to have better argument structure
        # But normalize to avoid just rewarding verbosity
        
        # Words per sentence (moderate is good, too short = underdeveloped)
        avg_words_per_sentence = num_words / num_sentences
        if avg_words_per_sentence < 5:
            length_quality = 2
        elif avg_words_per_sentence < 10:
            length_quality = 5
        elif avg_words_per_sentence < 25:
            length_quality = 8
        elif avg_words_per_sentence < 40:
            length_quality = 6
        else:
            length_quality = 4
        
        # Total content score - responses need enough content to build arguments
        content_mass = min(num_words / 50, 1.0) * 5  # Up to 5 points for having enough words
        
        # ====== FEATURE 6: Query relevance (topical alignment) ======
        query_content = get_content_words(query_clean)
        response_content = get_content_words(response_clean)
        
        if query_content and response_content:
            relevance_overlap = len(query_content & response_content)
            relevance_score = min(relevance_overlap / max(len(query_content), 1), 1.0) * 8
        else:
            relevance_score = 3
        
        # ====== FEATURE 7: Structural organization signals ======
        # Paragraphs, numbered points, structured presentation
        
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        has_enumeration = bool(re.search(r'(?:^|\n)\s*(?:\d+[.):]|\*|-|•)\s', response_clean))
        has_multiple_paragraphs = num_paragraphs >= 2
        
        structure_score = 0
        if has_multiple_paragraphs:
            structure_score += 3
        if has_enumeration:
            structure_score += 2
        if num_sentences >= 3:
            structure_score += 2
        structure_score = min(structure_score, 7)
        
        # ====== FEATURE 8: Hedging and epistemic markers (shows nuanced thinking) ======
        hedge_patterns = [
            r'\btend[s]? to\b', r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin most cases\b', r'\bit depends\b', r'\bto some extent\b',
            r'\barguably\b', r'\bperhaps\b', r'\bpossibly\b', r'\bmight\b',
            r'\bcould be\b', r'\bsome would argue\b', r'\bone could\b',
            r'\bit\'s worth noting\b', r'\bimportantly\b', r'\bnuance\b',
            r'\bthat said\b', r'\bhaving said that\b', r'\btrade-off\b'
        ]
        
        hedge_count = 0
        for p in hedge_patterns:
            hedge_count += len(re.findall(p, resp_lower))
        
        nuance_score = min(hedge_count / max(num_sentences * 0.1, 1), 1.0) * 6
        
        # ====== FEATURE 9: Absence of incoherence signals ======
        # Detect potential contradictions or confused reasoning
        
        contradiction_patterns = [
            r'\bbut wait\b', r'\bactually no\b', r'\bi mean\b.*\bi mean\b',
            r'\bsorry\b', r'\bi don\'t know\b', r'\bi\'m not sure\b',
            r'\bignore\b.*\babove\b', r'\bforget what\b'
        ]
        
        contradiction_count = 0
        for p in contradiction_patterns:
            contradiction_count += len(re.findall(p, resp_lower))
        
        incoherence_penalty = min(contradiction_count * 2, 8)
        
        # ====== FEATURE 10: First-person engagement and perspective ======
        # Responses that engage personally often show better reasoning
        perspective_markers = re.findall(
            r'\bin my experience\b|\bfrom my perspective\b|\bi think\b|\bi believe\b|\bi would\b|\bi\'ve\b|\bpersonally\b',
            resp_lower
        )
        perspective_score = min(len(perspective_markers) * 1.5, 5)
        
        # ====== COMPOSITE SCORE ======
        
        # Weighted combination
        total = (
            discourse_score * 1.5 +          # 0-15: logical connectives density
            discourse_variety_score * 1.2 +   # 0-12: variety of connector types
            reasoning_score * 1.3 +           # 0-13: explicit reasoning chains
            coherence_score * 1.0 +           # 0-10: sentence-to-sentence coherence
            non_sequitur_score * 0.5 +        # 0-2.5: penalty for non-sequiturs
            clause_score * 0.8 +              # 0-6.4: clause complexity
            length_quality * 0.6 +            # 0-4.8: sentence length quality
            content_mass * 0.8 +              # 0-4: enough content
            relevance_score * 0.7 +           # 0-5.6: topical relevance
            structure_score * 0.5 +           # 0-3.5: structural organization
            nuance_score * 0.6 +              # 0-3.6: nuanced reasoning
            perspective_score * 0.3 -         # 0-1.5: personal engagement
            incoherence_penalty * 1.0         # 0-8: penalty for confusion
        )
        
        # Normalize to 0-100 range
        # Max theoretical ~82, typical good response ~40-60
        normalized = max(0, min(total, 80)) / 80 * 100
        
        return round(normalized, 2)
        
    except Exception as e:
        # Fallback: return a middling score based on response length
        try:
            return min(len(str(response)) / 20, 50)
        except:
            return 25.0