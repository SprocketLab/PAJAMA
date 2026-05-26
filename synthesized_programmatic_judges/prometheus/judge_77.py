def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    
    This variant focuses on:
    1. Sentence-level specificity scoring (analyzing each sentence for concrete content)
    2. Information-to-filler ratio
    3. Structural depth indicators (nested explanations, step sequences)
    4. Named entity and technical term density
    5. Actionability scoring
    
    Different from other variants by operating at the sentence level and using
    a weighted sentence quality aggregation approach.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 0.5
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.5
        
        # ============================================================
        # 1. SENTENCE-LEVEL SPECIFICITY SCORING
        # ============================================================
        # Score each sentence for how "specific" vs "vague" it is
        
        # Vague/filler phrases (per-sentence detection)
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bmost people\b',
            r'\bit depends\b', r'\bvarious factors\b', r'\bthere are many\b',
            r'\bthere are various\b', r'\bin general\b', r'\bgenerally speaking\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bmight not\b', r'\bmay not\b', r'\bcould be\b',
            r'\bjust\b.*\btry\b', r'\bkeep working\b', r'\byou.ll get there\b',
            r'\bdon.t worry\b', r'\bit.s fine\b', r'\bno big deal\b',
            r'\bwhatever\b', r'\bstuff like that\b', r'\band so on\b',
            r'\band things\b', r'\bor something\b', r'\bor whatever\b',
            r'\bbasically\b', r'\bessentially\b',
        ]
        
        # Specificity indicators (per-sentence)
        specific_patterns = [
            (r'\b\d+\.?\d*\s*(?:percent|%|pounds?|ounces?|cups?|minutes?|hours?|days?|weeks?|months?|years?|miles?|km|meters?|feet|inches?|lbs?|oz|mg|kg|gb|mb|tb)\b', 2.0),  # measurements
            (r'\b\d{1,2}[:/]\d{2}\b', 1.5),  # times
            (r'\b(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b', 1.0),  # ordinals indicating sequence
            (r'\b(?:step|phase|stage)\s*\d+\b', 1.5),  # explicit steps
            (r'\bfor (?:example|instance)\b', 1.5),  # examples
            (r'\bsuch as\b', 1.3),  # examples
            (r'\bspecifically\b', 1.2),
            (r'\b(?:because|since|due to|as a result|therefore|consequently)\b', 1.0),  # causal reasoning
            (r'\b(?:research|studies?|data|evidence|according to)\b', 1.5),  # evidence references
            (r'\b\d+\b', 0.5),  # any number
            (r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', 1.0),  # proper nouns / named entities
        ]
        
        # Empathy/engagement patterns (important for the dataset's style)
        empathy_patterns = [
            (r'\bi (?:can |)(?:hear|see|understand|sense|feel|imagine)\b', 1.2),
            (r'\bit.s (?:completely |absolutely |totally |perfectly )?(?:understandable|natural|normal|okay|ok|fine|valid)\b', 0.8),
            (r'\byour (?:feelings?|emotions?|experience|situation|concerns?|frustration|pain)\b', 0.8),
        ]
        
        # Dismissive patterns
        dismissive_patterns = [
            r'\bjust\s+(?:get over|move on|deal with|forget|ignore)\b',
            r'\bit.s (?:just|only) a\b',
            r'\byou (?:should|need to) (?:just|simply)\b',
            r'\bget yourself together\b',
            r'\bstop (?:being|feeling)\b',
        ]
        
        sentence_scores = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            sent_score = 0.0
            word_count = len(sent.split())
            
            if word_count < 2:
                sentence_scores.append(0.0)
                continue
            
            # Base score from sentence length (moderate sentences are better)
            if 5 <= word_count <= 30:
                sent_score += 1.0
            elif word_count > 30:
                sent_score += 0.5
            else:
                sent_score += 0.3
            
            # Penalize vague phrases
            vague_count = 0
            for vp in vague_phrases:
                if re.search(vp, sent_lower):
                    vague_count += 1
            sent_score -= vague_count * 0.8
            
            # Reward specific patterns
            for pattern, weight in specific_patterns:
                matches = re.findall(pattern, sent, re.IGNORECASE)
                if matches:
                    sent_score += weight * min(len(matches), 3)
            
            # Reward empathy patterns
            for pattern, weight in empathy_patterns:
                if re.search(pattern, sent_lower):
                    sent_score += weight
            
            # Penalize dismissive patterns
            for dp in dismissive_patterns:
                if re.search(dp, sent_lower):
                    sent_score -= 1.5
            
            sentence_scores.append(sent_score)
        
        if not sentence_scores:
            return 1.0
        
        # Aggregate sentence scores: use a combination of mean and top-quartile
        sorted_scores = sorted(sentence_scores, reverse=True)
        top_quarter = max(1, len(sorted_scores) // 4)
        top_avg = sum(sorted_scores[:top_quarter]) / top_quarter
        overall_avg = sum(sentence_scores) / len(sentence_scores)
        
        # Weight: 60% overall average, 40% top quarter (rewards having some great sentences)
        sentence_quality = 0.6 * overall_avg + 0.4 * top_avg
        
        # ============================================================
        # 2. INFORMATION DENSITY: Unique content words / total words
        # ============================================================
        words = re.findall(r'\b[a-z]+\b', response_text.lower())
        total_words = len(words)
        
        if total_words < 5:
            return 1.0
        
        # Stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'but', 'and', 'or',
            'if', 'that', 'this', 'it', 'its', 'my', 'your', 'his', 'her',
            'our', 'their', 'what', 'which', 'who', 'whom', 'these', 'those',
            'am', 'about', 'up', 'down', 'also', 'i', 'me', 'we', 'you', 'he',
            'she', 'they', 'them', 'us', 'him', 'much', 'many', 'get', 'got',
        }
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        
        info_density = len(unique_content) / max(total_words, 1)
        
        # ============================================================
        # 3. STRUCTURAL DEPTH
        # ============================================================
        structural_score = 0.0
        
        # Numbered lists (1. 2. 3. or 1) 2) 3))
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_text)
        if len(numbered_items) >= 2:
            structural_score += min(len(numbered_items) * 0.5, 3.0)
        
        # Colon-based explanations ("X: explanation")
        colon_explanations = re.findall(r'[A-Za-z]+\s*:', response_text)
        if len(colon_explanations) >= 1:
            structural_score += min(len(colon_explanations) * 0.3, 1.5)
        
        # Parenthetical clarifications
        parentheticals = re.findall(r'\([^)]{5,}\)', response_text)
        structural_score += min(len(parentheticals) * 0.4, 1.2)
        
        # Conditional/nuanced reasoning ("if...then", "however", "although")
        nuance_words = re.findall(r'\b(?:however|although|whereas|nevertheless|on the other hand|conversely|in contrast|while|despite)\b', response_text.lower())
        structural_score += min(len(nuance_words) * 0.4, 1.6)
        
        # ============================================================
        # 4. TECHNICAL/DOMAIN TERM DENSITY
        # ============================================================
        # Words that are longer and less common tend to be more technical
        long_words = [w for w in content_words if len(w) >= 8]
        tech_density = len(long_words) / max(total_words, 1)
        
        # Compound terms (hyphenated or multi-word technical terms)
        compound_terms = re.findall(r'\b\w+-\w+\b', response_text)
        compound_score = min(len(compound_terms) * 0.2, 1.0)
        
        # ============================================================
        # 5. ACTIONABILITY SCORE
        # ============================================================
        # Detect imperative/instructional language
        action_verbs = re.findall(
            r'\b(?:try|start|begin|make|create|use|apply|consider|remember|ensure|'
            r'take|break|divide|focus|maintain|keep|avoid|include|add|remove|'
            r'check|verify|explore|identify|analyze|implement|practice|develop|'
            r'set|establish|build|write|read|listen|observe|notice|imagine|'
            r'grab|heat|cook|brown|stir|pour|mix|chop|slice)\b',
            response_text.lower()
        )
        actionability = min(len(action_verbs) * 0.15, 2.5)
        
        # ============================================================
        # 6. QUERY RELEVANCE (semantic overlap with query)
        # ============================================================
        query_words = set(re.findall(r'\b[a-z]+\b', query.lower())) - stop_words
        response_content = set(content_words)
        
        if query_words:
            overlap = len(query_words & response_content) / len(query_words)
        else:
            overlap = 0.0
        
        relevance_score = overlap * 2.0  # 0 to 2
        
        # ============================================================
        # 7. NEGATIVE SIGNALS
        # ============================================================
        negative_score = 0.0
        
        # Excessive hedging at the start
        first_sentence = sentences[0].lower() if sentences else ""
        hedging_start = re.findall(r'\b(?:hmm|well|so|um|uh|like|you know)\b', first_sentence)
        negative_score += len(hedging_start) * 0.3
        
        # "might not", "probably won't", "may not be able" - expressing inability
        inability = re.findall(r'\b(?:might not|may not|probably won.t|can.t|cannot|won.t be able)\b', response_text.lower())
        negative_score += len(inability) * 0.4
        
        # Repetitive phrasing (same bigrams appearing too often)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repetitive = sum(1 for _, c in bigram_counts.items() if c > 2)
        negative_score += repetitive * 0.3
        
        # ============================================================
        # 8. RESPONSE COMPLETENESS
        # ============================================================
        # Longer responses with maintained quality are better
        length_score = 0.0
        if total_words >= 50:
            length_score = min(math.log(total_words / 50 + 1) * 1.5, 2.0)
        elif total_words >= 20:
            length_score = 0.5
        
        # ============================================================
        # FINAL AGGREGATION
        # ============================================================
        raw_score = (
            sentence_quality * 1.8      # sentence-level quality (main signal)
            + info_density * 6.0        # information density
            + structural_score * 0.8    # structural depth
            + tech_density * 8.0        # technical term density
            + compound_score            # compound terms
            + actionability             # actionability
            + relevance_score * 0.8     # query relevance
            + length_score              # completeness
            - negative_score * 1.2      # negative signals
        )
        
        # Normalize to 1-5 range
        # Empirically, raw_score ranges roughly from -2 to 15
        normalized = 1.0 + (raw_score + 2) * (4.0 / 17.0)
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception:
        return 2.5