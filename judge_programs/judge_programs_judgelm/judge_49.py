def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    Focuses on:
    - Whether the response breaks down problems step-by-step
    - Visibility of intermediate conclusions
    - Explanations of 'why' behind claims
    - Ability for reader to follow and verify logic
    - Penalizes opaque answers that jump to conclusions without reasoning
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        import string
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        # If response is essentially empty or trivially short
        if len(response_stripped) < 3:
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Response length relative to query complexity
        # Longer, more substantive responses tend to show more reasoning
        # ============================================================
        query_words = query_stripped.split()
        response_words = response_stripped.split()
        num_response_words = len(response_words)
        num_query_words = len(query_words)
        
        # Very short responses (< 5 words) are likely opaque/no reasoning
        if num_response_words <= 2:
            return 0.5
        elif num_response_words <= 5:
            score += 0.5
        elif num_response_words <= 15:
            score += 1.5
        elif num_response_words <= 40:
            score += 2.5
        elif num_response_words <= 100:
            score += 3.0
        elif num_response_words <= 200:
            score += 3.2
        else:
            score += 3.0  # Very long might be rambling
        
        # ============================================================
        # FEATURE 2: Step-wise / sequential reasoning indicators
        # ============================================================
        step_patterns = [
            r'\bstep\s*\d+\b',
            r'\bfirst(?:ly)?\b',
            r'\bsecond(?:ly)?\b',
            r'\bthird(?:ly)?\b',
            r'\bnext\b',
            r'\bthen\b',
            r'\bfinally\b',
            r'\bafter\s+that\b',
            r'\bto\s+begin\b',
            r'\bin\s+conclusion\b',
            r'\bto\s+summarize\b',
            r'\boverall\b',
            r'\b(?:1|2|3|4|5)\)',
            r'^\s*\d+[\.\)]\s',
            r'^\s*[-•]\s',
        ]
        
        response_lower = response_stripped.lower()
        step_count = 0
        for pat in step_patterns:
            matches = re.findall(pat, response_lower, re.MULTILINE)
            step_count += len(matches)
        
        step_score = min(step_count * 0.3, 1.5)
        score += step_score
        
        # ============================================================
        # FEATURE 3: Causal / explanatory connectors
        # Words that indicate reasoning is being shown
        # ============================================================
        reasoning_connectors = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bas\s+a\s+result\b', r'\bconsequently\b',
            r'\bdue\s+to\b', r'\bthis\s+means\b', r'\bthis\s+is\s+because\b',
            r'\bthe\s+reason\b', r'\bwhich\s+means\b', r'\bso\s+that\b',
            r'\bin\s+order\s+to\b', r'\bthis\s+leads\s+to\b',
            r'\bimplies\b', r'\bsuggests\s+that\b', r'\bindicates\b',
            r'\bit\s+follows\b', r'\bgiven\s+that\b',
            r'\bas\s+such\b', r'\bfor\s+this\s+reason\b',
            r'\baccordingly\b',
        ]
        
        causal_count = 0
        for pat in reasoning_connectors:
            matches = re.findall(pat, response_lower)
            causal_count += len(matches)
        
        causal_score = min(causal_count * 0.35, 1.5)
        score += causal_score
        
        # ============================================================
        # FEATURE 4: Hedging / nuance / qualification indicators
        # Shows thoughtfulness and transparency about uncertainty
        # ============================================================
        hedge_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bon\s+the\s+other\s+hand\b',
            r'\bit\s+depends\b', r'\bnot\s+always\b', r'\bin\s+some\s+cases\b',
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bit\s+is\s+(?:important|worth)\b', r'\bkeep\s+in\s+mind\b',
            r'\bnote\s+that\b', r'\bdifficult\s+to\b', r'\bsubjective\b',
            r'\bcan\s+vary\b', r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bpossibly\b', r'\bpotentially\b',
        ]
        
        hedge_count = 0
        for pat in hedge_patterns:
            matches = re.findall(pat, response_lower)
            hedge_count += len(matches)
        
        hedge_score = min(hedge_count * 0.2, 1.0)
        score += hedge_score
        
        # ============================================================
        # FEATURE 5: Sentence structure complexity and variety
        # Multiple sentences suggest structured reasoning
        # ============================================================
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        if num_sentences <= 1:
            score += 0.0
        elif num_sentences <= 3:
            score += 0.8
        elif num_sentences <= 7:
            score += 1.3
        elif num_sentences <= 15:
            score += 1.5
        else:
            score += 1.2  # Too many might be rambling
        
        # Sentence length variety (std dev of word counts per sentence)
        if num_sentences >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Some variety is good (not all same length)
            variety_score = min(std_dev / 5.0, 0.5)
            score += variety_score
        
        # ============================================================
        # FEATURE 6: Coherence / relevance to query
        # Check word overlap between query and response
        # ============================================================
        query_tokens = set(re.findall(r'\b\w+\b', query_stripped.lower()))
        response_tokens = set(re.findall(r'\b\w+\b', response_lower))
        
        # Remove very common stopwords for relevance check
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'it', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what',
            'which', 'who', 'whom', 'where', 'when', 'how', 'why', 'and', 'or',
            'but', 'not', 'no', 'if', 'so', 'as', 'about', 'up', 'out', 'just',
            'also', 'than', 'more', 'very', 'too', 'here', 'there',
        }
        
        query_content = query_tokens - stopwords
        response_content = response_tokens - stopwords
        
        if query_content:
            overlap = query_content & response_content
            relevance_ratio = len(overlap) / len(query_content)
            relevance_score = relevance_ratio * 1.0
            score += min(relevance_score, 1.0)
        
        # ============================================================
        # FEATURE 7: Penalize garbage / repetition / incoherence
        # ============================================================
        
        # Check for excessive repetition of phrases
        words_list = response_lower.split()
        if len(words_list) > 10:
            # Check bigram repetition
            bigrams = [' '.join(words_list[i:i+2]) for i in range(len(words_list)-1)]
            from collections import Counter
            bigram_counts = Counter(bigrams)
            if bigrams:
                most_common_count = bigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(bigrams)
                if repetition_ratio > 0.15:
                    score -= 1.5
                elif repetition_ratio > 0.08:
                    score -= 0.5
        
        # Penalize responses that contain HTML/code artifacts when not asked for code
        query_asks_code = any(kw in query_stripped.lower() for kw in [
            'code', 'html', 'program', 'script', 'function', 'tag', 'css', 'javascript'
        ])
        if not query_asks_code:
            code_artifacts = len(re.findall(r'<[a-z]+>', response_lower))
            code_artifacts += len(re.findall(r'import\s+\w+', response_lower))
            code_artifacts += len(re.findall(r'def\s+\w+\(', response_lower))
            if code_artifacts > 2:
                score -= 1.5
        
        # Penalize responses that just repeat the query
        if response_stripped.strip().lower() == query_stripped.strip().lower():
            return 1.0
        
        # ============================================================
        # FEATURE 8: Structural formatting (lists, paragraphs, etc.)
        # ============================================================
        has_list = bool(re.search(r'^\s*[\-•\*]\s', response_stripped, re.MULTILINE))
        has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', response_stripped, re.MULTILINE))
        has_paragraphs = '\n\n' in response_stripped
        has_colon_structure = bool(re.search(r'\w+:\s+\w+', response_stripped))
        
        structure_score = 0
        if has_list:
            structure_score += 0.3
        if has_numbered:
            structure_score += 0.3
        if has_paragraphs:
            structure_score += 0.2
        if has_colon_structure:
            structure_score += 0.2
        
        score += min(structure_score, 0.7)
        
        # ============================================================
        # FEATURE 9: Explanation depth - looking for elaboration patterns
        # ============================================================
        elaboration_patterns = [
            r'\bfor\s+example\b', r'\bfor\s+instance\b', r'\bsuch\s+as\b',
            r'\bspecifically\b', r'\bin\s+particular\b', r'\bnamely\b',
            r'\bto\s+illustrate\b', r'\bconsider\b', r'\bimagine\b',
            r'\blet\'s\s+say\b', r'\bsuppose\b', r'\bin\s+other\s+words\b',
            r'\bthat\s+is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bthis\s+is\s+why\b', r'\bthe\s+key\b', r'\bimportant(?:ly)?\b',
            r'\bnotably\b', r'\bessentially\b', r'\bfundamentally\b',
        ]
        
        elab_count = 0
        for pat in elaboration_patterns:
            matches = re.findall(pat, response_lower)
            elab_count += len(matches)
        
        elab_score = min(elab_count * 0.3, 1.0)
        score += elab_score
        
        # ============================================================
        # FEATURE 10: Penalize responses that are clearly off-topic gibberish
        # ============================================================
        # Check if response has mostly non-alphabetic characters
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        total_chars = len(response_stripped)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.3:
                score -= 2.0
        
        # Check for excessive question marks (response asks questions instead of answering)
        question_count = response_stripped.count('?')
        if question_count > 3 and num_sentences > 0:
            q_ratio = question_count / num_sentences
            if q_ratio > 0.5:
                score -= 0.5
        
        # ============================================================
        # FEATURE 11: Response completeness
        # Penalize truncated responses
        # ============================================================
        if response_stripped.endswith(('...', '…')):
            pass  # Might be truncated but still okay
        # Check if last sentence seems cut off (no ending punctuation)
        last_char = response_stripped[-1] if response_stripped else ''
        if last_char not in '.!?"\')]}:;…' and num_response_words > 20:
            # Possibly truncated, slight penalty
            score -= 0.3
        
        # ============================================================
        # FEATURE 12: Proportionality bonus
        # Response should be proportional to query complexity
        # ============================================================
        query_has_question = '?' in query_stripped
        query_is_complex = num_query_words > 15 or any(
            kw in query_stripped.lower() for kw in [
                'explain', 'why', 'how', 'describe', 'analyze', 'compare',
                'discuss', 'elaborate', 'detail', 'reason'
            ]
        )
        
        if query_is_complex and num_response_words > 30:
            score += 0.5
        elif not query_is_complex and 5 <= num_response_words <= 80:
            # Simple query, concise but informative response
            score += 0.3
        
        # ============================================================
        # Normalize to 0-10 range
        # ============================================================
        # Theoretical max around 12-13, min around -3
        final_score = max(0.0, min(10.0, score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Map [0, 10] through a curve that emphasizes differences in the middle
        normalized = final_score / 10.0  # [0, 1]
        # Slight S-curve: push low scores lower and high scores higher
        adjusted = 1.0 / (1.0 + math.exp(-6 * (normalized - 0.45)))
        final_score = adjusted * 10.0
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score based on basic length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 5:
                return 2.0
            else:
                return 0.5
        except:
            return 1.0