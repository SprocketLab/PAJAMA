def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    analysis: causal/logical connectors, sentence dependency chains, 
    contradiction detection, and structural completeness.
    
    This variant focuses on:
    1. Sentence-level logical flow (entailment-like heuristics between consecutive sentences)
    2. Causal and logical connector density and proper usage
    3. Structural completeness (intro/body/conclusion patterns)
    4. Contradiction and repetition detection via negation patterns
    5. Sentence complexity and information progression
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Very short responses get low scores for coherence
        if len(response_stripped) < 5:
            return 0.5
        
        # Split into sentences using multiple delimiters
        def split_sentences(text):
            # Split on sentence-ending punctuation, but be careful with abbreviations
            raw = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text)
            sentences = []
            for s in raw:
                s = s.strip()
                if len(s) > 2:
                    sentences.append(s)
            # If no splits found, treat whole text as one sentence
            if not sentences and text.strip():
                sentences = [text.strip()]
            return sentences
        
        sentences = split_sentences(response_stripped)
        num_sentences = len(sentences)
        
        # ============================================================
        # FEATURE 1: Causal/Logical Connector Analysis
        # ============================================================
        causal_connectors = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\baccordingly\b', r'\bfor this reason\b'
        ]
        
        logical_connectors = [
            r'\bhowever\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\balthough\b',
            r'\bdespite\b', r'\bwhile\b', r'\bwhereas\b', r'\byet\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\badditionally\b', r'\balso\b', r'\bsimilarly\b',
            r'\blikewise\b', r'\bspecifically\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bin particular\b',
            r'\bin fact\b', r'\bindeed\b', r'\bclearly\b'
        ]
        
        concluding_connectors = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b',
            r'\bin summary\b', r'\bfinally\b', r'\bultimately\b',
            r'\bin short\b', r'\bto conclude\b'
        ]
        
        response_lower = response_stripped.lower()
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_connectors)
        logical_count = sum(len(re.findall(p, response_lower)) for p in logical_connectors)
        concluding_count = sum(len(re.findall(p, response_lower)) for p in concluding_connectors)
        
        total_connectors = causal_count + logical_count + concluding_count
        
        # Normalize by number of sentences
        if num_sentences > 0:
            connector_density = total_connectors / num_sentences
        else:
            connector_density = 0
        
        # Score: 0-1 range, sweet spot around 0.3-0.6 connectors per sentence
        connector_score = min(1.0, connector_density / 0.5) if connector_density <= 0.8 else max(0.5, 1.0 - (connector_density - 0.8))
        
        # ============================================================
        # FEATURE 2: Sentence-to-Sentence Lexical Coherence Chain
        # ============================================================
        def get_content_words(text):
            """Extract content words (non-stopwords) from text."""
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 'just', 'and', 'but', 'or', 'if', 'it', 'its', 'this',
                'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
                'which', 'who', 'whom'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        coherence_scores = []
        if num_sentences >= 2:
            for i in range(1, num_sentences):
                prev_words = get_content_words(sentences[i-1])
                curr_words = get_content_words(sentences[i])
                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    jaccard = overlap / union if union > 0 else 0
                    coherence_scores.append(jaccard)
                else:
                    coherence_scores.append(0.0)
        
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            # Check for monotonic decay (which might indicate rambling)
            coherence_variance = sum((c - avg_coherence)**2 for c in coherence_scores) / len(coherence_scores)
        else:
            avg_coherence = 0.3  # neutral for single sentence
            coherence_variance = 0
        
        # Good coherence: moderate overlap (not too high = repetitive, not too low = disjointed)
        # Sweet spot: 0.05-0.25
        if avg_coherence < 0.01:
            coherence_chain_score = 0.2
        elif avg_coherence < 0.05:
            coherence_chain_score = 0.4
        elif avg_coherence <= 0.3:
            coherence_chain_score = 0.8 + 0.2 * min(1.0, avg_coherence / 0.3)
        else:
            # Too much overlap might mean repetition
            coherence_chain_score = max(0.3, 1.0 - (avg_coherence - 0.3) * 2)
        
        # ============================================================
        # FEATURE 3: Information Progression (new content per sentence)
        # ============================================================
        seen_words = set()
        new_info_ratios = []
        for sent in sentences:
            words = get_content_words(sent)
            if words:
                new_words = words - seen_words
                ratio = len(new_words) / len(words)
                new_info_ratios.append(ratio)
                seen_words.update(words)
            else:
                new_info_ratios.append(0.0)
        
        if new_info_ratios:
            avg_new_info = sum(new_info_ratios) / len(new_info_ratios)
        else:
            avg_new_info = 0
        
        # Good: high new info ratio means content is progressing, not repeating
        info_progression_score = min(1.0, avg_new_info * 1.2)
        
        # ============================================================
        # FEATURE 4: Repetition Detection (exact and near-duplicate sentences)
        # ============================================================
        repetition_penalty = 0.0
        if num_sentences >= 2:
            normalized_sents = [re.sub(r'[^a-z\s]', '', s.lower()).strip() for s in sentences]
            sent_counter = Counter(normalized_sents)
            duplicates = sum(c - 1 for c in sent_counter.values() if c > 1)
            repetition_penalty = min(1.0, duplicates / max(1, num_sentences) * 2)
        
        # ============================================================
        # FEATURE 5: Contradiction/Negation Pattern Detection
        # ============================================================
        negation_patterns = [
            (r'\bis\b', r'\bis not\b'), (r'\bcan\b', r'\bcannot\b'),
            (r'\bwill\b', r'\bwill not\b'), (r'\bshould\b', r'\bshould not\b'),
            (r'\btrue\b', r'\bfalse\b'), (r'\bcorrect\b', r'\bincorrect\b'),
            (r'\byes\b', r'\bno\b'), (r'\balways\b', r'\bnever\b')
        ]
        
        contradiction_score = 0.0
        for pos_pat, neg_pat in negation_patterns:
            pos_matches = re.findall(pos_pat, response_lower)
            neg_matches = re.findall(neg_pat, response_lower)
            if pos_matches and neg_matches:
                # Having both might indicate contradiction or nuanced discussion
                # Only penalize if they appear close together
                contradiction_score += 0.05
        
        contradiction_penalty = min(0.5, contradiction_score)
        
        # ============================================================
        # FEATURE 6: Structural Completeness
        # ============================================================
        # Check if response seems truncated
        truncation_penalty = 0.0
        if response_stripped[-1] not in '.!?"\')]}':
            # Might be truncated
            truncation_penalty = 0.15
        if re.search(r'\b\w+$', response_stripped) and len(response_stripped) > 100:
            # Ends mid-word or mid-sentence for long responses
            truncation_penalty = 0.25
        
        # ============================================================
        # FEATURE 7: Response Substance and Relevance
        # ============================================================
        words_in_response = re.findall(r'[a-z]+', response_lower)
        word_count = len(words_in_response)
        
        # Very short responses lack argument structure
        if word_count < 3:
            substance_score = 0.1
        elif word_count < 10:
            substance_score = 0.3
        elif word_count < 25:
            substance_score = 0.5
        elif word_count < 50:
            substance_score = 0.7
        elif word_count <= 300:
            substance_score = 0.9
        else:
            substance_score = 0.85  # Very long might be rambling
        
        # ============================================================
        # FEATURE 8: Query-Response Alignment
        # ============================================================
        query_content = get_content_words(query)
        response_content = get_content_words(response_stripped)
        
        if query_content and response_content:
            relevance_overlap = len(query_content & response_content) / len(query_content)
            relevance_score = min(1.0, relevance_overlap * 1.5)
        else:
            relevance_score = 0.3
        
        # ============================================================
        # FEATURE 9: Noise / Off-topic Detection
        # ============================================================
        noise_penalty = 0.0
        
        # Check for code-like content when query doesn't ask for code
        code_indicators = ['import ', 'def ', 'class ', 'return ', '#!/', 'void ', 'int main']
        query_asks_code = any(w in query.lower() for w in ['code', 'program', 'function', 'script', 'html', 'python', 'java'])
        
        if not query_asks_code:
            code_lines = sum(1 for ci in code_indicators if ci in response_stripped)
            if code_lines > 0:
                noise_penalty += 0.3
        
        # Check for HTML when not asked
        query_asks_html = any(w in query.lower() for w in ['html', 'tag', 'web', 'markup'])
        if not query_asks_html and re.search(r'<[a-z]+>', response_lower):
            html_tags = len(re.findall(r'<[a-z]+>', response_lower))
            if html_tags > 2:
                noise_penalty += 0.2
        
        # Check for excessive "Output:" or "Input:" patterns (template-like)
        template_patterns = len(re.findall(r'\b(Input|Output|Question|Answer)\s*:', response_stripped))
        if template_patterns > 3:
            noise_penalty += min(0.4, template_patterns * 0.08)
        
        # ============================================================
        # FEATURE 10: Sentence Complexity Progression
        # ============================================================
        sent_lengths = [len(re.findall(r'\w+', s)) for s in sentences]
        if len(sent_lengths) >= 2:
            # Good writing has varied sentence lengths
            avg_len = sum(sent_lengths) / len(sent_lengths)
            if avg_len > 0:
                length_cv = (sum((l - avg_len)**2 for l in sent_lengths) / len(sent_lengths)) ** 0.5 / avg_len
            else:
                length_cv = 0
            # Moderate variation is good
            variety_score = min(1.0, length_cv / 0.5) if length_cv <= 1.0 else max(0.5, 1.0 - (length_cv - 1.0) * 0.5)
        else:
            variety_score = 0.4  # Single sentence, neutral
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        # Weighted combination
        raw_score = (
            connector_score * 1.0 +          # Logical connectors
            coherence_chain_score * 1.5 +     # Sentence-to-sentence coherence
            info_progression_score * 1.2 +    # Information progression
            substance_score * 2.0 +           # Response substance
            relevance_score * 1.5 +           # Query relevance
            variety_score * 0.8               # Sentence variety
        )
        
        max_raw = 1.0 + 1.5 + 1.2 + 2.0 + 1.5 + 0.8  # = 8.0
        
        # Normalize to 0-1
        normalized = raw_score / max_raw
        
        # Apply penalties
        total_penalty = repetition_penalty * 0.3 + contradiction_penalty * 0.15 + truncation_penalty * 0.5 + noise_penalty * 0.8
        
        final_normalized = max(0.0, normalized - total_penalty)
        
        # Scale to 0-10
        score = final_normalized * 10.0
        
        # Clamp
        score = max(0.5, min(10.0, score))
        
        return round(score, 2)
    
    except Exception:
        # Fallback: return a middling score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 1.0
        except Exception:
            return 1.0