def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using sentence-level
    dependency analysis, discourse marker patterns, and structural consistency checks.
    
    This variant focuses on:
    1. Sentence-level logical progression (causal/temporal chains)
    2. Discourse relation detection (contrast, elaboration, cause-effect, condition)
    3. Structural completeness (intro-body-conclusion pattern detection)
    4. Contradiction/repetition detection via negation pattern analysis
    5. Information density and development scoring
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
        
        # Very short responses get a baseline proportional to minimal coherence
        if len(response_stripped) < 5:
            return 0.5
        
        # === SENTENCE EXTRACTION ===
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        if not sentences:
            return 1.0
        
        num_sentences = len(sentences)
        
        # === 1. DISCOURSE RELATION DETECTION ===
        # Categorize discourse markers by their logical function
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bhence\b', r'\bso\b', r'\baccordingly\b', r'\bfor this reason\b',
            r'\bthat is why\b', r'\bleading to\b', r'\bcaused by\b'
        ]
        
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bincluding\b'
        ]
        
        contrast_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bdespite\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bbut\b', r'\byet\b',
            r'\bin contrast\b', r'\bon the contrary\b', r'\bstill\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bin case\b', r'\bwhen\b', r'\bwhenever\b', r'\bas long as\b'
        ]
        
        additive_markers = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\blikewise\b', r'\bsimilarly\b', r'\bnot only\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bto sum up\b',
            r'\bultimately\b', r'\bin the end\b', r'\ball in all\b'
        ]
        
        temporal_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bpreviously\b', r'\bmeanwhile\b',
            r'\binitially\b', r'\beventually\b', r'\bbefore\b', r'\bafter\b'
        ]
        
        resp_lower = response_stripped.lower()
        
        def count_markers(patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, resp_lower))
            return total
        
        causal_count = count_markers(causal_markers)
        elaboration_count = count_markers(elaboration_markers)
        contrast_count = count_markers(contrast_markers)
        conditional_count = count_markers(conditional_markers)
        additive_count = count_markers(additive_markers)
        conclusion_count = count_markers(conclusion_markers)
        temporal_count = count_markers(temporal_markers)
        
        total_discourse = (causal_count + elaboration_count + contrast_count +
                          conditional_count + additive_count + conclusion_count +
                          temporal_count)
        
        # Discourse diversity: how many different types are used
        discourse_types_used = sum(1 for c in [causal_count, elaboration_count,
                                                contrast_count, conditional_count,
                                                additive_count, conclusion_count,
                                                temporal_count] if c > 0)
        
        # Normalize discourse density by number of sentences
        discourse_density = total_discourse / max(num_sentences, 1)
        # Cap at reasonable level
        discourse_density_score = min(discourse_density, 1.0) * 10
        
        # Diversity bonus (using multiple types shows structured thinking)
        diversity_score = min(discourse_types_used / 4.0, 1.0) * 10
        
        # === 2. SENTENCE-LEVEL LOGICAL PROGRESSION ===
        # Check if consecutive sentences share semantic content (topic continuity)
        def get_content_words(text):
            # Remove common stop words and get content words
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                'as', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'out', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'because',
                'but', 'and', 'or', 'if', 'while', 'that', 'this', 'these',
                'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
                'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
                'their', 'what', 'which', 'who', 'whom'
            }
            words = re.findall(r'[a-z]+', text.lower())
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        # Topic continuity: measure content word overlap between consecutive sentences
        continuity_scores = []
        if num_sentences >= 2:
            for i in range(len(sentences) - 1):
                words_a = get_content_words(sentences[i])
                words_b = get_content_words(sentences[i + 1])
                if words_a and words_b:
                    # Dice coefficient for overlap
                    overlap = len(words_a & words_b)
                    dice = (2 * overlap) / (len(words_a) + len(words_b))
                    continuity_scores.append(dice)
                else:
                    continuity_scores.append(0.0)
        
        # Average continuity (moderate overlap is good; too high = repetitive, too low = incoherent)
        if continuity_scores:
            avg_continuity = sum(continuity_scores) / len(continuity_scores)
            # Optimal continuity is around 0.15-0.4
            if avg_continuity < 0.05:
                continuity_quality = 2.0  # Very low overlap - disconnected
            elif avg_continuity < 0.15:
                continuity_quality = 5.0
            elif avg_continuity <= 0.5:
                continuity_quality = 8.0  # Good overlap range
            elif avg_continuity <= 0.7:
                continuity_quality = 5.0  # Getting repetitive
            else:
                continuity_quality = 2.0  # Very repetitive
        else:
            continuity_quality = 4.0  # Single sentence - neutral
        
        # === 3. INFORMATION DEVELOPMENT (New ideas introduced per sentence) ===
        cumulative_words = set()
        new_info_per_sentence = []
        for sent in sentences:
            content = get_content_words(sent)
            new_words = content - cumulative_words
            if content:
                new_ratio = len(new_words) / len(content)
                new_info_per_sentence.append(new_ratio)
            else:
                new_info_per_sentence.append(0.0)
            cumulative_words |= content
        
        if new_info_per_sentence:
            avg_new_info = sum(new_info_per_sentence) / len(new_info_per_sentence)
            # Good responses develop ideas: high new info early, moderate later
            # Very high avg means no connection; very low means repetitive
            if 0.3 <= avg_new_info <= 0.8:
                info_dev_score = 8.0
            elif 0.2 <= avg_new_info < 0.3 or 0.8 < avg_new_info <= 0.9:
                info_dev_score = 6.0
            elif avg_new_info > 0.9:
                info_dev_score = 4.0  # Sentences barely connect
            else:
                info_dev_score = 3.0  # Too repetitive
        else:
            info_dev_score = 3.0
        
        # === 4. REPETITION / CIRCULAR REASONING DETECTION ===
        # Check for near-duplicate sentences
        duplicate_count = 0
        if num_sentences >= 2:
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    words_i = get_content_words(sentences[i])
                    words_j = get_content_words(sentences[j])
                    if words_i and words_j:
                        overlap = len(words_i & words_j)
                        union = len(words_i | words_j)
                        if union > 0 and overlap / union > 0.75:
                            duplicate_count += 1
        
        max_possible_pairs = max(num_sentences * (num_sentences - 1) / 2, 1)
        repetition_ratio = duplicate_count / max_possible_pairs
        repetition_penalty = repetition_ratio * 15  # Penalty for repetition
        
        # === 5. STRUCTURAL COMPLETENESS ===
        # Check for complete sentences (starting with capital, ending with punctuation)
        complete_sentences = 0
        for sent in sentences:
            sent_clean = sent.strip()
            if sent_clean and sent_clean[0].isupper() and sent_clean[-1] in '.!?':
                complete_sentences += 1
        
        completeness_ratio = complete_sentences / max(num_sentences, 1)
        completeness_score = completeness_ratio * 8
        
        # === 6. RESPONSE RELEVANCE TO QUERY ===
        query_content = get_content_words(query)
        response_content = get_content_words(response_stripped)
        
        if query_content and response_content:
            relevance_overlap = len(query_content & response_content)
            relevance_score = min(relevance_overlap / max(len(query_content), 1), 1.0) * 6
        else:
            relevance_score = 3.0
        
        # === 7. RESPONSE LENGTH AND SUBSTANCE ===
        word_count = len(response_stripped.split())
        
        if word_count < 3:
            length_score = 0.5
        elif word_count < 10:
            length_score = 2.0
        elif word_count < 25:
            length_score = 4.0
        elif word_count < 50:
            length_score = 6.0
        elif word_count < 150:
            length_score = 7.0
        elif word_count < 300:
            length_score = 6.5
        else:
            length_score = 5.5  # Very long might ramble
        
        # === 8. GARBAGE / OFF-TOPIC DETECTION ===
        garbage_penalty = 0.0
        
        # Check for code when query doesn't ask for code
        code_indicators = ['import ', 'def ', 'class ', 'return ', '#!/', 'void ', 'int main']
        query_asks_code = any(w in query.lower() for w in ['code', 'program', 'function', 'script', 'html', 'python', 'java'])
        
        if not query_asks_code:
            code_lines = sum(1 for indicator in code_indicators if indicator in response_stripped)
            if code_lines >= 2:
                garbage_penalty += 4.0
        
        # Check for HTML tags when not asked
        query_asks_html = any(w in query.lower() for w in ['html', 'tag', 'web', 'page'])
        if not query_asks_html:
            html_tags = len(re.findall(r'<[^>]+>', response_stripped))
            if html_tags > 3:
                garbage_penalty += 3.0
        
        # Check for excessive repeated patterns (like "Output:" repeated)
        repeated_pattern_matches = re.findall(r'(\b\w+:)\s*', response_stripped)
        if repeated_pattern_matches:
            pattern_counter = Counter(repeated_pattern_matches)
            max_repeat = max(pattern_counter.values())
            if max_repeat > 3:
                garbage_penalty += 2.0
        
        # Check for question-answer format when not appropriate (hallucinated Q&A)
        qa_patterns = len(re.findall(r'(?:Question|Answer|Input|Output)\s*:', response_stripped))
        if qa_patterns > 4 and 'question' not in query.lower():
            garbage_penalty += 3.0
        
        # === 9. NEGATION CONTRADICTION CHECK ===
        # Simple check: if the same claim appears both affirmed and negated
        contradiction_penalty = 0.0
        negation_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bcan\b', r'\bcannot\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\bshould\b', r'\bshould not\b'),
        ]
        for affirm, negate in negation_patterns:
            # Check if both appear in close proximity (within same paragraph context)
            has_affirm = bool(re.search(affirm, resp_lower))
            has_negate = bool(re.search(negate, resp_lower))
            # This is a very rough heuristic - only penalize if both appear multiple times
            if has_affirm and has_negate:
                affirm_count = len(re.findall(affirm, resp_lower))
                negate_count = len(re.findall(negate, resp_lower))
                if affirm_count > 1 and negate_count > 1:
                    contradiction_penalty += 0.5
        
        # === 10. SENTENCE VARIETY (syntactic diversity) ===
        # Check if sentences start with different words
        if num_sentences >= 3:
            first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
            unique_starts = len(set(first_words))
            variety_ratio = unique_starts / num_sentences
            variety_score = variety_ratio * 5
        else:
            variety_score = 3.0
        
        # === COMPOSITE SCORE ===
        # Weight the components
        raw_score = (
            discourse_density_score * 0.12 +   # Discourse marker density
            diversity_score * 0.10 +            # Discourse type diversity
            continuity_quality * 0.15 +         # Topic continuity
            info_dev_score * 0.10 +             # Information development
            completeness_score * 0.12 +         # Structural completeness
            relevance_score * 0.12 +            # Query relevance
            length_score * 0.15 +               # Substance/length
            variety_score * 0.08 +              # Syntactic variety
            0 * 0.06                            # Reserved
        )
        
        # Apply penalties
        final_score = raw_score - repetition_penalty - garbage_penalty - contradiction_penalty
        
        # Bonus for multi-sentence coherent responses
        if num_sentences >= 3 and discourse_types_used >= 2 and completeness_ratio > 0.5:
            final_score += 1.0
        
        # Penalty for single-word or near-empty responses
        if word_count <= 2:
            final_score = min(final_score, 1.0)
        elif word_count <= 5:
            final_score = min(final_score, 3.0)
        
        # Clamp to [0, 10]
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except Exception:
            return 3.0