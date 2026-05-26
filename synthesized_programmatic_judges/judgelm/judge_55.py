def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    a dependency/discourse structure analysis approach.
    
    This variant focuses on:
    1. Causal/logical connective density (tracking discourse markers that signal reasoning)
    2. Sentence-to-sentence coherence flow (progressive information building)
    3. Explanation depth via clause complexity
    4. Presence of intermediate reasoning markers vs. bare assertions
    5. Response completeness and non-degeneracy signals
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Tokenize into sentences using a regex approach
        sentences = re.split(r'(?<=[.!?])\s+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Tokenize into words
        words = re.findall(r'[a-zA-Z]+', response_stripped.lower())
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # ============================================================
        # FEATURE 1: Causal/Logical Connective Density
        # These are discourse markers that signal reasoning steps
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\baccordingly\b'
        ]
        
        explanatory_markers = [
            r'\bthis means\b', r'\bthis is because\b', r'\bin other words\b',
            r'\bthat is\b', r'\bspecifically\b', r'\bnamely\b',
            r'\bto clarify\b', r'\bto explain\b', r'\bthe reason\b',
            r'\bthis suggests\b', r'\bthis indicates\b', r'\bthis implies\b'
        ]
        
        sequential_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bafterwards\b', r'\bsubsequently\b',
            r'\bto begin\b', r'\bto start\b', r'\bstep\s+\d+\b',
            r'\binitially\b', r'\blastly\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\bmoreover\b', r'\bin addition\b'
        ]
        
        conditional_markers = [
            r'\bif\b', r'\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bprovided that\b', r'\bin case\b', r'\bwhen\b',
            r'\bwhereas\b', r'\bwhile\b', r'\balthough\b',
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\bdespite\b', r'\binstead\b', r'\bconversely\b'
        ]
        
        response_lower = response_stripped.lower()
        
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_connectives)
        explanatory_count = sum(len(re.findall(p, response_lower)) for p in explanatory_markers)
        sequential_count = sum(len(re.findall(p, response_lower)) for p in sequential_markers)
        conditional_count = sum(len(re.findall(p, response_lower)) for p in conditional_markers)
        
        # Normalize by word count (per 100 words)
        norm = 100.0 / max(word_count, 1)
        causal_density = causal_count * norm
        explanatory_density = explanatory_count * norm
        sequential_density = sequential_count * norm
        conditional_density = conditional_count * norm
        
        # ============================================================
        # FEATURE 2: Clause complexity — subordinate clause indicators
        # More complex sentences with subordination suggest reasoning
        # ============================================================
        subordinators = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b',
            r'\bwhere\b', r'\bwhen\b', r'\bbecause\b', r'\balthough\b',
            r'\bwhile\b', r'\bsince\b', r'\bunless\b', r'\buntil\b',
            r'\bafter\b', r'\bbefore\b', r'\bas\b'
        ]
        subordinate_count = sum(len(re.findall(p, response_lower)) for p in subordinators)
        subordinate_density = subordinate_count * norm
        
        # Comma density as proxy for clause complexity
        comma_count = response_stripped.count(',')
        comma_density = comma_count * norm
        
        # ============================================================
        # FEATURE 3: Sentence-to-sentence information progression
        # Measure how much new information each sentence adds
        # (using word set overlap between consecutive sentences)
        # ============================================================
        if len(sentences) >= 2:
            progression_scores = []
            for i in range(1, len(sentences)):
                prev_words = set(re.findall(r'[a-zA-Z]+', sentences[i-1].lower()))
                curr_words = set(re.findall(r'[a-zA-Z]+', sentences[i].lower()))
                if len(curr_words) == 0:
                    continue
                # New words introduced
                new_words = curr_words - prev_words
                # Shared context words
                shared_words = curr_words & prev_words
                
                new_ratio = len(new_words) / max(len(curr_words), 1)
                shared_ratio = len(shared_words) / max(len(curr_words), 1)
                
                # Good progression: some shared context + some new info
                # Score peaks when there's a balance
                prog_score = min(new_ratio, 0.8) * min(shared_ratio + 0.1, 0.5) * 2
                progression_scores.append(prog_score)
            
            avg_progression = sum(progression_scores) / max(len(progression_scores), 1)
        else:
            avg_progression = 0.0
        
        # ============================================================
        # FEATURE 4: Response substantiveness and non-degeneracy
        # ============================================================
        
        # Unique word ratio (penalize repetitive responses)
        unique_words = set(words)
        unique_ratio = len(unique_words) / max(word_count, 1)
        
        # Sentence count score
        num_sentences = len(sentences)
        
        # Detect repetition/degenerate patterns
        # Check for repeated sentences
        sentence_texts = [s.lower().strip() for s in sentences]
        unique_sentences = set(sentence_texts)
        sentence_uniqueness = len(unique_sentences) / max(len(sentence_texts), 1)
        
        # Check for repeated phrases (trigrams)
        if word_count >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                trigram_repetition = repeated_trigrams / max(len(trigrams), 1)
            else:
                trigram_repetition = 0
        else:
            trigram_repetition = 0
        
        # ============================================================
        # FEATURE 5: Detect "bare assertion" vs "supported claim"
        # Sentences that are very short with no connectives are bare assertions
        # ============================================================
        all_reasoning_patterns = (causal_connectives + explanatory_markers + 
                                   sequential_markers + conditional_markers)
        
        supported_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_marker = any(re.search(p, sent_lower) for p in all_reasoning_patterns)
            # Also count sentences with multiple clauses (commas, semicolons)
            has_complexity = sent.count(',') >= 1 or ';' in sent
            sent_words = len(re.findall(r'[a-zA-Z]+', sent))
            is_substantial = sent_words >= 8
            
            if (has_marker or has_complexity) and is_substantial:
                supported_sentences += 1
        
        support_ratio = supported_sentences / max(num_sentences, 1)
        
        # ============================================================
        # FEATURE 6: Query engagement — does response address the query?
        # ============================================================
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        # Remove very common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                     'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                     'my', 'your', 'his', 'its', 'our', 'their', 'and', 'or',
                     'but', 'not', 'no', 'if', 'so', 'what', 'which', 'who',
                     'how', 'when', 'where', 'why', 'all', 'each', 'every',
                     'any', 'some', 'more', 'most', 'other', 'than', 'too',
                     'very', 'just', 'about', 'up', 'out', 'into', 'over'}
        
        query_content_words = query_words - stopwords
        response_content_words = unique_words - stopwords
        
        if query_content_words:
            query_coverage = len(query_content_words & response_content_words) / max(len(query_content_words), 1)
        else:
            query_coverage = 0.5  # neutral if query has no content words
        
        # ============================================================
        # FEATURE 7: Detect garbage/off-topic content
        # ============================================================
        # HTML/code contamination
        html_tags = len(re.findall(r'<[^>]+>', response_stripped))
        code_indicators = len(re.findall(r'(?:import |def |class |print\(|return )', response_stripped))
        
        # Excessive question marks in response (echoing/not answering)
        question_marks = response_stripped.count('?')
        
        # Check for truncation
        is_truncated = 1.0 if response_stripped[-1] not in '.!?"\')' and word_count > 20 else 0.0
        
        # ============================================================
        # SCORING FORMULA
        # ============================================================
        
        score = 0.0
        
        # Base score from word count (logarithmic, rewarding substance)
        # Very short responses get low base, but diminishing returns
        length_score = min(math.log(max(word_count, 1) + 1) / math.log(200), 1.0) * 2.5
        score += length_score
        
        # Causal/logical connective contribution (0-1.5)
        connective_score = min(causal_density * 0.3 + explanatory_density * 0.4, 1.5)
        score += connective_score
        
        # Sequential/structural markers (0-1.0)
        sequential_score = min(sequential_density * 0.15, 1.0)
        score += sequential_score
        
        # Conditional/contrastive reasoning (0-1.0)
        conditional_score = min(conditional_density * 0.1, 1.0)
        score += conditional_score
        
        # Clause complexity (0-1.0)
        complexity_score = min((subordinate_density * 0.05 + comma_density * 0.03), 1.0)
        score += complexity_score
        
        # Information progression (0-0.75)
        score += avg_progression * 1.5
        
        # Support ratio — proportion of sentences that show reasoning (0-1.5)
        score += support_ratio * 1.5
        
        # Query engagement (0-0.75)
        score += query_coverage * 0.75
        
        # Sentence count bonus for multi-sentence responses (0-0.5)
        sentence_bonus = min(num_sentences / 6.0, 1.0) * 0.5
        score += sentence_bonus
        
        # ============================================================
        # PENALTIES
        # ============================================================
        
        # Repetition penalty
        repetition_penalty = trigram_repetition * 3.0
        score -= repetition_penalty
        
        # Low sentence uniqueness penalty
        if sentence_uniqueness < 0.7:
            score -= (0.7 - sentence_uniqueness) * 2.0
        
        # HTML/code contamination penalty
        if html_tags > 2:
            score -= min(html_tags * 0.3, 2.0)
        if code_indicators > 2:
            score -= min(code_indicators * 0.3, 2.0)
        
        # Too many questions in response (not explaining)
        if question_marks > 3:
            score -= min((question_marks - 3) * 0.2, 1.5)
        
        # Very low unique ratio (degenerate text)
        if unique_ratio < 0.4:
            score -= (0.4 - unique_ratio) * 3.0
        
        # Truncation mild penalty
        if is_truncated:
            score -= 0.3
        
        # Ultra-short response penalty
        if word_count <= 5:
            score *= 0.3
        elif word_count <= 10:
            score *= 0.5
        elif word_count <= 20:
            score *= 0.7
        
        # Single word/character response
        if word_count <= 1:
            return 0.5
        
        # ============================================================
        # NORMALIZE to 0-10 range
        # ============================================================
        score = max(0.0, min(score, 10.0))
        
        # Apply a mild sigmoid-like stretch to increase discrimination
        # Map the raw score through a function that spreads mid-range values
        normalized = score / 10.0  # 0 to 1
        # Stretch: push low scores lower and high scores higher
        stretched = normalized ** 0.85  # mild adjustment
        final_score = stretched * 10.0
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 3.0
            return 0.5
        except Exception:
            return 1.0