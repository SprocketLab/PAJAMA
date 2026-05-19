def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a STRUCTURAL and RHETORICAL analysis approach:
    - Sentence-level analysis of claim types (factual vs speculative vs hedged)
    - Rhetorical sophistication (use of qualifiers, conditional structures, evidence references)
    - Penalizes absolutist language patterns and repetition
    - Rewards proportional uncertainty based on query ambiguity
    - Analyzes sentence complexity and information density
    """
    try:
        if not response or not isinstance(response, str) or response.strip() == "":
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        query = query.strip()
        
        import re
        from collections import Counter
        import math
        
        # ---- Helper: split into sentences ----
        def split_sentences(text):
            # Split on sentence-ending punctuation
            sents = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sents if s.strip()]
        
        sentences = split_sentences(response)
        words = re.findall(r'[a-z]+(?:\'[a-z]+)?', response.lower())
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # ---- 1. QUERY AMBIGUITY DETECTION ----
        # Determines how much epistemic calibration we expect
        ambiguity_signals = [
            r'\bwhy\b', r'\bwhat\s+(?:did|does|do)\s+.*\bmean\b', r'\bexplain\b',
            r'\bopinion\b', r'\bbelieve\b', r'\bhypothetical\b', r'\bcompare\b',
            r'\bcontrast\b', r'\bdescribe\b', r'\binterpret\b', r'\banalyze\b',
            r'\bdiscuss\b', r'\bevaluate\b', r'\bcreative\b', r'\bimagine\b',
            r'\bpredict\b', r'\bfuture\b', r'\bshould\b', r'\bcould\b',
            r'\bmight\b', r'\bpossible\b', r'\bspeculate\b'
        ]
        factual_signals = [
            r'\blist\b', r'\bname\b', r'\brewrite\b', r'\bprovide\b',
            r'\bgenerate\b', r'\bcreate\b', r'\bwrite\b', r'\bcrop\b',
            r'\bconvert\b', r'\btranslate\b', r'\bcalculate\b', r'\bdefine\b'
        ]
        
        query_lower = query.lower()
        ambiguity_score = sum(1 for p in ambiguity_signals if re.search(p, query_lower))
        factual_score = sum(1 for p in factual_signals if re.search(p, query_lower))
        
        # 0 to 1 scale: how ambiguous/subjective is the query
        query_subjectivity = min(1.0, ambiguity_score * 0.15) - min(0.5, factual_score * 0.1)
        query_subjectivity = max(0.0, min(1.0, query_subjectivity + 0.3))  # baseline
        
        # ---- 2. SENTENCE-LEVEL CLAIM CLASSIFICATION ----
        # Classify each sentence as: hedged, evidenced, conditional, absolute, or neutral
        
        hedged_patterns = [
            r'\b(?:perhaps|possibly|arguably|conceivably|plausibly)\b',
            r'\b(?:tends?\s+to|generally|typically|usually|often|sometimes)\b',
            r'\b(?:may|might|could)\s+(?:be|have|suggest|indicate|mean)\b',
            r'\bit\s+(?:seems?|appears?)\b',
            r'\b(?:likely|unlikely|probable|improbable)\b',
            r'\b(?:in\s+(?:some|many|most)\s+cases)\b',
            r'\b(?:to\s+some\s+(?:extent|degree))\b',
            r'\b(?:not\s+necessarily|not\s+always)\b',
        ]
        
        evidence_patterns = [
            r'\b(?:research|studies|evidence|data|findings)\s+(?:suggests?|shows?|indicates?)\b',
            r'\b(?:according\s+to)\b',
            r'\b(?:for\s+(?:example|instance))\b',
            r'\b(?:such\s+as)\b',
            r'\b(?:e\.g\.|i\.e\.)\b',
            r'\b(?:specifically|in\s+particular)\b',
        ]
        
        conditional_patterns = [
            r'\b(?:if|when|unless|provided\s+that|assuming)\b.*,',
            r'\b(?:depending\s+on|contingent\s+upon)\b',
            r'\b(?:in\s+(?:this|that)\s+(?:case|scenario|context))\b',
            r'\b(?:under\s+(?:certain|some|these)\s+(?:conditions|circumstances))\b',
        ]
        
        absolute_patterns = [
            r'\b(?:always|never|definitely|certainly|undoubtedly|unquestionably)\b',
            r'\b(?:every(?:one|thing|body|where))\b',
            r'\b(?:no\s+(?:one|thing|body))\s+(?:can|will|does)\b',
            r'\b(?:it\s+is\s+(?:clear|obvious|evident|certain)\s+that)\b',
            r'\b(?:without\s+(?:a\s+)?doubt)\b',
            r'\b(?:there\s+is\s+no\s+(?:question|doubt))\b',
            r'\b(?:the\s+(?:only|best|worst|greatest))\b',
        ]
        
        hedged_count = 0
        evidenced_count = 0
        conditional_count = 0
        absolute_count = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            is_hedged = any(re.search(p, sent_lower) for p in hedged_patterns)
            is_evidenced = any(re.search(p, sent_lower) for p in evidence_patterns)
            is_conditional = any(re.search(p, sent_lower) for p in conditional_patterns)
            is_absolute = any(re.search(p, sent_lower) for p in absolute_patterns)
            
            if is_hedged:
                hedged_count += 1
            if is_evidenced:
                evidenced_count += 1
            if is_conditional:
                conditional_count += 1
            if is_absolute:
                absolute_count += 1
        
        num_sentences = max(1, len(sentences))
        
        hedged_ratio = hedged_count / num_sentences
        evidenced_ratio = evidenced_count / num_sentences
        conditional_ratio = conditional_count / num_sentences
        absolute_ratio = absolute_count / num_sentences
        
        # ---- 3. INFORMATION DENSITY & STRUCTURAL QUALITY ----
        
        # Unique word ratio (penalize repetition heavily)
        word_freq = Counter(words)
        unique_ratio = len(word_freq) / max(1, word_count)
        
        # Detect severe repetition (same phrase repeated)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)] if len(words) > 1 else []
        bigram_freq = Counter(bigrams)
        if bigrams:
            max_bigram_freq = max(bigram_freq.values())
            repetition_penalty = max(0, (max_bigram_freq - 3) * 0.5)
        else:
            repetition_penalty = 0
        
        # Average sentence length (words per sentence) - moderate is best
        avg_sent_len = word_count / num_sentences
        sent_len_score = 0
        if avg_sent_len < 4:
            sent_len_score = 0.2
        elif avg_sent_len < 8:
            sent_len_score = 0.5
        elif avg_sent_len <= 25:
            sent_len_score = 1.0
        elif avg_sent_len <= 40:
            sent_len_score = 0.7
        else:
            sent_len_score = 0.4
        
        # Number of distinct points/ideas (proxy: number of meaningful sentences)
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 4]
        content_richness = min(1.0, len(meaningful_sentences) / max(1, 3))
        
        # ---- 4. RHETORICAL SOPHISTICATION ----
        # Connective/discourse markers that show structured reasoning
        discourse_markers = [
            r'\b(?:however|nevertheless|nonetheless|although|though|while)\b',
            r'\b(?:furthermore|moreover|additionally|in\s+addition)\b',
            r'\b(?:therefore|thus|consequently|as\s+a\s+result)\b',
            r'\b(?:on\s+the\s+other\s+hand|in\s+contrast|conversely)\b',
            r'\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|finally)\b',
            r'\b(?:for\s+this\s+reason|because\s+of\s+this)\b',
            r'\b(?:importantly|significantly|notably)\b',
        ]
        
        response_lower = response.lower()
        discourse_count = sum(1 for p in discourse_markers if re.search(p, response_lower))
        discourse_score = min(1.0, discourse_count * 0.2)
        
        # ---- 5. PROPORTIONAL CALIBRATION SCORING ----
        # The key insight: calibration quality depends on query type
        
        # For subjective/ambiguous queries: reward hedging, penalize absolutes
        # For factual/task queries: don't penalize confidence, reward completeness
        
        calibration_score = 0.0
        
        # Base epistemic quality: hedging + evidence + conditionals are good
        epistemic_quality = (
            hedged_ratio * 2.0 +
            evidenced_ratio * 2.5 +
            conditional_ratio * 1.5
        )
        
        # Absolute penalty scales with query subjectivity
        absolute_penalty = absolute_ratio * query_subjectivity * 2.0
        
        # For factual queries, being direct is fine
        directness_bonus = 0
        if query_subjectivity < 0.3:
            directness_bonus = (1 - hedged_ratio) * 0.3  # reward directness for factual queries
        
        calibration_score = min(3.0, epistemic_quality) - absolute_penalty + directness_bonus
        
        # ---- 6. COMPLETENESS AND SUBSTANCE ----
        # Longer, more substantive responses tend to be better (up to a point)
        
        if word_count < 5:
            length_score = 0.1
        elif word_count < 15:
            length_score = 0.3
        elif word_count < 30:
            length_score = 0.6
        elif word_count < 80:
            length_score = 0.9
        elif word_count <= 200:
            length_score = 1.0
        elif word_count <= 400:
            length_score = 0.9
        else:
            length_score = 0.7
        
        # ---- 7. EXPLANATION DEPTH ----
        # Check if response explains reasoning, not just states facts
        explanation_markers = [
            r'\b(?:because|since|due\s+to|as\s+a\s+result\s+of)\b',
            r'\b(?:this\s+(?:means|implies|suggests|indicates))\b',
            r'\b(?:in\s+other\s+words|that\s+is\s+to\s+say)\b',
            r'\b(?:the\s+reason\s+(?:is|being|for))\b',
            r'\b(?:which\s+(?:means|suggests|implies))\b',
        ]
        
        explanation_count = sum(1 for p in explanation_markers if re.search(p, response_lower))
        explanation_score = min(1.0, explanation_count * 0.25)
        
        # ---- 8. QUERY RELEVANCE (lightweight) ----
        # Check if response words overlap with query words meaningfully
        query_words = set(re.findall(r'[a-z]+', query_lower))
        response_words = set(words)
        
        # Remove very common words for relevance check
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'it', 'its', 'this', 'that',
                     'and', 'or', 'but', 'not', 'no', 'if', 'so', 'as', 'what', 'how',
                     'when', 'where', 'who', 'which', 'than', 'then', 'also', 'just',
                     'more', 'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many',
                     'much', 'very', 'too', 'only', 'own', 'same', 'other', 'such'}
        
        q_content = query_words - stopwords
        r_content = response_words - stopwords
        
        if q_content:
            relevance = len(q_content & r_content) / len(q_content)
        else:
            relevance = 0.5  # neutral if query has no content words
        
        relevance_score = min(1.0, relevance)
        
        # ---- 9. DETECT DEGENERATE RESPONSES ----
        # Truncated responses, gibberish, pure repetition
        degenerate_penalty = 0
        
        # Check if response is truncated (ends mid-word or mid-sentence without punctuation)
        if response and response[-1] not in '.!?"\')':
            degenerate_penalty += 0.5
        
        # Check for extreme repetition
        if word_count > 10:
            most_common_word, most_common_count = word_freq.most_common(1)[0]
            if most_common_word not in stopwords and most_common_count / word_count > 0.2:
                degenerate_penalty += 1.0
        
        # Very short response for a complex query
        if word_count < 10 and len(query.split()) > 5:
            degenerate_penalty += 0.5
        
        # ---- FINAL SCORE COMPOSITION ----
        # Weights designed to make this variant structurally different from others
        
        score = (
            calibration_score * 1.2 +          # epistemic calibration (0-3.6)
            discourse_score * 1.5 +              # rhetorical sophistication (0-1.5)
            explanation_score * 1.3 +             # explanation depth (0-1.3)
            sent_len_score * 0.8 +                # sentence structure (0-0.8)
            content_richness * 1.0 +              # content richness (0-1.0)
            unique_ratio * 1.5 +                  # vocabulary diversity (0-1.5)
            length_score * 1.5 +                  # completeness (0-1.5)
            relevance_score * 1.2 +               # relevance (0-1.2)
            - repetition_penalty -                # repetition penalty
            - degenerate_penalty                  # degenerate penalty
        )
        
        # Normalize to 0-10 range
        # Theoretical max is about 12.4, min is negative
        score = max(0.0, min(10.0, score))
        
        return round(score, 3)
        
    except Exception:
        # Never crash - return a neutral score
        try:
            if response and len(response.strip()) > 0:
                return 3.0
        except Exception:
            pass
        return 0.0