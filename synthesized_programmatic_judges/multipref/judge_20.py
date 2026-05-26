def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question decomposition and coverage analysis
    - Information density via unique noun-phrase-like chunks
    - Structural depth (nested lists, multi-level headers, code blocks)
    - Specificity signals (numbers, proper nouns, examples, citations)
    - Ratio-based coverage of query terms and their semantic neighbors
    - Sentence-level topic progression (are new ideas introduced throughout?)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.0
        
        # ---- Helper functions ----
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sents if len(s.strip()) > 5]
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Common stopwords
        STOPS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'also', 'like', 'get', 'got', 'make', 'made'
        }
        
        query_tokens = tokenize(query)
        resp_tokens = tokenize(response)
        
        if len(resp_tokens) < 3:
            return 0.5
        
        query_content = [t for t in query_tokens if t not in STOPS and len(t) > 2]
        resp_content = [t for t in resp_tokens if t not in STOPS and len(t) > 2]
        
        # ============================================================
        # FEATURE 1: Query Decomposition and Sub-question Coverage
        # ============================================================
        # Extract question words and key phrases from query
        question_fragments = re.split(r'[,;?.]', query)
        question_fragments = [f.strip() for f in question_fragments if len(f.strip()) > 3]
        
        # For each fragment, check if its key content words appear in response
        fragment_coverage = 0.0
        if question_fragments:
            for frag in question_fragments:
                frag_tokens = [t for t in tokenize(frag) if t not in STOPS and len(t) > 2]
                if frag_tokens:
                    covered = sum(1 for t in frag_tokens if t in set(resp_tokens))
                    fragment_coverage += covered / len(frag_tokens)
            fragment_coverage /= len(question_fragments)
        else:
            fragment_coverage = 0.5
        
        # ============================================================
        # FEATURE 2: Information Density — unique content bigrams
        # ============================================================
        resp_bigrams = get_ngrams(resp_content, 2)
        unique_bigrams = len(set(resp_bigrams))
        total_bigrams = max(len(resp_bigrams), 1)
        bigram_diversity = unique_bigrams / total_bigrams  # 0 to 1
        
        # Absolute unique bigram count (normalized)
        bigram_richness = min(unique_bigrams / 80.0, 1.0)
        
        # ============================================================
        # FEATURE 3: Topic Progression — new concepts per sentence quarter
        # ============================================================
        sentences = get_sentences(response)
        num_sents = len(sentences)
        
        topic_progression_score = 0.0
        if num_sents >= 4:
            quarter = max(num_sents // 4, 1)
            quarters = [
                sentences[:quarter],
                sentences[quarter:2*quarter],
                sentences[2*quarter:3*quarter],
                sentences[3*quarter:]
            ]
            
            seen_words = set()
            quarters_with_new = 0
            for q_sents in quarters:
                q_text = ' '.join(q_sents)
                q_tokens = set(t for t in tokenize(q_text) if t not in STOPS and len(t) > 2)
                new_words = q_tokens - seen_words
                if len(new_words) >= 3:
                    quarters_with_new += 1
                seen_words.update(q_tokens)
            
            topic_progression_score = quarters_with_new / 4.0
        elif num_sents >= 1:
            topic_progression_score = 0.5
        
        # ============================================================
        # FEATURE 4: Specificity Signals
        # ============================================================
        # Numbers (dates, quantities, measurements)
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:kg|m|cm|mm|ft|lb|oz|mph|km|%|degrees?|°))?', response)
        number_score = min(len(numbers) / 8.0, 1.0)
        
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]{2,}', response)
        proper_noun_score = min(len(set(proper_nouns)) / 6.0, 1.0)
        
        # Examples signaled explicitly
        example_markers = len(re.findall(
            r'\b(?:for example|for instance|e\.g\.|such as|like\s+\w+\s+and|including|consider)\b',
            response.lower()
        ))
        example_score = min(example_markers / 3.0, 1.0)
        
        specificity = (number_score * 0.3 + proper_noun_score * 0.3 + example_score * 0.4)
        
        # ============================================================
        # FEATURE 5: Structural Depth
        # ============================================================
        # Numbered items (1. 2. 3. or 1) 2) 3) or a. b. c.)
        numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*|-|•|[a-z][\.\)])', response)
        list_depth = min(len(numbered_items) / 6.0, 1.0)
        
        # Headers (markdown style or ALL CAPS lines)
        headers = re.findall(r'(?:^|\n)\s*(?:#{1,4}\s|[A-Z][A-Z\s]{5,}:|\*\*[^*]+\*\*)', response)
        header_score = min(len(headers) / 4.0, 1.0)
        
        # Sub-items (indented or nested lists)
        sub_items = re.findall(r'(?:^|\n)\s{2,}(?:\*|-|•|\d+[\.\)])', response)
        nesting_score = min(len(sub_items) / 3.0, 1.0)
        
        # Code blocks or formatted content
        code_blocks = len(re.findall(r'```|`[^`]+`|\\\[|\\\(', response))
        code_score = min(code_blocks / 4.0, 1.0)
        
        structural_depth = (list_depth * 0.3 + header_score * 0.3 + 
                           nesting_score * 0.2 + code_score * 0.2)
        
        # ============================================================
        # FEATURE 6: Response Length Adequacy (relative to query complexity)
        # ============================================================
        query_complexity = len(query_content) + len(question_fragments)
        resp_length = len(resp_tokens)
        
        # Longer queries with more sub-parts need longer responses
        expected_min_length = max(50, query_complexity * 8)
        length_adequacy = min(resp_length / expected_min_length, 1.5) / 1.5
        
        # Also reward absolute length up to a point
        abs_length_score = min(resp_length / 200.0, 1.0)
        
        length_score = 0.5 * length_adequacy + 0.5 * abs_length_score
        
        # ============================================================
        # FEATURE 7: Completeness Indicators vs Incompleteness
        # ============================================================
        # Check for truncation signals
        truncation_penalty = 0.0
        last_50 = response[-50:] if len(response) > 50 else response
        # Ends mid-sentence
        if not re.search(r'[.!?:"\']$', response.rstrip()):
            truncation_penalty += 0.15
        # Ends with incomplete list item
        if re.search(r'(?:\d+\.\s*$|\*\s*$|-\s*$)', last_50):
            truncation_penalty += 0.1
        
        # Conclusion/summary signals
        conclusion_markers = re.findall(
            r'\b(?:in\s+(?:summary|conclusion)|overall|to\s+summarize|finally|in\s+short|key\s+takeaway|remember)\b',
            response.lower()
        )
        conclusion_bonus = min(len(conclusion_markers) * 0.1, 0.2)
        
        # ============================================================
        # FEATURE 8: Explanation Depth — causal/reasoning language
        # ============================================================
        reasoning_markers = re.findall(
            r'\b(?:because|therefore|thus|hence|consequently|as\s+a\s+result|this\s+means|'
            r'the\s+reason|due\s+to|leads?\s+to|causes?|explains?|since|'
            r'in\s+order\s+to|so\s+that|which\s+(?:means|allows|enables|ensures))\b',
            response.lower()
        )
        reasoning_score = min(len(reasoning_markers) / 5.0, 1.0)
        
        # ============================================================
        # FEATURE 9: Vocabulary Sophistication (unique content word ratio)
        # ============================================================
        if resp_content:
            unique_content = len(set(resp_content))
            vocab_sophistication = unique_content / len(resp_content)
            # Also count longer/technical words
            technical_words = [w for w in resp_content if len(w) >= 8]
            technical_ratio = min(len(technical_words) / max(len(resp_content), 1), 0.3) / 0.3
        else:
            vocab_sophistication = 0.0
            technical_ratio = 0.0
        
        # ============================================================
        # FEATURE 10: Aspect Breadth — count distinct "topic clusters"
        # ============================================================
        # Use content trigrams as proxies for distinct aspects
        resp_trigrams = get_ngrams(resp_content, 3)
        unique_trigrams = len(set(resp_trigrams))
        aspect_breadth = min(unique_trigrams / 50.0, 1.0)
        
        # ============================================================
        # COMBINE FEATURES
        # ============================================================
        score = (
            fragment_coverage * 12.0 +       # 0-12: query coverage
            bigram_richness * 10.0 +          # 0-10: information richness
            bigram_diversity * 5.0 +           # 0-5: diversity
            topic_progression_score * 8.0 +    # 0-8: sustained depth
            specificity * 8.0 +                # 0-8: concrete details
            structural_depth * 10.0 +          # 0-10: organization
            length_score * 10.0 +              # 0-10: adequate length
            reasoning_score * 7.0 +            # 0-7: explanation depth
            vocab_sophistication * 5.0 +       # 0-5: vocabulary
            technical_ratio * 5.0 +            # 0-5: technical depth
            aspect_breadth * 8.0 +             # 0-8: breadth
            conclusion_bonus * 10.0 -          # 0-2: conclusion
            truncation_penalty * 15.0          # 0-3.75: truncation penalty
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~ 12+10+5+8+8+10+10+7+5+5+8+2 = 90
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)
        
    except Exception:
        try:
            return max(0.0, min(len(str(response)) / 20.0, 50.0))
        except Exception:
            return 0.0