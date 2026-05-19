def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    Focuses on:
    - Presence of logical structure and step-by-step breakdown
    - Intermediate conclusions and explanations of 'why'
    - Transparency of reasoning process
    - Avoiding opaque jumps to conclusions
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        query_stripped = query.strip()
        
        score = 0.0
        
        # === 1. SENTENCE-LEVEL ANALYSIS ===
        # Split response into sentences
        import re
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        # Reward having multiple sentences (shows elaboration, not just a one-liner)
        if num_sentences == 0:
            return 0.5
        elif num_sentences == 1:
            score += 2.0
        elif num_sentences == 2:
            score += 5.0
        elif num_sentences <= 5:
            score += 8.0
        elif num_sentences <= 10:
            score += 10.0
        else:
            score += 9.0  # Very long might be rambling
        
        # === 2. REASONING INDICATORS (causal/logical connectors) ===
        # Words and phrases that indicate reasoning transparency
        causal_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bdue to\b', r'\bowing to\b', r'\bfor this reason\b',
            r'\bthis means\b', r'\bthis suggests\b', r'\bthis implies\b',
            r'\bin order to\b', r'\bso that\b', r'\bwhich means\b',
            r'\bwhich leads to\b', r'\bwhich causes\b',
            r'\bthe reason\b', r'\bis because\b',
        ]
        
        response_lower = response_stripped.lower()
        
        causal_count = 0
        for pattern in causal_markers:
            causal_count += len(re.findall(pattern, response_lower))
        
        # Score causal markers (up to 15 points)
        score += min(causal_count * 3.0, 15.0)
        
        # === 3. STEP-WISE / STRUCTURAL MARKERS ===
        step_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfourth\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bafter that\b',
            r'\bstep \d+\b', r'\b\d+\.\s', r'\b\d+\)\s',
            r'\binitially\b', r'\bsubsequently\b', r'\bafterward\b',
            r'\bin the first place\b', r'\bin addition\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bhowever\b',
            r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b',
        ]
        
        step_count = 0
        for pattern in step_markers:
            step_count += len(re.findall(pattern, response_lower))
        
        # Score step markers (up to 15 points)
        score += min(step_count * 2.5, 15.0)
        
        # === 4. EXPLANATORY DEPTH MARKERS ===
        explanatory_markers = [
            r'\bthis is because\b', r'\bthe reason is\b',
            r'\bin other words\b', r'\bthat is to say\b',
            r'\bput simply\b', r'\bto clarify\b', r'\bto explain\b',
            r'\bwhat this means\b', r'\bessentially\b',
            r'\bimportantly\b', r'\bsignificantly\b',
            r'\bit is important to\b', r'\bit is worth noting\b',
            r'\bnote that\b', r'\bkeep in mind\b',
            r'\blet\'s\b', r'\bconsider\b', r'\bthink of\b',
            r'\bimagine\b', r'\bsuppose\b',
        ]
        
        explanatory_count = 0
        for pattern in explanatory_markers:
            explanatory_count += len(re.findall(pattern, response_lower))
        
        score += min(explanatory_count * 3.0, 12.0)
        
        # === 5. CONTRASTIVE/COMPARATIVE REASONING ===
        contrast_markers = [
            r'\bwhile\b', r'\bwhereas\b', r'\balthough\b',
            r'\bbut\b', r'\byet\b', r'\bdespite\b',
            r'\bon one hand\b', r'\bon the other\b',
            r'\bcompared to\b', r'\bin comparison\b',
            r'\bdiffers?\b', r'\bsimilar\b', r'\bunlike\b',
            r'\bboth\b', r'\neither\b',
        ]
        
        contrast_count = 0
        for pattern in contrast_markers:
            contrast_count += len(re.findall(pattern, response_lower))
        
        score += min(contrast_count * 1.5, 8.0)
        
        # === 6. RESPONSE LENGTH RELATIVE TO QUERY COMPLEXITY ===
        # Longer queries often need more elaborate responses
        query_words = len(query_stripped.split())
        response_words = len(response_stripped.split())
        
        # Base content score from word count
        if response_words < 5:
            score += 0.0
        elif response_words < 15:
            score += 3.0
        elif response_words < 30:
            score += 6.0
        elif response_words < 60:
            score += 9.0
        elif response_words < 120:
            score += 10.0
        else:
            score += 8.0  # Diminishing returns for very long
        
        # === 7. CLAUSE COMPLEXITY (commas indicating subordinate clauses) ===
        comma_count = response_stripped.count(',')
        # More clauses often means more nuanced reasoning
        avg_commas_per_sentence = comma_count / max(num_sentences, 1)
        if avg_commas_per_sentence > 0.3 and avg_commas_per_sentence < 4.0:
            score += min(avg_commas_per_sentence * 3.0, 8.0)
        
        # === 8. PENALIZE REPETITION (sign of low quality / no real reasoning) ===
        words = response_lower.split()
        if len(words) > 3:
            # Check for repeated bigrams
            bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
            from collections import Counter
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            if total_bigrams > 0:
                most_common_freq = bigram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_freq / total_bigrams
                if repetition_ratio > 0.15 and total_bigrams > 5:
                    score -= 10.0 * repetition_ratio
            
            # Check for repeated words
            word_counts = Counter(words)
            if len(words) > 10:
                # Exclude common stop words from repetition check
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or',
                             'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                             'it', 'its', 'that', 'this', 'their', 'they', 'be', 'as',
                             'has', 'have', 'had', 'not', 'but', 'if', 'than', 'more'}
                content_words = [w for w in words if w not in stop_words and len(w) > 2]
                if content_words:
                    content_counts = Counter(content_words)
                    max_content_freq = content_counts.most_common(1)[0][1] if content_counts else 0
                    if max_content_freq > 4 and len(content_words) > 5:
                        score -= min(max_content_freq * 1.5, 10.0)
        
        # === 9. PENALIZE MERE ECHO OF QUERY ===
        # If response just paraphrases the query with minimal addition
        query_words_set = set(query_stripped.lower().split())
        response_words_set = set(response_lower.split())
        
        if len(response_words_set) > 0 and len(query_words_set) > 0:
            overlap = len(query_words_set & response_words_set)
            # High overlap with short response = just echoing
            if response_words < 15 and overlap / max(len(response_words_set), 1) > 0.7:
                score -= 5.0
        
        # === 10. STRUCTURED FORMATTING (lists, numbered items, headers) ===
        has_bullet_points = bool(re.search(r'^\s*[-•*]\s', response_stripped, re.MULTILINE))
        has_numbered_list = bool(re.search(r'^\s*\d+[.)]\s', response_stripped, re.MULTILINE))
        has_headers = bool(re.search(r'^#+\s', response_stripped, re.MULTILINE))
        
        if has_bullet_points:
            score += 3.0
        if has_numbered_list:
            score += 4.0
        if has_headers:
            score += 2.0
        
        # === 11. UNIQUE VOCABULARY RICHNESS ===
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            # Higher unique ratio = more diverse vocabulary = likely more substantive
            score += unique_ratio * 5.0
        
        # === 12. PRESENCE OF SPECIFIC/CONCRETE DETAILS ===
        # Numbers, percentages, proper nouns, etc. indicate specificity
        has_numbers = bool(re.search(r'\b\d+\b', response_stripped))
        has_quotes = bool(re.search(r'["\'].*?["\']', response_stripped))
        
        if has_numbers:
            score += 1.5
        if has_quotes:
            score += 1.0
        
        # === 13. MULTI-ASPECT COVERAGE ===
        # Count distinct topic shifts (approximated by paragraph/sentence transitions)
        paragraphs = [p.strip() for p in response_stripped.split('\n') if p.strip()]
        if len(paragraphs) > 1:
            score += min(len(paragraphs) * 1.5, 6.0)
        
        # === 14. RESPONSE COMPLETENESS ===
        # Penalize responses that appear truncated
        if response_stripped[-1] not in '.!?"\')':
            # Might be truncated
            if len(response_stripped) > 50:
                score -= 3.0
        
        # === 15. EMPTY/NOINPUT CHECK ===
        if response_stripped.lower() in ['<noinput>', 'noinput', 'n/a', 'none', '']:
            return 0.5
        
        # Normalize score to 0-100 range
        # Theoretical max is roughly: 10 + 15 + 15 + 12 + 8 + 10 + 8 + 5 + 4 + 2 + 3 + 6 + 1.5 + 1 = ~100
        # But with penalties it can go lower
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)
    
    except Exception as e:
        # Fallback: return a basic score based on response length
        try:
            if response and len(response.strip()) > 0:
                return min(len(response.strip().split()) * 0.2, 20.0)
            return 0.0
        except:
            return 0.0