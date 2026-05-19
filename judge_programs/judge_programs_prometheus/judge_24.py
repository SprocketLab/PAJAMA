def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a structural
    and semantic analysis approach based on:
    - Query decomposition (identifying sub-questions/aspects)
    - Response structural depth (nested information, examples, explanations)
    - Information density and specificity
    - Acknowledgment patterns and empathy markers
    - Action/advice density
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not query:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        response_words = response.split()
        query_words = query.split()
        
        if len(response_words) < 3:
            return 0.5
        
        score = 0.0
        
        # === 1. Query Aspect Extraction and Coverage (0-20 points) ===
        # Extract key noun phrases / content words from query as "aspects"
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'that', 'this', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'they',
            'their', 'them', 'he', 'she', 'him', 'her', 'who', 'whom', 'which',
            'what', 'need', 'want', 'person', 'way', 'thing', 'things', 'also',
            'like', 'get', 'make', 'know', 'think', 'see', 'come', 'go', 'take',
        }
        
        query_content_words = set()
        for w in query_words:
            cleaned = re.sub(r'[^a-z]', '', w.lower())
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                query_content_words.add(cleaned)
        
        response_lower = response.lower()
        
        if query_content_words:
            covered = sum(1 for w in query_content_words if w in response_lower)
            coverage_ratio = covered / len(query_content_words)
            score += coverage_ratio * 15
        
        # Check for semantic coverage via bigrams from query
        query_lower = query.lower()
        query_bigrams = set()
        q_tokens = re.findall(r'[a-z]+', query_lower)
        for i in range(len(q_tokens) - 1):
            if q_tokens[i] not in stopwords or q_tokens[i+1] not in stopwords:
                query_bigrams.add((q_tokens[i], q_tokens[i+1]))
        
        if query_bigrams:
            r_tokens = re.findall(r'[a-z]+', response_lower)
            r_bigrams = set()
            for i in range(len(r_tokens) - 1):
                r_bigrams.add((r_tokens[i], r_tokens[i+1]))
            bigram_overlap = len(query_bigrams & r_bigrams) / len(query_bigrams)
            score += bigram_overlap * 5
        
        # === 2. Response Structural Depth (0-15 points) ===
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Reward having multiple substantive sentences (diminishing returns)
        sentence_score = min(num_sentences / 6.0, 1.0) * 8
        score += sentence_score
        
        # Check for structured elements: numbered items, colons (definitions), 
        # parenthetical clarifications
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response))
        colon_definitions = len(re.findall(r':\s', response))
        parentheticals = len(re.findall(r'\([^)]+\)', response))
        
        structure_score = min((numbered_items * 1.5 + colon_definitions * 0.5 + parentheticals * 0.8) / 5.0, 1.0) * 7
        score += structure_score
        
        # === 3. Information Density & Specificity (0-20 points) ===
        # Measure proportion of "specific" words (longer, less common words)
        r_words_clean = [re.sub(r'[^a-z]', '', w.lower()) for w in response_words]
        r_words_clean = [w for w in r_words_clean if w]
        
        if r_words_clean:
            # Specificity: words with 7+ characters that aren't stopwords
            specific_words = [w for w in r_words_clean if len(w) >= 7 and w not in stopwords]
            specificity_ratio = len(specific_words) / len(r_words_clean)
            score += min(specificity_ratio / 0.20, 1.0) * 10
            
            # Unique content ratio (not just stopwords repeated)
            content_words = [w for w in r_words_clean if w not in stopwords]
            if content_words:
                unique_content = len(set(content_words))
                content_diversity = unique_content / len(content_words)
                score += min(content_diversity / 0.65, 1.0) * 5
            
            # Check for concrete/actionable language
            action_patterns = [
                r'\b(?:first|second|third|next|then|finally|start|begin|step)\b',
                r'\b(?:try|consider|remember|ensure|make sure|important|crucial)\b',
                r'\b(?:for example|for instance|such as|e\.g\.|i\.e\.)\b',
                r'\b(?:because|since|therefore|thus|consequently|as a result)\b',
                r'\b(?:however|although|nevertheless|despite|while|whereas)\b',
            ]
            
            action_count = 0
            for pattern in action_patterns:
                action_count += len(re.findall(pattern, response_lower))
            
            score += min(action_count / 5.0, 1.0) * 5
        
        # === 4. Empathy & Acknowledgment Patterns (0-10 points) ===
        # Many queries in the dataset involve emotional/support contexts
        emotional_query_words = {'feeling', 'feel', 'frustrated', 'stress', 'sad', 
                                 'lonely', 'heartbroken', 'devastated', 'struggling',
                                 'difficult', 'tough', 'worried', 'anxious', 'upset',
                                 'exhausted', 'tired', 'down', 'breakup', 'passed',
                                 'regret', 'angry', 'fear', 'pain', 'hurt'}
        
        query_is_emotional = any(w in query_lower for w in emotional_query_words)
        
        if query_is_emotional:
            empathy_markers = [
                r'\b(?:understand|understandable|sorry|hear|feeling|feel)\b',
                r'\b(?:natural|normal|okay|valid|completely|absolutely|genuinely)\b',
                r'\b(?:support|care|help|comfort|listen)\b',
                r'\b(?:it\'s okay|it\'s fine|that\'s okay|perfectly)\b',
            ]
            empathy_count = 0
            for pattern in empathy_markers:
                empathy_count += len(re.findall(pattern, response_lower))
            
            score += min(empathy_count / 4.0, 1.0) * 10
        else:
            # For non-emotional queries, reward thoroughness of explanation
            explanation_markers = [
                r'\b(?:means|meaning|refers|defined|definition|concept)\b',
                r'\b(?:works|function|process|method|approach|technique)\b',
                r'\b(?:advantage|benefit|drawback|limitation|difference)\b',
                r'\b(?:essentially|basically|specifically|particularly)\b',
            ]
            explanation_count = 0
            for pattern in explanation_markers:
                explanation_count += len(re.findall(pattern, response_lower))
            score += min(explanation_count / 3.0, 1.0) * 10
        
        # === 5. Appropriate Response Length (0-10 points) ===
        # Not too short (incomplete) and not excessively padded
        word_count = len(response_words)
        
        # Ideal range depends on query complexity
        query_complexity = len(query_content_words)
        
        if query_complexity > 8:
            ideal_min = 80
            ideal_max = 400
        elif query_complexity > 4:
            ideal_min = 50
            ideal_max = 350
        else:
            ideal_min = 30
            ideal_max = 250
        
        if word_count < ideal_min:
            length_score = (word_count / ideal_min) * 8
        elif word_count <= ideal_max:
            length_score = 10
        else:
            length_score = max(5, 10 - (word_count - ideal_max) / 100)
        
        score += length_score
        
        # === 6. Coherence via Discourse Connectors (0-10 points) ===
        discourse_connectors = [
            r'\b(?:moreover|furthermore|additionally|in addition)\b',
            r'\b(?:however|on the other hand|conversely|nevertheless)\b',
            r'\b(?:therefore|consequently|as a result|hence|thus)\b',
            r'\b(?:for example|for instance|specifically|in particular)\b',
            r'\b(?:first|second|third|finally|lastly|to begin|to start)\b',
            r'\b(?:similarly|likewise|in contrast|compared to)\b',
            r'\b(?:in summary|overall|in conclusion|to summarize)\b',
            r'\b(?:also|besides|meanwhile|instead|rather)\b',
        ]
        
        connector_types_found = 0
        for pattern in discourse_connectors:
            if re.search(pattern, response_lower):
                connector_types_found += 1
        
        score += min(connector_types_found / 4.0, 1.0) * 10
        
        # === 7. Penalty for Dismissiveness or Negativity (0 to -10 points) ===
        dismissive_patterns = [
            r'\bjust\s+(?:do|get|try|read|go|buy|find)\b',
            r'\b(?:you should be able|it\'s not that|don\'t worry about)\b',
            r'\b(?:simply|obviously|clearly you)\b',
            r'\bmaybe you\'re (?:just|not)\b',
            r'\b(?:can\'t|won\'t|might not|probably won\'t)\b',
        ]
        
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        penalty = min(dismissive_count * 2, 10)
        score -= penalty
        
        # === 8. Addressing the "how" - Practical/Actionable Content (0-10 points) ===
        # Check if response provides concrete steps or suggestions
        imperative_verbs = re.findall(
            r'(?:^|\.\s+|\n\s*)(?:[A-Z][a-z]+|[a-z]+)\s+(?:your|the|a|some|this|these)',
            response
        )
        
        suggestion_patterns = [
            r'\b(?:you (?:can|could|might|should|may))\b',
            r'\b(?:try to|consider|think about|look into|explore)\b',
            r'\b(?:one way|another way|option|approach|strategy)\b',
            r'\b(?:here are|here\'s|following|these are)\b',
            r'\b(?:recommend|suggest|advise|encourage)\b',
        ]
        
        suggestion_count = 0
        for pattern in suggestion_patterns:
            suggestion_count += len(re.findall(pattern, response_lower))
        
        score += min(suggestion_count / 3.0, 1.0) * 10
        
        # Normalize to 1-5 scale
        # Max theoretical raw score ~100, but typical range 20-85
        normalized = 1.0 + (score / 85.0) * 4.0
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
        
    except Exception:
        return 2.5