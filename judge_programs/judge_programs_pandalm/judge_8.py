def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using a TF-IDF inspired approach
    with query decomposition into intent components and coverage analysis.
    
    This variant uses:
    - Query intent decomposition (question words, action verbs, topic nouns)
    - Weighted term importance based on IDF-like rarity scoring
    - Response structure quality (sentence coherence, information density)
    - Query-response semantic flow analysis using term chains
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if not response or len(response) < 2:
            return 0.0
        
        # --- Tokenization helpers ---
        def tokenize(text):
            return re.findall(r'[a-zA-Z]+', text.lower())
        
        def get_sentences(text):
            sents = re.split(r'[.!?]+', text)
            return [s.strip() for s in sents if s.strip() and len(s.strip()) > 3]
        
        # Common stop words (high frequency, low information)
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'its', 'it', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who',
            'whom', 'also', 'well', 'back', 'even', 'still', 'new', 'get', 'got',
            'like', 'make', 'made', 'go', 'going', 'come', 'take', 'know', 'see',
            'think', 'say', 'said', 'one', 'two', 'first', 'way', 'thing', 'much',
            'many', 'any', 'give', 'given', 'us', 'am', 'an'
        }
        
        # Action/intent verbs that indicate what the query wants
        ACTION_VERBS = {
            'explain', 'describe', 'compare', 'contrast', 'list', 'provide',
            'generate', 'create', 'write', 'rewrite', 'summarize', 'analyze',
            'evaluate', 'discuss', 'define', 'identify', 'suggest', 'recommend',
            'show', 'demonstrate', 'illustrate', 'outline', 'classify', 'categorize',
            'translate', 'convert', 'calculate', 'determine', 'find', 'name',
            'give', 'tell', 'crop', 'reduce', 'add', 'come', 'design', 'develop',
            'build', 'implement', 'solve', 'fix', 'improve', 'modify', 'edit'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not query_tokens or not response_tokens:
            return 1.0
        
        # --- 1. Query Decomposition ---
        # Separate query into: action words, content/topic words, question indicators
        query_content_words = [w for w in query_tokens if w not in STOP_WORDS]
        query_action_words = [w for w in query_content_words if w in ACTION_VERBS]
        query_topic_words = [w for w in query_content_words if w not in ACTION_VERBS]
        
        response_content_words = [w for w in response_tokens if w not in STOP_WORDS]
        
        # --- 2. IDF-like term importance ---
        # Words that appear less frequently in English tend to be more topically important
        # Approximate by: shorter common words get lower weight, longer/rarer words get higher
        # Also: words appearing in query but rare overall get boosted
        
        # Build a pseudo-document frequency from combining query + response
        all_content = query_content_words + response_content_words
        term_freq = Counter(all_content)
        total_terms = len(all_content) if all_content else 1
        
        def term_importance(word):
            """Higher score for more important/specific terms."""
            base = 1.0
            # Length bonus: longer words tend to be more specific
            if len(word) >= 8:
                base += 1.5
            elif len(word) >= 6:
                base += 1.0
            elif len(word) >= 4:
                base += 0.5
            # Rarity bonus: terms that appear fewer times are more discriminative
            freq = term_freq.get(word, 0)
            if freq <= 1:
                base += 1.0
            elif freq <= 3:
                base += 0.5
            return base
        
        # --- 3. Weighted Topic Coverage ---
        response_word_set = set(response_tokens)
        response_content_set = set(response_content_words)
        
        if query_topic_words:
            topic_weights = {}
            for w in query_topic_words:
                topic_weights[w] = term_importance(w)
            
            total_weight = sum(topic_weights.values())
            covered_weight = sum(topic_weights[w] for w in topic_weights if w in response_word_set)
            
            # Also check for stemmed/partial matches
            for w in topic_weights:
                if w not in response_word_set and len(w) >= 5:
                    stem = w[:max(4, len(w)-2)]
                    for rw in response_word_set:
                        if rw.startswith(stem) or stem in rw:
                            covered_weight += topic_weights[w] * 0.6
                            break
            
            topic_coverage = min(1.0, covered_weight / total_weight) if total_weight > 0 else 0.5
        else:
            topic_coverage = 0.5
        
        # --- 4. Query-Response Term Chain Analysis ---
        # Check if the response builds a coherent chain of concepts from the query
        response_sentences = get_sentences(response)
        
        chain_score = 0.0
        if response_sentences and query_topic_words:
            query_topic_set = set(query_topic_words)
            sentences_with_query_terms = 0
            for sent in response_sentences:
                sent_tokens = set(tokenize(sent))
                if sent_tokens & query_topic_set:
                    sentences_with_query_terms += 1
            
            # Good responses reference query topics across multiple sentences
            if len(response_sentences) > 0:
                chain_score = min(1.0, sentences_with_query_terms / max(1, len(response_sentences)))
        else:
            chain_score = 0.3
        
        # --- 5. Response Substantiveness ---
        # Measure information density: unique content words / total words
        response_unique_content = set(response_content_words)
        
        if len(response_tokens) > 0:
            info_density = len(response_unique_content) / len(response_tokens)
        else:
            info_density = 0.0
        
        # Penalize very repetitive responses
        if response_content_words:
            content_counter = Counter(response_content_words)
            most_common_freq = content_counter.most_common(1)[0][1]
            repetition_ratio = most_common_freq / len(response_content_words)
            repetition_penalty = max(0.0, 1.0 - max(0.0, repetition_ratio - 0.3) * 2.0)
        else:
            repetition_penalty = 0.5
        
        # --- 6. Length Appropriateness ---
        query_len = len(query_tokens)
        resp_len = len(response_tokens)
        
        # Responses should generally be longer than queries for explanatory tasks
        if resp_len == 0:
            length_score = 0.0
        elif resp_len < 5:
            length_score = 0.2
        elif resp_len < 10:
            length_score = 0.4
        elif resp_len < 20:
            length_score = 0.6
        elif resp_len < 80:
            length_score = 0.9
        elif resp_len < 150:
            length_score = 1.0
        else:
            # Very long responses might have padding
            length_score = max(0.6, 1.0 - (resp_len - 150) / 500.0)
        
        # --- 7. Direct Address Detection ---
        # Check if the response directly addresses the query structure
        # e.g., "What is X?" -> response should mention X and define it
        # "Compare X and Y" -> response should mention both X and Y
        
        direct_address_score = 0.0
        query_lower = query.lower()
        
        # Detect comparison queries
        is_comparison = any(w in query_lower for w in ['compare', 'contrast', 'difference', 'versus', 'vs'])
        if is_comparison:
            # Extract potential comparison subjects (nouns after compare/contrast/between)
            # Check if response mentions both sides
            # Simple heuristic: check for "both", "while", "whereas", "but", "however"
            comparison_markers = ['both', 'while', 'whereas', 'but', 'however', 'differ', 'similar', 'unlike']
            resp_lower = response.lower()
            markers_found = sum(1 for m in comparison_markers if m in resp_lower)
            direct_address_score = min(1.0, markers_found / 2.0)
        
        # Detect definition/explanation queries
        is_explanation = any(w in query_lower for w in ['explain', 'what', 'describe', 'define', 'mean'])
        if is_explanation:
            explanation_markers = ['means', 'refers', 'is a', 'is the', 'describes', 'suggests', 'implies', 'indicates']
            resp_lower = response.lower()
            markers_found = sum(1 for m in explanation_markers if m in resp_lower)
            direct_address_score = max(direct_address_score, min(1.0, markers_found / 1.5))
        
        # Detect listing queries
        is_listing = any(w in query_lower for w in ['list', 'provide', 'examples', 'name', 'give'])
        if is_listing:
            # Count distinct items (commas, bullet points, numbered items)
            comma_items = response.count(',')
            bullet_items = len(re.findall(r'(?:^|\n)\s*[-•*]\s', response))
            numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s', response))
            list_items = max(comma_items, bullet_items + numbered_items)
            direct_address_score = max(direct_address_score, min(1.0, list_items / 3.0))
        
        # Detect action/creative queries
        is_creative = any(w in query_lower for w in ['write', 'create', 'generate', 'come up', 'design', 'rewrite'])
        if is_creative:
            # Creative responses should be substantial and not just echo the query
            if resp_len > 15 and info_density > 0.2:
                direct_address_score = max(direct_address_score, 0.7)
        
        if not (is_comparison or is_explanation or is_listing or is_creative):
            direct_address_score = 0.5  # neutral for unclassified queries
        
        # --- 8. Echo/Parroting Detection ---
        # Penalize responses that just repeat the query
        if query_content_words and response_content_words:
            query_content_set = set(query_content_words)
            response_only = [w for w in response_content_words if w not in query_content_set]
            novelty_ratio = len(response_only) / max(1, len(response_content_words))
        else:
            novelty_ratio = 0.5
        
        # Very low novelty = parroting; very high = might be off-topic
        if novelty_ratio < 0.1:
            echo_penalty = 0.3
        elif novelty_ratio < 0.2:
            echo_penalty = 0.6
        elif novelty_ratio > 0.95:
            echo_penalty = 0.7  # might be off-topic
        else:
            echo_penalty = 1.0
        
        # --- 9. Semantic Field Expansion ---
        # Good responses should introduce related terms that expand on query topics
        # We approximate this by checking if response has content words that are
        # thematically connected (share character subsequences with query terms)
        
        expansion_score = 0.0
        if query_topic_words and response_content_words:
            response_new_words = [w for w in set(response_content_words) if w not in set(query_tokens)]
            related_expansions = 0
            for rw in response_new_words:
                for qw in query_topic_words:
                    # Check for morphological relatedness
                    if len(qw) >= 4 and len(rw) >= 4:
                        # Shared prefix of length >= 4
                        shared_prefix_len = 0
                        for i in range(min(len(qw), len(rw))):
                            if qw[i] == rw[i]:
                                shared_prefix_len += 1
                            else:
                                break
                        if shared_prefix_len >= 4:
                            related_expansions += 1
                            break
                        # Shared character trigrams
                        qw_trigrams = {qw[i:i+3] for i in range(len(qw)-2)}
                        rw_trigrams = {rw[i:i+3] for i in range(len(rw)-2)}
                        if qw_trigrams and rw_trigrams:
                            trigram_overlap = len(qw_trigrams & rw_trigrams) / max(1, min(len(qw_trigrams), len(rw_trigrams)))
                            if trigram_overlap >= 0.5:
                                related_expansions += 0.5
                                break
            
            if response_new_words:
                expansion_score = min(1.0, related_expansions / max(1, len(query_topic_words)))
            else:
                expansion_score = 0.2
        else:
            expansion_score = 0.3
        
        # --- 10. Sentence Completeness ---
        # Check if response has well-formed sentences (not cut off)
        last_char = response.strip()[-1] if response.strip() else ''
        if last_char in '.!?':
            completeness = 1.0
        elif last_char in ',;:':
            completeness = 0.5
        else:
            completeness = 0.3
        
        # --- Combine all signals with weights ---
        # topic_coverage: 0-1 (how much of query topic is covered)
        # chain_score: 0-1 (how well response threads query topics)
        # info_density: 0-1 (information richness)
        # repetition_penalty: 0-1 (penalize repetition)
        # length_score: 0-1 (appropriate length)
        # direct_address_score: 0-1 (addresses query type)
        # echo_penalty: 0-1 (not just parroting)
        # expansion_score: 0-1 (semantic expansion)
        # completeness: 0-1 (well-formed)
        # novelty_ratio: 0-1 (new information)
        
        weights = {
            'topic_coverage': 3.0,
            'chain_score': 1.5,
            'info_density': 1.0,
            'repetition_penalty': 1.5,
            'length_score': 1.5,
            'direct_address': 2.0,
            'echo_penalty': 1.0,
            'expansion': 0.8,
            'completeness': 0.7,
        }
        
        raw_score = (
            weights['topic_coverage'] * topic_coverage +
            weights['chain_score'] * chain_score +
            weights['info_density'] * min(1.0, info_density * 2.5) +  # scale up since typical is 0.3-0.5
            weights['repetition_penalty'] * repetition_penalty +
            weights['length_score'] * length_score +
            weights['direct_address'] * direct_address_score +
            weights['echo_penalty'] * echo_penalty +
            weights['expansion'] * expansion_score +
            weights['completeness'] * completeness
        )
        
        total_weight = sum(weights.values())
        normalized_score = raw_score / total_weight  # 0 to 1
        
        # Scale to 0-100
        final_score = normalized_score * 100.0
        
        # Clamp
        final_score = max(0.0, min(100.0, final_score))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Minimal fallback
            if response and len(response.strip()) > 10:
                return 30.0
            return 5.0
        except Exception:
            return 5.0