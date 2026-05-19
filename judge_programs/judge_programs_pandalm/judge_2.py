def judging_function(query, response):
    """
    Evaluates response relevance using TF-based semantic matching with
    query decomposition, intent coverage analysis, and information density scoring.
    
    This variant focuses on:
    1. Query intent decomposition into key phrases/bigrams
    2. TF-weighted term importance matching
    3. Response information density and completeness
    4. Penalization for repetition, emptiness, off-topic content
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
        
        # --- Utility functions ---
        
        STOPWORDS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'although', 'though',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'up', 'about', 'also',
            'down', 'get', 'got', 'let', 'make', 'made', 'put', 'say', 'said',
            'see', 'go', 'went', 'gone', 'come', 'came', 'take', 'took', 'taken',
            'give', 'gave', 'given', 'tell', 'told', 'think', 'thought', 'know',
            'knew', 'known', 'want', 'like', 'well', 'back', 'much', 'way',
            'even', 'new', 'one', 'two', 'first', 'also', 'now', 'look', 'people',
            'any', 'still', 'every', 'thing', 'things'
        }
        
        # Instructional/meta words that appear in queries but shouldn't be matched literally
        INSTRUCTION_WORDS = {
            'describe', 'explain', 'provide', 'generate', 'create', 'write',
            'list', 'compare', 'contrast', 'rewrite', 'following', 'given',
            'input', 'example', 'examples', 'please', 'using', 'use',
            'answer', 'question', 'respond', 'discuss', 'analyze', 'evaluate',
            'consider', 'determine', 'identify', 'suggest', 'recommend',
            'summarize', 'outline', 'define', 'illustrate', 'demonstrate',
            'show', 'tell', 'find', 'name', 'state', 'mention', 'note',
            'come', 'crop', 'reduce', 'add'
        }
        
        def tokenize(text):
            """Tokenize text into lowercase words."""
            return re.findall(r'[a-z]+(?:\'[a-z]+)?', text.lower())
        
        def get_content_words(tokens):
            """Filter to content words only."""
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        def get_query_key_terms(tokens):
            """Get key terms from query, excluding instruction words."""
            return [t for t in tokens if t not in STOPWORDS and t not in INSTRUCTION_WORDS and len(t) > 1]
        
        def get_bigrams(tokens):
            """Generate bigrams from token list."""
            return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        
        def get_trigrams(tokens):
            """Generate trigrams from token list."""
            return [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
        
        # --- Tokenization ---
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if not response_tokens:
            return 0.0
        
        query_content = get_content_words(query_tokens)
        response_content = get_content_words(response_tokens)
        query_key_terms = get_query_key_terms(query_tokens)
        
        # --- Score Component 1: Key Term Coverage (0-25) ---
        # How many of the query's key terms appear in the response?
        
        if query_key_terms:
            response_content_set = set(response_content)
            query_key_set = set(query_key_terms)
            
            # Weighted by term frequency in query (repeated terms are more important)
            query_term_freq = Counter(query_key_terms)
            total_weight = sum(query_term_freq.values())
            covered_weight = sum(freq for term, freq in query_term_freq.items() 
                               if term in response_content_set)
            
            term_coverage = covered_weight / total_weight if total_weight > 0 else 0
        else:
            # If no key terms extractable, be lenient
            term_coverage = 0.5
        
        score_term_coverage = term_coverage * 25.0
        
        # --- Score Component 2: Bigram Topic Alignment (0-15) ---
        # Bigram overlap captures phrase-level topic alignment
        
        query_content_bigrams = get_bigrams(query_content)
        response_content_bigrams = get_bigrams(response_content)
        
        if query_content_bigrams and response_content_bigrams:
            q_bigram_set = set(query_content_bigrams)
            r_bigram_set = set(response_content_bigrams)
            bigram_overlap = len(q_bigram_set & r_bigram_set)
            bigram_score = min(bigram_overlap / max(len(q_bigram_set), 1), 1.0)
        else:
            bigram_score = 0.0
        
        # Also check all-token bigrams (includes function words for phrase matching)
        query_all_bigrams = get_bigrams(query_tokens)
        response_all_bigrams = get_bigrams(response_tokens)
        
        if query_all_bigrams and response_all_bigrams:
            qa_set = set(query_all_bigrams)
            ra_set = set(response_all_bigrams)
            all_bigram_overlap = len(qa_set & ra_set)
            all_bigram_score = min(all_bigram_overlap / max(len(qa_set), 1), 1.0)
        else:
            all_bigram_score = 0.0
        
        score_bigram = max(bigram_score, all_bigram_score) * 15.0
        
        # --- Score Component 3: Response Information Density (0-20) ---
        # Measures how much unique, substantive content the response provides
        
        unique_content_words = set(response_content)
        total_response_words = len(response_tokens)
        
        if total_response_words > 0:
            # Unique content ratio (penalizes repetition)
            unique_ratio = len(unique_content_words) / total_response_words
            
            # Vocabulary richness - number of unique content words
            vocab_richness = min(len(unique_content_words) / 30.0, 1.0)
            
            # Length adequacy - responses should be substantive but not just padding
            length_score = min(total_response_words / 25.0, 1.0)
            
            info_density = (unique_ratio * 0.3 + vocab_richness * 0.4 + length_score * 0.3)
        else:
            info_density = 0.0
        
        score_info_density = info_density * 20.0
        
        # --- Score Component 4: Repetition Penalty (0 to -15) ---
        # Heavily penalize responses with excessive repetition
        
        if total_response_words > 3:
            word_freq = Counter(response_tokens)
            max_freq = max(word_freq.values())
            # Check for pathological repetition
            most_common_content = Counter(response_content).most_common(1)
            if most_common_content:
                top_content_word, top_count = most_common_content[0]
                content_rep_ratio = top_count / max(len(response_content), 1)
            else:
                content_rep_ratio = 0
            
            # Check for repeated phrases (3+ word sequences)
            trigrams = get_trigrams(response_tokens)
            if trigrams:
                trigram_freq = Counter(trigrams)
                max_trigram_freq = max(trigram_freq.values())
                trigram_rep = max_trigram_freq / max(len(trigrams), 1)
            else:
                trigram_rep = 0
            
            # Sentence-level repetition
            sentences = re.split(r'[.!?]+', response.strip())
            sentences = [s.strip().lower() for s in sentences if s.strip()]
            if len(sentences) > 1:
                unique_sentences = set(sentences)
                sentence_rep = 1.0 - (len(unique_sentences) / len(sentences))
            else:
                sentence_rep = 0
            
            rep_penalty = max(content_rep_ratio - 0.3, 0) * 10 + \
                         max(trigram_rep - 0.15, 0) * 15 + \
                         sentence_rep * 5
            rep_penalty = min(rep_penalty, 15.0)
        else:
            rep_penalty = 0
        
        score_repetition = -rep_penalty
        
        # --- Score Component 5: Query Intent Addressing (0-20) ---
        # Does the response actually address what the query is asking?
        
        # Detect query type
        query_lower = query.lower()
        
        intent_score = 0.0
        
        # Check if response echoes/references key query concepts
        # Use character-level n-gram overlap for fuzzy matching
        def char_ngrams(text, n=3):
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        if len(query) > 5 and len(response) > 5:
            q_chars = char_ngrams(query, 4)
            r_chars = char_ngrams(response, 4)
            if q_chars:
                char_overlap = len(q_chars & r_chars) / len(q_chars)
                intent_score += char_overlap * 0.4
        
        # Check for semantic alignment via shared topic words
        # (words that appear in both but aren't stopwords or instruction words)
        shared_topic = set(query_key_terms) & set(response_content)
        if query_key_terms:
            topic_alignment = len(shared_topic) / len(set(query_key_terms))
            intent_score += topic_alignment * 0.6
        else:
            intent_score += 0.3  # lenient when query has few extractable terms
        
        score_intent = intent_score * 20.0
        
        # --- Score Component 6: Completeness & Elaboration (0-15) ---
        # Does the response provide sufficient detail?
        
        # Count distinct ideas (approximated by sentence count with unique content)
        sentences = re.split(r'[.!?]+', response.strip())
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        sentence_count = len(meaningful_sentences)
        
        # Check for elaboration markers
        elaboration_markers = ['for example', 'such as', 'including', 'specifically',
                              'in particular', 'moreover', 'additionally', 'furthermore',
                              'also', 'because', 'since', 'therefore', 'thus',
                              'this means', 'in other words', 'that is']
        
        response_lower = response.lower()
        elaboration_count = sum(1 for marker in elaboration_markers if marker in response_lower)
        
        # Structural indicators
        has_structure = bool(re.search(r'\d+[.)\]]|\n[-*•]|\n\d', response))
        
        completeness = 0.0
        completeness += min(sentence_count / 4.0, 1.0) * 0.5
        completeness += min(elaboration_count / 3.0, 1.0) * 0.3
        completeness += (0.2 if has_structure else 0.0)
        
        score_completeness = completeness * 15.0
        
        # --- Score Component 7: Response is not just echoing the query (0 to -5) ---
        # Penalize if response is essentially just the query repeated
        
        echo_penalty = 0.0
        if len(response_tokens) > 0 and len(query_tokens) > 0:
            # Check if response is a near-copy of query
            q_set = set(query_tokens)
            r_set = set(response_tokens)
            if r_set:
                overlap_with_query = len(r_set & q_set) / len(r_set)
                if overlap_with_query > 0.85 and len(response_tokens) <= len(query_tokens) * 1.2:
                    echo_penalty = (overlap_with_query - 0.85) * 30
        
        score_echo = -min(echo_penalty, 5.0)
        
        # --- Score Component 8: Emptiness / Nonsense Detection (0 to -10) ---
        
        empty_penalty = 0.0
        
        # Check for noinput or placeholder responses
        if '<noinput>' in response_lower or response_lower.strip() in ['', 'n/a', 'none', 'no response']:
            empty_penalty = 10.0
        
        # Check if response is mostly non-alphabetic
        alpha_chars = sum(1 for c in response if c.isalpha())
        if len(response) > 0:
            alpha_ratio = alpha_chars / len(response)
            if alpha_ratio < 0.3:
                empty_penalty = max(empty_penalty, 5.0)
        
        score_empty = -empty_penalty
        
        # --- Aggregate Score ---
        
        total = (score_term_coverage +    # 0-25
                score_bigram +             # 0-15
                score_info_density +       # 0-20
                score_repetition +         # -15 to 0
                score_intent +             # 0-20
                score_completeness +       # 0-15
                score_echo +               # -5 to 0
                score_empty)               # -10 to 0
        
        # Clamp to 0-100
        total = max(0.0, min(100.0, total))
        
        return round(total, 4)
    
    except Exception as e:
        # Never crash - return a neutral score
        try:
            if response and len(str(response).strip()) > 10:
                return 25.0
            return 0.0
        except:
            return 0.0