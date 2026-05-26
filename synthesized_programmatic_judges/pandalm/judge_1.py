def judging_function(query, response):
    """
    Evaluates relevance of a response to a query using word overlap,
    topic alignment, and content quality signals.
    
    Returns a score from 0-100 where higher = better relevance.
    """
    try:
        import re
        import math
        from collections import Counter
        
        # Handle edge cases
        if not response or not isinstance(response, str) or response.strip() == "":
            return 0.0
        if not query or not isinstance(query, str) or query.strip() == "":
            return 5.0  # Can't judge relevance without a query
        
        query = query.strip()
        response = response.strip()
        
        # Check for non-informative responses like <noinput>
        if response.lower() in ('<noinput>', 'noinput', 'n/a', 'none', ''):
            return 0.0
        
        # --- Tokenization ---
        def tokenize(text):
            """Lowercase tokenization, removing punctuation."""
            text = text.lower()
            tokens = re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text)
            return tokens
        
        # Common English stopwords
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
            'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
            'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'about', 'up', 'down', 'also', 'much',
            'get', 'got', 'like', 'make', 'made', 'well', 'back', 'even', 'still',
        }
        
        # Task/instruction words to filter out
        TASK_WORDS = {
            'explain', 'describe', 'provide', 'generate', 'write', 'create',
            'list', 'give', 'compare', 'contrast', 'rewrite', 'come', 'following',
            'sentence', 'example', 'examples', 'input', 'output', 'article',
            'paragraph', 'text', 'word', 'words', 'letter', 'please', 'answer',
            'question', 'tell', 'show', 'find', 'identify', 'define', 'discuss',
            'analyze', 'summarize', 'suggest', 'recommend', 'crop', 'reduce',
            'add', 'given', 'mean', 'meant', 'wrote'
        }
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        if len(response_tokens) == 0:
            return 1.0
        
        # Content words (removing stopwords)
        query_content = [t for t in query_tokens if t not in STOPWORDS]
        response_content = [t for t in response_tokens if t not in STOPWORDS]
        
        # Topic words: content words minus task words
        query_topic = [t for t in query_content if t not in TASK_WORDS]
        response_topic = [t for t in response_content if t not in TASK_WORDS]
        
        # --- SCORE 1: Content word overlap (Jaccard-like) ---
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        if len(query_content_set) > 0:
            overlap = query_content_set & response_content_set
            # Weighted recall: how many query content words appear in response
            content_recall = len(overlap) / len(query_content_set)
        else:
            content_recall = 0.5  # neutral if query has no content words
        
        # --- SCORE 2: Topic word overlap ---
        query_topic_set = set(query_topic)
        response_topic_set = set(response_topic)
        
        if len(query_topic_set) > 0:
            topic_overlap = query_topic_set & response_topic_set
            topic_recall = len(topic_overlap) / len(query_topic_set)
        else:
            topic_recall = 0.5
        
        # --- SCORE 3: Bigram overlap for phrase-level relevance ---
        def get_bigrams(tokens):
            return [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]
        
        query_bigrams = set(get_bigrams(query_tokens))
        response_bigrams = set(get_bigrams(response_tokens))
        
        if len(query_bigrams) > 0:
            bigram_recall = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_recall = 0.3
        
        # --- SCORE 4: Response length quality ---
        resp_len = len(response_tokens)
        query_len = len(query_tokens)
        
        # Ideal response is typically 2-10x the query length
        if resp_len < 3:
            length_score = 0.1
        elif resp_len < 8:
            length_score = 0.3
        elif resp_len < 15:
            length_score = 0.6
        elif resp_len < 80:
            length_score = 1.0
        elif resp_len < 150:
            length_score = 0.9
        else:
            length_score = 0.7
        
        # --- SCORE 5: Repetition penalty ---
        if len(response_tokens) > 0:
            token_counts = Counter(response_content)
            if len(response_content) > 0:
                max_freq = max(token_counts.values()) if token_counts else 0
                # Penalize if any content word appears too many times
                repetition_ratio = max_freq / len(response_content)
                if repetition_ratio > 0.5:
                    repetition_penalty = 0.3
                elif repetition_ratio > 0.3:
                    repetition_penalty = 0.6
                else:
                    repetition_penalty = 1.0
                
                # Also check unique ratio
                unique_ratio = len(set(response_tokens)) / len(response_tokens)
                if unique_ratio < 0.3:
                    repetition_penalty *= 0.4
                elif unique_ratio < 0.5:
                    repetition_penalty *= 0.7
            else:
                repetition_penalty = 0.8
        else:
            repetition_penalty = 0.5
        
        # --- SCORE 6: Does response directly address the query? ---
        # Check if response echoes key query terms in its opening
        response_first_tokens = response_tokens[:min(20, len(response_tokens))]
        response_first_set = set(response_first_tokens)
        
        if len(query_topic_set) > 0:
            early_mention = len(query_topic_set & response_first_set) / len(query_topic_set)
        else:
            early_mention = 0.5
        
        # --- SCORE 7: Substantive content check ---
        # Penalize responses that are just echoing the query without adding info
        response_only = response_content_set - query_content_set
        if len(response_content_set) > 0:
            novelty_ratio = len(response_only) / len(response_content_set)
        else:
            novelty_ratio = 0.0
        
        # We want some novelty (the response adds information) but also overlap
        # Sweet spot: response shares topic words but adds new content
        if novelty_ratio < 0.1:
            # Almost entirely echoing query
            novelty_score = 0.3
        elif novelty_ratio > 0.95:
            # Almost no overlap - possibly off-topic
            novelty_score = 0.4
        else:
            novelty_score = 0.5 + 0.5 * min(novelty_ratio, 0.7) / 0.7
        
        # --- SCORE 8: Sentence structure quality ---
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        if num_sentences >= 2:
            structure_score = 1.0
        elif num_sentences == 1 and resp_len > 10:
            structure_score = 0.7
        else:
            structure_score = 0.4
        
        # --- SCORE 9: TF-based cosine similarity ---
        def tf_vector(tokens):
            counter = Counter(tokens)
            total = len(tokens) if len(tokens) > 0 else 1
            return {word: count / total for word, count in counter.items()}
        
        query_tf = tf_vector(query_content)
        response_tf = tf_vector(response_content)
        
        all_words = set(query_tf.keys()) | set(response_tf.keys())
        
        if len(all_words) > 0:
            dot = sum(query_tf.get(w, 0) * response_tf.get(w, 0) for w in all_words)
            mag_q = math.sqrt(sum(v**2 for v in query_tf.values())) if query_tf else 0
            mag_r = math.sqrt(sum(v**2 for v in response_tf.values())) if response_tf else 0
            
            if mag_q > 0 and mag_r > 0:
                cosine_sim = dot / (mag_q * mag_r)
            else:
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0
        
        # --- SCORE 10: Check for degenerate/broken responses ---
        degenerate_penalty = 1.0
        
        # Check for truncation (ends mid-word or mid-sentence without punctuation)
        if response[-1] not in '.!?")\']' and len(response) > 50:
            # Might be truncated
            degenerate_penalty *= 0.85
        
        # Check for excessive repetition of phrases
        response_lower = response.lower()
        for phrase_len in [3, 4, 5]:
            words = response_tokens
            if len(words) >= phrase_len * 3:
                phrases = []
                for i in range(len(words) - phrase_len + 1):
                    phrases.append(' '.join(words[i:i+phrase_len]))
                phrase_counts = Counter(phrases)
                most_common_count = phrase_counts.most_common(1)[0][1] if phrase_counts else 0
                if most_common_count > 3:
                    degenerate_penalty *= 0.5
                    break
        
        # Check for garbage/nonsensical patterns
        if re.search(r'(.)\1{5,}', response):
            degenerate_penalty *= 0.5
        
        # --- Combine scores ---
        # Weights emphasize topic overlap and content quality
        score = (
            content_recall * 18 +       # 0-18: content word recall
            topic_recall * 20 +          # 0-20: topic word recall
            bigram_recall * 8 +          # 0-8: phrase overlap
            cosine_sim * 12 +            # 0-12: TF cosine similarity
            early_mention * 8 +          # 0-8: early mention of topic
            novelty_score * 10 +         # 0-10: balanced novelty
            length_score * 10 +          # 0-10: appropriate length
            structure_score * 6 +        # 0-6: sentence structure
            repetition_penalty * 4 +     # 0-4: repetition quality
            degenerate_penalty * 4       # 0-4: not degenerate
        )
        
        # Apply degenerate penalty as a multiplier too
        score *= degenerate_penalty
        
        # Ensure score is in [0, 100]
        score = max(0.0, min(100.0, score))
        
        return round(score, 4)
    
    except Exception as e:
        # Fallback: return a low-middle score
        try:
            if response and len(response.strip()) > 10:
                return 25.0
            return 5.0
        except:
            return 5.0