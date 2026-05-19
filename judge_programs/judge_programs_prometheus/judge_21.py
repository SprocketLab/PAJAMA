def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a 
    question-decomposition and topic-coverage approach.
    
    Algorithm: 
    1. Extract implicit sub-questions/aspects from the query
    2. Check how many query aspects are addressed in the response
    3. Measure structural depth (nested reasoning, elaboration patterns)
    4. Assess information density via unique concept coverage
    5. Check for acknowledgment patterns and responsive framing
    
    This differs from other variants by focusing on query-response alignment
    through keyword/topic overlap analysis and response adequacy patterns.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.5
        
        # ---- Helper functions ----
        def tokenize(text):
            """Simple tokenization: lowercase, split on non-alpha, filter short words."""
            words = re.findall(r'[a-z]+', text.lower())
            return words
        
        def get_content_words(words, min_len=3):
            """Filter out very common stop words and short words."""
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'would', 'could',
                'should', 'will', 'just', 'more', 'also', 'than', 'them', 'then', 'into',
                'some', 'when', 'what', 'which', 'their', 'there', 'this', 'that', 'with',
                'from', 'they', 'were', 'your', 'about', 'each', 'make', 'like', 'does',
                'how', 'its', 'may', 'might', 'very', 'after', 'before', 'being', 'here',
                'where', 'those', 'these', 'through', 'while', 'other', 'because', 'such',
                'between', 'still', 'over', 'same', 'much', 'most', 'only', 'any', 'both',
                'come', 'get', 'got', 'way', 'said', 'say', 'says', 'going', 'thing',
                'things', 'really', 'well', 'know', 'think', 'want', 'see', 'look',
                'need', 'use', 'try', 'keep', 'let', 'help', 'take', 'made', 'find',
                'back', 'even', 'give', 'good', 'new', 'used', 'work', 'first', 'last'
            }
            return [w for w in words if len(w) >= min_len and w not in stop_words]
        
        def get_bigrams(words):
            """Generate bigrams from word list."""
            return [(words[i], words[i+1]) for i in range(len(words)-1)]
        
        def get_trigrams(words):
            """Generate trigrams from word list."""
            return [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
        
        # ---- 1. Query-Response Topic Overlap (0-20 points) ----
        query_words = tokenize(query)
        response_words = tokenize(response)
        
        query_content = get_content_words(query_words)
        response_content = get_content_words(response_words)
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        if len(query_content_set) > 0:
            # What fraction of query's content words appear in the response?
            direct_overlap = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            direct_overlap = 0.5
        
        # Also check bigram overlap for phrase-level coverage
        query_bigrams = set(get_bigrams(query_content))
        response_bigrams = set(get_bigrams(response_content))
        
        if len(query_bigrams) > 0:
            bigram_overlap = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_overlap = 0.0
        
        topic_score = (direct_overlap * 14) + (bigram_overlap * 6)  # max ~20
        
        # ---- 2. Sub-question / Aspect Coverage (0-20 points) ----
        # Extract aspects from query by looking for question words, conjunctions, commas
        # that suggest multiple aspects
        
        # Count question indicators in query
        question_patterns = re.findall(
            r'\b(how|what|why|when|where|which|who|explain|describe|tell|guide|help|advice|'
            r'understand|manage|handle|cope|address|provide|ensure|maintain)\b',
            query.lower()
        )
        num_aspects = max(len(set(question_patterns)), 1)
        
        # Check how many of these question-aspect words are addressed in response
        addressed_aspects = 0
        for pattern in set(question_patterns):
            # Check if the response contains content related to this aspect
            # by looking for the word or semantically related response patterns
            if pattern in response.lower():
                addressed_aspects += 1
            else:
                # Check if the response at least contains sentences that seem to address it
                # by looking for explanatory or instructional language
                related_patterns = {
                    'how': r'(by |through |step|first|then|next|start|begin)',
                    'what': r'(is a |refers to|means|defined as|consists)',
                    'why': r'(because|reason|due to|since|as a result)',
                    'explain': r'(means|basically|essentially|in other words|simply put)',
                    'help': r'(here|suggest|recommend|try|consider)',
                    'advice': r'(suggest|recommend|try|consider|tip)',
                    'cope': r'(manage|handle|deal|process|accept|allow)',
                    'understand': r'(concept|idea|think of|imagine|picture)',
                    'manage': r'(organize|prioritize|schedule|plan|track)',
                    'handle': r'(approach|deal|address|respond|manage)',
                    'ensure': r'(make sure|verify|check|confirm|guarantee)',
                    'describe': r'(looks like|appears|features|characteristics)',
                    'guide': r'(step|first|then|next|follow|begin)',
                    'provide': r'(here|offer|give|present|include)',
                    'maintain': r'(keep|continue|sustain|preserve|uphold)',
                    'address': r'(tackle|deal|handle|resolve|solve)',
                }
                if pattern in related_patterns:
                    if re.search(related_patterns[pattern], response.lower()):
                        addressed_aspects += 0.7
        
        aspect_ratio = addressed_aspects / num_aspects if num_aspects > 0 else 0.5
        aspect_score = min(aspect_ratio * 20, 20)
        
        # ---- 3. Information Density & Unique Concepts (0-15 points) ----
        # Measure unique content words relative to total, rewarding rich vocabulary
        
        if len(response_words) > 0:
            unique_ratio = len(set(response_content)) / max(len(response_content), 1)
            # Ideal unique ratio is around 0.4-0.7 (too high = sparse, too low = repetitive)
            if unique_ratio > 0.7:
                density_factor = 0.7 + 0.3 * (1.0 - (unique_ratio - 0.7) / 0.3)
            elif unique_ratio < 0.3:
                density_factor = unique_ratio / 0.3 * 0.5
            else:
                density_factor = 0.5 + 0.5 * ((unique_ratio - 0.3) / 0.4)
        else:
            density_factor = 0.0
        
        # Count unique content words as a raw measure of concept coverage
        num_unique_concepts = len(set(response_content))
        concept_richness = min(num_unique_concepts / 40.0, 1.0)  # cap at 40 unique concepts
        
        density_score = (density_factor * 7) + (concept_richness * 8)  # max ~15
        
        # ---- 4. Structural Depth & Elaboration (0-15 points) ----
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Check for elaboration patterns (because, for example, such as, this means, etc.)
        elaboration_markers = re.findall(
            r'\b(because|therefore|for example|for instance|such as|this means|'
            r'in other words|specifically|in particular|moreover|furthermore|'
            r'additionally|as a result|consequently|hence|thus|namely|'
            r'to illustrate|consider|imagine|think of|note that|'
            r'importantly|essentially|basically|the reason)\b',
            response.lower()
        )
        num_elaborations = len(elaboration_markers)
        
        # Check for conditional/nuanced reasoning
        nuance_markers = re.findall(
            r'\b(however|although|while|on the other hand|alternatively|'
            r'depending on|in some cases|it depends|unless|except|'
            r'keep in mind|bear in mind|remember that|note that|'
            r'if you|when you|in case)\b',
            response.lower()
        )
        num_nuances = len(nuance_markers)
        
        # Sentence count score (reward adequate length, diminishing returns)
        sentence_score = min(num_sentences / 8.0, 1.0) * 5  # max 5
        
        # Elaboration score
        elab_score = min(num_elaborations / 4.0, 1.0) * 5  # max 5
        
        # Nuance score
        nuance_score = min(num_nuances / 3.0, 1.0) * 5  # max 5
        
        structural_score = sentence_score + elab_score + nuance_score  # max 15
        
        # ---- 5. Responsive Engagement & Empathy (0-10 points) ----
        # Check if response directly engages with the query's emotional/practical needs
        
        engagement_score = 0.0
        
        # Direct address patterns
        direct_address = len(re.findall(
            r'\b(you|your|you\'re|you\'ve|you\'ll)\b', response.lower()
        ))
        engagement_score += min(direct_address / 5.0, 1.0) * 3  # max 3
        
        # Acknowledgment patterns (showing understanding of the query)
        acknowledgment_patterns = re.findall(
            r'\b(understand|hear you|see that|sounds like|must be|'
            r'it\'s okay|it\'s natural|it\'s normal|perfectly|completely|'
            r'sorry to hear|i can see|appreciate|recognize|'
            r'understandable|valid|reasonable|makes sense)\b',
            response.lower()
        )
        engagement_score += min(len(acknowledgment_patterns) / 3.0, 1.0) * 3  # max 3
        
        # Action-oriented language (showing practical help)
        action_patterns = re.findall(
            r'\b(try|consider|start|begin|make sure|ensure|'
            r'recommend|suggest|here are|here is|follow|'
            r'can help|will help|should|could|let\'s)\b',
            response.lower()
        )
        engagement_score += min(len(action_patterns) / 4.0, 1.0) * 4  # max 4
        
        # ---- 6. Negative Signals / Penalties (0 to -10) ----
        penalties = 0.0
        
        # Dismissive language
        dismissive = re.findall(
            r'\b(just|simply|obviously|clearly|easy|no big deal|'
            r'get over it|move on|stop|don\'t worry about it|'
            r'not a big deal|whatever|anyway)\b',
            response.lower()
        )
        # "just" and "simply" are common, only penalize if frequent
        just_count = response.lower().count('just') + response.lower().count('simply')
        if just_count > 3:
            penalties -= (just_count - 3) * 0.5
        
        # Hedging / uncertainty that suggests incompleteness
        hedging = re.findall(
            r'\b(might not|may not|probably won\'t|can\'t really|'
            r'not sure|don\'t know|hard to say|it\'s unclear|'
            r'i guess|maybe|perhaps not)\b',
            response.lower()
        )
        if len(hedging) > 2:
            penalties -= (len(hedging) - 2) * 1.0
        
        # Very short response penalty
        if len(response) < 100:
            penalties -= 5.0
        elif len(response) < 200:
            penalties -= 2.0
        
        # Repetition penalty: check for repeated phrases
        response_trigrams = get_trigrams(response_words)
        if len(response_trigrams) > 0:
            trigram_counts = Counter(response_trigrams)
            repeated_trigrams = sum(1 for _, c in trigram_counts.items() if c > 2)
            penalties -= repeated_trigrams * 0.5
        
        # ---- 7. Response Length Adequacy (0-10 points) ----
        # Longer responses tend to be more complete, but with diminishing returns
        response_len = len(response)
        
        # Sigmoid-like curve: ramps up around 200-600 chars, plateaus after
        if response_len < 50:
            length_score = 1.0
        else:
            length_score = 10.0 * (1.0 - math.exp(-response_len / 400.0))
        length_score = min(length_score, 10.0)
        
        # ---- 8. Query Complexity Matching (0-10 points) ----
        # More complex queries need more comprehensive responses
        query_len = len(query)
        query_complexity = len(set(question_patterns))
        
        # Expected response effort based on query complexity
        if query_complexity >= 3:
            expected_min_sentences = 6
        elif query_complexity >= 2:
            expected_min_sentences = 4
        else:
            expected_min_sentences = 3
        
        complexity_match = min(num_sentences / expected_min_sentences, 1.5)
        complexity_score = min(complexity_match * 6.67, 10.0)  # max 10
        
        # ---- Combine all scores ----
        raw_score = (
            topic_score +        # 0-20: query topic coverage
            aspect_score +       # 0-20: sub-question coverage
            density_score +      # 0-15: information density
            structural_score +   # 0-15: elaboration depth
            engagement_score +   # 0-10: responsive engagement
            length_score +       # 0-10: length adequacy
            complexity_score +   # 0-10: complexity matching
            penalties            # -10 to 0: negative signals
        )
        # Total theoretical max: ~100, typical range 10-80
        
        # Normalize to 1-5 scale to match the examples
        # Map raw_score from range [0, 80] to [1, 5]
        normalized = 1.0 + 4.0 * (raw_score / 80.0)
        
        # Clamp to [0.5, 5.5]
        final_score = max(0.5, min(5.5, normalized))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middle score
        try:
            if response and len(response.strip()) > 100:
                return 3.0
            return 1.5
        except:
            return 2.0