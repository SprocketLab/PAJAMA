def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response to a query.
    Uses a multi-signal approach focusing on thoroughness, depth, and coverage.
    Returns a score from 0 to 100 where higher = better quality.
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        # Handle edge cases
        if not response or not isinstance(response, str):
            return 0
        if not query or not isinstance(query, str):
            return 5
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 5:
            return 1
        
        # ============================================================
        # FEATURE 1: Response Length & Depth Score (0-20)
        # Longer responses tend to be more complete and thorough
        # ============================================================
        resp_len = len(response)
        word_count = len(response.split())
        
        # Logarithmic scaling for length - diminishing returns
        if word_count <= 5:
            length_score = 1
        elif word_count <= 20:
            length_score = 3 + (word_count - 5) * 0.3
        elif word_count <= 50:
            length_score = 7 + (word_count - 20) * 0.2
        elif word_count <= 150:
            length_score = 13 + (word_count - 50) * 0.05
        elif word_count <= 300:
            length_score = 18 + (word_count - 150) * 0.013
        else:
            length_score = 20
        
        length_score = min(20, max(0, length_score))
        
        # ============================================================
        # FEATURE 2: Sentence Complexity & Structure (0-15)
        # More sentences, varied structure = more thorough coverage
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        # Sentence count score
        if num_sentences <= 1:
            sent_score = 2
        elif num_sentences <= 3:
            sent_score = 5
        elif num_sentences <= 6:
            sent_score = 9
        elif num_sentences <= 10:
            sent_score = 12
        else:
            sent_score = 14
        
        # Bonus for varied sentence lengths (indicates structured thinking)
        if num_sentences >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            if variance > 10:
                sent_score = min(15, sent_score + 1)
        
        sent_score = min(15, max(0, sent_score))
        
        # ============================================================
        # FEATURE 3: Query Term Coverage (0-20)
        # How well does the response address terms/concepts from the query
        # ============================================================
        # Extract meaningful words from query (remove stopwords)
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'would',
            'could', 'also', 'like', 'get', 'got', 'much', 'many', 'may', 'might',
            'really', 'think', 'know', 'want', 'see', 'way', 'make', 'go', 'going',
            'been', 'into', 'even', 'well', 'back', 'any', 'give', 'day', 'come',
            'im', 'ive', 'dont', 'doesnt', 'didnt', 'cant', 'wont', 'isnt', 'arent',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        query_words = extract_content_words(query)
        response_lower = response.lower()
        
        if query_words:
            unique_query_words = list(set(query_words))
            covered = sum(1 for w in unique_query_words if w in response_lower)
            coverage_ratio = covered / len(unique_query_words) if unique_query_words else 0
            query_coverage_score = coverage_ratio * 20
        else:
            query_coverage_score = 10  # neutral
        
        query_coverage_score = min(20, max(0, query_coverage_score))
        
        # ============================================================
        # FEATURE 4: Information Density & Specificity (0-15)
        # Specific details, examples, names, numbers indicate thoroughness
        # ============================================================
        
        # Count specific indicators
        # Numbers and statistics
        numbers = re.findall(r'\d+', response)
        num_count = min(len(numbers), 8)
        
        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', response)
        proper_count = min(len(proper_nouns), 10)
        
        # Technical/specific vocabulary (longer words)
        resp_words = response.split()
        long_words = [w for w in resp_words if len(w) > 8]
        long_word_ratio = len(long_words) / max(len(resp_words), 1)
        
        # Quoted text, references, citations
        has_quotes = 1 if '"' in response or "'" in response or '*' in response else 0
        has_parenthetical = 1 if '(' in response and ')' in response else 0
        
        # Examples and elaborations
        example_markers = ['for example', 'for instance', 'such as', 'e.g.', 'i.e.',
                          'specifically', 'in particular', 'namely', 'including',
                          'consider', 'imagine', 'suppose']
        example_count = sum(1 for m in example_markers if m in response_lower)
        
        specificity_score = (
            min(3, num_count * 0.4) +
            min(3, proper_count * 0.4) +
            min(3, long_word_ratio * 20) +
            has_quotes * 1.5 +
            has_parenthetical * 1 +
            min(3, example_count * 1.5)
        )
        specificity_score = min(15, max(0, specificity_score))
        
        # ============================================================
        # FEATURE 5: Structural Completeness Indicators (0-15)
        # Lists, multiple points, transitions, explanations
        # ============================================================
        
        # Check for list/enumeration patterns
        bullet_patterns = re.findall(r'(?:^|\n)\s*[-*•]\s', response)
        numbered_patterns = re.findall(r'(?:^|\n)\s*\d+[.)]\s', response)
        list_count = len(bullet_patterns) + len(numbered_patterns)
        
        # Transition/connective words (indicate multi-faceted coverage)
        transitions = [
            'however', 'moreover', 'furthermore', 'additionally', 'also',
            'on the other hand', 'in addition', 'first', 'second', 'third',
            'finally', 'meanwhile', 'nevertheless', 'consequently', 'therefore',
            'in contrast', 'similarly', 'likewise', 'alternatively', 'besides',
            'although', 'despite', 'whereas', 'while', 'yet', 'but',
            'another', 'beyond that', 'not only', 'as well'
        ]
        transition_count = sum(1 for t in transitions if t in response_lower)
        
        # Causal/explanatory markers (indicate depth)
        explanation_markers = [
            'because', 'since', 'due to', 'the reason', 'this means',
            'as a result', 'which leads', 'this is because', 'in other words',
            'essentially', 'fundamentally', 'the key', 'important',
            'significant', 'crucial', 'notably'
        ]
        explanation_count = sum(1 for e in explanation_markers if e in response_lower)
        
        # Code blocks (for technical queries)
        code_blocks = response.count('```')
        has_code = 1 if code_blocks >= 2 or '    ' in response else 0
        
        structure_score = (
            min(4, list_count * 1.2) +
            min(5, transition_count * 0.8) +
            min(4, explanation_count * 1.0) +
            has_code * 2
        )
        structure_score = min(15, max(0, structure_score))
        
        # ============================================================
        # FEATURE 6: Question Addressing & Sub-question Coverage (0-15)
        # Does the response address multiple aspects of the query?
        # ============================================================
        
        # Count question marks in query (sub-questions)
        query_questions = query.count('?')
        
        # Detect sub-topics in query via conjunctions and separators
        query_aspects = 0
        aspect_markers = [' and ', ' or ', '?', ';', '\n', ' also ', ' additionally ',
                         ' what ', ' how ', ' why ', ' when ', ' where ', ' which ']
        for marker in aspect_markers:
            query_aspects += query.lower().count(marker)
        query_aspects = max(1, min(query_aspects, 8))
        
        # Check for multiple distinct paragraphs/sections in response
        paragraphs = [p.strip() for p in response.split('\n') if len(p.strip()) > 15]
        num_paragraphs = max(1, len(paragraphs))
        
        # Unique content words in response (vocabulary richness = broader coverage)
        resp_content_words = extract_content_words(response)
        unique_resp_words = set(resp_content_words)
        vocab_richness = len(unique_resp_words) / max(len(resp_content_words), 1) if resp_content_words else 0
        
        # Multi-aspect coverage heuristic
        aspect_score = (
            min(5, num_paragraphs * 1.2) +
            min(5, len(unique_resp_words) / 8) +
            min(3, vocab_richness * 5) +
            min(2, (1 if word_count > query_aspects * 15 else 0) * 2)
        )
        aspect_score = min(15, max(0, aspect_score))
        
        # ============================================================
        # PENALTIES
        # ============================================================
        penalty = 0
        
        # Penalty for being just a redirect/link without substance
        redirect_phrases = [
            'you might be interested in', 'check out this', 'see this link',
            'please read our rules', 'welcome to /r/', 'this has been removed',
            'your post has been', 'please read the sidebar'
        ]
        for phrase in redirect_phrases:
            if phrase in response_lower:
                penalty += 15
        
        # Penalty for meta-responses that don't actually answer
        meta_phrases = [
            'i can help you with that', 'great question', 'that\'s a good question',
            'interesting question'
        ]
        meta_count = sum(1 for p in meta_phrases if p in response_lower)
        # Only penalize if the response is JUST meta without substance
        if meta_count > 0 and word_count < 30:
            penalty += 5
        
        # Penalty for very short responses relative to query complexity
        query_word_count = len(query.split())
        if query_word_count > 50 and word_count < 30:
            penalty += 10
        elif query_word_count > 30 and word_count < 20:
            penalty += 8
        
        # Penalty for truncated responses
        if response.rstrip()[-1:] not in '.!?")\']' and word_count > 20:
            # Might be truncated but could still be informative
            penalty += 2
        
        # Penalty for responses that are just opinions without elaboration
        if word_count < 25 and not any(m in response_lower for m in example_markers + explanation_markers):
            penalty += 5
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        raw_score = (
            length_score +          # 0-20
            sent_score +            # 0-15
            query_coverage_score +  # 0-20
            specificity_score +     # 0-15
            structure_score +       # 0-15
            aspect_score            # 0-15
        )
        # Max possible raw = 100
        
        final_score = raw_score - penalty
        
        # Clamp to 0-100
        final_score = max(0, min(100, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: simple length-based score
        try:
            return min(50, max(1, len(str(response).split()) / 3))
        except:
            return 25