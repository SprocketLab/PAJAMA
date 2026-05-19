def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and topic-coverage approach. Analyzes how many implicit sub-topics/aspects of the query
    are addressed in the response, plus structural depth indicators.
    
    This variant focuses on:
    1. Query decomposition into key terms/concepts and checking coverage
    2. Semantic field coverage (related word clusters)
    3. Response structure depth (explanatory patterns, examples, transitions)
    4. Information density and specificity metrics
    """
    import re
    import math
    import string
    from collections import Counter
    
    try:
        if not query or not response:
            return 0.0
        
        query = str(query)
        response = str(response)
        
        if len(response.strip()) < 10:
            return 0.5
        
        # ---- 1. QUERY KEYWORD COVERAGE ----
        # Extract meaningful words from query (content words)
        stop_words = {
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
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'am',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
            'about', 'up', 'down', 'also', 'any', 'every', 'much', 'many',
            'get', 'got', 'make', 'made', 'like', 'even', 'still', 'way',
            'well', 'back', 'also', 'go', 'going', 'come', 'take', 'know',
            'say', 'said', 'tell', 'told', 'thing', 'things', 'let', 'one',
            'two', 'first', 'new', 'now', 'people', 'person', 'time', 'long',
            'look', 'see', 'think', 'want', 'give', 'use', 'find', 'put',
            'must', 'however', 'something', 'someone', 'individual', 'scenario',
            'where', 'been', 'being', 'another', 'between'
        }
        
        def extract_content_words(text):
            text_lower = text.lower()
            words = re.findall(r'[a-z]+', text_lower)
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        query_words = extract_content_words(query)
        response_lower = response.lower()
        
        # Check what fraction of query content words appear in response
        if query_words:
            query_word_set = set(query_words)
            covered = sum(1 for w in query_word_set if w in response_lower)
            keyword_coverage = covered / len(query_word_set)
        else:
            keyword_coverage = 0.5
        
        # ---- 2. QUERY CONCEPT CLUSTER COVERAGE ----
        # Identify conceptual clusters from the query using bigrams/trigrams
        query_lower = query.lower()
        query_tokens = re.findall(r'[a-z]+', query_lower)
        
        # Extract bigrams from query
        query_bigrams = set()
        for i in range(len(query_tokens) - 1):
            bg = query_tokens[i] + ' ' + query_tokens[i+1]
            if query_tokens[i] not in stop_words or query_tokens[i+1] not in stop_words:
                query_bigrams.add(bg)
        
        if query_bigrams:
            bigram_hits = sum(1 for bg in query_bigrams if bg in response_lower)
            bigram_coverage = bigram_hits / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        # ---- 3. STRUCTURAL DEPTH ANALYSIS ----
        # Look for explanatory patterns that indicate thorough coverage
        
        # Causal/explanatory connectors
        explanation_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bas a result\b', r'\bdue to\b', r'\bthis means\b', r'\bin other words\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', r'\bin addition\b',
            r'\bon the other hand\b', r'\bhowever\b', r'\bnevertheless\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bnotably\b',
            r'\bremember\b', r'\bkeep in mind\b', r'\bnote that\b',
            r'\bhere\'s\b', r'\bhere are\b', r'\blet me\b', r'\blet\'s\b',
            r'\bimagine\b', r'\bthink of\b', r'\bconsider\b',
        ]
        
        explanation_count = 0
        for pat in explanation_patterns:
            explanation_count += len(re.findall(pat, response_lower))
        
        # Normalize: diminishing returns after ~8 explanatory markers
        explanation_score = min(1.0, explanation_count / 8.0)
        
        # ---- 4. INFORMATION DENSITY & SPECIFICITY ----
        response_words = re.findall(r'[a-z]+', response_lower)
        response_word_count = len(response_words)
        
        if response_word_count < 5:
            return 1.0
        
        # Unique content words ratio (vocabulary richness)
        response_content = [w for w in response_words if w not in stop_words and len(w) > 2]
        if response_content:
            unique_ratio = len(set(response_content)) / len(response_content)
        else:
            unique_ratio = 0.0
        
        # Specificity: presence of specific/concrete terms (numbers, proper nouns, technical terms)
        specific_patterns = [
            r'\d+',  # numbers
            r'\b[A-Z][a-z]+\b',  # capitalized words (proper nouns, emphasis)
        ]
        specificity_count = 0
        for pat in specific_patterns:
            specificity_count += len(re.findall(pat, response))
        
        # Long words (likely technical/specific)
        long_words = [w for w in response_content if len(w) > 7]
        long_word_ratio = len(long_words) / max(1, len(response_content))
        
        specificity_score = min(1.0, (specificity_count / 15.0) * 0.5 + long_word_ratio * 0.5)
        
        # ---- 5. RESPONSE LENGTH ADEQUACY ----
        # Longer responses tend to be more complete, but with diminishing returns
        # Calibrate based on query complexity
        query_content_count = len(query_words)
        expected_min_words = max(50, query_content_count * 8)
        
        length_ratio = response_word_count / expected_min_words
        length_score = min(1.0, length_ratio)
        # Slight bonus for substantial responses
        if response_word_count > 150:
            length_score = min(1.0, length_score + 0.1)
        
        # ---- 6. SENTENCE COUNT AND VARIETY ----
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentence_count = len(sentences)
        
        # Average sentence length variety (indicates structured explanation)
        if sentence_count > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = sum(sent_lengths) / len(sent_lengths)
            sent_len_variance = sum((l - avg_sent_len)**2 for l in sent_lengths) / len(sent_lengths)
            sent_variety = min(1.0, math.sqrt(sent_len_variance) / 10.0)
        else:
            sent_variety = 0.0
        
        sentence_score = min(1.0, sentence_count / 8.0)
        
        # ---- 7. EMPATHY/ENGAGEMENT DETECTION (for emotional queries) ----
        emotional_query_signals = [
            'feeling', 'feel', 'stress', 'frustrat', 'sad', 'lonely', 'heartbroken',
            'devastat', 'anxious', 'worried', 'upset', 'angry', 'depress',
            'comfort', 'support', 'help', 'struggling', 'difficult', 'tough',
            'breakup', 'passed away', 'died', 'loss', 'grief', 'regret',
            'exhaustion', 'desperate', 'despair', 'fear'
        ]
        
        is_emotional_query = sum(1 for sig in emotional_query_signals if sig in query_lower)
        
        empathy_patterns = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b', r'\bi\'m sorry\b',
            r'\bthat\'s understandable\b', r'\bit\'s okay\b', r'\bit\'s natural\b',
            r'\bcompletely\b', r'\bperfectly\b', r'\babsolutely\b',
            r'\byour feelings\b', r'\byour pain\b', r'\byour experience\b',
            r'\bvalid\b', r'\bnormal\b', r'\bnatural\b',
            r'\btake your time\b', r'\bdon\'t rush\b', r'\bgive yourself\b',
            r'\bwe\'re here\b', r'\bhere for you\b',
        ]
        
        empathy_count = sum(1 for pat in empathy_patterns if re.search(pat, response_lower))
        
        if is_emotional_query >= 2:
            empathy_score = min(1.0, empathy_count / 4.0)
        else:
            empathy_score = 0.5  # neutral for non-emotional queries
        
        # ---- 8. ACTIONABILITY / CONCRETE ADVICE ----
        action_patterns = [
            r'\btry\b', r'\bcould\b', r'\bshould\b', r'\brecommend\b',
            r'\bstart\b', r'\bbegin\b', r'\bmake sure\b', r'\bensure\b',
            r'\bstep\b', r'\bfirst\b', r'\bnext\b', r'\bthen\b',
            r'\bhere are\b', r'\bhere\'s\b', r'\byou can\b', r'\byou might\b',
            r'\bconsider\b', r'\bfocus on\b', r'\bpractice\b',
        ]
        
        action_count = sum(len(re.findall(pat, response_lower)) for pat in action_patterns)
        actionability_score = min(1.0, action_count / 6.0)
        
        # ---- 9. DISMISSIVENESS PENALTY ----
        # Detect dismissive or shallow patterns
        dismissive_patterns = [
            r'\bjust\b.*\bget over\b', r'\bjust\b.*\bmove on\b',
            r'\bit\'s not a big deal\b', r'\bstop\b.*\bcomplaining\b',
            r'\bget yourself together\b', r'\byou need to\b.*\bget rid\b',
            r'\bmaybe you\'re just not\b', r'\bjust keep\b.*\btrying\b',
            r'\byou should be able\b',
        ]
        
        dismissive_count = sum(1 for pat in dismissive_patterns if re.search(pat, response_lower))
        dismissive_penalty = min(0.5, dismissive_count * 0.15)
        
        # ---- 10. CONTRADICTION WITH QUERY INTENT ----
        # If query asks for X but response does opposite
        negation_penalty = 0.0
        
        # Check if response says "can't", "unable", "not possible" too much without offering alternatives
        inability_patterns = [r'\bcan\'t\b', r'\bcannot\b', r'\bunable\b', r'\bnot possible\b',
                             r'\bwon\'t\b', r'\bmight not\b', r'\bprobably won\'t\b']
        inability_count = sum(len(re.findall(pat, response_lower)) for pat in inability_patterns)
        if inability_count > 2:
            negation_penalty = min(0.3, (inability_count - 2) * 0.1)
        
        # ---- COMPOSITE SCORING ----
        # Weight the components based on importance for completeness
        
        # Determine if query is more emotional vs informational
        emotional_weight = min(1.0, is_emotional_query / 3.0)
        informational_weight = 1.0 - emotional_weight * 0.5
        
        score = 0.0
        
        # Core completeness metrics
        score += keyword_coverage * 1.5          # How many query concepts addressed
        score += bigram_coverage * 1.0           # Phrase-level coverage
        score += explanation_score * 1.2         # Depth of explanation
        score += length_score * 1.0              # Adequate length
        score += sentence_score * 0.8            # Multiple points covered
        score += sent_variety * 0.3              # Structural variety
        score += specificity_score * 0.8         # Concrete/specific content
        score += unique_ratio * 0.5              # Vocabulary richness
        score += actionability_score * 0.8       # Concrete advice/steps
        
        # Emotional context adjustment
        score += empathy_score * emotional_weight * 1.0
        
        # Penalties
        score -= dismissive_penalty * 2.0
        score -= negation_penalty * 2.0
        
        # Normalize to 1-5 scale
        # Max theoretical raw score is roughly: 1.5+1.0+1.2+1.0+0.8+0.3+0.8+0.5+0.8+1.0 = 8.9
        # Typical good response: ~5-6
        # Typical poor response: ~2-3
        
        max_raw = 8.0
        normalized = (score / max_raw) * 4.0 + 1.0  # Maps to 1-5
        
        # Clamp to 1-5
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return middle score on any error
        return 2.5