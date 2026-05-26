def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using:
    - Question/aspect extraction from query and coverage analysis
    - Information density via unique noun-like words and detail markers
    - Structural diversity (different sentence types, enumeration patterns)
    - Depth signals (explanations, examples, reasoning chains)
    - Empathy/acknowledgment coverage when query is emotional
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
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # ============================================================
        # 1. QUERY ASPECT EXTRACTION AND COVERAGE
        # ============================================================
        # Extract key noun phrases / content words from query
        # Use a simple approach: extract words 4+ chars, not stopwords
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
            'their', 'there', 'what', 'when', 'where', 'which', 'would',
            'could', 'should', 'about', 'after', 'before', 'between', 'through',
            'during', 'each', 'some', 'other', 'into', 'more', 'most', 'also',
            'just', 'than', 'then', 'them', 'these', 'those', 'very', 'being',
            'does', 'doing', 'done', 'having', 'here', 'how', 'much', 'many',
            'such', 'only', 'same', 'will', 'over', 'under', 'again', 'once',
            'both', 'your', 'myself', 'yourself', 'itself', 'ourselves',
            'person', 'individual', 'model', 'response', 'provide', 'following',
            'need', 'must', 'like', 'make', 'made', 'know', 'want', 'come',
            'take', 'give', 'tell', 'well', 'back', 'even', 'still', 'way',
            'because', 'every', 'good', 'great', 'help', 'first', 'able',
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if len(w) >= 4 and w not in stopwords]
        
        query_content = extract_content_words(query)
        response_content = extract_content_words(response)
        
        query_content_set = set(query_content)
        response_content_set = set(response_content)
        
        # How many query concepts appear in response
        if query_content_set:
            query_coverage = len(query_content_set & response_content_set) / len(query_content_set)
        else:
            query_coverage = 0.5
        
        # ============================================================
        # 2. QUESTION / SUB-QUESTION DETECTION AND ADDRESSING
        # ============================================================
        # Count question marks and question-like phrases in query
        question_words = re.findall(r'\b(how|what|why|when|where|who|which|can|could|would|should|does|is|are)\b', query_lower)
        num_question_aspects = max(len(set(question_words)), 1)
        
        # Check if response addresses these question types
        # E.g., "how" questions expect process/method words
        addressed_aspects = 0
        
        if 'how' in question_words:
            process_indicators = re.findall(r'\b(step|first|then|next|start|begin|process|method|way|approach|technique|follow|ensure|by)\b', response_lower)
            if len(process_indicators) >= 2:
                addressed_aspects += 1
        
        if 'what' in question_words:
            definition_indicators = re.findall(r'\b(is|are|means|refers|defined|called|known|type|kind|form)\b', response_lower)
            if len(definition_indicators) >= 2:
                addressed_aspects += 1
        
        if 'why' in question_words:
            reason_indicators = re.findall(r'\b(because|reason|since|due|cause|result|therefore|leads|consequence)\b', response_lower)
            if len(reason_indicators) >= 1:
                addressed_aspects += 1
        
        aspect_coverage_score = addressed_aspects / num_question_aspects if num_question_aspects > 0 else 0.5
        
        # ============================================================
        # 3. INFORMATION DENSITY AND RICHNESS
        # ============================================================
        response_words = re.findall(r'[a-z]+', response_lower)
        total_words = len(response_words)
        
        if total_words == 0:
            return 0.5
        
        unique_words = set(response_words)
        vocabulary_richness = len(unique_words) / max(total_words, 1)
        
        # Count "detail markers" — specific, concrete information signals
        detail_patterns = [
            r'\b\d+\b',  # numbers
            r'\b(specifically|particularly|especially|notably|importantly)\b',
            r'\b(example|instance|such as|like|e\.g\.|for instance)\b',
            r'\b(because|since|due to|as a result|therefore|consequently)\b',
            r'\b(however|although|while|whereas|nevertheless|on the other hand)\b',
            r'\b(additionally|furthermore|moreover|also|in addition)\b',
        ]
        
        detail_count = 0
        for pattern in detail_patterns:
            detail_count += len(re.findall(pattern, response_lower))
        
        detail_density = min(detail_count / max(total_words / 50, 1), 5.0)
        
        # ============================================================
        # 4. STRUCTURAL COMPLETENESS
        # ============================================================
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        # Sentence variety — different sentence starters
        starters = []
        for s in sentences:
            words = s.split()
            if words:
                starters.append(words[0].lower())
        
        starter_variety = len(set(starters)) / max(len(starters), 1) if starters else 0
        
        # Check for enumeration / structured content (numbered lists, lettered items)
        enumeration_patterns = re.findall(r'(?:^|\n)\s*(?:\d+[.):]|\-|\*|•|[a-z][.):])', response)
        has_structure = min(len(enumeration_patterns) / 3.0, 1.0)
        
        # Paragraph count
        paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 20]
        paragraph_score = min(len(paragraphs) / 3.0, 1.0)
        
        # ============================================================
        # 5. DEPTH SIGNALS — causal reasoning, elaboration
        # ============================================================
        causal_connectors = re.findall(
            r'\b(because|therefore|thus|hence|consequently|as a result|this means|'
            r'this leads|which means|so that|in order to|the reason|this is why|'
            r'this ensures|this helps|this allows|this enables|by doing)\b',
            response_lower
        )
        causal_depth = min(len(causal_connectors) / 3.0, 1.5)
        
        # Conditional/nuance handling
        nuance_words = re.findall(
            r'\b(if|unless|depending|might|may|could|sometimes|often|usually|'
            r'in some cases|it depends|however|although|while|whereas|'
            r'on the other hand|alternatively|keep in mind|note that|remember)\b',
            response_lower
        )
        nuance_score = min(len(nuance_words) / 4.0, 1.5)
        
        # ============================================================
        # 6. EMOTIONAL/EMPATHY COVERAGE (when query is emotional)
        # ============================================================
        emotional_query_words = re.findall(
            r'\b(feeling|frustrated|stressed|sad|lonely|heartbroken|devastated|'
            r'exhausted|struggling|difficult|tough|hard|worried|anxious|'
            r'overwhelmed|upset|angry|fear|despair|regret|pain|hurt|grief)\b',
            query_lower
        )
        is_emotional_query = len(emotional_query_words) >= 2
        
        empathy_score = 0.0
        if is_emotional_query:
            empathy_indicators = re.findall(
                r'\b(understand|sorry|hear|feel|natural|okay|normal|valid|'
                r'completely|perfectly|genuinely|truly|acknowledge|recognize|'
                r'it\'s okay|it\'s fine|take your time|breathe|rest|'
                r'don\'t hesitate|you\'re not alone|we\'re here|i\'m here|'
                r'support|care|matter|important|valued)\b',
                response_lower
            )
            empathy_score = min(len(empathy_indicators) / 4.0, 1.5)
            
            # Penalize dismissive responses
            dismissive = re.findall(
                r'\b(just|simply|get over|move on|deal with|stop|don\'t worry|'
                r'no big deal|not a big|calm down|relax)\b',
                response_lower
            )
            empathy_score -= min(len(dismissive) * 0.15, 0.8)
        
        # ============================================================
        # 7. RESPONSE LENGTH ADEQUACY (relative to query complexity)
        # ============================================================
        query_words_count = len(re.findall(r'\w+', query))
        
        # Longer/more complex queries deserve longer responses
        expected_min_words = max(query_words_count * 1.5, 40)
        length_adequacy = min(total_words / expected_min_words, 1.5)
        
        # But also penalize very short responses
        if total_words < 30:
            length_adequacy *= 0.5
        
        # ============================================================
        # 8. ACTIONABILITY — does response give concrete advice/steps?
        # ============================================================
        action_verbs = re.findall(
            r'\b(try|start|begin|consider|ensure|make sure|check|look|'
            r'use|apply|implement|create|build|set up|follow|avoid|'
            r'focus|practice|maintain|develop|improve|explore|reach out|'
            r'contact|ask|seek|break down|tackle|prioritize|schedule)\b',
            response_lower
        )
        actionability = min(len(action_verbs) / 4.0, 1.5)
        
        # ============================================================
        # 9. NEGATIVE SIGNALS — vagueness, uncertainty, incompleteness
        # ============================================================
        vague_phrases = re.findall(
            r'\b(maybe|probably|i guess|i think|not sure|might not|'
            r'can\'t really|hard to say|it depends|who knows|'
            r'something like|sort of|kind of|stuff|things)\b',
            response_lower
        )
        vagueness_penalty = min(len(vague_phrases) * 0.1, 0.8)
        
        # Negation-heavy responses (saying what can't be done rather than what can)
        negations = re.findall(
            r'\b(can\'t|cannot|won\'t|unable|not able|might not|may not|'
            r'don\'t know|no way|impossible)\b',
            response_lower
        )
        negation_penalty = min(len(negations) * 0.12, 0.6)
        
        # ============================================================
        # 10. SEMANTIC COVERAGE — unique topic clusters
        # ============================================================
        # Count distinct "topic" words (longer, more specific words)
        specific_words = [w for w in response_content if len(w) >= 6]
        topic_diversity = len(set(specific_words)) / max(total_words / 10, 1)
        topic_diversity = min(topic_diversity, 1.5)
        
        # ============================================================
        # COMPOSITE SCORING
        # ============================================================
        
        score = 0.0
        
        # Query coverage (0-1.5)
        score += query_coverage * 1.5
        
        # Aspect coverage (0-1.0)
        score += aspect_coverage_score * 1.0
        
        # Information density (0-1.0)
        score += min(detail_density * 0.5, 1.0)
        
        # Structural completeness (0-1.0)
        score += (has_structure * 0.4 + paragraph_score * 0.3 + starter_variety * 0.3)
        
        # Depth (0-1.5)
        score += min(causal_depth * 0.5 + nuance_score * 0.5, 1.5)
        
        # Empathy (0-1.0 when applicable)
        if is_emotional_query:
            score += max(empathy_score, 0) * 0.7
        
        # Length adequacy (0-1.5)
        score += min(length_adequacy, 1.5)
        
        # Actionability (0-1.0)
        score += min(actionability * 0.7, 1.0)
        
        # Topic diversity (0-1.0)
        score += min(topic_diversity * 0.7, 1.0)
        
        # Vocabulary richness bonus (0-0.5)
        score += min(vocabulary_richness * 0.7, 0.5)
        
        # Sentence count bonus (0-0.5)
        score += min(num_sentences / 10.0, 0.5)
        
        # Penalties
        score -= vagueness_penalty
        score -= negation_penalty
        
        # Clamp and scale to 1-5
        raw_max = 11.0  # approximate theoretical max
        normalized = max(score / raw_max, 0.0)
        normalized = min(normalized, 1.0)
        
        # Map to 1-5 scale with some nonlinearity for discrimination
        final_score = 1.0 + 4.0 * (normalized ** 0.85)
        
        # Round to 1 decimal
        final_score = round(final_score, 1)
        
        return max(min(final_score, 5.0), 1.0)
        
    except Exception:
        # Fallback: simple length-based score
        try:
            words = len(str(response).split())
            return max(min(1.0 + words / 50.0, 3.5), 1.0)
        except Exception:
            return 2.5