def judging_function(query, response):
    """
    Evaluate response relevance using a query-intent matching approach based on:
    - Semantic field extraction and coverage analysis
    - Question type detection and answer pattern matching
    - Key entity/concept tracking from query to response
    - Discourse coherence signals (connectives, elaboration markers)
    - Penalty for dismissive/deflective language patterns
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
        
        # --- Tokenization helpers ---
        def tokenize(text):
            return re.findall(r'[a-z]+', text.lower())
        
        def get_content_words(tokens):
            stop = {
                'a','an','the','is','are','was','were','be','been','being',
                'have','has','had','do','does','did','will','would','shall','should',
                'may','might','can','could','must','need','dare','ought',
                'i','me','my','mine','we','us','our','ours','you','your','yours',
                'he','him','his','she','her','hers','it','its','they','them','their','theirs',
                'this','that','these','those','what','which','who','whom','whose',
                'where','when','why','how','all','each','every','both','few','more',
                'most','other','some','such','no','nor','not','only','own','same',
                'so','than','too','very','just','because','as','until','while',
                'of','at','by','for','with','about','against','between','through',
                'during','before','after','above','below','to','from','up','down',
                'in','out','on','off','over','under','again','further','then','once',
                'and','but','or','if','because','although','though','since','unless',
                'also','however','therefore','moreover','furthermore','nevertheless',
                'there','here','now','still','already','yet','even','much',
                'get','got','make','made','take','took','go','went','come','came',
                'know','say','said','tell','told','think','see','look','want',
                'give','use','find','thing','things','way','ways','like','well',
                'really','actually','basically','simply','maybe','perhaps',
                'one','two','first','second','new','good','bad','right','left',
                'something','anything','nothing','everything','someone','anyone',
                'people','person','time','day','work','part','keep','let',
            }
            return [t for t in tokens if t not in stop and len(t) > 2]
        
        query_tokens = tokenize(query)
        resp_tokens = tokenize(response)
        query_content = get_content_words(query_tokens)
        resp_content = get_content_words(resp_tokens)
        
        if not query_content:
            query_content = [t for t in query_tokens if len(t) > 2]
        if not resp_content:
            resp_content = [t for t in resp_tokens if len(t) > 2]
        
        # === FEATURE 1: Semantic Field Coverage ===
        # Extract "semantic fields" from query - clusters of related words
        # by looking at co-occurring content words and checking coverage in response
        query_content_set = set(query_content)
        resp_content_set = set(resp_content)
        resp_text_lower = response.lower()
        query_text_lower = query.lower()
        
        # Direct concept coverage: what fraction of query concepts appear in response
        if query_content_set:
            direct_hits = sum(1 for w in query_content_set if w in resp_content_set)
            concept_coverage = direct_hits / len(query_content_set)
        else:
            concept_coverage = 0.0
        
        # Weighted concept coverage (weight by word importance = inverse frequency proxy)
        query_content_freq = Counter(query_content)
        if query_content_freq:
            weighted_hits = 0
            total_weight = 0
            for word, count in query_content_freq.items():
                # Longer, less common words are more important
                weight = math.log(1 + len(word)) * (1 + math.log(1 + count))
                total_weight += weight
                if word in resp_content_set:
                    weighted_hits += weight
            weighted_coverage = weighted_hits / total_weight if total_weight > 0 else 0
        else:
            weighted_coverage = 0.0
        
        # === FEATURE 2: Query Intent Detection & Answer Pattern Matching ===
        intent_score = 0.0
        
        # Detect query type
        is_how_to = bool(re.search(r'\bhow\b.*\b(to|would|can|could|should|do)\b', query_text_lower))
        is_explain = bool(re.search(r'\b(explain|describe|understand|concept|what is|what are)\b', query_text_lower))
        is_emotional = bool(re.search(r'\b(feeling|feel|emotion|stress|sad|lonely|frustrat|heartbr|devastat|comfort|support)\b', query_text_lower))
        is_advice = bool(re.search(r'\b(advice|help|assist|guide|suggest|recommend|cope|handle|manage)\b', query_text_lower))
        is_scenario = bool(re.search(r'\b(scenario|situation|case|imagine|suppose)\b', query_text_lower))
        
        # Check if response matches expected patterns
        if is_how_to:
            # Expect step-like language, instructional markers
            step_markers = len(re.findall(r'\b(first|then|next|after|finally|step|start|begin|now)\b', resp_text_lower))
            action_verbs = len(re.findall(r'\b(add|put|take|make|set|turn|open|close|move|place|heat|cook|mix|stir|pour|cut|grab|get|try|use|apply|create|build|write|check)\b', resp_text_lower))
            intent_score += min(1.0, (step_markers * 0.15 + action_verbs * 0.08))
        
        if is_explain:
            # Expect definitional/explanatory language
            explain_markers = len(re.findall(r'\b(means|refers|defined|concept|essentially|basically|works|because|due to|principle|imagine|think of|like a|similar to|analogy)\b', resp_text_lower))
            intent_score += min(1.0, explain_markers * 0.12)
        
        if is_emotional:
            # Expect empathetic language
            empathy_markers = len(re.findall(r'\b(understand|sorry|hear|feel|natural|okay|valid|normal|tough|hard|difficult|pain|griev|process|allow|permit|perfectly)\b', resp_text_lower))
            dismissive_markers = len(re.findall(r'\b(just get over|move on|stop|don\'t worry|no big deal|get yourself together|shouldn\'t feel)\b', resp_text_lower))
            intent_score += min(1.0, empathy_markers * 0.1) - dismissive_markers * 0.2
        
        if is_advice:
            # Expect actionable suggestions
            advice_markers = len(re.findall(r'\b(try|consider|could|might|suggest|recommend|helpful|tip|strategy|approach|option|way to|method|technique|practice|remember)\b', resp_text_lower))
            intent_score += min(1.0, advice_markers * 0.08)
        
        intent_score = max(0.0, min(2.0, intent_score))
        
        # === FEATURE 3: Key Entity/Concept Tracking ===
        # Extract likely key entities from query (capitalized words, quoted terms, domain terms)
        # Also look for noun-like longer words that are likely domain-specific
        domain_words = [w for w in query_content if len(w) >= 5]
        
        if domain_words:
            domain_in_response = sum(1 for w in set(domain_words) if w in resp_content_set)
            entity_coverage = domain_in_response / len(set(domain_words))
        else:
            entity_coverage = concept_coverage  # fallback
        
        # Check for proper nouns / special terms in query
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        # Filter out sentence starters (rough heuristic)
        proper_nouns = [p for p in proper_nouns if len(p) > 3]
        if proper_nouns:
            pn_hits = sum(1 for pn in proper_nouns if pn.lower() in resp_text_lower)
            pn_coverage = pn_hits / len(proper_nouns)
            entity_coverage = (entity_coverage + pn_coverage) / 2
        
        # === FEATURE 4: Discourse Coherence & Elaboration ===
        # Measure how well-structured and elaborated the response is
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Coherence connectives
        connectives = len(re.findall(
            r'\b(however|therefore|moreover|furthermore|additionally|consequently|'
            r'in addition|as a result|for instance|for example|in other words|'
            r'on the other hand|that said|meanwhile|similarly|likewise|'
            r'specifically|particularly|importantly|notably|indeed|'
            r'first|second|third|finally|also|besides|yet|still|thus|hence)\b',
            resp_text_lower
        ))
        
        coherence_score = min(1.0, connectives * 0.08 + min(1.0, num_sentences / 6) * 0.3)
        
        # Numbered/bulleted lists (structured advice)
        has_structure = bool(re.search(r'(\d+[\.\):]|\n\s*[-•*])', response))
        if has_structure:
            coherence_score += 0.15
        
        coherence_score = min(1.0, coherence_score)
        
        # === FEATURE 5: Responsiveness vs Deflection/Dismissiveness ===
        # Penalize responses that seem dismissive or off-topic
        deflection_penalty = 0.0
        
        # Dismissive phrases
        dismissive_patterns = [
            r'\bjust\s+(get|move|stop|do|try)\b',
            r'\bit\'s\s+(just|only|no big)\b',
            r'\byou\s+should\s+be\s+able\b',
            r'\bmaybe\s+you\'re\s+(just|not)\b',
            r'\bget yourself together\b',
            r'\bnothing wrong with\b',
        ]
        for pat in dismissive_patterns:
            if re.search(pat, resp_text_lower):
                deflection_penalty += 0.08
        
        # Check if response introduces lots of content NOT related to query
        if resp_content:
            resp_unique = set(resp_content)
            query_related = sum(1 for w in resp_unique if w in query_content_set or 
                              any(w.startswith(qw[:4]) or qw.startswith(w[:4]) 
                                  for qw in query_content_set if len(qw) >= 4 and len(w) >= 4))
            relatedness_ratio = query_related / len(resp_unique) if resp_unique else 0
        else:
            relatedness_ratio = 0.0
        
        # === FEATURE 6: Substring/Phrase Overlap ===
        # Check for multi-word phrase matches (bigrams, trigrams from query in response)
        def get_ngrams(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        query_bigrams = set(get_ngrams(query_tokens, 2))
        resp_bigrams = set(get_ngrams(resp_tokens, 2))
        query_trigrams = set(get_ngrams(query_tokens, 3))
        resp_trigrams = set(get_ngrams(resp_tokens, 3))
        
        bigram_overlap = len(query_bigrams & resp_bigrams) / len(query_bigrams) if query_bigrams else 0
        trigram_overlap = len(query_trigrams & resp_trigrams) / len(query_trigrams) if query_trigrams else 0
        
        phrase_score = bigram_overlap * 0.6 + trigram_overlap * 0.4
        
        # === FEATURE 7: Response Depth/Substance ===
        # A good response should have adequate length and depth
        resp_word_count = len(resp_tokens)
        depth_score = min(1.0, resp_word_count / 80)  # Expect at least ~80 words for full answer
        
        # Penalize very short responses unless query is simple
        if resp_word_count < 30 and len(query_tokens) > 15:
            depth_score *= 0.5
        
        # === FEATURE 8: Tone Alignment ===
        # Check if response tone matches what query expects
        tone_score = 0.5  # neutral default
        
        # Casual query detection
        casual_markers_query = len(re.findall(r'\b(hey|cool|awesome|gonna|wanna|gotta|yeah|nah|dude|bro|chill|vibe|killer|whip|grab|stuff)\b', query_text_lower))
        casual_markers_resp = len(re.findall(r'\b(hey|cool|awesome|gonna|wanna|gotta|yeah|nah|dude|bro|chill|vibe|killer|whip|grab|stuff|alright|let\'s)\b', resp_text_lower))
        
        formal_markers_query = len(re.findall(r'\b(furthermore|therefore|consequently|moreover|regarding|concerning|shall|hereby)\b', query_text_lower))
        formal_markers_resp = len(re.findall(r'\b(furthermore|therefore|consequently|moreover|regarding|concerning|shall|hereby)\b', resp_text_lower))
        
        if casual_markers_query > 1:
            tone_score = min(1.0, 0.3 + casual_markers_resp * 0.15)
        elif formal_markers_query > 0:
            tone_score = min(1.0, 0.3 + formal_markers_resp * 0.15)
        else:
            tone_score = 0.6  # neutral is fine for most
        
        # === COMBINE ALL FEATURES ===
        # Weighted combination
        raw_score = (
            concept_coverage * 1.8 +          # Core relevance
            weighted_coverage * 1.5 +          # Weighted concept match
            entity_coverage * 1.2 +            # Key entity tracking
            intent_score * 1.3 +               # Intent matching
            phrase_score * 0.8 +               # Phrase-level overlap
            coherence_score * 0.7 +            # Discourse quality
            relatedness_ratio * 1.0 +          # Topic focus
            depth_score * 0.6 +                # Adequate depth
            tone_score * 0.4 -                 # Tone alignment
            deflection_penalty * 2.0           # Dismissiveness penalty
        )
        
        # Normalize to 1-5 scale
        # Theoretical max ~9.3, typical good ~5-7, typical bad ~1-3
        max_theoretical = 9.3
        normalized = (raw_score / max_theoretical) * 4.0 + 1.0
        
        # Apply sigmoid-like stretching for better discrimination
        # Center around 3.0
        centered = normalized - 3.0
        stretched = 3.0 + 2.0 * math.tanh(centered * 0.8)
        
        # Clamp to [0.5, 5.5]
        final_score = max(0.5, min(5.5, stretched))
        
        return round(final_score, 2)
        
    except Exception:
        try:
            # Minimal fallback
            query_words = set(re.findall(r'[a-z]+', query.lower()))
            resp_words = set(re.findall(r'[a-z]+', response.lower()))
            if query_words:
                overlap = len(query_words & resp_words) / len(query_words)
                return round(1.0 + overlap * 4.0, 2)
            return 2.5
        except Exception:
            return 2.5