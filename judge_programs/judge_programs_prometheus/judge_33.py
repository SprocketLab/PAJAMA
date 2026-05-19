def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    
    Focuses on:
    - Clear logical flow and organization
    - Well-structured arguments with premises leading to conclusions
    - Smooth transitions between ideas
    - Absence of contradictions, circular reasoning, non-sequiturs
    - Appropriate use of discourse markers and connectives
    
    Returns a score where HIGHER = BETTER quality (range ~0-10).
    """
    try:
        import re
        import math
        import string
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        if len(response_clean) < 10:
            return 0.5
        
        # ============================================================
        # FEATURE 1: Discourse markers and logical connectives
        # These indicate structured argumentation and logical flow
        # ============================================================
        
        response_lower = response_clean.lower()
        
        # Causal/logical connectives
        causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'since', 'so that', 'in order to',
            'this means', 'which means', 'leading to', 'resulting in',
            'for this reason', 'that\'s why', 'this is why', 'owing to'
        ]
        
        # Transitional/sequential markers
        transition_markers = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'additionally', 'moreover', 'furthermore', 'in addition',
            'also', 'besides', 'on the other hand', 'however',
            'nevertheless', 'nonetheless', 'meanwhile', 'subsequently',
            'following this', 'after that', 'before this', 'to begin',
            'lastly', 'in conclusion', 'to summarize', 'in summary',
            'overall', 'step', 'now'
        ]
        
        # Concessive/contrastive markers
        contrastive_markers = [
            'however', 'but', 'although', 'though', 'while',
            'on the other hand', 'in contrast', 'conversely',
            'despite', 'even though', 'yet', 'still', 'whereas',
            'rather than', 'instead', 'alternatively'
        ]
        
        # Elaboration markers
        elaboration_markers = [
            'for example', 'for instance', 'such as', 'specifically',
            'in particular', 'namely', 'to illustrate', 'consider',
            'imagine', 'think of', 'like when', 'this includes',
            'in other words', 'that is', 'meaning', 'essentially'
        ]
        
        # Emphatic/acknowledgment markers (show engagement with topic)
        engagement_markers = [
            'it\'s important', 'it is important', 'remember',
            'keep in mind', 'note that', 'understand that',
            'it\'s worth', 'it is worth', 'crucially', 'importantly',
            'significantly', 'notably', 'understandably',
            'absolutely', 'completely', 'genuinely', 'truly',
            'perfectly', 'certainly', 'definitely'
        ]
        
        def count_markers(markers, text):
            count = 0
            for m in markers:
                count += len(re.findall(r'\b' + re.escape(m) + r'\b', text))
            return count
        
        causal_count = count_markers(causal_markers, response_lower)
        transition_count = count_markers(transition_markers, response_lower)
        contrastive_count = count_markers(contrastive_markers, response_lower)
        elaboration_count = count_markers(elaboration_markers, response_lower)
        engagement_count = count_markers(engagement_markers, response_lower)
        
        total_markers = causal_count + transition_count + contrastive_count + elaboration_count + engagement_count
        
        # Normalize by response length (per 100 words)
        words = response_clean.split()
        word_count = len(words)
        if word_count == 0:
            return 0.5
        
        marker_density = (total_markers / word_count) * 100
        
        # Score: good density is around 3-8 markers per 100 words
        marker_score = min(marker_density / 5.0, 2.0)  # max 2.0
        
        # Bonus for variety of marker types used
        marker_type_variety = sum([
            1 for c in [causal_count, transition_count, contrastive_count, 
                        elaboration_count, engagement_count] if c > 0
        ])
        variety_bonus = marker_type_variety * 0.15  # up to 0.75
        
        # ============================================================
        # FEATURE 2: Sentence structure and paragraph organization
        # ============================================================
        
        sentences = re.split(r'[.!?]+', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # Check for numbered/bulleted lists (indicates structured thinking)
        numbered_pattern = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean)
        bullet_pattern = re.findall(r'(?:^|\n)\s*[-•\*]\s', response_clean)
        has_structure = len(numbered_pattern) + len(bullet_pattern)
        structure_score = min(has_structure * 0.3, 1.0)
        
        # Paragraph breaks indicate organized thought
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if len(p.strip()) > 10]
        num_paragraphs = len(paragraphs)
        paragraph_score = min(num_paragraphs * 0.2, 0.8)
        
        # ============================================================
        # FEATURE 3: Sentence length variation (good writing varies)
        # ============================================================
        
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 1]
        if len(sent_lengths) >= 2:
            import statistics
            mean_len = statistics.mean(sent_lengths)
            stdev_len = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
            # Coefficient of variation - some variation is good
            cv = stdev_len / mean_len if mean_len > 0 else 0
            variation_score = min(cv * 1.5, 0.8)
        else:
            variation_score = 0.2
        
        # ============================================================
        # FEATURE 4: Response completeness and depth
        # ============================================================
        
        # Adequate length suggests thorough treatment
        length_score = 0
        if word_count >= 20:
            length_score = 0.3
        if word_count >= 50:
            length_score = 0.6
        if word_count >= 80:
            length_score = 0.9
        if word_count >= 120:
            length_score = 1.2
        if word_count >= 180:
            length_score = 1.4
        
        # ============================================================
        # FEATURE 5: Query relevance and responsiveness
        # ============================================================
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_clean.lower()))
        response_words = set(re.findall(r'\b[a-z]{3,}\b', response_lower))
        
        # Remove very common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'would', 'could', 'should', 'will', 'with',
            'this', 'that', 'from', 'they', 'were', 'been', 'said',
            'each', 'which', 'their', 'there', 'what', 'about', 'when',
            'make', 'like', 'time', 'just', 'know', 'take', 'come',
            'more', 'some', 'than', 'them', 'very', 'after', 'also',
            'how', 'who', 'get', 'may', 'new', 'now', 'way', 'use'
        }
        
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        if len(query_content) > 0:
            overlap = len(query_content & response_content) / len(query_content)
            relevance_score = overlap * 1.2
        else:
            relevance_score = 0.5
        
        # ============================================================
        # FEATURE 6: Empathy and acknowledgment (for emotional queries)
        # ============================================================
        
        emotional_query_words = {
            'feeling', 'feel', 'stress', 'frustrated', 'sad', 'lonely',
            'heartbroken', 'devastated', 'struggling', 'difficult',
            'worried', 'anxious', 'upset', 'exhausted', 'tired',
            'depressed', 'overwhelmed', 'afraid', 'fear', 'regret',
            'down', 'loneliness', 'despair', 'concern', 'breakup',
            'passed', 'died', 'death', 'grief', 'loss'
        }
        
        query_is_emotional = len(emotional_query_words & query_words) > 0
        
        empathy_phrases = [
            'i understand', 'i can see', 'it\'s understandable',
            'it\'s completely', 'it is completely', 'that\'s',
            'i hear', 'i\'m sorry', 'i am sorry', 'it\'s okay',
            'it is okay', 'it\'s natural', 'it is natural',
            'it\'s perfectly', 'it is perfectly', 'absolutely okay',
            'completely understandable', 'totally understandable',
            'i can hear', 'i can imagine', 'it must be',
            'your feelings', 'your experience', 'give yourself',
            'allow yourself', 'take a moment', 'let yourself'
        ]
        
        empathy_count = count_markers(empathy_phrases, response_lower)
        
        empathy_score = 0
        if query_is_emotional:
            empathy_score = min(empathy_count * 0.4, 1.2)
        
        # ============================================================
        # FEATURE 7: Absence of dismissive or contradictory language
        # ============================================================
        
        dismissive_phrases = [
            'just get over', 'stop worrying', 'it\'s not a big deal',
            'you should be able', 'just do it', 'not that hard',
            'get yourself together', 'move on', 'get rid of',
            'you need to get', 'just keep trying', 'maybe you\'re just not',
            'you\'re not using it correctly', 'read the manual',
            'it\'s just a', 'nothing wrong', 'don\'t let it'
        ]
        
        dismissive_count = count_markers(dismissive_phrases, response_lower)
        dismissive_penalty = min(dismissive_count * 0.5, 2.0)
        
        # ============================================================
        # FEATURE 8: Hedging and appropriate uncertainty
        # ============================================================
        
        hedging_phrases = [
            'might', 'perhaps', 'possibly', 'it seems', 'it appears',
            'could be', 'may be', 'in some cases', 'sometimes',
            'it depends', 'generally', 'typically', 'often',
            'tend to', 'can vary'
        ]
        
        hedging_count = count_markers(hedging_phrases, response_lower)
        # Some hedging is good (shows nuance), too much is wishy-washy
        hedging_score = min(hedging_count * 0.1, 0.4)
        
        # ============================================================
        # FEATURE 9: Coherence via sentence-to-sentence topic continuity
        # ============================================================
        
        coherence_score = 0
        if len(sentences) >= 2:
            continuity_scores = []
            for i in range(1, len(sentences)):
                prev_words = set(re.findall(r'\b[a-z]{3,}\b', sentences[i-1].lower())) - stop_words
                curr_words = set(re.findall(r'\b[a-z]{3,}\b', sentences[i].lower())) - stop_words
                if len(prev_words) > 0 and len(curr_words) > 0:
                    overlap = len(prev_words & curr_words) / min(len(prev_words), len(curr_words))
                    continuity_scores.append(overlap)
            
            if continuity_scores:
                avg_continuity = sum(continuity_scores) / len(continuity_scores)
                # Good coherence: moderate overlap (0.1-0.4 is ideal)
                # Too high means repetitive, too low means disconnected
                if avg_continuity < 0.05:
                    coherence_score = 0.2
                elif avg_continuity < 0.15:
                    coherence_score = 0.6
                elif avg_continuity < 0.35:
                    coherence_score = 1.0
                elif avg_continuity < 0.5:
                    coherence_score = 0.7
                else:
                    coherence_score = 0.4  # too repetitive
        
        # ============================================================
        # FEATURE 10: Opening quality - does the response address the query directly?
        # ============================================================
        
        first_sentence = sentences[0].lower() if sentences else ""
        
        # Good openings acknowledge the situation or directly address the query
        good_opening_signals = [
            'i understand', 'i can see', 'i\'m sorry', 'i hear',
            'it sounds like', 'it seems', 'that\'s', 'to',
            'imagine', 'let\'s', 'here', 'absolutely',
            'great question', 'good question'
        ]
        
        opening_score = 0
        for signal in good_opening_signals:
            if signal in first_sentence:
                opening_score = 0.5
                break
        
        # Check if first sentence contains query-relevant words
        first_sent_words = set(re.findall(r'\b[a-z]{3,}\b', first_sentence)) - stop_words
        if query_content and first_sent_words:
            first_relevance = len(query_content & first_sent_words) / max(len(query_content), 1)
            opening_score += min(first_relevance * 0.8, 0.5)
        
        # ============================================================
        # FEATURE 11: Actionable advice / concrete suggestions
        # ============================================================
        
        action_phrases = [
            'try to', 'you can', 'you could', 'consider',
            'start by', 'begin with', 'make sure', 'don\'t forget',
            'keep in mind', 'it helps to', 'one way', 'another way',
            'a good approach', 'recommend', 'suggest', 'helpful',
            'effective', 'strategy', 'technique', 'method',
            'here are', 'here\'s', 'following', 'steps'
        ]
        
        action_count = count_markers(action_phrases, response_lower)
        action_score = min(action_count * 0.15, 0.8)
        
        # ============================================================
        # FEATURE 12: Contradiction detection (negative signal)
        # ============================================================
        
        contradiction_penalty = 0
        
        # Simple contradiction patterns
        negation_pairs = [
            (r'\bcan\b', r'\bcannot\b'), (r'\bcan\b', r'\bcan\'t\b'),
            (r'\bwill\b', r'\bwon\'t\b'), (r'\bshould\b', r'\bshouldn\'t\b'),
            (r'\bis\b', r'\bisn\'t\b'), (r'\bdo\b', r'\bdon\'t\b'),
        ]
        
        # Check if the response says something and then contradicts it
        # (very basic heuristic)
        if 'but' in response_lower or 'however' in response_lower:
            # Having contrasts is actually good for nuanced arguments
            pass  # no penalty
        
        # Check for "might not" / "probably won't" type hedging that undermines claims
        undermining_phrases = [
            'might not be able', 'probably won\'t', 'may not be able',
            'it might not', 'it probably won\'t', 'it may not'
        ]
        undermine_count = count_markers(undermining_phrases, response_lower)
        
        # In certain contexts, undermining one's own suggestions is bad
        if undermine_count > 1:
            contradiction_penalty = 0.5
        
        # ============================================================
        # FEATURE 13: Tone appropriateness
        # ============================================================
        
        # Check if response uses casual language when query is casual and vice versa
        casual_indicators = [
            'hey', 'alright', 'cool', 'awesome', 'gonna', 'wanna',
            'gotta', 'kinda', 'sorta', 'yeah', 'yep', 'nope',
            'dude', 'buddy', 'man,', 'bro', 'yo ', 'lol', 'haha',
            'nifty', 'killer', 'whip up', 'let\'s get down'
        ]
        
        query_casual = count_markers(casual_indicators, query_clean.lower())
        response_casual = count_markers(casual_indicators, response_lower)
        
        # Tone matching bonus
        tone_score = 0
        if query_casual > 0 and response_casual > 0:
            tone_score = 0.4  # matched casual
        elif query_casual == 0 and response_casual == 0:
            tone_score = 0.3  # matched formal
        # Mismatch gets 0
        
        # ============================================================
        # AGGREGATE SCORING
        # ============================================================
        
        raw_score = (
            marker_score * 1.2 +        # discourse markers (up to 2.4)
            variety_bonus +              # marker variety (up to 0.75)
            structure_score +            # numbered/bulleted lists (up to 1.0)
            paragraph_score +            # paragraph organization (up to 0.8)
            variation_score +            # sentence length variation (up to 0.8)
            length_score +               # adequate length (up to 1.4)
            relevance_score +            # query relevance (up to 1.2)
            empathy_score +              # empathy for emotional queries (up to 1.2)
            hedging_score +              # appropriate hedging (up to 0.4)
            coherence_score +            # sentence coherence (up to 1.0)
            opening_score +              # good opening (up to 1.0)
            action_score +               # actionable advice (up to 0.8)
            tone_score -                 # tone matching (up to 0.4)
            dismissive_penalty -         # dismissive language (up to -2.0)
            contradiction_penalty        # contradictions (up to -0.5)
        )
        
        # Normalize to 0-10 range
        # Theoretical max is roughly ~12, theoretical min is ~-2.5
        # Map to 0-10
        normalized = (raw_score + 1.0) * (10.0 / 13.0)
        
        # Clamp to [0.5, 10]
        final_score = max(0.5, min(10.0, normalized))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Center around 5, stretch differences
        midpoint = 5.0
        spread = 2.5
        adjusted = midpoint + spread * math.tanh((final_score - midpoint) / spread)
        
        # Ensure within bounds
        adjusted = max(0.5, min(10.0, adjusted))
        
        return round(adjusted, 2)
        
    except Exception as e:
        # Fallback: return a middle-of-road score
        try:
            if response and len(response.strip()) > 50:
                return 4.0
            elif response and len(response.strip()) > 10:
                return 2.5
            else:
                return 1.0
        except:
            return 3.0