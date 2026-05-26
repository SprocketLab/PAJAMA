def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Sentence-level coherence chains (entity/topic threading)
    - Discourse marker quality and placement
    - Argument depth estimation via clause analysis
    - Contradiction detection via negation patterns
    - Information density and progression
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        query = query.strip() if query else ""
        
        # ---- Tokenization helpers ----
        def get_sentences(text):
            # Split on sentence-ending punctuation, but handle abbreviations roughly
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sents if len(s.strip()) > 2]
        
        def get_words(text):
            return re.findall(r'[a-zA-Z]+', text.lower())
        
        def get_content_words(words):
            stop = {'the','a','an','is','are','was','were','be','been','being',
                    'have','has','had','do','does','did','will','would','shall',
                    'should','may','might','can','could','i','you','he','she',
                    'it','we','they','me','him','her','us','them','my','your',
                    'his','its','our','their','this','that','these','those',
                    'and','or','but','if','then','else','when','where','how',
                    'what','which','who','whom','whose','why','not','no','nor',
                    'so','too','very','just','also','of','in','on','at','to',
                    'for','with','by','from','as','into','about','between',
                    'through','after','before','during','above','below','up',
                    'down','out','off','over','under','again','further','than',
                    'more','most','some','any','all','each','every','both',
                    'few','many','much','other','another','such','only','own',
                    'same','here','there','now','then'}
            return [w for w in words if w not in stop and len(w) > 1]
        
        sentences = get_sentences(response_stripped)
        all_words = get_words(response_stripped)
        content_words = get_content_words(all_words)
        
        num_sentences = len(sentences)
        num_words = len(all_words)
        
        # ---- 1. ENTITY THREADING SCORE ----
        # Measures how well sentences share entities/topics with their neighbors
        # (not just adjacent but within a window) - coherence chain
        entity_thread_score = 0.0
        if num_sentences >= 2:
            sent_content = []
            for s in sentences:
                w = get_words(s)
                cw = set(get_content_words(w))
                sent_content.append(cw)
            
            chain_scores = []
            for i in range(1, len(sent_content)):
                # Check overlap with previous 1-3 sentences (discourse window)
                window = min(i, 3)
                best_overlap = 0.0
                for j in range(max(0, i - window), i):
                    prev = sent_content[j]
                    curr = sent_content[i]
                    if len(prev) > 0 and len(curr) > 0:
                        # Dice coefficient
                        overlap = 2 * len(prev & curr) / (len(prev) + len(curr))
                        best_overlap = max(best_overlap, overlap)
                chain_scores.append(best_overlap)
            
            if chain_scores:
                entity_thread_score = sum(chain_scores) / len(chain_scores)
                # Penalize if chain breaks completely (score = 0) for many transitions
                zero_chains = sum(1 for s in chain_scores if s < 0.01)
                if len(chain_scores) > 0:
                    break_ratio = zero_chains / len(chain_scores)
                    entity_thread_score *= (1.0 - 0.5 * break_ratio)
        elif num_sentences == 1 and num_words >= 5:
            entity_thread_score = 0.3  # Single sentence, can't measure threading
        
        # ---- 2. DISCOURSE MARKER QUALITY ----
        # Categorize discourse markers by function
        causal_markers = ['because', 'therefore', 'thus', 'hence', 'consequently',
                         'as a result', 'due to', 'since', 'so that', 'for this reason',
                         'accordingly', 'it follows']
        contrast_markers = ['however', 'although', 'nevertheless', 'on the other hand',
                           'in contrast', 'despite', 'yet', 'whereas', 'conversely',
                           'but', 'while', 'even though', 'nonetheless']
        elaboration_markers = ['for example', 'for instance', 'specifically',
                              'in particular', 'such as', 'namely', 'that is',
                              'in other words', 'to illustrate', 'moreover',
                              'furthermore', 'additionally', 'in addition']
        sequence_markers = ['first', 'second', 'third', 'finally', 'next',
                           'then', 'subsequently', 'lastly', 'to begin',
                           'in conclusion', 'to summarize', 'overall']
        
        response_lower = response_stripped.lower()
        
        causal_count = sum(1 for m in causal_markers if m in response_lower)
        contrast_count = sum(1 for m in contrast_markers if m in response_lower)
        elaboration_count = sum(1 for m in elaboration_markers if m in response_lower)
        sequence_count = sum(1 for m in sequence_markers if m in response_lower)
        
        total_markers = causal_count + contrast_count + elaboration_count + sequence_count
        # Variety of marker types used (0-4)
        marker_types = sum(1 for c in [causal_count, contrast_count, elaboration_count, sequence_count] if c > 0)
        
        # Normalize marker density per sentence
        marker_density = total_markers / max(num_sentences, 1)
        # Optimal density is around 0.3-0.8 markers per sentence
        if marker_density <= 0.8:
            density_quality = marker_density / 0.8
        else:
            density_quality = max(0.3, 1.0 - (marker_density - 0.8) * 0.3)
        
        discourse_score = 0.4 * min(density_quality, 1.0) + 0.6 * (marker_types / 4.0)
        
        # ---- 3. ARGUMENT DEPTH via clause complexity ----
        # Count subordinate clauses, conditional structures, etc.
        subordinators = ['because', 'although', 'while', 'whereas', 'since',
                        'unless', 'if', 'when', 'where', 'after', 'before',
                        'until', 'though', 'even if', 'provided that',
                        'in order to', 'so that', 'as long as']
        
        clause_count = 0
        for sub in subordinators:
            clause_count += len(re.findall(r'\b' + re.escape(sub) + r'\b', response_lower))
        
        # Relative clauses
        relative_count = len(re.findall(r'\b(which|that|who|whom|whose)\b', response_lower))
        clause_count += relative_count * 0.5
        
        # Normalize by sentence count
        clause_density = clause_count / max(num_sentences, 1)
        # Optimal is around 0.5-2.0 per sentence
        depth_score = min(clause_density / 1.5, 1.0)
        
        # ---- 4. CONTRADICTION / INCOHERENCE DETECTION ----
        contradiction_penalty = 0.0
        
        # Check for repeated sentences (sign of incoherence/generation failure)
        if num_sentences >= 2:
            sent_normalized = [re.sub(r'\s+', ' ', s.lower().strip()) for s in sentences]
            sent_counter = Counter(sent_normalized)
            repeated = sum(c - 1 for c in sent_counter.values() if c > 1)
            repetition_ratio = repeated / max(num_sentences, 1)
            contradiction_penalty += min(repetition_ratio * 2.0, 0.5)
        
        # Check for negation contradictions (simple heuristic)
        # Look for "X is Y" followed by "X is not Y" patterns
        negation_pairs = re.findall(r'(\b\w+\b) is (?:not |n\'t )', response_lower)
        affirmation_pairs = re.findall(r'(\b\w+\b) is (?!not |n\'t )(\w+)', response_lower)
        for neg_subj in negation_pairs:
            for aff_subj, _ in affirmation_pairs:
                if neg_subj == aff_subj:
                    # Could be contradiction or nuance - mild penalty
                    contradiction_penalty += 0.05
        
        contradiction_penalty = min(contradiction_penalty, 0.6)
        
        # ---- 5. INFORMATION PROGRESSION ----
        # Check if new information is introduced progressively
        progression_score = 0.0
        if num_sentences >= 2:
            cumulative_content = set()
            new_info_ratios = []
            for s in sentences:
                w = get_words(s)
                cw = set(get_content_words(w))
                if len(cw) > 0:
                    new_words = cw - cumulative_content
                    new_ratio = len(new_words) / len(cw)
                    new_info_ratios.append(new_ratio)
                    cumulative_content.update(cw)
            
            if new_info_ratios:
                # Good progression: early sentences introduce lots of new info,
                # later ones still add some but also connect back
                avg_new = sum(new_info_ratios) / len(new_info_ratios)
                # We want a balance: not 100% new (disconnected) and not 0% new (repetitive)
                if avg_new > 0.5:
                    progression_score = 1.0 - (avg_new - 0.5) * 0.6
                else:
                    progression_score = avg_new * 2.0
                progression_score = max(0.0, min(1.0, progression_score))
        elif num_sentences == 1:
            progression_score = 0.3
        
        # ---- 6. RESPONSE COMPLETENESS & STRUCTURE ----
        # Check if response ends properly (not mid-sentence)
        completeness_score = 0.5
        if response_stripped[-1] in '.!?)"\'':
            completeness_score = 1.0
        elif response_stripped[-1] in ',;:':
            completeness_score = 0.2
        elif len(response_stripped) < 20:
            completeness_score = 0.3
        
        # Check for structural elements (lists, paragraphs)
        has_structure = 0.0
        if re.search(r'\n\s*\n', response_stripped):  # Paragraphs
            has_structure += 0.3
        if re.search(r'^\s*[\d\-\*\•]', response_stripped, re.MULTILINE):  # Lists
            has_structure += 0.3
        if re.search(r':\s*\n', response_stripped):  # Colon before list
            has_structure += 0.2
        has_structure = min(has_structure, 0.5)
        
        # ---- 7. RELEVANCE TO QUERY ----
        query_words = set(get_content_words(get_words(query)))
        response_content = set(content_words)
        
        if query_words and response_content:
            relevance = len(query_words & response_content) / max(len(query_words), 1)
        elif not query_words:
            relevance = 0.5
        else:
            relevance = 0.1
        
        # ---- 8. NOISE / GARBAGE DETECTION ----
        noise_penalty = 0.0
        
        # HTML/code in non-code responses
        if '<' in response_stripped and '>' in response_stripped:
            html_tags = re.findall(r'<[^>]+>', response_stripped)
            if len(html_tags) > 3:
                # Check if query asks for HTML
                if 'html' not in query.lower() and 'tag' not in query.lower():
                    noise_penalty += 0.3
        
        # Random code blocks when not asked
        if 'import ' in response_stripped and 'def ' in response_stripped:
            if 'code' not in query.lower() and 'python' not in query.lower() and 'program' not in query.lower():
                noise_penalty += 0.3
        
        # Very short responses (likely low quality)
        length_factor = 1.0
        if num_words < 3:
            length_factor = 0.15
        elif num_words < 8:
            length_factor = 0.35
        elif num_words < 15:
            length_factor = 0.55
        elif num_words < 25:
            length_factor = 0.75
        elif num_words > 300:
            # Very long might be rambling
            length_factor = max(0.7, 1.0 - (num_words - 300) / 1000)
        
        # ---- 9. LOGICAL CONNECTIVE CHAIN ANALYSIS ----
        # Beyond just counting markers, check if they're properly placed
        # (at sentence beginnings or between clauses)
        proper_marker_placement = 0.0
        if total_markers > 0:
            properly_placed = 0
            for s in sentences:
                s_lower = s.lower().strip()
                # Check if sentence starts with a discourse marker
                all_markers = causal_markers + contrast_markers + elaboration_markers + sequence_markers
                for m in all_markers:
                    if s_lower.startswith(m):
                        properly_placed += 1
                        break
                    # Or after a comma (mid-sentence)
                    if ', ' + m in s_lower:
                        properly_placed += 0.5
                        break
            proper_marker_placement = properly_placed / max(total_markers, 1)
        
        # ---- COMBINE SCORES ----
        # Weights emphasizing logical coherence
        raw_score = (
            2.0 * entity_thread_score +      # Topic threading
            1.5 * discourse_score +            # Discourse markers
            1.2 * depth_score +                # Argument complexity
            1.5 * progression_score +          # Information progression
            1.0 * completeness_score +         # Structural completeness
            0.5 * has_structure +              # Formatting structure
            1.0 * relevance +                  # Query relevance
            0.8 * proper_marker_placement      # Marker placement quality
        )
        
        max_possible = 2.0 + 1.5 + 1.2 + 1.5 + 1.0 + 0.5 + 1.0 + 0.8  # = 9.5
        
        # Normalize to 0-10
        normalized = (raw_score / max_possible) * 10.0
        
        # Apply penalties
        normalized *= (1.0 - contradiction_penalty)
        normalized *= (1.0 - noise_penalty)
        normalized *= length_factor
        
        # Ensure bounds
        final_score = max(0.5, min(10.0, normalized))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return middle score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            return 1.0
        except:
            return 2.0