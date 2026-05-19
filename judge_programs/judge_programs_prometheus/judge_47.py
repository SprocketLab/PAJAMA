def judging_function(query, response):
    """
    Evaluates clarity and conciseness using information-theoretic and structural coherence metrics.
    
    Algorithm focus: 
    - Information density (unique information per token ratio)
    - Sentence-level coherence via topic continuity (shared content words between adjacent sentences)
    - Filler/weasel word density
    - Precision of language (specific vs vague word ratio)
    - Response-query alignment (relevance without bloat)
    - Compression ratio (how much meaning per character)
    - Clause complexity estimation
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 1.0
        
        # Tokenize
        words = re.findall(r'[a-zA-Z]+', response.lower())
        if len(words) < 3:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        if not sentences:
            return 1.0
        
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        
        # ---- Feature 1: Information Entropy per word ----
        # Higher entropy = more diverse vocabulary = less repetitive
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize entropy by log2(total_words) to get relative entropy
        max_entropy = math.log2(total_words) if total_words > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        # Score: 0-10
        entropy_score = normalized_entropy * 10.0
        
        # ---- Feature 2: Topic Continuity (coherence between adjacent sentences) ----
        # Measures how well ideas flow by checking content word overlap between adjacent sentences
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
            'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
            'them', 'their', 'theirs', 'themselves', 'about', 'up', 'also',
            'there', 'here', 'while', 'although', 'though', 'even', 'still',
            'already', 'much', 'many', 'well', 'back', 'also', 'get', 'got',
            'like', 'make', 'go', 'going', 'know', 'take', 'come', 'think',
            'see', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work',
            'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin',
            'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'hold',
            'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
            'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change',
            'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
            'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win',
            'offer', 'remember', 'consider', 'appear', 'buy', 'wait', 'serve',
            'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach',
            'kill', 'remain', 'don', 't', 's', 're', 've', 'll', 'd', 'm',
        }
        
        def content_words(text):
            ws = re.findall(r'[a-zA-Z]+', text.lower())
            return set(w for w in ws if w not in stop_words and len(w) > 2)
        
        if len(sentences) >= 2:
            continuity_scores = []
            for i in range(len(sentences) - 1):
                cw1 = content_words(sentences[i])
                cw2 = content_words(sentences[i + 1])
                if cw1 and cw2:
                    overlap = len(cw1 & cw2)
                    union = len(cw1 | cw2)
                    # We want moderate overlap - too much = repetitive, too little = incoherent
                    ratio = overlap / union if union > 0 else 0
                    # Optimal around 0.15-0.35
                    if ratio <= 0.25:
                        cont_score = ratio / 0.25
                    else:
                        # Penalize excessive overlap (repetition)
                        cont_score = max(0, 1.0 - (ratio - 0.25) / 0.75)
                    continuity_scores.append(cont_score)
                else:
                    continuity_scores.append(0.3)  # neutral
            
            coherence_score = (sum(continuity_scores) / len(continuity_scores)) * 10.0
        else:
            coherence_score = 5.0  # single sentence, neutral
        
        # ---- Feature 3: Filler and Weasel Word Density ----
        filler_patterns = [
            r'\bbasically\b', r'\bactually\b', r'\bliterally\b', r'\bkind of\b',
            r'\bsort of\b', r'\byou know\b', r'\bi mean\b', r'\blike\b(?!\s+(?:a|the|this|that|an))',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b', r'\bsomewhat\b',
            r'\bquite\b', r'\brather\b', r'\bpretty much\b', r'\bmore or less\b',
            r'\bin a way\b', r'\bto be honest\b', r'\bhonestly\b', r'\bfrankly\b',
            r'\bat the end of the day\b', r'\bit is what it is\b',
            r'\bneedless to say\b', r'\bit goes without saying\b',
            r'\bas a matter of fact\b', r'\bin other words\b',
            r'\bthat being said\b', r'\bhaving said that\b',
            r'\banyway\b', r'\banyhow\b', r'\bwhatever\b',
            r'\bmight not\b', r'\bmight\b', r'\bcould be\b',
        ]
        
        filler_count = 0
        response_lower = response.lower()
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, response_lower))
        
        filler_density = filler_count / total_words if total_words > 0 else 0
        # Lower density is better
        filler_score = max(0, 10.0 - filler_density * 150)
        
        # ---- Feature 4: Specificity / Concreteness Proxy ----
        # Count numbers, proper nouns (capitalized words not at sentence start), 
        # and specific action verbs vs vague language
        
        vague_words = {
            'thing', 'things', 'stuff', 'something', 'somehow', 'somewhere',
            'anything', 'everything', 'nothing', 'someone', 'anyone', 'everyone',
            'nobody', 'whatever', 'whichever', 'wherever', 'whenever', 'however',
            'good', 'bad', 'nice', 'great', 'fine', 'ok', 'okay', 'interesting',
            'cool', 'awesome', 'terrible', 'horrible', 'wonderful', 'amazing',
            'lot', 'lots', 'bunch', 'tons', 'bit', 'little', 'big', 'small',
            'very', 'really', 'extremely', 'incredibly', 'absolutely',
            'definitely', 'certainly', 'obviously', 'clearly',
            'etc', 'etcetera', 'whatnot',
        }
        
        vague_count = sum(1 for w in words if w in vague_words)
        vague_density = vague_count / total_words if total_words > 0 else 0
        
        # Count numbers (specificity indicator)
        number_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', response))
        number_bonus = min(number_count * 0.3, 2.0)
        
        specificity_score = max(0, 10.0 - vague_density * 100) + number_bonus
        specificity_score = min(specificity_score, 10.0)
        
        # ---- Feature 5: Sentence Complexity Distribution (Clause estimation) ----
        # Count commas, conjunctions, relative pronouns per sentence as proxy for clause count
        clause_markers = [
            r',', r'\band\b', r'\bbut\b', r'\bor\b', r'\bwhich\b', r'\bthat\b',
            r'\bwho\b', r'\bwhere\b', r'\bwhen\b', r'\bbecause\b', r'\balthough\b',
            r'\bwhile\b', r'\bif\b', r'\bsince\b', r'\bunless\b',
        ]
        
        clause_counts = []
        for sent in sentences:
            clause_count = 1  # base clause
            for marker in clause_markers:
                clause_count += len(re.findall(marker, sent.lower()))
            sent_words = len(re.findall(r'[a-zA-Z]+', sent))
            if sent_words > 0:
                # Clauses per word - moderate complexity is ideal
                complexity = clause_count / sent_words
                clause_counts.append(complexity)
        
        if clause_counts:
            avg_complexity = sum(clause_counts) / len(clause_counts)
            # Optimal complexity around 0.08-0.15
            if avg_complexity <= 0.12:
                complexity_score = (avg_complexity / 0.12) * 10.0
            else:
                complexity_score = max(0, 10.0 - (avg_complexity - 0.12) * 60)
        else:
            complexity_score = 5.0
        
        # ---- Feature 6: Query-Response Alignment (relevance without bloat) ----
        query_content = content_words(query)
        response_content = content_words(response)
        
        if query_content and response_content:
            # What fraction of query content words appear in response
            coverage = len(query_content & response_content) / len(query_content)
            # Penalize if response has way more unique content words than query
            # (might indicate tangential rambling)
            expansion_ratio = len(response_content) / max(len(query_content), 1)
            
            # Ideal expansion: 1.5-4x the query content words
            if expansion_ratio <= 3.0:
                expansion_penalty = 0
            else:
                expansion_penalty = min((expansion_ratio - 3.0) * 0.5, 3.0)
            
            alignment_score = coverage * 10.0 - expansion_penalty
            alignment_score = max(0, min(10.0, alignment_score))
        else:
            alignment_score = 5.0
        
        # ---- Feature 7: Structural Clarity Signals ----
        # Numbered lists, colons for definitions, direct address
        structural_score = 5.0  # baseline
        
        # Numbered/bulleted lists (clear structure)
        list_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|[-•*])\s+', response))
        if list_items >= 2:
            structural_score += min(list_items * 0.7, 3.0)
        
        # Colons used for definitions/explanations
        colon_count = response.count(':')
        if 1 <= colon_count <= 5:
            structural_score += 0.5
        
        # Direct address ("you", "your") - engagement
        direct_address = len(re.findall(r'\byou(?:r|rs|rself)?\b', response_lower))
        if direct_address > 0:
            structural_score += min(direct_address * 0.15, 1.0)
        
        # Penalize excessive exclamation marks (unprofessional)
        exclamation_count = response.count('!')
        if exclamation_count > 3:
            structural_score -= min((exclamation_count - 3) * 0.3, 2.0)
        
        structural_score = max(0, min(10.0, structural_score))
        
        # ---- Feature 8: Redundancy Detection via Trigram Repetition ----
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
            trigram_repetition_rate = repeated_trigrams / len(trigrams) if trigrams else 0
            redundancy_score = max(0, 10.0 - trigram_repetition_rate * 80)
        else:
            redundancy_score = 5.0
        
        # ---- Feature 9: Empathy/Engagement Markers (for appropriate queries) ----
        # Detect if query is emotional/seeking comfort
        emotional_query_words = {'feeling', 'feel', 'stressed', 'frustrated', 'sad', 'lonely',
                                  'heartbroken', 'devastated', 'struggling', 'worried', 'anxious',
                                  'upset', 'angry', 'confused', 'overwhelmed', 'exhausted',
                                  'regret', 'fear', 'pain', 'hurt', 'down', 'difficult'}
        
        query_words_set = set(re.findall(r'[a-zA-Z]+', query.lower()))
        is_emotional_query = len(query_words_set & emotional_query_words) >= 2
        
        empathy_markers = [
            r'\bi understand\b', r'\bi can see\b', r'\bi hear\b', r'\bi\'m sorry\b',
            r'\bunderstandable\b', r'\bcompletely\b', r'\babsolutely\b',
            r'\bit\'s okay\b', r'\bit\'s ok\b', r'\bit\'s fine\b', r'\bit\'s natural\b',
            r'\bit\'s normal\b', r'\bperfectly\b', r'\bvalid\b',
        ]
        
        empathy_count = sum(1 for p in empathy_markers if re.search(p, response_lower))
        
        if is_emotional_query:
            empathy_score = min(empathy_count * 2.0, 8.0) + 2.0
        else:
            empathy_score = 5.0  # neutral for non-emotional queries
        
        empathy_score = min(10.0, empathy_score)
        
        # ---- Feature 10: Appropriate Length ----
        # Not too short (unhelpful) or too long (bloated)
        # Optimal: roughly 50-200 words for most responses
        if total_words < 20:
            length_score = total_words / 20 * 4.0
        elif total_words <= 50:
            length_score = 4.0 + (total_words - 20) / 30 * 3.0
        elif total_words <= 200:
            length_score = 7.0 + (min(total_words, 150) - 50) / 100 * 3.0
        elif total_words <= 300:
            length_score = 10.0 - (total_words - 200) / 100 * 2.0
        else:
            length_score = max(3.0, 8.0 - (total_words - 300) / 200 * 3.0)
        
        length_score = max(0, min(10.0, length_score))
        
        # ---- Combine all features with weights ----
        weights = {
            'entropy': 0.10,
            'coherence': 0.10,
            'filler': 0.12,
            'specificity': 0.10,
            'complexity': 0.08,
            'alignment': 0.12,
            'structural': 0.10,
            'redundancy': 0.08,
            'empathy': 0.10,
            'length': 0.10,
        }
        
        scores = {
            'entropy': entropy_score,
            'coherence': coherence_score,
            'filler': filler_score,
            'specificity': specificity_score,
            'complexity': complexity_score,
            'alignment': alignment_score,
            'structural': structural_score,
            'redundancy': redundancy_score,
            'empathy': empathy_score,
            'length': length_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 1-5 range to match examples
        final_score = 1.0 + (final_score / 10.0) * 4.0
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
    
    except Exception as e:
        return 3.0  # Safe default