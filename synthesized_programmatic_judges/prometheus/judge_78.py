def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a sentence-level analysis approach.
    
    This variant analyzes responses at the sentence level, scoring each sentence for
    information density, then aggregates. It also uses a unique approach of measuring
    "specificity gradients" - how much each sentence adds new specific information
    beyond what previous sentences established.
    
    Different from other variants by focusing on:
    1. Sentence-level information density scoring
    2. Specificity gradient (incremental information gain per sentence)
    3. Ratio of substantive vs filler sentences
    4. Precision of language (avoiding weasel words, measuring definitive statements)
    5. Structural coherence signals (numbered steps, cause-effect, conditional logic)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        import re
        import math
        from collections import Counter
        
        response_clean = response.strip()
        if len(response_clean) < 10:
            return 0.5
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'(?<=[.!?])\s+|(?<=:)\s*\n|\n{1,}', response_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 1.0
        
        # ---- Feature 1: Sentence-level information density ----
        # Score each sentence for how much concrete information it carries
        
        def sentence_info_density(sent):
            """Score a single sentence for information density."""
            score = 0.0
            words = sent.split()
            if not words:
                return 0.0
            
            # Numbers and quantities
            numbers = re.findall(r'\b\d+[\d.,]*\b', sent)
            score += len(numbers) * 1.5
            
            # Proper nouns (capitalized words not at sentence start)
            proper_nouns = re.findall(r'(?<!^)(?<!\. )(?<!\n)\b[A-Z][a-z]+\b', sent)
            score += len(proper_nouns) * 1.0
            
            # Specific measurements/units
            units = re.findall(r'\b\d+\s*(?:pounds?|lbs?|oz|ounces?|cups?|tbsp|tsp|minutes?|hours?|days?|weeks?|months?|years?|percent|%|degrees?|miles?|km|meters?|feet|inches?|gallons?|liters?|mg|kg|gb|mb|tb)\b', sent, re.IGNORECASE)
            score += len(units) * 2.0
            
            # Technical/domain terms (longer words tend to be more specific)
            technical_words = [w for w in words if len(w) > 8]
            score += len(technical_words) * 0.5
            
            # Quoted terms or emphasized terms
            quoted = re.findall(r'["\']([^"\']+)["\']', sent)
            score += len(quoted) * 0.8
            
            # Parenthetical clarifications (shows precision)
            parens = re.findall(r'\([^)]+\)', sent)
            score += len(parens) * 1.0
            
            # Specific action verbs vs generic ones
            specific_verbs = re.findall(r'\b(?:implement|configure|execute|analyze|calculate|measure|identify|specify|demonstrate|illustrate|establish|integrate|optimize|transform|generate|extract|validate|deploy)\b', sent, re.IGNORECASE)
            score += len(specific_verbs) * 0.7
            
            # Normalize by sentence length to get density
            density = score / max(len(words), 1)
            return density
        
        sentence_densities = [sentence_info_density(s) for s in sentences]
        avg_density = sum(sentence_densities) / len(sentence_densities) if sentence_densities else 0
        max_density = max(sentence_densities) if sentence_densities else 0
        
        # ---- Feature 2: Specificity gradient ----
        # Measure how much new specific information each sentence adds
        def compute_specificity_gradient(sents):
            if len(sents) < 2:
                return 0.5
            
            seen_content_words = set()
            new_info_counts = []
            
            for sent in sents:
                words = set(re.findall(r'\b[a-z]{4,}\b', sent.lower()))
                new_words = words - seen_content_words
                ratio = len(new_words) / max(len(words), 1)
                new_info_counts.append(ratio)
                seen_content_words.update(words)
            
            # Good responses maintain a steady flow of new information
            if not new_info_counts:
                return 0
            
            # Average new information ratio (higher = more diverse content)
            avg_new = sum(new_info_counts) / len(new_info_counts)
            
            # Penalize if later sentences add nothing new (repetitive)
            if len(new_info_counts) > 2:
                later_avg = sum(new_info_counts[len(new_info_counts)//2:]) / len(new_info_counts[len(new_info_counts)//2:])
                if later_avg < 0.1:
                    avg_new *= 0.7
            
            return avg_new
        
        specificity_gradient = compute_specificity_gradient(sentences)
        
        # ---- Feature 3: Substantive vs filler sentence ratio ----
        filler_patterns = [
            r'\b(?:it depends|there are (?:many|various|several|different) (?:factors|ways|reasons|things))\b',
            r'\b(?:many people|some people|a lot of people)\s+(?:think|believe|say|feel)\b',
            r'\b(?:in general|generally speaking|as a general rule|broadly speaking)\b',
            r'\b(?:it\'?s? (?:important|worth|good|nice) to (?:note|mention|remember|keep in mind))\b',
            r'\b(?:there are pros and cons|it varies|results may vary)\b',
            r'\b(?:at the end of the day|when all is said and done|all things considered)\b',
            r'\b(?:you know|I mean|kind of|sort of|basically|essentially|actually|literally)\b',
        ]
        
        filler_sentence_count = 0
        for sent in sentences:
            is_filler = False
            for pattern in filler_patterns:
                if re.search(pattern, sent, re.IGNORECASE):
                    is_filler = True
                    break
            # Also check if sentence is very short and generic
            words = sent.split()
            if len(words) < 5 and not re.search(r'\d', sent):
                is_filler = True
            if is_filler:
                filler_sentence_count += 1
        
        substantive_ratio = 1.0 - (filler_sentence_count / max(len(sentences), 1))
        
        # ---- Feature 4: Definitiveness of language ----
        # Measure ratio of definitive statements vs hedging
        hedging_words = re.findall(
            r'\b(?:maybe|perhaps|possibly|probably|might|could be|somewhat|rather|quite|fairly|'
            r'a bit|a little|tend to|seems? like|appears? to|I think|I guess|I suppose|'
            r'not sure|uncertain|unclear|debatable|arguably)\b',
            response_clean, re.IGNORECASE
        )
        
        definitive_words = re.findall(
            r'\b(?:specifically|precisely|exactly|always|never|must|shall|will|ensures?|'
            r'guarantees?|requires?|necessitates?|demonstrates?|proves?|confirms?|'
            r'according to|based on|results in|leads to|causes?|produces?|creates?)\b',
            response_clean, re.IGNORECASE
        )
        
        total_stance = len(hedging_words) + len(definitive_words)
        if total_stance > 0:
            definitiveness = len(definitive_words) / total_stance
        else:
            definitiveness = 0.5
        
        # ---- Feature 5: Structural coherence signals ----
        # Numbered/lettered steps, cause-effect chains, conditional logic
        structural_score = 0.0
        
        # Numbered items
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response_clean)
        structural_score += min(len(numbered_items) * 0.3, 2.0)
        
        # Lettered items
        lettered_items = re.findall(r'(?:^|\n)\s*[a-z][\.\)]\s', response_clean)
        structural_score += min(len(lettered_items) * 0.3, 1.0)
        
        # Cause-effect language
        causal = re.findall(
            r'\b(?:because|therefore|consequently|as a result|this (?:means|leads|causes|ensures)|'
            r'due to|since|thus|hence|so that|in order to|which (?:means|results|leads))\b',
            response_clean, re.IGNORECASE
        )
        structural_score += min(len(causal) * 0.2, 1.5)
        
        # Conditional logic
        conditionals = re.findall(
            r'\b(?:if .{5,40}(?:then|,)|when .{5,40}(?:,|then)|unless|provided that|'
            r'in case|assuming)\b',
            response_clean, re.IGNORECASE
        )
        structural_score += min(len(conditionals) * 0.3, 1.0)
        
        # Normalize structural score
        structural_score = min(structural_score / 3.0, 1.0)
        
        # ---- Feature 6: Response engagement with query ----
        # How well the response addresses the specific query
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        response_words = set(re.findall(r'\b[a-z]{3,}\b', response_clean.lower()))
        
        # Remove very common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
                      'was', 'one', 'our', 'out', 'has', 'have', 'had', 'been', 'would',
                      'could', 'should', 'will', 'may', 'might', 'this', 'that', 'with',
                      'they', 'from', 'what', 'which', 'when', 'where', 'how', 'who',
                      'their', 'there', 'then', 'than', 'them', 'these', 'those', 'some',
                      'into', 'also', 'just', 'more', 'most', 'very', 'about', 'being'}
        
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        if query_content:
            query_coverage = len(query_content & response_content) / len(query_content)
        else:
            query_coverage = 0.5
        
        # ---- Feature 7: Lexical sophistication ----
        # Ratio of uncommon/specific words (proxy: words not in top-frequency list)
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
            'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his',
            'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
            'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
            'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like',
            'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
            'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
            'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'very', 'much', 'too', 'really',
            'thing', 'things', 'something', 'lot', 'right', 'still', 'try', 'feel', 'need',
            'should', 'may', 'might', 'been', 'was', 'were', 'are', 'is', 'am', 'being',
            'had', 'has', 'did', 'does', 'done', 'doing', 'going', 'got', 'getting'
        }
        
        all_words = re.findall(r'\b[a-z]+\b', response_clean.lower())
        if all_words:
            uncommon_count = sum(1 for w in all_words if w not in common_words and len(w) > 3)
            lexical_sophistication = uncommon_count / len(all_words)
        else:
            lexical_sophistication = 0
        
        # ---- Feature 8: Empathy and engagement signals (for emotional queries) ----
        emotional_query = bool(re.search(
            r'\b(?:feel|feeling|emotion|stress|frustrat|sad|lonely|heartbrok|devastat|'
            r'comfort|support|struggling|difficult|tough|hard time|upset)\b',
            query, re.IGNORECASE
        ))
        
        empathy_score = 0.5  # neutral default
        if emotional_query:
            empathy_markers = re.findall(
                r'\b(?:understand|hear you|sorry|completely|absolutely|natural|okay|'
                r'valid|normal|perfectly|genuinely|sincerely|acknowledge|recognize|'
                r'it\'s okay|it\'s fine|it\'s natural|don\'t worry)\b',
                response_clean, re.IGNORECASE
            )
            dismissive_markers = re.findall(
                r'\b(?:just get over|move on|stop|don\'t be|shouldn\'t feel|'
                r'get yourself together|toughen up|deal with it|suck it up)\b',
                response_clean, re.IGNORECASE
            )
            empathy_count = len(empathy_markers)
            dismiss_count = len(dismissive_markers)
            empathy_score = min(empathy_count * 0.15, 1.0) - min(dismiss_count * 0.3, 0.8)
            empathy_score = max(0, min(1, empathy_score + 0.3))
        
        # ---- Feature 9: Response length adequacy ----
        word_count = len(all_words)
        # Moderate length is good; too short = insufficient detail, too long with low density = bad
        if word_count < 20:
            length_score = 0.2
        elif word_count < 50:
            length_score = 0.5
        elif word_count < 100:
            length_score = 0.8
        elif word_count < 200:
            length_score = 1.0
        elif word_count < 400:
            length_score = 0.9
        else:
            length_score = 0.8
        
        # ---- Feature 10: Actionability ----
        # Does the response provide actionable guidance?
        action_patterns = re.findall(
            r'\b(?:first|second|third|next|then|finally|start by|begin with|'
            r'try to|make sure|ensure|consider|remember to|don\'t forget|'
            r'step \d|you (?:can|should|could|need to|might want to)|'
            r'here\'s how|here are|follow these|the key is)\b',
            response_clean, re.IGNORECASE
        )
        actionability = min(len(action_patterns) * 0.12, 1.0)
        
        # ---- AGGREGATION ----
        # Weighted combination of all features
        weights = {
            'avg_density': 1.5,
            'max_density': 0.5,
            'specificity_gradient': 1.2,
            'substantive_ratio': 1.3,
            'definitiveness': 0.8,
            'structural_score': 1.0,
            'query_coverage': 0.7,
            'lexical_sophistication': 0.8,
            'empathy_score': 0.9 if emotional_query else 0.3,
            'length_score': 0.6,
            'actionability': 1.0,
        }
        
        features = {
            'avg_density': min(avg_density * 3, 1.0),  # scale up density
            'max_density': min(max_density * 2, 1.0),
            'specificity_gradient': specificity_gradient,
            'substantive_ratio': substantive_ratio,
            'definitiveness': definitiveness,
            'structural_score': structural_score,
            'query_coverage': query_coverage,
            'lexical_sophistication': min(lexical_sophistication * 2, 1.0),
            'empathy_score': empathy_score,
            'length_score': length_score,
            'actionability': actionability,
        }
        
        total_weight = sum(weights.values())
        raw_score = sum(features[k] * weights[k] for k in features) / total_weight
        
        # Map from [0, 1] to [1, 5] scale
        final_score = 1.0 + raw_score * 4.0
        
        # Apply slight non-linearity to spread scores
        # Using a sigmoid-like transformation centered at 3
        centered = final_score - 3.0
        spread_score = 3.0 + 2.5 * math.tanh(centered * 0.8)
        
        # Clamp to [1, 5]
        spread_score = max(1.0, min(5.0, spread_score))
        
        return round(spread_score, 2)
        
    except Exception:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 50:
                return 2.5
            return 1.5
        except Exception:
            return 2.0