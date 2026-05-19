def judging_function(query, response):
    """
    Evaluates evidence density and specificity by analyzing the presence of
    specific details, named entities, precise language, and actionable content.
    
    Uses a token-level classification approach: categorizes each word/token
    into specificity tiers and computes a weighted density score.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.0
        
        words = response.split()
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        score = 0.0
        
        # === 1. NUMBER DENSITY (precise quantities, dates, percentages) ===
        # Find all number-like tokens
        number_patterns = [
            r'\b\d+\.?\d*%\b',          # percentages
            r'\b\d{4}\b',                # years
            r'\$\d+[\d,.]*',             # dollar amounts
            r'\b\d+[\d,]*\.?\d*\b',      # general numbers
            r'\b\d+(?:st|nd|rd|th)\b',   # ordinals
        ]
        number_count = 0
        for pat in number_patterns:
            number_count += len(re.findall(pat, response))
        
        number_density = min(number_count / max(total_words, 1) * 100, 15)
        score += number_density * 1.5
        
        # === 2. CAPITALIZED PROPER NOUNS (named entities proxy) ===
        # Words that are capitalized but not at sentence start
        sentences = re.split(r'[.!?]\s+', response)
        proper_noun_count = 0
        for sent in sentences:
            sent_words = sent.split()
            for i, w in enumerate(sent_words):
                if i > 0 and len(w) > 1 and w[0].isupper() and not w.isupper():
                    proper_noun_count += 1
        
        proper_density = min(proper_noun_count / max(total_words, 1) * 100, 15)
        score += proper_density * 1.2
        
        # === 3. SPECIFICITY VOCABULARY TIERS ===
        # Tier 1: Highly specific action/detail words
        tier1_words = {
            'specifically', 'precisely', 'exactly', 'approximately', 'namely',
            'including', 'such as', 'for example', 'for instance', 'e.g.',
            'i.e.', 'particularly', 'notably', 'according', 'measured',
            'calculated', 'estimated', 'recorded', 'documented', 'identified',
            'categorize', 'categorized', 'tracked', 'designed', 'implemented',
            'configured', 'processed', 'transmitted', 'displayed', 'generated',
            'personalized', 'handwritten', 'customized', 'tailored'
        }
        
        # Tier 2: Moderately specific descriptive words
        tier2_words = {
            'allows', 'enables', 'provides', 'includes', 'contains',
            'features', 'requires', 'involves', 'creates', 'produces',
            'displays', 'sends', 'receives', 'stores', 'manages',
            'connects', 'supports', 'handles', 'delivers', 'performs',
            'structured', 'organized', 'focused', 'detailed', 'specific',
            'concrete', 'practical', 'actionable', 'relevant', 'unique',
            'succulent', 'delectable', 'reminders', 'alerts', 'budgeting',
            'income', 'expenses', 'spending', 'photo', 'message'
        }
        
        # Vague/filler words and phrases (negative signal)
        vague_words = {
            'various', 'many', 'some', 'several', 'numerous', 'lots',
            'things', 'stuff', 'somewhat', 'quite', 'rather', 'fairly',
            'basically', 'essentially', 'generally', 'typically', 'usually',
            'often', 'sometimes', 'perhaps', 'maybe', 'probably',
            'might', 'could', 'somehow', 'something', 'anything',
            'everything', 'whatever', 'whoever', 'wherever', 'however',
            'etc', 'overall', 'aspect', 'aspects'
        }
        
        lower_response = response.lower()
        lower_words = lower_response.split()
        
        tier1_count = sum(1 for w in lower_words if w.strip('.,;:!?()[]"\'') in tier1_words)
        tier2_count = sum(1 for w in lower_words if w.strip('.,;:!?()[]"\'') in tier2_words)
        vague_count = sum(1 for w in lower_words if w.strip('.,;:!?()[]"\'') in vague_words)
        
        # Check for vague phrases
        vague_phrases = [
            'many people', 'it depends', 'there are various', 'in many ways',
            'a lot of', 'kind of', 'sort of', 'more or less', 'to some extent',
            'in general', 'as a whole', 'for the most part', 'by and large',
            'all in all', 'at the end of the day', 'when it comes to',
            'it is important to', 'it is worth noting', 'it should be noted',
            'needless to say', 'goes without saying', 'it is well known'
        ]
        vague_phrase_count = sum(1 for phrase in vague_phrases if phrase in lower_response)
        
        tier1_score = min(tier1_count / max(total_words, 1) * 100, 10) * 2.0
        tier2_score = min(tier2_count / max(total_words, 1) * 100, 15) * 1.0
        vague_penalty = min(vague_count / max(total_words, 1) * 100, 20) * 0.5
        vague_phrase_penalty = vague_phrase_count * 1.5
        
        score += tier1_score + tier2_score - vague_penalty - vague_phrase_penalty
        
        # === 4. UNIQUE CONTENT TOKENS (lexical diversity of substantive words) ===
        # Filter out stopwords and measure unique meaningful words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my', 'who',
            'which', 'what', 'also', 'about', 'up'
        }
        
        content_words = [w.strip('.,;:!?()[]"\'').lower() for w in words 
                        if w.strip('.,;:!?()[]"\'').lower() not in stopwords 
                        and len(w.strip('.,;:!?()[]"\'')) > 2]
        
        unique_content = set(content_words)
        content_count = len(content_words)
        unique_count = len(unique_content)
        
        if content_count > 0:
            # Unique content ratio (penalizes repetition)
            uniqueness_ratio = unique_count / content_count
            # Reward having more unique content words (absolute count matters)
            unique_richness = min(unique_count, 50) * 0.3
            score += uniqueness_ratio * 5 + unique_richness
        
        # === 5. REPETITION PENALTY ===
        # Detect excessive repetition of the same word
        if content_words:
            word_freq = Counter(content_words)
            most_common_freq = word_freq.most_common(1)[0][1]
            if content_count > 3:
                repetition_ratio = most_common_freq / content_count
                if repetition_ratio > 0.3:
                    score -= (repetition_ratio - 0.3) * 30
        
        # === 6. SENTENCE-LEVEL DETAIL DENSITY ===
        # Reward sentences that pack in multiple specific details
        sentences_raw = re.split(r'[.!?]+', response)
        sentences_raw = [s.strip() for s in sentences_raw if len(s.strip()) > 5]
        num_sentences = len(sentences_raw)
        
        if num_sentences > 0:
            detail_rich_sentences = 0
            for sent in sentences_raw:
                sent_lower = sent.lower()
                sent_words = sent.split()
                detail_markers = 0
                
                # Check for numbers in sentence
                if re.search(r'\d', sent):
                    detail_markers += 1
                
                # Check for specific descriptors (adjectives/adverbs that add info)
                specific_descriptors = {
                    'red', 'blue', 'green', 'large', 'small', 'first', 'second',
                    'third', 'primary', 'secondary', 'main', 'key', 'critical',
                    'essential', 'fundamental', 'advanced', 'basic', 'complex',
                    'simple', 'digital', 'physical', 'social', 'economic',
                    'political', 'cultural', 'technical', 'scientific',
                    'musical', 'visual', 'written', 'spoken', 'personal',
                    'professional', 'financial', 'educational', 'environmental',
                    'handwritten', 'personalized', 'heartfelt', 'creative'
                }
                desc_count = sum(1 for w in sent_lower.split() 
                               if w.strip('.,;:!?()[]"\'') in specific_descriptors)
                if desc_count >= 1:
                    detail_markers += 1
                if desc_count >= 2:
                    detail_markers += 1
                
                # Check for listing/enumeration patterns
                if ',' in sent and len(sent.split(',')) >= 3:
                    detail_markers += 1
                
                # Check for parenthetical details
                if '(' in sent or '-' in sent:
                    detail_markers += 0.5
                
                if detail_markers >= 2:
                    detail_rich_sentences += 1
            
            detail_sentence_ratio = detail_rich_sentences / num_sentences
            score += detail_sentence_ratio * 8
        
        # === 7. STRUCTURAL COMPLEXITY (compound/complex sentences carry more info) ===
        # Count subordinate clause markers
        clause_markers = [
            'which', 'that', 'where', 'when', 'while', 'although', 'because',
            'since', 'unless', 'whereas', 'whereby', 'wherein'
        ]
        clause_count = sum(1 for w in lower_words 
                          if w.strip('.,;:!?()[]"\'') in clause_markers)
        score += min(clause_count * 0.5, 4)
        
        # === 8. RESPONSE LENGTH BONUS (with diminishing returns) ===
        # Longer responses tend to have more details, but cap the bonus
        length_bonus = math.log(max(total_words, 1) + 1) * 2
        score += min(length_bonus, 10)
        
        # === 9. ENUMERATION / LIST DETECTION ===
        # Lists and enumerations are evidence of specificity
        list_patterns = [
            r'\b\d+\)',    # 1) 2) 3)
            r'\b\d+\.',    # 1. 2. 3.
            r'^\s*[-•*]',  # bullet points
            r'\bfirst\b.*\bsecond\b',
            r'\bfirstly\b',
        ]
        list_score = 0
        for pat in list_patterns:
            if re.search(pat, response, re.MULTILINE | re.IGNORECASE):
                list_score += 1.5
        score += min(list_score, 5)
        
        # === 10. COMMA-SEPARATED SPECIFICS ===
        # Count comma-separated items (indicates listing specific things)
        comma_segments = response.split(',')
        if len(comma_segments) >= 3:
            # Check that segments are substantive (not just filler)
            substantive_segments = sum(1 for seg in comma_segments 
                                      if len(seg.strip().split()) >= 2)
            if substantive_segments >= 3:
                score += min(substantive_segments * 0.8, 5)
        
        # === 11. TECHNICAL/DOMAIN TERMS ===
        # Longer words (6+ chars) that aren't common tend to be more specific/technical
        technical_words = [w for w in content_words if len(w) >= 7]
        unique_technical = set(technical_words)
        tech_score = min(len(unique_technical) * 0.6, 8)
        score += tech_score
        
        # === 12. EMPTY/MINIMAL RESPONSE PENALTY ===
        if total_words < 5:
            score = max(score * 0.2, 0)
        elif total_words < 10:
            score = max(score * 0.5, 0)
        
        # Normalize to 0-100 range
        score = max(0.0, min(score, 100.0))
        
        return round(score, 2)
        
    except Exception:
        return 0.0