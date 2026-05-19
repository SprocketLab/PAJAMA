def judging_function(query, response):
    """
    Evaluates factual accuracy indicators in an LLM response.
    
    This variant uses a unique approach based on:
    1. Named entity density (capitalized multi-word phrases, proper nouns)
    2. Citation/reference pattern detection
    3. Specificity signals (dates, numbers, named sources)
    4. Hallucination red-flag detection (absolute claims, unsourced precise stats)
    5. Appropriate epistemic calibration (hedging vs overconfidence ratio)
    6. Discourse structure analysis (claim-evidence patterns)
    7. Information density via unique content word ratio
    8. Response completeness and engagement with query
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        resp = response.strip()
        q = query.strip()
        
        if len(resp) < 10:
            return 0.5
        
        score = 0.0
        
        # --- 1. Named Entity / Proper Noun Density ---
        # Look for capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]+', resp)
        proper_noun_count = 0
        total_words = resp.split()
        num_words = max(len(total_words), 1)
        
        for sent in sentences:
            sent = sent.strip()
            words = sent.split()
            if len(words) < 2:
                continue
            # Skip first word (sentence starter), check rest for capitalization
            for w in words[1:]:
                cleaned = re.sub(r'[^a-zA-Z]', '', w)
                if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                    proper_noun_count += 1
        
        proper_noun_density = proper_noun_count / num_words
        # Reward moderate density, penalize extreme (likely hallucinated name-dropping)
        if proper_noun_density < 0.15:
            score += proper_noun_density * 40  # up to ~6 points
        else:
            score += 6.0 - (proper_noun_density - 0.15) * 20  # diminishing returns
        score = max(score, 0)
        
        # --- 2. Specific Reference Patterns ---
        # Citations, book titles, user references, URLs
        citation_patterns = [
            r'\*[A-Z][^*]+\*',           # *Italicized titles*
            r'"[A-Z][^"]{3,}"',           # "Quoted titles"
            r'u/\w+',                      # Reddit user references
            r'(?:according to|as (?:noted|described|mentioned) (?:by|in))',  # Attribution phrases
            r'(?:chapter|section|page|vol\.?|volume)\s+\d+',  # Document references
            r'(?:19|20)\d{2}',            # Years (1900-2099)
            r'(?:St\.|Dr\.|Prof\.|Mr\.|Mrs\.)\s+[A-Z]',  # Titled names
        ]
        
        citation_score = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, resp, re.IGNORECASE)
            citation_score += len(matches) * 1.5
        
        score += min(citation_score, 10)  # cap at 10
        
        # --- 3. Numeric Specificity ---
        # Dates, quantities, percentages — but not overly precise unsourced stats
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|percent)?\b', resp)
        num_count = len(numbers)
        
        # Moderate numbers = good specificity; too many without sources = suspicious
        if num_count <= 8:
            score += num_count * 0.8
        else:
            score += 6.4 - (num_count - 8) * 0.3
        score = max(score, 0)
        
        # --- 4. Hallucination Red-Flags ---
        red_flag_penalty = 0
        
        # Absolute/sensationalist language
        absolute_patterns = [
            r'\b(?:always|never|every single|without exception|guaranteed|undeniable|unquestionable)\b',
            r'\b(?:everyone knows|it is a fact that|obviously|clearly everyone)\b',
            r'\b(?:exposed|shocking|they don\'t want you to know|wake up|sheeple)\b',
            r'\b(?:conspiracy|cover.?up|mainstream media lies|big pharma)\b',
            r'\b(?:100% (?:certain|sure|proven|guaranteed))\b',
            r'\b(?:absolutely (?:certain|proven|guaranteed|no doubt))\b',
        ]
        
        for pattern in absolute_patterns:
            matches = re.findall(pattern, resp, re.IGNORECASE)
            red_flag_penalty += len(matches) * 2.0
        
        # Overly precise unsourced statistics (e.g., "exactly 73.4% of people")
        precise_stats = re.findall(r'\b\d+\.\d{2,}%', resp)
        red_flag_penalty += len(precise_stats) * 1.5
        
        # Exclamation marks (sensationalism)
        exclamation_count = resp.count('!')
        if exclamation_count > 2:
            red_flag_penalty += (exclamation_count - 2) * 0.5
        
        # ALL CAPS words (shouting/sensationalism)
        all_caps = re.findall(r'\b[A-Z]{4,}\b', resp)
        # Filter out common acronyms
        common_acronyms = {'HTTP', 'HTML', 'HVAC', 'NASA', 'NATO', 'AUTO', 'NULL', 'FROM', 
                          'JOIN', 'LEFT', 'WHERE', 'SELECT', 'TABLE', 'COMMENT', 'INTO',
                          'CREATE', 'MIRI', 'SQL', 'VARCHAR'}
        non_acronym_caps = [w for w in all_caps if w not in common_acronyms]
        red_flag_penalty += len(non_acronym_caps) * 0.8
        
        score -= min(red_flag_penalty, 12)
        
        # --- 5. Epistemic Calibration ---
        # Appropriate hedging for uncertain claims
        hedging_phrases = [
            r'\b(?:might|may|could|perhaps|possibly|likely|unlikely|probably|tends? to)\b',
            r'\b(?:it seems|it appears|in my (?:experience|understanding|opinion))\b',
            r'\b(?:generally|typically|often|sometimes|usually|frequently)\b',
            r'\b(?:one (?:possibility|interpretation|view)|some (?:argue|suggest|believe))\b',
            r'\b(?:there\'s a chance|if I recall|as far as I know|I think|I believe)\b',
            r'\b(?:roughly|approximately|around|about|circa|estimated)\b',
            r'\b(?:for example|for instance|such as|e\.g\.|i\.e\.)\b',
        ]
        
        hedge_count = 0
        for pattern in hedging_phrases:
            matches = re.findall(pattern, resp, re.IGNORECASE)
            hedge_count += len(matches)
        
        # Reward moderate hedging (shows epistemic humility)
        hedge_ratio = hedge_count / max(num_words / 50, 1)
        if hedge_ratio < 5:
            score += hedge_ratio * 1.5
        else:
            score += 7.5 - (hedge_ratio - 5) * 0.5  # too much hedging = wishy-washy
        
        # --- 6. Claim-Evidence Structure ---
        # Look for patterns: claim followed by evidence/explanation
        evidence_markers = [
            r'\b(?:because|since|due to|as a result|therefore|thus|consequently)\b',
            r'\b(?:this (?:means|implies|suggests|indicates))\b',
            r'\b(?:the reason|one reason|a key factor)\b',
            r'\b(?:research|studies?|evidence|data|findings?|experiments?)\b',
            r'\b(?:specifically|in particular|notably|importantly)\b',
        ]
        
        evidence_count = 0
        for pattern in evidence_markers:
            matches = re.findall(pattern, resp, re.IGNORECASE)
            evidence_count += len(matches)
        
        score += min(evidence_count * 0.8, 6)
        
        # --- 7. Information Density ---
        # Ratio of unique content words to total words (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'while',
            'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'his', 'she', 'her', 'they', 'their', 'what', 'which',
            'who', 'whom', 'up', 'about', 'also', 'them', 'these', 'those',
        }
        
        content_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{2,}\b', resp) 
                        if w.lower() not in stop_words]
        
        if content_words:
            unique_content = set(content_words)
            info_density = len(unique_content) / max(len(content_words), 1)
            # Higher density = more diverse vocabulary = more informative
            score += info_density * 8  # up to ~8 points
        
        # --- 8. Response Length & Completeness ---
        # Longer responses tend to be more thorough (with diminishing returns)
        char_len = len(resp)
        if char_len < 50:
            length_score = 0
        elif char_len < 100:
            length_score = 1
        elif char_len < 200:
            length_score = 2
        elif char_len < 400:
            length_score = 4
        elif char_len < 800:
            length_score = 6
        else:
            length_score = 6 + math.log(char_len / 800) * 1.5
        
        score += min(length_score, 10)
        
        # --- 9. Query Engagement ---
        # Check if response addresses key terms from the query
        query_content_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', q) 
            if w.lower() not in stop_words
        )
        resp_content_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', resp) 
            if w.lower() not in stop_words
        )
        
        if query_content_words:
            engagement = len(query_content_words & resp_content_words) / len(query_content_words)
            score += engagement * 5  # up to 5 points
        
        # --- 10. Structural Quality ---
        # Multiple sentences show developed thought
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        if sentence_count >= 3:
            score += min((sentence_count - 2) * 0.5, 4)
        
        # Presence of examples (concrete vs abstract)
        example_indicators = re.findall(
            r'\b(?:for example|for instance|such as|e\.g\.|like when|consider)\b', 
            resp, re.IGNORECASE
        )
        score += min(len(example_indicators) * 1.5, 4)
        
        # --- 11. Penalize meta/non-answers ---
        meta_patterns = [
            r'\bplease read our rules\b',
            r'\byour (?:comments|posts?) will be removed\b',
            r'\bwhile you wait for an answer\b',
            r'\bwelcome to /r/\b',
            r'\bI cannot help\b',
            r'\bI\'m not able to\b',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, resp, re.IGNORECASE):
                score -= 8
        
        # --- 12. Code block handling ---
        # For technical queries, code blocks are valuable
        code_blocks = re.findall(r'```[\s\S]*?```', resp)
        if code_blocks:
            # Check if query seems technical
            tech_indicators = re.findall(
                r'\b(?:SQL|SELECT|CREATE|TABLE|code|function|program|algorithm|API)\b', 
                q, re.IGNORECASE
            )
            if tech_indicators:
                score += min(len(code_blocks) * 3, 6)
            # Well-formatted code with keywords
            for block in code_blocks:
                sql_keywords = re.findall(
                    r'\b(?:SELECT|FROM|JOIN|WHERE|LEFT|RIGHT|ON|AS|GROUP|ORDER|INSERT)\b', 
                    block, re.IGNORECASE
                )
                if sql_keywords:
                    score += min(len(sql_keywords) * 0.3, 3)
        
        # --- 13. Contrast/nuance indicators ---
        contrast_words = re.findall(
            r'\b(?:however|although|on the other hand|conversely|nevertheless|'
            r'while.*?,|whereas|that said|in contrast|but also|trade.?off)\b',
            resp, re.IGNORECASE
        )
        score += min(len(contrast_words) * 1.2, 5)
        
        # Normalize to 0-100 range
        # Theoretical max is around 70-80, typical good response 30-50
        score = max(0, score)
        normalized = min(score * 1.5, 100)
        
        return round(normalized, 2)
        
    except Exception:
        return 5.0