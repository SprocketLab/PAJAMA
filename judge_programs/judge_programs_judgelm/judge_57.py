def judging_function(query, response):
    """
    Evaluates response quality focusing on Epistemic Calibration and Uncertainty Communication.
    
    Rewards:
    - Appropriate hedging language for uncertain claims
    - Distinguishing established facts from speculation
    - Acknowledging limitations of knowledge
    - Proportional confidence to claim certainty
    
    Penalizes:
    - Overconfident claims on ambiguous topics
    - Presenting speculation as fact
    - Empty or incoherent responses
    - False precision on inherently uncertain questions
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        import re
        import math
        from collections import Counter
        
        resp_lower = response.lower()
        query_lower = query.lower()
        
        score = 5.0  # Start at midpoint
        
        # === 1. BASIC QUALITY CHECKS ===
        
        # Penalize extremely short responses (likely low effort)
        word_count = len(response_stripped.split())
        if word_count <= 2:
            score -= 3.0
        elif word_count <= 5:
            score -= 1.5
        elif word_count >= 15:
            score += 0.5
        
        # Penalize responses that are just punctuation or whitespace-like
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        if alpha_chars < 3:
            return 0.5
        
        alpha_ratio = alpha_chars / max(len(response_stripped), 1)
        if alpha_ratio < 0.3:
            score -= 2.0
        
        # === 2. EPISTEMIC HEDGING LANGUAGE (positive signals) ===
        
        # Words/phrases that indicate appropriate uncertainty communication
        hedging_phrases = [
            r'\blikely\b', r'\bunlikely\b', r'\bprobably\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\bmay\b', r'\bmight\b', r'\bcould be\b',
            r'\bit is possible\b', r'\bit seems\b', r'\bappears to\b',
            r'\bsuggests?\b', r'\bresearch suggests\b', r'\bstudies suggest\b',
            r'\bevidence suggests\b', r'\bgenerally\b', r'\btypically\b',
            r'\btends? to\b', r'\boften\b', r'\busually\b',
            r'\bin many cases\b', r'\bin some cases\b', r'\bin most cases\b',
            r'\bto some extent\b', r'\bto a degree\b',
            r'\bapproximately\b', r'\broughly\b', r'\babout\b',
            r'\bestimated\b', r'\baround\b',
            r'\bit depends\b', r'\bdepending on\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bcan vary\b', r'\bvaries?\b', r'\bvarying\b',
            r'\bsome\b.*\bbelieve\b', r'\bsome\b.*\bargue\b',
            r'\baccording to\b', r'\bbased on\b',
        ]
        
        hedging_count = 0
        for pattern in hedging_phrases:
            matches = re.findall(pattern, resp_lower)
            hedging_count += len(matches)
        
        # Reward hedging but with diminishing returns
        hedging_score = min(hedging_count * 0.3, 2.0)
        score += hedging_score
        
        # === 3. EPISTEMIC HUMILITY / LIMITATION ACKNOWLEDGMENT ===
        
        humility_phrases = [
            r'\bdifficult to\b.*\bdetermine\b', r'\bdifficult to\b.*\bprovide\b',
            r'\bdifficult to\b.*\bsay\b', r'\bhard to\b.*\bsay\b',
            r'\bnot entirely clear\b', r'\bnot certain\b',
            r'\bi\'m not sure\b', r'\bi am not sure\b',
            r'\bunclear\b', r'\buncertain\b', r'\buncertainty\b',
            r'\bwe don\'t know\b', r'\bnot well understood\b',
            r'\bdebated?\b', r'\bcontroversial\b', r'\bcontested\b',
            r'\bsubjective\b', r'\bopen to interpretation\b',
            r'\blimited\b.*\binformation\b', r'\blimited\b.*\bdata\b',
            r'\bmore research\b', r'\bfurther research\b',
            r'\bhowever\b', r'\balthough\b', r'\bnonetheless\b',
            r'\bon the other hand\b', r'\bthat said\b',
            r'\bit\'s worth noting\b', r'\bimportant to note\b',
            r'\bkeep in mind\b', r'\bbe aware\b',
            r'\bnot without\b.*\bcriticism\b', r'\bnot without\b.*\bcontroversy\b',
            r'\bcaveat\b', r'\bnuance\b', r'\bcomplexity\b',
        ]
        
        humility_count = 0
        for pattern in humility_phrases:
            if re.search(pattern, resp_lower):
                humility_count += 1
        
        humility_score = min(humility_count * 0.4, 2.0)
        score += humility_score
        
        # === 4. OVERCONFIDENCE DETECTION (negative signals) ===
        
        overconfident_phrases = [
            r'\bdefinitely\b', r'\babsolutely\b', r'\bwithout a doubt\b',
            r'\bundoubtedly\b', r'\bunquestionably\b', r'\bindisputably\b',
            r'\bcertainly\b', r'\bwithout question\b',
            r'\balways\b', r'\bnever\b', r'\beveryone knows\b',
            r'\bit is clear that\b', r'\bobviously\b', r'\bclearly\b',
            r'\bno doubt\b', r'\bguaranteed\b',
            r'\bthe fact is\b', r'\bthe truth is\b',
            r'\b100%\b', r'\b100 percent\b',
            r'\bproven\b.*\bfact\b', r'\bscientific fact\b',
        ]
        
        overconfidence_count = 0
        for pattern in overconfident_phrases:
            matches = re.findall(pattern, resp_lower)
            overconfidence_count += len(matches)
        
        # Penalize overconfidence but consider context
        overconfidence_penalty = min(overconfidence_count * 0.25, 1.5)
        score -= overconfidence_penalty
        
        # === 5. QUERY COMPLEXITY / AMBIGUITY DETECTION ===
        # Questions that are more ambiguous or complex should reward hedging more
        
        ambiguity_indicators = [
            r'\bhow many\b', r'\bwhat is the\b.*\bbest\b',
            r'\bshould\b', r'\bis it ok\b', r'\bis it okay\b',
            r'\bwhat do you think\b', r'\bopinion\b',
            r'\bwhy\b', r'\bhow\b', r'\bcan you explain\b',
            r'\bhistory\b', r'\bmeaning\b', r'\binterpretation\b',
            r'\brecent\b', r'\bfuture\b', r'\bpredict\b',
            r'\bbelieve\b', r'\bfeel\b',
        ]
        
        query_ambiguity = 0
        for pattern in ambiguity_indicators:
            if re.search(pattern, query_lower):
                query_ambiguity += 1
        
        # For ambiguous queries, hedging is more valuable
        if query_ambiguity >= 2:
            score += hedging_score * 0.3  # Bonus for hedging on ambiguous queries
            score += humility_score * 0.3
        
        # === 6. COHERENCE AND STRUCTURE ===
        
        # Check for repetitive text (sign of low quality)
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) >= 2:
            unique_sentences = set(s.lower() for s in sentences)
            repetition_ratio = len(unique_sentences) / len(sentences)
            if repetition_ratio < 0.5:
                score -= 2.0
            elif repetition_ratio < 0.75:
                score -= 1.0
        
        # Check for garbage/random text patterns
        garbage_patterns = [
            r'(?:Input:|Output:){3,}',  # Repeated Input/Output markers
            r'(?:Question:.*?Answer:.*?){3,}',  # Repeated Q&A that's off-topic
            r'<[a-z]+>.*?</[a-z]+>.*?<[a-z]+>.*?</[a-z]+>.*?<[a-z]+>',  # Excessive HTML
            r'```.*?```.*?```',  # Excessive code blocks
            r'(?:def |import |class ){3,}',  # Code dump
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, resp_lower, re.DOTALL):
                score -= 2.0
                break
        
        # === 7. RELEVANCE TO QUERY ===
        
        # Simple keyword overlap check
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        response_words = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
        
        # Remove very common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                      'have', 'been', 'some', 'them', 'than', 'its', 'over',
                      'such', 'that', 'this', 'with', 'will', 'each', 'from',
                      'they', 'which', 'their', 'said', 'what', 'how', 'who',
                      'where', 'when', 'why', 'does', 'did', 'about', 'into',
                      'more', 'other', 'there', 'also', 'just', 'very'}
        
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        if len(query_content) > 0:
            overlap = len(query_content & response_content) / len(query_content)
            if overlap < 0.1:
                score -= 1.5  # Very low relevance
            elif overlap >= 0.3:
                score += 0.5  # Good relevance
        
        # === 8. SOURCING AND ATTRIBUTION ===
        
        source_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstudies?\b',
            r'\bexperts?\b', r'\bscholars?\b', r'\bhistorians?\b',
            r'\bscientists?\b', r'\breported\b', r'\bfound that\b',
            r'\bpublished\b', r'\bjournal\b', r'\buniversity\b',
            r'\bsource\b', r'\breference\b', r'\bcited\b',
        ]
        
        source_count = 0
        for pattern in source_patterns:
            if re.search(pattern, resp_lower):
                source_count += 1
        
        # Mild reward for sourcing (shows epistemic care)
        source_score = min(source_count * 0.15, 0.8)
        score += source_score
        
        # === 9. BALANCED PERSPECTIVE INDICATORS ===
        
        balance_phrases = [
            r'\bon one hand\b', r'\bon the other hand\b',
            r'\bwhile\b.*\b(also|however|but)\b',
            r'\bboth\b.*\band\b', r'\bpros and cons\b',
            r'\badvantages and disadvantages\b',
            r'\bdifferent perspectives\b', r'\bvarious\b.*\bviews\b',
            r'\bsome\b.*\bwhile others\b', r'\bsome\b.*\bbut others\b',
        ]
        
        balance_count = 0
        for pattern in balance_phrases:
            if re.search(pattern, resp_lower):
                balance_count += 1
        
        balance_score = min(balance_count * 0.3, 1.0)
        score += balance_score
        
        # === 10. LENGTH QUALITY INTERACTION ===
        # Longer responses with good epistemic markers are better
        # Longer responses without them might be overconfident rambling
        
        if word_count > 30:
            epistemic_density = (hedging_count + humility_count) / (word_count / 50.0)
            if epistemic_density > 1.0:
                score += 0.5  # Good density of epistemic markers
            elif epistemic_density < 0.2 and query_ambiguity >= 1:
                score -= 0.5  # Long response on ambiguous topic without hedging
        
        # === 11. COMPLETENESS CHECK ===
        # Truncated responses are still better than empty ones
        if response_stripped.endswith(('...', '…')):
            pass  # Neutral - might be intentional
        elif len(response_stripped) > 50 and not response_stripped[-1] in '.!?"\')':
            # Might be truncated
            score -= 0.3
        
        # === 12. DISTINGUISH FACTUAL VS OPINION QUERIES ===
        
        factual_indicators = [
            r'\bidentify\b', r'\blist\b', r'\bname\b', r'\bcreate\b',
            r'\brewrite\b', r'\btranslate\b', r'\bconvert\b',
            r'\bcalculate\b', r'\bdefine\b',
        ]
        
        is_factual_query = False
        for pattern in factual_indicators:
            if re.search(pattern, query_lower):
                is_factual_query = True
                break
        
        # For purely factual queries, hedging is less important
        # but we still reward clear, confident-but-accurate responses
        if is_factual_query:
            # Don't penalize lack of hedging for factual queries
            # But still reward coherence
            score += 0.3 if word_count >= 3 else 0
        
        # === FINAL CLAMPING ===
        score = max(0.0, min(10.0, score))
        
        # Round to 1 decimal
        score = round(score, 1)
        
        return score
        
    except Exception as e:
        # Fallback: return a neutral score
        try:
            if response and len(response.strip()) > 10:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except:
            return 0.0