def judging_function(query, response):
    """
    Evaluates evidence density and specificity in LLM responses.
    
    This variant focuses on:
    - Named entity density (capitalized multi-word phrases, proper nouns)
    - Numeric/quantitative information density
    - Specificity markers vs vagueness markers ratio
    - Technical/domain vocabulary richness
    - Structural elaboration (parentheticals, citations, qualifications with specifics)
    - Information-to-filler ratio
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_text = response.strip()
        if len(response_text) < 10:
            return 0.5
        
        words = response_text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # ---- 1. Named Entity / Proper Noun Density ----
        # Look for capitalized words that aren't sentence starters
        sentences = re.split(r'[.!?]\s+', response_text)
        proper_noun_count = 0
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) > 1:
                # Words after the first that are capitalized (likely proper nouns)
                for w in sent_words[1:]:
                    cleaned = re.sub(r'[^a-zA-Z]', '', w)
                    if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                        proper_noun_count += 1
        
        # Multi-word capitalized phrases (e.g., "University of Cambridge", "Effective Altruism")
        capitalized_phrases = re.findall(r'(?:[A-Z][a-z]+(?:\s+(?:of|the|and|in|for|de|von|van|du|la|el)\s+)?[A-Z][a-z]+)', response_text)
        
        # Specific references: book titles in italics/quotes, user mentions, URLs
        italic_refs = re.findall(r'\*[^*]+\*', response_text)
        quoted_refs = re.findall(r'"[^"]{3,}"', response_text)
        backtick_refs = re.findall(r'`[^`]+`', response_text)
        
        named_entity_score = (
            proper_noun_count * 1.5 +
            len(capitalized_phrases) * 2.0 +
            len(italic_refs) * 3.0 +
            len(quoted_refs) * 2.5 +
            len(backtick_refs) * 1.5
        )
        # Normalize per 100 words
        named_entity_density = (named_entity_score / max(word_count, 1)) * 100
        
        # ---- 2. Numeric / Quantitative Information ----
        # Specific numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b', response_text)
        dates = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', response_text)
        measurements = re.findall(r'\b\d+\s*(?:lb|lbs|kg|oz|mg|g|ml|L|ft|m|cm|mm|inch|inches|hours?|minutes?|seconds?|days?|weeks?|months?|years?|mph|km|mi)\b', response_text, re.IGNORECASE)
        fractions = re.findall(r'\b\d+/\d+\b', response_text)
        ranges = re.findall(r'\b\d+\s*[-–]\s*\d+\b', response_text)
        
        numeric_score = (
            len(numbers) * 1.0 +
            len(dates) * 2.0 +
            len(measurements) * 2.5 +
            len(fractions) * 1.5 +
            len(ranges) * 2.0
        )
        numeric_density = (numeric_score / max(word_count, 1)) * 100
        
        # ---- 3. Specificity vs Vagueness Markers ----
        vague_phrases = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are (?:many|various|several|some)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bfor the most part\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bprobably\b', r'\bmaybe\b', r'\bperhaps\b',
            r'\bi think\b', r'\bi guess\b', r'\bi suppose\b',
            r'\ba lot of\b', r'\bquite a few\b', r'\btons of\b',
            r'\bstuff\b', r'\bthings\b', r'\bwhatever\b',
            r'\betc\.?\b', r'\band so on\b', r'\band whatnot\b',
            r'\byou know\b', r'\bbasically\b',
            r'\bjust\b', r'\breally\b', r'\bvery\b',
            r'\bsomehow\b', r'\bsomewhat\b',
        ]
        
        specificity_phrases = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies\b',
            r'\bevidence\b', r'\bdata\b', r'\bstatistic\w*\b',
            r'\bexperiment\w*\b', r'\banalysis\b', r'\bsurvey\b',
            r'\bpublished\b', r'\bjournal\b', r'\breport\w*\b',
            r'\bcite[ds]?\b', r'\breference\w*\b',
            r'\bconcretely\b', r'\bprecisely\b', r'\bexactly\b',
            r'\bin fact\b', r'\bnotably\b',
        ]
        
        response_lower = response_text.lower()
        
        vague_count = sum(len(re.findall(p, response_lower)) for p in vague_phrases)
        specific_count = sum(len(re.findall(p, response_lower)) for p in specificity_phrases)
        
        # Ratio: positive means more specific, negative means more vague
        if vague_count + specific_count > 0:
            specificity_ratio = (specific_count - vague_count * 0.5) / (vague_count + specific_count + 1)
        else:
            specificity_ratio = 0.0
        
        # ---- 4. Technical / Domain Vocabulary Richness ----
        # Unique words ratio (type-token ratio) for longer words (likely domain-specific)
        long_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{7,}\b', response_text)]
        if long_words:
            unique_long = len(set(long_words))
            ttr_long = unique_long / len(long_words) if len(long_words) > 0 else 0
            long_word_density = len(long_words) / max(word_count, 1)
        else:
            ttr_long = 0
            long_word_density = 0
        
        # Words with special characters (technical terms, code, etc.)
        technical_tokens = re.findall(r'\b\w+[_./]\w+\b', response_text)
        code_blocks = re.findall(r'```[\s\S]*?```', response_text)
        inline_code = re.findall(r'`[^`]+`', response_text)
        
        technical_score = (
            len(technical_tokens) * 1.5 +
            len(code_blocks) * 5.0 +
            len(inline_code) * 2.0
        )
        technical_density = (technical_score / max(word_count, 1)) * 100
        
        # ---- 5. Structural Elaboration ----
        # Parenthetical explanations (show depth)
        parentheticals = re.findall(r'\([^)]{5,}\)', response_text)
        
        # Enumeration / structured listing
        numbered_items = re.findall(r'(?:^|\n)\s*\d+[.)]\s', response_text)
        bullet_items = re.findall(r'(?:^|\n)\s*[-*•]\s', response_text)
        
        # Conditional/nuanced statements (showing depth)
        conditional_patterns = [
            r'\bif\s+you\b', r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bon the other hand\b', r'\bthat said\b', r'\bnevertheless\b',
            r'\bin contrast\b', r'\bconversely\b',
        ]
        conditional_count = sum(len(re.findall(p, response_lower)) for p in conditional_patterns)
        
        # Causal/explanatory connectors
        causal_patterns = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bas a result\b', r'\bdue to\b', r'\bsince\b', r'\bconsequently\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
        ]
        causal_count = sum(len(re.findall(p, response_lower)) for p in causal_patterns)
        
        structural_score = (
            len(parentheticals) * 2.0 +
            len(numbered_items) * 1.5 +
            len(bullet_items) * 1.0 +
            conditional_count * 1.0 +
            causal_count * 1.5
        )
        structural_density = (structural_score / max(word_count, 1)) * 100
        
        # ---- 6. Sentence-level information density ----
        # Average content words per sentence (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'it',
            'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'whom',
        }
        
        content_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', response_text) 
                        if w.lower() not in stop_words and len(w) > 2]
        content_ratio = len(content_words) / max(word_count, 1)
        
        # Unique content word diversity
        if content_words:
            content_diversity = len(set(content_words)) / len(content_words)
        else:
            content_diversity = 0
        
        # ---- 7. Response length bonus (diminishing returns) ----
        # Longer responses tend to have more evidence, but with diminishing returns
        length_score = math.log(max(word_count, 1) + 1) / math.log(500)
        length_score = min(length_score, 1.5)  # Cap the bonus
        
        # ---- 8. Dialogue/narrative engagement (for creative tasks) ----
        # Detect if this is a creative/dialogue response
        dialogue_markers = re.findall(r'[*][^*]+[*]', response_text)  # action markers
        speech_marks = re.findall(r'(?:^|\n)\s*\w+:', response_text)  # Speaker: format
        
        engagement_score = min(len(dialogue_markers) * 1.5 + len(speech_marks) * 2.0, 10)
        
        # ---- COMPOSITE SCORING ----
        # Weight each dimension
        score = 0.0
        
        # Named entities (0-15 points)
        score += min(named_entity_density * 1.2, 15)
        
        # Numeric density (0-12 points)
        score += min(numeric_density * 3.0, 12)
        
        # Specificity ratio (-5 to 10 points)
        score += max(min(specificity_ratio * 10, 10), -5)
        
        # Technical vocabulary (0-10 points)
        score += min(technical_density * 2.0, 10)
        score += min(long_word_density * 8, 8)
        score += min(ttr_long * 5, 5)
        
        # Structural elaboration (0-12 points)
        score += min(structural_density * 3.0, 12)
        
        # Content density (0-10 points)
        score += content_ratio * 8
        score += content_diversity * 4
        
        # Length (0-10 points)
        score += length_score * 6.5
        
        # Engagement (0-5 points for creative responses)
        score += min(engagement_score * 0.5, 5)
        
        # ---- Penalty for extremely short responses ----
        if word_count < 20:
            score *= 0.4
        elif word_count < 40:
            score *= 0.7
        elif word_count < 60:
            score *= 0.85
        
        # ---- Penalty for pure meta-responses (just linking elsewhere) ----
        meta_patterns = [
            r'\byou might be interested in\b',
            r'\bplease read our rules\b',
            r'\bwelcome to\b',
            r'\bwhile you wait\b',
        ]
        meta_count = sum(1 for p in meta_patterns if re.search(p, response_lower))
        if meta_count > 0:
            score *= max(0.5, 1.0 - meta_count * 0.2)
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, score * 0.12))
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a middle score based on length
        try:
            return min(5.0, len(str(response).split()) * 0.05)
        except:
            return 2.5