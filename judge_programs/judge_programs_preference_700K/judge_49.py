def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    Focuses on:
    - Explicit reasoning steps and logical progression
    - Intermediate conclusions made visible
    - Explanations of 'why' behind claims
    - Ability for reader to follow and verify logic
    - Penalizes opaque, jump-to-conclusion answers
    
    Returns a score where HIGHER = BETTER quality.
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
        import string
        from collections import Counter
        
        score = 0.0
        
        # ============================================================
        # FEATURE 1: Response length and substance (0-15 points)
        # Longer responses tend to show more reasoning steps
        # ============================================================
        word_count = len(response_stripped.split())
        
        if word_count < 10:
            length_score = 0.0
        elif word_count < 25:
            length_score = 3.0
        elif word_count < 50:
            length_score = 6.0
        elif word_count < 100:
            length_score = 9.0
        elif word_count < 200:
            length_score = 12.0
        else:
            length_score = 15.0
        
        score += length_score
        
        # ============================================================
        # FEATURE 2: Sentence count and multi-sentence reasoning (0-10 points)
        # Multiple sentences suggest step-by-step explanation
        # ============================================================
        sentences = re.split(r'[.!?]+(?:\s|$)', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        sentence_count = len(sentences)
        
        if sentence_count <= 1:
            sentence_score = 0.0
        elif sentence_count == 2:
            sentence_score = 3.0
        elif sentence_count <= 4:
            sentence_score = 6.0
        elif sentence_count <= 7:
            sentence_score = 8.0
        else:
            sentence_score = 10.0
        
        score += sentence_score
        
        # ============================================================
        # FEATURE 3: Reasoning/transition markers (0-15 points)
        # Words that indicate logical flow and reasoning steps
        # ============================================================
        reasoning_markers = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bthis means\b', r'\bthis is because\b', r'\bthe reason\b',
            r'\bdue to\b', r'\bin other words\b', r'\bput differently\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bfor instance\b',
            r'\bfor example\b', r'\bsuch as\b', r'\blike\b',
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bwhile\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bnevertheless\b',
            r'\bif\b.*\bthen\b', r'\bassuming\b', r'\bgiven that\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bfinally\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\badditionally\b', r'\balso\b', r'\bin addition\b',
            r'\bso\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bin short\b', r'\bto summarize\b', r'\boverall\b',
            r'\bessentially\b', r'\bbasically\b', r'\bfundamentally\b',
        ]
        
        response_lower = response_stripped.lower()
        marker_count = 0
        for pattern in reasoning_markers:
            matches = re.findall(pattern, response_lower)
            marker_count += len(matches)
        
        # Normalize by word count to avoid pure length bias, then scale
        if word_count > 0:
            marker_density = marker_count / word_count
        else:
            marker_density = 0
        
        # Raw count matters too
        raw_marker_score = min(marker_count * 1.2, 10.0)
        density_marker_score = min(marker_density * 100, 5.0)
        reasoning_marker_score = raw_marker_score + density_marker_score
        reasoning_marker_score = min(reasoning_marker_score, 15.0)
        
        score += reasoning_marker_score
        
        # ============================================================
        # FEATURE 4: Causal/explanatory language (0-12 points)
        # Phrases that explain WHY something is the case
        # ============================================================
        causal_patterns = [
            r'\bbecause\b', r'\bthe reason (?:is|being|for)\b',
            r'\bthis is (?:because|due|why)\b', r'\bcaused by\b',
            r'\bleads to\b', r'\bresults in\b', r'\bexplains why\b',
            r'\bthe key (?:point|idea|insight|thing)\b',
            r'\bwhat this means\b', r'\bimportantly\b',
            r'\bthe trade-off\b', r'\bthe tradeoff\b',
            r'\bin practice\b', r'\bin theory\b',
            r'\btends to be\b', r'\btypically\b',
            r'\bwhy\b', r'\bhow\b',
            r'\bworks by\b', r'\bfunctions as\b',
        ]
        
        causal_count = 0
        for pattern in causal_patterns:
            causal_count += len(re.findall(pattern, response_lower))
        
        causal_score = min(causal_count * 2.0, 12.0)
        score += causal_score
        
        # ============================================================
        # FEATURE 5: Structured enumeration / listing (0-8 points)
        # Numbered lists, bullet points, or explicit step markers
        # ============================================================
        enumeration_patterns = [
            r'^\s*\d+[\.\)]\s', r'^\s*[-*•]\s', r'^\s*\w+\)\s',
            r'\bstep \d+\b', r'\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b',
            r'\b(?:1|one)\.\s', r'\b(?:2|two)\.\s',
        ]
        
        lines = response_stripped.split('\n')
        enum_count = 0
        for line in lines:
            for pattern in enumeration_patterns:
                if re.search(pattern, line.strip(), re.IGNORECASE):
                    enum_count += 1
                    break
        
        # Also check for sequential markers in text
        sequential_words = [r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bthen\b', r'\bnext\b', r'\bfinally\b', r'\blastly\b']
        seq_found = sum(1 for p in sequential_words if re.search(p, response_lower))
        
        enum_score = min(enum_count * 2.0 + seq_found * 1.0, 8.0)
        score += enum_score
        
        # ============================================================
        # FEATURE 6: Qualification and nuance (0-8 points)
        # Shows awareness of complexity, edge cases, caveats
        # ============================================================
        nuance_patterns = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bit depends\b', r'\bin some cases\b', r'\bnot always\b',
            r'\bsometimes\b', r'\boften\b', r'\brarely\b',
            r'\bmight\b', r'\bcould\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bto some extent\b', r'\bmore or less\b',
            r'\bon one hand\b', r'\bon the other\b',
            r'\bcaveat\b', r'\bexception\b', r'\bnuance\b',
            r'\bthat said\b', r'\bhaving said that\b',
            r'\bkeep in mind\b', r'\bnote that\b', r'\bworth noting\b',
            r'\bimportant to\b', r'\bplease be\b',
        ]
        
        nuance_count = 0
        for pattern in nuance_patterns:
            nuance_count += len(re.findall(pattern, response_lower))
        
        nuance_score = min(nuance_count * 1.5, 8.0)
        score += nuance_score
        
        # ============================================================
        # FEATURE 7: Elaboration depth - avg sentence length (0-7 points)
        # Very short sentences may indicate surface-level answers
        # ============================================================
        if sentences:
            avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sent_len < 5:
                depth_score = 1.0
            elif avg_sent_len < 10:
                depth_score = 3.0
            elif avg_sent_len < 18:
                depth_score = 5.5
            elif avg_sent_len < 25:
                depth_score = 7.0
            else:
                depth_score = 6.0  # Very long sentences can be hard to follow
        else:
            depth_score = 0.0
        
        score += depth_score
        
        # ============================================================
        # FEATURE 8: Concrete examples and evidence (0-8 points)
        # References to specific cases, examples, analogies
        # ============================================================
        example_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.\b', r'\bi\.e\.\b', r'\bconsider\b',
            r'\bimagine\b', r'\bsuppose\b', r'\blet\'s say\b',
            r'\bin my experience\b', r'\bfrom experience\b',
            r'\banalog(?:y|ous)\b', r'\bsimilar to\b',
            r'\blike when\b', r'\bthink of\b',
            r'\bspecific(?:ally)?\b', r'\bconcrete\b',
        ]
        
        example_count = 0
        for pattern in example_patterns:
            example_count += len(re.findall(pattern, response_lower))
        
        example_score = min(example_count * 2.5, 8.0)
        score += example_score
        
        # ============================================================
        # FEATURE 9: Query engagement / relevance signals (0-7 points)
        # Does the response engage with the query's key terms?
        # ============================================================
        query_lower = query.lower()
        # Extract meaningful words from query (>3 chars, not stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'would', 'could', 'should', 'will',
                     'what', 'when', 'where', 'which', 'who', 'how', 'that',
                     'this', 'with', 'from', 'they', 'were', 'their', 'there',
                     'about', 'more', 'some', 'them', 'than', 'into', 'does',
                     'like', 'just', 'very', 'much', 'most', 'also', 'being',
                     'other', 'your'}
        
        query_words = set()
        for w in re.findall(r'\b[a-z]+\b', query_lower):
            if len(w) > 3 and w not in stopwords:
                query_words.add(w)
        
        if query_words:
            matched = sum(1 for w in query_words if w in response_lower)
            overlap_ratio = matched / len(query_words)
            relevance_score = overlap_ratio * 7.0
        else:
            relevance_score = 3.5  # neutral
        
        score += relevance_score
        
        # ============================================================
        # FEATURE 10: Multi-perspective / comparative reasoning (0-7 points)
        # Shows multiple viewpoints or considers alternatives
        # ============================================================
        perspective_patterns = [
            r'\bboth\b', r'\beither\b', r'\bneither\b',
            r'\bon one hand\b', r'\bon the other\b',
            r'\balternatively\b', r'\banother (?:way|perspective|view|approach)\b',
            r'\bsome (?:people|argue|say|think|believe)\b',
            r'\bothers (?:argue|say|think|believe)\b',
            r'\bwhile some\b', r'\bconversely\b',
            r'\bpros and cons\b', r'\badvantages and disadvantages\b',
            r'\bin contrast\b', r'\bcompared to\b',
            r'\bdepends on\b', r'\bvaries\b',
        ]
        
        perspective_count = 0
        for pattern in perspective_patterns:
            perspective_count += len(re.findall(pattern, response_lower))
        
        perspective_score = min(perspective_count * 2.0, 7.0)
        score += perspective_score
        
        # ============================================================
        # FEATURE 11: Paragraph structure (0-5 points)
        # Multiple paragraphs suggest organized thought
        # ============================================================
        paragraphs = [p.strip() for p in response_stripped.split('\n\n') if p.strip() and len(p.strip()) > 20]
        if len(paragraphs) >= 4:
            para_score = 5.0
        elif len(paragraphs) >= 3:
            para_score = 4.0
        elif len(paragraphs) >= 2:
            para_score = 3.0
        elif len(lines) >= 3:
            para_score = 2.0
        else:
            para_score = 0.5
        
        score += para_score
        
        # ============================================================
        # PENALTY: Opaque / dismissive responses (-0 to -8 points)
        # ============================================================
        opaque_penalty = 0.0
        
        # Very short responses that just state a conclusion
        if word_count < 20 and marker_count < 2:
            opaque_penalty += 5.0
        
        # Responses that are just links or references without explanation
        if re.search(r'^(?:https?://|www\.)', response_stripped) and word_count < 15:
            opaque_penalty += 4.0
        
        # Purely meta/administrative responses (like mod messages)
        meta_patterns = [r'\bplease read our rules\b', r'\byour (?:comment|post) (?:was|has been) removed\b',
                        r'\bwelcome to /r/\b']
        for pattern in meta_patterns:
            if re.search(pattern, response_lower):
                opaque_penalty += 6.0
                break
        
        # Single declarative statement with no elaboration
        if sentence_count <= 1 and word_count < 30:
            opaque_penalty += 3.0
        
        score -= opaque_penalty
        
        # ============================================================
        # BONUS: Explicit reasoning chain markers (0-5 points)
        # ============================================================
        chain_patterns = [
            r'\bif\b.{5,60}\bthen\b',
            r'\bwhen\b.{5,60}\bthen\b',
            r'\bso\b.{3,40}\bbecause\b',
            r'\bsince\b.{5,60}\btherefore\b',
            r'\bgiven\b.{5,60}\bwe can\b',
        ]
        
        chain_count = 0
        for pattern in chain_patterns:
            chain_count += len(re.findall(pattern, response_lower))
        
        chain_bonus = min(chain_count * 2.5, 5.0)
        score += chain_bonus
        
        # ============================================================
        # Normalize to 0-100 range
        # Max theoretical ≈ 15+10+15+12+8+8+7+8+7+7+5+5 = 107
        # ============================================================
        score = max(0.0, score)
        score = min(score, 100.0)
        
        return round(score, 2)
    
    except Exception as e:
        # Fallback: return a neutral score based on length
        try:
            return min(len(str(response).split()) * 0.3, 50.0)
        except:
            return 25.0