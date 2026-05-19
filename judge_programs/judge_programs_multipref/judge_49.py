def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    Higher scores indicate better quality in showing reasoning process.
    
    This variant focuses on:
    - Structural markers of step-by-step reasoning
    - Explicit logical connectives and transitional phrases
    - Presence of intermediate conclusions
    - Explanation depth (why behind claims)
    - Reader-followable logic flow
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_lower = response.lower()
        response_stripped = response.strip()
        
        if len(response_stripped) < 10:
            return 0.0
        
        score = 0.0
        
        # === 1. STEP-BY-STEP STRUCTURAL MARKERS (0-20 points) ===
        import re
        
        # Numbered steps (1. 2. 3. or Step 1, Step 2, etc.)
        numbered_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)', response_lower)
        num_steps = len(numbered_pattern)
        step_score = min(num_steps * 2.5, 12.0)
        
        # Bullet points or dashes as list items
        bullet_pattern = re.findall(r'(?:^|\n)\s*[-•\*]\s+\S', response)
        bullet_count = len(bullet_pattern)
        step_score += min(bullet_count * 1.0, 5.0)
        
        # Bold/markdown headers indicating sections
        header_pattern = re.findall(r'(?:\*\*[^*]+\*\*|###?\s+\S|#{1,3}\s+\d)', response)
        header_count = len(header_pattern)
        step_score += min(header_count * 1.5, 5.0)
        
        score += min(step_score, 20.0)
        
        # === 2. LOGICAL CONNECTIVES AND REASONING WORDS (0-20 points) ===
        # Words that signal reasoning process
        reasoning_markers = [
            'because', 'therefore', 'thus', 'hence', 'since',
            'as a result', 'consequently', 'this means', 'this implies',
            'it follows', 'given that', 'due to', 'the reason',
            'which means', 'which leads', 'so that', 'in order to',
            'this is because', 'the key reason', 'this suggests'
        ]
        
        reasoning_count = 0
        for marker in reasoning_markers:
            reasoning_count += response_lower.count(marker)
        
        reasoning_score = min(reasoning_count * 2.0, 12.0)
        
        # Transitional/sequencing phrases
        transition_markers = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'to begin', 'moving on', 'additionally', 'furthermore',
            'moreover', 'in addition', 'on the other hand', 'however',
            'in contrast', 'meanwhile', 'subsequently', 'lastly',
            'to start', 'after that', 'following this', 'at this point'
        ]
        
        transition_count = 0
        for marker in transition_markers:
            transition_count += response_lower.count(marker)
        
        reasoning_score += min(transition_count * 1.5, 8.0)
        
        score += min(reasoning_score, 20.0)
        
        # === 3. INTERMEDIATE CONCLUSIONS AND EXPLANATIONS (0-15 points) ===
        # Phrases that indicate intermediate conclusions
        intermediate_markers = [
            'this tells us', 'we can see that', 'this shows',
            'from this', 'we can conclude', 'this indicates',
            'note that', 'notice that', 'importantly',
            'the key point', 'in other words', 'to put it',
            'what this means', 'the takeaway', 'so we know',
            'we now have', 'at this stage', 'so far',
            'let\'s', 'let us', 'now we', 'now that we',
            'having established', 'with this in mind',
            'keep in mind', 'remember that'
        ]
        
        intermediate_count = 0
        for marker in intermediate_markers:
            intermediate_count += response_lower.count(marker)
        
        score += min(intermediate_count * 2.5, 15.0)
        
        # === 4. EXPLANATORY DEPTH - "WHY" EXPLANATIONS (0-12 points) ===
        # Direct why-explanations
        why_patterns = [
            r'\bwhy\b', r'\bthe reason\b', r'\bthis is because\b',
            r'\bexplain\b', r'\bexplanation\b', r'\bunderstand\b',
            r'\breasoning\b', r'\brationale\b', r'\bjustif'
        ]
        
        why_count = 0
        for pat in why_patterns:
            why_count += len(re.findall(pat, response_lower))
        
        score += min(why_count * 2.0, 12.0)
        
        # === 5. STRUCTURAL ORGANIZATION QUALITY (0-15 points) ===
        # Paragraph count (well-organized responses have multiple paragraphs)
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        para_count = len(paragraphs)
        
        if para_count >= 3:
            score += 5.0
        elif para_count >= 2:
            score += 2.5
        
        # Sentence count and variety (more sentences = more detailed reasoning)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sent_count = len(sentences)
        
        if sent_count >= 8:
            score += 5.0
        elif sent_count >= 5:
            score += 3.0
        elif sent_count >= 3:
            score += 1.5
        
        # Line breaks indicating structured content
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        line_count = len(lines)
        if line_count >= 6:
            score += 5.0
        elif line_count >= 4:
            score += 3.0
        elif line_count >= 2:
            score += 1.0
        
        # === 6. ENGAGEMENT AND READER-GUIDANCE (0-10 points) ===
        engagement_markers = [
            'let\'s break', 'let\'s dive', 'let\'s look',
            'let\'s start', 'let\'s consider', 'let\'s explore',
            'here\'s how', 'here are', 'here is',
            'to summarize', 'in summary', 'to wrap up',
            'the bottom line', 'in conclusion',
            'you can see', 'as you can', 'you\'ll notice',
            'think of it', 'imagine', 'consider this',
            'for example', 'for instance', 'such as',
            'specifically', 'in particular', 'namely'
        ]
        
        engagement_count = 0
        for marker in engagement_markers:
            engagement_count += response_lower.count(marker)
        
        score += min(engagement_count * 2.0, 10.0)
        
        # === 7. RESPONSE COMPLETENESS SIGNAL (0-5 points) ===
        # Responses that seem to have a proper conclusion
        words = response_lower.split()
        word_count = len(words)
        
        # Reasonable length bonus (not too short, not just cut off)
        if word_count >= 80:
            score += 3.0
        elif word_count >= 50:
            score += 1.5
        
        # Check if response seems to end mid-sentence (truncated)
        last_char = response_stripped[-1] if response_stripped else ''
        if last_char in '.!?':
            score += 2.0
        elif last_char in ':,':
            score -= 1.0  # Likely truncated
        
        # === 8. PENALTY FOR OPAQUE/FLAT RESPONSES (subtract up to -8) ===
        # If response is just a flat list without explanation
        if num_steps == 0 and bullet_count == 0 and header_count == 0:
            if reasoning_count < 2 and transition_count < 2:
                score -= 5.0  # Very flat, no structure or reasoning shown
        
        # If response jumps to conclusion without any build-up
        if sent_count <= 2 and reasoning_count == 0:
            score -= 3.0
        
        # === 9. MATHEMATICAL/ANALYTICAL REASONING BONUS (0-8 points) ===
        # For technical queries, showing calculations step by step
        math_markers = [
            r'=\s*[\d\(]', r'\bcalculat', r'\bsubstitut',
            r'\bsolv', r'\bequation', r'\bformula',
            r'\bplug\s+in', r'\bapply', r'\busing\b',
            r'\bwe\s+get\b', r'\bwe\s+find\b', r'\bwe\s+have\b',
            r'\bresult\b', r'\byields?\b'
        ]
        
        math_count = 0
        for pat in math_markers:
            math_count += len(re.findall(pat, response_lower))
        
        if math_count >= 3:
            score += min(math_count * 1.5, 8.0)
        
        # === 10. CONTEXTUAL FRAMING (0-5 points) ===
        # Does the response set up context before diving in?
        first_100_chars = response_lower[:min(150, len(response_lower))]
        framing_starters = [
            'great question', 'that\'s a', 'this is a',
            'to answer', 'to understand', 'before we',
            'let me', 'i\'ll', 'we need to', 'the key to',
            'when it comes to', 'there are several',
            'a classic', 'interesting', 'good question',
            'certainly', 'absolutely', 'of course'
        ]
        
        for marker in framing_starters:
            if marker in first_100_chars:
                score += 1.5
                break
        
        # Introduction that sets expectations
        if any(phrase in first_100_chars for phrase in ['here are', 'here\'s', 'let\'s break', 'step by step', 'steps to']):
            score += 2.0
        
        # Mentioning the approach/method before executing
        approach_markers = ['approach', 'method', 'strategy', 'way to', 'process', 'technique']
        if any(m in first_100_chars for m in approach_markers):
            score += 1.5
        
        # Normalize to 0-100 range and clamp
        final_score = max(0.0, min(score, 100.0))
        
        return round(final_score, 2)
    
    except Exception:
        return 0.0