def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    causal/logical chain analysis approach. This variant focuses on:
    1. Causal connective density (because, therefore, since, thus, hence, so that, etc.)
    2. Explicit reasoning markers (let's, we can, this means, note that, consider, etc.)
    3. Sequential reasoning signals (first...then...finally patterns)
    4. Question-answer self-dialogue patterns (rhetorical questions followed by answers)
    5. Intermediate conclusion markers (so, thus, this shows, this implies, etc.)
    6. Depth of explanation via clause complexity
    7. Ratio of "reasoning sentences" vs "assertion sentences"
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""

        import re
        import math

        text = response.strip()
        if len(text) < 10:
            return 0.0

        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences (rough but effective)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\d*#])', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        words = re.findall(r'[a-zA-Z]+', clean_text.lower())
        num_words = max(len(words), 1)
        
        score = 0.0

        # ============================================================
        # FEATURE 1: Causal connective density
        # These words explicitly show WHY something is the case
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis is because\b', r'\bthe reason\b', r'\bcaused by\b',
            r'\bleads to\b', r'\bresults in\b', r'\bwhich means\b',
            r'\bimplying\b', r'\bimplies\b', r'\bit follows\b',
            r'\bgiven that\b', r'\bassuming\b', r'\baccordingly\b',
        ]
        causal_count = 0
        lower_text = clean_text.lower()
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, lower_text))
        
        # Normalize by sentence count - reward density
        causal_density = causal_count / num_sentences
        # Score: up to 15 points
        score += min(causal_density * 12, 15.0)

        # ============================================================
        # FEATURE 2: Explicit reasoning/thinking process markers
        # Words that show the author is walking through their thinking
        # ============================================================
        reasoning_markers = [
            r"\blet's\b", r"\blet us\b", r"\bwe can\b", r"\bwe need\b",
            r"\bwe know\b", r"\bwe have\b", r"\bwe see\b",
            r"\bnote that\b", r"\bnotice that\b", r"\bobserve that\b",
            r"\bconsider\b", r"\brecall\b", r"\bremember that\b",
            r"\bthink about\b", r"\bthink of\b",
            r"\bthis means\b", r"\bthis tells us\b", r"\bthis shows\b",
            r"\bthis implies\b", r"\bthis indicates\b", r"\bthis suggests\b",
            r"\bin other words\b", r"\bput differently\b",
            r"\bto understand\b", r"\bto see why\b", r"\bto determine\b",
            r"\bto find\b", r"\bto calculate\b", r"\bto solve\b",
            r"\bby substituting\b", r"\bby applying\b", r"\bby using\b",
            r"\bif we\b", r"\bwhen we\b", r"\bonce we\b",
            r"\bbreak it down\b", r"\bbreak this down\b",
            r"\bstep by step\b", r"\bone by one\b",
            r"\bthe key\b", r"\bthe idea\b", r"\bthe point\b",
            r"\bin this case\b", r"\bin particular\b",
        ]
        reasoning_count = 0
        for pattern in reasoning_markers:
            reasoning_count += len(re.findall(pattern, lower_text))
        
        reasoning_density = reasoning_count / num_sentences
        # Score: up to 15 points
        score += min(reasoning_density * 10, 15.0)

        # ============================================================
        # FEATURE 3: Sequential reasoning signals
        # Detect ordered reasoning (first, second, next, then, finally)
        # Different from bullet detection - looks for inline sequencing
        # ============================================================
        seq_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bfourth(?:ly)?\b', r'\bnext\b', r'\bthen\b',
            r'\bfinally\b', r'\blastly\b', r'\bafterward\b',
            r'\bsubsequently\b', r'\bfollowing that\b',
            r'\bafter that\b', r'\bbefore that\b',
            r'\bto begin\b', r'\bto start\b', r'\bstarting with\b',
            r'\bmoving on\b', r'\bnow\b(?=.*(?:let|we|consider))',
        ]
        seq_count = 0
        for pattern in seq_markers:
            seq_count += len(re.findall(pattern, lower_text))
        
        # Check for numbered inline steps like "Step 1:", "1.", "1)"
        numbered_steps = len(re.findall(r'(?:step\s*\d|^\s*\d+[.)]\s)', clean_text, re.MULTILINE | re.IGNORECASE))
        seq_count += numbered_steps
        
        # Reward having multiple sequential markers (shows progression)
        if seq_count >= 2:
            seq_score = min(seq_count * 2.0, 12.0)
        else:
            seq_score = seq_count * 0.5
        score += seq_score

        # ============================================================
        # FEATURE 4: Reasoning sentence ratio
        # A "reasoning sentence" contains at least one causal/reasoning word
        # vs an "assertion sentence" that just states facts
        # ============================================================
        reasoning_sentence_patterns = (
            causal_connectives + reasoning_markers + 
            [r'\bif\b.*\bthen\b', r'\balthough\b', r'\bhowever\b', 
             r'\bon the other hand\b', r'\bwhile\b.*\b(?:also|but|however)\b']
        )
        
        reasoning_sentences = 0
        for sent in sentences:
            sent_lower = sent.lower()
            for pattern in reasoning_sentence_patterns:
                if re.search(pattern, sent_lower):
                    reasoning_sentences += 1
                    break
        
        reasoning_ratio = reasoning_sentences / num_sentences
        # Score: up to 12 points
        score += reasoning_ratio * 12.0

        # ============================================================
        # FEATURE 5: Clause complexity per sentence
        # More clauses per sentence = more nuanced reasoning
        # Measured by subordinating conjunctions and relative pronouns
        # ============================================================
        clause_markers = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b',
            r'\bwhere\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\beven though\b', r'\bwhereas\b', r'\bunless\b',
            r'\bprovided\b', r'\bwhether\b',
        ]
        clause_count = 0
        for pattern in clause_markers:
            clause_count += len(re.findall(pattern, lower_text))
        
        avg_clauses = clause_count / num_sentences
        # Moderate complexity is good (1-3 clauses per sentence on average)
        clause_score = min(avg_clauses * 3.0, 8.0)
        score += clause_score

        # ============================================================
        # FEATURE 6: Intermediate conclusion markers
        # Phrases that summarize mid-reasoning before moving on
        # ============================================================
        intermediate_markers = [
            r'\bso\b,?\s', r'\bso far\b', r'\bat this point\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin short\b',
            r'\boverall\b', r'\bin conclusion\b', r'\bto conclude\b',
            r'\bthe takeaway\b', r'\bthe key takeaway\b',
            r'\bwhat this means\b', r'\bthe bottom line\b',
            r'\bputting it all together\b', r'\bcombining\b',
            r'\bfrom this\b', r'\bfrom the above\b',
            r'\bbased on\b', r'\bgiven this\b', r'\bgiven these\b',
        ]
        intermediate_count = 0
        for pattern in intermediate_markers:
            intermediate_count += len(re.findall(pattern, lower_text))
        
        score += min(intermediate_count * 1.5, 8.0)

        # ============================================================
        # FEATURE 7: Self-dialogue / rhetorical question patterns
        # Questions posed and then answered show reasoning transparency
        # ============================================================
        # Count question marks in the response
        questions_in_response = clean_text.count('?')
        # Rhetorical questions followed by answers are great for reasoning
        # Pattern: sentence ending with ? followed by a sentence that answers
        qa_pairs = re.findall(r'\?\s+[A-Z][^.!?]*[.!]', clean_text)
        qa_score = min(len(qa_pairs) * 2.5, 8.0)
        # Also reward any questions as they show engagement with the problem
        qa_score += min(questions_in_response * 0.5, 3.0)
        score += qa_score

        # ============================================================
        # FEATURE 8: Explicit labeling of reasoning components
        # e.g., "Given:", "Find:", "Approach:", "Solution:", "Analysis:"
        # ============================================================
        component_labels = re.findall(
            r'(?:given|find|approach|solution|analysis|assumption|constraint|'
            r'observation|claim|proof|argument|evidence|conclusion|method|'
            r'reasoning|explanation|answer|result|calculation|formula|'
            r'definition|example|case|scenario|problem|strategy|plan|'
            r'identify|determine|evaluate|compare|contrast)\s*[:—\-]',
            lower_text
        )
        score += min(len(component_labels) * 2.0, 8.0)

        # ============================================================
        # FEATURE 9: Contrast and qualification language
        # Shows nuanced thinking, not just one-sided assertions
        # ============================================================
        contrast_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\bnonetheless\b', r'\bconversely\b', r'\bin contrast\b',
            r'\balternatively\b', r'\bbut\b', r'\byet\b',
            r'\bdespite\b', r'\bin spite of\b',
            r'\bwhile\s+(?:this|it|that)\b', r'\balthough\b',
            r'\beven so\b', r'\bthat said\b', r'\bhaving said that\b',
        ]
        contrast_count = 0
        for pattern in contrast_patterns:
            contrast_count += len(re.findall(pattern, lower_text))
        
        score += min(contrast_count * 1.5, 7.0)

        # ============================================================
        # FEATURE 10: Response length bonus (longer responses tend to
        # have more room for reasoning, but diminishing returns)
        # ============================================================
        length_score = math.log(max(num_words, 1) + 1) * 0.8
        score += min(length_score, 5.0)

        # ============================================================
        # FEATURE 11: Penalty for pure list-only responses with no
        # connecting reasoning (just items without explanation)
        # ============================================================
        # Count lines that are just short list items (< 15 words) with bullets/numbers
        lines = text.split('\n')
        short_list_items = 0
        total_content_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            total_content_lines += 1
            if re.match(r'^[\d\-\*•►▪]+[.):\s]', stripped):
                line_words = len(stripped.split())
                if line_words < 15:
                    short_list_items += 1
        
        if total_content_lines > 0:
            list_only_ratio = short_list_items / total_content_lines
            if list_only_ratio > 0.7:
                # Heavy penalty for responses that are almost entirely short list items
                score *= 0.6

        # ============================================================
        # FEATURE 12: Introductory framing / context setting
        # Good reasoning starts by framing the problem
        # ============================================================
        first_two_sentences = ' '.join(sentences[:2]).lower() if len(sentences) >= 2 else lower_text[:200]
        framing_patterns = [
            r'\bto (?:answer|address|solve|understand|tackle|approach)\b',
            r'\bthe (?:question|problem|issue|challenge|key)\b',
            r'\blet me\b', r"\blet's\b", r'\bi\'ll\b',
            r'\bhere\'s how\b', r'\bhere is\b',
            r'\bwe need to\b', r'\bwe should\b',
            r'\bthis (?:is|depends|requires|involves)\b',
            r'\bthere are (?:several|multiple|a few|many)\b',
        ]
        framing_count = 0
        for pattern in framing_patterns:
            if re.search(pattern, first_two_sentences):
                framing_count += 1
        
        score += min(framing_count * 1.5, 4.0)

        # ============================================================
        # Normalize to 0-100 range
        # Maximum theoretical: ~15+15+12+12+8+8+11+8+7+5+4 ≈ 105
        # Typical good: 40-70, typical bad: 10-30
        # ============================================================
        final_score = max(0.0, min(score, 100.0))
        
        return round(final_score, 2)

    except Exception:
        return 0.0