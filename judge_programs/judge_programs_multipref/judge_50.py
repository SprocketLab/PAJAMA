def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    sentence-level analysis approach. Analyzes the logical flow between
    sentences, presence of causal/explanatory connectors, and the ratio
    of reasoning sentences to assertion sentences.
    
    Different from Variant 1 (which uses bullet/list detection, header detection,
    paragraph analysis, transition words). This variant focuses on:
    - Sentence-level causal chain analysis
    - Explanation density (why/because/therefore patterns)
    - Reasoning verb detection
    - Progressive elaboration scoring
    - Question-response alignment depth
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.0
        
        # Split into sentences using a more careful approach
        def split_sentences(text):
            # Split on sentence-ending punctuation followed by space or end
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])$', text)
            # Also split on newlines that seem to separate thoughts
            expanded = []
            for p in parts:
                sub = re.split(r'\n+', p)
                expanded.extend(sub)
            # Filter out very short fragments
            return [s.strip() for s in expanded if len(s.strip()) > 5]
        
        sentences = split_sentences(response_stripped)
        num_sentences = max(len(sentences), 1)
        
        # === FEATURE 1: Causal/Explanatory Connector Density ===
        # These are words/phrases that show reasoning is being made explicit
        causal_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bthis implies\b', r'\bwhich means\b', r'\bwhich implies\b',
            r'\bso that\b', r'\bin order to\b', r'\bthe reason\b',
            r'\bthis is because\b', r'\bthis is why\b', r'\bthat\'s why\b',
            r'\bit follows\b', r'\bwe can conclude\b', r'\bwe can see\b',
            r'\bthis shows\b', r'\bthis demonstrates\b', r'\bthis indicates\b',
            r'\baccordingly\b', r'\bas such\b', r'\bgiven that\b',
            r'\bassuming\b', r'\bif we\b', r'\bwhen we\b',
        ]
        
        response_lower = response_stripped.lower()
        causal_count = 0
        for pat in causal_patterns:
            causal_count += len(re.findall(pat, response_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 8.0, 10.0)
        
        # === FEATURE 2: Reasoning Verb Presence ===
        # Verbs that indicate active reasoning/explanation
        reasoning_verbs = [
            r'\bconsider\b', r'\banalyze\b', r'\bexamine\b', r'\bevaluate\b',
            r'\bcalculate\b', r'\bdetermine\b', r'\bidentify\b', r'\bcompare\b',
            r'\bnote that\b', r'\bobserve\b', r'\brecall\b', r'\brecognize\b',
            r'\bunderstand\b', r'\bthink about\b', r'\breason\b',
            r'\bbreak down\b', r'\bbreak it down\b', r'\blet\'s\b',
            r'\bfirst,?\s+we\b', r'\bnext,?\s+we\b', r'\bthen,?\s+we\b',
            r'\bstart by\b', r'\bbegin by\b', r'\bwork through\b',
            r'\bfigure out\b', r'\bsolve\b', r'\bapply\b', r'\buse\b',
            r'\bsubstitut\b', r'\bplug\b', r'\bcompute\b',
            r'\bwe need to\b', r'\bwe can\b', r'\bwe know\b',
            r'\bwe have\b', r'\bwe get\b', r'\bwe find\b',
        ]
        
        reasoning_verb_count = 0
        for pat in reasoning_verbs:
            reasoning_verb_count += len(re.findall(pat, response_lower))
        
        reasoning_verb_density = reasoning_verb_count / num_sentences
        reasoning_verb_score = min(reasoning_verb_density * 6.0, 10.0)
        
        # === FEATURE 3: Progressive Elaboration ===
        # Check if sentences build on each other (referencing previous content)
        progressive_markers = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bsuch\b', r'\bthe above\b', r'\bmentioned\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\balso\b', r'\banother\b', r'\bon top of\b',
            r'\bbased on\b', r'\bbuilding on\b', r'\bfrom this\b',
            r'\bwith this\b', r'\bgiven this\b',
        ]
        
        progressive_sentences = 0
        for sent in sentences[1:]:  # Skip first sentence
            sent_lower = sent.lower()
            for pat in progressive_markers:
                if re.search(pat, sent_lower):
                    progressive_sentences += 1
                    break
        
        if num_sentences > 1:
            progressive_ratio = progressive_sentences / (num_sentences - 1)
        else:
            progressive_ratio = 0
        progressive_score = progressive_ratio * 10.0
        
        # === FEATURE 4: Intermediate Conclusion Markers ===
        # Phrases that signal intermediate conclusions being made visible
        intermediate_patterns = [
            r'\bso\b', r'\bso,\b', r'\btherefore\b', r'\bthus\b',
            r'\bwhich gives\b', r'\bwhich leads\b', r'\bresulting in\b',
            r'\bwe get\b', r'\bwe find\b', r'\bwe obtain\b',
            r'\bthis gives us\b', r'\bthis tells us\b', r'\bthis means\b',
            r'\bin other words\b', r'\bput differently\b',
            r'\bto summarize\b', r'\bin summary\b', r'\bin short\b',
            r'\bthe result is\b', r'\bthe answer is\b',
            r'\boverall\b', r'\bin conclusion\b',
            r'\b=\b',  # equations showing work
        ]
        
        intermediate_count = 0
        for pat in intermediate_patterns:
            intermediate_count += len(re.findall(pat, response_lower))
        
        intermediate_density = intermediate_count / num_sentences
        intermediate_score = min(intermediate_density * 5.0, 10.0)
        
        # === FEATURE 5: Explicit Labeling of Steps/Phases ===
        # Different from bullet detection - looking for linguistic step markers
        step_label_patterns = [
            r'\bstep\s*\d', r'\bstep\s*#?\s*\d', r'\bphase\s*\d',
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bfourth(?:ly)?\b', r'\bfifth(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bto begin\b', r'\bto start\b', r'\bafter that\b',
            r'\bonce\s+(?:you|we|this)\b', r'\bnow\s+(?:we|that|let)\b',
            r'\bat this point\b', r'\bmoving on\b',
        ]
        
        step_label_count = 0
        for pat in step_label_patterns:
            step_label_count += len(re.findall(pat, response_lower))
        
        step_density = step_label_count / num_sentences
        step_score = min(step_density * 7.0, 10.0)
        
        # === FEATURE 6: Explanation Depth via Sentence Length Variance ===
        # Good reasoning tends to have a mix of short concluding statements
        # and longer explanatory sentences, rather than uniform length
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variance is good (shows mix of explanation and conclusion)
            # Very low variance = monotonous, very high = disorganized
            cv = std_dev / max(mean_len, 1)  # coefficient of variation
            # Sweet spot around 0.4-0.8
            if cv < 0.1:
                depth_score = 2.0
            elif cv < 0.3:
                depth_score = 5.0
            elif cv < 0.6:
                depth_score = 8.0
            elif cv < 1.0:
                depth_score = 7.0
            else:
                depth_score = 4.0
        else:
            depth_score = 3.0
        
        # === FEATURE 7: Conversational Reasoning Engagement ===
        # Phrases that invite the reader to follow along
        engagement_patterns = [
            r'\blet\'s\b', r'\blet us\b', r'\bimagine\b', r'\bsuppose\b',
            r'\bconsider\b', r'\bnotice\b', r'\bkeep in mind\b',
            r'\bremember\b', r'\bthink of\b', r'\bpicture\b',
            r'\byou can see\b', r'\byou\'ll notice\b', r'\bas you can\b',
            r'\bhere\'s (?:why|how|what)\b', r'\bthe key\b',
            r'\bthe idea\b', r'\bthe point\b', r'\bthe trick\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bnotably\b',
            r'\bdive into\b', r'\blet\'s dive\b', r'\bwalk through\b',
        ]
        
        engagement_count = 0
        for pat in engagement_patterns:
            engagement_count += len(re.findall(pat, response_lower))
        
        engagement_score = min(engagement_count * 2.0, 10.0)
        
        # === FEATURE 8: Absence of Opaque Assertions ===
        # Penalize sentences that make claims without any reasoning markers
        # An "opaque assertion" is a declarative sentence with no reasoning words
        reasoning_indicators = set()
        all_reasoning_pats = causal_patterns + reasoning_verbs + intermediate_patterns + step_label_patterns + engagement_patterns
        
        opaque_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_reasoning = False
            for pat in all_reasoning_pats:
                if re.search(pat, sent_lower):
                    has_reasoning = True
                    break
            if not has_reasoning:
                # Check if it's a question (questions can be part of reasoning)
                if not sent.strip().endswith('?'):
                    opaque_count += 1
        
        opaque_ratio = opaque_count / num_sentences
        # Lower opaque ratio is better
        transparency_score = (1.0 - opaque_ratio) * 10.0
        
        # === FEATURE 9: Mathematical/Logical Notation ===
        # Presence of equations, formulas, logical symbols showing work
        math_patterns = [
            r'[=+\-*/^]', r'\d+\s*[×÷±]', r'\bx\s*=\b', r'\by\s*=\b',
            r'\d+\s*=\s*\d+', r'\(.*\d.*\)', r'\bif\b.*\bthen\b',
        ]
        
        math_count = 0
        for pat in math_patterns:
            math_count += len(re.findall(pat, response_stripped))
        
        # Normalize - just a small bonus
        math_score = min(math_count * 0.5, 5.0)
        
        # === FEATURE 10: Response Substantiveness ===
        # Very short responses can't show much reasoning
        word_count = len(response_stripped.split())
        if word_count < 20:
            length_factor = 0.3
        elif word_count < 50:
            length_factor = 0.6
        elif word_count < 100:
            length_factor = 0.8
        elif word_count < 300:
            length_factor = 1.0
        else:
            length_factor = 0.95  # Slight penalty for extreme length without proportional reasoning
        
        # === COMBINE SCORES ===
        # Weighted combination emphasizing the most important reasoning features
        weights = {
            'causal': 2.0,
            'reasoning_verb': 1.5,
            'progressive': 1.2,
            'intermediate': 1.5,
            'step': 1.8,
            'depth': 0.8,
            'engagement': 1.3,
            'transparency': 1.0,
            'math': 0.5,
        }
        
        total_weight = sum(weights.values())
        
        raw_score = (
            weights['causal'] * causal_score +
            weights['reasoning_verb'] * reasoning_verb_score +
            weights['progressive'] * progressive_score +
            weights['intermediate'] * intermediate_score +
            weights['step'] * step_score +
            weights['depth'] * depth_score +
            weights['engagement'] * engagement_score +
            weights['transparency'] * transparency_score +
            weights['math'] * math_score
        ) / total_weight
        
        # Apply length factor
        final_score = raw_score * length_factor
        
        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 0.0