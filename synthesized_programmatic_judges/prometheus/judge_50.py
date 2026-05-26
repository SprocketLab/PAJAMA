def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation in LLM responses.
    
    This variant focuses on:
    - Causal/logical connective density (because, therefore, since, thus, etc.)
    - Transition words that signal reasoning progression
    - Question acknowledgment and reframing patterns
    - Explanation depth via subordinate clause analysis
    - Ratio of explanatory vs declarative sentences
    - Progressive elaboration detection (building on previous points)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not query or not response:
            return 1.0
        
        if len(response.strip()) < 20:
            return 1.0
        
        response_lower = response.lower()
        words = response_lower.split()
        word_count = len(words)
        
        if word_count < 5:
            return 1.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # ---- Feature 1: Causal/logical connectives ----
        # Words that show WHY something is the case or connect reasoning steps
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bdue to\b', r'\bthis means\b', r'\bwhich means\b',
            r'\bso that\b', r'\bin order to\b', r'\bthis is because\b',
            r'\bthe reason\b', r'\bthis leads to\b', r'\bit follows\b',
            r'\bgiven that\b', r'\bowing to\b', r'\bas such\b',
            r'\bthis is due to\b', r'\bthis is why\b',
        ]
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, response_lower))
        
        causal_density = causal_count / num_sentences
        causal_score = min(causal_density * 8.0, 3.0)
        
        # ---- Feature 2: Transition/progression markers ----
        # Words that signal step-wise progression in reasoning
        transition_markers = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bafter that\b', r'\bfinally\b', r'\blastly\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\balso\b', r'\bin addition\b', r'\bon top of that\b',
            r'\bfirst of all\b', r'\bto begin\b', r'\bto start\b',
            r'\bonce\b', r'\bnow\b', r'\bat this point\b',
            r'\bstep\b', r'\bhere\b', r'\blet\'s\b', r'\blets\b',
        ]
        transition_count = 0
        for pattern in transition_markers:
            transition_count += len(re.findall(pattern, response_lower))
        
        transition_density = transition_count / num_sentences
        transition_score = min(transition_density * 4.0, 2.5)
        
        # ---- Feature 3: Elaboration/explanation markers ----
        # Phrases that indicate the response is explaining or clarifying
        elaboration_markers = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bin other words\b', r'\bthat is\b', r'\bi\.e\.\b',
            r'\bspecifically\b', r'\bto clarify\b', r'\bto put it\b',
            r'\bthink of\b', r'\bimagine\b', r'\bconsider\b',
            r'\blike\b.*\bswitch\b', r'\banalog\b', r'\bmetaphor\b',
            r'\bin particular\b', r'\bto illustrate\b', r'\bjust like\b',
            r'\bpicture\b', r'\bsuppose\b', r'\blet me explain\b',
            r'\bwhat this means\b', r'\bput simply\b', r'\bsimply put\b',
        ]
        elaboration_count = 0
        for pattern in elaboration_markers:
            elaboration_count += len(re.findall(pattern, response_lower))
        
        elaboration_score = min(elaboration_count * 1.2, 2.5)
        
        # ---- Feature 4: Conditional/hypothetical reasoning ----
        conditional_markers = [
            r'\bif\b', r'\bwhen\b', r'\bwhenever\b', r'\bunless\b',
            r'\bprovided that\b', r'\bassuming\b', r'\bin case\b',
            r'\bwould\b', r'\bcould\b', r'\bmight\b', r'\bshould\b',
            r'\bwhat if\b', r'\beven if\b',
        ]
        conditional_count = 0
        for pattern in conditional_markers:
            conditional_count += len(re.findall(pattern, response_lower))
        
        conditional_density = conditional_count / num_sentences
        conditional_score = min(conditional_density * 2.5, 1.5)
        
        # ---- Feature 5: Subordinate clause complexity ----
        # Count commas and subordinating conjunctions as proxy for complex sentences
        # that show reasoning chains
        subordinators = [
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\beven though\b',
            r'\bwhether\b', r'\bso that\b', r'\bin order that\b',
            r'\bbefore\b', r'\bafter\b', r'\buntil\b',
        ]
        subordinate_count = 0
        for pattern in subordinators:
            subordinate_count += len(re.findall(pattern, response_lower))
        
        # Comma density as proxy for clause complexity
        comma_count = response.count(',')
        comma_density = comma_count / num_sentences
        clause_complexity_score = min((subordinate_count * 0.5 + comma_density * 0.4), 1.5)
        
        # ---- Feature 6: Acknowledgment and empathy markers ----
        # Shows the response engages with the query rather than jumping to conclusions
        acknowledgment_markers = [
            r'\bi understand\b', r'\bi can see\b', r'\bthat\'s\b.*\bunderstandable\b',
            r'\bit\'s\b.*\bokay\b', r'\bit\'s\b.*\bnatural\b', r'\bit\'s\b.*\bnormal\b',
            r'\bi hear\b', r'\bi\'m sorry\b', r'\bcompletely understandable\b',
            r'\bperfectly\b.*\bfine\b', r'\bperfectly\b.*\bokay\b',
            r'\babsolutely\b.*\bokay\b', r'\byou\'re right\b',
            r'\bgood question\b', r'\bgreat question\b',
            r'\blet me\b', r'\blet\'s\b',
            r'\byou might\b', r'\byou could\b', r'\byou may\b',
        ]
        ack_count = 0
        for pattern in acknowledgment_markers:
            ack_count += len(re.findall(pattern, response_lower))
        
        ack_score = min(ack_count * 0.8, 2.0)
        
        # ---- Feature 7: Contrast and nuance markers ----
        # Shows balanced reasoning, considering multiple sides
        contrast_markers = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\bbut\b', r'\byet\b', r'\binstead\b', r'\brather\b',
            r'\bdespite\b', r'\bin contrast\b', r'\balternatively\b',
            r'\bthat said\b', r'\bhaving said that\b', r'\bstill\b',
            r'\bnonetheless\b',
        ]
        contrast_count = 0
        for pattern in contrast_markers:
            contrast_count += len(re.findall(pattern, response_lower))
        
        contrast_score = min(contrast_count * 0.7, 1.5)
        
        # ---- Feature 8: Imperative vs explanatory sentence ratio ----
        # Opaque responses tend to be mostly imperative ("Do X", "Try Y")
        # Good responses mix imperatives with explanations
        imperative_starters = [
            r'^(do|try|make|get|go|call|take|use|buy|find|keep|start|stop|just|remember)\b'
        ]
        explanatory_starters = [
            r'^(this|it|the|a|an|that|these|those|by|when|if|since|because|as|while|here|now|imagine|think|consider|knowing|understanding)\b'
        ]
        
        imp_count = 0
        exp_count = 0
        for s in sentences:
            s_lower = s.strip().lower()
            for pat in imperative_starters:
                if re.match(pat, s_lower):
                    imp_count += 1
                    break
            for pat in explanatory_starters:
                if re.match(pat, s_lower):
                    exp_count += 1
                    break
        
        if num_sentences > 0:
            exp_ratio = exp_count / num_sentences
            imp_ratio = imp_count / num_sentences
            # Reward explanatory, mildly penalize purely imperative
            explanation_ratio_score = exp_ratio * 2.0 - imp_ratio * 0.5
            explanation_ratio_score = max(min(explanation_ratio_score, 1.5), -0.5)
        else:
            explanation_ratio_score = 0.0
        
        # ---- Feature 9: Progressive reference (referring back to earlier points) ----
        # Detect words like "this", "these", "that" at sentence beginnings which
        # indicate building on previous reasoning
        progressive_refs = 0
        for s in sentences:
            s_lower = s.strip().lower()
            if re.match(r'^(this|these|that|those|such|the above|as mentioned|as noted|as we)', s_lower):
                progressive_refs += 1
        
        progressive_score = min((progressive_refs / num_sentences) * 4.0, 1.5)
        
        # ---- Feature 10: Numbered/ordered reasoning (not bullet detection, but inline ordering) ----
        # Detect inline ordering like "first... second... third..." or "1)... 2)..."
        inline_order_patterns = re.findall(r'\b(?:first|firstly|second|secondly|third|thirdly|fourth|fifth|finally|lastly)\b', response_lower)
        numbered_patterns = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', response)
        
        ordering_count = len(inline_order_patterns) + len(numbered_patterns)
        ordering_score = min(ordering_count * 0.6, 2.0)
        
        # ---- Feature 11: Response length adequacy ----
        # Very short responses are likely opaque; moderate length is good
        length_score = 0.0
        if word_count < 30:
            length_score = -0.5
        elif word_count < 60:
            length_score = 0.5
        elif word_count < 150:
            length_score = 1.0
        elif word_count < 300:
            length_score = 0.8
        else:
            length_score = 0.6  # Very long might be verbose without reasoning
        
        # ---- Feature 12: Dismissive/opaque language penalty ----
        dismissive_patterns = [
            r'\bjust\b.*\bdo\b', r'\bjust\b.*\bget\b', r'\bjust\b.*\btry\b',
            r'\bmaybe you\'re\b.*\bnot\b', r'\byou should be able\b',
            r'\bit\'s not that\b.*\bhard\b', r'\bjust keep\b',
            r'\bget over it\b', r'\bmove on\b', r'\bget yourself together\b',
        ]
        dismissive_count = 0
        for pattern in dismissive_patterns:
            dismissive_count += len(re.findall(pattern, response_lower))
        
        dismissive_penalty = min(dismissive_count * 0.6, 2.0)
        
        # ---- Feature 13: Query engagement ----
        # Check if response addresses specific terms from the query
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query_lower))
        response_words_set = set(re.findall(r'\b[a-z]{4,}\b', response_lower))
        
        # Remove very common words
        common_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
                       'their', 'will', 'would', 'could', 'should', 'about', 'which',
                       'when', 'what', 'there', 'your', 'more', 'some', 'than', 'them',
                       'other', 'into', 'very', 'just', 'also', 'most', 'only'}
        query_content_words = query_words - common_words
        
        if query_content_words:
            overlap = len(query_content_words & response_words_set) / len(query_content_words)
            engagement_score = min(overlap * 1.5, 1.0)
        else:
            engagement_score = 0.5
        
        # ---- Feature 14: Sentence variety (mix of short and long) ----
        # Good reasoning often has varied sentence lengths
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variety is good (not all same length, not wildly different)
            if 3 <= std_dev <= 15:
                variety_score = 0.8
            elif std_dev > 15:
                variety_score = 0.4
            else:
                variety_score = 0.3
        else:
            variety_score = 0.2
        
        # ---- Combine all features ----
        raw_score = (
            causal_score * 1.2 +          # Max ~3.6
            transition_score * 1.0 +       # Max ~2.5
            elaboration_score * 1.0 +      # Max ~2.5
            conditional_score * 0.8 +      # Max ~1.2
            clause_complexity_score * 0.8 + # Max ~1.2
            ack_score * 0.9 +              # Max ~1.8
            contrast_score * 0.8 +         # Max ~1.2
            explanation_ratio_score * 1.0 + # Max ~1.5
            progressive_score * 0.9 +      # Max ~1.35
            ordering_score * 1.0 +         # Max ~2.0
            length_score * 0.8 +           # Max ~0.8
            engagement_score * 0.7 +       # Max ~0.7
            variety_score * 0.5 +          # Max ~0.4
            - dismissive_penalty * 1.0     # Penalty
        )
        
        # Normalize to 1-5 scale
        # Theoretical max is roughly ~20, but practical max is ~10-12
        # Map roughly: 0 -> 1, 10 -> 5
        normalized = 1.0 + (raw_score / 10.0) * 4.0
        
        # Clamp to [1, 5]
        final_score = max(1.0, min(5.0, normalized))
        
        return round(final_score, 2)
    
    except Exception:
        return 2.5