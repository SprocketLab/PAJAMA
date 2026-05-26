def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    sentence-level reasoning chain analysis approach.
    
    This variant focuses on:
    1. Causal/logical connective density between sentences
    2. Progressive depth (do sentences build on each other?)
    3. Explicit reasoning markers (because, therefore, since, this means, etc.)
    4. Question-answer self-dialogue patterns
    5. Hedging and qualification (shows nuanced thinking)
    6. Evidence/example grounding per claim
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        import re
        import math
        
        text = response.strip()
        if len(text) < 20:
            return 0.5
        
        # Split into sentences (rough but effective)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\*\#\d])', text)
        # Also split on newlines that start new thoughts
        expanded = []
        for s in sentences:
            parts = re.split(r'\n\s*\n|\n(?=[A-Z\*\#\d\-])', s)
            expanded.extend([p.strip() for p in parts if p.strip()])
        sentences = expanded
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5
        
        # ---- Feature 1: Causal/Logical Connective Density ----
        # These are words/phrases that connect reasoning steps
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bwhich means\b',
            r'\bwhich leads to\b', r'\bit follows\b', r'\baccordingly\b',
            r'\bfor this reason\b', r'\bgiven that\b', r'\bgiven this\b',
            r'\bas such\b', r'\bthat\'s why\b', r'\bthis is because\b',
            r'\bthe reason\b', r'\bif\s+.+\s+then\b', r'\bso\b',
        ]
        
        text_lower = text.lower()
        causal_count = 0
        for pattern in causal_connectives:
            causal_count += len(re.findall(pattern, text_lower))
        
        # Normalize by number of sentences
        causal_density = causal_count / max(num_sentences, 1)
        causal_score = min(causal_density * 8, 10)  # scale to 0-10
        
        # ---- Feature 2: Progressive Elaboration Markers ----
        # Words that indicate building on previous points
        progressive_markers = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bafter that\b', r'\bfinally\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin addition\b', r'\bbeyond that\b', r'\bbuilding on\b',
            r'\bnow\b', r'\bonce\b', r'\bwith that\b',
            r'\bstep\s*\d', r'\bphase\s*\d', r'\bstage\s*\d',
        ]
        
        prog_count = 0
        for pattern in progressive_markers:
            prog_count += len(re.findall(pattern, text_lower))
        
        prog_density = prog_count / max(num_sentences, 1)
        prog_score = min(prog_density * 6, 10)
        
        # ---- Feature 3: Explanation Depth - "Why" behind claims ----
        # Look for explanatory patterns
        explanation_patterns = [
            r'\bthis is because\b', r'\bthe reason (?:is|being)\b',
            r'\bto understand (?:why|this|how)\b', r'\blet me explain\b',
            r'\bhere\'s why\b', r'\bthe idea (?:is|behind)\b',
            r'\bin other words\b', r'\bput (?:simply|differently)\b',
            r'\bwhat this means\b', r'\bto clarify\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b',
            r'\bconsider\b', r'\bimagine\b', r'\bsuppose\b',
            r'\bthink of\b', r'\blet\'s\b', r'\blet us\b',
            r'\bnote that\b', r'\bnotice that\b', r'\bobserve that\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bkey\b',
            r'\bwe (?:can|need|should|must|have)\b',
        ]
        
        explain_count = 0
        for pattern in explanation_patterns:
            explain_count += len(re.findall(pattern, text_lower))
        
        explain_density = explain_count / max(num_sentences, 1)
        explain_score = min(explain_density * 5, 10)
        
        # ---- Feature 4: Sentence-to-Sentence Coherence ----
        # Check if consecutive sentences reference each other (pronouns, "this", "that", etc.)
        reference_words = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bit\b', r'\bthey\b', r'\bsuch\b', r'\babove\b',
            r'\bprevious\b', r'\bmentioned\b', r'\bearlier\b',
        ]
        
        coherence_count = 0
        for i in range(1, len(sentences)):
            sent_lower = sentences[i].lower()
            # Check if sentence starts with or contains a reference to previous content
            first_30_chars = sent_lower[:60]
            for pattern in reference_words:
                if re.search(pattern, first_30_chars):
                    coherence_count += 1
                    break
        
        coherence_ratio = coherence_count / max(num_sentences - 1, 1)
        coherence_score = coherence_ratio * 10
        
        # ---- Feature 5: Intermediate Conclusion Markers ----
        # Phrases that signal intermediate conclusions before final answer
        intermediate_patterns = [
            r'\bso far\b', r'\bat this point\b', r'\bwe (?:now |can )?\s*(?:see|know|have|conclude)\b',
            r'\bthis (?:gives|tells|shows|reveals|indicates|suggests|confirms)\b',
            r'\bfrom (?:this|here|the above)\b', r'\bputting .+ together\b',
            r'\bcombining\b', r'\bsubstitut\w+\b', r'\bplug\w*\s*(?:in|into)\b',
            r'\bsolving\b', r'\bsimplif\w+\b', r'\bcalculat\w+\b',
            r'\bevaluat\w+\b', r'\bcomputing\b', r'\bresult\w*\b',
            r'\byield\w*\b', r'\bwhich (?:gives|is|equals|results)\b',
        ]
        
        intermediate_count = 0
        for pattern in intermediate_patterns:
            intermediate_count += len(re.findall(pattern, text_lower))
        
        intermediate_density = intermediate_count / max(num_sentences, 1)
        intermediate_score = min(intermediate_density * 6, 10)
        
        # ---- Feature 6: Structural Segmentation ----
        # Count distinct reasoning segments (numbered items, bold headers, etc.)
        numbered_items = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|\*\*(?:Step|Part|Phase)\s*\d)', text))
        bold_headers = len(re.findall(r'\*\*[^*]+\*\*', text))
        hash_headers = len(re.findall(r'#{1,4}\s+\S', text))
        
        segment_count = numbered_items + bold_headers * 0.5 + hash_headers
        # Diminishing returns
        segment_score = min(math.log1p(segment_count) * 3, 10)
        
        # ---- Feature 7: Hedging and Qualification (nuanced reasoning) ----
        hedge_patterns = [
            r'\bhowever\b', r'\balthough\b', r'\bwhile\b', r'\bbut\b',
            r'\bon the other hand\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bthat said\b', r'\bin contrast\b', r'\bdespite\b',
            r'\beven though\b', r'\bmay\b', r'\bmight\b', r'\bcould\b',
            r'\bpossibly\b', r'\bpotentially\b', r'\btypically\b',
            r'\bgenerally\b', r'\busually\b', r'\bit depends\b',
            r'\bnot necessarily\b', r'\bnot always\b',
        ]
        
        hedge_count = 0
        for pattern in hedge_patterns:
            hedge_count += len(re.findall(pattern, text_lower))
        
        hedge_density = hedge_count / max(num_sentences, 1)
        hedge_score = min(hedge_density * 5, 8)
        
        # ---- Feature 8: Sentence Length Variance ----
        # Good reasoning often mixes short summary sentences with longer explanatory ones
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        if len(sent_lengths) >= 3:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Moderate variance is good (not too uniform, not too wild)
            cv = std_dev / max(mean_len, 1)  # coefficient of variation
            # Sweet spot around 0.4-0.8
            if 0.3 <= cv <= 1.0:
                variance_score = 5
            elif cv < 0.3:
                variance_score = cv * 15  # too uniform
            else:
                variance_score = max(5 - (cv - 1.0) * 3, 1)
        else:
            variance_score = 2
        
        # ---- Feature 9: Invitational/Metacognitive Framing ----
        # "Let's", "Let me", "We can", etc. - shows guided reasoning
        meta_patterns = [
            r'\blet\'s\b', r'\blet me\b', r'\blet us\b',
            r'\bwe can\b', r'\bwe need\b', r'\bwe should\b',
            r'\bwe\'ll\b', r'\bwe have\b', r'\bwe know\b',
            r'\bto do this\b', r'\bto find\b', r'\bto solve\b',
            r'\bto determine\b', r'\bto calculate\b', r'\bto understand\b',
            r'\bour\b', r'\brecall\b', r'\bremember\b',
            r'\bbreak\w* (?:it |this )?down\b', r'\bwork through\b',
            r'\bstep by step\b', r'\bone by one\b',
        ]
        
        meta_count = 0
        for pattern in meta_patterns:
            meta_count += len(re.findall(pattern, text_lower))
        
        meta_score = min(meta_count * 1.5, 8)
        
        # ---- Feature 10: Response Length Appropriateness ----
        # Very short responses rarely show reasoning
        word_count = len(text.split())
        if word_count < 30:
            length_score = 1
        elif word_count < 60:
            length_score = 3
        elif word_count < 100:
            length_score = 5
        elif word_count < 200:
            length_score = 7
        elif word_count < 400:
            length_score = 8
        else:
            length_score = 7  # slight penalty for excessive length
        
        # ---- Feature 11: Opening Context/Framing ----
        # Good reasoning often starts by framing the problem
        first_100 = text_lower[:min(200, len(text_lower))]
        framing_patterns = [
            r'\bto (?:answer|address|solve|tackle|understand)\b',
            r'\bthis is a\b', r'\bthis question\b',
            r'\bthe (?:key|main|core|central|important)\b',
            r'\bgreat question\b', r'\bgood question\b',
            r'\bthere are (?:several|multiple|a few|many)\b',
            r'\bfirst,? (?:we|let|it)\b',
            r'\bidentif\w+\b', r'\bdefin\w+\b',
        ]
        
        framing_count = 0
        for pattern in framing_patterns:
            if re.search(pattern, first_100):
                framing_count += 1
        
        framing_score = min(framing_count * 2.5, 6)
        
        # ---- Weighted Combination ----
        weights = {
            'causal': 0.18,
            'progressive': 0.12,
            'explain': 0.15,
            'coherence': 0.10,
            'intermediate': 0.10,
            'segment': 0.08,
            'hedge': 0.05,
            'variance': 0.04,
            'meta': 0.08,
            'length': 0.05,
            'framing': 0.05,
        }
        
        scores = {
            'causal': causal_score,
            'progressive': prog_score,
            'explain': explain_score,
            'coherence': coherence_score,
            'intermediate': intermediate_score,
            'segment': segment_score,
            'hedge': hedge_score,
            'variance': variance_score,
            'meta': meta_score,
            'length': length_score,
            'framing': framing_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        return 3.0