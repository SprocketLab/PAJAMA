def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level logical flow analysis, causal/explanatory connective density,
    and progressive information building patterns.
    
    This variant focuses on:
    1. Causal/explanatory connective analysis (because, therefore, since, thus, etc.)
    2. Sentence-to-sentence information progression (new info introduced gradually)
    3. Clause complexity and subordination depth
    4. Question-response alignment through progressive elaboration
    5. Evidence of intermediate reasoning (conditional statements, qualifications)
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # Tokenize into sentences using multiple delimiters
        def split_sentences(text):
            # Split on sentence-ending punctuation, but be careful with abbreviations
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*', text)
            # Also split on newlines
            result = []
            for s in sents:
                sub = re.split(r'\n+', s)
                result.extend(sub)
            return [s.strip() for s in result if s.strip() and len(s.strip()) > 2]
        
        sentences = split_sentences(response_stripped)
        num_sentences = len(sentences)
        
        # Tokenize into words
        words = re.findall(r'[a-zA-Z]+', response_stripped.lower())
        num_words = len(words)
        
        if num_words < 1:
            return 0.5
        
        # ============================================================
        # FEATURE 1: Causal/Explanatory Connective Density
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bsince\b', r'\bas a result\b', r'\bconsequently\b',
            r'\bso that\b', r'\bdue to\b', r'\bowing to\b',
            r'\bthis means\b', r'\bthis implies\b', r'\bwhich means\b',
            r'\bfor this reason\b', r'\bthat is why\b', r'\baccordingly\b',
        ]
        
        explanatory_connectives = [
            r'\bin other words\b', r'\bspecifically\b', r'\bnamely\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bto illustrate\b', r'\bin particular\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bto clarify\b',
        ]
        
        sequential_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bfollowing this\b', r'\bin the first place\b',
            r'\bto begin\b', r'\bto start\b', r'\bstep\b',
            r'\binitially\b', r'\blastly\b', r'\bfurthermore\b',
            r'\bmoreover\b', r'\badditionally\b', r'\bin addition\b',
        ]
        
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bdespite\b',
            r'\bwhile\b', r'\bwhereas\b', r'\bbut\b', r'\byet\b',
            r'\binstead\b', r'\brather\b',
        ]
        
        response_lower = response_stripped.lower()
        
        def count_patterns(patterns):
            total = 0
            for p in patterns:
                total += len(re.findall(p, response_lower))
            return total
        
        causal_count = count_patterns(causal_connectives)
        explanatory_count = count_patterns(explanatory_connectives)
        sequential_count = count_patterns(sequential_connectives)
        contrastive_count = count_patterns(contrastive_connectives)
        
        total_connectives = causal_count + explanatory_count + sequential_count + contrastive_count
        
        # Normalize by word count - connective density
        connective_density = total_connectives / max(num_words, 1) * 100
        # Score: higher density = more transparent reasoning, cap at reasonable level
        connective_score = min(connective_density * 3.0, 10.0)
        
        # Bonus for variety of connective types used
        types_used = sum([1 for c in [causal_count, explanatory_count, sequential_count, contrastive_count] if c > 0])
        variety_bonus = types_used * 0.5  # 0 to 2.0
        
        # ============================================================
        # FEATURE 2: Sentence-to-Sentence Information Progression
        # ============================================================
        # Measure how much new information each sentence introduces
        # compared to previous sentences (progressive elaboration)
        
        progression_score = 0.0
        if num_sentences >= 2:
            prev_words_set = set()
            new_info_ratios = []
            for sent in sentences:
                sent_words = set(re.findall(r'[a-zA-Z]{3,}', sent.lower()))
                if len(sent_words) > 0:
                    if len(prev_words_set) == 0:
                        new_info_ratios.append(1.0)
                    else:
                        new_words = sent_words - prev_words_set
                        new_ratio = len(new_words) / len(sent_words)
                        new_info_ratios.append(new_ratio)
                    prev_words_set.update(sent_words)
            
            if new_info_ratios:
                # Ideal: each sentence introduces ~30-70% new info
                # Too high = disconnected, too low = repetitive
                good_progression = sum(1 for r in new_info_ratios if 0.2 <= r <= 0.85)
                progression_score = (good_progression / len(new_info_ratios)) * 10.0
        elif num_sentences == 1:
            # Single sentence - limited progression possible
            progression_score = 2.0
        
        # ============================================================
        # FEATURE 3: Clause Complexity and Subordination
        # ============================================================
        # Count subordinate clauses as evidence of nuanced reasoning
        
        subordinators = [
            r'\bif\b', r'\bwhen\b', r'\bwhile\b', r'\balthough\b',
            r'\bunless\b', r'\bprovided that\b', r'\bassuming\b',
            r'\bgiven that\b', r'\bin case\b', r'\beven though\b',
            r'\bwhether\b', r'\bwherever\b', r'\bwhenever\b',
        ]
        
        subordination_count = count_patterns(subordinators)
        # Average commas per sentence as proxy for clause complexity
        comma_count = response_stripped.count(',')
        avg_commas = comma_count / max(num_sentences, 1)
        
        # Subordination density
        sub_density = subordination_count / max(num_sentences, 1)
        clause_complexity_score = min((sub_density * 3.0 + min(avg_commas, 3.0) * 0.8), 10.0)
        
        # ============================================================
        # FEATURE 4: Explicit Reasoning Markers
        # ============================================================
        # Phrases that signal the author is showing their reasoning
        
        reasoning_markers = [
            r'\bthis is because\b', r'\bthe reason\b', r'\bthis suggests\b',
            r'\bwe can (?:see|conclude|infer|determine)\b',
            r'\bit follows that\b', r'\bfrom this\b',
            r'\bin order to\b', r'\bthe key (?:point|idea|insight)\b',
            r'\blet\'?s (?:consider|examine|look|think|analyze|break)\b',
            r'\bnote that\b', r'\bimportantly\b', r'\bcrucially\b',
            r'\bobserve that\b', r'\brecall that\b',
            r'\bto understand\b', r'\bto see why\b',
            r'\bthis shows\b', r'\bthis demonstrates\b',
            r'\bthis indicates\b', r'\bas we can see\b',
            r'\bput (?:simply|differently|another way)\b',
            r'\bin summary\b', r'\bto summarize\b', r'\bin conclusion\b',
            r'\boverall\b', r'\bin short\b',
        ]
        
        reasoning_marker_count = count_patterns(reasoning_markers)
        reasoning_marker_score = min(reasoning_marker_count * 1.5, 8.0)
        
        # ============================================================
        # FEATURE 5: Response Substantiveness and Coherence Length
        # ============================================================
        # Not just length, but meaningful length relative to query complexity
        
        query_words = re.findall(r'[a-zA-Z]+', query.lower())
        query_len = len(query_words)
        
        # Penalize extremely short responses
        if num_words <= 3:
            length_score = 0.5
        elif num_words <= 10:
            length_score = 2.0
        elif num_words <= 30:
            length_score = 4.0
        elif num_words <= 80:
            length_score = 6.0
        elif num_words <= 200:
            length_score = 8.0
        else:
            length_score = 7.5  # Very long can be less focused
        
        # ============================================================
        # FEATURE 6: Absence of Noise / Garbage Detection
        # ============================================================
        # Detect repetition, HTML artifacts, code dumps, nonsense
        
        noise_penalty = 0.0
        
        # Check for excessive repetition
        if num_sentences >= 3:
            sent_texts = [s.lower().strip() for s in sentences]
            unique_sents = set(sent_texts)
            repetition_ratio = len(unique_sents) / len(sent_texts)
            if repetition_ratio < 0.5:
                noise_penalty += 3.0
            elif repetition_ratio < 0.7:
                noise_penalty += 1.5
        
        # Check for HTML tags (not in query context)
        html_tags = re.findall(r'<[^>]+>', response_stripped)
        if len(html_tags) > 3 and 'html' not in query.lower() and 'tag' not in query.lower():
            noise_penalty += 2.0
        
        # Check for code-like content when not asked for code
        code_indicators = response_stripped.count('def ') + response_stripped.count('import ') + response_stripped.count('class ')
        if code_indicators > 2 and 'code' not in query.lower() and 'program' not in query.lower() and 'python' not in query.lower():
            noise_penalty += 2.0
        
        # Check for "Question:" / "Answer:" / "Input:" / "Output:" repetition patterns
        meta_patterns = len(re.findall(r'(?:Question|Answer|Input|Output)\s*:', response_stripped))
        if meta_patterns > 2:
            noise_penalty += 2.0
        
        # Word-level repetition (bigram repetition)
        if num_words > 10:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                most_common_freq = bigram_counts.most_common(1)[0][1]
                if most_common_freq > max(5, len(bigrams) * 0.1):
                    noise_penalty += 2.0
        
        # ============================================================
        # FEATURE 7: Multi-sentence Logical Flow
        # ============================================================
        # Check if sentences build on each other using anaphoric references
        
        anaphoric_refs = [
            r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
            r'\bit\b', r'\bsuch\b', r'\bthe above\b', r'\bas mentioned\b',
            r'\bas noted\b', r'\bas stated\b',
        ]
        
        anaphoric_count = 0
        if num_sentences >= 2:
            # Check sentences after the first one for anaphoric references
            for sent in sentences[1:]:
                sent_lower = sent.lower()
                for pattern in anaphoric_refs:
                    if re.search(pattern, sent_lower):
                        anaphoric_count += 1
                        break  # Count once per sentence
        
        if num_sentences >= 2:
            anaphoric_ratio = anaphoric_count / (num_sentences - 1)
        else:
            anaphoric_ratio = 0
        
        flow_score = min(anaphoric_ratio * 6.0, 5.0)
        
        # ============================================================
        # FEATURE 8: Qualification and Nuance
        # ============================================================
        # Detect hedging, qualification, and nuanced statements
        
        qualification_markers = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin most cases\b', r'\bit depends\b', r'\bmay\b',
            r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\blikely\b', r'\bunlikely\b', r'\btend to\b',
            r'\bnot necessarily\b', r'\bnot always\b',
            r'\bto some extent\b', r'\brelatively\b',
            r'\bapproximately\b', r'\broughly\b',
        ]
        
        qual_count = count_patterns(qualification_markers)
        qualification_score = min(qual_count * 0.8, 4.0)
        
        # ============================================================
        # COMBINE SCORES
        # ============================================================
        
        # Weighted combination
        raw_score = (
            connective_score * 0.20 +       # Causal/explanatory connectives
            variety_bonus * 0.10 +           # Variety of connective types
            progression_score * 0.15 +       # Information progression
            clause_complexity_score * 0.10 + # Clause complexity
            reasoning_marker_score * 0.10 +  # Explicit reasoning markers
            length_score * 0.15 +            # Substantiveness
            flow_score * 0.10 +              # Logical flow
            qualification_score * 0.10       # Nuance
        )
        
        # Apply noise penalty
        raw_score = max(raw_score - noise_penalty, 0.0)
        
        # Scale to 0-10 range
        # The theoretical max is around 10 but practically it's lower
        # Apply a mild sigmoid-like scaling to spread scores
        final_score = min(max(raw_score * 1.3, 0.0), 10.0)
        
        # Ensure minimum discrimination: very short/empty responses get low scores
        if num_words <= 2:
            final_score = min(final_score, 1.0)
        elif num_words <= 5:
            final_score = min(final_score, 2.5)
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: return a middling score
        try:
            if response and len(response.strip()) > 20:
                return 3.0
            return 1.0
        except Exception:
            return 1.0