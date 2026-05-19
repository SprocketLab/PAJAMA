def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a SENTENCE-LEVEL analysis approach:
    - Analyzes each sentence for epistemic stance (certain, hedged, speculative)
    - Measures the distribution of epistemic stances across the response
    - Evaluates whether the query topic warrants uncertainty
    - Checks for nuanced reasoning patterns (acknowledging multiple perspectives)
    - Penalizes absolutist language patterns and rewards graduated confidence
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 1.0
        
        import re
        import math
        from collections import Counter
        
        response_lower = response.lower().strip()
        query_lower = query.lower().strip()
        
        if len(response_lower) < 5:
            return 0.5
        
        # === SENTENCE SEGMENTATION ===
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        # === TOPIC UNCERTAINTY DETECTION ===
        # Determine if the query is about topics that inherently require uncertainty
        uncertain_topic_signals = [
            'why', 'explain', 'what causes', 'what is the reason', 'how does',
            'compare', 'contrast', 'opinion', 'think', 'believe', 'argue',
            'debate', 'controversial', 'hypothetical', 'predict', 'future',
            'best', 'worst', 'should', 'would', 'could', 'might',
            'meaning', 'interpret', 'significance', 'impact', 'effect',
            'philosophy', 'theory', 'hypothesis'
        ]
        
        factual_task_signals = [
            'rewrite', 'generate', 'create', 'write', 'list', 'provide',
            'describe', 'crop', 'convert', 'translate', 'summarize',
            'come up with', 'give me', 'make', 'compose', 'draft',
            'name', 'identify', 'define', 'calculate', 'count'
        ]
        
        topic_uncertainty_level = 0.0
        for signal in uncertain_topic_signals:
            if signal in query_lower:
                topic_uncertainty_level += 1.0
        topic_uncertainty_level = min(topic_uncertainty_level / 3.0, 1.0)
        
        is_factual_task = False
        for signal in factual_task_signals:
            if signal in query_lower:
                is_factual_task = True
                break
        
        # === SENTENCE-LEVEL EPISTEMIC STANCE CLASSIFICATION ===
        # Classify each sentence into epistemic categories
        
        # High confidence / absolutist markers
        absolutist_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\bundoubtedly\b', r'\bwithout a doubt\b', r'\bobviously\b',
            r'\bclearly\b', r'\bof course\b', r'\beveryone knows\b',
            r'\bit is (a )?fact\b', r'\bno question\b', r'\bwithout question\b',
            r'\babsolutely\b', r'\bunquestionably\b', r'\bindisputably\b',
            r'\bthe truth is\b', r'\bthe fact is\b', r'\bplain and simple\b',
            r'\bno doubt\b', r'\bguaranteed\b'
        ]
        
        # Appropriate hedging / calibration markers
        calibration_patterns = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b', r'\boften\b',
            r'\btends to\b', r'\bin many cases\b', r'\bin some cases\b',
            r'\blikely\b', r'\bprobably\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bcan\b',
            r'\bresearch suggests\b', r'\bstudies suggest\b', r'\bevidence suggests\b',
            r'\bit appears\b', r'\bit seems\b', r'\bseems to\b',
            r'\bto some extent\b', r'\bin part\b', r'\bpartially\b',
            r'\bapproximately\b', r'\broughly\b', r'\babout\b',
            r'\bone perspective\b', r'\bsome argue\b', r'\bsome believe\b',
            r'\bdepending on\b', r'\bit depends\b', r'\bvaries\b',
            r'\bnot necessarily\b', r'\bnot always\b'
        ]
        
        # Nuance / multi-perspective markers
        nuance_patterns = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\balthough\b', r'\bwhile\b', r'\bwhereas\b', r'\bbut\b',
            r'\bin contrast\b', r'\bconversely\b', r'\bdespite\b',
            r'\bboth\b.*\band\b', r'\bnot only\b.*\bbut also\b',
            r'\bsome\b.*\bwhile others\b', r'\bsome\b.*\bother\b',
            r'\bdifferent perspectives\b', r'\bmultiple\b', r'\bvarious\b',
            r'\bon one hand\b', r'\balternatively\b', r'\bthat said\b',
            r'\bmore complex\b', r'\bnuanced\b'
        ]
        
        # Source attribution patterns
        source_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstudies\b', r'\bexperts\b',
            r'\bscientists\b', r'\bscholars\b', r'\bdata\b', r'\bevidence\b',
            r'\bfindings\b', r'\bliterature\b', r'\bsurvey\b'
        ]
        
        # Overconfident speculation patterns
        overconfident_spec_patterns = [
            r'\bwill definitely\b', r'\bwill always\b', r'\bwill never\b',
            r'\bis the only\b', r'\bthe only way\b', r'\bthere is no other\b',
            r'\beveryone\b', r'\bnobody\b', r'\ball people\b', r'\bno one\b',
            r'\bimpossible\b', r'\bperfect\b', r'\bflawless\b',
            r'\bthe best\b', r'\bthe worst\b', r'\bthe most\b'
        ]
        
        def count_pattern_matches(text, patterns):
            count = 0
            for pat in patterns:
                count += len(re.findall(pat, text, re.IGNORECASE))
            return count
        
        # === COMPUTE SCORES PER SENTENCE ===
        sentence_stances = []
        for sent in sentences:
            sent_lower = sent.lower()
            abs_count = count_pattern_matches(sent_lower, absolutist_patterns)
            cal_count = count_pattern_matches(sent_lower, calibration_patterns)
            nuance_count = count_pattern_matches(sent_lower, nuance_patterns)
            overconf_count = count_pattern_matches(sent_lower, overconfident_spec_patterns)
            
            # Classify sentence stance
            total_markers = abs_count + cal_count + nuance_count + overconf_count
            if total_markers == 0:
                stance = 'neutral'
            elif cal_count + nuance_count > abs_count + overconf_count:
                stance = 'calibrated'
            elif abs_count + overconf_count > cal_count + nuance_count:
                stance = 'absolutist'
            else:
                stance = 'mixed'
            
            sentence_stances.append({
                'stance': stance,
                'abs': abs_count,
                'cal': cal_count,
                'nuance': nuance_count,
                'overconf': overconf_count
            })
        
        # === AGGREGATE METRICS ===
        stance_counts = Counter(s['stance'] for s in sentence_stances)
        total_abs = sum(s['abs'] for s in sentence_stances)
        total_cal = sum(s['cal'] for s in sentence_stances)
        total_nuance = sum(s['nuance'] for s in sentence_stances)
        total_overconf = sum(s['overconf'] for s in sentence_stances)
        total_source = count_pattern_matches(response_lower, source_patterns)
        
        # === SCORING COMPONENTS ===
        
        # 1. Calibration ratio score (0-25)
        # Ratio of calibrated language to absolutist language
        total_epistemic = total_abs + total_cal + total_overconf + 0.001
        calibration_ratio = (total_cal + total_nuance * 0.5) / total_epistemic
        calibration_score = min(calibration_ratio * 20, 25)
        
        # 2. Overconfidence penalty (0 to -15)
        overconf_rate = total_overconf / max(num_sentences, 1)
        absolutist_rate = total_abs / max(num_sentences, 1)
        overconf_penalty = -min((overconf_rate * 8 + absolutist_rate * 4), 15)
        
        # 3. Nuance and multi-perspective score (0-20)
        nuance_rate = total_nuance / max(num_sentences, 1)
        nuance_score = min(nuance_rate * 15, 20)
        
        # 4. Source attribution bonus (0-10)
        source_score = min(total_source * 2.5, 10)
        
        # 5. Stance distribution score (0-15)
        # Reward diverse epistemic stances (not all sentences same stance)
        num_stance_types = len([k for k, v in stance_counts.items() if v > 0])
        if num_sentences >= 3:
            distribution_score = min(num_stance_types * 4, 15)
        else:
            distribution_score = min(num_stance_types * 3, 10)
        
        # 6. Response completeness and structure (0-20)
        # Longer, more structured responses tend to be better
        word_count = len(response.split())
        
        # Length score with diminishing returns
        if word_count < 10:
            length_score = 2
        elif word_count < 30:
            length_score = 8
        elif word_count < 60:
            length_score = 14
        elif word_count < 120:
            length_score = 18
        else:
            length_score = 20
        
        # 7. Repetition penalty
        words = response_lower.split()
        if len(words) > 5:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                max_bigram_freq = max(bigram_counts.values())
                repetition_ratio = max_bigram_freq / len(bigrams)
                if repetition_ratio > 0.15:
                    repetition_penalty = -10 * min(repetition_ratio, 0.5)
                else:
                    repetition_penalty = 0
            else:
                repetition_penalty = 0
        else:
            repetition_penalty = 0
        
        # 8. Conditional modifier: adjust based on topic type
        # For factual tasks, don't penalize lack of hedging as much
        # For uncertain topics, reward hedging more
        if is_factual_task:
            # For factual/creative tasks, reduce importance of calibration
            calibration_score *= 0.3
            overconf_penalty *= 0.3
            nuance_score *= 0.3
            source_score *= 0.3
            # Boost length/completeness importance
            length_score *= 1.3
        elif topic_uncertainty_level > 0.5:
            # For uncertain topics, boost calibration importance
            calibration_score *= 1.5
            overconf_penalty *= 1.5
            nuance_score *= 1.4
        
        # 9. Sentence variety score (0-10)
        # Check if sentences have varied structure (not all starting the same way)
        if num_sentences >= 2:
            first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
            first_word_counts = Counter(first_words)
            max_first_word_freq = max(first_word_counts.values()) if first_word_counts else 0
            variety_ratio = 1.0 - (max_first_word_freq / num_sentences)
            variety_score = variety_ratio * 10
        else:
            variety_score = 3
        
        # 10. Information density: unique content words / total words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                      'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'only', 'own',
                      'same', 'than', 'too', 'very', 'just', 'because', 'that',
                      'this', 'these', 'those', 'it', 'its', 'they', 'them',
                      'their', 'we', 'us', 'our', 'he', 'him', 'his', 'she',
                      'her', 'i', 'me', 'my', 'you', 'your', 'which', 'who',
                      'whom', 'what', 'where', 'when', 'how', 'while', 'also'}
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = set(content_words)
        if len(content_words) > 0:
            info_density = len(unique_content) / len(content_words)
            info_score = info_density * 10
        else:
            info_score = 2
        
        # === FINAL SCORE AGGREGATION ===
        raw_score = (
            calibration_score +      # 0-25 (or adjusted)
            overconf_penalty +        # -15 to 0
            nuance_score +            # 0-20
            source_score +            # 0-10
            distribution_score +      # 0-15
            length_score +            # 0-20 (or adjusted)
            repetition_penalty +      # -5 to 0
            variety_score +           # 0-10
            info_score                # 0-10
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~130, min ~ -30
        # Practical range: roughly 5-80
        final_score = max(0.0, min(100.0, raw_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: return a middling score based on response length
        try:
            return min(max(len(str(response).split()) * 0.3, 1.0), 50.0)
        except:
            return 5.0