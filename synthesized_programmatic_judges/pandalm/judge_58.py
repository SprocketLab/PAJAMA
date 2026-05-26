def judging_function(query, response):
    """
    Evaluates response quality focusing on epistemic calibration and uncertainty communication.
    
    This variant uses a sentence-level analysis approach:
    - Analyzes each sentence for epistemic stance (certain, hedged, speculative)
    - Measures the ratio of appropriately hedged claims vs overconfident assertions
    - Evaluates structural completeness and coherence
    - Penalizes repetition and empty content
    - Rewards nuanced, multi-perspective responses
    
    Different from Variant 1 (which uses vocabulary diversity, word overlap, confidence markers)
    by focusing on sentence-level epistemic classification and structural analysis.
    """
    try:
        if not response or not isinstance(response, str) or response.strip() == "":
            return 0.0
        if not query or not isinstance(query, str):
            return 2.0
        
        response_stripped = response.strip()
        query_stripped = query.strip()
        
        # === SENTENCE-LEVEL EPISTEMIC ANALYSIS ===
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_stripped)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if len(sentences) == 0:
            return 1.0
        
        # Categorize each sentence's epistemic stance
        # Hedging phrases (appropriate uncertainty markers)
        hedging_patterns = [
            r'\bmight\b', r'\bcould\b', r'\bmay\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\blikely\b', r'\bunlikely\b', r'\bprobably\b', r'\bapparently\b',
            r'\bseems?\b', r'\bappears?\b', r'\btends?\sto\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\boften\b', r'\bsometimes\b',
            r'\bin\s+some\s+cases\b', r'\bit\s+is\s+(?:widely\s+)?(?:believed|thought|considered)\b',
            r'\bresearch\s+suggests?\b', r'\bstudies?\s+(?:show|indicate|suggest)\b',
            r'\bevidence\s+suggests?\b', r'\baccording\s+to\b',
            r'\bone\s+(?:could|might|may)\b', r'\bin\s+general\b',
            r'\bto\s+some\s+(?:extent|degree)\b', r'\barguably\b',
            r'\bcan\s+be\b', r'\btend\s+to\b', r'\bsome\b.*\bwhile\b.*\bothers?\b',
            r'\bnot\s+necessarily\b', r'\bdepends?\b', r'\bvaries?\b',
            r'\bsuggests?\s+that\b', r'\bindicates?\s+that\b',
        ]
        
        # Overconfidence patterns (presenting speculation as fact)
        overconfidence_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\bundoubtedly\b', r'\bwithout\s+(?:a\s+)?doubt\b', r'\bobviously\b',
            r'\bclearly\b', r'\beveryone\s+knows\b', r'\bit\s+is\s+(?:a\s+)?fact\b',
            r'\bno\s+question\b', r'\babsolutely\b', r'\bguaranteed\b',
            r'\bimpossible\b', r'\bperfect(?:ly)?\b',
            r'\bthe\s+only\b', r'\bthe\s+best\b', r'\bthe\s+worst\b',
            r'\bnothing\s+(?:can|could|will)\b', r'\beverything\s+(?:is|was|will)\b',
        ]
        
        # Explanatory/elaboration patterns (good for quality)
        elaboration_patterns = [
            r'\bfor\s+(?:example|instance)\b', r'\bsuch\s+as\b',
            r'\bin\s+other\s+words\b', r'\bthis\s+means\b',
            r'\bspecifically\b', r'\bin\s+particular\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bhowever\b', r'\bon\s+the\s+other\s+hand\b', r'\bconversely\b',
            r'\bwhile\b', r'\bwhereas\b', r'\balthough\b', r'\bdespite\b',
            r'\bin\s+contrast\b', r'\bnevertheless\b',
            r'\bbecause\b', r'\bsince\b', r'\bdue\s+to\b', r'\bas\s+a\s+result\b',
            r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
        ]
        
        # Contrastive/nuance patterns (showing multiple perspectives)
        nuance_patterns = [
            r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\bwhile\b',
            r'\bon\s+the\s+other\s+hand\b', r'\bin\s+contrast\b',
            r'\bconversely\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bdepending\s+on\b', r'\bit\s+depends\b',
            r'\bboth\b.*\band\b', r'\bnot\s+only\b.*\bbut\s+also\b',
            r'\bsome\b.*\bwhile\b', r'\bsome\b.*\bothers\b',
        ]
        
        def count_pattern_matches(text, patterns):
            count = 0
            text_lower = text.lower()
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            return count
        
        response_lower = response_stripped.lower()
        
        # Count matches across entire response
        hedge_count = count_pattern_matches(response_stripped, hedging_patterns)
        overconfidence_count = count_pattern_matches(response_stripped, overconfidence_patterns)
        elaboration_count = count_pattern_matches(response_stripped, elaboration_patterns)
        nuance_count = count_pattern_matches(response_stripped, nuance_patterns)
        
        # === SENTENCE-LEVEL CLASSIFICATION ===
        sentence_scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            sent_hedge = count_pattern_matches(sent, hedging_patterns)
            sent_overconf = count_pattern_matches(sent, overconfidence_patterns)
            sent_elab = count_pattern_matches(sent, elaboration_patterns)
            
            # Score each sentence: hedged/elaborated = good, overconfident = penalized
            s_score = 5.0  # baseline
            s_score += min(sent_hedge * 1.5, 4.0)
            s_score -= min(sent_overconf * 1.0, 3.0)
            s_score += min(sent_elab * 1.0, 3.0)
            
            # Sentence length quality (too short = low info, too long = potentially rambling)
            words_in_sent = len(sent.split())
            if words_in_sent < 4:
                s_score -= 1.5
            elif words_in_sent > 8:
                s_score += 0.5
            if words_in_sent > 40:
                s_score -= 0.5
            
            sentence_scores.append(max(0, min(10, s_score)))
        
        avg_sentence_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 5.0
        
        # === REPETITION DETECTION (sentence-level) ===
        # Check for repeated sentences or near-duplicate content
        unique_sentences = set()
        repetition_penalty = 0.0
        for sent in sentences:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sent.lower().strip())
            normalized = re.sub(r'[^\w\s]', '', normalized)
            if normalized in unique_sentences:
                repetition_penalty += 2.0
            unique_sentences.add(normalized)
        
        # Word-level repetition (bigram repetition ratio)
        words = response_lower.split()
        if len(words) > 4:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            unique_bigrams = set(bigrams)
            bigram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 1.0
        else:
            bigram_diversity = 0.5
        
        # Extreme repetition check (same word repeated many times)
        from collections import Counter
        word_counts = Counter(words)
        total_words = len(words)
        if total_words > 0:
            max_word_freq = max(word_counts.values())
            # Exclude common stop words from this check
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with',
                         'as', 'at', 'by', 'from', 'or', 'but', 'not', 'this', 'they',
                         'its', 'their', 'his', 'her', 'he', 'she', 'we', 'you', 'i'}
            content_words = {w: c for w, c in word_counts.items() if w not in stop_words and len(w) > 2}
            if content_words:
                max_content_freq = max(content_words.values())
                if max_content_freq > max(5, total_words * 0.15):
                    repetition_penalty += 3.0
        
        # === STRUCTURAL COMPLETENESS ===
        # Check if response ends mid-sentence (truncation)
        truncation_penalty = 0.0
        if not response_stripped[-1] in '.!?"\')':
            truncation_penalty = 2.0
        elif response_stripped.endswith('...'):
            truncation_penalty = 1.0
        
        # === RESPONSE LENGTH AND SUBSTANCE ===
        # Measure informational density
        total_words = len(words)
        
        length_score = 0.0
        if total_words < 5:
            length_score = -3.0
        elif total_words < 15:
            length_score = -1.0
        elif total_words < 30:
            length_score = 1.0
        elif total_words < 80:
            length_score = 2.5
        elif total_words < 150:
            length_score = 2.0
        else:
            length_score = 1.5
        
        # === QUERY-RESPONSE ALIGNMENT ===
        # Check if response addresses the query topic
        query_words = set(re.findall(r'\b\w{3,}\b', query_stripped.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response_lower))
        
        # Remove very common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                       'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                       'have', 'been', 'will', 'that', 'this', 'with', 'from',
                       'they', 'what', 'which', 'their', 'would', 'there', 'could',
                       'about', 'than', 'into', 'them', 'these', 'other', 'some'}
        
        query_content = query_words - common_words
        response_content = response_words - common_words
        
        if query_content:
            overlap = len(query_content & response_content) / len(query_content)
            relevance_score = overlap * 3.0  # 0 to 3
        else:
            relevance_score = 1.5
        
        # === INFORMATION RICHNESS ===
        # Count distinct informational elements (unique content words)
        info_richness = len(response_content) / max(total_words, 1) if total_words > 0 else 0
        richness_score = min(info_richness * 8.0, 3.0)
        
        # === EPISTEMIC CALIBRATION COMPOSITE ===
        # Determine if query is factual vs opinion-based vs creative
        query_lower = query_stripped.lower()
        
        # Factual queries should have more hedging on uncertain topics
        factual_indicators = ['what is', 'what are', 'explain', 'describe', 'how does',
                            'why does', 'define', 'what happens', 'what did']
        opinion_indicators = ['compare', 'contrast', 'evaluate', 'argue', 'discuss',
                            'opinion', 'think', 'believe', 'should', 'better']
        creative_indicators = ['write', 'create', 'generate', 'come up with', 'make',
                             'rewrite', 'provide examples', 'give examples']
        
        is_factual = any(ind in query_lower for ind in factual_indicators)
        is_opinion = any(ind in query_lower for ind in opinion_indicators)
        is_creative = any(ind in query_lower for ind in creative_indicators)
        
        # Epistemic calibration score
        epistemic_score = 0.0
        
        if is_opinion:
            # Opinion queries benefit from hedging and nuance
            epistemic_score += min(hedge_count * 0.8, 3.0)
            epistemic_score += min(nuance_count * 1.0, 3.0)
            epistemic_score -= min(overconfidence_count * 0.5, 2.0)
        elif is_factual:
            # Factual queries: moderate hedging is good, overconfidence is slightly penalized
            epistemic_score += min(hedge_count * 0.5, 2.0)
            epistemic_score += min(elaboration_count * 0.8, 2.5)
            epistemic_score -= min(overconfidence_count * 0.3, 1.5)
        elif is_creative:
            # Creative queries: less need for hedging, more for substance
            epistemic_score += min(elaboration_count * 0.5, 2.0)
            epistemic_score += richness_score * 0.5
        else:
            # Default: balanced approach
            epistemic_score += min(hedge_count * 0.5, 2.0)
            epistemic_score += min(elaboration_count * 0.6, 2.0)
            epistemic_score -= min(overconfidence_count * 0.4, 1.5)
        
        # === FINAL COMPOSITE SCORE ===
        # Weights chosen to balance all components
        final_score = (
            avg_sentence_score * 0.30 +      # sentence-level epistemic quality (0-10 range, weighted ~3)
            length_score * 0.8 +               # length appropriateness (-3 to 2.5, weighted)
            relevance_score * 1.0 +            # query relevance (0-3)
            richness_score * 0.8 +             # information richness (0-3)
            epistemic_score * 0.7 +            # epistemic calibration (variable)
            bigram_diversity * 2.0 -           # diversity bonus (0-2)
            repetition_penalty * 0.8 -         # repetition penalty
            truncation_penalty * 0.6           # truncation penalty
        )
        
        # Normalize to 0-10 range
        # Expected range: roughly -5 to 15, map to 0-10
        final_score = max(0.0, min(10.0, (final_score + 2.0) * 0.5))
        
        # Additional floor/ceiling adjustments
        # Very short responses (< 10 words) cap at 5
        if total_words < 10:
            final_score = min(final_score, 5.0)
        
        # Responses that are just echoing the query with nothing added
        if total_words < 15 and overlap > 0.8 if query_content else False:
            final_score = min(final_score, 4.0)
        
        # Empty-ish responses
        if total_words < 3:
            final_score = min(final_score, 1.0)
        
        return round(final_score, 3)
        
    except Exception as e:
        # Fallback: return a neutral score based on response length
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            elif response and len(response.strip()) > 0:
                return 2.0
            return 0.0
        except:
            return 0.0