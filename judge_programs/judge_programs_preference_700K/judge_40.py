def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using a discourse-level analysis
    approach based on:
    1. Causal/logical connective chain analysis (tracking reasoning chains)
    2. Sentence-level entailment proxy via shared entity/concept threading
    3. Structural progression detection (claim → evidence → conclusion patterns)
    4. Contradiction detection via negation pattern analysis
    5. Discourse depth and elaboration metrics
    """
    try:
        import re
        import math
        from collections import Counter, defaultdict

        if not response or not isinstance(response, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 10:
            return 0.5

        # --- Tokenization helpers ---
        def get_sentences(text):
            # Split on sentence-ending punctuation, keeping non-empty
            parts = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in parts if s.strip() and len(s.strip()) > 3]

        def get_words(text):
            return re.findall(r"[a-zA-Z']+", text.lower())

        def get_content_words(words):
            stop = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                    'do','does','did','will','would','shall','should','may','might','can','could',
                    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
                    'they','them','their','this','that','these','those','what','which','who','whom',
                    'and','but','or','nor','not','no','so','if','then','than','too','very','just',
                    'about','above','after','again','all','also','am','any','as','at','back','because',
                    'before','between','both','by','came','come','could','day','did','do','each',
                    'even','for','from','get','got','had','has','him','how','in','into','its','let',
                    'like','long','look','make','many','most','much','must','new','now','of','on',
                    'one','only','or','other','out','over','own','said','same','see','should','show',
                    'side','since','so','some','still','such','take','tell','than','that','the',
                    'their','them','then','there','these','thing','think','those','through','time',
                    'to','two','under','up','upon','us','use','very','want','way','well','went',
                    'were','when','where','while','why','with','work','world','year'}
            return [w for w in words if w not in stop and len(w) > 2]

        sentences = get_sentences(response_stripped)
        all_words = get_words(response_stripped)
        
        if len(all_words) < 3:
            return 1.0

        # ============================================================
        # FEATURE 1: Causal/Logical Reasoning Chain Density
        # Detect chains of reasoning connectives and measure their density
        # ============================================================
        causal_connectives = [
            r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bas a result\b', r'\bdue to\b',
            r'\bsince\b', r'\bso that\b', r'\bin order to\b',
            r'\bthis means\b', r'\bwhich means\b', r'\bimplying\b',
            r'\bimplies\b', r'\bleads to\b', r'\bresults in\b',
            r'\bcaused by\b', r'\bgiven that\b', r'\bfor this reason\b',
            r'\bit follows\b', r'\baccordingly\b'
        ]
        
        conditional_connectives = [
            r'\bif\b.*\bthen\b', r'\bwhen\b.*\bthen\b',
            r'\bassuming\b', r'\bprovided that\b', r'\bunless\b',
            r'\bin the case\b', r'\bsuppose\b', r'\bwere to\b'
        ]
        
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bnamely\b', r'\bsuch as\b',
            r'\bto illustrate\b', r'\bin other words\b', r'\bthat is\b',
            r'\bi\.e\.\b', r'\be\.g\.\b', r'\bmore specifically\b'
        ]
        
        contrast_connectives = [
            r'\bhowever\b', r'\bon the other hand\b', r'\bnevertheless\b',
            r'\bnonetheless\b', r'\balthough\b', r'\beven though\b',
            r'\bdespite\b', r'\bin contrast\b', r'\bconversely\b',
            r'\bwhile\b.*\b(also|still|yet)\b', r'\brather\b',
            r'\binstead\b', r'\byet\b'
        ]
        
        concession_connectives = [
            r'\badmittedly\b', r'\bgranted\b', r'\bof course\b',
            r'\bto be fair\b', r'\bwhile it.s true\b', r'\bit.s worth noting\b'
        ]
        
        response_lower = response_stripped.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectives, response_lower)
        conditional_count = count_patterns(conditional_connectives, response_lower)
        elaboration_count = count_patterns(elaboration_connectives, response_lower)
        contrast_count = count_patterns(contrast_connectives, response_lower)
        concession_count = count_patterns(concession_connectives, response_lower)
        
        total_connectives = causal_count + conditional_count + elaboration_count + contrast_count + concession_count
        
        # Normalize by number of sentences
        num_sentences = max(len(sentences), 1)
        connective_density = total_connectives / num_sentences
        
        # Diversity of connective types used (out of 5 categories)
        connective_types_used = sum([
            causal_count > 0,
            conditional_count > 0,
            elaboration_count > 0,
            contrast_count > 0,
            concession_count > 0
        ])
        connective_diversity = connective_types_used / 5.0

        # ============================================================
        # FEATURE 2: Entity/Concept Threading (Coherence via topic continuity)
        # Measure how well concepts carry across consecutive sentences
        # ============================================================
        if len(sentences) >= 2:
            sentence_content_words = []
            for s in sentences:
                w = get_words(s)
                cw = set(get_content_words(w))
                sentence_content_words.append(cw)
            
            # Compute consecutive sentence overlap (entity threading)
            thread_scores = []
            for i in range(1, len(sentence_content_words)):
                prev = sentence_content_words[i-1]
                curr = sentence_content_words[i]
                if len(prev) == 0 and len(curr) == 0:
                    thread_scores.append(0.5)
                elif len(prev) == 0 or len(curr) == 0:
                    thread_scores.append(0.0)
                else:
                    overlap = len(prev & curr)
                    # Normalized by min set size (ensures strong threading)
                    min_size = min(len(prev), len(curr))
                    thread_scores.append(overlap / max(min_size, 1))
            
            avg_threading = sum(thread_scores) / max(len(thread_scores), 1)
            
            # Also measure long-range coherence (every sentence with the first 2 sentences)
            if len(sentence_content_words) >= 3:
                anchor = sentence_content_words[0] | (sentence_content_words[1] if len(sentence_content_words) > 1 else set())
                long_range_scores = []
                for i in range(2, len(sentence_content_words)):
                    curr = sentence_content_words[i]
                    if len(anchor) > 0 and len(curr) > 0:
                        overlap = len(anchor & curr)
                        long_range_scores.append(overlap / max(min(len(anchor), len(curr)), 1))
                long_range_coherence = sum(long_range_scores) / max(len(long_range_scores), 1) if long_range_scores else 0.5
            else:
                long_range_coherence = 0.5
        else:
            avg_threading = 0.3
            long_range_coherence = 0.3

        # ============================================================
        # FEATURE 3: Structural Progression (Claim-Evidence-Conclusion)
        # Detect discourse roles of sentences
        # ============================================================
        claim_indicators = [
            r'\bi (?:think|believe|argue|contend|maintain|suggest)\b',
            r'\bit is\b', r'\bthe (?:key|main|central|primary)\b',
            r'\bfundamentally\b', r'\bessentially\b', r'\bin my (?:view|opinion)\b',
            r'\bthe (?:answer|point|issue|problem|question)\b',
            r'\bwhat matters\b', r'\bthe truth is\b'
        ]
        
        evidence_indicators = [
            r'\bfor example\b', r'\bfor instance\b', r'\bevidence\b',
            r'\bdata\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\baccording to\b', r'\bin fact\b', r'\bstatistic\b',
            r'\bhistorically\b', r'\bin practice\b', r'\bexperience\b',
            r'\btypically\b', r'\bgenerally\b', r'\boften\b',
            r'\bspecifically\b', r'\bsuch as\b'
        ]
        
        conclusion_indicators = [
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
            r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bultimately\b', r'\ball in all\b', r'\bto sum up\b',
            r'\bthe bottom line\b', r'\bin short\b', r'\bso,?\s\b'
        ]
        
        claim_count = count_patterns(claim_indicators, response_lower)
        evidence_count = count_patterns(evidence_indicators, response_lower)
        conclusion_count = count_patterns(conclusion_indicators, response_lower)
        
        # Reward having a mix of discourse roles
        discourse_roles_present = sum([claim_count > 0, evidence_count > 0, conclusion_count > 0])
        discourse_structure_score = discourse_roles_present / 3.0
        
        # Extra reward for evidence richness
        evidence_density = min(evidence_count / max(num_sentences, 1), 1.0)

        # ============================================================
        # FEATURE 4: Contradiction/Negation Consistency Analysis
        # Check for potential self-contradictions via negation flip patterns
        # ============================================================
        negation_words = {'not', "n't", 'never', 'no', 'neither', 'nor', 'none', 'nothing', 'nowhere', 'nobody'}
        
        sentence_polarities = []
        for s in sentences:
            s_words = get_words(s)
            neg_count = sum(1 for w in s_words if w in negation_words or w.endswith("n't"))
            # Simple polarity: even negations = positive, odd = negative
            sentence_polarities.append(neg_count % 2)
        
        # Check for rapid polarity flips (potential contradictions)
        if len(sentence_polarities) >= 3:
            flip_count = 0
            for i in range(2, len(sentence_polarities)):
                # Pattern: positive -> negative -> positive or vice versa without connectives
                if sentence_polarities[i] != sentence_polarities[i-1] and sentence_polarities[i-1] != sentence_polarities[i-2]:
                    # Check if there's a contrast connective to justify the flip
                    s_lower = sentences[i].lower() if i < len(sentences) else ""
                    has_justification = any(re.search(p, s_lower) for p in contrast_connectives + concession_connectives)
                    if not has_justification:
                        flip_count += 1
            
            contradiction_penalty = min(flip_count * 0.15, 0.5)
        else:
            contradiction_penalty = 0.0

        # ============================================================
        # FEATURE 5: Discourse Depth and Elaboration
        # Measure how deeply the response explores the topic
        # ============================================================
        content_words = get_content_words(all_words)
        
        # Unique concept density
        unique_content = set(content_words)
        concept_richness = len(unique_content) / max(len(content_words), 1) if content_words else 0
        # Sweet spot: not too repetitive, not too scattered
        # Optimal around 0.4-0.7
        concept_richness_score = 1.0 - abs(concept_richness - 0.55) * 2.0
        concept_richness_score = max(0, min(1, concept_richness_score))
        
        # Response substantiveness (length proxy, but with diminishing returns)
        word_count = len(all_words)
        # Log scale for length: reward longer responses but with diminishing returns
        length_score = min(math.log(max(word_count, 1) + 1) / math.log(300), 1.0)
        
        # Sentence complexity: average words per sentence
        if num_sentences > 0:
            avg_sent_len = word_count / num_sentences
            # Optimal range: 12-25 words per sentence
            if avg_sent_len < 8:
                complexity_score = avg_sent_len / 8.0 * 0.6
            elif avg_sent_len <= 25:
                complexity_score = 0.6 + 0.4 * min((avg_sent_len - 8) / 17.0, 1.0)
            else:
                complexity_score = max(0.5, 1.0 - (avg_sent_len - 25) / 40.0)
        else:
            complexity_score = 0.3

        # ============================================================
        # FEATURE 6: Argument Qualification and Nuance
        # Detect qualified statements (sign of careful reasoning)
        # ============================================================
        qualification_patterns = [
            r'\bgenerally\b', r'\btypically\b', r'\busually\b',
            r'\bin most cases\b', r'\btends to\b', r'\boften\b',
            r'\bto some extent\b', r'\bpartially\b', r'\bsomewhat\b',
            r'\bit depends\b', r'\bcontext\b', r'\bnuance\b',
            r'\bcaveat\b', r'\bexception\b', r'\bthat said\b',
            r'\bkeep in mind\b', r'\bnote that\b', r'\bimportant(ly)?\b',
            r'\bworth (?:noting|mentioning|considering)\b'
        ]
        
        qualification_count = count_patterns(qualification_patterns, response_lower)
        qualification_score = min(qualification_count / max(num_sentences * 0.3, 1), 1.0)

        # ============================================================
        # FEATURE 7: Query Relevance (semantic overlap with query)
        # ============================================================
        if query and isinstance(query, str):
            query_words = set(get_content_words(get_words(query.lower())))
            response_content_set = unique_content
            if query_words and response_content_set:
                relevance = len(query_words & response_content_set) / max(len(query_words), 1)
                relevance_score = min(relevance * 2.0, 1.0)  # Scale up
            else:
                relevance_score = 0.3
        else:
            relevance_score = 0.5

        # ============================================================
        # FEATURE 8: Reasoning Pattern Detection
        # Look for explicit reasoning structures
        # ============================================================
        reasoning_patterns = [
            r'\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b',  # Enumeration
            r'\bon one hand\b.*\bon the other\b',  # Balanced argument
            r'\bthe reason\b.*\bis\b',  # Explicit reason-giving
            r'\bthis is because\b',
            r'\bthe (?:key|main|important) (?:point|thing|issue|factor)\b',
            r'\bnot only\b.*\bbut also\b',  # Additive reasoning
            r'\bwhile\b.*\b(?:also|still|however)\b',  # Concessive reasoning
        ]
        
        reasoning_structure_count = count_patterns(reasoning_patterns, response_lower)
        reasoning_score = min(reasoning_structure_count * 0.25, 1.0)
        
        # Also check for enumeration/ordered arguments
        enum_patterns = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b',
            r'\b1[\.\)]\s', r'\b2[\.\)]\s', r'\b3[\.\)]\s',
            r'\bfirstly\b', r'\bsecondly\b', r'\bthirdly\b',
            r'\banother\b', r'\badditionally\b', r'\bfurthermore\b',
            r'\bmoreover\b', r'\balso\b', r'\bin addition\b'
        ]
        enum_count = count_patterns(enum_patterns, response_lower)
        enumeration_score = min(enum_count / max(num_sentences * 0.2, 1), 1.0)

        # ============================================================
        # FEATURE 9: Opening/Framing Quality
        # Does the response start by framing the answer clearly?
        # ============================================================
        first_sentence = sentences[0].lower() if sentences else ""
        
        framing_patterns = [
            r'^(?:essentially|basically|in short|to answer|the (?:short|simple|quick) answer)',
            r'^(?:yes|no|it depends|great question|good question)',
            r'^(?:there are|this is|the (?:key|main|central))',
            r'^(?:so,?\s|well,?\s|actually)',
            r'(?:let me|i\'ll|allow me)',
        ]
        
        has_framing = any(re.search(p, first_sentence) for p in framing_patterns)
        framing_score = 0.7 if has_framing else 0.3
        
        # Also check if response directly addresses the query topic in first sentence
        if query and isinstance(query, str):
            query_content = set(get_content_words(get_words(query.lower())))
            first_sent_content = set(get_content_words(get_words(first_sentence)))
            if query_content and first_sent_content:
                first_sent_relevance = len(query_content & first_sent_content) / max(len(query_content), 1)
                framing_score = max(framing_score, min(first_sent_relevance * 2.5, 1.0))

        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        # Weighted combination
        score = (
            connective_density * 8.0 +          # Reasoning connective density (0-~1.0 range, scaled)
            connective_diversity * 6.0 +         # Variety of logical connective types
            avg_threading * 5.0 +                # Local coherence via entity threading
            long_range_coherence * 3.0 +         # Global coherence
            discourse_structure_score * 5.0 +    # Claim-evidence-conclusion structure
            evidence_density * 4.0 +             # Evidence richness
            concept_richness_score * 3.0 +       # Vocabulary balance
            length_score * 8.0 +                 # Substantiveness
            complexity_score * 4.0 +             # Sentence complexity
            qualification_score * 3.0 +          # Nuance and qualification
            relevance_score * 5.0 +              # Query relevance
            reasoning_score * 5.0 +              # Explicit reasoning patterns
            enumeration_score * 3.0 +            # Ordered argumentation
            framing_score * 3.0 +                # Opening quality
            -contradiction_penalty * 8.0         # Contradiction penalty
        )
        
        # Normalize to 0-10 range
        # Max theoretical raw: ~8 + 6 + 5 + 3 + 5 + 4 + 3 + 8 + 4 + 3 + 5 + 5 + 3 + 3 = ~65
        max_raw = 65.0
        normalized_score = (score / max_raw) * 10.0
        
        # Clamp
        normalized_score = max(0.0, min(10.0, normalized_score))
        
        return round(normalized_score, 3)
        
    except Exception:
        try:
            # Minimal fallback
            if response and isinstance(response, str):
                return min(len(response.split()) / 30.0, 5.0)
            return 0.0
        except Exception:
            return 0.0