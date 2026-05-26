def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using
    sentence-level logical flow analysis, causal/explanatory connective density,
    and progressive information accumulation patterns.
    
    This variant focuses on:
    1. Causal/explanatory connective analysis (not just word overlap)
    2. Sentence-level information progression (new concepts introduced per sentence)
    3. Logical scaffolding detection (premises -> conclusions pattern)
    4. Depth of elaboration ratio
    5. Anti-patterns: repetition, incoherence, off-topic drift
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response_stripped = response.strip()
        if len(response_stripped) == 0:
            return 0.0
        
        # --- Feature 1: Causal/Explanatory Connective Density ---
        # These indicate reasoning steps and explanations
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bcaused by\b', r'\bleads to\b', r'\bleading to\b',
            r'\bso that\b', r'\bin order to\b', r'\bfor this reason\b',
        ]
        explanatory_connectives = [
            r'\bthis means\b', r'\bin other words\b', r'\bthat is\b',
            r'\bspecifically\b', r'\bnamely\b', r'\bfor example\b',
            r'\bfor instance\b', r'\bsuch as\b', r'\bto illustrate\b',
            r'\bin particular\b', r'\bthis is because\b', r'\bthe reason\b',
        ]
        sequential_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bafterward\b', r'\bsubsequently\b',
            r'\bto begin\b', r'\bfollowing this\b', r'\bin the first place\b',
            r'\bstep\s+\d+\b', r'\b\d+\)\s', r'\b\d+\.\s',
        ]
        contrastive_connectives = [
            r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\bdespite\b',
            r'\bwhile\b', r'\byet\b', r'\bbut\b', r'\binstead\b',
        ]
        
        resp_lower = response_stripped.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectives, resp_lower)
        explanatory_count = count_patterns(explanatory_connectives, resp_lower)
        sequential_count = count_patterns(sequential_connectives, resp_lower)
        contrastive_count = count_patterns(contrastive_connectives, resp_lower)
        
        total_connectives = causal_count + explanatory_count + sequential_count + contrastive_count
        
        # Normalize by response length (per 100 words)
        words = response_stripped.split()
        word_count = len(words)
        
        if word_count < 2:
            return 0.5
        
        connective_density = (total_connectives / max(word_count, 1)) * 100
        # Score: 0-2.5 points, sweet spot around 3-8 connectives per 100 words
        connective_score = min(2.5, connective_density * 0.5)
        
        # Bonus for diversity of connective types used
        types_used = sum(1 for c in [causal_count, explanatory_count, sequential_count, contrastive_count] if c > 0)
        connective_diversity_bonus = types_used * 0.2  # up to 0.8
        
        # --- Feature 2: Sentence-level Information Progression ---
        # Split into sentences
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 1.0
        
        # Measure how many NEW content words each sentence introduces
        # Good reasoning builds progressively
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                'under', 'again', 'further', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 'just', 'it', 'its', 'this', 'that',
                'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
                'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
                'who', 'whom', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
                'about', 'up', 'down', 'also', 'then', 'them',
            }
            w = re.findall(r'[a-z]+', text.lower())
            return set(w2 for w2 in w if w2 not in stop_words and len(w2) > 2)
        
        seen_words = set()
        new_word_ratios = []
        for sent in sentences:
            content = get_content_words(sent)
            if len(content) == 0:
                new_word_ratios.append(0)
            else:
                new_words = content - seen_words
                new_word_ratios.append(len(new_words) / len(content))
            seen_words.update(content)
        
        # Good progression: each sentence introduces some new concepts but not entirely new
        # (which would indicate incoherence)
        if len(new_word_ratios) > 1:
            avg_new_ratio = sum(new_word_ratios[1:]) / len(new_word_ratios[1:])
            # Sweet spot: 0.3-0.7 new words per sentence (building on previous but adding)
            if 0.25 <= avg_new_ratio <= 0.75:
                progression_score = 1.5
            elif 0.15 <= avg_new_ratio <= 0.85:
                progression_score = 1.0
            else:
                progression_score = 0.3  # Too repetitive or too scattered
        else:
            progression_score = 0.5
        
        # --- Feature 3: Logical Scaffolding / Multi-sentence Reasoning ---
        # Detect if response has premise-conclusion structure
        # Look for patterns like "X. Therefore Y." or "Since X, Y."
        premise_indicators = [
            r'\bgiven that\b', r'\bassuming\b', r'\bif\b.*\bthen\b',
            r'\bsince\b', r'\bbecause\b', r'\bconsidering\b',
            r'\bnote that\b', r'\bobserve that\b', r'\brecall that\b',
        ]
        conclusion_indicators = [
            r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bso\b',
            r'\bwe can conclude\b', r'\bthis means\b', r'\bit follows\b',
            r'\bin conclusion\b', r'\boverall\b', r'\bin summary\b',
        ]
        
        has_premises = count_patterns(premise_indicators, resp_lower) > 0
        has_conclusions = count_patterns(conclusion_indicators, resp_lower) > 0
        
        scaffolding_score = 0.0
        if has_premises and has_conclusions:
            scaffolding_score = 1.5
        elif has_premises or has_conclusions:
            scaffolding_score = 0.7
        
        # --- Feature 4: Depth of Elaboration ---
        # Longer, more substantive sentences indicate deeper reasoning
        # vs very short terse answers
        avg_sentence_length = word_count / max(num_sentences, 1)
        
        # Sweet spot: 10-25 words per sentence for explanatory text
        if 10 <= avg_sentence_length <= 30:
            elaboration_score = 1.5
        elif 7 <= avg_sentence_length <= 40:
            elaboration_score = 1.0
        elif 5 <= avg_sentence_length <= 50:
            elaboration_score = 0.5
        else:
            elaboration_score = 0.2
        
        # Multi-sentence bonus: reasoning typically requires multiple sentences
        if num_sentences >= 4:
            multi_sent_bonus = 0.8
        elif num_sentences >= 2:
            multi_sent_bonus = 0.4
        else:
            multi_sent_bonus = 0.0
        
        # --- Feature 5: Anti-patterns ---
        anti_pattern_penalty = 0.0
        
        # 5a: Excessive repetition detection (sentence-level)
        if num_sentences >= 2:
            sentence_texts = [s.lower().strip() for s in sentences]
            unique_sents = set(sentence_texts)
            repetition_ratio = 1 - (len(unique_sents) / len(sentence_texts))
            if repetition_ratio > 0.3:
                anti_pattern_penalty += 2.0
            elif repetition_ratio > 0.15:
                anti_pattern_penalty += 1.0
        
        # 5b: Code/HTML dump detection (not reasoning)
        code_indicators = len(re.findall(r'[{}<>/\\;=]', response_stripped))
        code_ratio = code_indicators / max(len(response_stripped), 1)
        if code_ratio > 0.05:
            anti_pattern_penalty += min(2.0, code_ratio * 20)
        
        # 5c: Very short response (likely no reasoning shown)
        if word_count < 5:
            anti_pattern_penalty += 3.0
        elif word_count < 10:
            anti_pattern_penalty += 1.5
        elif word_count < 20:
            anti_pattern_penalty += 0.5
        
        # 5d: Gibberish / nonsense detection - high ratio of non-alpha chars
        alpha_chars = sum(1 for c in response_stripped if c.isalpha())
        alpha_ratio = alpha_chars / max(len(response_stripped), 1)
        if alpha_ratio < 0.5:
            anti_pattern_penalty += 2.0
        
        # 5e: Response is just echoing the query without adding reasoning
        query_words = get_content_words(query)
        response_words = get_content_words(response_stripped)
        if len(response_words) > 0 and len(query_words) > 0:
            overlap = len(response_words & query_words)
            if len(response_words) > 0:
                echo_ratio = overlap / len(response_words)
                if echo_ratio > 0.8 and word_count < 30:
                    anti_pattern_penalty += 1.0
        
        # 5f: Detect off-topic rambling (e.g., random questions, unrelated content)
        question_marks = response_stripped.count('?')
        if question_marks > 3 and num_sentences > 0:
            q_ratio = question_marks / num_sentences
            if q_ratio > 0.5:
                anti_pattern_penalty += 1.0
        
        # --- Feature 6: Explicit reasoning markers ---
        reasoning_phrases = [
            r'\blet me explain\b', r'\bhere\'s why\b', r'\bthe reason is\b',
            r'\bthis is because\b', r'\bto understand\b', r'\blet\'s consider\b',
            r'\blet\'s break\b', r'\bfirst,?\s+we\b', r'\bwe need to\b',
            r'\bwe can see\b', r'\bthis suggests\b', r'\bthis indicates\b',
            r'\bimportantly\b', r'\bnotably\b', r'\bkey point\b',
            r'\bin fact\b', r'\bthe key\b', r'\bthe main\b',
        ]
        reasoning_marker_count = count_patterns(reasoning_phrases, resp_lower)
        reasoning_marker_score = min(1.0, reasoning_marker_count * 0.3)
        
        # --- Feature 7: Query-response relevance (basic) ---
        # Ensure the response addresses the query at all
        if len(query_words) > 0 and len(response_words) > 0:
            relevance = len(response_words & query_words) / max(len(query_words), 1)
            relevance_score = min(1.0, relevance * 1.5)
        else:
            relevance_score = 0.3
        
        # --- Combine all features ---
        raw_score = (
            connective_score          # 0-2.5: causal/explanatory language
            + connective_diversity_bonus  # 0-0.8: variety of connective types
            + progression_score        # 0-1.5: information builds progressively
            + scaffolding_score        # 0-1.5: premise-conclusion structure
            + elaboration_score        # 0-1.5: appropriate sentence depth
            + multi_sent_bonus         # 0-0.8: multiple sentences
            + reasoning_marker_score   # 0-1.0: explicit reasoning phrases
            + relevance_score          # 0-1.0: addresses the query
            - anti_pattern_penalty     # 0-8+: various quality penalties
        )
        
        # Max theoretical positive: ~10.6
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # Map to 0-10 with better discrimination
        # Center around 5, stretch differences
        normalized = final_score / 10.0  # 0-1
        # Apply power curve to spread middle values
        adjusted = normalized ** 0.8 * 10.0
        
        return round(max(0.0, min(10.0, adjusted)), 2)
        
    except Exception:
        try:
            if response and len(response.strip()) > 20:
                return 3.0
            return 1.0
        except Exception:
            return 1.0