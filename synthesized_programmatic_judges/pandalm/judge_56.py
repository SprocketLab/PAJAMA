def judging_function(query, response):
    """
    Evaluates reasoning transparency and step-wise formulation using a
    discourse coherence and explanatory depth analysis approach.
    
    This variant focuses on:
    1. Causal/logical connective density (tracking discourse flow markers)
    2. Clause-level complexity (subordinate clauses indicating elaboration)
    3. Information progression (new concept introduction rate across sentences)
    4. Explanatory ratio (proportion of text devoted to explaining vs asserting)
    5. Referential coherence (anaphoric references linking ideas together)
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response = response.strip()
        query = query.strip()
        
        if len(response) < 5:
            return 0.5
        
        import re
        from collections import Counter
        
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.5
        
        # Tokenize into words (lowercase)
        words = re.findall(r'[a-zA-Z]+', response.lower())
        num_words = len(words)
        
        if num_words < 3:
            return 1.0
        
        # ---- Feature 1: Causal/Logical Connective Density ----
        # These are discourse markers that signal reasoning steps
        causal_connectives = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bso that\b',
            r'\bin order to\b', r'\bthis means\b', r'\bthis implies\b',
            r'\bwhich means\b', r'\bwhich leads to\b', r'\bleading to\b',
            r'\bcausing\b', r'\bresulting in\b'
        ]
        
        elaboration_connectives = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bthat is\b', r'\bi\.e\.\b', r'\be\.g\.\b',
            r'\bin other words\b', r'\bto illustrate\b', r'\bto clarify\b'
        ]
        
        contrast_connectives = [
            r'\bhowever\b', r'\bwhereas\b', r'\bon the other hand\b',
            r'\bin contrast\b', r'\bnevertheless\b', r'\balthough\b',
            r'\bwhile\b', r'\bdespite\b', r'\bunlike\b', r'\bbut\b',
            r'\byet\b', r'\bconversely\b'
        ]
        
        sequence_connectives = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bnext\b',
            r'\bthen\b', r'\bfinally\b', r'\bsubsequently\b',
            r'\bafterward\b', r'\bfollowing this\b', r'\bin the first place\b',
            r'\bto begin with\b', r'\blast\b', r'\binitially\b',
            r'\bstep\b', r'\bfirstly\b', r'\bsecondly\b', r'\bthirdly\b'
        ]
        
        additive_connectives = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bnot only\b', r'\bas well\b'
        ]
        
        resp_lower = response.lower()
        
        causal_count = sum(len(re.findall(p, resp_lower)) for p in causal_connectives)
        elaboration_count = sum(len(re.findall(p, resp_lower)) for p in elaboration_connectives)
        contrast_count = sum(len(re.findall(p, resp_lower)) for p in contrast_connectives)
        sequence_count = sum(len(re.findall(p, resp_lower)) for p in sequence_connectives)
        additive_count = sum(len(re.findall(p, resp_lower)) for p in additive_connectives)
        
        # Weight different connective types differently
        total_connective_score = (
            causal_count * 3.0 +
            elaboration_count * 2.5 +
            sequence_count * 2.0 +
            contrast_count * 1.5 +
            additive_count * 1.0
        )
        
        # Normalize by number of sentences to get density
        connective_density = total_connective_score / max(num_sentences, 1)
        # Cap and scale to 0-20
        connective_feature = min(connective_density * 5.0, 20.0)
        
        # ---- Feature 2: Clause Complexity (subordination depth) ----
        # Count subordinating conjunctions and relative pronouns as indicators
        # of complex, elaborative sentence structures
        subordinators = [
            r'\bwhich\b', r'\bthat\b', r'\bwho\b', r'\bwhom\b',
            r'\bwhere\b', r'\bwhen\b', r'\bif\b', r'\bunless\b',
            r'\buntil\b', r'\bafter\b', r'\bbefore\b', r'\bonce\b',
            r'\bwhenever\b', r'\bwherever\b'
        ]
        
        subordination_count = sum(len(re.findall(p, resp_lower)) for p in subordinators)
        # Commas often indicate clause boundaries in complex sentences
        comma_count = response.count(',')
        
        # Average clauses per sentence (rough proxy)
        avg_commas_per_sent = comma_count / max(num_sentences, 1)
        avg_sub_per_sent = subordination_count / max(num_sentences, 1)
        
        clause_complexity = (avg_commas_per_sent * 0.8 + avg_sub_per_sent * 1.5)
        clause_feature = min(clause_complexity * 3.0, 15.0)
        
        # ---- Feature 3: Information Progression ----
        # Measure how new concepts are introduced across sentences
        # Good reasoning progressively introduces new terms
        sentence_word_sets = []
        for s in sentences:
            s_words = set(re.findall(r'[a-zA-Z]{3,}', s.lower()))
            # Remove very common words
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                'have', 'been', 'this', 'that', 'with', 'from', 'they',
                'will', 'would', 'there', 'their', 'what', 'about', 'which',
                'when', 'make', 'like', 'could', 'into', 'than', 'then',
                'also', 'more', 'some', 'them', 'very', 'just', 'where',
                'most', 'should', 'does', 'each', 'much', 'these', 'other'
            }
            s_words -= stopwords
            sentence_word_sets.append(s_words)
        
        if len(sentence_word_sets) > 1:
            new_concept_ratios = []
            cumulative_concepts = set()
            for i, sw in enumerate(sentence_word_sets):
                if i == 0:
                    cumulative_concepts = sw.copy()
                    continue
                if len(sw) > 0:
                    new_concepts = sw - cumulative_concepts
                    ratio = len(new_concepts) / max(len(sw), 1)
                    new_concept_ratios.append(ratio)
                    cumulative_concepts |= sw
            
            if new_concept_ratios:
                avg_new_concept_ratio = sum(new_concept_ratios) / len(new_concept_ratios)
            else:
                avg_new_concept_ratio = 0.0
        else:
            avg_new_concept_ratio = 0.0
        
        # Good progression: moderate ratio (not too repetitive, not too disconnected)
        # Optimal around 0.3-0.6
        if avg_new_concept_ratio < 0.15:
            progression_score = avg_new_concept_ratio * 30  # repetitive
        elif avg_new_concept_ratio <= 0.65:
            progression_score = 5.0 + (avg_new_concept_ratio - 0.15) * 10
        else:
            progression_score = max(0, 10.0 - (avg_new_concept_ratio - 0.65) * 15)
        
        progression_feature = min(max(progression_score, 0), 12.0)
        
        # ---- Feature 4: Explanatory Ratio ----
        # Proportion of text that is explanatory (contains reasoning markers)
        explanation_patterns = [
            r'\bthis means\b', r'\bthis is because\b', r'\bthe reason\b',
            r'\bexplain\b', r'\bindicat\w+\b', r'\bsuggests?\b',
            r'\bimplies?\b', r'\bdemonstrat\w+\b', r'\bshows? that\b',
            r'\brefers? to\b', r'\bdescribes?\b', r'\brepresents?\b',
            r'\bin this case\b', r'\bin this context\b', r'\bhere\b',
            r'\bwhat this means\b', r'\bput simply\b', r'\bessentially\b',
            r'\bfundamentally\b', r'\bat its core\b', r'\bbasically\b'
        ]
        
        explanatory_sentences = 0
        for s in sentences:
            s_lower = s.lower()
            for p in explanation_patterns:
                if re.search(p, s_lower):
                    explanatory_sentences += 1
                    break
        
        explanatory_ratio = explanatory_sentences / max(num_sentences, 1)
        explanatory_feature = explanatory_ratio * 15.0  # up to 15
        
        # ---- Feature 5: Referential Coherence ----
        # Anaphoric references that link back to previous content
        anaphoric_patterns = [
            r'\bthis\b', r'\bthese\b', r'\bthat\b', r'\bthose\b',
            r'\bit\b', r'\bits\b', r'\bsuch\b', r'\bthe former\b',
            r'\bthe latter\b', r'\babove\b', r'\bprevious\b',
            r'\baforementioned\b'
        ]
        
        # Only count in non-first sentences (anaphora needs antecedent)
        anaphoric_count = 0
        if len(sentences) > 1:
            for s in sentences[1:]:
                s_lower = s.lower()
                for p in anaphoric_patterns:
                    anaphoric_count += len(re.findall(p, s_lower))
        
        anaphoric_density = anaphoric_count / max(num_sentences - 1, 1)
        # Moderate density is good (0.5-3 per sentence)
        if anaphoric_density < 0.3:
            coherence_score = anaphoric_density * 10
        elif anaphoric_density <= 3.0:
            coherence_score = 3.0 + (anaphoric_density - 0.3) * 2.5
        else:
            coherence_score = max(3, 10.0 - (anaphoric_density - 3.0) * 1.0)
        
        coherence_feature = min(max(coherence_score, 0), 10.0)
        
        # ---- Feature 6: Response Substantiveness ----
        # Longer, more detailed responses tend to show more reasoning
        # But penalize extremely repetitive content
        unique_words = set(words)
        type_token_ratio = len(unique_words) / max(num_words, 1)
        
        # Penalize very low TTR (repetitive)
        if type_token_ratio < 0.3:
            repetition_penalty = (0.3 - type_token_ratio) * 30
        else:
            repetition_penalty = 0
        
        # Length bonus (logarithmic, with diminishing returns)
        import math
        length_bonus = min(math.log(max(num_words, 1) + 1) * 2.0, 12.0)
        
        # Sentence count bonus (more sentences = more steps, up to a point)
        sentence_bonus = min(num_sentences * 0.8, 8.0)
        
        substantive_feature = length_bonus + sentence_bonus - repetition_penalty
        substantive_feature = max(substantive_feature, 0)
        
        # ---- Feature 7: Query-Response Alignment ----
        # Check if the response addresses the query's core concepts
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        query_stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'this', 'that', 'with', 'from', 'they',
            'will', 'would', 'there', 'their', 'what', 'about', 'which',
            'when', 'make', 'like', 'could', 'into', 'than', 'then',
            'explain', 'describe', 'write', 'give', 'provide', 'how',
            'why', 'what', 'compare', 'contrast', 'discuss'
        }
        query_content = query_words - query_stopwords
        resp_words_set = set(words)
        
        if query_content:
            alignment = len(query_content & resp_words_set) / max(len(query_content), 1)
        else:
            alignment = 0.5
        
        alignment_feature = alignment * 8.0  # up to 8
        
        # ---- Feature 8: Structural Variety ----
        # Measure if the response uses varied sentence structures
        # (different sentence lengths indicate different types of statements)
        sent_lengths = [len(re.findall(r'[a-zA-Z]+', s)) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Moderate variation is good (0.2-0.6)
            if cv < 0.1:
                variety_score = cv * 30  # too uniform
            elif cv <= 0.7:
                variety_score = 3.0 + (cv - 0.1) * 8
            else:
                variety_score = max(2, 8.0 - (cv - 0.7) * 5)
        else:
            variety_score = 1.0
        
        variety_feature = min(max(variety_score, 0), 8.0)
        
        # ---- Combine all features ----
        total_score = (
            connective_feature * 1.0 +      # up to 20
            clause_feature * 0.8 +           # up to 12
            progression_feature * 0.7 +      # up to 8.4
            explanatory_feature * 0.9 +      # up to 13.5
            coherence_feature * 0.6 +        # up to 6
            substantive_feature * 0.5 +      # variable
            alignment_feature * 0.4 +        # up to 3.2
            variety_feature * 0.4            # up to 3.2
        )
        
        # Scale to 0-100 range
        # Theoretical max is roughly 20 + 9.6 + 5.88 + 12.15 + 6 + ~10 + 3.2 + 3.2 ≈ 70
        final_score = min(max(total_score, 0), 100)
        
        return round(final_score, 2)
        
    except Exception:
        try:
            if response and len(response.strip()) > 0:
                return 5.0
            return 0.0
        except Exception:
            return 0.0