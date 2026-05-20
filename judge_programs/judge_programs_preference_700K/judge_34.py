def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure of an LLM response.
    Uses analysis of discourse markers, argument patterns, logical flow,
    and structural coherence indicators.
    
    Returns a score where HIGHER = BETTER quality.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            return 0.0
        
        response_stripped = response.strip()
        if len(response_stripped) < 5:
            return 0.5
        
        import re
        import math
        from collections import Counter
        
        # ===== FEATURE 1: Discourse and Logical Connectors =====
        # Words/phrases that indicate logical flow and structured argumentation
        
        causal_connectors = [
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bthus\b',
            r'\bhence\b', r'\bconsequently\b', r'\bas a result\b',
            r'\bdue to\b', r'\bowing to\b', r'\bthis means\b',
            r'\bwhich leads to\b', r'\bso that\b', r'\bfor this reason\b',
            r'\bit follows that\b', r'\baccordingly\b'
        ]
        
        contrastive_connectors = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bnevertheless\b',
            r'\bon the other hand\b', r'\bconversely\b', r'\byet\b',
            r'\bdespite\b', r'\bin contrast\b', r'\bwhile\b',
            r'\bnonetheless\b', r'\beven though\b', r'\bthat said\b'
        ]
        
        additive_connectors = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b',
            r'\bin addition\b', r'\balso\b', r'\bbesides\b',
            r'\bwhat\'s more\b', r'\bon top of that\b', r'\blikewise\b',
            r'\bsimilarly\b'
        ]
        
        sequential_connectors = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\bto begin\b',
            r'\bin the first place\b', r'\bsubsequently\b', r'\blastly\b',
            r'\bto start\b', r'\bafter that\b'
        ]
        
        exemplification = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bto illustrate\b', r'\bspecifically\b', r'\bin particular\b',
            r'\bnamely\b', r'\be\.g\.\b', r'\bi\.e\.\b', r'\bconsider\b'
        ]
        
        conclusion_markers = [
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\boverall\b', r'\bin short\b', r'\bultimately\b',
            r'\bthe key point\b', r'\bto sum up\b', r'\ball in all\b',
            r'\bthe bottom line\b'
        ]
        
        resp_lower = response.lower()
        
        def count_patterns(patterns, text):
            total = 0
            for p in patterns:
                total += len(re.findall(p, text))
            return total
        
        causal_count = count_patterns(causal_connectors, resp_lower)
        contrastive_count = count_patterns(contrastive_connectors, resp_lower)
        additive_count = count_patterns(additive_connectors, resp_lower)
        sequential_count = count_patterns(sequential_connectors, resp_lower)
        exemplification_count = count_patterns(exemplification, resp_lower)
        conclusion_count = count_patterns(conclusion_markers, resp_lower)
        
        total_connectors = (causal_count + contrastive_count + additive_count +
                           sequential_count + exemplification_count + conclusion_count)
        
        # Diversity of connector types used (0-6 types)
        connector_types_used = sum([
            causal_count > 0,
            contrastive_count > 0,
            additive_count > 0,
            sequential_count > 0,
            exemplification_count > 0,
            conclusion_count > 0
        ])
        
        # ===== FEATURE 2: Sentence-level analysis =====
        sentences = re.split(r'[.!?]+', response_stripped)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        num_sentences = max(len(sentences), 1)
        
        words = response_stripped.split()
        num_words = max(len(words), 1)
        
        avg_sentence_length = num_words / num_sentences
        
        # ===== FEATURE 3: Paragraph structure =====
        paragraphs = [p.strip() for p in response_stripped.split('\n\n') if p.strip()]
        num_paragraphs = len(paragraphs)
        
        # Also count single newline separated blocks
        lines = [l.strip() for l in response_stripped.split('\n') if l.strip()]
        num_lines = len(lines)
        
        # ===== FEATURE 4: Argument quality indicators =====
        # Hedging and nuance (shows careful reasoning)
        hedging_patterns = [
            r'\btend(?:s)? to\b', r'\bgenerally\b', r'\btypically\b',
            r'\bin most cases\b', r'\bit depends\b', r'\bnot necessarily\b',
            r'\bto some extent\b', r'\barguably\b', r'\bpotentially\b',
            r'\boften\b', r'\busually\b', r'\bmight\b', r'\bcould\b',
            r'\bperhaps\b', r'\bpossibly\b', r'\bnuance\b'
        ]
        hedging_count = count_patterns(hedging_patterns, resp_lower)
        
        # Evidence and reference indicators
        evidence_patterns = [
            r'\baccording to\b', r'\bresearch\b', r'\bstud(?:y|ies)\b',
            r'\bevidence\b', r'\bdata\b', r'\bhistorically\b',
            r'\bin practice\b', r'\bin theory\b', r'\bempirically\b',
            r'\bthe literature\b', r'\bscholars?\b', r'\bexperts?\b',
            r'\bsource\b', r'\bcitation\b'
        ]
        evidence_count = count_patterns(evidence_patterns, resp_lower)
        
        # Analytical depth markers
        analysis_patterns = [
            r'\bthe reason\b', r'\bthis is because\b', r'\bthe key\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bfundamentally\b',
            r'\bthe distinction\b', r'\bdistinguish\b', r'\bthe difference\b',
            r'\bin other words\b', r'\bput differently\b', r'\bmore precisely\b',
            r'\bto clarify\b', r'\bthe point is\b', r'\bthe issue\b',
            r'\bthe problem\b', r'\bthe question\b', r'\bthe argument\b',
            r'\bimplication\b', r'\bperspective\b', r'\bframework\b'
        ]
        analysis_count = count_patterns(analysis_patterns, resp_lower)
        
        # ===== FEATURE 5: Structural formatting =====
        # Bullet points, numbered lists, headers
        bullet_count = len(re.findall(r'^\s*[-*•]\s', response, re.MULTILINE))
        numbered_count = len(re.findall(r'^\s*\d+[.)]\s', response, re.MULTILINE))
        has_formatting = bullet_count + numbered_count
        
        # Bold/emphasis markers (markdown)
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', response))
        
        # ===== FEATURE 6: Coherence via topic consistency =====
        # Check if response words relate to query words
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        response_words_set = set(re.findall(r'\b[a-zA-Z]{3,}\b', resp_lower))
        
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'would', 'could', 'should', 'their', 'there', 'about', 'which',
            'when', 'what', 'where', 'with', 'this', 'that', 'from', 'they',
            'will', 'each', 'make', 'like', 'just', 'over', 'such', 'take',
            'than', 'them', 'very', 'some', 'into', 'most', 'other', 'also',
            'more', 'how', 'does', 'did', 'its', 'may', 'any', 'these', 'those'
        }
        
        query_content = query_words - stopwords
        response_content = response_words_set - stopwords
        
        if query_content:
            topic_overlap = len(query_content & response_content) / len(query_content)
        else:
            topic_overlap = 0.5
        
        # ===== FEATURE 7: Contradiction detection (simple heuristic) =====
        contradiction_patterns = [
            (r'\bis\b.*\bis not\b', r'\bis not\b.*\bis\b'),
            (r'\balways\b', r'\bnever\b'),
        ]
        contradiction_score = 0
        for pat1, pat2 in contradiction_patterns:
            if re.search(pat1, resp_lower) and re.search(pat2, resp_lower):
                contradiction_score += 1
        
        # ===== FEATURE 8: Response substantiveness =====
        # Longer, more detailed responses tend to have better argument structure
        # Use log scale to avoid over-rewarding length
        length_score = min(math.log(num_words + 1, 2) / math.log(500, 2), 1.0)
        
        # ===== FEATURE 9: Vocabulary richness (type-token ratio) =====
        all_words_lower = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', response)]
        if all_words_lower:
            # Use root TTR to normalize for length
            unique_words = len(set(all_words_lower))
            ttr = unique_words / math.sqrt(len(all_words_lower)) if len(all_words_lower) > 0 else 0
        else:
            ttr = 0
        
        # ===== FEATURE 10: Sentence length variance =====
        # Good writing varies sentence length
        if num_sentences > 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance_sl = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance_sl)
            # Normalize: some variance is good, too much is bad
            sent_variety_score = min(std_sl / 8.0, 1.0)
        else:
            sent_variety_score = 0.2
        
        # ===== FEATURE 11: Addresses the query directly =====
        # Check if response starts by engaging with the topic
        first_100 = resp_lower[:min(200, len(resp_lower))]
        direct_engagement_patterns = [
            r'\byes\b', r'\bno\b', r'\bessentially\b', r'\bthe\b',
            r'\bgreat question\b', r'\bthat\'s\b', r'\bthis\b',
            r'\bso\b', r'\bwell\b', r'\bin\b', r'\bif\b',
            r'\bwhen\b', r'\ba lot\b', r'\bbeing\b'
        ]
        direct_start = any(re.search(p, first_100) for p in direct_engagement_patterns)
        
        # ===== FEATURE 12: Multi-perspective / nuanced reasoning =====
        perspective_patterns = [
            r'\bon one hand\b', r'\bon the other\b', r'\bfrom .{1,30} perspective\b',
            r'\bsome .{1,20} argue\b', r'\bothers .{1,20} believe\b',
            r'\bboth\b', r'\beither\b', r'\bneither\b',
            r'\bthe trade-off\b', r'\btrade.?off\b', r'\bbalance\b',
            r'\bcomplexity\b', r'\bnuance\b', r'\bsubtlet\b',
            r'\bit\'s not that simple\b', r'\bmore complex\b'
        ]
        perspective_count = count_patterns(perspective_patterns, resp_lower)
        
        # ===== FEATURE 13: Explanation depth =====
        # Clauses per sentence (approximated by commas, semicolons, dashes)
        clause_separators = len(re.findall(r'[,;:\-—]', response))
        clauses_per_sentence = clause_separators / num_sentences if num_sentences > 0 else 0
        
        # ===== SCORING =====
        # Normalize features and combine with weights
        
        # Connector density (per 100 words)
        connector_density = (total_connectors / num_words) * 100
        
        scores = {}
        
        # 1. Logical connectors (density and diversity) - weight: 15
        connector_density_score = min(connector_density / 5.0, 1.0)  # cap at 5 per 100 words
        connector_diversity_score = connector_types_used / 6.0
        scores['connectors'] = (connector_density_score * 0.5 + connector_diversity_score * 0.5) * 15
        
        # 2. Causal reasoning specifically - weight: 10
        causal_density = (causal_count / num_words) * 100
        scores['causal'] = min(causal_density / 2.0, 1.0) * 10
        
        # 3. Length/substantiveness - weight: 12
        scores['length'] = length_score * 12
        
        # 4. Hedging/nuance - weight: 8
        hedging_density = (hedging_count / num_words) * 100
        scores['hedging'] = min(hedging_density / 3.0, 1.0) * 8
        
        # 5. Evidence references - weight: 7
        evidence_density = (evidence_count / num_words) * 100
        scores['evidence'] = min(evidence_density / 2.0, 1.0) * 7
        
        # 6. Analytical depth - weight: 10
        analysis_density = (analysis_count / num_words) * 100
        scores['analysis'] = min(analysis_density / 3.0, 1.0) * 10
        
        # 7. Topic relevance - weight: 8
        scores['relevance'] = topic_overlap * 8
        
        # 8. Vocabulary richness - weight: 6
        scores['vocabulary'] = min(ttr / 8.0, 1.0) * 6
        
        # 9. Sentence variety - weight: 5
        scores['sent_variety'] = sent_variety_score * 5
        
        # 10. Structure/formatting - weight: 5
        formatting_score = min(has_formatting / 4.0, 1.0) * 0.3
        paragraph_score = min(num_paragraphs / 3.0, 1.0) * 0.4
        multi_line_score = min(num_lines / 4.0, 1.0) * 0.3
        scores['structure'] = (formatting_score + paragraph_score + multi_line_score) * 5
        
        # 11. Multi-perspective reasoning - weight: 8
        perspective_density = (perspective_count / num_words) * 100
        scores['perspective'] = min(perspective_density / 2.0, 1.0) * 8
        
        # 12. Clause complexity - weight: 4
        scores['clause_complexity'] = min(clauses_per_sentence / 4.0, 1.0) * 4
        
        # 13. Contradiction penalty - weight: -3 each
        scores['contradiction_penalty'] = -contradiction_score * 3
        
        # 14. Exemplification bonus - weight: 5
        exemplification_density = (exemplification_count / num_words) * 100
        scores['exemplification'] = min(exemplification_density / 2.0, 1.0) * 5
        
        # 15. Sequential organization bonus - weight: 4
        scores['sequential'] = min(sequential_count / 3.0, 1.0) * 4
        
        # 16. Direct engagement bonus - weight: 2
        scores['engagement'] = 2.0 if direct_start else 0.0
        
        # Sum all scores
        total_score = sum(scores.values())
        
        # Normalize to 0-10 range
        # Max theoretical: ~109, but practical max around 60-70
        # Min practical: around 2-5
        normalized = max(0.0, min(10.0, total_score / 7.5))
        
        # Apply slight sigmoid-like transformation for better discrimination
        # This spreads out the middle range
        centered = normalized - 5.0
        transformed = 5.0 + 5.0 * (2.0 / (1.0 + math.exp(-0.6 * centered)) - 1.0)
        
        return round(max(0.0, min(10.0, transformed)), 3)
        
    except Exception:
        # Fallback: return a neutral score based on length
        try:
            return min(max(len(str(response).split()) / 50.0, 0.5), 5.0)
        except Exception:
            return 2.0