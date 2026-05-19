def judging_function(query, response):
    """
    Evaluates logical coherence and argument structure using:
    - Sentence-level dependency chain analysis (do later sentences reference/build on earlier ones?)
    - Discourse marker categorization (causal, contrastive, additive, temporal)
    - Contradiction detection via negation pattern analysis
    - Information density and progression measurement
    - Structural completeness (intro/body/conclusion patterns)
    - Repetition/circularity detection via sentence similarity using character trigrams
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not response.strip():
            return 0.0
        
        if not query or not query.strip():
            return 3.0
        
        response_clean = response.strip()
        query_clean = query.strip()
        
        # Split into sentences using multiple delimiters
        def split_sentences(text):
            # Split on sentence-ending punctuation
            raw = re.split(r'(?<=[.!?])\s+', text)
            # Also split on newlines if they separate content
            sentences = []
            for r in raw:
                parts = re.split(r'\n+', r)
                sentences.extend(parts)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        sentences = split_sentences(response_clean)
        words = re.findall(r'[a-zA-Z]+', response_clean.lower())
        word_count = len(words)
        
        # === 1. Basic length and substance check ===
        if word_count < 3:
            return 0.5
        
        length_score = min(1.0, math.log(1 + word_count) / math.log(1 + 80))
        
        # === 2. Character trigram based circularity/repetition detection ===
        def char_trigrams(text):
            text = text.lower()
            trigrams = Counter()
            for i in range(len(text) - 2):
                trigrams[text[i:i+3]] += 1
            return trigrams
        
        def trigram_similarity(t1, t2):
            c1 = char_trigrams(t1)
            c2 = char_trigrams(t2)
            if not c1 or not c2:
                return 0.0
            intersection = sum((c1 & c2).values())
            union = sum((c1 | c2).values())
            return intersection / union if union > 0 else 0.0
        
        # Detect circular reasoning: high similarity between non-adjacent sentences
        circularity_penalty = 0.0
        if len(sentences) >= 3:
            high_sim_count = 0
            pair_count = 0
            for i in range(len(sentences)):
                for j in range(i + 2, len(sentences)):
                    sim = trigram_similarity(sentences[i], sentences[j])
                    if sim > 0.7:
                        high_sim_count += 1
                    pair_count += 1
            if pair_count > 0:
                circularity_penalty = min(1.0, (high_sim_count / max(pair_count, 1)) * 3.0)
        
        # Also detect consecutive repetition
        consecutive_rep = 0
        if len(sentences) >= 2:
            for i in range(len(sentences) - 1):
                sim = trigram_similarity(sentences[i], sentences[i+1])
                if sim > 0.75:
                    consecutive_rep += 1
            consecutive_rep_ratio = consecutive_rep / (len(sentences) - 1)
        else:
            consecutive_rep_ratio = 0.0
        
        repetition_penalty = min(1.0, (circularity_penalty + consecutive_rep_ratio) / 1.5)
        
        # === 3. Discourse marker categorization and scoring ===
        discourse_markers = {
            'causal': [
                'because', 'therefore', 'thus', 'hence', 'consequently',
                'as a result', 'due to', 'since', 'so that', 'for this reason',
                'it follows', 'leading to', 'caused by', 'this means', 'implies',
                'accordingly', 'thereby'
            ],
            'contrastive': [
                'however', 'although', 'nevertheless', 'on the other hand',
                'in contrast', 'despite', 'whereas', 'but', 'yet', 'still',
                'conversely', 'nonetheless', 'even though', 'while', 'instead',
                'rather than', 'on the contrary'
            ],
            'additive': [
                'furthermore', 'moreover', 'in addition', 'additionally',
                'also', 'besides', 'similarly', 'likewise', 'as well as',
                'not only', 'along with', 'coupled with'
            ],
            'temporal': [
                'first', 'second', 'third', 'then', 'next', 'finally',
                'subsequently', 'previously', 'before', 'after', 'meanwhile',
                'initially', 'eventually', 'at first', 'in the end', 'lastly',
                'to begin', 'following this'
            ],
            'exemplification': [
                'for example', 'for instance', 'such as', 'specifically',
                'in particular', 'to illustrate', 'namely', 'including'
            ],
            'conclusion': [
                'in conclusion', 'to summarize', 'in summary', 'overall',
                'in short', 'to conclude', 'ultimately', 'all in all'
            ]
        }
        
        response_lower = response_clean.lower()
        marker_counts = {}
        total_markers = 0
        category_diversity = 0
        
        for category, markers in discourse_markers.items():
            count = 0
            for marker in markers:
                occurrences = len(re.findall(r'\b' + re.escape(marker) + r'\b', response_lower))
                count += occurrences
            marker_counts[category] = count
            total_markers += count
            if count > 0:
                category_diversity += 1
        
        # Score based on density and diversity of discourse markers
        marker_density = total_markers / max(word_count, 1) * 100  # per 100 words
        marker_density_score = min(1.0, marker_density / 4.0)  # expect ~4 per 100 words max
        diversity_score = min(1.0, category_diversity / 3.0)  # having 3+ categories is good
        
        discourse_score = 0.5 * marker_density_score + 0.5 * diversity_score
        
        # === 4. Information progression: do sentences introduce new content? ===
        def get_content_words(text):
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                'our', 'their', 'and', 'or', 'but', 'not', 'no', 'if', 'so', 'than',
                'too', 'very', 'just', 'about', 'up', 'out', 'all', 'some', 'any',
                'each', 'every', 'both', 'few', 'more', 'most', 'other', 'such',
                'only', 'own', 'same', 'here', 'there', 'when', 'where', 'how', 'what',
                'which', 'who', 'whom', 'why'
            }
            wds = re.findall(r'[a-zA-Z]+', text.lower())
            return set(w for w in wds if w not in stop_words and len(w) > 2)
        
        progression_score = 0.0
        if len(sentences) >= 2:
            cumulative_words = set()
            new_info_ratios = []
            for sent in sentences:
                content = get_content_words(sent)
                if content:
                    new_words = content - cumulative_words
                    ratio = len(new_words) / len(content)
                    new_info_ratios.append(ratio)
                    cumulative_words.update(content)
            
            if new_info_ratios:
                # Good progression means each sentence adds some new info
                avg_new = sum(new_info_ratios) / len(new_info_ratios)
                progression_score = min(1.0, avg_new / 0.5)  # 50%+ new content is great
        else:
            progression_score = 0.4  # single sentence gets moderate score
        
        # === 5. Sentence-level coherence chain: adjacent sentences share some content ===
        coherence_chain_score = 0.0
        if len(sentences) >= 2:
            chain_links = []
            for i in range(len(sentences) - 1):
                c1 = get_content_words(sentences[i])
                c2 = get_content_words(sentences[i + 1])
                if c1 and c2:
                    shared = len(c1 & c2)
                    total = len(c1 | c2)
                    link_strength = shared / total if total > 0 else 0
                    chain_links.append(link_strength)
            
            if chain_links:
                # We want moderate overlap (0.1-0.4) - too low means disconnected, too high means repetitive
                avg_link = sum(chain_links) / len(chain_links)
                if avg_link < 0.05:
                    coherence_chain_score = 0.2
                elif avg_link < 0.15:
                    coherence_chain_score = 0.7
                elif avg_link < 0.35:
                    coherence_chain_score = 1.0
                elif avg_link < 0.5:
                    coherence_chain_score = 0.6
                else:
                    coherence_chain_score = 0.3  # too much overlap = repetitive
        else:
            coherence_chain_score = 0.3
        
        # === 6. Negation contradiction detection ===
        contradiction_penalty = 0.0
        negation_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bcan\b', r'\bcannot\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\bshould\b', r'\bshould not\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\bcorrect\b', r'\bincorrect\b'),
            (r'\bpossible\b', r'\bimpossible\b'),
        ]
        
        for pos_pat, neg_pat in negation_patterns:
            pos_matches = re.findall(pos_pat, response_lower)
            neg_matches = re.findall(neg_pat, response_lower)
            if pos_matches and neg_matches:
                # Check if they appear in close proximity (same paragraph context)
                # This is a soft signal, not definitive
                contradiction_penalty += 0.05
        
        contradiction_penalty = min(0.3, contradiction_penalty)
        
        # === 7. Structural completeness ===
        # Does the response have identifiable structure?
        structural_score = 0.0
        
        # Check for enumeration/listing structure
        has_enumeration = bool(re.search(r'(?:^|\n)\s*(?:\d+[.)]|[-•*])\s', response_clean))
        
        # Check for paragraph structure
        paragraphs = [p.strip() for p in response_clean.split('\n\n') if p.strip()]
        has_paragraphs = len(paragraphs) >= 2
        
        # Check for complete sentences (ending with punctuation)
        complete_sentences = sum(1 for s in sentences if re.search(r'[.!?]$', s.strip()))
        completeness_ratio = complete_sentences / max(len(sentences), 1)
        
        if has_enumeration:
            structural_score += 0.3
        if has_paragraphs:
            structural_score += 0.2
        structural_score += completeness_ratio * 0.5
        structural_score = min(1.0, structural_score)
        
        # === 8. Relevance to query ===
        query_content = get_content_words(query_clean)
        response_content = get_content_words(response_clean)
        
        if query_content and response_content:
            relevance = len(query_content & response_content) / len(query_content)
            relevance_score = min(1.0, relevance * 2.0)  # having half the query words is good
        else:
            relevance_score = 0.3
        
        # === 9. Garbage/noise detection ===
        noise_penalty = 0.0
        
        # HTML/code artifacts
        html_count = len(re.findall(r'<[^>]+>', response_clean))
        if html_count > 2:
            noise_penalty += min(0.4, html_count * 0.05)
        
        # Excessive special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()\-]', response_clean)) / max(len(response_clean), 1)
        if special_ratio > 0.1:
            noise_penalty += min(0.3, special_ratio * 2)
        
        # Repeated patterns (like "Output:" appearing multiple times is ok for lists)
        # But exact duplicate lines are bad
        lines = [l.strip() for l in response_clean.split('\n') if l.strip()]
        if lines:
            unique_lines = set(lines)
            duplicate_ratio = 1.0 - len(unique_lines) / len(lines)
            if duplicate_ratio > 0.3:
                noise_penalty += min(0.3, duplicate_ratio * 0.5)
        
        noise_penalty = min(0.6, noise_penalty)
        
        # === 10. Response truncation detection ===
        truncation_penalty = 0.0
        if response_clean and response_clean[-1] not in '.!?"\')]}':
            # Might be truncated
            last_sentence = sentences[-1] if sentences else ""
            if len(last_sentence) > 20 and not re.search(r'[.!?]$', last_sentence):
                truncation_penalty = 0.1
        
        # === Combine all scores ===
        # Weights chosen to emphasize logical structure aspects
        raw_score = (
            0.12 * length_score +
            0.18 * discourse_score +
            0.15 * progression_score +
            0.15 * coherence_chain_score +
            0.15 * structural_score +
            0.10 * relevance_score +
            0.15 * (1.0 - repetition_penalty)
        )
        
        # Apply penalties
        raw_score = raw_score * (1.0 - noise_penalty)
        raw_score = raw_score * (1.0 - contradiction_penalty)
        raw_score = raw_score - truncation_penalty
        
        # Scale to 0-10
        final_score = max(0.0, min(10.0, raw_score * 10.0))
        
        # Apply floor for very short but non-empty responses
        if word_count < 5:
            final_score = min(final_score, 2.0)
        elif word_count < 10:
            final_score = min(final_score, 4.0)
        
        return round(final_score, 2)
        
    except Exception:
        # Fallback: return a middle-ground score based on length
        try:
            words = len(response.split()) if response else 0
            return min(5.0, max(0.5, words * 0.1))
        except Exception:
            return 3.0