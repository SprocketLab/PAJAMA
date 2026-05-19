def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of an LLM response.
    
    This variant uses a sentence-level analysis approach, focusing on:
    - Sentence length variance and rhythm
    - Transition word usage for logical flow
    - Information density distribution across sentences
    - Hierarchical structure detection (markdown-style)
    - Repetition penalty at multiple granularities
    - Response completeness and truncation detection
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) == 0:
            return 0.0
        
        score = 0.0
        
        # ============================================================
        # 1. SENTENCE-LEVEL RHYTHM ANALYSIS (0-15 points)
        # Good writing varies sentence length for readability
        # ============================================================
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        rhythm_score = 0.0
        if num_sentences >= 2:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                std_dev = math.sqrt(variance)
                # Coefficient of variation - moderate variation is good
                cv = std_dev / mean_len if mean_len > 0 else 0
                # Sweet spot: CV between 0.2 and 0.6
                if 0.15 <= cv <= 0.7:
                    rhythm_score = 10.0
                elif cv < 0.15:
                    rhythm_score = cv / 0.15 * 7.0  # Too uniform
                else:
                    rhythm_score = max(0, 10.0 - (cv - 0.7) * 5)  # Too erratic
                
                # Bonus for having a mix of short and long sentences
                short = sum(1 for l in sentence_lengths if l <= 8)
                medium = sum(1 for l in sentence_lengths if 9 <= l <= 20)
                long_s = sum(1 for l in sentence_lengths if l > 20)
                categories_used = (short > 0) + (medium > 0) + (long_s > 0)
                rhythm_score += categories_used * 1.5
            else:
                rhythm_score = 2.0
        elif num_sentences == 1:
            # Single sentence - minimal structure
            wc = sentence_lengths[0] if sentence_lengths else 0
            if wc >= 10:
                rhythm_score = 4.0
            else:
                rhythm_score = 2.0
        
        score += min(15.0, rhythm_score)
        
        # ============================================================
        # 2. TRANSITION AND CONNECTIVE WORDS (0-15 points)
        # Measures logical flow between ideas
        # ============================================================
        transition_categories = {
            'addition': ['also', 'additionally', 'furthermore', 'moreover', 'in addition', 'besides', 'as well'],
            'contrast': ['however', 'but', 'although', 'nevertheless', 'on the other hand', 'whereas', 'while', 'yet', 'in contrast', 'conversely'],
            'cause_effect': ['therefore', 'thus', 'consequently', 'as a result', 'because', 'since', 'due to', 'hence'],
            'sequence': ['first', 'second', 'third', 'then', 'next', 'finally', 'subsequently', 'lastly', 'initially'],
            'example': ['for example', 'for instance', 'such as', 'specifically', 'in particular', 'namely'],
            'summary': ['in conclusion', 'to summarize', 'overall', 'in summary', 'ultimately', 'in short'],
            'emphasis': ['importantly', 'notably', 'significantly', 'especially', 'particularly', 'indeed'],
        }
        
        response_lower = response.lower()
        categories_found = set()
        total_transitions = 0
        
        for category, words in transition_categories.items():
            for word in words:
                count = response_lower.count(word)
                if count > 0:
                    categories_found.add(category)
                    total_transitions += count
        
        transition_score = 0.0
        # Reward diversity of transition types
        transition_score += len(categories_found) * 2.5
        # Reward appropriate density (not too many, not too few)
        if num_sentences > 0:
            transition_density = total_transitions / num_sentences
            if 0.1 <= transition_density <= 0.6:
                transition_score += 5.0
            elif transition_density > 0.6:
                transition_score += max(0, 5.0 - (transition_density - 0.6) * 5)
        
        score += min(15.0, transition_score)
        
        # ============================================================
        # 3. STRUCTURAL ELEMENT DIVERSITY (0-20 points)
        # Detects different formatting elements used
        # ============================================================
        structural_score = 0.0
        elements_found = 0
        
        lines = response.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        
        # Detect numbered lists (e.g., "1.", "1)", "Step 1:")
        numbered_pattern = re.compile(r'^\s*(\d+[\.\)\:]|step\s+\d+)', re.IGNORECASE)
        numbered_items = [l for l in non_empty_lines if numbered_pattern.match(l)]
        if len(numbered_items) >= 2:
            structural_score += 5.0
            elements_found += 1
        
        # Detect bullet points (-, *, •, ▪)
        bullet_pattern = re.compile(r'^\s*[-*•▪►→]\s+')
        bullet_items = [l for l in non_empty_lines if bullet_pattern.match(l)]
        if len(bullet_items) >= 2:
            structural_score += 5.0
            elements_found += 1
        
        # Detect headers (markdown # or ALL CAPS lines or lines ending with :)
        header_pattern = re.compile(r'^\s*(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}:?\s*$)')
        colon_header = re.compile(r'^[A-Z][^.!?]{2,30}:\s*$')
        headers = [l for l in non_empty_lines if header_pattern.match(l) or colon_header.match(l)]
        if len(headers) >= 1:
            structural_score += 4.0
            elements_found += 1
        
        # Detect paragraph breaks (multiple paragraphs separated by blank lines)
        paragraph_count = 0
        in_paragraph = False
        for line in lines:
            if line.strip():
                if not in_paragraph:
                    paragraph_count += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        
        if paragraph_count >= 2:
            structural_score += 3.0 + min(3.0, (paragraph_count - 2) * 0.75)
            elements_found += 1
        
        # Detect bold/italic formatting
        if re.search(r'\*\*.+?\*\*|__.+?__', response):
            structural_score += 2.0
            elements_found += 1
        if re.search(r'(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*)|(?<!_)_(?!_).+?(?<!_)_(?!_)', response):
            structural_score += 1.0
        
        # Bonus for using multiple different structural elements
        if elements_found >= 3:
            structural_score += 3.0
        elif elements_found >= 2:
            structural_score += 1.5
        
        score += min(20.0, structural_score)
        
        # ============================================================
        # 4. INFORMATION DENSITY AND DISTRIBUTION (0-15 points)
        # Checks that content is spread across the response, not front-loaded
        # ============================================================
        density_score = 0.0
        words = response.split()
        total_words = len(words)
        
        if total_words >= 10:
            # Split response into thirds and check content distribution
            third = total_words // 3
            parts = [
                ' '.join(words[:third]),
                ' '.join(words[third:2*third]),
                ' '.join(words[2*third:])
            ]
            
            # Count unique content words per part
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'as', 'into', 'through', 'during', 'before', 'after', 'and',
                         'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                         'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                         'than', 'too', 'very', 'just', 'it', 'its', 'that', 'this',
                         'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                         'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
                         'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'}
            
            unique_content_per_part = []
            for part in parts:
                part_words = re.findall(r'[a-z]+', part.lower())
                content_words = set(w for w in part_words if w not in stop_words and len(w) > 2)
                unique_content_per_part.append(len(content_words))
            
            # Check distribution evenness
            if all(u > 0 for u in unique_content_per_part):
                max_u = max(unique_content_per_part)
                min_u = min(unique_content_per_part)
                if max_u > 0:
                    evenness = min_u / max_u
                    density_score += evenness * 8.0
            
            # Overall content richness (unique words / total words ratio)
            all_content = re.findall(r'[a-z]+', response.lower())
            content_only = [w for w in all_content if w not in stop_words and len(w) > 2]
            if len(content_only) > 0:
                uniqueness_ratio = len(set(content_only)) / len(content_only)
                density_score += uniqueness_ratio * 7.0
        elif total_words >= 3:
            density_score = 4.0
        
        score += min(15.0, density_score)
        
        # ============================================================
        # 5. REPETITION PENALTY (0 to -15 points)
        # Penalizes repetitive phrases and words at multiple granularities
        # ============================================================
        repetition_penalty = 0.0
        
        if total_words >= 5:
            # Word-level repetition
            word_list = re.findall(r'[a-z]+', response.lower())
            word_counts = Counter(word_list)
            
            # Check for excessively repeated content words
            content_word_counts = {w: c for w, c in word_counts.items() 
                                   if w not in {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                               'to', 'of', 'in', 'for', 'on', 'with', 'at',
                                               'and', 'but', 'or', 'that', 'this', 'it', 'its',
                                               'be', 'been', 'have', 'has', 'had'} and len(w) > 2}
            
            if content_word_counts:
                max_repeat = max(content_word_counts.values())
                if max_repeat > 5 and total_words < 100:
                    repetition_penalty -= (max_repeat - 5) * 1.5
            
            # Bigram repetition
            if len(word_list) >= 4:
                bigrams = [f"{word_list[i]} {word_list[i+1]}" for i in range(len(word_list)-1)]
                bigram_counts = Counter(bigrams)
                excessive_bigrams = sum(1 for c in bigram_counts.values() if c > 3)
                repetition_penalty -= excessive_bigrams * 2.0
            
            # Trigram repetition (phrase-level)
            if len(word_list) >= 6:
                trigrams = [f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}" for i in range(len(word_list)-2)]
                trigram_counts = Counter(trigrams)
                excessive_trigrams = sum(1 for c in trigram_counts.values() if c > 2)
                repetition_penalty -= excessive_trigrams * 3.0
            
            # Sentence-level repetition
            if num_sentences >= 2:
                sentence_texts = [re.sub(r'[^a-z\s]', '', s.lower()).strip() for s in sentences]
                sent_counter = Counter(sentence_texts)
                duplicate_sents = sum(c - 1 for c in sent_counter.values() if c > 1)
                repetition_penalty -= duplicate_sents * 4.0
        
        score += max(-15.0, repetition_penalty)
        
        # ============================================================
        # 6. COMPLETENESS AND TRUNCATION DETECTION (0-10 points)
        # ============================================================
        completeness_score = 0.0
        
        # Check if response ends properly
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        if last_char in '.!?)"\'':
            completeness_score += 4.0
        elif last_char == ':':
            completeness_score += 1.0  # Might be truncated after a header
        else:
            # Likely truncated
            completeness_score -= 2.0
        
        # Check for balanced parentheses/brackets
        open_parens = response.count('(') - response.count(')')
        open_brackets = response.count('[') - response.count(']')
        if open_parens == 0 and open_brackets == 0:
            completeness_score += 2.0
        else:
            completeness_score -= abs(open_parens) + abs(open_brackets)
        
        # Appropriate length relative to query complexity
        query_words = len(query.split()) if query else 5
        response_to_query_ratio = total_words / max(1, query_words)
        
        if 1.5 <= response_to_query_ratio <= 30:
            completeness_score += 4.0
        elif response_to_query_ratio < 1.5:
            completeness_score += response_to_query_ratio * 2.0
        else:
            completeness_score += max(0, 4.0 - (response_to_query_ratio - 30) * 0.1)
        
        score += min(10.0, max(-3.0, completeness_score))
        
        # ============================================================
        # 7. TOPIC SENTENCE AND OPENING QUALITY (0-10 points)
        # ============================================================
        opening_score = 0.0
        
        if num_sentences >= 1:
            first_sentence = sentences[0]
            first_words = first_sentence.split()
            
            # Check if first sentence establishes context
            if len(first_words) >= 5:
                opening_score += 3.0
            elif len(first_words) >= 3:
                opening_score += 1.5
            
            # Check if response addresses the query topic
            query_content_words = set(re.findall(r'[a-z]{3,}', query.lower())) if query else set()
            query_content_words -= {'the', 'what', 'how', 'why', 'when', 'where', 'which',
                                    'does', 'can', 'could', 'would', 'should', 'will',
                                    'explain', 'describe', 'provide', 'give', 'write',
                                    'following', 'given', 'input'}
            
            first_sent_words = set(re.findall(r'[a-z]{3,}', first_sentence.lower()))
            if query_content_words:
                overlap = len(query_content_words & first_sent_words) / len(query_content_words)
                opening_score += overlap * 4.0
            else:
                opening_score += 2.0
            
            # Check if paragraphs/sections start with clear topic sentences
            if paragraph_count >= 2:
                # Find first sentence of each paragraph
                para_starters = []
                current_para_start = True
                for line in lines:
                    if line.strip():
                        if current_para_start:
                            para_starters.append(line.strip())
                            current_para_start = False
                    else:
                        current_para_start = True
                
                # Topic sentences should be reasonably long
                good_starters = sum(1 for s in para_starters if len(s.split()) >= 5)
                if len(para_starters) > 0:
                    opening_score += (good_starters / len(para_starters)) * 3.0
        
        score += min(10.0, opening_score)
        
        # ============================================================
        # 8. WALL-OF-TEXT PENALTY (0 to -10 points)
        # ============================================================
        wall_penalty = 0.0
        
        if total_words > 80 and paragraph_count <= 1 and len(numbered_items) == 0 and len(bullet_items) == 0:
            # Long response with no structural breaks
            wall_penalty = -min(10.0, (total_words - 80) * 0.05)
        elif total_words > 150 and paragraph_count <= 2 and len(numbered_items) == 0 and len(bullet_items) == 0:
            wall_penalty = -min(7.0, (total_words - 150) * 0.03)
        
        score += wall_penalty
        
        # ============================================================
        # FINAL NORMALIZATION
        # ============================================================
        # Theoretical range: roughly -25 to 85, normalize to 0-100
        score = max(0.0, min(100.0, score + 15.0))  # shift up and clamp
        
        # Scale to 0-10 for cleaner output
        final_score = round(score / 10.0, 2)
        
        return final_score
        
    except Exception as e:
        # Fallback: return a minimal score based on response length
        try:
            if response and len(response.strip()) > 0:
                return 3.0
            return 0.0
        except:
            return 0.0