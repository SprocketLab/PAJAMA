def judging_function(query, response):
    """
    Evaluates language quality and readability using a unique approach based on:
    - Punctuation sophistication (variety and correctness of punctuation usage)
    - Clause density and sentence complexity via comma/semicolon analysis
    - Discourse markers and cohesion signals
    - Lexical sophistication via word frequency tiers (common vs uncommon words)
    - Sentence rhythm variation (standard deviation of sentence lengths)
    - Capitalization correctness
    - Response completeness and structural signals
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 5:
            return 0.5
        
        # ---- 1. Punctuation Sophistication Score ----
        # Measures variety and appropriate density of punctuation
        punct_types = {
            'period': len(re.findall(r'\.(?!\.\.)(?!\d)', text)),
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'dash': len(re.findall(r'[—–-]{1,2}', text)),
            'question': text.count('?'),
            'exclamation': text.count('!'),
            'parentheses': text.count('(') + text.count(')'),
            'quotes': text.count('"') + text.count("'") + text.count('"') + text.count('"'),
            'ellipsis': len(re.findall(r'\.{3}|…', text)),
        }
        
        punct_variety = sum(1 for v in punct_types.values() if v > 0)
        # Normalize: max variety around 7-8 types is excellent
        punct_variety_score = min(punct_variety / 6.0, 1.0) * 10
        
        total_punct = sum(punct_types.values())
        words = re.findall(r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b', text)
        word_count = len(words)
        
        if word_count == 0:
            return 1.0
        
        # Punctuation density (punctuation per word) - sweet spot around 0.1-0.2
        punct_density = total_punct / max(word_count, 1)
        if punct_density < 0.02:
            punct_density_score = 2.0
        elif punct_density < 0.08:
            punct_density_score = 5.0 + (punct_density - 0.02) * 50
        elif punct_density <= 0.25:
            punct_density_score = 8.0
        else:
            punct_density_score = max(3.0, 8.0 - (punct_density - 0.25) * 15)
        
        # ---- 2. Clause Density & Sentence Complexity ----
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)
        
        # Commas per sentence as proxy for clause complexity
        commas_per_sent = punct_types['comma'] / num_sentences
        # Sweet spot: 1-3 commas per sentence
        if commas_per_sent < 0.3:
            clause_score = 4.0
        elif commas_per_sent <= 3.5:
            clause_score = 4.0 + (commas_per_sent - 0.3) * 1.875
        else:
            clause_score = max(4.0, 10.0 - (commas_per_sent - 3.5) * 1.5)
        
        # Semicolons and colons boost (sophisticated punctuation)
        sophisticated_punct = punct_types['semicolon'] + punct_types['colon'] + punct_types['dash']
        sophistication_bonus = min(sophisticated_punct * 0.5, 2.0)
        clause_score = min(10.0, clause_score + sophistication_bonus)
        
        # ---- 3. Discourse Markers & Cohesion ----
        text_lower = text.lower()
        
        # Categorized discourse markers
        additive = ['furthermore', 'moreover', 'additionally', 'in addition', 'also', 'besides',
                     'as well', 'not only', 'likewise', 'similarly']
        adversative = ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'although',
                       'though', 'yet', 'but', 'whereas', 'while', 'despite', 'in contrast',
                       'conversely', 'instead', 'rather']
        causal = ['therefore', 'thus', 'consequently', 'hence', 'as a result', 'because',
                  'since', 'so that', 'due to', 'owing to', 'accordingly']
        temporal = ['first', 'second', 'third', 'then', 'next', 'finally', 'subsequently',
                    'previously', 'meanwhile', 'afterwards', 'initially', 'eventually']
        exemplifying = ['for example', 'for instance', 'such as', 'specifically',
                        'in particular', 'namely', 'to illustrate']
        summarizing = ['in summary', 'in conclusion', 'overall', 'essentially', 'in short',
                       'to summarize', 'in other words', 'that is']
        
        categories_found = 0
        total_markers = 0
        for category in [additive, adversative, causal, temporal, exemplifying, summarizing]:
            cat_count = sum(1 for marker in category if marker in text_lower)
            if cat_count > 0:
                categories_found += 1
                total_markers += cat_count
        
        # Score based on variety of discourse marker categories and density
        marker_density = total_markers / max(num_sentences, 1)
        discourse_score = min(categories_found * 1.5, 6.0) + min(marker_density * 3.0, 4.0)
        discourse_score = min(10.0, discourse_score)
        
        # ---- 4. Lexical Sophistication via Word Frequency Tiers ----
        # Define very common words (top ~200 most frequent English words)
        very_common = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
            'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'has', 'had',
            'did', 'does', 'am', 'being', 'having', 'doing', 'got', 'more', 'very',
            'much', 'too', 'really', 'here', 'still', 'should', 'may', 'might',
            'must', 'shall', 'need', 'own', 'thing', 'things', 'right', 'same',
            'each', 'tell', 'many', 'those', 'such', 'through', 'while', 'where',
            'before', 'between', 'both', 'under', 'never', 'always', 'last', 'long',
            'great', 'little', 'old', 'big', 'few', 'off', 'down', 'first', 'every',
        }
        
        words_lower = [w.lower() for w in words]
        
        if word_count > 0:
            common_ratio = sum(1 for w in words_lower if w in very_common) / word_count
            uncommon_ratio = 1.0 - common_ratio
        else:
            uncommon_ratio = 0.0
        
        # Average word length of uncommon words (longer uncommon words = more sophisticated)
        uncommon_words = [w for w in words_lower if w not in very_common and len(w) > 2]
        if uncommon_words:
            avg_uncommon_len = sum(len(w) for w in uncommon_words) / len(uncommon_words)
        else:
            avg_uncommon_len = 0
        
        # Sweet spot for uncommon ratio: 0.4-0.7
        if uncommon_ratio < 0.2:
            lexical_score = 3.0
        elif uncommon_ratio <= 0.7:
            lexical_score = 3.0 + (uncommon_ratio - 0.2) * 14.0
        else:
            lexical_score = max(5.0, 10.0 - (uncommon_ratio - 0.7) * 10.0)
        
        # Bonus for longer uncommon words (avg 6-9 chars is good)
        if avg_uncommon_len >= 6:
            lexical_score = min(10.0, lexical_score + min((avg_uncommon_len - 6) * 0.5, 2.0))
        
        # ---- 5. Sentence Rhythm Variation ----
        sent_lengths = []
        for s in sentences:
            s_words = re.findall(r'\b[a-zA-Z]+\b', s)
            if s_words:
                sent_lengths.append(len(s_words))
        
        if len(sent_lengths) >= 2:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_dev = math.sqrt(variance)
            # Coefficient of variation
            cv = std_dev / max(mean_len, 1)
            # Good rhythm: CV between 0.3 and 0.8
            if cv < 0.1:
                rhythm_score = 3.0  # Too monotonous
            elif cv <= 0.8:
                rhythm_score = 3.0 + (cv - 0.1) * 10.0
                rhythm_score = min(10.0, rhythm_score)
            else:
                rhythm_score = max(4.0, 10.0 - (cv - 0.8) * 5.0)
            
            # Penalize very short average sentence length
            if mean_len < 5:
                rhythm_score *= 0.6
            elif mean_len < 8:
                rhythm_score *= 0.8
            elif mean_len > 35:
                rhythm_score *= 0.8
        elif len(sent_lengths) == 1:
            # Single sentence - neutral
            rhythm_score = 5.0
        else:
            rhythm_score = 3.0
        
        # ---- 6. Capitalization Correctness ----
        # Check sentence-initial capitalization
        cap_correct = 0
        cap_total = 0
        for s in sentences:
            s = s.strip()
            if s and s[0].isalpha():
                cap_total += 1
                if s[0].isupper():
                    cap_correct += 1
        
        if cap_total > 0:
            cap_score = (cap_correct / cap_total) * 10.0
        else:
            cap_score = 5.0
        
        # Check for excessive ALL CAPS words (more than expected)
        all_caps_words = [w for w in words if len(w) > 2 and w.isupper() and w not in 
                          {'SQL', 'HTML', 'CSS', 'API', 'USA', 'UK', 'EU', 'AI', 'PE', 'EIT',
                           'HVAC', 'DC', 'NOT', 'AND', 'THE', 'PhD', 'CEO', 'CTO', 'HTTP',
                           'JSON', 'XML', 'RAM', 'CPU', 'GPU', 'DNS', 'URL', 'FAQ', 'PDF',
                           'EE', 'MIRI'}]
        if word_count > 0:
            caps_ratio = len(all_caps_words) / word_count
            if caps_ratio > 0.05:
                cap_score *= 0.8
        
        # ---- 7. Response Length & Completeness ----
        # Longer, more substantive responses tend to be better (up to a point)
        if word_count < 10:
            length_score = 2.0
        elif word_count < 25:
            length_score = 4.0
        elif word_count < 50:
            length_score = 6.0
        elif word_count < 100:
            length_score = 7.5
        elif word_count < 200:
            length_score = 9.0
        elif word_count < 400:
            length_score = 10.0
        else:
            length_score = 9.5
        
        # Check if response seems truncated
        if text[-1] not in '.!?"\')]}…' and word_count > 20:
            # Might be truncated, slight penalty
            length_score *= 0.9
        
        # ---- 8. Structural Elements ----
        structural_score = 5.0
        
        # Lists (bullet points, numbered)
        has_lists = bool(re.search(r'(?m)^\s*[-*•]\s', text) or re.search(r'(?m)^\s*\d+[.)]\s', text))
        if has_lists:
            structural_score += 1.5
        
        # Code blocks
        has_code = '```' in text or bool(re.search(r'(?m)^    \S', text))
        if has_code:
            structural_score += 1.0
        
        # Bold/italic formatting
        has_formatting = bool(re.search(r'\*\*.+?\*\*', text) or re.search(r'\*.+?\*', text))
        if has_formatting:
            structural_score += 0.5
        
        # Multiple paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            structural_score += 1.0
        if len(paragraphs) >= 3:
            structural_score += 0.5
        
        structural_score = min(10.0, structural_score)
        
        # ---- 9. Spelling Heuristic (repeated characters, common error patterns) ----
        spelling_score = 10.0
        
        # Words with 3+ repeated consecutive characters (likely errors)
        repeated_chars = len(re.findall(r'\b\w*(.)\1{2,}\w*\b', text))
        spelling_score -= min(repeated_chars * 1.0, 3.0)
        
        # Very short words that are likely typos (single consonants that aren't 'I' or common abbrevs)
        single_chars = re.findall(r'\b([a-zA-Z])\b', text)
        bad_singles = [c for c in single_chars if c.lower() not in {'i', 'a', 'o', 'x'}]
        if word_count > 0:
            bad_single_ratio = len(bad_singles) / word_count
            if bad_single_ratio > 0.03:
                spelling_score -= 2.0
        
        # Double spaces (sloppy formatting)
        double_spaces = len(re.findall(r'  +', text))
        if double_spaces > 3:
            spelling_score -= 1.0
        
        spelling_score = max(2.0, spelling_score)
        
        # ---- Combine all scores with weights ----
        weights = {
            'punct_variety': 0.08,
            'punct_density': 0.07,
            'clause': 0.10,
            'discourse': 0.15,
            'lexical': 0.12,
            'rhythm': 0.10,
            'capitalization': 0.06,
            'length': 0.15,
            'structural': 0.08,
            'spelling': 0.09,
        }
        
        scores = {
            'punct_variety': punct_variety_score,
            'punct_density': punct_density_score,
            'clause': clause_score,
            'discourse': discourse_score,
            'lexical': lexical_score,
            'rhythm': rhythm_score,
            'capitalization': cap_score,
            'length': length_score,
            'structural': structural_score,
            'spelling': spelling_score,
        }
        
        final_score = sum(weights[k] * scores[k] for k in weights)
        
        # Scale to 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        return round(final_score, 3)
        
    except Exception:
        # Fallback: return a middle score based on length
        try:
            return min(5.0, len(str(response)) / 100.0)
        except Exception:
            return 2.5