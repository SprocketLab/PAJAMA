def judging_function(query, response):
    """
    Evaluate language quality and readability using a substantially different approach:
    - Sentence complexity via clause detection (commas, conjunctions, relative pronouns)
    - Punctuation sophistication (variety of punctuation marks used)
    - Cohesion via pronoun and demonstrative usage
    - Lexical sophistication via average word rarity (approximated by word length distribution skewness)
    - Sentence length variance (good writing has varied sentence lengths)
    - Discourse markers and hedging language
    - Capitalization correctness
    - Repetition penalty via bigram/trigram repetition rates
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        text = response.strip()
        if len(text) < 10:
            return 0.5
        
        words = re.findall(r"[a-zA-Z']+", text)
        if len(words) < 3:
            return 0.5
        
        # Sentence splitting - more careful than simple period split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if not sentences:
            sentences = [text]
        
        num_words = len(words)
        num_sentences = max(len(sentences), 1)
        words_lower = [w.lower() for w in words]
        
        score = 0.0
        
        # ===== 1. Sentence Length Variance (good writing varies sentence length) =====
        sent_word_counts = []
        for s in sentences:
            sw = re.findall(r"[a-zA-Z']+", s)
            sent_word_counts.append(len(sw))
        
        if len(sent_word_counts) > 1:
            mean_sl = sum(sent_word_counts) / len(sent_word_counts)
            variance_sl = sum((x - mean_sl) ** 2 for x in sent_word_counts) / len(sent_word_counts)
            std_sl = math.sqrt(variance_sl)
            # Coefficient of variation - higher means more variety
            cv = std_sl / max(mean_sl, 1)
            # Sweet spot: cv between 0.3 and 0.8 is good variety
            if 0.2 <= cv <= 1.0:
                variance_score = min(cv / 0.5, 1.0) * 10
            elif cv < 0.2:
                variance_score = cv / 0.2 * 5
            else:
                variance_score = max(10 - (cv - 1.0) * 5, 3)
            score += variance_score
        else:
            score += 3  # single sentence, neutral
        
        # ===== 2. Clause Complexity (subordinate clauses, relative pronouns) =====
        subordinating_conjunctions = {
            'because', 'although', 'though', 'while', 'whereas', 'since',
            'unless', 'until', 'before', 'after', 'whenever', 'wherever',
            'whether', 'if', 'once', 'provided', 'assuming', 'given'
        }
        relative_pronouns = {'which', 'whom', 'whose', 'whereby', 'wherein'}
        
        sub_count = sum(1 for w in words_lower if w in subordinating_conjunctions)
        rel_count = sum(1 for w in words_lower if w in relative_pronouns)
        clause_density = (sub_count + rel_count) / max(num_sentences, 1)
        # Good writing: ~0.5-2.0 clauses per sentence
        clause_score = min(clause_density / 1.0, 1.0) * 10
        score += clause_score
        
        # ===== 3. Punctuation Sophistication =====
        punct_types = set()
        for ch in text:
            if ch in '.,;:!?-—–()[]"\'…':
                punct_types.add(ch)
        
        # Count specific sophisticated punctuation
        semicolons = text.count(';')
        colons = text.count(':')
        dashes = text.count('—') + text.count('–') + text.count(' - ')
        parens = text.count('(')
        
        punct_variety_score = min(len(punct_types) / 6.0, 1.0) * 5
        sophistication_bonus = min((semicolons + colons + dashes + parens) / max(num_sentences, 1), 1.0) * 5
        score += punct_variety_score + sophistication_bonus
        
        # ===== 4. Cohesion: Discourse Connectives and Transition Phrases =====
        # Different from variant 3's "transition words" - focusing on cohesive ties
        cohesive_ties = {
            'however', 'moreover', 'furthermore', 'additionally', 'consequently',
            'nevertheless', 'meanwhile', 'subsequently', 'therefore', 'thus',
            'hence', 'accordingly', 'likewise', 'similarly', 'conversely',
            'nonetheless', 'alternatively', 'specifically', 'notably', 'importantly'
        }
        
        # Also check multi-word connectives
        text_lower = text.lower()
        multi_word_connectives = [
            'on the other hand', 'in addition', 'as a result', 'for instance',
            'for example', 'in contrast', 'in particular', 'in other words',
            'at the same time', 'to this end', 'with that said', 'that said',
            'having said that', 'it is worth noting', 'to be specific',
            'more importantly', 'in fact', 'as such', 'to that end'
        ]
        
        single_connective_count = sum(1 for w in words_lower if w in cohesive_ties)
        multi_connective_count = sum(1 for phrase in multi_word_connectives if phrase in text_lower)
        
        total_connectives = single_connective_count + multi_connective_count
        connective_density = total_connectives / max(num_sentences, 1)
        cohesion_score = min(connective_density / 0.5, 1.0) * 10
        score += cohesion_score
        
        # ===== 5. Bigram and Trigram Repetition Penalty =====
        if len(words_lower) >= 4:
            bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
            trigrams = [(words_lower[i], words_lower[i+1], words_lower[i+2]) for i in range(len(words_lower)-2)]
            
            # Filter out very common bigrams (stop word pairs)
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'it', 'and',
                         'or', 'but', 'not', 'no', 'do', 'does', 'did', 'has', 'have', 'had',
                         'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'}
            
            content_bigrams = [b for b in bigrams if b[0] not in stop_words or b[1] not in stop_words]
            content_trigrams = [t for t in trigrams if any(w not in stop_words for w in t)]
            
            if content_bigrams:
                bigram_counts = Counter(content_bigrams)
                repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
                bigram_rep_rate = repeated_bigrams / max(len(set(content_bigrams)), 1)
            else:
                bigram_rep_rate = 0
            
            if content_trigrams:
                trigram_counts = Counter(content_trigrams)
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
                trigram_rep_rate = repeated_trigrams / max(len(set(content_trigrams)), 1)
            else:
                trigram_rep_rate = 0
            
            # Lower repetition is better
            repetition_penalty = (bigram_rep_rate * 5 + trigram_rep_rate * 10)
            score -= min(repetition_penalty, 10)
        
        # ===== 6. Lexical Sophistication via Word Length Distribution Shape =====
        word_lengths = [len(w) for w in words]
        if word_lengths:
            mean_wl = sum(word_lengths) / len(word_lengths)
            
            # Skewness of word length distribution - positive skew means more long words
            if len(word_lengths) > 2:
                std_wl = math.sqrt(sum((x - mean_wl)**2 for x in word_lengths) / len(word_lengths))
                if std_wl > 0:
                    skewness = sum(((x - mean_wl) / std_wl)**3 for x in word_lengths) / len(word_lengths)
                else:
                    skewness = 0
                
                # Moderate positive skewness (0.5-1.5) suggests good vocabulary
                if 0.3 <= skewness <= 2.0:
                    skew_score = 8
                elif skewness > 2.0:
                    skew_score = 5
                else:
                    skew_score = max(3 + skewness * 5, 0)
            else:
                skew_score = 4
            
            # Proportion of "sophisticated" words (7+ characters)
            sophisticated_ratio = sum(1 for wl in word_lengths if wl >= 7) / len(word_lengths)
            # Sweet spot: 15-35% sophisticated words
            if 0.10 <= sophisticated_ratio <= 0.40:
                soph_score = 7
            elif sophisticated_ratio > 0.40:
                soph_score = 5  # might be overly complex
            else:
                soph_score = sophisticated_ratio / 0.10 * 5
            
            score += (skew_score + soph_score) / 2
        
        # ===== 7. Capitalization Correctness =====
        # Check if sentences start with capital letters
        cap_correct = 0
        cap_total = 0
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped[0].isalpha():
                cap_total += 1
                if s_stripped[0].isupper():
                    cap_correct += 1
        
        # Check for inappropriate ALL CAPS words (excluding common acronyms)
        all_caps_words = [w for w in words if w.isupper() and len(w) > 2 
                         and w not in ('THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'YOU',
                                       'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT',
                                       'AI', 'API', 'CPU', 'GPU', 'RAM', 'USB', 'HTML',
                                       'CSS', 'SQL', 'PDF', 'URL', 'HTTP', 'HTTPS')]
        caps_penalty = min(len(all_caps_words) * 2, 5)
        
        if cap_total > 0:
            cap_score = (cap_correct / cap_total) * 5 - caps_penalty
        else:
            cap_score = 3
        score += max(cap_score, 0)
        
        # ===== 8. Pronoun and Demonstrative Cohesion =====
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
                    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
                    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}
        demonstratives = {'this', 'that', 'these', 'those', 'here', 'there', 'such'}
        
        pronoun_count = sum(1 for w in words_lower if w in pronouns)
        demo_count = sum(1 for w in words_lower if w in demonstratives)
        
        # Pronoun density - indicates engagement and cohesion
        pronoun_density = (pronoun_count + demo_count) / max(num_words, 1)
        # Sweet spot: 5-15% pronouns
        if 0.04 <= pronoun_density <= 0.18:
            pronoun_score = 7
        elif pronoun_density > 0.18:
            pronoun_score = 4  # too many, might be vague
        else:
            pronoun_score = pronoun_density / 0.04 * 5
        score += pronoun_score
        
        # ===== 9. Response Completeness and Structure =====
        # Check for numbered lists or bullet points (structured responses)
        has_numbered_list = bool(re.search(r'\d+[\.\)]\s+\w', text))
        has_bullet_list = bool(re.search(r'[-•*]\s+\w', text))
        has_paragraphs = text.count('\n\n') >= 1
        
        structure_score = 0
        if has_numbered_list:
            structure_score += 3
        if has_paragraphs:
            structure_score += 2
        if has_bullet_list and not has_numbered_list:
            structure_score += 2
        
        # Good average sentence length (12-25 words)
        avg_sent_len = num_words / max(num_sentences, 1)
        if 10 <= avg_sent_len <= 28:
            structure_score += 3
        elif avg_sent_len < 10:
            structure_score += max(avg_sent_len / 10 * 3, 0)
        else:
            structure_score += max(3 - (avg_sent_len - 28) * 0.2, 0)
        
        score += min(structure_score, 8)
        
        # ===== 10. Empathy and Engagement Markers =====
        empathy_phrases = [
            "i understand", "i can see", "it's understandable", "it's completely",
            "it's perfectly", "that's understandable", "i hear you", "i'm sorry",
            "i apologize", "we understand", "we appreciate", "we value",
            "don't hesitate", "feel free", "let's", "let us", "together",
            "you're not alone", "it's okay", "it's natural", "it's normal"
        ]
        
        empathy_count = sum(1 for phrase in empathy_phrases if phrase in text_lower)
        empathy_score = min(empathy_count * 2, 8)
        score += empathy_score
        
        # ===== 11. Type-Token Ratio (different from simple vocabulary diversity) =====
        # Use a moving-average TTR for length independence
        if len(words_lower) > 20:
            window_size = 20
            ttr_values = []
            for i in range(0, len(words_lower) - window_size + 1, window_size // 2):
                window = words_lower[i:i + window_size]
                ttr_values.append(len(set(window)) / len(window))
            if ttr_values:
                mattr = sum(ttr_values) / len(ttr_values)
            else:
                mattr = len(set(words_lower)) / len(words_lower)
        else:
            mattr = len(set(words_lower)) / max(len(words_lower), 1)
        
        # MATTR typically 0.6-0.9 for good text
        ttr_score = min(max((mattr - 0.4) / 0.4, 0), 1.0) * 8
        score += ttr_score
        
        # ===== 12. Length Appropriateness =====
        # Longer, more developed responses tend to be better (up to a point)
        length_score = 0
        if num_words >= 50:
            length_score = min((num_words - 30) / 100, 1.0) * 5
        elif num_words >= 20:
            length_score = 2
        else:
            length_score = 1
        score += length_score
        
        # Normalize to roughly 1-5 scale
        # Max theoretical score ~85, min ~0
        # Map to 1-5
        normalized = 1 + (score / 75) * 4
        normalized = max(1.0, min(5.0, normalized))
        
        return round(normalized, 2)
    
    except Exception:
        return 2.5