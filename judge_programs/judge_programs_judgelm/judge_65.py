def judging_function(query, response):
    """
    Evaluates structural organization and formatting of an LLM response.
    Returns a score where HIGHER = BETTER quality.
    
    This variant focuses on a feature-weighted linear model approach,
    analyzing multiple structural dimensions and combining them with
    learned weights.
    """
    try:
        if not response or not isinstance(response, str):
            return 0.0
        
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        
        if len(response) == 0:
            return 0.0
        
        import re
        import math
        
        scores = {}
        
        # === Feature 1: Minimum viable length ===
        # Very short responses are almost always poorly structured
        resp_len = len(response)
        if resp_len < 5:
            return 0.5
        elif resp_len < 20:
            scores['length'] = 1.0
        elif resp_len < 50:
            scores['length'] = 3.0
        elif resp_len < 150:
            scores['length'] = 5.0
        elif resp_len < 500:
            scores['length'] = 6.5
        elif resp_len < 1500:
            scores['length'] = 7.0
        else:
            scores['length'] = 6.0  # very long can be wall-of-text
        
        # === Feature 2: Sentence structure ===
        # Count sentences by splitting on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            scores['sentences'] = 1.0
        elif num_sentences == 1:
            scores['sentences'] = 3.0
        elif 2 <= num_sentences <= 5:
            scores['sentences'] = 7.0
        elif 6 <= num_sentences <= 12:
            scores['sentences'] = 8.0
        else:
            scores['sentences'] = 6.5
        
        # === Feature 3: Paragraph structure ===
        # Split by double newlines or single newlines
        paragraphs = re.split(r'\n\s*\n', response)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)
        
        lines = response.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        num_lines = len(lines)
        
        if num_paragraphs >= 2 and resp_len > 100:
            scores['paragraphs'] = 8.0
        elif num_lines >= 3 and resp_len > 80:
            scores['paragraphs'] = 7.0
        elif num_paragraphs == 1 and resp_len > 300:
            # Wall of text penalty
            scores['paragraphs'] = 3.0
        else:
            scores['paragraphs'] = 5.0
        
        # === Feature 4: Formatting elements ===
        # Check for numbered lists
        numbered_list = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+\S', response)
        # Check for bullet points
        bullet_points = re.findall(r'(?:^|\n)\s*[-•*]\s+\S', response)
        # Check for headers (markdown style or caps)
        headers_md = re.findall(r'(?:^|\n)\s*#{1,6}\s+\S', response)
        # Check for bold/italic markdown
        bold_italic = re.findall(r'\*{1,3}.+?\*{1,3}', response)
        # Check for colons used as labels
        label_colons = re.findall(r'(?:^|\n)\s*[A-Z][A-Za-z\s]{1,30}:\s', response)
        # HTML tags
        html_tags = re.findall(r'<[a-zA-Z][^>]*>', response)
        
        formatting_score = 5.0  # baseline
        has_lists = len(numbered_list) >= 2 or len(bullet_points) >= 2
        has_headers = len(headers_md) >= 1
        has_labels = len(label_colons) >= 1
        
        if has_lists:
            formatting_score += 2.0
        if has_headers:
            formatting_score += 1.5
        if has_labels:
            formatting_score += 1.0
        if len(bold_italic) >= 1:
            formatting_score += 0.5
        
        # Cap formatting score
        formatting_score = min(formatting_score, 10.0)
        scores['formatting'] = formatting_score
        
        # === Feature 5: Repetition detection ===
        # Repetitive text is a sign of poor quality
        words = response.lower().split()
        if len(words) > 10:
            # Check for repeated phrases (trigrams)
            trigrams = []
            for i in range(len(words) - 2):
                trigrams.append(' '.join(words[i:i+3]))
            
            if trigrams:
                from collections import Counter
                trigram_counts = Counter(trigrams)
                max_trigram_freq = max(trigram_counts.values())
                unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
                
                if max_trigram_freq > 5 or unique_ratio < 0.4:
                    scores['repetition'] = 2.0
                elif max_trigram_freq > 3 or unique_ratio < 0.6:
                    scores['repetition'] = 4.0
                else:
                    scores['repetition'] = 7.5
            else:
                scores['repetition'] = 5.0
        else:
            scores['repetition'] = 5.0
        
        # === Feature 6: Coherence signals ===
        # Look for transition words, topic sentences
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'moreover',
            'in addition', 'on the other hand', 'for example', 'for instance',
            'in conclusion', 'first', 'second', 'third', 'finally', 'also',
            'meanwhile', 'consequently', 'as a result', 'in summary',
            'specifically', 'notably', 'importantly'
        ]
        resp_lower = response.lower()
        transition_count = sum(1 for tw in transition_words if tw in resp_lower)
        
        if transition_count >= 3:
            scores['coherence'] = 8.0
        elif transition_count >= 1:
            scores['coherence'] = 6.0
        else:
            scores['coherence'] = 4.5
        
        # === Feature 7: Noise and garbage detection ===
        # Check for broken/garbage content
        garbage_signals = 0
        
        # Repeated characters
        if re.search(r'(.)\1{5,}', response):
            garbage_signals += 2
        
        # Lots of "Output:" or "Input:" repetitions (seen in bad examples)
        if response.count('Output:') > 2 or response.count('Input:') > 2:
            garbage_signals += 2
        
        # Random code when not asked for code
        code_keywords = ['import ', 'def ', 'class ', 'return ', 'print(']
        is_code_query = any(kw in query.lower() for kw in ['code', 'python', 'program', 'function', 'script', 'html', 'css', 'javascript'])
        if not is_code_query:
            code_count = sum(1 for ck in code_keywords if ck in response)
            if code_count >= 3:
                garbage_signals += 2
        
        # Truncation detection (ends mid-word or mid-sentence without punctuation)
        if resp_len > 100 and not response[-1] in '.!?"\')]}':
            last_words = response[-30:]
            if not any(c in last_words for c in '.!?'):
                garbage_signals += 1
        
        # Question-answer format pollution (asking new questions)
        qa_pollution = len(re.findall(r'(?:^|\n)(?:Question|Answer):', response))
        if qa_pollution > 2:
            garbage_signals += 2
        
        if garbage_signals >= 4:
            scores['noise'] = 1.0
        elif garbage_signals >= 2:
            scores['noise'] = 3.0
        elif garbage_signals >= 1:
            scores['noise'] = 5.0
        else:
            scores['noise'] = 7.5
        
        # === Feature 8: Proportionality to query ===
        # Response should be proportional to query complexity
        query_len = len(query.strip())
        query_words = len(query.split())
        resp_words = len(words)
        
        if query_words > 0 and resp_words > 0:
            ratio = resp_words / max(query_words, 1)
            if ratio < 0.3:
                # Very short response to a real question
                scores['proportionality'] = 2.0
            elif 0.3 <= ratio < 1.0:
                scores['proportionality'] = 4.0
            elif 1.0 <= ratio <= 15.0:
                scores['proportionality'] = 7.0
            elif ratio <= 30.0:
                scores['proportionality'] = 6.0
            else:
                scores['proportionality'] = 4.5
        else:
            scores['proportionality'] = 5.0
        
        # === Feature 9: Sentence length variance ===
        # Good writing has varied sentence lengths
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len > 0:
                variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean_len if mean_len > 0 else 0
                
                if 0.2 <= cv <= 0.8:
                    scores['sent_variance'] = 7.5
                elif cv < 0.2:
                    scores['sent_variance'] = 5.0  # too uniform
                else:
                    scores['sent_variance'] = 5.5  # too varied
            else:
                scores['sent_variance'] = 4.0
        else:
            scores['sent_variance'] = 5.0
        
        # === Feature 10: Appropriate conciseness for simple queries ===
        # For simple identification/classification queries, concise well-formatted is better
        simple_query_signals = ['identify', 'which', 'name the', 'list', 'what is the']
        is_simple = any(sq in query.lower() for sq in simple_query_signals)
        
        if is_simple and resp_words <= 30 and num_sentences <= 3:
            # Short, focused answers to simple queries are good
            if has_labels or has_lists:
                scores['conciseness'] = 9.0
            else:
                scores['conciseness'] = 7.0
        elif is_simple and resp_words > 100:
            scores['conciseness'] = 4.0
        else:
            scores['conciseness'] = 6.0
        
        # === Combine scores with weights ===
        weights = {
            'length': 0.08,
            'sentences': 0.10,
            'paragraphs': 0.12,
            'formatting': 0.15,
            'repetition': 0.15,
            'coherence': 0.08,
            'noise': 0.18,
            'proportionality': 0.06,
            'sent_variance': 0.04,
            'conciseness': 0.04,
        }
        
        total_score = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in scores:
                total_score += scores[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 3.0
        
        # === Apply global adjustments ===
        
        # Extreme brevity penalty (single word or very short)
        if resp_len < 10:
            final_score = min(final_score, 2.0)
        elif resp_len < 25 and num_sentences <= 1:
            final_score = min(final_score, 3.5)
        
        # Heavy repetition of lines
        if num_lines > 3:
            unique_lines = set(l.strip().lower() for l in lines if l.strip())
            if len(unique_lines) / num_lines < 0.5:
                final_score *= 0.5
        
        # Clamp to [0.5, 10.0]
        final_score = max(0.5, min(10.0, final_score))
        
        return round(final_score, 2)
    
    except Exception:
        # Fallback: return a middle-ground score
        try:
            if response and len(response.strip()) > 20:
                return 4.0
            return 2.0
        except Exception:
            return 3.0