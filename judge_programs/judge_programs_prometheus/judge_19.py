def judging_function(query, response):
    """
    Evaluates completeness and coverage of an LLM response using a question-decomposition
    and topic-coverage approach. This variant focuses on:
    1. Query decomposition - identifying sub-topics/aspects the query asks about
    2. Response coverage mapping - checking how many query aspects are addressed
    3. Structural depth analysis - measuring explanation depth per topic
    4. Information density and specificity scoring
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not query or not response:
            return 0.0
        
        query = str(query).strip()
        response = str(response).strip()
        
        if len(response) < 10:
            return 0.0
        
        # ========== 1. QUERY DECOMPOSITION ==========
        # Extract key aspects/demands from the query
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract significant content words from query (nouns, verbs, adjectives)
        # Remove very common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
            'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these', 'those',
            'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'whose',
            'also', 'like', 'get', 'got', 'make', 'made', 'way', 'need', 'needs',
            'must', 'much', 'many', 'well', 'back', 'even', 'new', 'want', 'first',
            'last', 'long', 'great', 'little', 'right', 'still', 'find', 'give',
            'tell', 'say', 'said', 'know', 'take', 'come', 'think', 'look', 'good',
            'people', 'person', 'individual', 'one', 'two', 'three'
        }
        
        def extract_content_words(text):
            words = re.findall(r'[a-z]+', text.lower())
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        # Extract bigrams for better topic matching
        def extract_bigrams(text):
            words = re.findall(r'[a-z]+', text.lower())
            content_words = [w for w in words if len(w) > 2]
            bigrams = []
            for i in range(len(content_words) - 1):
                bigrams.append(content_words[i] + '_' + content_words[i+1])
            return bigrams
        
        query_words = extract_content_words(query)
        response_words = extract_content_words(response)
        
        query_bigrams = set(extract_bigrams(query))
        response_bigrams = set(extract_bigrams(response))
        
        # ========== 2. TOPIC COVERAGE SCORE ==========
        # What fraction of query's key concepts appear in the response?
        
        # Use unique query words as "aspects" to cover
        query_word_set = set(query_words)
        response_word_set = set(response_words)
        
        if len(query_word_set) > 0:
            word_coverage = len(query_word_set & response_word_set) / len(query_word_set)
        else:
            word_coverage = 0.5
        
        # Bigram coverage
        if len(query_bigrams) > 0:
            bigram_coverage = len(query_bigrams & response_bigrams) / len(query_bigrams)
        else:
            bigram_coverage = 0.5
        
        topic_coverage_score = 0.6 * word_coverage + 0.4 * bigram_coverage
        
        # ========== 3. QUERY DEMAND ANALYSIS ==========
        # Identify what the query is asking for (multiple demands = need more coverage)
        
        # Count distinct question types / demands in query
        demand_patterns = [
            r'\bhow\b', r'\bwhat\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b',
            r'\bexplain\b', r'\bdescribe\b', r'\bprovide\b', r'\badvice\b',
            r'\bhelp\b', r'\bguide\b', r'\bsteps?\b', r'\bmanage\b',
            r'\bhandle\b', r'\bcope\b', r'\bunderstand\b', r'\bcomfort\b',
            r'\bassist\b', r'\bsupport\b', r'\brecommend\b', r'\bsuggest\b',
            r'\bensure\b', r'\baddress\b', r'\brespond\b', r'\badapt\b',
        ]
        
        demand_count = sum(1 for p in demand_patterns if re.search(p, query_lower))
        demand_count = max(demand_count, 1)
        
        # ========== 4. RESPONSE STRUCTURAL DEPTH ==========
        # Measure how deeply the response explores topics
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        num_sentences = len(sentences)
        
        # Measure average sentence length (longer = more detailed, up to a point)
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            avg_sentence_len = 0
        
        # Optimal sentence length around 15-25 words
        sentence_quality = min(avg_sentence_len / 18.0, 1.5) if avg_sentence_len > 0 else 0
        
        # Response length relative to query complexity
        response_words_count = len(response.split())
        length_ratio = response_words_count / max(demand_count * 30, 1)
        length_score = min(length_ratio, 2.0) / 2.0  # cap at 1.0
        
        # ========== 5. SPECIFICITY & CONCRETENESS ==========
        # Check for specific, actionable, or detailed content
        
        # Specific patterns: numbers, examples, named entities, action verbs
        specificity_markers = 0
        
        # Numbers and quantities
        specificity_markers += len(re.findall(r'\d+', response)) * 0.5
        
        # Enumeration / structured content (1. 2. 3. or a) b) c))
        enumeration = len(re.findall(r'(?:^|\n)\s*(?:\d+[.):]|[a-z][.):]|\*|\-)\s', response))
        specificity_markers += enumeration * 1.5
        
        # Causal/explanatory connectors (because, therefore, this means, as a result)
        explanation_words = [
            r'\bbecause\b', r'\btherefore\b', r'\bthis means\b', r'\bas a result\b',
            r'\bdue to\b', r'\bin order to\b', r'\bso that\b', r'\bwhich means\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bimportantly\b', r'\bcrucially\b', r'\bessentially\b',
        ]
        explanation_count = sum(len(re.findall(p, response_lower)) for p in explanation_words)
        specificity_markers += explanation_count * 1.0
        
        # Concrete action words
        action_patterns = [
            r'\btry\b', r'\bstart\b', r'\bfocus\b', r'\bconsider\b',
            r'\bremember\b', r'\bensure\b', r'\bcreate\b', r'\bbuild\b',
            r'\bimplement\b', r'\bapply\b', r'\buse\b', r'\bpractice\b',
            r'\bdevelop\b', r'\bmaintain\b', r'\bestablish\b',
        ]
        action_count = sum(len(re.findall(p, response_lower)) for p in action_patterns)
        specificity_markers += action_count * 0.5
        
        specificity_score = min(specificity_markers / 10.0, 1.0)
        
        # ========== 6. EMPATHY & ACKNOWLEDGMENT (for emotional queries) ==========
        emotional_query_words = [
            'feeling', 'feel', 'frustrated', 'stress', 'sad', 'lonely',
            'heartbroken', 'devastated', 'exhausted', 'struggling', 'difficult',
            'worried', 'anxious', 'upset', 'angry', 'disappointed', 'regret',
            'comfort', 'support', 'listening', 'emotion', 'pain', 'grief',
            'breakup', 'passed away', 'died', 'loss'
        ]
        
        is_emotional_query = sum(1 for w in emotional_query_words if w in query_lower) >= 2
        
        empathy_score = 0.5  # neutral default
        if is_emotional_query:
            empathy_markers = [
                r"\bi understand\b", r"\bi can see\b", r"\bi hear\b",
                r"\bi'm sorry\b", r"\bunderstandable\b", r"\bnatural\b",
                r"\bit's okay\b", r"\bit's fine\b", r"\bperfectly\b",
                r"\bcompletely\b", r"\babsolutely\b", r"\bvalid\b",
                r"\bfeel this way\b", r"\bgenuinely\b", r"\bsincerely\b",
                r"\backnowledge\b", r"\brecognize\b", r"\brespect\b",
            ]
            empathy_count = sum(1 for p in empathy_markers if re.search(p, response_lower))
            empathy_score = min(empathy_count / 3.0, 1.0)
            
            # Check for dismissive language (penalize)
            dismissive_patterns = [
                r'\bjust get over\b', r'\bstop feeling\b', r'\bget yourself together\b',
                r'\byou should be\b', r"\bdon't let it\b", r'\bget rid of\b',
                r'\bjust a\b', r'\bmaybe you\'re just\b',
            ]
            dismissive_count = sum(1 for p in dismissive_patterns if re.search(p, response_lower))
            empathy_score = max(0, empathy_score - dismissive_count * 0.2)
        
        # ========== 7. NEGATIVE INDICATORS ==========
        # Things that suggest incompleteness or poor quality
        
        negative_score = 0.0
        
        # Vagueness / hedging without substance
        vague_phrases = [
            r'\bmaybe\b', r'\bprobably\b', r'\bmight not\b',
            r'\bnot be able\b', r'\bwon\'t be able\b', r'\bcannot\b',
            r'\bmay not\b',
        ]
        vague_count = sum(len(re.findall(p, response_lower)) for p in vague_phrases)
        
        # Negativity/inability focus (saying what can't be done instead of what can)
        inability_phrases = [
            r'\bit might not\b', r'\bit may not\b', r'\bit probably won\'t\b',
            r'\bit can\'t\b', r'\bit won\'t\b',
        ]
        inability_count = sum(len(re.findall(p, response_lower)) for p in inability_phrases)
        
        negative_score = min((vague_count * 0.1 + inability_count * 0.2), 0.5)
        
        # ========== 8. RESPONSE APPROPRIATENESS ==========
        # Does the response match the type of query?
        
        # Check if query asks for directions/specifics and response is appropriately specific or asks for clarification
        asks_clarification = bool(re.search(r'\bcan you\b.*\bmore\b|\bwhat\b.*\brefer|\bspecif', response_lower))
        is_ambiguous_query = bool(re.search(r'ambiguous|no.*context|unclear', query_lower))
        
        appropriateness_bonus = 0.0
        if is_ambiguous_query and asks_clarification:
            appropriateness_bonus = 0.3
        
        # ========== 9. MULTI-ASPECT ADDRESSING ==========
        # Split query into clauses and check coverage
        
        query_clauses = re.split(r'[.!?,;]+', query)
        query_clauses = [c.strip() for c in query_clauses if len(c.strip()) > 15]
        
        clauses_addressed = 0
        for clause in query_clauses:
            clause_words = extract_content_words(clause)
            if clause_words:
                clause_coverage = len(set(clause_words) & response_word_set) / len(set(clause_words))
                if clause_coverage >= 0.3:
                    clauses_addressed += 1
        
        if query_clauses:
            clause_coverage_score = clauses_addressed / len(query_clauses)
        else:
            clause_coverage_score = 0.5
        
        # ========== 10. UNIQUE INFORMATION DENSITY ==========
        # Ratio of unique content words to total words (avoid repetition)
        
        if response_words_count > 0:
            response_content = extract_content_words(response)
            if response_content:
                unique_ratio = len(set(response_content)) / len(response_content)
                # Penalize very repetitive text
                info_density = min(unique_ratio / 0.5, 1.0)
            else:
                info_density = 0.3
        else:
            info_density = 0.0
        
        # ========== FINAL SCORING ==========
        
        # Weight the components
        weights = {
            'topic_coverage': 2.0,
            'clause_coverage': 1.8,
            'length': 1.2,
            'specificity': 1.5,
            'sentence_quality': 0.8,
            'empathy': 1.0 if is_emotional_query else 0.2,
            'info_density': 0.8,
            'appropriateness': 1.0,
            'negative': -1.5,
        }
        
        raw_score = (
            weights['topic_coverage'] * topic_coverage_score +
            weights['clause_coverage'] * clause_coverage_score +
            weights['length'] * length_score +
            weights['specificity'] * specificity_score +
            weights['sentence_quality'] * min(sentence_quality, 1.0) +
            weights['empathy'] * empathy_score +
            weights['info_density'] * info_density +
            weights['appropriateness'] * appropriateness_bonus +
            weights['negative'] * negative_score
        )
        
        total_positive_weight = sum(v for v in weights.values() if v > 0)
        
        # Normalize to 0-1 range
        normalized = raw_score / total_positive_weight
        normalized = max(0.0, min(1.0, normalized))
        
        # Map to 1-5 scale
        final_score = 1.0 + normalized * 4.0
        
        # Apply a slight sigmoid-like transformation for better discrimination
        # Center around 3.0
        centered = final_score - 3.0
        discriminated = 3.0 + 2.0 * (2.0 / (1.0 + math.exp(-1.5 * centered)) - 1.0)
        
        # Clamp to [1, 5]
        discriminated = max(1.0, min(5.0, discriminated))
        
        return round(discriminated, 2)
        
    except Exception as e:
        # Fallback: return middle score
        return 2.5