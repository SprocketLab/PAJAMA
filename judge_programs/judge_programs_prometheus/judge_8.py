def judging_function(query, response):
    """
    Evaluates response relevance using a query-intent matching approach based on:
    - Semantic field coverage (how many distinct query topics are addressed)
    - Question-type alignment (detecting what kind of answer is expected)
    - Discourse coherence signals (connectives, referential phrases)
    - Empathy/engagement markers when appropriate
    - Anti-patterns (off-topic, dismissive, contradictory signals)
    
    This variant focuses on intent decomposition and topic coverage rather than
    simple word overlap or n-gram/cosine approaches.
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
            return 0.5
        
        # Normalize text
        def normalize(text):
            return re.sub(r'[^\w\s]', ' ', text.lower())
        
        def get_words(text):
            return [w for w in normalize(text).split() if len(w) > 1]
        
        # Common stop words to filter
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'don', 'now', 'it', 'its',
            'they', 'them', 'their', 'this', 'that', 'these', 'those', 'he', 'she',
            'him', 'her', 'his', 'we', 'us', 'our', 'you', 'your', 'my', 'me',
            'if', 'or', 'and', 'but', 'because', 'while', 'about', 'up', 'which',
            'what', 'who', 'whom', 'also', 'am', 'i', 'like', 'get', 'got',
        }
        
        def content_words(text):
            return [w for w in get_words(text) if w not in stop_words and len(w) > 2]
        
        query_words = get_words(query)
        resp_words = get_words(response)
        query_content = content_words(query)
        resp_content = content_words(response)
        
        if not query_content or not resp_content:
            return 1.0
        
        # ============================================================
        # 1. INTENT DECOMPOSITION: Extract key "topic chunks" from query
        # ============================================================
        # Extract noun-phrase-like chunks by finding sequences of content words
        # that appear close together in the query
        
        def extract_topic_phrases(text, content_ws):
            """Extract meaningful 2-3 word phrases from text as topic units."""
            words = get_words(text)
            phrases = set()
            content_set = set(content_ws)
            
            for i in range(len(words)):
                if words[i] in content_set:
                    # Single content word
                    phrases.add(words[i])
                    # Bigram with next content word within window of 3
                    for j in range(i+1, min(i+4, len(words))):
                        if words[j] in content_set:
                            phrases.add(words[i] + '_' + words[j])
                            break
            return phrases
        
        query_topics = extract_topic_phrases(query, query_content)
        
        # Check how many query topics are covered in response
        resp_content_set = set(resp_content)
        resp_text_lower = normalize(response)
        
        topics_covered = 0
        total_topics = len(query_topics) if query_topics else 1
        
        for topic in query_topics:
            if '_' in topic:
                # Bigram topic - check if both words appear in response
                parts = topic.split('_')
                if all(p in resp_content_set for p in parts):
                    topics_covered += 1
                elif any(p in resp_content_set for p in parts):
                    topics_covered += 0.4
            else:
                if topic in resp_content_set:
                    topics_covered += 1
        
        topic_coverage_score = topics_covered / total_topics if total_topics > 0 else 0
        
        # ============================================================
        # 2. QUERY INTENT TYPE DETECTION & ALIGNMENT
        # ============================================================
        query_lower = query.lower()
        
        # Detect query intent types
        is_emotional = any(w in query_lower for w in [
            'feeling', 'feel', 'emotion', 'stress', 'frustrat', 'sad', 'happy',
            'heartbroken', 'devastat', 'loneli', 'lonely', 'despair', 'comfort',
            'exhaust', 'anger', 'anxious', 'anxiety', 'depress', 'grief',
            'passed away', 'breakup', 'break up', 'regret', 'tether',
            'down after', 'struggling'
        ])
        
        is_howto = any(w in query_lower for w in [
            'how to', 'how can', 'how would', 'how do', 'guide', 'explain',
            'steps', 'process', 'method', 'recipe', 'cook', 'make',
            'instructions', 'approach', 'strategy', 'way to'
        ])
        
        is_technical = any(w in query_lower for w in [
            'ai model', 'algorithm', 'system', 'design', 'implement',
            'computing', 'quantum', 'program', 'software', 'model',
            'track', 'manage', 'detect'
        ])
        
        is_ambiguous = any(w in query_lower for w in [
            'ambiguous', 'unclear', 'no context', 'no previous context',
            'vague', 'interpret'
        ])
        
        is_style_match = any(w in query_lower for w in [
            'casual', 'informal', 'slang', 'laid-back', 'language style',
            'manner of speaking', 'tone', 'mirror', 'adapt', 'communication'
        ])
        
        resp_lower = response.lower()
        
        # Emotional alignment score
        emotional_alignment = 0.0
        if is_emotional:
            empathy_markers = [
                'understand', 'sorry', 'hear', 'feel', 'okay', 'natural',
                'valid', 'normal', 'tough', 'hard', 'difficult', 'pain',
                'grieve', 'process', 'heal', 'support', 'care', 'listen',
                'completely', 'absolutely', 'genuinely', 'truly',
                'it\'s okay', 'it\'s natural', 'it\'s understandable',
                'perfectly', 'remember', 'breathe'
            ]
            empathy_count = sum(1 for m in empathy_markers if m in resp_lower)
            emotional_alignment = min(empathy_count / 5.0, 1.0)
            
            # Penalize dismissive language in emotional contexts
            dismissive = [
                'just get over', 'move on', 'get yourself together',
                'not a big deal', 'shouldn\'t feel', 'stop feeling',
                'get over it', 'just a', 'maybe you\'re just not'
            ]
            dismissive_count = sum(1 for d in dismissive if d in resp_lower)
            emotional_alignment -= dismissive_count * 0.3
            emotional_alignment = max(0, emotional_alignment)
        
        # How-to alignment: check for structured guidance
        howto_alignment = 0.0
        if is_howto:
            # Check for actionable content
            action_indicators = [
                'first', 'then', 'next', 'step', 'start', 'begin',
                'try', 'consider', 'make sure', 'remember', 'add',
                'use', 'take', 'put', 'place', 'combine', 'mix',
                'explore', 'demonstrate', 'develop', 'create'
            ]
            action_count = sum(1 for a in action_indicators if a in resp_lower)
            howto_alignment = min(action_count / 4.0, 1.0)
            
            # Check for numbered/bulleted lists
            has_list = bool(re.search(r'(\d+[\.\)]\s|[-•]\s)', response))
            if has_list:
                howto_alignment = min(howto_alignment + 0.2, 1.0)
        
        # Technical alignment
        technical_alignment = 0.0
        if is_technical:
            # Check for technical specificity
            specificity_markers = [
                'detect', 'recognize', 'store', 'maintain', 'handle',
                'design', 'implement', 'process', 'algorithm', 'model',
                'system', 'approach', 'method', 'function', 'stack',
                'context', 'track', 'record', 'manage', 'ensure',
                'imagine', 'concept', 'works', 'unlike', 'classical'
            ]
            spec_count = sum(1 for s in specificity_markers if s in resp_lower)
            technical_alignment = min(spec_count / 4.0, 1.0)
            
            # Penalize vagueness in technical context
            vague_markers = [
                'might not', 'probably won\'t', 'may not be able',
                'it might not', 'it probably'
            ]
            vague_count = sum(1 for v in vague_markers if v in resp_lower)
            technical_alignment -= vague_count * 0.25
            technical_alignment = max(0, technical_alignment)
        
        # Ambiguity handling alignment
        ambiguity_alignment = 0.0
        if is_ambiguous:
            # Good response should ask for clarification
            clarification_markers = [
                'clarif', 'more detail', 'more information', 'specify',
                'which', 'where', 'what place', 'can you', 'could you',
                'without', 'further detail', 'destination'
            ]
            clar_count = sum(1 for c in clarification_markers if c in resp_lower)
            ambiguity_alignment = min(clar_count / 2.0, 1.0)
            
            # Penalize making assumptions without clarifying
            if clar_count == 0 and len(resp_words) > 30:
                ambiguity_alignment = -0.3  # Penalty for assuming
        
        # Style matching alignment
        style_alignment = 0.0
        if is_style_match:
            # Check if response discusses adaptation/mirroring
            if 'casual' in query_lower or 'informal' in query_lower or 'slang' in query_lower:
                # Response should use casual language
                casual_markers = [
                    'hey', 'alright', 'gonna', 'wanna', 'cool', 'awesome',
                    'killer', 'whip', 'grab', 'let\'s', 'yo', 'dude',
                    'nifty', 'wild', 'sweet', 'rad', 'hang tight',
                    'down to it', 'get down', 'first things first'
                ]
                casual_count = sum(1 for c in casual_markers if c in resp_lower)
                style_alignment = min(casual_count / 3.0, 1.0)
            elif 'mirror' in query_lower or 'adapt' in query_lower:
                adapt_markers = [
                    'mirror', 'adapt', 'match', 'formal', 'serious',
                    'tone', 'language', 'respect', 'body language',
                    'posture', 'eye contact', 'acknowledge'
                ]
                adapt_count = sum(1 for a in adapt_markers if a in resp_lower)
                style_alignment = min(adapt_count / 3.0, 1.0)
        
        # Combine intent alignment scores
        intent_scores = []
        if is_emotional:
            intent_scores.append(emotional_alignment)
        if is_howto:
            intent_scores.append(howto_alignment)
        if is_technical:
            intent_scores.append(technical_alignment)
        if is_ambiguous:
            intent_scores.append(ambiguity_alignment)
        if is_style_match:
            intent_scores.append(style_alignment)
        
        intent_alignment = max(intent_scores) if intent_scores else 0.5
        
        # ============================================================
        # 3. DISCOURSE COHERENCE: Does response flow logically?
        # ============================================================
        
        # Check for discourse connectives that signal structured reasoning
        connectives = [
            'however', 'moreover', 'furthermore', 'additionally', 'therefore',
            'consequently', 'meanwhile', 'nevertheless', 'instead', 'although',
            'because', 'since', 'while', 'whereas', 'indeed', 'specifically',
            'for instance', 'for example', 'in addition', 'on the other hand',
            'as a result', 'in other words', 'that said', 'keep in mind',
            'remember that', 'it\'s important', 'it is important'
        ]
        connective_count = sum(1 for c in connectives if c in resp_lower)
        coherence_score = min(connective_count / 4.0, 1.0)
        
        # Sentence count and average length as proxy for developed response
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 5]
        num_sentences = len(sentences)
        
        if num_sentences > 0:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / num_sentences
            # Good responses tend to have moderate sentence length (10-25 words)
            if 10 <= avg_sentence_len <= 25:
                coherence_score += 0.2
            elif avg_sentence_len < 5 or avg_sentence_len > 40:
                coherence_score -= 0.1
        
        coherence_score = max(0, min(1, coherence_score))
        
        # ============================================================
        # 4. RESPONSE DEPTH: Measure substantive content density
        # ============================================================
        
        # Unique content words ratio (vocabulary richness)
        unique_content = set(resp_content)
        vocab_richness = len(unique_content) / max(len(resp_content), 1)
        
        # Content density: ratio of content words to total words
        content_density = len(resp_content) / max(len(resp_words), 1)
        
        # Response length adequacy (not too short, not padded)
        resp_len = len(resp_words)
        if resp_len < 20:
            length_score = 0.3
        elif resp_len < 40:
            length_score = 0.6
        elif resp_len < 150:
            length_score = 1.0
        else:
            length_score = 0.85  # Very long might be padded
        
        depth_score = (vocab_richness * 0.3 + content_density * 0.3 + length_score * 0.4)
        depth_score = min(1.0, depth_score)
        
        # ============================================================
        # 5. DIRECT ADDRESS: Does response directly engage with query?
        # ============================================================
        
        # Check if response references key query entities/concepts
        # Use a "query fingerprint" - the most distinctive words
        query_word_freq = Counter(query_content)
        # Words that appear in query but are relatively specific
        distinctive_query_words = [w for w, c in query_word_freq.items() 
                                    if len(w) > 3]
        
        if distinctive_query_words:
            addressed = sum(1 for w in distinctive_query_words if w in resp_content_set)
            direct_address = addressed / len(distinctive_query_words)
        else:
            direct_address = 0.5
        
        # Check for second-person engagement ("you", "your")
        engagement_words = ['you', 'your', "you're", "you've", "you'll"]
        engagement_count = sum(1 for w in get_words(response) if w in engagement_words)
        engagement_ratio = min(engagement_count / max(len(resp_words) * 0.05, 1), 1.0)
        
        # ============================================================
        # 6. ANTI-PATTERN DETECTION
        # ============================================================
        
        penalty = 0.0
        
        # Repetitive phrases (sign of low quality)
        resp_bigrams = []
        rw = get_words(response)
        for i in range(len(rw) - 1):
            resp_bigrams.append(rw[i] + ' ' + rw[i+1])
        if resp_bigrams:
            bigram_counts = Counter(resp_bigrams)
            max_repeat = max(bigram_counts.values()) if bigram_counts else 0
            if max_repeat > 3:
                penalty += 0.15
        
        # Generic/hollow phrases
        hollow_phrases = [
            'keep working on it', "you'll get there", 'just keep',
            'it is what it is', 'that\'s life', 'part of life',
            'nothing wrong with', 'get some rest', 'feel better in the morning',
            'just a job', 'maybe you should buy a new',
            'read the manual', 'you should be able to handle'
        ]
        hollow_count = sum(1 for h in hollow_phrases if h in resp_lower)
        penalty += hollow_count * 0.1
        
        # Contradicting the query's premise
        if is_ambiguous and not any(c in resp_lower for c in ['clarif', 'detail', 'specify', 'which', 'where exactly', 'more information']):
            if len(resp_words) > 50:
                penalty += 0.2  # Long response to ambiguous query without clarifying
        
        # Check for tone mismatch in emotional contexts
        if is_emotional:
            cold_phrases = ['noted that', 'it is noted', 'we suggest that you',
                           'you should be able', 'just a']
            cold_count = sum(1 for c in cold_phrases if c in resp_lower)
            penalty += cold_count * 0.15
        
        penalty = min(penalty, 0.6)
        
        # ============================================================
        # 7. WEIGHTED COMBINATION
        # ============================================================
        
        # Different weights based on query type
        if is_emotional:
            weights = {
                'topic_coverage': 0.15,
                'intent_alignment': 0.35,
                'coherence': 0.10,
                'depth': 0.15,
                'direct_address': 0.15,
                'engagement': 0.10
            }
        elif is_howto:
            weights = {
                'topic_coverage': 0.20,
                'intent_alignment': 0.25,
                'coherence': 0.15,
                'depth': 0.15,
                'direct_address': 0.15,
                'engagement': 0.10
            }
        elif is_technical:
            weights = {
                'topic_coverage': 0.20,
                'intent_alignment': 0.30,
                'coherence': 0.15,
                'depth': 0.20,
                'direct_address': 0.10,
                'engagement': 0.05
            }
        elif is_ambiguous:
            weights = {
                'topic_coverage': 0.10,
                'intent_alignment': 0.40,
                'coherence': 0.10,
                'depth': 0.10,
                'direct_address': 0.15,
                'engagement': 0.15
            }
        else:
            weights = {
                'topic_coverage': 0.25,
                'intent_alignment': 0.20,
                'coherence': 0.15,
                'depth': 0.15,
                'direct_address': 0.15,
                'engagement': 0.10
            }
        
        raw_score = (
            weights['topic_coverage'] * topic_coverage_score +
            weights['intent_alignment'] * intent_alignment +
            weights['coherence'] * coherence_score +
            weights['depth'] * depth_score +
            weights['direct_address'] * direct_address +
            weights['engagement'] * engagement_ratio
        )
        
        # Apply penalty
        raw_score = raw_score - penalty
        
        # Scale to 1-5 range
        final_score = 1.0 + raw_score * 4.0
        final_score = max(1.0, min(5.0, final_score))
        
        return round(final_score, 2)
        
    except Exception as e:
        # Fallback: basic relevance
        try:
            q_words = set(query.lower().split())
            r_words = set(response.lower().split())
            overlap = len(q_words & r_words) / max(len(q_words), 1)
            return 1.0 + overlap * 3.0
        except:
            return 2.5