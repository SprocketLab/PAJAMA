def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a knowledge graph density approach.
    
    This variant builds a "knowledge atom" model: it identifies distinct knowledge claims
    in the response by detecting subject-predicate-object-like structures, proper nouns,
    technical terms, and cross-references between claims. It measures the density of
    interconnected specific claims rather than just counting surface features.
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        words = response.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s.strip()) > 5]
        num_sentences = max(len(sentences), 1)
        
        # === 1. KNOWLEDGE ATOM DETECTION ===
        # Detect distinct factual claims via clause-level analysis
        # A "knowledge atom" is a clause containing at least one specific referent
        
        clauses = re.split(r'[.!?;,\n]', response)
        clauses = [c.strip() for c in clauses if len(c.strip()) > 10]
        
        knowledge_atoms = 0
        for clause in clauses:
            has_specific = False
            # Check for: numbers, proper nouns (capitalized mid-sentence words), 
            # quoted terms, technical markers
            if re.search(r'\d+', clause):
                has_specific = True
            if re.search(r'(?<!\. )\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*', clause):
                has_specific = True
            if re.search(r'["\'\*`]', clause):
                has_specific = True
            # Causal/explanatory connectives suggest structured reasoning
            if re.search(r'\b(because|since|therefore|thus|hence|due to|as a result|caused by|leads to|results in)\b', clause, re.I):
                has_specific = True
            if has_specific:
                knowledge_atoms += 1
        
        atom_density = knowledge_atoms / num_sentences if num_sentences > 0 else 0
        
        # === 2. NAMED ENTITY PROXY DETECTION ===
        # Detect proper nouns, titles, specific references without NLP libraries
        # Look for capitalized multi-word sequences (not sentence starters)
        
        # Remove sentence-initial capitals by looking at mid-sentence caps
        mid_caps = re.findall(r'(?<=[a-z]\s)[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*', response)
        # Also find patterns like "St. Peter", "Dr. Smith"
        titled_entities = re.findall(r'\b(?:St\.|Dr\.|Mr\.|Mrs\.|Prof\.|Mt\.)\s+[A-Z][a-z]+', response)
        # Book/work titles in italics or quotes
        quoted_titles = re.findall(r'[*_][A-Z][^*_]+[*_]', response)
        # Parenthetical references
        paren_refs = re.findall(r'\([^)]{3,60}\)', response)
        
        entity_count = len(mid_caps) + len(titled_entities) * 2 + len(quoted_titles) * 2 + len(paren_refs) * 0.5
        entity_density = entity_count / (word_count / 100) if word_count > 0 else 0
        
        # === 3. NUMERIC PRECISION SCORING ===
        # Different types of numbers carry different specificity weight
        
        # Years (high specificity)
        years = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', response)
        # Percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', response)
        # Decimal numbers (precise measurements)
        decimals = re.findall(r'\d+\.\d+', response)
        # Plain numbers
        plain_nums = re.findall(r'\b\d+\b', response)
        # Monetary values
        money = re.findall(r'[$€£¥]\s*\d+|\d+\s*(?:dollars|euros|pounds|USD|EUR)', response)
        
        numeric_score = (
            len(years) * 1.5 +
            len(percentages) * 2.0 +
            len(decimals) * 1.8 +
            len(money) * 2.0 +
            len(plain_nums) * 0.5
        )
        numeric_density = numeric_score / (word_count / 50) if word_count > 0 else 0
        
        # === 4. SPECIFICITY VOCABULARY RATIO ===
        # Ratio of specific/technical words to generic filler words
        
        lower_response = response.lower()
        
        # Vague/hedge phrases (penalize)
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bvarious factors\b', r'\bthere are (?:many|various|several)\b',
            r'\bgenerally speaking\b', r'\bin general\b', r'\bmore or less\b',
            r'\bkind of\b', r'\bsort of\b', r'\bpretty much\b',
            r'\bi think\b', r'\bi guess\b', r'\bi suppose\b',
            r'\bmaybe\b', r'\bperhaps\b', r'\bprobably\b',
            r'\bnot sure\b', r'\bdon\'t know\b', r'\bhard to say\b',
            r'\bthat said\b', r'\bhaving said that\b',
            r'\byou know\b', r'\bi mean\b', r'\bbasically\b',
            r'\bessentially\b', r'\bsomething like\b',
            r'\bstuff\b', r'\bthings\b(?!\s+(?:like|such))',
            r'\bwhatever\b', r'\betc\.?\b',
        ]
        
        vague_count = sum(len(re.findall(p, lower_response)) for p in vague_patterns)
        vague_rate = vague_count / num_sentences
        
        # Specificity markers (reward)
        specific_patterns = [
            r'\bspecifically\b', r'\bfor example\b', r'\bfor instance\b',
            r'\bsuch as\b', r'\bnamely\b', r'\bin particular\b',
            r'\baccording to\b', r'\bresearch (?:shows|suggests|indicates|found)\b',
            r'\bstud(?:y|ies)\b', r'\bexperiment\b',
            r'\bdata\b', r'\bevidence\b', r'\bsource\b',
            r'\bcited\b', r'\breference\b', r'\bpublished\b',
        ]
        
        specific_count = sum(len(re.findall(p, lower_response)) for p in specific_patterns)
        
        # === 5. STRUCTURAL DEPTH ===
        # Measure how deeply the response engages with the topic
        # via subordinate clauses, conditional reasoning, contrast
        
        depth_markers = [
            r'\bhowever\b', r'\balthough\b', r'\bwhereas\b', r'\bwhile\b',
            r'\bon the other hand\b', r'\bin contrast\b', r'\bnevertheless\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bif\s+\w+\s+\w+,\s+then\b', r'\bwhen\s+\w+\s+\w+,\b',
            r'\bthe (?:trade-off|tradeoff|difference|distinction|key)\b',
        ]
        
        depth_count = sum(len(re.findall(p, lower_response, re.I)) for p in depth_markers)
        
        # === 6. UNIQUE CONCEPT DENSITY ===
        # Count unique multi-word phrases (bigrams/trigrams) as proxy for concept diversity
        
        clean_words = re.findall(r'[a-z]+', lower_response)
        if len(clean_words) >= 2:
            bigrams = set()
            for i in range(len(clean_words) - 1):
                bg = (clean_words[i], clean_words[i+1])
                bigrams.add(bg)
            bigram_diversity = len(bigrams) / max(len(clean_words) - 1, 1)
        else:
            bigram_diversity = 0
        
        # === 7. RESPONSE ENGAGEMENT WITH QUERY ===
        # Check if response addresses specific elements from the query
        
        query_lower = query.lower() if query else ""
        query_words = set(re.findall(r'[a-z]{4,}', query_lower))
        response_words_set = set(re.findall(r'[a-z]{4,}', lower_response))
        
        # Remove very common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will',
                     'would', 'could', 'should', 'about', 'their', 'there', 'which',
                     'when', 'what', 'your', 'more', 'some', 'than', 'them', 'they',
                     'into', 'also', 'just', 'like', 'very', 'much', 'most', 'only',
                     'other', 'does', 'doing', 'being'}
        
        query_content = query_words - stopwords
        response_content = response_words_set - stopwords
        
        if query_content:
            query_coverage = len(query_content & response_content) / len(query_content)
        else:
            query_coverage = 0.5
        
        # === 8. ELABORATION PATTERNS ===
        # Detect explanatory structures that indicate deep engagement
        
        elaboration_patterns = [
            r'\bthis (?:means|implies|suggests|indicates)\b',
            r'\bin other words\b', r'\bthat is(?:,| to say)\b',
            r'\bthe reason (?:is|being|for)\b',
            r'\bwhat this means\b', r'\bto put it\b',
            r'\bto be (?:more )?specific\b',
        ]
        elaboration_count = sum(len(re.findall(p, lower_response, re.I)) for p in elaboration_patterns)
        
        # === 9. CODE/TECHNICAL CONTENT DETECTION ===
        code_blocks = len(re.findall(r'```', response))
        inline_code = len(re.findall(r'`[^`]+`', response))
        technical_symbols = len(re.findall(r'[{}()\[\]=<>]', response))
        has_technical = (code_blocks > 0 or inline_code > 0 or technical_symbols > 10)
        
        # === 10. DIALOGUE/NARRATIVE RICHNESS ===
        # For creative/narrative responses, detect scene-setting, action, dialogue
        dialogue_markers = len(re.findall(r'[*][^*]+[*]', response))  # asterisk actions
        direct_speech = len(re.findall(r'["\'].*?["\']', response))
        
        # === COMPOSITE SCORING ===
        
        # Length component (logarithmic, diminishing returns)
        length_score = min(math.log(word_count + 1, 2) / math.log(500, 2), 1.0) * 15
        
        # Knowledge atom density (0-20)
        atom_score = min(atom_density * 10, 20)
        
        # Entity density (0-15)
        entity_score = min(entity_density * 3, 15)
        
        # Numeric precision (0-12)
        num_score = min(numeric_density * 4, 12)
        
        # Specificity vs vagueness (can be negative)
        specificity_balance = (specific_count * 2.0 - vague_count * 1.5)
        spec_score = max(min(specificity_balance * 2, 10), -8)
        
        # Structural depth (0-8)
        depth_score = min(depth_count * 1.5, 8)
        
        # Bigram diversity bonus (0-5)
        diversity_score = bigram_diversity * 5
        
        # Query engagement (0-5)
        engagement_score = query_coverage * 5
        
        # Elaboration bonus (0-5)
        elab_score = min(elaboration_count * 2, 5)
        
        # Technical content bonus (0-5)
        tech_score = 5 if has_technical else 0
        
        # Dialogue/narrative richness (0-5)
        narrative_score = min((dialogue_markers + direct_speech) * 1.0, 5)
        
        # Combine all components
        total = (
            length_score +          # 0-15
            atom_score +            # 0-20
            entity_score +          # 0-15
            num_score +             # 0-12
            spec_score +            # -8 to 10
            depth_score +           # 0-8
            diversity_score +       # 0-5
            engagement_score +      # 0-5
            elab_score +            # 0-5
            tech_score +            # 0-5
            narrative_score         # 0-5
        )
        
        # Normalize to 0-100 range
        # Theoretical max ~105, theoretical min ~ -8
        score = max(0, min(100, total))
        
        return round(score, 2)
        
    except Exception:
        try:
            return max(0, min(50, len(str(response)) / 20))
        except Exception:
            return 0.0