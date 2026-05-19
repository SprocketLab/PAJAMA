def judging_function(query, response):
    """
    Evaluates evidence density and specificity using a novel approach based on:
    1. Named entity density (capitalized multi-word phrases, proper nouns)
    2. Numeric/quantitative content detection
    3. Specificity markers (action verbs, technical terms, domain-specific language)
    4. Anti-repetition penalty (unique information ratio)
    5. Structural informativeness (information per sentence)
    6. Rare word ratio (words not in top-500 common English words as proxy for specificity)
    """
    try:
        import re
        import math
        from collections import Counter
        
        if not response or not isinstance(response, str):
            return 0.0
        
        response = response.strip()
        if len(response) < 5:
            return 0.5
        
        # Top ~300 most common/generic English words (stopwords + common filler)
        COMMON_WORDS = set("""
        the a an is are was were be been being have has had do does did will would shall
        should can could may might must need dare ought to of in for on with at by from
        as into through during before after above below between out off over under again
        further then once here there when where why how all each every both few more most
        other some such no nor not only own same so than too very just don t s d ll ve re
        m o ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
        shouldn wasn weren won wouldn that this these those i me my myself we our ours
        ourselves you your yours yourself yourselves he him his himself she her hers
        herself it its itself they them their theirs themselves what which who whom
        and but if or because while although since until unless also still already yet
        even though however therefore moreover furthermore additionally meanwhile
        about after again against between into through during before after above below
        up down out off over under around along across behind beside besides beyond
        get got gets getting go goes going gone went come came comes coming make made
        makes making take took takes taking give gave gives giving say said says saying
        know knew knows knowing think thought thinks thinking see saw sees seeing want
        wanted wants wanting use used uses using find found finds finding tell told tells
        telling ask asked asks asking work worked works working seem seemed seems seeming
        feel felt feels feeling try tried tries trying leave left leaves leaving call
        called calls calling keep kept keeps keeping let lets letting begin began begins
        beginning show showed shows showing hear heard hears hearing play played plays
        playing run ran runs running move moved moves moving like liked likes liking
        live lived lives living believe believed believes believing hold held holds
        holding bring brought brings bringing happen happened happens happening write
        wrote writes writing provide provided provides providing sit sat sits sitting
        stand stood stands standing lose lost loses losing pay paid pays paying meet
        met meets meeting include included includes including continue continued
        set learn change lead understand watch follow stop create speak read allow
        add spend grow open walk win offer remember love consider appear buy wait
        serve die send expect build stay fall cut reach kill remain suggest raise
        pass sell require report decide pull develop many much very really well just
        also still already even now back then too quite rather almost enough never
        always often sometimes usually probably certainly perhaps maybe actually
        simply clearly nearly likely finally basically generally especially recently
        particularly currently certainly indeed therefore thus hence consequently
        thing things way ways people person man woman time times year years day days
        world life hand part place case week point company number group problem fact
        it them something anything everything nothing someone anyone everyone
        good bad big small great little long short high low old new first last
        different important large young right next early possible able free
        several whole special sure real full better best certain clear
        various possible likely able sure important different
        """.split())
        
        words = re.findall(r'\b[a-zA-Z]+\b', response)
        total_words = len(words)
        
        if total_words < 3:
            return 1.0
        
        # === Feature 1: Numeric/Quantitative Content ===
        # Count numbers, percentages, dates, measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', response)
        num_score = min(len(numbers) * 3.0, 15.0)
        
        # Detect measurement units and quantifiers
        measurement_patterns = [
            r'\b\d+\s*(?:km|mi|miles|meters|feet|inches|cm|mm|kg|lbs|pounds|grams|mg|'
            r'hours|minutes|seconds|days|weeks|months|years|dollars|euros|cents|'
            r'percent|%|GB|MB|KB|TB|Hz|MHz|GHz|watts|volts|amps|calories|degrees|'
            r'mph|kph|rpm|psi|dB|px|em|rem)\b',
        ]
        measurement_count = sum(len(re.findall(p, response, re.IGNORECASE)) for p in measurement_patterns)
        measurement_score = min(measurement_count * 4.0, 12.0)
        
        # === Feature 2: Named Entity / Proper Noun Density ===
        # Detect capitalized words not at sentence start
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        proper_noun_count = 0
        for sent in sentences:
            sent_words = sent.split()
            if len(sent_words) > 1:
                for w in sent_words[1:]:  # skip first word (sentence start)
                    clean_w = re.sub(r'[^a-zA-Z]', '', w)
                    if clean_w and clean_w[0].isupper() and len(clean_w) > 1 and clean_w.lower() not in COMMON_WORDS:
                        proper_noun_count += 1
        
        proper_noun_density = proper_noun_count / max(total_words, 1) * 100
        proper_noun_score = min(proper_noun_density * 3.0, 12.0)
        
        # === Feature 3: Rare/Specific Word Ratio ===
        lower_words = [w.lower() for w in words if len(w) > 2]
        if lower_words:
            rare_words = [w for w in lower_words if w not in COMMON_WORDS]
            rare_ratio = len(rare_words) / len(lower_words)
            rare_score = rare_ratio * 20.0  # 0-20 scale
        else:
            rare_score = 0.0
        
        # === Feature 4: Unique Information Density (anti-repetition) ===
        if total_words > 0:
            unique_words = set(w.lower() for w in words)
            unique_ratio = len(unique_words) / total_words
            # Penalize heavy repetition
            if unique_ratio < 0.3:
                uniqueness_score = 0.0
            else:
                uniqueness_score = min(unique_ratio * 10.0, 8.0)
        else:
            uniqueness_score = 0.0
        
        # N-gram repetition penalty (bigrams and trigrams)
        lower_word_list = [w.lower() for w in words]
        if len(lower_word_list) >= 3:
            trigrams = [tuple(lower_word_list[i:i+3]) for i in range(len(lower_word_list)-2)]
            trigram_counts = Counter(trigrams)
            if trigrams:
                repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
                repetition_ratio = repeated_trigrams / len(trigram_counts)
                repetition_penalty = min(repetition_ratio * 15.0, 15.0)
            else:
                repetition_penalty = 0.0
        else:
            repetition_penalty = 0.0
        
        # === Feature 5: Specificity Markers ===
        # Words/phrases that signal concrete, specific information
        specificity_patterns = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\bspecifically\b', r'\bin particular\b', r'\bnamely\b',
            r'\bincluding\b', r'\be\.g\.\b', r'\bi\.e\.\b',
            r'\baccording to\b', r'\bresearch shows\b', r'\bstudies\b',
            r'\bdata\b', r'\bevidence\b', r'\bresults\b',
        ]
        specificity_count = sum(len(re.findall(p, response, re.IGNORECASE)) for p in specificity_patterns)
        specificity_score = min(specificity_count * 2.5, 10.0)
        
        # === Feature 6: Vagueness Penalty ===
        vague_patterns = [
            r'\bmany people\b', r'\bsome people\b', r'\bit depends\b',
            r'\bthere are (?:many|various|several|different)\b',
            r'\bin many ways\b', r'\ba lot of\b', r'\bvarious factors\b',
            r'\band so on\b', r'\band more\b', r'\betc\.?\b',
            r'\bthings like that\b', r'\band stuff\b',
            r'\bgenerally speaking\b', r'\bin general\b',
            r'\bkind of\b', r'\bsort of\b', r'\bmore or less\b',
            r'\bcan be\b', r'\bmay be\b', r'\bmight be\b',
        ]
        vague_count = sum(len(re.findall(p, response, re.IGNORECASE)) for p in vague_patterns)
        vague_penalty = min(vague_count * 1.5, 10.0)
        
        # === Feature 7: Information per Sentence ===
        # Longer, more substantive sentences (but not run-on) carry more info
        num_sentences = max(len(sentences), 1)
        avg_words_per_sentence = total_words / num_sentences
        
        # Sweet spot: 10-25 words per sentence
        if avg_words_per_sentence < 5:
            sentence_info_score = 2.0
        elif avg_words_per_sentence <= 25:
            sentence_info_score = min(avg_words_per_sentence * 0.4, 8.0)
        else:
            sentence_info_score = max(8.0 - (avg_words_per_sentence - 25) * 0.2, 3.0)
        
        # === Feature 8: Actionable/Concrete Verb Density ===
        action_verbs = set([
            'create', 'build', 'design', 'implement', 'develop', 'configure',
            'install', 'deploy', 'execute', 'calculate', 'measure', 'analyze',
            'compare', 'evaluate', 'select', 'choose', 'identify', 'define',
            'specify', 'categorize', 'classify', 'organize', 'prioritize',
            'track', 'monitor', 'record', 'document', 'submit', 'process',
            'send', 'receive', 'connect', 'integrate', 'combine', 'separate',
            'extract', 'transform', 'convert', 'generate', 'produce', 'display',
            'render', 'format', 'encode', 'decode', 'encrypt', 'compress',
            'crop', 'resize', 'reduce', 'increase', 'adjust', 'modify',
            'customize', 'personalize', 'optimize', 'enhance', 'improve',
        ])
        action_count = sum(1 for w in lower_word_list if w in action_verbs)
        action_density = action_count / max(total_words, 1) * 100
        action_score = min(action_density * 3.0, 8.0)
        
        # === Feature 9: Content Length Bonus (diminishing returns) ===
        # Longer responses tend to have more evidence, but with diminishing returns
        length_score = min(math.log(max(total_words, 1) + 1) * 1.5, 8.0)
        
        # === Feature 10: Parenthetical/Elaboration Detection ===
        # Parentheses, colons, dashes often introduce specific details
        elaboration_markers = len(re.findall(r'[(:—–\-]', response))
        # Commas in lists suggest enumeration of specifics
        comma_count = response.count(',')
        list_indicators = len(re.findall(r'\b(?:\d+[.)]\s|•|→|►)', response))
        
        elaboration_score = min((elaboration_markers * 0.3 + comma_count * 0.2 + list_indicators * 1.5), 8.0)
        
        # === Feature 11: Quoted/Referenced Content ===
        quotes = len(re.findall(r'"[^"]{3,}"', response))
        quote_score = min(quotes * 2.0, 6.0)
        
        # === Combine all features ===
        total_score = (
            num_score +           # 0-15: numbers
            measurement_score +   # 0-12: measurements
            proper_noun_score +   # 0-12: proper nouns
            rare_score +          # 0-20: rare/specific vocabulary
            uniqueness_score +    # 0-8: unique words
            specificity_score +   # 0-10: specificity markers
            sentence_info_score + # 2-8: info per sentence
            action_score +        # 0-8: action verbs
            length_score +        # 0-8: length bonus
            elaboration_score +   # 0-8: elaboration markers
            quote_score +         # 0-6: quotes
            - vague_penalty -     # 0-10: vagueness penalty
            - repetition_penalty  # 0-15: repetition penalty
        )
        
        # Normalize to 0-100 range
        # Max theoretical: ~115, Min: ~-25
        normalized = max(0.0, min(100.0, (total_score + 5) * 0.85))
        
        return round(normalized, 2)
        
    except Exception:
        return 5.0