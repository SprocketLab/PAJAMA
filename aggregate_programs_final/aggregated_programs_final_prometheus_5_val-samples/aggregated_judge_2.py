def judging_function(query, response):
    try:
        import re, math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not isinstance(query, str):
            query = ""

        resp = response.strip()
        if len(resp) < 8:
            return 1.5
        ql = query.lower()
        rl = resp.lower()

        words = re.findall(r"[a-zA-Z']+", rl)
        n_words = len(words)
        if n_words < 3:
            return 1.5

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)

        # ---------- Detect query type ----------
        emotion_words = {'stress','stressed','anxious','anxiety','overwhelm','overwhelmed',
                         'frustrat','frustrated','sad','lonely','loneliness','heartbroken',
                         'devastated','crying','sleepless','struggl','exhausted','upset',
                         'discouraged','disappointment','disappointed','feeling','emotional',
                         'depressed','worried','angry','grief','grieving','passed','died',
                         'breakup','heartbreak','support','encourage','encouragement','comfort'}

        emotional_score_q = sum(1 for w in re.findall(r"[a-z']+", ql) if w in emotion_words)
        is_emotional = emotional_score_q >= 1

        tone_match_q = bool(re.search(
            r'\b(adapt|match|mirror|tailor)\b.*\b(tone|style|language|register|mood)\b', ql)) or \
                       bool(re.search(r'\b(formal|informal|casual|slang|laid[- ]back)\b', ql))

        clarification_q = bool(re.search(
            r'\b(no (prior |)information|don\'?t (have|know)|ambiguous|unclear|'
            r'unspecified|without (any )?(details|context)|without providing)\b', ql))

        # ---------- Empathy / validation markers ----------
        empathy_pats = [
            r"\bi (can |)(see|hear|understand|sense|imagine|know|get) (that|how|what|you|your)\b",
            r"\bi'?m (genuinely |truly |really |so )?sorry\b",
            r"\bsorry to hear\b",
            r"\bthat'?s (completely |totally |absolutely |perfectly )?(understandable|valid|okay|ok|fine|normal|natural)\b",
            r"\bit'?s (completely |totally |absolutely |perfectly )?(understandable|okay|ok|fine|natural|normal|valid)\b",
            r"\byou'?re not alone\b",
            r"\bnot alone in (this|feeling)\b",
            r"\bvalid feelings?\b",
            r"\byour feelings? (are|matter|is)\b",
            r"\bi'?m here (for you|to listen|with you)\b",
            r"\bcompletely natural\b", r"\btotally natural\b",
            r"\bgive yourself (permission|time|space|grace)\b",
            r"\bit'?s okay to (feel|cry|not|be)\b",
            r"\btake (a moment|your time)\b",
            r"\blet yourself\b",
            r"\bthat sounds (really |truly |incredibly )?(hard|tough|difficult|challenging|painful)\b",
            r"\bappreciate you (sharing|opening up)\b",
            r"\bthank you for sharing\b",
            r"\bwhat you'?re (going through|experiencing|feeling)\b",
        ]
        empathy_count = sum(1 for p in empathy_pats if re.search(p, rl))

        # ---------- Dismissive / judgmental markers ----------
        dismissive_pats = [
            r"\bjust\s+(get over|move on|deal with|forget|stop|ignore|do|try|keep|remember)\b",
            r"\byou'?re just not (cut out|good)\b",
            r"\bmaybe you'?re just\b",
            r"\bstop (being|feeling|complaining|worrying)\b",
            r"\bget yourself together\b",
            r"\bsuck it up\b",
            r"\bcut out for\b",
            r"\bnot a big deal\b", r"\bno big deal\b",
            r"\bthat'?s a bummer\b",
            r"\byou should be able to\b",
            r"\bbe grateful\b",
            r"\bothers have it (worse|harder)\b",
            r"\beveryone (has|goes through|deals with)\b.*\b(struggles?|problems?|stress)\b",
            r"\bwe all have\b",
            r"\blower your expectations\b",
            r"\bnot cut out\b",
            r"\bpush through\b",
            r"\baccept your limitations\b",
        ]
        dismissive_count = sum(1 for p in dismissive_pats if re.search(p, rl))

        # ---------- Constructive advice signals ----------
        constructive = [
            r"\b(try|consider|might|could|may want to|one (way|approach)|here are some|"
            r"some (ways|things|strategies|ideas)|break (it|things) down|prioritize|"
            r"reach out|talk to|seek (help|support|professional))\b",
            r"\b(deep (breath|breathing)|relaxation|mindfulness|meditation|exercise|"
            r"journal|write|sleep|rest|self[- ]care)\b",
        ]
        constructive_count = sum(len(re.findall(p, rl)) for p in constructive)

        # ---------- Question back to user (engagement) ----------
        questions = resp.count('?')

        # ---------- Tone-match detection ----------
        casual_markers = len(re.findall(
            r"\b(hey|yo|mate|gonna|wanna|gotta|ain'?t|killer|whip up|cool|"
            r"awesome|alright|ya|y'?all|chill)\b", rl))
        formal_markers = len(re.findall(
            r"\b(cordially|esteemed|elucidation|reminiscent|herein|whereby|"
            r"furthermore|moreover|consequently|aforementioned|pursuant)\b", rl))

        query_casual = bool(re.search(
            r"\b(casual|laid[- ]back|slang|informal|chill|hey|yo)\b", ql))
        query_formal = bool(re.search(
            r"\b(formal|professional|academic|scholarly|cordial)\b", ql))

        # ---------- Clarification-asking detection ----------
        asks_for_clarification = bool(re.search(
            r"\b(could you (provide|share|tell|specify|elaborate|clarify)|"
            r"can you (tell|share|give|describe)|"
            r"what (kind|type|specific)|"
            r"more (details|information|context)|"
            r"can you (provide|give))\b", rl))

        # ---------- Off-topic detection: response doesn't share content words with query ----------
        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on',
                'with','at','by','from','it','this','that','and','or','but','if','i',
                'you','we','they','he','she','have','has','had','will','would','can',
                'could','should','do','does','did','your','my','our','their','what',
                'how','when','why','where','about'}
        q_content = {w for w in re.findall(r"[a-z']+", ql) if w not in STOP and len(w) > 3}
        r_content = {w for w in re.findall(r"[a-z']+", rl) if w not in STOP and len(w) > 3}
        if q_content:
            overlap_ratio = len(q_content & r_content) / len(q_content)
        else:
            overlap_ratio = 0.5

        # ---------- Compose score ----------
        score = 3.0  # midpoint base

        # Universal signals
        score += min(0.7, overlap_ratio * 1.0)
        if 25 <= n_words <= 280:
            score += 0.3
        elif n_words < 15:
            score -= 0.5
        elif n_words > 400:
            score -= 0.2

        # Emotional query handling
        if is_emotional:
            score += min(1.5, empathy_count * 0.45)
            score -= dismissive_count * 0.8
            score += min(0.6, constructive_count * 0.08)
            # validation without solution-only mode
            if empathy_count == 0 and dismissive_count == 0:
                score -= 0.3  # clinical-only response in emotional context
            if empathy_count >= 2 and constructive_count >= 1:
                score += 0.4  # the gold standard: validation + actionable
        else:
            # In non-emotional contexts, mild empathy still slightly positive
            score += min(0.3, empathy_count * 0.1)
            score -= min(0.3, dismissive_count * 0.15)

        # Tone-match handling
        if tone_match_q:
            # response that demonstrates tone awareness wins
            tone_aware = len(re.findall(
                r"\b(adjust|adapt|match|mirror|tailor|tune|switch|modify)\b.*"
                r"\b(tone|style|language|response|word|register)\b", rl))
            score += min(0.6, tone_aware * 0.3)
            # description-of-failure penalty (response describes NOT adapting)
            fail_desc = bool(re.search(
                r"\b(does not|did not|doesn'?t|didn'?t|fails? to|cannot|can'?t)\b.{0,40}"
                r"\b(adapt|match|mirror|adjust|tailor|respond)\b", rl))
            if fail_desc:
                score -= 0.6

        if query_casual:
            score += min(0.6, casual_markers * 0.15)
            score -= min(0.5, formal_markers * 0.2)
        elif query_formal:
            score += min(0.5, formal_markers * 0.15)

        # Clarification-asking
        if clarification_q:
            if asks_for_clarification:
                score += 0.9
            # Penalize obvious off-topic ramblings
            if overlap_ratio < 0.15 and n_words > 40:
                score -= 1.0

        # Question engagement (mild bonus)
        if 1 <= questions <= 3:
            score += 0.15

        # Generic platitude penalty (catches case 9-style responses)
        platitudes = len(re.findall(
            r"\b(keep working|you'?ll figure|don'?t give up|stay positive|"
            r"believe in yourself|good luck|you got this|you can do it|"
            r"you'?ll be fine|hope (you|things) (find|work|get) (out|better))\b", rl))
        if platitudes >= 2 and constructive_count == 0:
            score -= 0.5

        # Sentence-level filler
        if n_words > 0:
            filler_rate = len(re.findall(
                r"\b(basically|literally|honestly|just|stuff|things)\b", rl)) / n_words
            score -= min(0.4, filler_rate * 5)

        score = max(1.0, min(5.0, score))
        return round(score, 3)

    except Exception:
        return 3.0
