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
            return 2.0
        rl = resp.lower()
        ql = query.lower()

        words = re.findall(r"[a-zA-Z']+", rl)
        n_words = len(words)
        if n_words < 3:
            return 2.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if len(s.strip()) > 2]
        n_sents = max(len(sentences), 1)

        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on',
                'with','at','by','from','it','this','that','and','or','but','if','i',
                'you','we','they','have','has','will','would','can','could','should',
                'do','does','did','your','my','our','their','what','how','when','why',
                'where','about','as','also','am','being','been','him','her','them','us'}

        # ---------- 1. Negative-scenario detection ----------
        # Response describes a system/agent that FAILS at the task
        negative_scenario_pats = [
            r"\bdoes not (ask|inquire|consider|address|adapt|match|mirror|understand)\b",
            r"\bdoesn'?t (ask|inquire|consider|address|adapt|match|mirror|understand|seek)\b",
            r"\bfails? to (ask|inquire|consider|address|adapt|match|mirror|understand|recognize)\b",
            r"\bdid not (ask|inquire|consider|address|adapt|match|mirror|seek)\b",
            r"\bdidn'?t (ask|inquire|consider|address|adapt|match|mirror|seek)\b",
            r"\bwithout (seeking|asking|inquiring|considering) (clarification|details|context)\b",
            r"\blacking (empathy|understanding|consideration|adaptation)\b",
            r"\bremain(s|ed) indifferent\b",
            r"\bneglect(s|ing|ed) (the|to|its)\b",
            r"\bnot adapt(ing|ed)?\b",
            r"\bcontinues? (in the same|to use) (?:light[- ]hearted|casual|formal|informal)\b.{0,40}\bdespite\b",
            r"\bcrack(s|ing) jokes\b.{0,30}\bserious\b",
            r"\bdisregard(s|ed|ing)?\b",
            r"\binstead of (asking|seeking|considering|clarifying|adapting)\b",
            r"\bproceeds? (directly|without)\b",
            r"\bperson(?:'s)? speech remain(?:s|ed)\b",
        ]
        neg_scenario_count = sum(1 for p in negative_scenario_pats if re.search(p, rl))

        # Positive-scenario opposites: "did ask", "adapts", "shows empathy"
        positive_scenario_pats = [
            r"\b(politely|kindly) (interject|ask|inquire)",
            r"\bask(s|ed|ing)? (for )?clarif",
            r"\b(shows?|demonstrat\w+|exhibit\w+) empathy\b",
            r"\b(adapt|adjust|tailor|mirror|match)(s|ed|ing)? (its|their|the) (tone|style|language|response)\b",
            r"\backnowledg\w+ the (customer|user|speaker|person)'?s (mood|frustration|emotion|feelings?)\b",
            r"\b(offer|provide|present)(s|ed|ing)? (a |an |)(possible|specific|relevant) (interpretation|suggestion|option)\b",
            r"\b(empathetic|understanding) tone\b",
            r"\bsoothe\b", r"\binterject\b", r"\bdeliberate\b",
        ]
        pos_scenario_count = sum(1 for p in positive_scenario_pats if re.search(p, rl))

        # ---------- 2. Off-topic detection ----------
        q_content = {w for w in re.findall(r"[a-z']+", ql) if w not in STOP and len(w) > 3}
        r_content_set = {w for w in words if w not in STOP and len(w) > 3}
        if q_content:
            overlap = len(q_content & r_content_set) / len(q_content)
        else:
            overlap = 0.5

        # Strong topic signals (proper nouns / specific terms in query)
        q_specific_terms = re.findall(r"\b[a-z]{5,}\b", ql)
        q_specific_terms = [w for w in q_specific_terms if w not in STOP]
        if q_specific_terms:
            specific_in_resp = sum(1 for w in q_specific_terms if w in rl)
            specific_overlap = specific_in_resp / len(q_specific_terms)
        else:
            specific_overlap = overlap

        # ---------- 3. Clarification-asking detection ----------
        ambiguity_signals = bool(re.search(
            r"\b(no (prior )?information|don'?t (have|know)|ambiguous|unclear|"
            r"unspecified|without (providing|any|specific)|uncertain inquiry|"
            r"insufficient (details|information|context))\b", ql))

        asks_clarify = (bool(re.search(
            r"\b(could you (provide|share|tell|specify|elaborate|clarify|give)|"
            r"can you (tell|share|give|describe|specify|provide)|"
            r"what (kind|type|specific|exact)|"
            r"more (details|information|context|specifics)|"
            r"could you describe|would you mind sharing|"
            r"please (provide|share|specify|elaborate)|"
            r"need (more|further|additional))\b", rl)) or
            bool(re.search(r"\?", resp)) and bool(re.search(
                r"\b(provide|share|tell|specify|clarify|describe|details)\b", rl)))

        # ---------- 4. Direct-answer-with-substance ----------
        # Response that actually addresses the question with content
        provides_substance = (n_words >= 35 and overlap >= 0.2)

        # ---------- 5. Action / concrete advice ----------
        action_re = re.compile(
            r'\b(try|consider|start|begin|use|apply|add|remove|check|ensure|'
            r'first|second|third|next|then|finally|step|click|select|enter|'
            r'pour|stir|mix|preheat|grease|combine|whisk|knead|bake)\b',
            re.IGNORECASE)
        actions = len(action_re.findall(resp))

        # Numbered steps
        numbered = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', resp))

        # ---------- 6. Generic non-answer detection ----------
        nonanswer_pats = [
            r"\byou'?ll figure (it|this) out\b",
            r"\bjust keep (trying|working|going|at it)\b",
            r"\bgood luck (with )?(your|the )?\w*\s*$",
            r"\bdon'?t (worry|give up)\b",
            r"\byou'?ll be fine\b",
            r"\byou got this\b",
            r"\bbelieve in yourself\b",
            r"\bdouble[- ]check your work\b",
            r"\bread(?:ing)? documentation\b",
            r"\bsearch(?:ing)? (online|for similar issues)\b",
            r"\bdon'?t have (?:the |any |specific )?(?:information|specifics|recommendations?)\b",
        ]
        nonanswer = sum(1 for p in nonanswer_pats if re.search(p, rl))

        # ---------- 7. Compose ----------
        score = 3.0

        # Topic adherence baseline
        score += overlap * 1.2
        score += min(0.6, specific_overlap * 1.0)

        # Scenario polarity
        # Only weight when the query is asking about a system/agent's behavior
        scenario_query = bool(re.search(
            r"\b(model|chatbot|AI|system|agent|assistant|waiter|representative|teacher)\b", ql))
        if scenario_query:
            # Check explicit failure framing in query
            query_describes_problem = bool(re.search(
                r"\b(struggl|fail|unable|difficult|trouble|challenge|issue|problem)\b", ql))
            if pos_scenario_count > neg_scenario_count:
                score += min(0.8, (pos_scenario_count - neg_scenario_count) * 0.3)
            elif neg_scenario_count > pos_scenario_count and not query_describes_problem:
                # Response is describing failure but query isn't asking for failure analysis
                score -= min(1.0, (neg_scenario_count - pos_scenario_count) * 0.4)
            elif neg_scenario_count >= 2:
                # Always mild penalty for heavily negative-scenario response
                score -= 0.4

        # Clarification logic
        if ambiguity_signals:
            if asks_clarify:
                score += 1.0
            elif overlap < 0.15 and n_words > 30:
                # off-topic ramble when clarification was needed
                score -= 1.2

        # General off-topic
        if overlap < 0.08 and n_words > 60:
            score -= 1.0

        # Direct-answer-with-substance bonus
        if provides_substance:
            score += 0.4
        if numbered >= 2:
            score += 0.3
        if actions >= 3:
            score += 0.3

        # Non-answer penalty
        score -= min(1.2, nonanswer * 0.4)

        # Length sanity
        if n_words < 15:
            score -= 0.4
        elif n_words > 500:
            score -= 0.1

        # ---------- 8. Empathy / dismissiveness shading (emotional context) ----------
        emo_q = bool(re.search(
            r"\b(stress|anxious|overwhelm|frustrat|sad|lonely|heartbroken|"
            r"crying|sleepless|struggl|exhausted|upset|discouraged|"
            r"disappointment|feeling)\b", ql))
        if emo_q:
            empathy = len(re.findall(
                r"\b(i understand|i hear|i'?m sorry|it'?s okay|completely understandable|"
                r"natural to|not alone|your feelings|i can see|valid)\b", rl))
            score += min(0.7, empathy * 0.3)
            dismissive = len(re.findall(
                r"\b(get over it|just move on|just deal|you'?re just not|"
                r"suck it up|cut out for|lower your expectations|stop (being|feeling)|"
                r"maybe you'?re just|be grateful|push through)\b", rl))
            score -= dismissive * 0.7

        # Audience mismatch shading (lighter than Program 4)
        if re.search(r"\b(simple|simpler|layman|beginner|7th grade|8th grade|"
                     r"new to|first time|language barrier|foreigner|kid)\b", ql):
            sci = len(re.findall(
                r"\b(biochemical|autotrophic|chlorophyll|chloroplast|photolysis|"
                r"redox|quantum|inertia|momentum|asymptotic|reminiscent|decennium|"
                r"elucidation|cordially|esteemed)\b", rl))
            if sci >= 2:
                score -= 0.5
            # reward simple framings
            if re.search(r"\b(imagine|think of|like a|just like|picture|as if)\b", rl):
                score += 0.4

        # Clamp
        score = max(1.0, min(5.0, score))
        return round(score, 3)

    except Exception:
        return 3.0
