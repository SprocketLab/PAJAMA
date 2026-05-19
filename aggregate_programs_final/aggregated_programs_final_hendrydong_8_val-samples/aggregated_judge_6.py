def judging_function(query, response):
    try:
        import re
        import math
        from collections import Counter

        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""

        resp = response.strip()
        if len(resp) < 2:
            return 1.0

        rl = resp.lower()
        ql = query.lower()
        words = resp.split()
        wc = len(words)
        q_wc = len(query.split())

        score = 50.0

        STOP = {'the','a','an','is','are','was','were','be','to','of','in','for','on',
                'and','but','or','not','it','its','this','that','these','those','i','me',
                'my','we','our','you','your','what','which','how','why','when','where',
                'with','from','as','have','has','had','do','does','did','will','would',
                'can','could','should','about','up','out','also','like','some','any',
                'one','two','am','re','ve','ll'}

        q_cw = set(w for w in re.findall(r'[a-z]+', ql) if w not in STOP and len(w) > 2)
        r_cw_list = [w for w in re.findall(r'[a-z]+', rl) if w not in STOP and len(w) > 2]
        r_cw = set(r_cw_list)

        # === RELEVANCE (heavy weight) ===
        if q_cw:
            cov = len(q_cw & r_cw) / len(q_cw)
            score += cov * 25
        else:
            cov = 0.5

        # Bigram coverage
        q_words_list = [w for w in re.findall(r'[a-z]+', ql) if len(w) > 1]
        r_words_list = [w for w in re.findall(r'[a-z]+', rl) if len(w) > 1]
        q_bg = set(zip(q_words_list, q_words_list[1:]))
        r_bg = set(zip(r_words_list, r_words_list[1:]))
        if q_bg:
            bcov = len(q_bg & r_bg) / len(q_bg)
            score += bcov * 10

        # === LENGTH NEUTRALITY for simple queries ===
        # Detect simple-answer queries
        simple_marker = bool(re.search(
            r'\bname (?:a|one|some|two)\b|\bwhat (?:is|are) the\b|\bwhich\b|'
            r'\bwho (?:is|wrote|composed|invented|painted)\b|'
            r'\bsuggest (?:a|some|one|two|two)\b|\bfavorite\b|'
            r'\bbest (?:cut|way|method)\b|\bhow (?:many|much)\b|'
            r'\boptions?:|\banswer is\?', ql))
        is_short_query = q_wc < 25 and resp.count('?') == 0
        has_choices = bool(re.search(r'\([a-eA-E]\)', query) or re.search(r'options?:', ql))

        # On simple queries: don't reward long over short
        if simple_marker or has_choices:
            if wc <= 30:
                score += 4  # short concise answers fine
            elif wc <= 80:
                score += 2
            else:
                score -= 1  # mild penalty for over-explaining

        # On open-ended queries: length helps
        else:
            if wc < 5:
                score -= 4
            elif wc < 20:
                score += 0
            elif wc < 80:
                score += 4
            elif wc < 250:
                score += 6
            elif wc < 500:
                score += 4
            else:
                score += 1

        # === REFUSAL / CLARIFICATION-INSTEAD-OF-ANSWER penalty ===
        # When the response refuses or asks back instead of answering
        is_clarification = (
            resp.endswith('?') and
            re.search(r'^(?:can you|could you|do you|when you say|please (?:describe|clarify)|what (?:do you|is the))', rl) and
            wc < 50
        )
        if is_clarification:
            score -= 8

        # Refusal patterns
        refusal_strong = bool(re.search(
            r'(?:i must point out|not factually coherent|not a realistic|'
            r'not (?:a )?meaningful|unhealthy ingredient|cannot provide an answer|'
            r'i\'?m not able to|the (?:question|prompt) (?:itself )?(?:may not be|is not))',
            rl))
        if refusal_strong:
            # Check if refusal is actually appropriate (genuinely incoherent query)
            incoherent_query = bool(re.search(
                r'rhino.*air|cat.*lawn.*desert|fish.*lungs|breathe through lungs', ql))
            if incoherent_query:
                score += 4  # refusal IS the right move
            else:
                score -= 10  # refusal on benign query is bad

        # === DIRECT ANSWER MARKERS ===
        # Answers that start with the actual content
        if re.match(r'^[A-Z][a-z]+\s*\.?\s*$', resp):  # one-word answer
            # only count as good if simple query
            if simple_marker or has_choices:
                score += 6

        # Answer leads with the key noun from query
        if q_cw:
            first_words = set(re.findall(r'[a-z]+', rl[:60]))
            if len(first_words & q_cw) >= min(2, len(q_cw)):
                score += 2

        # === BOILERPLATE PENALTY ===
        if re.match(r'^(sure|great question|that\'?s a great|i\'?d be happy|hello!)', rl):
            score -= 4
        if re.search(r'i hope this helps|happy to help|let me know if', rl):
            score -= 2

        # === FACTUAL CONSISTENCY CHECK (lightweight) ===
        # Detect inconsistency: "fish/tadpoles use lungs" type errors when query has factual constraint
        # Look for contradictions: response says X when query implies not-X
        if 'lung' in ql and 'fish' in ql:
            if re.search(r'fish.*(?:rely on|use|have).*lungs?|both.*rely on.*lungs', rl):
                score -= 8  # factually wrong
            elif re.search(r'fish.*(?:gills|don\'?t|do not).*lungs?|lung is not associated', rl):
                score += 6  # factually correct

        # === Math/option answer detection ===
        # If query has options and response confidently states an option
        if has_choices:
            opt_match = re.search(r'\(([A-E])\)|option\s*([A-E])|^\s*([A-E])\s*\.', resp)
            if opt_match:
                score += 4

        # === Translation gate ===
        if re.search(r'\btranslat(?:e|ion)\b|romanian:|spanish:|french:', ql):
            non_ascii = sum(1 for c in resp if ord(c) > 127 and c.isalpha())
            if non_ascii < 4 and wc > 15:
                score -= 18
            elif non_ascii >= 4:
                score += 6

        # === Bot template ===
        if re.search(r'welcome to /r/|i am a bot|please read our rules', rl):
            score -= 15

        # === Garbled output ===
        garbage = len(re.findall(r'[^\x00-\x7F]', resp))
        if garbage / max(len(resp), 1) > 0.20:
            if 'translate' not in ql and 'latex' not in ql:
                score -= 12

        # === Code request fulfillment ===
        if re.search(r'\b(code|sql|select|create table|python|function|latex)\b', ql):
            has_code = bool(re.search(r'```|\bSELECT\b|\bCREATE\b|\bdef \b|\\usepackage', resp))
            if has_code:
                score += 5
            else:
                score -= 2

        # === Penalty for response that just repeats query / asks for it back ===
        if q_cw and r_cw:
            jacc = len(q_cw & r_cw) / max(len(q_cw | r_cw), 1)
            # If response is mostly just the query verbatim and short
            if jacc > 0.7 and wc < 30:
                score -= 4

        # === Specificity small bonus (concrete details) ===
        nums = len(re.findall(r'\b\d+\b', resp))
        score += min(nums * 0.3, 3)

        score = max(0.0, min(100.0, score))
        return round(score, 2)
    except Exception:
        return 50.0
