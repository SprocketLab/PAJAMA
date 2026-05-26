def judging_function(query, response):
    """
    Evaluates structural organization and formatting quality of a response.
    
    This variant focuses on:
    - Hierarchical depth analysis (nested structures)
    - Visual separation and chunking patterns
    - Formatting diversity score (variety of formatting elements used)
    - Sentence length variance as a proxy for structural rhythm
    - Inline formatting (bold, italic, code, quotes)
    - Response-to-query proportionality and structural appropriateness
    """
    import re
    import math
    from collections import Counter
    
    try:
        if not response or not isinstance(response, str):
            return 0.0
        if not query or not isinstance(query, str):
            query = ""
        
        response = response.strip()
        if len(response) < 10:
            return 0.5
        
        lines = response.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        total_chars = len(response)
        
        # ============================================================
        # FEATURE 1: Formatting Element Diversity Score
        # Instead of counting individual elements, measure the VARIETY
        # of different formatting mechanisms used
        # ============================================================
        formatting_types_present = set()
        
        # Check for numbered lists (various styles)
        if re.search(r'^\s*\d+[\.\)]\s+\S', response, re.MULTILINE):
            formatting_types_present.add('numbered_list')
        
        # Check for bullet points (various styles)
        if re.search(r'^\s*[-*•▪▸►]\s+\S', response, re.MULTILINE):
            formatting_types_present.add('bullet_list')
        
        # Check for markdown headers
        if re.search(r'^#{1,6}\s+\S', response, re.MULTILINE):
            formatting_types_present.add('md_header')
        
        # Check for bold text
        if re.search(r'\*\*[^*]+\*\*', response) or re.search(r'__[^_]+__', response):
            formatting_types_present.add('bold')
        
        # Check for italic text
        if re.search(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response) or re.search(r'(?<!_)_(?!_)[^_]+_(?!_)', response):
            formatting_types_present.add('italic')
        
        # Check for code blocks
        if '```' in response:
            formatting_types_present.add('code_block')
        
        # Check for inline code
        if re.search(r'`[^`]+`', response):
            formatting_types_present.add('inline_code')
        
        # Check for blockquotes
        if re.search(r'^\s*>\s+', response, re.MULTILINE):
            formatting_types_present.add('blockquote')
        
        # Check for colon-based definitions/labels
        if re.search(r'^[A-Z][^:]{2,30}:\s+\S', response, re.MULTILINE):
            formatting_types_present.add('label_value')
        
        # Check for parenthetical asides/citations
        if re.search(r'\([^)]{5,}\)', response):
            formatting_types_present.add('parenthetical')
        
        # Check for em-dashes used for structure
        if ' -- ' in response or ' — ' in response:
            formatting_types_present.add('em_dash')
        
        # Check for URLs/links
        if re.search(r'https?://|www\.|\[.*?\]\(.*?\)', response):
            formatting_types_present.add('links')
        
        diversity_count = len(formatting_types_present)
        # Logarithmic scaling: diversity has diminishing returns
        diversity_score = min(2.0, math.log1p(diversity_count) * 0.9)
        
        # ============================================================
        # FEATURE 2: Visual Chunking Analysis
        # Analyze how the response is broken into visual "chunks"
        # separated by blank lines or structural elements
        # ============================================================
        # Split by blank lines to find chunks
        chunks = re.split(r'\n\s*\n', response)
        chunks = [c.strip() for c in chunks if c.strip()]
        num_chunks = len(chunks)
        
        # Measure chunk size consistency (coefficient of variation)
        if num_chunks > 1:
            chunk_lengths = [len(c) for c in chunks]
            mean_chunk = sum(chunk_lengths) / len(chunk_lengths)
            if mean_chunk > 0:
                variance = sum((cl - mean_chunk) ** 2 for cl in chunk_lengths) / len(chunk_lengths)
                std_chunk = math.sqrt(variance)
                cv = std_chunk / mean_chunk  # coefficient of variation
                # Moderate variation is good (not all identical, not wildly different)
                # Sweet spot around 0.3-0.7
                if cv < 0.1:
                    chunk_consistency_score = 0.3  # too uniform
                elif cv < 0.8:
                    chunk_consistency_score = 1.0  # good variation
                else:
                    chunk_consistency_score = max(0.2, 1.0 - (cv - 0.8) * 0.5)
            else:
                chunk_consistency_score = 0.2
        else:
            chunk_consistency_score = 0.0
        
        # Reward having multiple chunks, scaled by response length
        if total_chars > 200:
            expected_chunks = max(2, total_chars // 300)
            chunk_ratio = min(num_chunks, expected_chunks * 2) / expected_chunks
            chunk_quantity_score = min(1.5, chunk_ratio * 0.8)
        elif total_chars > 50:
            chunk_quantity_score = 0.4 if num_chunks >= 2 else 0.1
        else:
            chunk_quantity_score = 0.2
        
        chunking_score = chunk_consistency_score * 0.5 + chunk_quantity_score
        
        # ============================================================
        # FEATURE 3: Sentence Rhythm Analysis
        # Good writing has varied sentence lengths creating a "rhythm"
        # ============================================================
        # Extract sentences (rough approximation)
        sentence_pattern = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response)
        sentences = [s.strip() for s in sentence_pattern if len(s.strip()) > 5]
        
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            if mean_sl > 0:
                variance_sl = sum((sl - mean_sl) ** 2 for sl in sent_lengths) / len(sent_lengths)
                std_sl = math.sqrt(variance_sl)
                cv_sl = std_sl / mean_sl
                # Good rhythm: CV between 0.3 and 0.8
                if cv_sl < 0.15:
                    rhythm_score = 0.3  # monotonous
                elif cv_sl < 0.9:
                    rhythm_score = 1.0
                else:
                    rhythm_score = 0.6  # too erratic
            else:
                rhythm_score = 0.3
        elif len(sentences) >= 1:
            rhythm_score = 0.4
        else:
            rhythm_score = 0.2
        
        # ============================================================
        # FEATURE 4: Hierarchical Nesting Depth
        # Detect indentation levels, nested lists, sub-sections
        # ============================================================
        indent_levels = set()
        for line in non_empty_lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                indent_levels.add(indent)
        
        # Check for nested list items
        nested_list_items = len(re.findall(r'^\s{2,}[-*•\d]', response, re.MULTILINE))
        
        nesting_score = 0.0
        if len(indent_levels) >= 3:
            nesting_score = 1.0
        elif len(indent_levels) >= 2:
            nesting_score = 0.7
        elif len(indent_levels) >= 1:
            nesting_score = 0.4
        
        if nested_list_items >= 3:
            nesting_score = min(1.2, nesting_score + 0.4)
        
        # ============================================================
        # FEATURE 5: Opening and Closing Structure
        # Good responses often have an intro, body, and conclusion
        # ============================================================
        structure_flow_score = 0.0
        
        if num_chunks >= 3:
            first_chunk = chunks[0]
            last_chunk = chunks[-1]
            middle_chunks = chunks[1:-1]
            
            # Check if first chunk is shorter (intro-like)
            avg_middle_len = sum(len(c) for c in middle_chunks) / len(middle_chunks) if middle_chunks else total_chars
            
            if len(first_chunk) < avg_middle_len * 1.5:
                structure_flow_score += 0.3
            
            # Check if last chunk has concluding indicators
            conclusion_words = ['overall', 'in summary', 'in conclusion', 'to summarize', 
                              'hope this', 'in short', 'ultimately', 'finally', 'in the end',
                              'to sum up', 'all in all', 'the bottom line']
            last_lower = last_chunk.lower()
            if any(cw in last_lower for cw in conclusion_words):
                structure_flow_score += 0.4
            
            # Having intro-body-conclusion structure
            if len(first_chunk) > 20 and len(last_chunk) > 20:
                structure_flow_score += 0.3
        elif num_chunks == 2:
            structure_flow_score = 0.3
        
        structure_flow_score = min(1.0, structure_flow_score)
        
        # ============================================================
        # FEATURE 6: Wall-of-Text Penalty
        # Penalize long unbroken text blocks
        # ============================================================
        wall_penalty = 0.0
        
        # Find the longest unbroken paragraph
        max_chunk_len = max((len(c) for c in chunks), default=0) if chunks else total_chars
        
        if max_chunk_len > 800:
            wall_penalty = min(2.5, (max_chunk_len - 800) / 400)
        elif max_chunk_len > 500 and total_chars > 600:
            wall_penalty = 0.5
        
        # Also penalize if response is long but has very few line breaks
        if total_chars > 300:
            line_break_density = response.count('\n') / (total_chars / 100)
            if line_break_density < 0.5:
                wall_penalty += min(1.0, (0.5 - line_break_density) * 2)
        
        # ============================================================
        # FEATURE 7: Structural Signposting
        # Words/phrases that help navigate the response
        # ============================================================
        signpost_patterns = [
            r'\bfirst(?:ly)?\b', r'\bsecond(?:ly)?\b', r'\bthird(?:ly)?\b',
            r'\bnext\b', r'\bthen\b', r'\bfinally\b', r'\blastly\b',
            r'\bfor example\b', r'\bfor instance\b', r'\bspecifically\b',
            r'\bin particular\b', r'\bon the other hand\b', r'\bhowever\b',
            r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b',
            r'\bin contrast\b', r'\bconversely\b', r'\bnevertheless\b',
            r'\bthat said\b', r'\bimportantly\b', r'\bnotably\b',
            r'\bto clarify\b', r'\bto be clear\b', r'\bin other words\b',
            r'\bthe key\b', r'\bthe main\b', r'\bthe point\b',
        ]
        
        response_lower = response.lower()
        signpost_count = 0
        unique_signposts = set()
        for pat in signpost_patterns:
            matches = re.findall(pat, response_lower)
            if matches:
                signpost_count += len(matches)
                unique_signposts.add(pat)
        
        # Reward VARIETY of signposts more than raw count
        signpost_variety = len(unique_signposts)
        signpost_score = min(1.5, signpost_variety * 0.25)
        
        # ============================================================
        # FEATURE 8: Response Length Appropriateness
        # Not too short for complex queries, not unnecessarily padded
        # ============================================================
        query_complexity = 0
        query_lower = query.lower()
        
        # Estimate query complexity
        query_words = len(query.split())
        if query_words > 50:
            query_complexity = 3
        elif query_words > 25:
            query_complexity = 2
        else:
            query_complexity = 1
        
        # Check for multi-part questions
        question_marks = query.count('?')
        if question_marks > 1:
            query_complexity = max(query_complexity, 2)
        
        response_words = len(response.split())
        
        length_appropriateness = 0.5  # default
        if query_complexity >= 2:
            # Complex query: longer, well-structured response is better
            if response_words > 80:
                length_appropriateness = 1.0
            elif response_words > 40:
                length_appropriateness = 0.7
            else:
                length_appropriateness = 0.3
        else:
            # Simple query: moderate length is fine
            if 20 <= response_words <= 300:
                length_appropriateness = 0.8
            elif response_words > 300:
                length_appropriateness = 0.6
            else:
                length_appropriateness = 0.4
        
        # ============================================================
        # FEATURE 9: Inline Formatting Richness
        # Use of emphasis, code, quotes within text
        # ============================================================
        inline_score = 0.0
        
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', response))
        italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response))
        inline_code_count = len(re.findall(r'`[^`]+`', response))
        
        inline_elements = bold_count + italic_count + inline_code_count
        if inline_elements > 0:
            # Reward moderate use, penalize overuse
            if inline_elements <= 10:
                inline_score = min(1.0, inline_elements * 0.2)
            else:
                inline_score = max(0.3, 1.0 - (inline_elements - 10) * 0.05)
        
        # ============================================================
        # FEATURE 10: Semantic Section Detection
        # Look for topic shifts indicated by structural markers
        # ============================================================
        section_markers = 0
        for line in non_empty_lines:
            stripped = line.strip()
            # Lines that look like section headers (short, possibly ending with colon)
            if len(stripped.split()) <= 8 and stripped.endswith(':') and len(stripped) > 3:
                section_markers += 1
            # All-caps short lines
            elif stripped.isupper() and len(stripped.split()) <= 6 and len(stripped) > 3:
                section_markers += 1
            # Bold-only lines (likely headers)
            elif re.match(r'^\*\*[^*]+\*\*$', stripped):
                section_markers += 1
        
        section_score = min(1.0, section_markers * 0.3)
        
        # ============================================================
        # COMBINE ALL FEATURES
        # ============================================================
        
        # Weighted combination
        raw_score = (
            diversity_score * 1.5 +       # Formatting variety: 0-3.0
            chunking_score * 1.3 +         # Visual chunking: 0-2.6
            rhythm_score * 0.7 +           # Sentence rhythm: 0-0.7
            nesting_score * 0.6 +          # Hierarchical depth: 0-0.72
            structure_flow_score * 0.8 +   # Intro/body/conclusion: 0-0.8
            signpost_score * 0.8 +         # Signposting: 0-1.2
            length_appropriateness * 1.0 + # Length fit: 0-1.0
            inline_score * 0.5 +           # Inline formatting: 0-0.5
            section_score * 0.6 -          # Section headers: 0-0.6
            wall_penalty * 1.2             # Wall-of-text: 0-3.0
        )
        
        # Normalize to 0-10 range
        # Theoretical max ~11.12, typical good response ~5-7
        final_score = max(0.0, min(10.0, raw_score))
        
        # Apply a slight sigmoid-like transformation to spread scores
        # This maps the ~1-8 range more evenly across 0-10
        midpoint = 4.0
        steepness = 0.5
        transformed = 10.0 / (1.0 + math.exp(-steepness * (final_score - midpoint)))
        
        return round(transformed, 3)
        
    except Exception:
        try:
            # Minimal fallback: just check basic structure
            if not response or len(response.strip()) < 10:
                return 0.5
            lines = response.strip().split('\n')
            non_empty = [l for l in lines if l.strip()]
            if len(non_empty) > 3:
                return 4.0
            elif len(non_empty) > 1:
                return 2.5
            return 1.5
        except Exception:
            return 2.0