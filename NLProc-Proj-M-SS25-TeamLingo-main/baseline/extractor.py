import re

def clean_list_answer(raw_ans: str) -> list:
    """
    Splits a newline-separated answer into items,
    strips bullets/whitespace, and de-duplicates while preserving order.
    """
    items = [line.strip("-• ").strip() for line in raw_ans.splitlines() if line.strip()]
    seen = set()
    out = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

class Extractor:
    # ... your existing numeric/definition methods ...

    def extract_list(
        self,
        raw_ans: str,
        expected_keywords: list = None,
        generator=None,
        contexts: list = None
    ) -> list:
        items = clean_list_answer(raw_ans)
        # Fallback: if we missed keywords, re-generate with top context only
        if expected_keywords and generator and contexts:
            missing = [
                kw for kw in expected_keywords
                if kw.lower() not in [it.lower() for it in items]
            ]
            if missing and contexts:
                # simple bullet prompt
                prompt = "List each answer as “- <item>” on its own line."
                raw2 = generator.generate(prompt)
                items = clean_list_answer(raw2)
        return items
