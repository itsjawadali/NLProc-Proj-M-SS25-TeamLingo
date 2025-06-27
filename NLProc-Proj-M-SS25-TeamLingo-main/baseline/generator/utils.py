import re

def _truncate_txt(txt: str, max_words: int = 50) -> str:
    """
    Truncate a long context string to at most `max_words` words,
    appending an ellipsis if truncated.
    """
    words = txt.split()
    if len(words) <= max_words:
        return txt
    return " ".join(words[:max_words]) + " …"

def build_prompt(question: str, contexts: list) -> str:
    """
    Simple Q→A prompt: list each truncated context, then ask for the answer.
    """
    # Truncate each context to 50 words
    truncated = [_truncate_txt(ctx, max_words=50) for ctx in contexts]
    ctx_text  = "\n\n".join(truncated)
    return f"Answer the question: {question}\n\n{ctx_text}\n\nAnswer:"

def build_explanation_prompt(question: str, contexts: list) -> str:
    """
    Paragraph‐style prompt for descriptive/explanatory questions.
    Each context is truncated to 50 words to avoid token‐overflow.
    """
    # Truncate and then number each context snippet
    max_words = 50
    pieces = []
    for i, txt in enumerate(contexts):
        t = _truncate_txt(txt, max_words=max_words)
        pieces.append(f"[{i+1}] {t}")
    ctx_block = "\n\n".join(pieces)

    return (
        "You are a climate scientist. Using ONLY the information below, write a concise "
        "explanatory paragraph that answers the question.\n\n"
        f"Question: {question}\n\n"
        "Contexts:\n"
        f"{ctx_block}\n\n"
        "Explanation:"
    )

def classify_qtype(question: str) -> str:
    """
    Heuristic classifier: numeric, list, explanation, definition, or general.
    """
    q = question.lower().strip()
    if re.search(r"\bhow much\b|\bhow many\b", q):
        return "numeric"
    if re.match(r"^(describe|explain)\b", q) \
       or re.search(r"\bwhat (is the role of|are the impacts of|does .* mean|impact)\b", q):
        return "explanation"
    if re.search(r"\bwhich\b|\blist\b|\bwhat types of\b|\bwhat (are|which)\b", q):
        return "list"
    if re.search(r"\bdefine\b|\bwhat is\b|\bimpact\b", q):
        return "definition"
    return "general"
