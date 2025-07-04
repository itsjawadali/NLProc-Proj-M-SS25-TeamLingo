# baseline/generator/utils.py
def classify_qtype(question: str) -> str:
    ql = question.lower()
    if ql.startswith(("what is","define")):    return "definition"
    if ql.startswith(("list","how many")):     return "list"
    if ql.startswith(("compare","difference")): return "compare"
    if ql.startswith(("why","how does")):       return "explanation"
    return "general"
