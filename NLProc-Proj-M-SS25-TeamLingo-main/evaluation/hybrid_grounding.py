# evaluation/hybrid_grounding.py

import re
from typing import List, Tuple
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# 1) Load resources once
_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_nlp = spacy.load("en_core_web_sm")
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

CITATION_TAG = r"\[source\]"

def has_citation(answer: str) -> bool:
    """Check for at least one [source] tag."""
    return bool(re.search(CITATION_TAG, answer))

def semantic_score(pred: str, gold: str) -> float:
    """Return cosine similarity via SBERT."""
    emb1 = _sbert.encode(pred, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = _sbert.encode(gold, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb1, emb2).item()

def rouge_l_score(pred: str, gold: str) -> float:
    """Return ROUGE-L recall score."""
    scores = _rouge.score(gold, pred)
    return scores["rougeL"].recall

def entity_overlap(pred: str, gold_items: List[str]) -> float:
    """
    Extract named entities from pred and compare to gold_items.
    Returns fraction of gold_items recovered.
    """
    doc = _nlp(pred)
    pred_ents = {ent.text.lower() for ent in doc.ents}
    gold_set = {g.lower() for g in gold_items}
    if not gold_set:
        return 0.0
    return len(pred_ents & gold_set) / len(gold_set)

def is_grounded(
    answer: str,
    gold_chunk: str,
    question_type: str,
    gold_list: List[str] = None
) -> Tuple[bool, dict]:
    """
    Returns (grounded, reasons) where reasons include the individual scores.
    - answer: generated answer containing [source] tags
    - gold_chunk: the text of the chunk the test expects
    - question_type: one of 'numeric', 'definition', 'list', 'general'
    - gold_list: for 'list' questions, the list of expected items
    """
    reasons = {}
    # Step 1: citation check
    cit = has_citation(answer)
    reasons["has_citation"] = cit
    if not cit:
        return False, reasons

    # strip citation tags for content comparison
    clean_ans = re.sub(CITATION_TAG, "", answer).strip()

    # Step 2: semantic similarity
    sem = semantic_score(clean_ans, gold_chunk)
    reasons["semantic_cosine"] = sem
    if sem >= 0.7:
        return True, reasons

    # Step 3: for lists, entity overlap
    if question_type == "list" and gold_list is not None:
        ent_ol = entity_overlap(clean_ans, gold_list)
        reasons["entity_overlap"] = ent_ol
        if ent_ol >= 0.5:
            return True, reasons

    # Step 4: ROUGE-L fallback
    rouge_l = rouge_l_score(clean_ans, gold_chunk)
    reasons["rougeL_recall"] = rouge_l
    if rouge_l >= 0.4:
        return True, reasons

    # else ungrounded
    return False, reasons
