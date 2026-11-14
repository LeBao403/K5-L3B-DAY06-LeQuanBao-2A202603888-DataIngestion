from typing import List, Dict


def evaluate_query(question: dict, hits: List[dict]) -> dict:
    """
    Compute Hit@1 and Hit@3 for a single question based on:
    - gold_page: chunk belongs to the target page.
    - evidence_quote: chunk contains the quote or key facts.
    """
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote", "").lower().strip()
    gold_ids = set(question.get("gold", []))

    def is_match(hit: dict) -> bool:
        # Match by chunk_id if present
        if gold_ids and hit.get("chunk_id") in gold_ids:
            return True
        # Match by page
        if gold_page is not None and hit.get("page") == gold_page:
            return True
        # Match by evidence text overlap
        hit_text = hit.get("text", "").lower()
        if evidence_quote and (evidence_quote in hit_text or any(w in hit_text for w in evidence_quote.split()[:5])):
            return True
        return False

    matches = [is_match(h) for h in hits]
    
    hit1 = int(matches[0]) if matches else 0
    hit3 = int(any(matches[:3])) if matches else 0

    return {"hit1": hit1, "hit3": hit3, "gold_in_hits": bool(hit1 or hit3)}


def summarize(traces: List[dict]) -> dict:
    summary: dict = {}
    for t in traces:
        key = f"{t['strategy']}_{t['split']}"
        if key not in summary:
            summary[key] = {"hit1": 0, "hit3": 0, "total": 0}
        summary[key]["hit1"] += t["hit1"]
        summary[key]["hit3"] += t["hit3"]
        summary[key]["total"] += 1

    for stats in summary.values():
        stats["hit1_ratio"] = stats["hit1"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["hit3_ratio"] = stats["hit3"] / stats["total"] if stats["total"] > 0 else 0.0
    return summary
