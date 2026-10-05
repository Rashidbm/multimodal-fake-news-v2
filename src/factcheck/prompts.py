"""Prompts, tasks and label sets.

Bump ``PROMPT_VERSION`` whenever any text here changes; it is part of the
configuration hash, so old run directories will refuse to resume.
"""
from __future__ import annotations

PROMPT_VERSION = "2026-10-05.1"

# Default task: does the evidence support the claim made by the text + image?
CLAIM_LABELS = {
    "supported": "Reliable evidence confirms the claim as stated, including what the image is presented as showing.",
    "refuted": "Reliable evidence contradicts the claim or shows the image is presented with a false context.",
    "conflicting": "Reliable sources disagree, or the claim is partly true and partly false.",
    "insufficient": "The available evidence is not enough to decide.",
}

# MMFakeBench's own four-way labels (fake_cls), arXiv 2406.08772 section 3.
MMFAKE_LABELS = {
    "original": "Real news: the text is accurate and the image genuinely belongs to it.",
    "textual_veracity_distortion": "The text is false: a rumour, fabricated or edited claim, whatever the image.",
    "visual_veracity_distortion": "The text is accurate but the image is manipulated or AI-generated.",
    "mismatch": "Text and image are each authentic but do not belong together (out-of-context pairing).",
}

TASKS = {"claim": CLAIM_LABELS, "mmfakebench": MMFAKE_LABELS}
DEFAULT_TASK = "claim"

# Labels that assert something about the world and therefore must cite evidence
# when evidence was provided (Config.require_citation).
CITATION_REQUIRED = {"claim": {"supported", "refuted", "conflicting"}, "mmfakebench": set()}


def labels(task: str) -> list[str]:
    try:
        return list(TASKS[task])
    except KeyError:
        raise ValueError(f"unknown task {task!r}; choose one of {sorted(TASKS)}") from None


def _label_block(task: str) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in TASKS[task].items())


QUERY_SYSTEM = (
    "You are a careful fact-checker. You receive a news post: a text and an image. "
    "Write web search queries that would find independent reporting, official statements or "
    "earlier publications that confirm or contradict the post. Search for the specific people, "
    "places, dates, numbers and events it mentions, and for what the image appears to show. "
    "Do not include words like 'fake' or 'fact check' unless they help find the original event. "
    "Call the web_search tool exactly once."
)


def query_user(text: str, max_queries: int) -> str:
    return (f"Post text:\n\"\"\"\n{text}\n\"\"\"\n\nThe post's image is attached. "
            f"Give between 1 and {max_queries} short, distinct search queries.")


def verdict_system(task: str, with_evidence: bool) -> str:
    rules = [
        "You are a careful fact-checker. Decide one label for the news post (text + image).",
        f"Labels:\n{_label_block(task)}",
        "Base the decision on what the text claims and what the image shows.",
    ]
    if with_evidence:
        rules += [
            "Evidence items retrieved from the web are given with ids like E1. They may be irrelevant, "
            "outdated or wrong; judge their reliability. Do not use knowledge that contradicts "
            "the evidence without saying so in the rationale.",
            "Support the decision with citations: each citation gives an evidence id and a quote copied "
            "word-for-word from that item's text. Quote short exact spans (one sentence or less). "
            "Never cite an id that was not given.",
        ]
    else:
        rules += ["No evidence is provided; decide from the post itself and your own knowledge. "
                  "Return an empty citations list."]
    rules.append("Answer with JSON only: {\"label\", \"confidence\" (0 to 1), \"rationale\", \"citations\"}.")
    return "\n\n".join(rules)


def format_evidence(evidence: list[dict], assessments: dict | None = None) -> str:
    if not evidence:
        return "Evidence: none was found."
    parts = ["Evidence:"]
    for ev in evidence:
        head = f"[{ev['id']}] {ev.get('title') or '(no title)'}\nURL: {ev['url']}"
        if ev.get("published"):
            head += f"\nPublished: {ev['published']}"
        if assessments and ev["id"] in assessments:
            a = assessments[ev["id"]]
            head += f"\nAssessment: relevance={a['relevance']}, stance={a['stance']}. {a.get('note', '')}".rstrip()
        parts.append(f"{head}\nText: {ev['text']}")
    return "\n\n".join(parts)


def verdict_user(text: str, evidence: list[dict] | None, assessments: dict | None = None) -> str:
    out = f"Post text:\n\"\"\"\n{text}\n\"\"\"\n\nThe post's image is attached."
    if evidence is not None:
        out += "\n\n" + format_evidence(evidence, assessments)
    return out


ASSESS_SYSTEM = (
    "You are a careful fact-checker. For every evidence item, judge it against the news post "
    "(text + image) before any verdict is made.\n"
    "relevance: high = directly about the same event, people or image; low = related background; "
    "none = unrelated.\n"
    "stance: supports, refutes or neutral towards the post as presented.\n"
    "note: one sentence on what the item actually says that matters, or why it does not matter.\n"
    "Assess every id exactly once. Answer with JSON only: {\"assessments\": [...]}."
)


def assess_user(text: str, evidence: list[dict]) -> str:
    return verdict_user(text, evidence)
