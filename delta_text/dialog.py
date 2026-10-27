"""Ответ на реплику (Delta LM, раздел 3.6, шаг 3): выбор между гипотезами.

    Есть применимые гипотезы?
    ├─ нет → UNKNOWN с указанием, что именно не понято
    └─ да → Лучшие (с наименьшим числом допущений) дают один ответ?
            ├─ да → ответить; допущения назвать явно («Понял так: …»)
            └─ нет → задать один уточняющий вопрос по месту расхождения

Статусы ответа:
- `proven` — понимание без допущений, все выводы ядра доказаны, оговорок данных нет;
- `assumed` — ответ получен, но опирается на названные допущения или оговорки;
- `quoted` — понятия нет в деревьях, приведено определение из словаря RazumWiki;
- `unknown` — ответа нет, указано почему;
- `clarify` — нужен ответ человека на уточняющий вопрос.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from delta_text.knowledge import KnowledgeBase, default_knowledge
from delta_text.reason import NotApplicable, TaskResult, execute, wiki_lines
from delta_text.understand import Hypothesis, Understanding, understand


@dataclass
class Reply:
    text: str
    status: str  # proven | assumed | quoted | unknown | clarify
    assumptions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    certificates: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def respond(text: str, kb: KnowledgeBase | None = None) -> Reply:
    kb = kb or default_knowledge()
    u = understand(text, kb)
    lemmas = frozenset(lemma for t in u.tokens if t.kind == "word" for lemma in t.lemmas)
    if not u.mentions:
        quoted = None if u.other_sense else _wiki_fallback(u, kb)
        return quoted or Reply(_not_understood(u, kb), "unknown", trace=_trace(u, []))

    done: list[tuple[Hypothesis, TaskResult]] = []
    rejected: list[tuple[Hypothesis, str]] = []
    for h in u.hypotheses:
        try:
            done.append((h, execute(h.intent, h.concepts, kb, lemmas)))
        except NotApplicable as exc:
            rejected.append((h, str(exc)))
    if not done:
        h, reason = min(rejected, key=lambda item: len(item[0].assumptions))
        flagged = {k for _, k in u.other_sense}
        if flagged and set(h.concepts) <= flagged:
            return Reply(_not_understood(u, kb), "unknown", h.assumptions, trace=_trace(u, done, rejected))
        names = ", ".join(f"«{kb.concepts[k].name}»" for k in h.concepts)
        text = f"Не могу ответить: {reason}. Нашёл понятия: {names}."
        if u.unrecognized:
            text += " Не знаю: " + ", ".join(f"«{w}»" for w in u.unrecognized) + "."
        return Reply(text, "unknown", h.assumptions, trace=_trace(u, done, rejected))

    fewest = min(len(h.assumptions) for h, _ in done)
    best = [(h, r) for h, r in done if len(h.assumptions) == fewest]
    distinct = {"\n".join(r.lines): (h, r) for h, r in best}
    if len(distinct) > 1:
        return Reply(_clarifying_question(u, [h for h, _ in distinct.values()], kb), "clarify",
                     trace=_trace(u, done, rejected))

    h, r = best[0]
    r.caveats += [
        f"возможно, «{text}» здесь употреблено не в физико-химическом смысле (догадка обученного дерева смысла)"
        for text, key in u.other_sense if key in h.concepts
    ]
    lines = []
    if h.assumptions:
        lines.append(f"Понял так: {r.restatement}.")
    lines.extend(r.lines)
    if r.caveats:
        lines.append("Оговорка: " + "; ".join(dict.fromkeys(r.caveats)) + ".")
    if h.assumptions:
        lines.append("Допущения: " + "; ".join(h.assumptions) + ". Если я понял неверно, уточните.")
    status = "proven" if r.proven and not h.assumptions and not r.caveats else "assumed"
    if not r.proven:
        status = "unknown" if not h.assumptions else "assumed"
    certs = [c for c in r.certificates if c] + ([h.intent_certificate] if h.intent_certificate else [])
    return Reply("\n".join(lines), status, h.assumptions, list(dict.fromkeys(r.caveats)), certs,
                 _trace(u, done, rejected, chosen=h, plan=r.plan))


def _clarifying_question(u: Understanding, options: list[Hypothesis], kb: KnowledgeBase) -> str:
    """Один вопрос по первому упоминанию, на котором гипотезы расходятся."""
    for idx, mention in enumerate(u.mentions):
        picks = list(dict.fromkeys(h.choice[idx] for h in options))
        if len(picks) > 1:
            variants = " или ".join(f"«{kb.concepts[mention.candidates[p].concept].name}»" for p in picks)
            return f"Уточните: «{mention.text}» — это {variants}?"
    intents = " или ".join(f"«{h.intent_label.lower()}»" for h in options)
    return f"Уточните, что вы хотите узнать: {intents}?"


DEFINE_CUES = {"что", "такой", "означать", "значить", "определение", "кто"}


def _wiki_fallback(u: Understanding, kb: KnowledgeBase) -> Reply | None:
    """Вопрос «что это» о понятии вне деревьев: цитата из словаря RazumWiki.

    Ядро такой ответ не выводит, поэтому статус — `quoted`, а не `proven`.
    """
    words = [t for t in u.tokens if t.kind == "word"]
    content = [t for t in words if not t.is_service and not (set(t.lemmas) & DEFINE_CUES)]
    asks_define = any(set(t.lemmas) & DEFINE_CUES for t in words) or len(content) == 1
    if not content or not asks_define:
        return None
    phrases = [f"{a.text} {b.text}" for a, b in zip(content, content[1:])]
    for t in content:
        phrases += [t.text, *t.lemmas]
    entry_lines = wiki_lines(kb, *phrases)
    if not entry_lines:
        return None
    text = ("Этого понятия нет в моих деревьях знаний, поэтому вывести ответ ядром не могу.\n"
            + entry_lines[0])
    return Reply(text, "quoted", trace=_trace(u, []))


def _not_understood(u: Understanding, kb: KnowledgeBase) -> str:
    if u.other_sense:
        words = "; ".join(f"«{text}» здесь, похоже, не «{kb.concepts[key].name}»" for text, key in u.other_sense)
        return (f"Похоже, вопрос не о физике или химии: {words}. "
                "Это догадка обученного дерева смысла слова; если я ошибся, переформулируйте вопрос.")
    if u.unrecognized:
        words = ", ".join(f"«{w}»" for w in u.unrecognized)
        noun = "понятия" if len(u.unrecognized) == 1 else "понятий"
        return f"Не понял: не знаю {noun} {words}. Я знаю понятия о материи и химических элементах."
    return "Не нашёл в вопросе известных мне понятий. Я знаю понятия о материи и химических элементах."


def _trace(u: Understanding, done, rejected=(), chosen: Hypothesis | None = None, plan=None) -> dict[str, Any]:
    return {
        "mentions": [
            {"text": m.text, "candidates": [{"concept": c.concept, "assumption": c.assumption} for c in m.candidates]}
            for m in u.mentions
        ],
        "unrecognized": u.unrecognized,
        "other_sense": [{"text": t, "concept": k} for t, k in u.other_sense],
        "hypotheses": [
            {"intent": h.intent, "concepts": h.concepts, "assumptions": h.assumptions} for h, _ in done
        ],
        "rejected": [{"intent": h.intent, "concepts": h.concepts, "reason": reason} for h, reason in rejected],
        "chosen": None if chosen is None else {"intent": chosen.intent, "concepts": chosen.concepts},
        "plan": plan or [],
    }
