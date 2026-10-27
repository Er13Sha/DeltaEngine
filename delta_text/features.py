"""Признаки фразы и упоминания для деревьев понимания (тип вопроса, смысл слова).

Одни и те же функции используются при обучении (`learn_intent`) и при понимании
(`understand`), поэтому признаки не расходятся.
"""
from __future__ import annotations

from typing import Any

from delta_kernel import Tree
from delta_text.annotate import Token
from delta_text.knowledge import Concept, KnowledgeBase
from delta_text.morph import normalize


def utterance_record(tokens: list[Token], concepts: list[Concept]) -> dict[str, Any]:
    """Все признаки фразы; дереву передаётся проекция на его схему (`project`)."""
    words = [t for t in tokens if t.kind == "word"]
    return {
        "lemmas": sorted({lemma for t in words for lemma in t.lemmas}),
        "first": words[0].lemmas[0] if words else "",
        "qmark": any(t.text == "?" for t in tokens),
        "n_concepts": len(concepts),
        "pair": pair_kind(concepts),
    }


def mention_record(tokens: list[Token], start: int, end: int, concept_key: str, kb: KnowledgeBase) -> dict[str, Any]:
    concept = kb.concepts[concept_key]
    kind = "class" if concept.kind == "class" else concept_key.split(":")[0]
    context = {lemma for i, t in enumerate(tokens) if t.kind == "word" and not start <= i < end for lemma in t.lemmas}
    return {
        "word": normalize(" ".join(t.text for t in tokens[start:end])),
        "kind": kind,
        "context": sorted(context),
    }


def project(tree: Tree, record: dict[str, Any]) -> dict[str, Any]:
    """Оставляет только атрибуты схемы дерева."""
    return {k: v for k, v in record.items() if k in tree.schema.attributes}


def pair_kind(concepts: list[Concept]) -> str:
    if len(concepts) != 2:
        return "none"
    a, b = concepts
    return "instance_class" if a.kind == "instance" and b.kind == "class" else "other"
