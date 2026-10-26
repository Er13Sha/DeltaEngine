"""Адаптер морфологии русского языка (pymorphy3, словарь OpenCorpora).

Вне доверенного ядра: морфология предлагает варианты разбора, но не решает,
какой из них верен. Выбор делают вилки понимания.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Части речи, которые не несут понятий: предлоги, союзы, частицы, местоимения, междометия.
SERVICE_POS = {"PREP", "CONJ", "PRCL", "NPRO", "INTJ", "PRED"}


@dataclass(frozen=True)
class Parse:
    lemma: str
    pos: str
    score: float


@lru_cache(maxsize=1)
def _analyzer():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


@lru_cache(maxsize=65536)
def parses(word: str) -> tuple[Parse, ...]:
    """Все варианты разбора слова (леммы нормализованы: строчные, ё → е)."""
    out = []
    seen = set()
    for p in _analyzer().parse(word):
        key = (normalize(p.normal_form), str(p.tag.POS))
        if key in seen:
            continue
        seen.add(key)
        out.append(Parse(key[0], key[1], p.score))
    return tuple(out)


@lru_cache(maxsize=65536)
def is_known(word: str) -> bool:
    return _analyzer().word_is_known(word)


def name_pattern(phrase: str) -> tuple[frozenset[str], ...]:
    """Образец многословного имени: для каждого слова — множество допустимых лемм.

    Слово текста совпадает со словом имени, если их варианты лемм пересекаются:
    так «щелочных металлов» совпадает с именем «щелочной металл».
    """
    words = [w for w in phrase.replace("-", " ").split() if w]
    return tuple(frozenset({normalize(w)} | {p.lemma for p in parses(w)}) for w in words)
