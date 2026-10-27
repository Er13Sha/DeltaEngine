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
    case: str = ""  # падеж (nomn, gent, …); пусто у неизменяемых частей речи


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
        key = (normalize(p.normal_form), str(p.tag.POS), str(p.tag.case or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(Parse(key[0], key[1], p.score, key[2]))
    return tuple(out)


@lru_cache(maxsize=65536)
def is_known(word: str) -> bool:
    return _analyzer().word_is_known(word)


def name_pattern(phrase: str) -> tuple[frozenset[str], ...]:
    """Образец многословного имени: для каждого слова — множество допустимых лемм.

    Слово текста совпадает со словом имени, если их варианты лемм пересекаются:
    так «щелочных металлов» совпадает с именем «щелочной металл».

    Имена записаны в начальной форме, поэтому берутся только разборы в именительном
    падеже (или без падежа) той же части речи, что и самый вероятный разбор.
    Иначе в образец попадают посторонние леммы: «метан» — краткое причастие
    от «метать», «поле» — предложный падеж от «пол», «сера» — краткое от «серый».
    """
    words = [w for w in phrase.replace("-", " ").split() if w]
    pattern = []
    for w in words:
        variants = parses(w)
        top_pos = variants[0].pos if variants else ""
        lemmas = {p.lemma for p in variants if p.pos == top_pos and p.case in ("nomn", "")}
        pattern.append(frozenset({normalize(w)} | lemmas))
    return tuple(pattern)
