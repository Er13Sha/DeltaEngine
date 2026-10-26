"""Разметка текста (Delta LM, раздел 3.1): токены и варианты морфологического разбора.

Детерминированно и без обучения. Если у слова несколько разборов, передаются все.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from delta_text.morph import SERVICE_POS, is_known, normalize, parses

# Кириллическое слово (с дефисом внутри), латинское слово/формула (H2O, Na), число, прочий знак.
_TOKEN = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*|[A-Za-z][A-Za-z0-9]*|\d+|[^\sА-Яа-яЁёA-Za-z\d]")


@dataclass(frozen=True)
class Token:
    text: str
    kind: str  # word | latin | number | punct
    lemmas: tuple[str, ...]  # все варианты лемм; для latin/number/punct — нормализованный текст
    pos: tuple[str, ...]
    known: bool

    @property
    def is_service(self) -> bool:
        """Самый вероятный разбор — служебная часть речи."""
        return self.kind == "word" and bool(self.pos) and self.pos[0] in SERVICE_POS


def annotate(text: str) -> list[Token]:
    tokens = []
    for m in _TOKEN.finditer(text):
        raw = m.group(0)
        if re.fullmatch(r"[А-Яа-яЁё-]+", raw):
            variants = parses(raw)
            tokens.append(Token(
                raw, "word",
                tuple(dict.fromkeys(p.lemma for p in variants)),
                tuple(dict.fromkeys(p.pos for p in variants)),
                is_known(raw),
            ))
        elif raw[0].isascii() and raw[0].isalpha():
            tokens.append(Token(raw, "latin", (raw,), (), True))
        elif raw.isdigit():
            tokens.append(Token(raw, "number", (raw,), (), True))
        else:
            tokens.append(Token(raw, "punct", (normalize(raw),), (), True))
    return tokens
