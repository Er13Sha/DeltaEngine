"""Понимание (Delta LM, разделы 3.2 и 3.6): текст → гипотезы смысловой формы.

1. В тексте ищутся упоминания понятий: многословные имена по леммам (сначала
   длинные), символы и формулы. Неоднозначное имя даёт несколько кандидатов.
2. Незнакомое или не найденное в словаре понятий слово проверяется на опечатку:
   кандидаты — имена на расстоянии правки 1 (2 для длинных слов). Каждая такая
   догадка — допущение.
3. Дерево смысла слова (если есть) отсеивает упоминания не в физико-химическом
   смысле: «в Германии» — не германий, «щитовидная железа» — не железо.
4. Для каждой комбинации кандидатов тип вопроса определяет дерево ядра `intent`.
   Вилки-догадки этого дерева помечены как гипотезы, поэтому угаданный тип вопроса
   тоже становится допущением.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from delta_text.annotate import Token, annotate
from delta_text.features import mention_record, project, utterance_record
from delta_text.knowledge import KnowledgeBase
from delta_text.morph import normalize

MAX_HYPOTHESES = 32
INTENT_NAMES = {
    "DEFINE": "вопрос «что это»",
    "DIFFERENCE": "вопрос о различии",
    "IS_A": "вопрос о принадлежности классу",
    "CLASSIFY": "вопрос о классе",
    "OTHER": "вопрос другого вида («почему», «как», «где»…)",
}


@dataclass(frozen=True)
class Candidate:
    concept: str
    assumption: str = ""  # пусто — точное совпадение
    typo: bool = False  # догадка об опечатке (а не уточнение прилагательным)


@dataclass
class Mention:
    start: int
    end: int
    text: str
    candidates: list[Candidate]


@dataclass
class Hypothesis:
    intent: str
    concepts: list[str]
    assumptions: list[str]
    intent_label: str
    intent_certificate: dict[str, Any] | None
    choice: tuple[int, ...]  # номер кандидата для каждого упоминания


@dataclass
class Understanding:
    tokens: list[Token]
    mentions: list[Mention]
    unrecognized: list[str]
    hypotheses: list[Hypothesis] = field(default_factory=list)
    # упоминания, отсеянные деревом смысла: (текст, понятие)
    other_sense: list[tuple[str, str]] = field(default_factory=list)


def understand(text: str, kb: KnowledgeBase) -> Understanding:
    tokens = annotate(text)
    mentions, unrecognized = find_mentions(tokens, kb)
    result = Understanding(tokens, mentions, unrecognized, other_sense=_other_sense(tokens, mentions, kb))
    learned = kb.intent.id.endswith(".learned")
    choices = itertools.islice(
        itertools.product(*(range(len(m.candidates)) for m in mentions)), MAX_HYPOTHESES
    )
    for choice in choices:
        picked = [m.candidates[i] for m, i in zip(mentions, choice)]
        concepts = list(dict.fromkeys(c.concept for c in picked))
        assumptions = [c.assumption for c in picked if c.assumption]
        record = utterance_record(tokens, [kb.concepts[k] for k in concepts])
        tree = kb.intent
        cls = tree.classify(project(tree, record))
        leaf = tree.nodes[cls.leaf] if cls.leaf else None
        if learned and (leaf is None or leaf.meta.get("unknown")):
            # обученное дерево не знает (мало примеров) — догадка по ручному дереву
            tree = kb.hand_intent
            cls = tree.classify(project(tree, record))
            leaf = tree.nodes[cls.leaf] if cls.leaf else None
        # ручное дерево хранит тип в meta.intent или id листа, обученное — в meta.class
        intent = (leaf.meta.get("intent") or leaf.meta.get("class") or leaf.id) if leaf is not None else "NONE"
        label = leaf.label if leaf is not None else ""
        if label == intent or label == "UNKNOWN" or not label:  # у обученного дерева метка листа — сам тип
            label = INTENT_NAMES.get(intent, intent)
        if cls.assumptions:
            if tree is kb.intent and learned:
                assumptions.append(f"тип вопроса ({label}) определён обученным деревом, его вилки ещё не приняты человеком")
            else:
                assumptions.append(f"тип вопроса угадан: {label.lower()}")
        result.hypotheses.append(Hypothesis(intent, concepts, assumptions, label, cls.certificate, choice))
    return result


def _other_sense(tokens: list[Token], mentions: list[Mention], kb: KnowledgeBase) -> list[tuple[str, str]]:
    """Точные совпадения, которые дерево смысла относит не к нашей области.

    Это догадка обученного дерева, поэтому упоминание не выбрасывается: если на вопрос
    можно ответить, ответ получает оговорку; если нельзя — отказ объясняется смыслом слова.
    Догадки об опечатках не проверяются: дерево обучено только на точных совпадениях.
    """
    if kb.sense is None:
        return []
    out: list[tuple[str, str]] = []
    for m in mentions:
        for c in m.candidates:
            if c.typo:
                continue
            record = project(kb.sense, mention_record(tokens, m.start, m.end, c.concept, kb))
            result = kb.sense.classify(record, certify=False)
            leaf = kb.sense.nodes[result.leaf] if result.leaf else None
            if leaf is not None and leaf.meta.get("class") == "other":
                out.append((m.text, c.concept))
    return out


def find_mentions(tokens: list[Token], kb: KnowledgeBase) -> tuple[list[Mention], list[str]]:
    mentions: list[Mention] = []
    unrecognized: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "latin":
            keys = kb.latin_index.get(tok.text) or kb.latin_index.get(tok.text.lower()) or []
            if keys:
                # Короткий символ часто не химия: «F = m*a», «P/S», «Re:». Это допущение.
                short = len(tok.text) <= 2 and tok.text.isalpha()
                mentions.append(Mention(i, i + 1, tok.text, [
                    Candidate(k, f"«{tok.text}» понято как символ «{kb.concepts[k].name}»" if short else "")
                    for k in keys
                ]))
            else:
                unrecognized.append(tok.text)
            i += 1
            continue
        if tok.kind != "word":
            i += 1
            continue
        match = _match_names(tokens, i, kb)
        if match is not None:
            length, keys = match
            text = " ".join(t.text for t in tokens[i : i + length])
            modifier = _modifier(tokens, i, kb)
            if modifier is None:
                candidates = [Candidate(k) for k in keys]
            else:
                # «тёмная материя» — частное понятие; дерево знает только «материю»
                candidates = [
                    Candidate(k, f"«{modifier.text} {text}» понято как «{kb.concepts[k].name}»: "
                                 f"уточнение «{modifier.text}» в дереве не описано")
                    for k in keys
                ]
                if modifier.text in unrecognized:
                    unrecognized.remove(modifier.text)
            mentions.append(Mention(i, i + length, text, candidates))
            i += length
            continue
        if not tok.is_service and not (set(tok.lemmas) & kb.intent_words):
            # Известное словарю существительное или прилагательное — это другое понятие
            # или признак, а не опечатка («борщ» не «бор», «серые» не «сера»). Опечаткой
            # считается незнакомое слово или слово другой части речи, случайно совпавшее
            # со словарём («натри» — повелительное «натереть»).
            known_noun = tok.known and tok.pos[:1] in (("NOUN",), ("ADJF",), ("ADJS",))
            typos = [] if known_noun else _typo_candidates(tok, kb)
            if typos:
                mentions.append(Mention(i, i + 1, tok.text, typos))
            elif known_noun or not tok.known:
                unrecognized.append(tok.text)
        i += 1
    return mentions, unrecognized


def _match_names(tokens: list[Token], i: int, kb: KnowledgeBase) -> tuple[int, list[str]] | None:
    best_len = 0
    keys: list[str] = []
    for pattern, key in kb.patterns:  # отсортированы по убыванию длины
        n = len(pattern)
        if n < best_len:
            break
        window = tokens[i : i + n]
        if len(window) < n or any(t.kind != "word" for t in window):
            continue
        if all(set(t.lemmas) & allowed or normalize(t.text) in allowed for t, allowed in zip(window, pattern)):
            best_len = n
            if key not in keys:
                keys.append(key)
    return (best_len, keys) if keys else None


def _typo_candidates(tok: Token, kb: KnowledgeBase) -> list[Candidate]:
    word = normalize(tok.text)
    limit = 1 if len(word) <= 6 else 2
    found: dict[str, tuple[int, str]] = {}
    for name, keys in kb.single_words.items():
        if abs(len(name) - len(word)) > limit:
            continue
        d = min(levenshtein(word, name, limit), *(levenshtein(l, name, limit) for l in tok.lemmas))
        if 0 < d <= limit:
            for key in keys:
                if key not in found or d < found[key][0]:
                    found[key] = (d, name)
    best = min((d for d, _ in found.values()), default=None)
    return [
        Candidate(key, f"«{tok.text}» понято как «{name}» (возможная опечатка)", typo=True)
        for key, (d, name) in sorted(found.items())
        if d == best
    ]


PRONOMINAL = {"какой", "такой", "этот", "тот", "весь", "каждый", "любой", "свой", "мой", "наш", "ваш",
              "твой", "другой", "сам", "самый", "который", "чей", "некоторый", "иной", "один"}


def _modifier(tokens: list[Token], i: int, kb: KnowledgeBase) -> Token | None:
    """Прилагательное или причастие прямо перед именем понятия, не входящее в имя."""
    if i == 0:
        return None
    prev = tokens[i - 1]
    if prev.kind != "word" or not prev.pos or prev.pos[0] not in ("ADJF", "PRTF"):
        return None
    if set(prev.lemmas) & (PRONOMINAL | kb.intent_words):
        return None
    return prev


def levenshtein(a: str, b: str, limit: int) -> int:
    """Расстояние правки; если оно больше limit, возвращается limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return min(prev[-1], limit + 1)
