"""Диалог с Delta LM в терминале.

    python -m delta_app.chat "Чем жидкость отличается от газа?"
    python -m delta_app.chat            # интерактивный режим
    python -m delta_app.chat --trace "натри"   # показать гипотезы и сертификаты
"""
from __future__ import annotations

import argparse
import json
import sys

from delta_text.dialog import Reply, respond
from delta_text.knowledge import KnowledgeBase, default_knowledge
from delta_text.learn_intent import LEARNED

STATUS = {
    "proven": "доказано ядром",
    "assumed": "с допущениями",
    "quoted": "цитата из словаря, ядром не проверено",
    "unknown": "нет ответа",
    "clarify": "нужно уточнение",
}


def show(reply: Reply, trace: bool) -> None:
    print(reply.text)
    print(f"[{STATUS[reply.status]}]")
    if trace:
        print(json.dumps({"trace": reply.trace, "certificates": reply.certificates}, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delta LM: вопросы о материи и химических элементах")
    parser.add_argument("question", nargs="*", help="вопрос; без него — интерактивный режим")
    parser.add_argument("--trace", action="store_true", help="показать гипотезы, план и сертификаты")
    parser.add_argument("--learned", action="store_true",
                        help="тип вопроса определять деревом, обученным индуктором (data/trees/intent.learned.json)")
    args = parser.parse_args(argv)
    kb = KnowledgeBase(intent_tree=LEARNED) if args.learned else default_knowledge()
    if args.question:
        show(respond(" ".join(args.question), kb), args.trace)
        return 0
    print("Delta LM. Спросите о материи или химических элементах. Пустая строка — выход.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        show(respond(line, kb), args.trace)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
