"""Вопросы из экспорта Telegram-чата — без имён участников (для обучения понимания).

Запускается у владельца экспорта (сырой экспорт содержит имена людей и в репозиторий
не попадает):

    python scripts/data_acquisition/tg_questions.py C:\\IT\\ML\\DeltaEngine\\ChatExport_2026-10-27 data/corpora/telegram_questions.raw.json

В файл попадают только вопросительные предложения (оканчиваются на «?») длиной
10–200 символов: текст, дата и номер сообщения. Имена, ники, ссылки и упоминания
@… удаляются. Файл затем размечается так же, как mailru_questions.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parse_tg import TG  # noqa: E402

_URL = re.compile(r"<?https?://\S+>?")
_MENTION = re.compile(r"@\w+")


def questions(export_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(export_dir, "messages*.html")),
                   key=lambda p: int(re.search(r"messages(\d*)\.html", os.path.basename(p)).group(1) or 1))
    out, seen = [], set()
    for path in files:
        parser = TG()
        parser.feed(open(path, encoding="utf-8", errors="replace").read())
        for m in parser.msgs:
            if m["service"] or not m["text"]:
                continue
            text = _MENTION.sub("", _URL.sub("", m["text"]))
            for sentence in re.split(r"(?<=[?!.])\s+", " ".join(text.split())):
                sentence = sentence.strip(" -–—…»«\"")
                if not sentence.endswith("?") or not 10 <= len(sentence) <= 200:
                    continue
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({"text": sentence, "date": m["date"][:10], "message": m["id"]})
    return out


def main() -> int:
    export_dir, out = sys.argv[1], sys.argv[2]
    rows = questions(export_dir)
    doc = {
        "format": "telegram-questions@0.1",
        "source": "Telegram-чат «РАЗУМНАЯ ЦИВИЛИЗАЦИЯ: Научный Дискуссионный Клуб» (экспорт владельца)",
        "privacy": "только тексты вопросов; имена, ники, ссылки и упоминания удалены",
        "questions": rows,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"вопросов: {len(rows)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
