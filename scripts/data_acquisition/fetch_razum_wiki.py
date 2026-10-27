"""Выгрузка определений из RazumWiki (https://my.razum.wiki, лицензия CC BY-SA).

Определения лежат в двух видах страниц:
- страницы букв («А», «Б», …): записи вида `<mark>Термин</mark> — определение`;
- отдельные страницы терминов («Материя», «Вещество»): первый абзац — определение.

    python scripts/data_acquisition/fetch_razum_wiki.py <raw_dir> <out.json>

raw_dir — каталог для сырых страниц (кэш; повторный запуск не скачивает их заново).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

SITE = "https://my.razum.wiki"
API = f"{SITE}/api.php"
LETTERS = set("АБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ")


def get(url: str, attempts: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DeltaEngine research crawler (one request per 0.2 s)"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:  # обрыв соединения, таймаут: повторить с паузой 2, 4, 8, 16 с
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def all_titles() -> list[str]:
    titles, cont = [], {}
    while True:
        q = {"action": "query", "list": "allpages", "aplimit": "500", "format": "json", **cont}
        data = json.loads(get(f"{API}?{urllib.parse.urlencode(q)}"))
        titles += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data:
            return titles
        cont = data["continue"]


def raw_page(title: str, raw_dir: Path) -> str:
    path = raw_dir / (hashlib.sha1(title.encode()).hexdigest() + ".wiki")
    if not path.exists():
        url = f"{SITE}/index.php?title={urllib.parse.quote(title)}&action=raw"
        path.write_bytes(get(url))
        time.sleep(0.2)
    return path.read_text(encoding="utf-8", errors="replace")


def clean(text: str) -> str:
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[ссылка|текст]] → текст
    text = re.sub(r"'''?", "", text)
    text = text.replace("́", "")  # знак ударения
    return re.sub(r"\s+", " ", text).strip()


_MARK = re.compile(r"<mark>(.*?)</mark>\s*(.*?)(?=\n\s*\n|\n?<mark>|\Z)", re.S)
_DASH = re.compile(r"^\s*(?:\([^)]*\)\s*)?[-—–,:]\s*")


def parse_letter_page(title: str, text: str) -> list[dict]:
    out = []
    for term, body in _MARK.findall(text):
        term, body = clean(term), clean(body)
        body = _DASH.sub("", body, count=1) if _DASH.match(body) else body
        if term and body:
            out.append({"term": term, "definition": body, "page": title, "kind": "letter_page"})
    return out


def parse_term_page(title: str, text: str) -> dict | None:
    if text.lstrip().startswith("#REDIRECT") or text.lstrip().startswith("#перенаправление"):
        return None
    first = next((p for p in re.split(r"\n\s*\n", text) if clean(p) and not p.lstrip().startswith(("{|", "[[Категория", "=="))), "")
    body = clean(first)
    m = re.match(r"^(.{1,120}?)\s+[-—–]\s+(.+)$", body)
    if m and clean(m.group(1)).lower().startswith(title.lower()[:4]):
        body = m.group(2)
    if not body:
        return None
    return {"term": title, "definition": body, "page": title, "kind": "term_page"}


def main(raw_dir: Path, out_path: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    titles = all_titles()
    entries: list[dict] = []
    digest = hashlib.sha256()
    for i, title in enumerate(titles):
        text = raw_page(title, raw_dir)
        digest.update(f"{title}:{hashlib.sha256(text.encode()).hexdigest()}\n".encode())
        if title in LETTERS:
            entries += parse_letter_page(title, text)
        else:
            entry = parse_term_page(title, text)
            if entry:
                entries.append(entry)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(titles)}", file=sys.stderr)
    for e in entries:
        e["url"] = f"{SITE}/index.php?title={urllib.parse.quote(e['page'])}"
    result = {
        "format": "razum-wiki-terms@0.1",
        "source": SITE,
        "license": "CC BY-SA (RazumWiki, siteinfo rightsinfo)",
        "fetched": date.today().isoformat(),
        "pages": len(titles),
        "aggregate_sha256": digest.hexdigest(),
        "hash_method": "sha256 over lines 'title:sha256(raw wikitext)' in API order",
        "entries": entries,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"страниц {len(titles)}, определений {len(entries)} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
