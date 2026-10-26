import collections
import re
import sys

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
blocks = [b for b in text.split("\n\n") if b.strip()]
msgs = []
for b in blocks:
    head, _, body = b.partition("\n")
    m = re.match(r"\[(\d\d)\.(\d\d)\.(\d{4}) [^\]]*\]\s*(.*)", head)
    if m:
        d, mo, y, who = m.groups()
        who = who.replace(" (ответ)", "").strip()
        msgs.append((y, mo, who, body))
    elif head.startswith("[служебное]"):
        msgs.append(("svc", "", "", head + " " + body))
lines = []
real = [m for m in msgs if m[0] != "svc"]
lines.append(f"messages {len(real)}")
lines.append("by year: " + str(sorted(collections.Counter(m[0] for m in real).items())))
lines.append("authors: " + str(collections.Counter(m[2] for m in real).most_common(40)))
lines.append("distinct authors: " + str(len(set(m[2] for m in real))))
bymonth = collections.Counter(f"{m[0]}-{m[1]}" for m in real)
lines.append("top months: " + str(bymonth.most_common(10)))
svc = [m[3] for m in msgs if m[0] == "svc"]
topics = [s for s in svc if "topic" in s.lower()]
lines.append(f"service msgs {len(svc)}; topic-related {len(topics)}")
for t in topics[:80]:
    lines.append("  " + t[:160])
pats = {
    "дельт": r"дельт",
    "вилк": r"вилк|дихотом",
    "ИИ/искусств.интеллект": r"искусственн\w+ интеллект|\bИИ\b|\bAI\b",
    "нейросет": r"нейро",
    "GPT/LLM": r"GPT|ЧатЖПТ|чат ?жпт|\bLLM\b|языков\w+ модел|Claude|Gemini|Grok|DeepSeek",
    "развёртка": r"развёртк|развертк",
    "функц-характ": r"функци\w+[- ]характеристик",
    "определени": r"определени",
    "Катющик": r"Катющик",
    "алгоритм": r"алгоритм",
    "программ": r"программ",
}
for k, p in pats.items():
    n = sum(1 for m in real if re.search(p, m[3], re.I))
    lines.append(f"{k}: {n}")
open(out, "w", encoding="utf-8").write("\n".join(lines))
