"""Extract messages matching a regex with neighbours. Usage: tg_grep.py <chat.txt> <out.txt> <ctx> <regex>"""
import re
import sys

src, out, ctx, pat = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
text = open(src, encoding="utf-8").read()
blocks = [b.strip("\n") for b in text.split("\n\n") if b.strip()]
rx = re.compile(pat, re.I)
hits = [i for i, b in enumerate(blocks) if not b.startswith("[служебное]") and rx.search(b)]
shown = set()
res = []
last = -10
for i in hits:
    lo, hi = max(0, i - ctx), min(len(blocks), i + ctx + 1)
    if lo > last + 1 and res:
        res.append("-----")
    for j in range(max(lo, last + 1), hi):
        mark = ">>" if j == i else "  "
        res.append(f"{mark}{blocks[j]}\n")
    last = max(last, hi - 1)
open(out, "w", encoding="utf-8").write("\n".join(res))
print("hits", len(hits), "chars", sum(len(r) for r in res))
