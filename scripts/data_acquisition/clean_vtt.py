import re, sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

ts_re = re.compile(r"(\d+):(\d+):(\d+)\.\d+ --> ")
tag_re = re.compile(r"<[^>]+>")

out = []
prev = ""
last_stamp = -999
cur_ts = 0
for block in text.split("\n\n"):
    lines = [l for l in block.strip().split("\n") if l.strip()]
    if not lines:
        continue
    m = None
    body = []
    for l in lines:
        mm = ts_re.match(l)
        if mm:
            m = mm
            continue
        if l.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        body.append(tag_re.sub("", l).strip())
    if m is None:
        continue
    cur_ts = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    for l in body:
        if not l or l == prev:
            continue
        prev = l
        if cur_ts - last_stamp >= 30:
            out.append(f"\n[{cur_ts // 60:02d}:{cur_ts % 60:02d}] ")
            last_stamp = cur_ts
        out.append(l + " ")

open(dst, "w", encoding="utf-8").write("".join(out))
print("chars:", sum(len(x) for x in out))
