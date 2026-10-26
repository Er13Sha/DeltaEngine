"""List topics of phpBB forums. Usage: list_forum.py <out.tsv> <forum_id> [<forum_id> ...]"""
import html
import re
import sys
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research fetch)"}
BASE = "https://forum.razum.wiki/"


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


row_re = re.compile(r'<a href="\./viewtopic\.php\?f=(\d+)&amp;(?:sid=\w+&amp;)?t=(\d+)[^"]*"[^>]*class="topictitle"[^>]*>(.*?)</a>', re.S)
replies_re = re.compile(r'<dd class="posts">\s*(\d+)', re.S)


def main():
    out = sys.argv[1]
    fids = sys.argv[2:]
    rows = []
    for fid in fids:
        start = 0
        seen = set()
        while True:
            url = f"{BASE}viewforum.php?f={fid}" + (f"&start={start}" if start else "")
            page = get(url)
            found = row_re.findall(page)
            replies = replies_re.findall(page)
            new = 0
            for i, (f, t, title) in enumerate(found):
                if t in seen:
                    continue
                seen.add(t)
                new += 1
                rep = replies[i] if i < len(replies) else "?"
                rows.append((f, t, rep, html.unescape(re.sub(r"<[^>]+>", "", title)).strip()))
            if new == 0:
                break
            start += 25
            time.sleep(0.8)
            if start > 800:
                break
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("topics:", len(rows))


if __name__ == "__main__":
    main()
