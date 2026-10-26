"""Fetch a phpBB topic (all pages) and dump post text. Usage: fetch_forum.py <viewtopic-url> <out.txt>"""
import html
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

UA = {"User-Agent": "Mozilla/5.0 (research fetch; contact: local user)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return raw.decode("utf-8", errors="replace")


class Posts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.posts = []
        self.depth = 0
        self.mode = None  # 'content' | 'author'
        self.buf = []
        self.cur = {}
        self.stack = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "div" and "postbody" in cls.split():
            self.cur = {"author": "", "date": "", "text": ""}
            self.stack.append(("postbody", 1))
            return
        if self.stack:
            top, d = self.stack[-1]
            if tag == "div":
                if "content" in cls.split() and top == "postbody":
                    self.mode = "content"
                    self.buf = []
                    self.stack.append(("content", 1))
                    return
                self.stack[-1] = (top, d + 1)
            if tag == "p" and "author" in cls.split() and top == "postbody":
                self.mode = "author"
                self.buf = []
            if tag in ("br",) and self.mode == "content":
                self.buf.append("\n")
            if tag in ("script", "style"):
                self.skip += 1
            if tag == "blockquote" and self.mode == "content":
                self.buf.append("\n> ")
            if tag == "img" and self.mode == "content":
                alt = a.get("alt", "")
                src = a.get("src", "")
                if "smilies" not in src:
                    self.buf.append(f"[img {alt or src}]")
            if tag == "a" and self.mode == "content":
                href = a.get("href", "")
                if href.startswith("http"):
                    self.buf.append(f"<{href}> ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if not self.stack:
            return
        top, d = self.stack[-1]
        if tag == "p" and self.mode == "author":
            self.cur["author"] = " ".join("".join(self.buf).split())
            self.mode = None
            self.buf = []
        if tag == "div":
            d -= 1
            if d == 0:
                self.stack.pop()
                if top == "content":
                    self.cur["text"] = re.sub(r"\n{3,}", "\n\n", "".join(self.buf)).strip()
                    self.mode = None
                    self.buf = []
                elif top == "postbody":
                    self.posts.append(self.cur)
                    self.cur = {}
            else:
                self.stack[-1] = (top, d)

    def handle_data(self, data):
        if self.skip:
            return
        if self.mode in ("content", "author"):
            self.buf.append(data)


def main():
    url, out = sys.argv[1], sys.argv[2]
    base = re.sub(r"&start=\d+", "", url)
    all_posts = []
    start = 0
    seen_first = None
    while True:
        page_url = base + (f"&start={start}" if start else "")
        page = get(page_url)
        p = Posts()
        p.feed(page)
        if not p.posts:
            break
        sig = p.posts[0]["text"][:200]
        if seen_first == sig:
            break
        if start == 0:
            seen_first = sig
        all_posts.extend(p.posts)
        title = re.search(r"<title>(.*?)</title>", page, re.S)
        if start == 0 and title:
            print("TITLE:", html.unescape(title.group(1)).strip())
        total = re.search(r"(\d+)\s*(?:сообщени|post)", page)
        if len(p.posts) < 10 and start > 0:
            break
        start += len(p.posts)
        time.sleep(1.0)
        if start > 2000:
            break
    with open(out, "w", encoding="utf-8") as f:
        for i, post in enumerate(all_posts, 1):
            f.write(f"\n===== #{i} | {post['author']} =====\n{post['text']}\n")
    print("posts:", len(all_posts), "chars:", sum(len(x['text']) for x in all_posts))


if __name__ == "__main__":
    main()
