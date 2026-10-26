"""Parse a Telegram Desktop HTML export into plain text. Usage: parse_tg.py <export_dir> <out.txt>"""
import glob
import os
import re
import sys
from html.parser import HTMLParser


class TG(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.msgs = []
        self.cur = None
        self.msg_depth = 0
        self.field = None
        self.field_depth = 0
        self.buf = []
        self.last_from = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div" and "message" in cls and "default" in cls or (tag == "div" and "message" in cls and "service" in cls):
            self.cur = {"id": a.get("id", ""), "from": "", "date": "", "text": "", "reply": "", "media": [], "joined": "joined" in cls, "service": "service" in cls, "fwd": ""}
            self.msg_depth = 1
            self.field = None
            return
        if self.cur is None:
            return
        if tag == "div":
            self.msg_depth += 1
            if self.field is None:
                if self.cur["service"] and "details" in cls and "body" in cls:
                    self.field, self.field_depth, self.buf = "text", 1, []
                    return
                if "from_name" in cls:
                    self.field, self.field_depth, self.buf = "from", 1, []
                    return
                if "text" in cls and "forwarded" not in cls:
                    self.field, self.field_depth, self.buf = "text", 1, []
                    return
                if "reply_to" in cls:
                    self.field, self.field_depth, self.buf = "reply", 1, []
                    return
                if "date" in cls and "details" in cls:
                    self.cur["date"] = a.get("title", "")
                if "from_name" in cls:
                    pass
                if "title" in cls and "bold" in cls:
                    self.field, self.field_depth, self.buf = "media", 1, []
                    return
                if "description" in cls:
                    self.field, self.field_depth, self.buf = "media", 1, []
                    return
            else:
                self.field_depth += 1
        elif self.field is not None:
            if tag == "br":
                self.buf.append("\n")
            if tag == "a" and self.field == "text":
                href = a.get("href", "")
                if href.startswith("http"):
                    self.buf.append(f"<{href}> ")
            if tag == "a" and self.field == "reply":
                self.cur["reply"] = a.get("href", "")
        if tag == "a" and self.field is None:
            href = a.get("href", "")
            if "photos/" in href or "files/" in href or "video_files/" in href or "voice_messages/" in href:
                self.cur["media"].append(href)

    def handle_endtag(self, tag):
        if self.cur is None:
            return
        if tag == "div":
            if self.field is not None:
                self.field_depth -= 1
                if self.field_depth == 0:
                    txt = "".join(self.buf).strip()
                    if self.field == "media":
                        self.cur["media"].append(txt)
                    elif self.field == "text":
                        self.cur["text"] = (self.cur["text"] + "\n" + txt).strip() if self.cur["text"] else txt
                    elif self.field == "reply":
                        self.cur["reply"] = (txt or self.cur["reply"])
                    else:
                        self.cur[self.field] = txt
                    self.field = None
                    self.buf = []
            self.msg_depth -= 1
            if self.msg_depth == 0:
                self.msgs.append(self.cur)
                self.cur = None

    def handle_data(self, data):
        if self.field is not None:
            self.buf.append(data)


def main():
    d, out = sys.argv[1], sys.argv[2]
    files = glob.glob(os.path.join(d, "messages*.html"))

    def key(p):
        m = re.search(r"messages(\d*)\.html", os.path.basename(p))
        return int(m.group(1) or 1)

    files.sort(key=key)
    allm = []
    last_from = ""
    for f in files:
        p = TG()
        p.feed(open(f, encoding="utf-8", errors="replace").read())
        for m in p.msgs:
            if m["joined"] and not m["from"]:
                m["from"] = last_from
            elif m["from"]:
                last_from = m["from"]
            allm.append(m)
    with open(out, "w", encoding="utf-8") as fh:
        for m in allm:
            media = "; ".join(x for x in m["media"] if x)
            line = f"[{m['date']}] {m['from']}"
            if m["service"]:
                line = f"[служебное] {m['date']}"
            elif m["reply"]:
                line += " (ответ)"
            fh.write(line + "\n")
            if m["text"]:
                fh.write(m["text"] + "\n")
            if media:
                fh.write(f"[медиа: {media}]\n")
            fh.write("\n")
    print("files", len(files), "messages", len(allm), "chars", sum(len(m['text']) for m in allm))


if __name__ == "__main__":
    main()
