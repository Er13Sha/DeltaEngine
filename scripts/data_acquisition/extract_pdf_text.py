import sys, pypdf
r = pypdf.PdfReader(sys.argv[1])
print("PAGES", len(r.pages))
out = []
for i, p in enumerate(r.pages):
    out.append(f"--- page {i+1} ---\n" + (p.extract_text() or ""))
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
