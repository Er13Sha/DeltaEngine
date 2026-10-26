import glob, hashlib, json, os, sys

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "messages*.html")), key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
agg = hashlib.sha256()
total = 0
for f in files:
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    agg.update(os.path.basename(f).encode() + b":" + h.encode() + b"\n")
    total += os.path.getsize(f)
print(json.dumps({"files": len(files), "bytes": total, "aggregate_sha256": agg.hexdigest()}))
