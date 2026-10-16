import glob, json
from collections import defaultdict

f = sorted(glob.glob("results/*_free_*.jsonl"))[-1]
R = [json.loads(l) for l in open(f, encoding="utf-8")]
T = {t["id"]: t for t in json.load(open("tasks/tasks.json", encoding="utf-8"))}
print("file:", f)

by_h = defaultdict(list)
for r in R:
    by_h[r["task"]["horizon"]].append(r["score"]["strict_success"])
for h in sorted(by_h):
    v = by_h[h]
    print(f"horizon {h}: success {sum(v)}/{len(v)}")

print("\n[calendar tasks: gold vs model's create_event]")
for r in R:
    t = T[r["task"]["id"]]
    need = t["success_condition"].get("calendar_contains")
    if not need or t["horizon"] == 1:
        continue
    ev = [e["action"]["arguments"] for e in r["trace"]["events"]
          if (e.get("action") or {}).get("name") == "create_event"]
    got = ev[-1] if ev else None
    ok = r["score"]["strict_success"]
    print(f"{t['id']:16s} {'O' if ok else 'X'} gold time={need['time']} title={need['title']} | "
          f"model={None if not got else (got.get('time'), got.get('title'), got.get('location'), got.get('date'))}")