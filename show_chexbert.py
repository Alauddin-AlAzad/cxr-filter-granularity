"""Print every run in chexbert.json on one line, plus the comparison names.
Run:  python show_chexbert.py
Only numbers are printed - safe to paste in chat."""
import json

d = json.load(open("chexbert.json"))

print(f"{'run':24s} {'F1-14':>7s} {'95% CI':>17s} {'P':>7s} {'R':>7s} {'F1-top5':>8s}  n")
for name, v in d["summaries"].items():
    a = v.get("all14", {})
    t = v.get("top5", {})
    print(f"{name:24s} {a.get('f1', float('nan')):7.4f} "
          f"[{a.get('lo', float('nan')):.4f},{a.get('hi', float('nan')):.4f}] "
          f"{a.get('precision', float('nan')):7.4f} {a.get('recall', float('nan')):7.4f} "
          f"{t.get('f1', float('nan')):8.4f}  {v.get('n', '?')}")

print("\ncomparisons:")
for name, c in d["comparisons"].items():
    for lab, r in c.items():
        print(f"  {name:32s} {lab:9s} diff {r['diff']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}]  distinguishable: {r['distinguishable']}")