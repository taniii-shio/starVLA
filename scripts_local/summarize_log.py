"""学習ログの action_dit_loss / world_loss を step 窓ごとに平均して表にする。標準ライブラリのみ。VM でも手元でも動く。

  python3 scripts_local/summarize_log.py /data/logs/A_short_detach.log [他のログ ...] [--bin 250]

複数のログを渡すと、同じ窓を横に並べる (A / B / C の比較用)。最初の 100 step は桁違いに大きい値があるので除く。
"""

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

STEP_RE = re.compile(r"Step (\d+), Loss: \{")
PAIR_RE = re.compile(r"'([A-Za-z_][\w/.]*)':\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)")


def parse(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="replace"))
    marks = list(STEP_RE.finditer(text))
    rows = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.end(): min(end, m.end() + 2000)]
        close = chunk.find("}")
        chunk = chunk[:close] if close >= 0 else chunk
        d = {k: float(v) for k, v in PAIR_RE.findall(chunk)}
        d["step"] = int(m.group(1))
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--bin", type=int, default=250)
    args = ap.parse_args()
    tables = {}
    for p in args.logs:
        acc = defaultdict(lambda: defaultdict(list))
        for r in parse(p):
            if r["step"] <= 100:
                continue
            b = (r["step"] - 1) // args.bin
            for k in ("action_dit_loss", "world_loss"):
                if k in r:
                    acc[k][b].append(r[k])
        tables[Path(p).stem] = acc
    names = list(tables)
    for key in ("action_dit_loss", "world_loss"):
        if not any(tables[n][key] for n in names):
            continue
        print(f"\n## {key} ({args.bin} step 窓の平均)")
        print("step 窓".ljust(14) + "".join(n[:22].rjust(24) for n in names))
        bins = sorted({b for n in names for b in tables[n][key]})
        for b in bins:
            cells = []
            for n in names:
                v = tables[n][key].get(b)
                cells.append(f"{st.mean(v):.4f} (n={len(v)})".rjust(24) if v else "-".rjust(24))
            print(f"{b * args.bin + 1}-{(b + 1) * args.bin}".ljust(14) + "".join(cells))


if __name__ == "__main__":
    main()
