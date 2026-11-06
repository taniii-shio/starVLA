"""GRAD_LOG=true で出力した grad_log.jsonl を step 窓ごとに集計する。標準ライブラリのみ。VM でも手元でも動く。

  python3 scripts_local/summarize_gradnorm.py /data/checkpoints/A_g/grad_log.jsonl /data/checkpoints/C_g/grad_log.jsonl [--bin 250]

列の意味 (analysis の「検証の判定」表に対応):
  clip%      クリップ係数が 1 未満 (= 全体ノルムが上限を超えた) step の割合
  coef       クリップ係数の平均 / 標準偏差の比 (cv = 標準偏差 / 平均。大きいほど step ごとにぶれる)
  head_share ヘッドの寄与 = head_norm^2 / total_norm^2 の平均
  total, head, rest  ノルムの中央値 (rest = ヘッド以外 = sqrt(total^2 - head^2))
最初の 100 step は桁違いに大きいので除く。
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(rows, bin_size):
    acc = defaultdict(list)
    for r in rows:
        if r["step"] > 100:
            acc[(r["step"] - 1) // bin_size].append(r)
    out = []
    for b, rs in sorted(acc.items()):
        coef = [r["clip_coef"] for r in rs]
        share = [r["head_norm"] ** 2 / r["total_norm"] ** 2 for r in rs if r["total_norm"] > 0]
        out.append({
            "win": f"{b * bin_size + 1}-{(b + 1) * bin_size}",
            "n": len(rs),
            "clip%": 100 * sum(c < 0.999 for c in coef) / len(coef),
            "coef": st.mean(coef),
            "cv": (st.pstdev(coef) / st.mean(coef)) if st.mean(coef) > 0 else float("nan"),
            "share": st.mean(share) if share else float("nan"),
            "total": st.median(r["total_norm"] for r in rs),
            "head": st.median(r["head_norm"] for r in rs),
            "rest": st.median(r["rest_norm"] for r in rs),
            "act": st.mean(r["action_loss"] for r in rs),
        })
    return out


def sanity(name, rows):
    bad = [r["step"] for r in rows if r["head_norm"] > r["total_norm"] * 1.02 + 1e-6]
    nan = [r["step"] for r in rows if r["total_norm"] != r["total_norm"]]
    print(f"[{name}] 行数={len(rows)}  head>total の step={len(bad)} {bad[:5]}  total が NaN の step={len(nan)}")
    if bad:
        print("  -> head_norm が total_norm を超えている。フックの積算か get_global_grad_norm の値が想定と違う")
    if nan:
        print("  -> get_global_grad_norm が None。全体ノルムが取れていない")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--bin", type=int, default=250)
    args = ap.parse_args()
    for p in args.logs:
        name = Path(p).parent.name
        rows = load(p)
        sanity(name, rows)
        print(f"\n## {name}")
        print("window".ljust(11) + "n".rjust(5) + "clip%".rjust(8) + "coef".rjust(8) + "cv".rjust(7)
              + "head_share".rjust(12) + "total".rjust(9) + "head".rjust(9) + "rest".rjust(9) + "act_loss".rjust(10))
        for s in summarize(rows, args.bin):
            print(s["win"].ljust(11) + str(s["n"]).rjust(5) + f"{s['clip%']:.0f}".rjust(8) + f"{s['coef']:.3f}".rjust(8)
                  + f"{s['cv']:.2f}".rjust(7) + f"{s['share']:.3f}".rjust(12) + f"{s['total']:.2f}".rjust(9)
                  + f"{s['head']:.3f}".rjust(9) + f"{s['rest']:.2f}".rjust(9) + f"{s['act']:.3f}".rjust(10))


if __name__ == "__main__":
    main()
