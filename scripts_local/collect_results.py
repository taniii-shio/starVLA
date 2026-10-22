"""学習ログと評価結果を 1 つの JSON にまとめ、成功動画を 1 つの tar にする。VM で実行する (標準ライブラリのみ)。

  python scripts_local/collect_results.py --out /data/export --max_videos_per_arm 10

出力:
  <out>/results.json   {"train": {arm: {"keys": [...], "steps": [...], <key>: [...]}},
                        "eval":  {arm: {suite: {"tasks": {task: {"success": n, "total": n}}, "success": n, "total": n}}}}
  <out>/videos/<arm>/<suite>/*.mp4   成功動画 (アームごとに最大 N 本。タスクが偏らないよう、タスクを巡回して選ぶ)
  <out>/export.tar                   上の全部 (Mac への転送用)

評価の成否は動画のファイル名 (rollout_<task>_episode<i>_success|failure.mp4) から数える。
"""

import argparse
import json
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

ARMS = {"B": "B_s0_5k", "A": "A_s0_5k", "C": "C_s0_5k"}
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
VIDEO_RE = re.compile(r"^rollout_(?P<task>.+)_episode(?P<ep>\d+)_(?P<res>success|failure)\.mp4$")
STEP_RE = re.compile(r"Step (\d+), Loss: \{")
PAIR_RE = re.compile(r"'([A-Za-z_][\w/.]*)':\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_train_log(path: Path) -> dict:
    """'Step N, Loss: {...}' を拾う。rich の折り返しで行が割れても取れるよう、次の 'Step' までを 1 塊として読む。"""
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    marks = list(STEP_RE.finditer(text))
    rows = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.end() : min(end, m.end() + 2000)]
        close = chunk.find("}")
        if close >= 0:
            chunk = chunk[:close]
        rows[int(m.group(1))] = {k: float(v) for k, v in PAIR_RE.findall(chunk)}
    steps = sorted(rows)
    keys = sorted({k for r in rows.values() for k in r})
    out = {"keys": keys, "steps": steps}
    for k in keys:
        out[k] = [rows[s].get(k) for s in steps]  # 欠けた行は null
    return out


def scan_videos(video_dir: Path):
    # 同じ (task, episode) で success と failure が両方ある場合は、smoke の古い動画が残っている。更新時刻が新しい方を採る
    latest = {}
    for p in video_dir.glob("*.mp4"):
        m = VIDEO_RE.match(p.name)
        if not m:
            continue
        key = (m["task"], int(m["ep"]))
        if key not in latest or p.stat().st_mtime > latest[key][1].stat().st_mtime:
            latest[key] = (m["res"], p)
    tasks = defaultdict(lambda: {"success": 0, "total": 0})
    successes = []
    for (task, _), (res, p) in sorted(latest.items()):
        tasks[task]["total"] += 1
        if res == "success":
            tasks[task]["success"] += 1
            successes.append((task, p))
    return dict(tasks), successes


def pick_round_robin(successes, n):
    """タスクを巡回して n 本選ぶ (同じタスクに偏らない)。"""
    by_task = defaultdict(list)
    for task, p in successes:
        by_task[task].append(p)
    picked, queues = [], [list(v) for _, v in sorted(by_task.items())]
    while queues and len(picked) < n:
        for q in list(queues):
            if len(picked) >= n:
                break
            picked.append(q.pop(0))
            if not q:
                queues.remove(q)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/export")
    ap.add_argument("--logs", default="/data/logs")
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--max_videos_per_arm", type=int, default=10)
    ap.add_argument("--ckpt_tag", default="checkpoints_steps_5000_pytorch_model.pt")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "videos").exists():
        shutil.rmtree(out / "videos")
    out.mkdir(parents=True, exist_ok=True)

    data = {"train": {}, "eval": {}, "ckpt_tag": args.ckpt_tag}
    for arm, run in ARMS.items():
        log = Path(args.logs) / f"{run}.log"
        if log.exists():
            data["train"][arm] = parse_train_log(log)
            tr = data["train"][arm]
            print(f"train {arm}: {len(tr['steps'])} 行, keys={tr['keys']}")
        else:
            print(f"train {arm}: ログなし ({log})")

        data["eval"][arm] = {}
        all_success = []
        for suite in SUITES:
            vdir = Path(args.results) / suite / f"{run}_{args.ckpt_tag}"
            if not vdir.is_dir():
                continue
            tasks, succ = scan_videos(vdir)
            s = sum(t["success"] for t in tasks.values())
            n = sum(t["total"] for t in tasks.values())
            data["eval"][arm][suite] = {"tasks": tasks, "success": s, "total": n}
            all_success += [(suite, task, p) for task, p in succ]
            print(f"eval {arm} {suite}: {s}/{n} ({len(tasks)} タスク)")

        # 成功動画: スイートを巡回しつつ、タスクが偏らないように選ぶ
        by_suite = defaultdict(list)
        for suite, task, p in all_success:
            by_suite[suite].append((task, p))
        budget, picked = args.max_videos_per_arm, []
        per_suite = max(1, budget // max(1, len(by_suite))) if by_suite else 0
        for suite, items in by_suite.items():
            picked += [(suite, p) for p in pick_round_robin(items, per_suite)]
        picked = picked[:budget]
        for suite, p in picked:
            dst = out / "videos" / arm / suite
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        print(f"videos {arm}: 成功 {len(all_success)} 本中 {len(picked)} 本を選択")

    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False))
    tar_path = out / "export.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(out / "results.json", arcname="results.json")
        if (out / "videos").exists():
            tf.add(out / "videos", arcname="videos")
    print(f"\n{tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
