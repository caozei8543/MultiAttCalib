"""Print a Markdown comparison from actual evaluation summaries; never embeds paper results."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",action="append",required=True,help="LABEL=path/to/eval-directory")
    a=p.parse_args()
    rows=[]
    digest=None
    for item in a.result:
        label,path=item.split("=",1)
        path=Path(path)
        metrics=json.loads((path/"summary.json").read_text())
        if (path/"evaluation.json").is_file():
            meta=json.loads((path/"evaluation.json").read_text())
            key=(meta["manifest_sha256"],meta["trials_per_pair"])
            if digest is not None and digest!=key:
                raise ValueError("Evaluation manifest or trial definition differs between runs")
            digest=key
        rows.append((label,metrics))
    if len({(r["n"],r["rotation_threshold_deg"],r["translation_threshold_cm"]) for _,r in rows})!=1:
        raise ValueError("Trial count or thresholds differ")
    print("| Method | Rotation MAE (deg) | Translation MAE (cm) | Joint success (%) |")
    print("|---|---:|---:|---:|")
    for label,r in rows:
        print(f"| {label} | {r['rotation_mae_deg']:.2f} | {r['translation_mae_cm']:.2f} | {r['joint_success_percent']:.1f} |")


if __name__=="__main__":
    main()
