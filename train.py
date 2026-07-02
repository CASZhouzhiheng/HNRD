import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hnrd.data import SUPPORTED_DATASETS, load_dataset
from hnrd.models import HNRD
from hnrd.operators import IncidenceOperator, edge_list_to_index
from hnrd.utils import balanced_split, fix_seed


def accuracy(logits, labels, index):
    pred = logits[index].argmax(dim=-1)
    return float((pred == labels[index]).float().mean().item() * 100.0)


@torch.no_grad()
def evaluate(model, x, operator, labels, split):
    model.eval()
    logits = model(x, operator)
    return {name: accuracy(logits, labels, idx) for name, idx in split.items()}


def train_once(args, seed):
    fix_seed(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = load_dataset(args.dataset, args.data_root)
    x = data["features"].float().to(device)
    labels = data["labels"].long().view(-1).to(device)
    edge_index = edge_list_to_index(data["edge_list"], data["num_nodes"], device)
    operator = IncidenceOperator.from_edge_index(edge_index, data["num_nodes"], device)
    split_cpu = balanced_split(data["labels"], args.train_prop, args.valid_prop)
    split = {key: value.to(device) for key, value in split_cpu.items()}

    model = HNRD(
        in_dim=x.shape[1],
        hidden_dim=args.hidden,
        out_dim=data["num_classes"],
        layers=args.layers,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        step_size=args.step_size,
        learnable_step=args.learnable_step,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"valid": -1.0, "test": -1.0, "epoch": -1}
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, operator)
        loss = F.cross_entropy(logits[split["train"]], labels[split["train"]])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            scores = evaluate(model, x, operator, labels, split)
            if scores["valid"] >= best["valid"]:
                best = {"valid": scores["valid"], "test": scores["test"], "epoch": epoch + 1}
            print(
                f"seed={seed:02d} epoch={epoch + 1:03d} "
                f"valid={scores['valid']:.2f} test={scores['test']:.2f} "
                f"best_test={best['test']:.2f}",
                flush=True,
            )
    return best


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="HNRD")
    parser.add_argument("--config", default="configs")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--input-dropout", type=float, default=None)
    parser.add_argument("--hidden-dropout", type=float, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--learnable-step", action="store_true", default=None)
    parser.add_argument("--fixed-step", action="store_false", dest="learnable_step")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-prop", type=float, default=None)
    parser.add_argument("--valid-prop", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = None
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, cfg.get("learning_rate" if key == "lr" else key, value))
    return args


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path("runs") / args.dataset.lower().replace("-", "_"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        best = train_once(args, seed)
        rows.append({"seed": seed, "valid": best["valid"], "test": best["test"], "epoch": best["epoch"]})

    tests = np.asarray([row["test"] for row in rows], dtype=np.float64)
    summary = {
        "dataset": args.dataset,
        "model": "HNRD",
        "num_seeds": len(rows),
        "test_mean": float(tests.mean()),
        "test_std": float(tests.std(ddof=1) if len(tests) > 1 else 0.0),
    }

    dataset_slug = args.dataset.lower().replace("-", "_")
    raw_path = out_dir / f"{dataset_slug}_hnrd_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "valid", "test", "epoch"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{dataset_slug}_hnrd_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"HNRD {args.dataset}: test={summary['test_mean']:.2f} +- {summary['test_std']:.2f} "
        f"over {summary['num_seeds']} seeds",
        flush=True,
    )
    print(f"saved={raw_path}", flush=True)


if __name__ == "__main__":
    main()
