"""Verify logging-only changes and snapshot recovery preserve executed trajectories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("left", type=Path)
parser.add_argument("right", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
comparisons = []
for left in sorted(args.left.glob("*/trajectory.npz")):
    right = args.right / left.parent.name / left.name
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        shared = sorted(set(a.files) & set(b.files))
        for key in shared:
            assert np.array_equal(a[key], b[key]), (left.parent.name, key)
    with left.with_name("metrics.csv").open() as a, right.with_name("metrics.csv").open() as b:
        rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
    assert len(rows_a) == len(rows_b)
    fields = set(rows_a[0]) & set(rows_b[0])
    assert all(all(x[key] == y[key] for key in fields) for x, y in zip(rows_a, rows_b))
    comparisons.append({"case": left.parent.name, "rows": len(rows_a), "common_csv_fields": len(fields), "identical_arrays": shared})
assert len(comparisons) == 12
record = {"all_bitwise_equal": True, "left": str(args.left), "right": str(args.right), "cases": comparisons}
with args.output.open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps({"case_count": len(comparisons), "all_bitwise_equal": True}))
