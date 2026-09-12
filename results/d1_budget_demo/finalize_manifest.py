"""Inventory this isolated delivery including retained failed-attempt evidence."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
output = root / "delivery_manifest.json"
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path != output:
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
manifest = {"schema": "d1-isolated-demo-delivery-v1", "files": files,
    "published_video_candidates": ["comparison.mp4", "zero.mp4", "ppo.mp4", "gui_retry/window.mp4"],
    "retained_failed_attempt": "gui/", "demo_operations_modify_repository_source": False}
with output.open("x") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
    "manifest": str(output)}, indent=2))
