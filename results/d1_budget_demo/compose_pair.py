"""Put the two state-aligned development recordings side by side without trimming."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


first, second = (json.loads((HERE / f"{name}.json").read_text()) for name in ("zero", "ppo"))
assert first["frame_state_indices"] == second["frame_state_indices"]
assert first["fps"] == second["fps"] == 20
assert first["frames"] == second["frames"] == 1201
assert first["simulation_duration_s"] == second["simulation_duration_s"]
assert first["compiled_model_sha256"] == second["compiled_model_sha256"]
command = ["rtk", "proxy", "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
    "-threads", "1", "-i", str(HERE / "zero.mp4"),
    "-threads", "1", "-i", str(HERE / "ppo.mp4"),
    "-filter_complex_threads", "1", "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
    "-map", "[v]", "-an", "-threads", "1", "-c:v", "libx264", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(HERE / "comparison.mp4")]
subprocess.run(command, check=True)
report = {"layout": "left zero residual; right predetermined seed31000 budget262144 PPO",
    "command": command, "input_sha256": {f"{name}.mp4": sha(HERE / f"{name}.mp4")
        for name in ("zero", "ppo")}, "video_sha256": sha(HERE / "comparison.mp4"),
    "frames": 1201, "fps": 20, "playback_duration_s": 60.05,
    "all_frame_state_indices_match": True, "frame_0_is_reset": True,
    "last_frame_is_terminal": True, "physics_resimulation_in_composition": False,
    "claim_limit": "One predetermined development case; full 60s, no selection or trimming"}
with (HERE / "comparison.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
subprocess.run(["rtk", "proxy", "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
    "-threads", "1", "-ss", "30", "-i", str(HERE / "comparison.mp4"),
    "-frames:v", "1", str(HERE / "comparison.png")], check=True)
print(json.dumps(report, indent=2))
