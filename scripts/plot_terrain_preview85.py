"""Plot the two round-85 probes from their independently verified recording."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def labels(cases):
    return [name.replace("up_", "上坡  ").replace("down_", "下坡  ") for name in cases]


def plot(audit, output):
    if audit["audit"] != "PASS":
        raise ValueError("a verified readback is required")
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    choices = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "PingFang SC")
    font = next((name for name in choices if name in fonts), None)
    if font is None:
        raise RuntimeError("install a Chinese font such as Noto Sans CJK to render the labels")
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.23, top=0.78, wspace=0.52)
    teal, slate, red = "#00796b", "#687887", "#a83e2c"

    sensing = audit["heldout"]
    for y, case in enumerate(sensing.values()):
        lead = [case["first_front_contact_s"] - case["mounts"][a]["first_confirmation_s"]
                for a in ("74", "82")]
        left.plot(lead, [y, y], "-", color="#a6b8bb", lw=1.8)
        left.plot(lead[0], y, "o", mfc="white", mec=slate, mew=1.8, label="74°" if y == 0 else None)
        left.plot(lead[1], y, "o", color=teal, label="82°" if y == 0 else None)
    left.set(xlim=(0, 2.55), xticks=[0, 0.5, 1, 1.5, 2, 2.5],
             xlabel="正确坡向确认至前轮触坡的时间（秒）")
    left.set_title("预瞄能提前多久？", loc="left", weight="bold", pad=29)
    left.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
                borderaxespad=0, handlelength=1.0, columnspacing=1.4)

    actions = audit["development"]
    for y, case in enumerate(actions.values()):
        delta = case["first_pos_neg_applied_torque_linf_nm"]
        separated = case["checks"]["first_applied_torques_separate"]
        color = teal if separated else red
        right.hlines(y, 0, delta, color="#b8c8c7", lw=2)
        right.plot(delta, y, "o" if separated else "x", color=color, markersize=7, mew=2)
        right.annotate(f"{delta:.2f}", (delta, y), xytext=(8, 0),
                       textcoords="offset points", va="center", color=color)
    right.set(xlim=(-0.13, 2.55), xticks=[0, 0.5, 1, 1.5, 2],
              xlabel="正负指令的首拍实际腿关节力矩差（N·m）")
    right.set_title("第一拍能把动作施加到腿上吗？", loc="left", weight="bold", pad=29)
    right.text(0, 1.025, "×  两个方向的实发力矩相同", transform=right.transAxes, color=red)

    for ax, cases in ((left, sensing), (right, actions)):
        ax.set_yticks(range(len(cases)), labels(cases))
        ax.set_ylim(len(cases) - 0.45, -0.55)
        ax.grid(axis="x", color="#e4e8eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b4bdc4")
        ax.tick_params(axis="both", length=0, pad=6)
        ax.xaxis.labelpad = 11
    fig.text(0.5, 0.035, "左图为零预瞄动作的留出路线。右图另用六个开发起点，每点从同一状态施加负、零、正脉冲。",
             ha="center", color="#43515c", fontsize=10)
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.audit.read_text(encoding="utf-8")), args.output)
