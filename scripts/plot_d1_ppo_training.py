"""D1 budget study 的离线 PPO 训练诊断绘图（每次 train() 调用一个点）。

仅读取训练产物；不导入项目代码 / Torch / MuJoCo，不访问网络。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMA = "d1-budget-study-v1"
OUTPUT_SCHEMA = "d1-ppo-training-diagnostics-v1"
METRICS = (
    "train/std",
    "train/approx_kl",
    "train/clip_fraction",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/explained_variance",
)
METRIC_SEMANTICS = {
    "train/std": "本次 train() 调用结束后的 mean(exp(log_std))（更新后的策略标准差）。",
    "train/approx_kl": "仅最后一个已完成 epoch 的 minibatch 均值，不是全部 epoch 的均值。",
    "train/clip_fraction": (
        "所有 epoch / minibatch 中 abs(ratio - 1) > clip_range 的比例均值；"
        "不等于 min 实际选择裁剪分支的比例，后者还取决于 advantage 的符号。"
    ),
    "train/value_loss": "本次 train() 调用内所有 epoch / minibatch 的均值。",
    "train/policy_gradient_loss": "本次 train() 调用内所有 epoch / minibatch 的均值。",
    "train/explained_variance": "更新前的旧 value 预测对完整 rollout 的解释方差（OLD values）。",
    "_excluded": (
        "updates.jsonl 顶层的 approx_reverse_kl_after / explained_variance_old_values 等审计字段"
        "基于前 128 个 env-major 样本子集，语义不同，本脚本不使用也不混用。"
    ),
    "_source": "stable_baselines3/ppo/ppo.py::PPO.train（本地依赖，仅阅读未导入）",
}
# 图上文本使用英文：默认 matplotlib 字体不含中文字形，中文会渲染成方块。
FOOTNOTE = (
    "Semantics: train/approx_kl is the minibatch mean of the LAST completed epoch only; "
    "train/explained_variance uses the OLD pre-update value predictions over the full rollout; "
    "train/std is post-update mean exp(log_std); clip_fraction / value_loss / policy_gradient_loss "
    "are means over all minibatches of all epochs in that train() call. "
    "x axis = recorded actual cumulative env transitions (not update or epoch index). "
    "Top-level 128-sample audit fields (approx_reverse_kl_after, explained_variance_old_values) are NOT used."
)
MODE_COLORS = {"shared2": "#1f77b4", "independent8": "#d62728"}
SEED_STYLES = ("-", "--", ":", "-.")


def _no_dup_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"JSON 重复键: {key!r}")
        out[key] = value
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Inputs:
    """记录每个输入文件加载时的 SHA256，用于发布前复核是否被改动。"""

    def __init__(self, study: Path) -> None:
        self.study = study
        self.hashes: dict[str, str] = {}

    def record(self, path: Path) -> str:
        self.check_path(path)
        rel = path.relative_to(self.study).as_posix()
        digest = _sha256(path)
        previous = self.hashes.get(rel)
        if previous is not None and previous != digest:
            raise ValueError(f"输入文件在读取期间发生变化: {rel}")
        self.hashes[rel] = digest
        return digest

    def check_path(self, path: Path) -> None:
        if ".." in path.parts or not path.resolve().is_relative_to(self.study):
            raise ValueError(f"输入路径必须位于 study 内部: {path}")

    def read_bytes(self, path: Path) -> bytes:
        self.check_path(path)
        payload = path.read_bytes()
        rel = path.relative_to(self.study).as_posix()
        digest = hashlib.sha256(payload).hexdigest()
        if rel in self.hashes and self.hashes[rel] != digest:
            raise ValueError(f"输入文件在读取期间发生变化: {rel}")
        self.hashes[rel] = digest
        return payload

    def load_json(self, path: Path):
        if not path.is_file():
            raise ValueError(f"缺少必需文件: {path}")
        try:
            value = json.loads(self.read_bytes(path), object_pairs_hook=_no_dup_keys)
            _require(isinstance(value, dict), f"{path}: 必须是 JSON 对象")
            return value
        except ValueError as exc:
            raise ValueError(f"{path}: JSON 解析失败: {exc}") from None

    def recheck(self) -> None:
        for rel, digest in self.hashes.items():
            path = self.study / rel
            self.check_path(path)
            if not path.is_file():
                raise ValueError(f"输入文件在发布前消失: {rel}")
            if _sha256(path) != digest:
                raise ValueError(f"输入文件在发布前被修改: {rel}")


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _number(value, where: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{where}: 需要数值，实际为 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: 非有限数值 {value!r}")
    return number


def _integer(value, where: str) -> int:
    _require(type(value) is int, f"{where}: 必须为整数（不允许 bool 或 float）")
    return value


def load_protocol(inputs: Inputs) -> dict:
    protocol = inputs.load_json(inputs.study / "protocol.json")
    _require(protocol.get("schema") == SCHEMA, f"顶层 protocol.schema 必须为 {SCHEMA}")
    kind = protocol.get("kind")
    _require(kind in ("formal", "smoke"), f"未知 protocol.kind: {kind!r}")
    modes = list(protocol["action_modes"])
    seeds = list(protocol["training_seeds"])
    budgets = list(protocol["checkpoint_budgets"])
    _require(set(modes) == set(MODE_COLORS) and len(modes) == 2, "action_modes 必须为 shared2、independent8")
    _require(len(seeds) == (3 if kind == "formal" else 1), "training_seeds 数量与 kind 不符")
    _require(all(_integer(s, "training seed") >= 0 for s in seeds), "training seed 必须非负")
    _require(len(set(seeds)) == len(seeds), "training_seeds 重复")
    _require(budgets and all(_integer(b, "budget") > 0 for b in budgets), "budget 必须为正整数")
    _require(budgets == sorted(set(budgets)), "checkpoint_budgets 必须唯一且递增")
    workers = _integer(protocol["workers"], "workers")
    n_steps = _integer(protocol["n_steps"], "n_steps")
    _require(workers > 0 and n_steps > 0, "workers 和 n_steps 必须为正整数")
    increment = workers * n_steps
    max_budget = max(budgets)
    _require(all(b % increment == 0 for b in budgets), "budget 不是 workers*n_steps 的整数倍")
    calls_per_model = max_budget // increment

    models = protocol["models"]
    identities = [(m["action_mode"], _integer(m["seed"], "model.seed")) for m in models]
    for model in models:
        expected = f"{model['action_mode']}_seed{model['seed']}"
        _require(model["model_id"] == model["training_run"] == expected, "model_id/training_run 身份不符")
    _require(len(set(identities)) == len(identities), "protocol.models 存在重复身份")
    _require(
        set(identities) == {(mode, seed) for mode in modes for seed in seeds}
        and len(identities) == len(modes) * len(seeds),
        "protocol.models 不是 action_modes × training_seeds 的唯一笛卡尔积",
    )
    _require(
        _integer(protocol["expected_train_calls"], "expected_train_calls") == len(models) * calls_per_model,
        "expected_train_calls != models * max_budget / (workers*n_steps)",
    )
    declared_pairs = [(c["model_id"], _integer(c["budget"], "checkpoint.budget"))
                      for c in protocol["checkpoints"]]
    expected_pairs = {(m["model_id"], b) for m in models for b in budgets}
    _require(len(declared_pairs) == len(expected_pairs) and set(declared_pairs) == expected_pairs,
             "checkpoints 必须与 models × budgets 一一对应")
    for entry in protocol["checkpoints"]:
        prefix = f"{entry['model_id']}/checkpoints/budget{entry['budget']}"
        _require(entry["relative_metadata"] == f"{prefix}/checkpoint.json"
                 and entry["relative_zip"] == f"{prefix}/checkpoint.zip"
                 and entry["training_run"] == entry["model_id"], "checkpoint 路径/身份不符")
    return {
        "kind": kind,
        "models": models,
        "budgets": budgets,
        "workers": workers,
        "n_steps": n_steps,
        "increment": increment,
        "max_budget": max_budget,
        "calls_per_model": calls_per_model,
        "checkpoints": protocol["checkpoints"],
        "expected_train_calls": protocol["expected_train_calls"],
    }


def load_updates(inputs: Inputs, run_dir: Path, spec: dict) -> list[dict]:
    path = run_dir / "updates" / "updates.jsonl"
    _require(path.is_file(), f"缺少 updates.jsonl: {path}")
    rows: list[dict] = []
    previous_timesteps = 0
    payload = inputs.read_bytes(path).decode("utf-8")
    if payload:
        for line_index, raw in enumerate(payload.splitlines()):
            where = f"{path}:{line_index + 1}"
            _require(raw.strip() != "", f"{where}: 空行")
            try:
                row = json.loads(raw, object_pairs_hook=_no_dup_keys)
            except ValueError as exc:
                raise ValueError(f"{where}: JSON 解析失败: {exc}") from None
            _require(isinstance(row, dict), f"{where}: 行必须是 JSON 对象")
            _require(
                _integer(row.get("audit_index"), f"{where}: audit_index") == line_index,
                f"{where}: audit_index 应为 {line_index}，实际 {row.get('audit_index')!r}",
            )
            timesteps = _integer(row.get("num_timesteps"), f"{where}: num_timesteps")
            _require(timesteps > previous_timesteps, f"{where}: num_timesteps 未严格递增")
            _require(
                timesteps - previous_timesteps == spec["increment"],
                f"{where}: transition 增量应为 {spec['increment']}，实际 {timesteps - previous_timesteps}",
            )
            previous_timesteps = timesteps
            metrics_blob = row.get("logger_train_metrics")
            _require(isinstance(metrics_blob, dict), f"{where}: 缺少 logger_train_metrics 对象")
            values = {}
            for metric in METRICS:
                _require(metric in metrics_blob, f"{where}: 缺少指标 {metric}")
                values[metric] = _number(metrics_blob[metric], f"{where}: {metric}")
            rows.append({"audit_index": line_index, "num_timesteps": timesteps, "metrics": values})
    _require(
        len(rows) == spec["calls_per_model"],
        f"{path}: 期望 {spec['calls_per_model']} 行 train 调用记录，实际 {len(rows)}",
    )
    _require(
        previous_timesteps == spec["max_budget"],
        f"{path}: 最终 num_timesteps 应为 {spec['max_budget']}，实际 {previous_timesteps}",
    )
    return rows


def load_model(inputs: Inputs, spec: dict, model: dict) -> dict:
    model_id, mode, seed = model["model_id"], model["action_mode"], model["seed"]
    run_dir = inputs.study / model["training_run"]
    _require(run_dir.is_dir(), f"{model_id}: 缺少训练目录 {run_dir}")

    training = inputs.load_json(run_dir / "training.json")
    _require(training.get("training_mode") == "single_continuous_learn", f"{model_id}: training_mode 不符")
    _require(_integer(training.get("learn_calls"), "learn_calls") == 1, f"{model_id}: learn_calls 必须为 1")
    _require(
        _integer(training["num_timesteps"], "training.num_timesteps") == spec["max_budget"],
        f"{model_id}: num_timesteps={training['num_timesteps']} 与最大 budget {spec['max_budget']} 不符（训练未完成）",
    )
    _require(list(training["checkpoint_budgets"]) == spec["budgets"], f"{model_id}: checkpoint_budgets 与顶层不符")
    _require(training["action_contract"]["action_mode"] == mode, f"{model_id}: action_contract.action_mode 不符")
    _require(bool(training.get("training_finished_at_utc")), f"{model_id}: 缺少 training_finished_at_utc（运行未完成）")

    run_protocol = inputs.load_json(run_dir / "protocol.json")
    arguments = run_protocol["arguments"]
    _require(arguments.get("mode") == "train", f"{model_id}: arguments.mode 必须为 train")
    _require(_integer(arguments["seed"], "arguments.seed") == seed, f"{model_id}: arguments.seed 与顶层不符")
    _require(arguments["action_mode"] == mode, f"{model_id}: arguments.action_mode 与顶层不符")
    _require(_integer(arguments["workers"], "arguments.workers") == spec["workers"], f"{model_id}: arguments.workers 与顶层不符")
    _require(_integer(arguments["steps"], "arguments.steps") == spec["max_budget"], f"{model_id}: arguments.steps 与最大 budget 不符")
    _require(
        _integer(run_protocol["ppo_settings"]["n_steps"], "ppo_settings.n_steps") == spec["n_steps"],
        f"{model_id}: ppo_settings.n_steps 与顶层不符",
    )

    declared = [c for c in spec["checkpoints"] if c["model_id"] == model_id]
    _require(
        sorted(c["budget"] for c in declared) == sorted(spec["budgets"]),
        f"{model_id}: 声明的 checkpoint budget 集合与顶层不符",
    )
    for entry in declared:
        budget = entry["budget"]
        meta_path = inputs.study / entry["relative_metadata"]
        zip_path = inputs.study / entry["relative_zip"]
        meta = inputs.load_json(meta_path)
        extra = meta["extra"]
        _require(meta.get("action_mode") == mode, f"{meta_path}: action_mode 不符")
        _require(_integer(extra["training_seed"], "extra.training_seed") == seed, f"{meta_path}: extra.training_seed 不符")
        _require(_integer(extra["checkpoint_budget"], "extra.checkpoint_budget") == budget, f"{meta_path}: extra.checkpoint_budget 不符")
        _require(_integer(extra["num_timesteps"], "extra.num_timesteps") == budget, f"{meta_path}: extra.num_timesteps 不符")
        _require(
            _integer(extra["after_update_index"], "extra.after_update_index") == budget // spec["increment"] - 1,
            f"{meta_path}: extra.after_update_index 与 budget/(workers*n_steps) 不符",
        )
        _require(
            _integer(entry["expected_update_index"], "expected_update_index") == extra["after_update_index"],
            f"{meta_path}: after_update_index 与顶层 expected_update_index 不符",
        )
        _require(zip_path.is_file(), f"缺少 checkpoint zip: {zip_path}")
        if inputs.record(zip_path) != meta["model_sha256"]:
            raise ValueError(f"{zip_path}: zip SHA256 与 checkpoint.json 的 model_sha256 不一致")

    return {
        "model_id": model_id,
        "action_mode": mode,
        "seed": seed,
        "rows": load_updates(inputs, run_dir, spec),
    }


def render(stage: Path, spec: dict, series: list[dict]) -> None:
    seeds = sorted({s["seed"] for s in series})
    _require(len(seeds) <= len(SEED_STYLES), "seed 数量超过可用线型数量")
    figure, axes = plt.subplots(2, 3, figsize=(19, 10))
    for axis, metric in zip(axes.ravel(), METRICS):
        for entry in series:
            axis.plot(
                [r["num_timesteps"] for r in entry["rows"]],
                [r["metrics"][metric] for r in entry["rows"]],
                color=MODE_COLORS.get(entry["action_mode"], "#555555"),
                linestyle=SEED_STYLES[seeds.index(entry["seed"])],
                linewidth=1.0,
                label=f"{entry['model_id']} (seed {entry['seed']})",
            )
        axis.set_title(metric)
        axis.set_xlabel("cumulative env transitions (recorded)")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), fontsize=9)
    figure.suptitle(f"PPO training diagnostics - not controller performance ({spec['kind']})", fontsize=16)
    figure.text(0.5, 0.115, FOOTNOTE, ha="center", va="top", fontsize=8, wrap=True)
    if spec["kind"] == "smoke":
        figure.text(
            0.5, 0.5, "SMOKE - PIPELINE CHECK ONLY", ha="center", va="center",
            fontsize=52, color="red", alpha=0.32, rotation=28, zorder=10,
        )
    figure.tight_layout(rect=(0, 0.16, 1, 0.95))
    figure.savefig(stage / "training_diagnostics.png", dpi=150)
    plt.close(figure)

    with (stage / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model_id", "action_mode", "seed", "audit_index", "num_timesteps", "metric", "value"]
        )
        for entry in series:
            for row in entry["rows"]:
                for metric in METRICS:
                    writer.writerow([
                        entry["model_id"], entry["action_mode"], entry["seed"],
                        row["audit_index"], row["num_timesteps"], metric, repr(row["metrics"][metric]),
                    ])


def write_manifest(stage: Path, script: Path, spec: dict, series: list[dict], inputs: Inputs) -> None:
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "study_schema": SCHEMA,
        "kind": spec["kind"],
        "study_path": str(inputs.study),
        "counts": {
            "models": len(series),
            "train_calls_per_model": spec["calls_per_model"],
            "train_calls_total": sum(len(s["rows"]) for s in series),
            "expected_train_calls": spec["expected_train_calls"],
            "transitions_per_train_call": spec["increment"],
            "max_budget": spec["max_budget"],
            "csv_rows": sum(len(s["rows"]) for s in series) * len(METRICS),
            "metrics_per_train_call": len(METRICS),
        },
        "models": [
            {k: s[k] for k in ("model_id", "action_mode", "seed")} | {"train_calls": len(s["rows"])}
            for s in series
        ],
        "metrics": list(METRICS),
        "metrics_semantics": METRIC_SEMANTICS,
        "script": {"path": str(script), "sha256": _sha256(script)},
        "versions": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__},
        "inputs_sha256": dict(sorted(inputs.hashes.items())),
        "outputs_sha256": {
            name: _sha256(stage / name) for name in ("training_diagnostics.png", "training_metrics.csv")
        },
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_paths(study_arg: str, output_arg: str) -> tuple[Path, Path]:
    study = Path(study_arg).expanduser().resolve()
    _require(not os.path.lexists(Path(output_arg).expanduser()), f"输出路径已存在，拒绝覆盖: {output_arg}")
    output = Path(output_arg).expanduser().resolve()
    _require(study.is_dir(), f"study 目录不存在: {study}")
    _require(not output.exists(), f"输出路径已存在，拒绝覆盖: {output}")
    _require(output != study and study not in output.parents, f"输出目录不能位于输入 study 内部: {output}")
    _require(output.parent.is_dir(), f"输出父目录不存在: {output.parent}")
    return study, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D1 budget study 每次 PPO train 调用的离线训练诊断")
    parser.add_argument("--study", required=True, help="d1_budget_study / d1_budget_smoke_final 目录")
    parser.add_argument("--output", required=True, help="新建输出目录（必须不存在）")
    args = parser.parse_args(argv)
    script = Path(__file__).resolve()
    try:
        study, output = resolve_paths(args.study, args.output)
        inputs = Inputs(study)
        spec = load_protocol(inputs)
        series = [load_model(inputs, spec, model) for model in spec["models"]]
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
        try:
            render(stage, spec, series)
            write_manifest(stage, script, spec, series, inputs)
            inputs.recheck()
            _require(not os.path.lexists(output), f"输出路径在发布前被创建: {output}")
            os.rename(stage, output)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"已写出 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
