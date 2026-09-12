"""d1 budget study 检查点张量与哈希的独立审计（仅 stdlib/NumPy/torch）。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import stat
import zipfile
from pathlib import Path

import numpy as np
import torch

ENTRIES = {
    "data",
    "pytorch_variables.pth",
    "policy.pth",
    "policy.optimizer.pth",
    "_stable_baselines3_version",
    "system_info.txt",
}
LIMIT = 64 * 1024 * 1024
COUNTS = {2: 19141, 8: 19537}
MODES = {"shared2": 2, "independent8": 8}
META_SCHEMA = "d1-locomotion-checkpoint-v1"
LIMITATIONS = [
    "仅校验 policy.pth 张量与文件哈希，不验证 optimizer/pytorch_variables/data 字段",
    "不重放训练、不做任何数学或科学有效性判断",
    "不解析 NPZ 或源码归档；这些记录由既有独立分析器另行核验",
    "仅适用于受信任的本地项目检查点；weights_only 对不可信文件不构成安全保证",
]


def _need(cond: object, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _shapes(a: int) -> dict:
    exp = {
        "log_std": (a,),
        "action_net.weight": (a, 64),
        "action_net.bias": (a,),
        "value_net.weight": (1, 64),
        "value_net.bias": (1,),
    }
    for net in ("policy_net", "value_net"):
        exp[f"mlp_extractor.{net}.0.weight"] = (64, 82)
        exp[f"mlp_extractor.{net}.0.bias"] = (64,)
        exp[f"mlp_extractor.{net}.2.weight"] = (64, 64)
        exp[f"mlp_extractor.{net}.2.bias"] = (64,)
    return exp


def _pairs(items: list) -> dict:
    out: dict = {}
    for key, val in items:
        _need(key not in out, f"JSON 重复键: {key}")
        out[key] = val
    return out


def _float(text: str) -> float:
    val = float(text)
    _need(math.isfinite(val), f"JSON 非有限数值: {text}")
    return val


def _const(text: str) -> float:
    raise ValueError(f"JSON 非有限常量: {text}")


def _loads(text: str) -> object:
    return json.loads(text, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_const)


def _safe(path: Path) -> Path:
    path = Path(path).absolute()
    _need(".." not in path.parts and "\\" not in str(path), "路径必须规范且无父级跳转")
    for node in (path, *path.parents):
        _need(not node.is_symlink(), f"路径含符号链接: {node}")
    return path


def _file(path: Path) -> Path:
    _safe(path)
    _need(path.is_file(), f"缺少常规文件: {path}")
    return path


def _json(path: Path) -> dict:
    obj = _loads(_file(path).read_text("utf-8"))
    _need(isinstance(obj, dict), f"JSON 顶层非对象: {path}")
    return obj


def _jsonl(path: Path) -> list:
    rows = []
    for num, line in enumerate(_file(path).read_text("utf-8").splitlines(), 1):
        _need(bool(line.strip()), f"{path}:{num} 空记录")
        row = _loads(line)
        _need(isinstance(row, dict), f"{path}:{num} 行非对象")
        rows.append(row)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(_file(path).read_bytes()).hexdigest()


def hash_checkpoint(path: Path, action_size: int) -> dict:
    """独立读取检查点 ZIP 中的 policy.pth，返回参数哈希、ZIP 哈希与参数量。"""
    path = Path(path)
    _need(
        type(action_size) is int and action_size in COUNTS, f"不支持的 action_size: {action_size}"
    )
    _file(path)
    _need(path.stat().st_size <= LIMIT, f"归档超过 64MiB: {path}")
    raw = path.read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = zf.infolist()
            _need(sorted(i.filename for i in infos) == sorted(ENTRIES), f"ZIP 条目不符: {path}")
            total = 0
            for info in infos:
                _need(not info.flag_bits & 0x1, f"加密条目: {info.filename}")
                _need(not info.is_dir(), f"目录条目: {info.filename}")
                kind = stat.S_IFMT(info.external_attr >> 16)
                _need(kind in (0, stat.S_IFREG), f"非常规条目: {info.filename}")
                total += info.file_size
                _need(total <= LIMIT, f"解压总量超过 64MiB: {path}")
            blob = zf.read("policy.pth")
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as exc:
        raise ValueError(f"ZIP 读取失败 {path}: {exc}") from exc
    try:
        state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"policy.pth 加载失败 {path}: {exc}") from exc
    exp = _shapes(action_size)
    _need(isinstance(state, dict) and set(state) == set(exp), f"state_dict 键不符: {path}")
    digest, count = hashlib.sha256(), 0
    for name, tensor in sorted(state.items()):
        _need(isinstance(tensor, torch.Tensor), f"{name} 非张量")
        _need(tensor.dtype == torch.float32 and tensor.layout == torch.strided, f"{name} 类型不符")
        _need(tuple(tensor.shape) == exp[name], f"{name} 形状不符: {tuple(tensor.shape)}")
        array = tensor.detach().cpu().numpy()
        _need(bool(np.isfinite(array).all()), f"{name} 含非有限值")
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        count += int(array.size)
    _need(count == COUNTS[action_size], f"参数量不符: {count}")
    return {
        "parameter_sha256": digest.hexdigest(),
        "zip_sha256": hashlib.sha256(raw).hexdigest(),
        "parameter_count": count,
    }


def _expect(record: dict, expected: dict, label: str) -> None:
    _need(type(record) is dict, f"{label}: 必须为对象")
    for name, value in expected.items():
        _need(
            name in record and type(record[name]) is type(value) and record[name] == value,
            f"{label}: {name} 不符",
        )


def _read_bound_checkpoint(directory, size, seed, budget, index, record, witness, final=False):
    """Bind directly read tensors to file records; this is not training replay."""
    stem = "" if final else f"budget{budget}/"
    _expect(
        record,
        {"model_path": stem + "checkpoint.zip", "metadata_path": stem + "checkpoint.json"},
        "checkpoint paths",
    )
    path, meta_path = directory / record["model_path"], directory / record["metadata_path"]
    result = hash_checkpoint(path, size)
    meta_sha = _sha(meta_path)
    common = {
        "zip_sha256": result["zip_sha256"],
        "metadata_sha256": meta_sha,
        "after_update_index": index,
    }
    _expect(record, common, "checkpoint file hashes")
    _expect(
        witness,
        {**common, "reloaded_policy_param_sha256": result["parameter_sha256"]},
        "reload witness versus directly read tensors",
    )
    meta = _json(meta_path)
    _expect(
        meta,
        {
            "schema": META_SCHEMA,
            "action_dim": size,
            "observation_dim": 82,
            "action_mode": "shared2" if size == 2 else "independent8",
            "history_length": 1,
            "model_sha256": result["zip_sha256"],
        },
        "metadata",
    )
    extra = {"training_seed": seed, "num_timesteps": budget}
    if not final:
        extra.update(
            checkpoint_budget=budget,
            after_update_index=index,
            param_sha256_after=result["parameter_sha256"],
        )
    _expect(meta["extra"], extra, "metadata extra")
    return {**result, "metadata_sha256": meta_sha, "budget": budget, "after_update_index": index}


def audit(study: Path) -> dict:
    """Complement the existing study analyzer with actual policy.pth tensor reads."""
    study = _safe(study)
    _need(study.is_dir(), "study 必须为已有目录")
    failure = study / "failure.json"
    _need(not failure.exists() and not failure.is_symlink(), "study 存在失败记录")
    protocol = _json(study / "protocol.json")
    _expect(protocol, {"schema": "d1-budget-study-v1"}, "protocol")
    _expect(_json(study / "summary.json"), {"status": "commands_completed"}, "summary")
    input_sha256 = {name: _sha(study / name) for name in ("protocol.json", "summary.json")}
    budgets, models = protocol["checkpoint_budgets"], protocol["models"]
    _need(
        type(budgets) is list
        and bool(budgets)
        and all(type(b) is int and b > 0 and b % 512 == 0 for b in budgets),
        "无效预算",
    )
    _need(budgets == sorted(set(budgets)), "预算必须严格递增")
    _need(type(models) is list and bool(models), "缺少模型")
    results, names = [], set()
    for model in models:
        mode, seed, name = model["action_mode"], model["seed"], model["model_id"]
        _need(
            type(mode) is str and mode in MODES and type(seed) is int and seed >= 0,
            "无效模型模式或种子",
        )
        _need(name == f"{mode}_seed{seed}" and name not in names, "无效或重复模型目录")
        names.add(name)
        directory, size = study / name, MODES[mode]
        manifest = _json(directory / "checkpoints/manifest.json")
        _expect(
            manifest,
            {
                "schema": "d1-budget-checkpoints-v1",
                "complete": True,
                "failed": False,
                "failure": None,
                "pending_budgets": [],
                "budgets": budgets,
            },
            "manifest",
        )
        reload = _json(directory / "checkpoint_reload.json")
        _expect(
            reload,
            {
                "schema": "d1-budget-checkpoint-reload-v1",
                "phase": "after_training",
                "training_final_num_timesteps": budgets[-1],
            },
            "reload",
        )
        saved, witnesses = manifest["saved"], reload["records"]
        _need(
            type(saved) is list
            and type(witnesses) is list
            and len(saved) == len(witnesses) == len(budgets),
            "检查点数量不符",
        )
        updates = _jsonl(directory / "updates/updates.jsonl")
        _need(len(updates) == budgets[-1] // 512, "更新记录数量不符")
        selected = []
        for budget, record, witness in zip(budgets, saved, witnesses, strict=True):
            index = budget // 512 - 1
            _expect(
                record,
                {"budget": budget, "num_timesteps": budget, "after_update_index": index},
                "saved checkpoint",
            )
            _expect(witness, {"budget": budget}, "reload budget")
            result = _read_bound_checkpoint(
                directory / "checkpoints", size, seed, budget, index, record, witness
            )
            parameter = result["parameter_sha256"]
            _expect(record, {"param_sha256_after": parameter}, "manifest tensor hash")
            _expect(
                updates[index],
                {"audit_index": index, "num_timesteps": budget, "param_sha256_after": parameter},
                "selected update tensor hash",
            )
            selected.append(result)
        final_record = reload["final_checkpoint"]
        _expect(final_record, {"num_timesteps": budgets[-1]}, "root final timestep")
        final = _read_bound_checkpoint(
            directory,
            size,
            seed,
            budgets[-1],
            budgets[-1] // 512 - 1,
            final_record,
            final_record,
            final=True,
        )
        _need(
            final["parameter_sha256"] == selected[-1]["parameter_sha256"],
            "root final 参数与末预算不符",
        )
        _expect(
            _json(directory / "training.json"),
            {"num_timesteps": budgets[-1], "checkpoint_sha256": final["zip_sha256"]},
            "training final ZIP hash",
        )
        results.append({"model_id": name, "checkpoints": selected, "final_checkpoint": final})
        for relative in (
            "updates/updates.jsonl",
            "checkpoints/manifest.json",
            "checkpoint_reload.json",
            "training.json",
        ):
            input_sha256[f"{name}/{relative}"] = _sha(directory / relative)
    return {
        "schema": "d1-budget-weights-v1",
        "status": "weights_verified",
        "study": str(study),
        "input_sha256": input_sha256,
        "models": results,
        "checkpoints_checked": len(models) * len(budgets),
        "root_finals_checked": len(models),
        "total_zip_files_checked": len(models) * (len(budgets) + 1),
        "limitations": LIMITATIONS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    study, output = _safe(args.study), _safe(args.output)
    _need(not output.exists(), "输出目录必须为新目录")
    _need(not output.is_relative_to(study) and not study.is_relative_to(output), "输入输出重叠")
    report = audit(study)
    output.mkdir(parents=True)
    # The completion marker is last: interruption leaves visibly partial files, never a success marker.
    (output / ".partial").touch(exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    source = Path(__file__)
    with (output / source.name).open("xb") as stream:
        stream.write(source.read_bytes())
    hashes = {name: _sha(output / name) for name in ("report.json", source.name)}
    pending_manifest = output / "manifest.json.partial"
    with pending_manifest.open("x", encoding="utf-8") as stream:
        json.dump({"status": "weights_verified", "sha256": hashes}, stream, indent=2)
        stream.write("\n")
    (output / ".partial").unlink()
    pending_manifest.rename(output / "manifest.json")
    return report


if __name__ == "__main__":
    main()
