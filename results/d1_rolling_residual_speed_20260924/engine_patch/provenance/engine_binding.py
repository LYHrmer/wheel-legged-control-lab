"""Strict runtime binding and counter bridge for the isolated EPA DSO.

Importing this module does not import MuJoCo or call an engine function. The
root-owned C/D runners alone instantiate EngineGuard inside their budgeted
processes. Its fixed offsets are valid only for the ELF hashes below.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any

INSTALLED = Path("/home/lyh/.local/lib/python3.10/site-packages/mujoco")
ORIGINAL = INSTALLED / "libmujoco.so.3.12.0"
FUNCTIONS = INSTALLED / "_functions.cpython-310-x86_64-linux-gnu.so"
ROLLOUT = INSTALLED / "_rollout.cpython-310-x86_64-linux-gnu.so"
ELF_SHA256 = {
    ORIGINAL: "bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8",
    FUNCTIONS: "c91f9a7a4f5d12023201c1c4faab893f4593cc953f53756cd9ac561d08e18a46",
    ROLLOUT: "f0fcc4ffe83b44829703ec5c6a7e65ee2ea7ad40648f07c8bd3515b80e49f877",
}
# Fixed readelf -rW R_X86_64_JUMP_SLOT locations on the ELF files above.
SLOTS = (
    (ORIGINAL, 0x561410, "mjc_ccd"),
    (ORIGINAL, 0x5617E0, "mj_step"),
    (FUNCTIONS, 0x1182D8, "mj_step"),
    (ROLLOUT, 0xA64A0, "mj_step"),
)
# Fixed objdump -d original lib at mjc_penetration+0x1d9:
# 0x185cd9 call mjc_ccd@plt (5 bytes); return address is 0x185cde.
NATIVE_CCD_RETURN_OFFSET = 0x185CDE
# Fixed TryCompile call at 0x3a3543: 5-byte call mj_step@plt.
NATIVE_CONSTRUCTION_RETURN_OFFSET = 0x3A3548
ORIGINAL_STEP_OFFSET = 0x21A500


class _CState(ctypes.Structure):
    _fields_ = [
        ("construction_attempts", ctypes.c_uint64),
        ("construction_returns", ctypes.c_uint64),
        ("control_attempts", ctypes.c_uint64),
        ("control_returns", ctypes.c_uint64),
        ("ccd_attempts", ctypes.c_uint64),
        ("ccd_returns", ctypes.c_uint64),
        ("violations", ctypes.c_uint64),
        ("construction_limit", ctypes.c_uint64),
        ("control_limit", ctypes.c_uint64),
        ("first_ccd_caller", ctypes.c_size_t),
        ("first_construction_step_caller", ctypes.c_size_t),
        ("first_control_step_caller", ctypes.c_size_t),
        ("original_step", ctypes.c_size_t),
        ("target_model", ctypes.c_size_t),
        ("target_data", ctypes.c_size_t),
        ("armed", ctypes.c_int),
        ("phase", ctypes.c_int),
    ]


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapped_bases(expected_paths: tuple[Path, ...]) -> dict[Path, int]:
    bases: dict[Path, int] = {}
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        offset = int(parts[2], 16)
        # The fixed ELFs have a PT_LOAD at p_offset=0, p_vaddr=0. Other
        # segments have p_vaddr != p_offset after page alignment, so their
        # /proc/maps start-offset is not the ELF load bias.
        if offset != 0:
            continue
        path_text = parts[5].removesuffix(" (deleted)")
        path = Path(path_text).resolve()
        if path not in expected_paths:
            continue
        base = int(parts[0].split("-", maxsplit=1)[0], 16)
        if path in bases and bases[path] != base:
            raise RuntimeError(f"multiple ELF bases for {path}")
        bases[path] = base
    return bases


class EngineGuard:
    """A thin ctypes wrapper around the preloaded isolation DSO."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        if os.environ.get("LD_PRELOAD") != str(self.path):
            raise RuntimeError("LD_PRELOAD must be the exact isolated DSO path")
        if os.environ.get("LD_BIND_NOW") != "1":
            raise RuntimeError("LD_BIND_NOW must equal 1")
        if os.environ.get("LD_LIBRARY_PATH"):
            raise RuntimeError("LD_LIBRARY_PATH must be empty")
        self._lib = ctypes.CDLL(str(self.path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self._lib.epa01_arm.argtypes = (ctypes.c_uint, ctypes.c_uint)
        self._lib.epa01_arm.restype = ctypes.c_int
        self._lib.epa01_set_phase.argtypes = (ctypes.c_int,)
        self._lib.epa01_set_phase.restype = ctypes.c_int
        self._lib.epa01_set_target.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self._lib.epa01_set_target.restype = ctypes.c_int
        self._lib.epa01_get_state.argtypes = (ctypes.POINTER(_CState), ctypes.c_size_t)
        self._lib.epa01_get_state.restype = ctypes.c_int
        self._lib.epa01_original_step_address.argtypes = ()
        self._lib.epa01_original_step_address.restype = ctypes.c_size_t
        self._libc = ctypes.CDLL(None)
        self._libc.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
        self._libc.dladdr.restype = ctypes.c_int

    def arm(self, construction_limit: int = 4, control_limit: int = 12000) -> None:
        if (not isinstance(construction_limit, int) or isinstance(construction_limit, bool)
                or not 0 <= construction_limit <= 2**32 - 1
                or not isinstance(control_limit, int) or isinstance(control_limit, bool)
                or not 0 <= control_limit <= 2**32 - 1):
            raise ValueError("limits must be unsigned 32-bit integers")
        if self._lib.epa01_arm(construction_limit, control_limit) != 0:
            raise RuntimeError("engine guard has already been armed")

    def set_phase(self, phase: int) -> None:
        if phase not in (0, 1, 2) or self._lib.epa01_set_phase(phase) != 0:
            raise RuntimeError(f"rejected engine phase {phase}")

    def set_target(self, model_address: int | None, data_address: int | None) -> None:
        model = 0 if model_address is None else int(model_address)
        data = 0 if data_address is None else int(data_address)
        if model < 0 or data < 0 or self._lib.epa01_set_target(model, data) != 0:
            raise RuntimeError("rejected engine target")

    def _dladdr(self, address: int) -> dict[str, Any] | None:
        if not address:
            return None
        info = _DlInfo()
        if not self._libc.dladdr(ctypes.c_void_p(address), ctypes.byref(info)):
            return None
        path = Path(os.fsdecode(info.dli_fname)).resolve() if info.dli_fname else None
        base = int(info.dli_fbase or 0)
        return {
            "path": str(path) if path else None,
            "base": base,
            "address": address,
            "offset": address - base,
            "symbol": os.fsdecode(info.dli_sname) if info.dli_sname else None,
        }

    def snapshot(self) -> dict[str, Any]:
        state = _CState()
        if self._lib.epa01_get_state(ctypes.byref(state), ctypes.sizeof(state)) != 0:
            raise RuntimeError("failed to read engine guard state")
        result: dict[str, Any] = {
            field_name: int(getattr(state, field_name)) for field_name, _ in _CState._fields_
        }
        caller = self._dladdr(result["first_ccd_caller"])
        result["first_ccd_caller_dladdr"] = caller
        result["native_ccd_caller_verified"] = bool(
            caller
            and caller["path"] == str(ORIGINAL)
            and caller["offset"] == NATIVE_CCD_RETURN_OFFSET
            and _sha256(ORIGINAL) == ELF_SHA256[ORIGINAL]
        )
        for label in ("first_construction_step_caller", "first_control_step_caller"):
            result[label + "_dladdr"] = self._dladdr(result[label])
        construction_caller = result["first_construction_step_caller_dladdr"]
        result["native_construction_caller_verified"] = bool(
            construction_caller
            and construction_caller["path"] == str(ORIGINAL)
            and construction_caller["offset"] == NATIVE_CONSTRUCTION_RETURN_OFFSET
            and _sha256(ORIGINAL) == ELF_SHA256[ORIGINAL]
        )
        return result

    def binding_proof(self) -> dict[str, Any]:
        paths = (self.path, ORIGINAL, FUNCTIONS, ROLLOUT)
        bases = _mapped_bases(paths)
        hashes = {str(path): _sha256(path) for path in paths}
        hash_ok = all(hashes[str(path)] == digest for path, digest in ELF_SHA256.items())
        wrapper_ccd = int(ctypes.cast(self._lib.mjc_ccd, ctypes.c_void_p).value or 0)
        wrapper_step = int(ctypes.cast(self._lib.mj_step, ctypes.c_void_p).value or 0)
        wrapper_ccd_info = self._dladdr(wrapper_ccd)
        wrapper_step_info = self._dladdr(wrapper_step)
        original_step = int(self._lib.epa01_original_step_address())
        original_info = self._dladdr(original_step)
        slots: list[dict[str, Any]] = []
        for path, offset, symbol in SLOTS:
            base = bases.get(path)
            target = (int(ctypes.c_void_p.from_address(base + offset).value or 0)
                      if base is not None and hash_ok else None)
            expected = wrapper_ccd if symbol == "mjc_ccd" else wrapper_step
            slots.append({
                "elf": str(path), "elf_sha256": hashes[str(path)],
                "base": base, "relocation_offset": offset,
                "relocation_type": "R_X86_64_JUMP_SLOT", "symbol": symbol,
                "target": target, "expected_wrapper": expected,
                "passed": target == expected if target is not None else False,
            })
        passed = bool(
            hash_ok
            and self.path in bases
            and all(path in bases for path in (ORIGINAL, FUNCTIONS, ROLLOUT))
            and wrapper_ccd_info and wrapper_ccd_info["path"] == str(self.path)
            and wrapper_step_info and wrapper_step_info["path"] == str(self.path)
            and original_info and original_info["path"] == str(ORIGINAL)
            and original_info["offset"] == ORIGINAL_STEP_OFFSET
            and all(slot["passed"] for slot in slots)
        )
        return {
            "passed": passed,
            "dso_path": str(self.path),
            "dso_sha256": hashes[str(self.path)],
            "elf_hashes": hashes,
            "elf_bases": {str(path): bases.get(path) for path in paths},
            "wrapper_ccd": wrapper_ccd_info,
            "wrapper_step": wrapper_step_info,
            "rtld_next_original_step": original_info,
            "jump_slots": slots,
            "counters": self.snapshot(),
        }
