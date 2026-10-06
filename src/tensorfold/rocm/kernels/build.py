"""Build the RDNA extensions for the GPU present, one gfx target, wavefront 32."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any

# gfx11 WMMA (RDNA3, RDNA 3.5) and gfx12 WMMA (RDNA4); RDNA2 uses dot2, and RDNA1 is not a target.
WMMA = {
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201",
}
RDNA = WMMA | {
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
}
RDNA1 = {"gfx1010", "gfx1011", "gfx1012"}
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again


def _rocm_lib_flags() -> list[str]:
    """Flags for a ROCm tree whose device bitcode is not on hipcc's default search path."""

    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    bitcode = os.path.join(rocm, "lib", "llvm", "amdgcn", "bitcode")
    if not os.path.isdir(bitcode):
        return []
    return [f"--rocm-path={rocm}", f"--rocm-device-lib-path={bitcode}"]


def gfx_name() -> str:
    """The visible device's gfx target, without the feature suffix (``gfx1100:xnack-``)."""

    import torch

    name = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if name in RDNA1:
        raise RuntimeError(f"TensorFold does not target RDNA1 ({name})")
    if name not in RDNA:
        raise RuntimeError(f"TensorFold's RDNA kernels do not have a schedule for {name} "
                           f"({torch.cuda.get_device_name(0)})")
    return name


def load(name: str, sources: list[str], **kwargs: Any) -> Any:
    """torch's JIT load for this GPU only, named by a digest of the sources, their headers and the flags."""

    from torch.utils import cpp_extension

    gfx = gfx_name()
    wmma = gfx in WMMA
    # Undo torch's HIP half-operator macros (rocWMMA needs the float constructor) and point hipcc at device bitcode.
    kwargs["extra_cuda_cflags"] = ["-mno-wavefrontsize64", "-ffp-contract=off",
                                   "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__",
                                   f"-DTENSORFOLD_RDNA_WMMA={1 if wmma else 0}",
                                   *_rocm_lib_flags(),
                                   *kwargs.get("extra_cuda_cflags", [])]
    kwargs.setdefault("with_cuda", True)
    name = f"{name}_{gfx}_{_digest(sources, kwargs)}"
    held = _announce(cpp_extension, name, sources, kwargs.get("build_directory"), gfx, wmma)
    timer = None
    if held is not None:
        timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, held)
        timer.daemon = True
        timer.start()
    previous = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = gfx
    try:
        return cpp_extension.load(name=name, sources=sources, **kwargs)
    finally:
        if timer is not None:
            timer.cancel()
        if previous is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = previous


def _digest(sources: list[str], kwargs: dict[str, Any]) -> str:
    """Twelve hex digits over the flags, every source, and every header in the source and include directories.

    A ``.hip`` file a source includes counts too: attention.hip includes attention_fa.hip, and an edit there has to
    rebuild that extension (and only that one).
    """

    folders = {os.path.dirname(os.path.abspath(source)) for source in sources}
    folders.update(os.path.abspath(path) for path in kwargs.get("extra_include_paths", ()))
    headers = sorted(os.path.join(folder, entry) for folder in folders for entry in os.listdir(folder)
                     if entry.endswith((".hpp", ".h", ".cuh")))
    listed = {os.path.abspath(source) for source in sources}
    for source in sources:
        with open(source, encoding="utf-8", errors="replace") as handle:
            for name in re.findall(r'^\s*#\s*include\s+"([^"]+\.hip)"', handle.read(), flags=re.M):
                path = os.path.join(os.path.dirname(os.path.abspath(source)), name)
                if os.path.exists(path) and path not in listed and path not in headers:
                    headers.append(path)
    digest = hashlib.sha256(repr(sorted((k, repr(v)) for k, v in kwargs.items() if k != "verbose")).encode())
    for path in [*sources, *headers]:
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def _announce(cpp_extension: Any, name: str, sources: list[str], directory: str | None, gfx: str,
              wmma: bool) -> tuple[str, tuple[int, int]] | None:
    """Say whether ``name`` builds or waits on a lock; return (lock path, (inode, mtime)) when it waits."""

    try:
        directory = directory or cpp_extension._get_build_directory(name, verbose=False)   # private in torch
    except Exception:  # noqa: BLE001 - no lookup in this torch: build without the lines
        return None
    lock = os.path.join(directory, "lock")          # torch's FileBaton for this extension
    try:
        seen = os.stat(lock)
    except OSError:
        seen = None
    if seen is not None:
        _say(f"RDNA extension {name} waits on the build lock {lock}; {HINT}")
        return lock, (seen.st_ino, seen.st_mtime_ns)
    if _needs_build(os.path.join(directory, name + getattr(cpp_extension, "LIB_EXT", ".so")), sources):
        _say(f"building RDNA extension {name} for {gfx} ({'wmma' if wmma else 'gemv'}) "
             "(first start after an install or update; later starts reuse it)")
    return None


def _needs_build(module: str, sources: list[str]) -> bool:
    """No built module, or a source newer than it (ninja then compiles again)."""

    try:
        built = os.path.getmtime(module)
    except OSError:
        return True
    for source in sources:
        try:
            if os.path.getmtime(source) > built:
                return True
        except OSError:
            continue
    return False


def _still_waiting(lock: str, identity: tuple[int, int]) -> None:
    """Repeat the hint while the lock seen before the build is still the one there."""

    try:
        now = os.stat(lock)
    except OSError:
        return
    if (now.st_ino, now.st_mtime_ns) == identity:
        _say(f"still waiting after {LOCK_WAIT_SECONDS:g} s on the build lock {lock}; {HINT}")


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["RDNA", "WMMA", "gfx_name", "load"]
