"""Fetch the locked SegMAN sources and build the required CUDA extensions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
CUDA_SOURCE = ROOT / "cuda"
BUILD = ROOT / "build"


def copy_source(destination, data):
    if destination.is_file() and destination.read_bytes() == data:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def prepare_sources(build_dir):
    provenance = json.loads((ROOT.parent / "configs/source_provenance.json").read_text(encoding="utf-8"))
    lock = provenance["segman_build_sources"]
    repo = lock["repository"].removeprefix("https://github.com/")
    origin = f"https://raw.githubusercontent.com/{repo}/{lock['commit']}"
    for relative, record in lock["files"].items():
        target = build_dir / relative
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
            continue
        with urlopen(f"{origin}/{record['path']}", timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Locked SegMAN source checksum mismatch for {relative}")
        copy_source(target, data)

    overlay = CUDA_SOURCE / "oflex"
    for source in overlay.rglob("*"):
        if source.is_file():
            copy_source(build_dir / "oflex" / source.relative_to(overlay), source.read_bytes())


def build_extensions(inference_only):
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").is_file():
        raise RuntimeError("Set CUDA_HOME to the CUDA 11.8 toolkit directory")

    prepare_sources(BUILD)
    version = ["-O3", "-std=c++17"]
    oflex = BUILD / "oflex"
    kernels = oflex / "cusoflex"
    arch_flags = [
        flag
        for arch in ("70", "80", "89", "90")
        for flag in ("-gencode", f"arch=compute_{arch},code=sm_{arch}")
    ]
    scan_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
        "--ptxas-options=-v", "-lineinfo", "--threads", "4",
    ]
    modules = [CUDAExtension(
        "selective_scan_cuda_oflex", [str(kernels / name) for name in (
            "selective_scan_oflex.cpp", "selective_scan_core_fwd.cu", "selective_scan_core_bwd.cu",
        )], include_dirs=[str(oflex)],
        extra_compile_args={"cxx": version, "nvcc": version + scan_flags + arch_flags},
    )]

    if not inference_only:
        resize = CUDA_SOURCE / "resize"
        resize_flags = {"cxx": version, "nvcc": version + ["--fmad=false"]}
        for name in ("r01", "r05"):
            modules.append(CUDAExtension(
                f"{name}_resize_backward_ext",
                [str(resize / f"{name}_resize_backward{suffix}") for suffix in (".cpp", "_cuda.cu")],
                extra_compile_args=resize_flags,
            ))
        binding_source = (resize / "bicubic_backward.cpp").read_bytes()
        for precision in ("float", "double"):
            binding = BUILD / f"bicubic_{precision}.cpp"
            copy_source(binding, binding_source)
            modules.append(CUDAExtension(
                f"bicubic_{precision}_ext",
                [str(binding), str(resize / f"bicubic_backward_{precision}_cuda.cu")],
                extra_compile_args=resize_flags,
            ))

    os.chdir(CUDA_SOURCE)
    setup(name="abh_cuda_ops", packages=[], ext_modules=modules,
          cmdclass={"build_ext": BuildExtension}, script_args=["build_ext", "--inplace"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-only", action="store_true", help="Build Oflex for inference and cache export")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    os.environ.setdefault("MAX_JOBS", "4")
    build_extensions(args.inference_only)


if __name__ == "__main__":
    main()
