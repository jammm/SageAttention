"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import sys
import threading
import warnings

from packaging.version import parse, Version
from setuptools import find_packages, setup


# Skip extension builds in CI or when explicitly requested.
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}


def split_flags(value: str):
    return value.strip().split() if value and value.strip() else []


def split_paths(value: str):
    return [p.strip() for p in value.split(os.pathsep) if p.strip()] if value else []


def rocm_sdk_path(which: str):
    try:
        return subprocess.check_output(["rocm-sdk", "path", f"--{which}"], text=True).strip()
    except Exception:
        return None


def detect_rocm_arches(torch):
    arch_env = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if arch_env:
        return [a.strip() for a in arch_env.replace(";", ",").split(",") if a.strip()]

    archs = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            arch = getattr(props, "gcnArchName", "")
            if arch:
                archs.append(arch.split(":", 1)[0])
    return archs


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def resolve_parallel():
    parallel = None
    for env_name in ("EXT_PARALLEL", "MAX_JOBS"):
        if parallel is None and env_name in os.environ:
            try:
                parallel = int(os.getenv(env_name))
            except ValueError:
                pass
    return 4 if parallel is None else parallel


if not SKIP_CUDA_BUILD:
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    parallel = resolve_parallel()

    if torch.version.hip is not None:
        from torch.utils.cpp_extension import ROCM_HOME

        sdk_rocm_root = rocm_sdk_path("root")
        sdk_core_root = None
        if sdk_rocm_root:
            candidate = os.path.join(os.path.dirname(sdk_rocm_root), "_rocm_sdk_core")
            if os.path.exists(os.path.join(candidate, "bin", "hipcc.exe")):
                sdk_core_root = candidate

        explicit_rocm_home = os.getenv("ROCM_HOME")
        if os.name == "nt" and explicit_rocm_home == sdk_rocm_root and sdk_core_root:
            rocm_home = sdk_core_root
        else:
            rocm_home = explicit_rocm_home or sdk_core_root or ROCM_HOME or sdk_rocm_root
        rocm_root = sdk_rocm_root or rocm_home
        if not rocm_home or not rocm_root:
            raise RuntimeError("Cannot find ROCm. Activate the ROCm Python environment or set ROCM_HOME.")

        path_parts = [
            os.path.join(rocm_home, "lib", "llvm", "bin"),
            os.path.join(rocm_home, "bin"),
        ]
        if sdk_rocm_root and sdk_rocm_root != rocm_home:
            path_parts += [
                os.path.join(sdk_rocm_root, "lib", "llvm", "bin"),
                os.path.join(sdk_rocm_root, "bin"),
            ]
        os.environ["PATH"] = os.pathsep.join(path_parts + [os.environ.get("PATH", "")])
        os.environ["ROCM_HOME"] = rocm_home
        if os.name == "nt":
            os.environ["HIP_PATH"] = rocm_home
        cpp_extension.ROCM_HOME = rocm_home

        amd_arches = detect_rocm_arches(torch) or ["gfx1201"]
        os.environ.setdefault("PYTORCH_ROCM_ARCH", ";".join(amd_arches))
        print(f"Target AMD GPU architectures: {amd_arches}")

        has_gfx12 = any(arch.startswith("gfx12") for arch in amd_arches)
        if os.name == "nt":
            cxx_flags = ["/O2", "/std:c++17", f"/D_GLIBCXX_USE_CXX11_ABI={ABI}", "/DENABLE_BF16"]
        else:
            cxx_flags = ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-DENABLE_BF16"]

        hip_flags = [
            "-O3",
            "-std=c++17",
            "-ffast-math",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
        ]
        if os.getenv("SAGEATTN_GFX12_SAFE_BUILD", "0").upper() not in {"1", "TRUE", "YES"}:
            hip_flags += [
                "-mllvm",
                "--lsr-drop-solution=1",
                "-mllvm",
                "-enable-post-misched=1",
                "-mllvm",
                "-amdgpu-early-inline-all=true",
                "-mllvm",
                "-amdgpu-function-calls=false",
            ]
        for arch in amd_arches:
            hip_flags.append(f"--offload-arch={arch}")

        cxx_flags += split_flags(os.getenv("CXX_APPEND_FLAGS", ""))
        hip_flags += split_flags(os.getenv("NVCC_APPEND_FLAGS", ""))
        hip_flags += split_flags(os.getenv("HIPCC_APPEND_FLAGS", ""))

        include_dirs = [
            os.path.join(rocm_root, "include"),
            os.path.join(rocm_home, "include"),
        ]
        include_dirs += split_paths(os.getenv("SAGEATTN_ROCM_EXTRA_INCLUDE_DIRS", ""))
        repo_parent = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        therock_include_candidates = split_paths(os.getenv("SAGEATTN_THEROCK_INCLUDE_DIRS", ""))
        therock_include_candidates.append(
            os.path.join(repo_parent, "TheRock", "build", "dist", "rocm", "include")
        )
        for include_dir in therock_include_candidates:
            if os.path.exists(os.path.join(include_dir, "thrust", "complex.h")):
                include_dirs.append(include_dir)

        if has_gfx12:
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_gfx12_native",
                    sources=[
                        "csrc/qattn/pybind_gfx12_native.cpp",
                        "csrc/qattn/qk_int_sv_gfx12_native.cu",
                    ],
                    include_dirs=list(dict.fromkeys(include_dirs)),
                    extra_compile_args={"cxx": cxx_flags, "nvcc": hip_flags},
                )
            )
        else:
            warnings.warn(
                "ROCm build detected, but no gfx12 target architecture was selected; "
                "skipping the gfx12 native attention extension."
            )
        ext_modules.append(
            CUDAExtension(
                name="sageattention._fused",
                sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
                include_dirs=list(dict.fromkeys(include_dirs)),
                extra_compile_args={"cxx": cxx_flags, "nvcc": hip_flags},
            )
        )
    else:
        from torch.utils.cpp_extension import CUDA_HOME

        HAS_SM80 = False
        HAS_SM86 = False
        HAS_SM89 = False
        HAS_SM90 = False
        HAS_SM100 = False
        HAS_SM120 = False
        HAS_SM121 = False

        CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
        NVCC_FLAGS = [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--use_fast_math",
            "--threads=8",
            "-Xptxas=-v",
            "-diag-suppress=174",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
        ]
        CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
        CXX_FLAGS += split_flags(os.getenv("CXX_APPEND_FLAGS", ""))
        NVCC_FLAGS += split_flags(os.getenv("NVCC_APPEND_FLAGS", ""))

        if CUDA_HOME is None:
            raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build the package.")

        compute_capabilities = set()
        arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
        if arch_list_env:
            for item in arch_list_env.replace(",", ";").split(";"):
                it = item.strip()
                if not it:
                    continue
                it = it.lower().replace("sm_", "").replace("compute_", "")
                it = it.replace("a", "")
                if it.endswith("+ptx"):
                    it = it[:-4]
                    compute_capabilities.add(f"{it}+PTX")
                else:
                    if len(it) == 2 and it.isdigit():
                        it = f"{it[0]}.{it[1]}"
                    compute_capabilities.add(it)

        if not compute_capabilities:
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                if major < 8:
                    warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
                    continue
                compute_capabilities.add(f"{major}.{minor}")

        nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

        if not compute_capabilities:
            raise RuntimeError(
                "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
        else:
            print(f"Target compute capabilities: {compute_capabilities}")

        if nvcc_cuda_version < Version("12.0"):
            raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
        if nvcc_cuda_version < Version("12.4") and any(cc.startswith("8.9") for cc in compute_capabilities):
            raise RuntimeError("CUDA 12.4 or higher is required for compute capability 8.9.")
        if nvcc_cuda_version < Version("12.3") and any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError("CUDA 12.3 or higher is required for compute capability 9.0.")
        if nvcc_cuda_version < Version("12.8") and any(cc.startswith("12.0") for cc in compute_capabilities):
            raise RuntimeError("CUDA 12.8 or higher is required for compute capability 12.0.")

        for capability in compute_capabilities:
            if capability.startswith("8.0"):
                HAS_SM80 = True
                num = "80"
            elif capability.startswith("8.6"):
                HAS_SM86 = True
                num = "86"
            elif capability.startswith("8.9"):
                HAS_SM89 = True
                num = "89"
            elif capability.startswith("9.0"):
                HAS_SM90 = True
                num = "90a"
            elif capability.startswith("10.0"):
                HAS_SM100 = True
                num = "100a"
            elif capability.startswith("12.0"):
                HAS_SM120 = True
                num = "120a"
            elif capability.startswith("12.1"):
                HAS_SM121 = True
                num = "121a"
            else:
                continue
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
            if capability.endswith("+PTX"):
                NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

        if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120 or HAS_SM121:
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_sm80",
                    sources=[
                        "csrc/qattn/pybind_sm80.cpp",
                        "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
                    ],
                    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                )
            )

        if HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120 or HAS_SM121:
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_sm89",
                    sources=[
                        "csrc/qattn/pybind_sm89.cpp",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
                        "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                    ],
                    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                )
            )

        if HAS_SM90:
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_sm90",
                    sources=[
                        "csrc/qattn/pybind_sm90.cpp",
                        "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
                    ],
                    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                    extra_link_args=["-lcuda"],
                )
            )

        ext_modules.append(
            CUDAExtension(
                name="sageattention._fused",
                sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            )
        )

    if ext_modules:
        os.environ.setdefault("MAX_JOBS", "32")

        class BuildExtensionSeparateDir(BuildExtension):
            build_extension_patch_lock = threading.Lock()
            thread_ext_name_map = {}

            def finalize_options(self):
                if parallel is not None:
                    self.parallel = parallel
                super().finalize_options()

            def build_extension(self, ext):
                with self.build_extension_patch_lock:
                    if not getattr(self.compiler, "_compile_separate_output_dir", False):
                        compile_orig = self.compiler.compile

                        def compile_new(*args, **kwargs):
                            return compile_orig(*args, **{
                                **kwargs,
                                "output_dir": os.path.join(
                                    kwargs["output_dir"],
                                    self.thread_ext_name_map[threading.current_thread().ident]),
                            })

                        self.compiler.compile = compile_new
                        self.compiler._compile_separate_output_dir = True
                self.thread_ext_name_map[threading.current_thread().ident] = ext.name
                return super().build_extension(ext)

        cmdclass = {"build_ext": BuildExtensionSeparateDir}


setup(
    name="sageattention",
    version="2.2.0",
    author="SageAttention team",
    license="Apache 2.0 License",
    description="Accurate and efficient plug-and-play low-bit attention.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/SageAttention",
    packages=find_packages(),
    python_requires=">=3.9",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
