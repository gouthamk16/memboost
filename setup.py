import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUDA home
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

setup(
    name="memboost",
    version="0.1.0",
    description="Mixed-precision 2/4-bit weight quantization engine for LLMs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy"],
    extras_require={"cuda": ["torch"]},
    ext_modules=[
        CUDAExtension(
            name="memboost._C",
            sources=[
                "core/bindings.cpp",
                "core/torch_ops.cu",
                "core/quantize.cu",
            ],
            include_dirs=[
                os.path.join(os.path.dirname(__file__), "core"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-UTTEST_QUANTIZE",  # Make sure test main is excluded
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
