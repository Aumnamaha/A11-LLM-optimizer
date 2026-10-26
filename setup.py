import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# MSVC (Windows) uses different flag syntax than GCC/Clang (Linux/macOS).
if os.name == "nt":
    cxx_flags = ["/DGLOG_USE_GLOG_EXPORT", "/std:c++20"]
else:
    cxx_flags = ["-DGLOG_USE_GLOG_EXPORT", "-std=c++20"]

setup(
    name="a11_cuda_engine",
    ext_modules=[
        CppExtension(
            name="a11_cuda_engine",
            sources=["csrc/engine_cuda.cpp"],
            extra_compile_args={"cxx": cxx_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
