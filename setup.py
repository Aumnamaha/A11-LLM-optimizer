from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='a11_cuda_engine',
    ext_modules=[
        CppExtension(
            name='a11_cuda_engine',
            sources=['csrc/engine_cuda.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
