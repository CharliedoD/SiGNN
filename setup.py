"""Build script for the C++ neighborhood sampling extension."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="sample_neighbor",
    ext_modules=[
        CppExtension(
            "sample_neighbor",
            sources=["signn/sampling/sample_neighbor.cpp"],
            extra_compile_args=["-g"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
