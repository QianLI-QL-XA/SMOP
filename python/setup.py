#======================================================================
# setup.py -- build/install the smop Python package (pybind11 + C++ core)
#
#   pip install .
#   python -c "import smop; print(smop.version())"
#======================================================================
import os
import sys

from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # package root (smop/)
INCLUDE = os.path.join(ROOT, "include")
EIGEN = os.path.join(ROOT, "third_party", "eigen-3.4.0")
if not os.path.isdir(EIGEN):
    EIGEN = os.path.join(ROOT, "third_party", "eigen")
if not os.path.isdir(EIGEN):
    raise RuntimeError(
        "Eigen headers not found under third_party/. Download Eigen 3.4.x "
        "and place it at smop/third_party/eigen-3.4.0/")

ext_modules = [
    Pybind11Extension(
        "smop._core",
        [os.path.join("src", "smop_py.cpp")],
        include_dirs=[INCLUDE, EIGEN],
        cxx_std=17,
        extra_compile_args=["-O2"] if sys.platform != "win32" else ["/O2"],
        extra_link_args=[],
    ),
]

setup(
    name="smop",
    version="0.1.0",
    description="Level-set method for BMOP (min ||x||_1 s.t. ||Ax-b||<=delta), C++ core",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy"],
    zip_safe=False,
)
