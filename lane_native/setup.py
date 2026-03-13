"""
lane_native — pip-installable build wrapper
Usage:  cd lane_native && pip install .
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        # pybind11 cmake dir
        try:
            import pybind11
            cmake_args.append(
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"
            )
        except ImportError:
            pass

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )

        build_args = ["cmake", "--build", ".", "--config", "Release", "-j4"]
        subprocess.run(build_args, cwd=build_temp, check=True)


setup(
    name="lane_native",
    version="1.0.0",
    description="C++ accelerated lane detection for lane_opencv.py",
    ext_modules=[CMakeExtension("lane_native")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)
