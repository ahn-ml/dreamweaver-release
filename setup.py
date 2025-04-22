from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name="dreamweaver",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
)