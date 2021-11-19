#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# readme = open("README.md").read()
version = find_version("amulator", "__init__.py")


# Run the setup
setup(
    name="amulator",
    version=version,
    description="Emulator using GPyTorch and Ignite.",
    # long_description=readme,
    # long_description_content_type="text/markdown",
    author="Stijn Debackere",
    url="https://github.com/StijnDebackere/amulator",
    author_email="debackere@strw.leidenuniv.nl",
    license="MIT",
    classifiers=["Development Status :: 4 - Beta", "Programming Language :: Python :: 3"],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "dill>=0.3.0",
        "gpytorch>=1.5",
        "pytorch-ignite>=0.4.7",
        "torch>=1.9.0",
        "threadpoolctl>=2.2.0",
        "tqdm>=4",
    ]
)
