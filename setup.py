# setup.py
from setuptools import setup, find_packages

setup(
    name="UniMolXC",
    version="0.0.1",
    description="A package for incorporating machine learning into eXchange-Correlation functionals.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BariumOxide13716/Uni-Mol-XC",
    packages=find_packages(include=["UniMolXC", "UniMolXC.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "ase"
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)