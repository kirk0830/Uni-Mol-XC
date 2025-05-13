# Uni-Mol-XC

## Introduction

wabiwaibibabo

## Installation

Build the package with the following steps:

1. create virtual environment
2. install dependencies in the order DeePMD-kit (torch backend), then Huggingface, UniMol-Tools, finally this package.

```bash
# create virtual environment
# such as conda
conda create -n mlxc python=3.10

# then activate it
conda activate mlxc

# install dependencies
# install DeePMD-kit
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install deepmd-kit

# install Huggingface
pip install huggingface_hub

# install UniMol-Tools
pip install unimol_tools --upgrade

# install this package
pip install .
```

## Unit Test

```bash
# run unit test
bash .github/scripts/ut_driver.sh
```
