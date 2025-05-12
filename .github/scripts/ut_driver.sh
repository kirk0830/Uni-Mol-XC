#!/usr/bin/bash

set -e # Exit on error

# abacus interface
python3 ./UniMolXC/abacus/control.py -v
python3 ./UniMolXC/abacus/inputio.py -v
python3 ./UniMolXC/abacus/struio.py -v

# flow


# geometry
python3 ./UniMolXC/geometry/cluster.py -v
