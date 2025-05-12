#!/usr/bin/bash

set -e # Exit on error

# abacus interface
python3 ./UniMolXC/abacus/control.py -v
python3 ./UniMolXC/abacus/inputio.py -v
python3 ./UniMolXC/abacus/struio.py -v

# geometry
python3 ./UniMolXC/geometry/manip/cluster.py -v
python3 ./UniMolXC/geometry/repr/_deepmd.py -v

