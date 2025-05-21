#!/usr/bin/bash

set -e # Exit on error

# abacus interface
python3 ./UniMolXC/abacus/control.py -v
python3 ./UniMolXC/abacus/inputio.py -v
python3 ./UniMolXC/abacus/struio.py -v

# geometry
python3 ./UniMolXC/geometry/manip/cluster.py -v

# network
python3 ./UniMolXC/network/utility/xcfit.py -v
python3 ./UniMolXC/network/utility/xcloss.py -v
