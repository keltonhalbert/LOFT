#!/usr/bin/env bash

#NOTE! YOU MUST BE ON AN RTX FRONTERA NODE

module load cuda/12.2
module load hdf5/1.14.0
module load netcdf/4.9.0

make clean
make
