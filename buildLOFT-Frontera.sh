#!/usr/bin/env bash

#NOTE! YOU MUST BE ON AN RTX FRONTERA NODE

#These need to be typed by hand to stick, so do that, and then 'module
#save loft' so you can 'module restore loft' later.

#module load cuda/12.2
#module load hdf5/1.14.0
#module load netcdf/4.9.0

make clean
make
