#!/usr/bin/env bash

module load cuda
module load hdf5
module load netcdf

ln -s ./Makefiles/Makefile.frontera ./Makefile
make
