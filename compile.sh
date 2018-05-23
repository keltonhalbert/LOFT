#!/bin/bash
module unload darshan

module use /sw/bw/thg/modulefiles
module load phdf5
module load zfp
module load h5zzfp
module load szip
module load cray-netcdf
module load cudatoolkit


make -f Makefile.bw all 
