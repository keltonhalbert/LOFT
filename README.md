# LOFT - Lagrangian Offline Flow Trajectories
LOFT is an offline trajectory integration package that leverages Graphical Processing Units (GPUs) to integrate up to tens of millions of lagrangian trajectories through CM1 simulations. Included is the ability to compute budgets of momentum and vorticity to track along the trajectories, as well as various scalar fields. 

* Data must be written from [CM1r19.8-LOFS](https://github.com/leighorf/cm1r19.8-LOFS), which uses a specialized data format built on distributed HDF5 files. The primary advantage is I/O performance when running CM1, and additional storage perofrmance through the usage of ZFP compression. 

* In order for LOFT to read the data from CM1, the [LOFS-read package must be installed](https://github.com/leighorf/LOFS-read). 

* Additional Requirements:
  * NVIDIA CUDA 10.1+ 
    * Note: It is assumed the GPU is compute compatability 61 or higher to leverage Unified Memory
  * NetCDF4 C++ API
  * ZFP
  * H5Z-ZFP plugin for HDF5
