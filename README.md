# LOFT - Lagrangian Offline Flow Trajectories
LOFT is an offline trajectory integration package that leverages Graphical Processing Units (GPUs) to integrate up to tens of millions of Lagrangian trajectories through CM1 simulations. Included is the ability to compute budgets of momentum and vorticity to track along the trajectories, as well as various scalar fields. 

While its use cases are at the moment incredibly specific, the idea is that anyone who desires lossy compression of Large Eddy Simulations (LES) and high temporal resolution saves from simulations, these tools can be used to analyze the flow in a variety of ways using Lagrangian particle trajectories. In the event this isn't useful, it can provide some guidance on how to write CUDA C++ code for fluid dynamics problems, compute stencils that could be linked to for other projects, or just be an open resource so that results from publications can be verified. 

As of right now, it's not at 100% generalization. This assumes CM1 is running with no terrain and uses the zero-flux boundary condition, with free slip being a relatively straightforward addition to add in the future (and then perhaps semi-slip or no-slip). Additionally, the CM1 mesh is assumed to be isotropic (dx == dy == dz) for right now while tests are being developed. Stretched meshes will be included in the future. 

* Data must be written from [CM1r19.8-LOFS](https://github.com/leighorf/cm1r19.8-LOFS), which uses a specialized data format built on distributed HDF5 files. The primary advantage is I/O performance when running CM1, and additional storage perofrmance through the usage of ZFP compression. 

* In order for LOFT to read the data from CM1, the [LOFS-read package must be installed](https://github.com/leighorf/LOFS-read). 

* Additional Requirements:
  * NVIDIA CUDA 10.1+ 
    * Note: It is assumed the GPU is compute compatability 61 or higher to leverage Unified Memory
  * [NetCDF4 C++ API](https://github.com/Unidata/netcdf-cxx4)
  * [ZFP Floating Point Compression](https://computing.llnl.gov/projects/floating-point-compression)
  * [H5Z-ZFP](https://h5z-zfp.readthedocs.io/en/latest/) plugin for HDF5
