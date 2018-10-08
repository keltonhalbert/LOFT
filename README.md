# The TACOS Project
TACOS - Trajectory Analysis for CUDA Optimized Systems

Name clearly subject to change

To run in an interactive Blue Waters session (including himem node)

```
qsub -I -l nodes=1:ppn=16:xk -l walltime=1:00:00 -q high     
qsub -I -l nodes=1:ppn=16:xkhimem -l walltime=1:00:00 -q high
```

Example usage:
```
mpirun -n 4 ./run --histpath=/apollo/orfstore/khalbert/24May2011-every-time-step/3D --base=24May2011-svc-5500s --x0=-3500 --y0=-1400 --z0=15 --nx=100 --ny=100 --nz=100 --dx=10 --dy=10 --dz=10 --time=5500 --ntimes=900
```

