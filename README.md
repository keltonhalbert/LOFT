# The TACOS Project
TACOS - Trajectory Analysis for CUDA Optimized Systems

Name clearly subject to change

To run in an interactive Blue Waters session (including himem node)

```
qsub -I -l nodes=1:ppn=16:xk -l walltime=1:00:00 -q high     
qsub -I -l nodes=1:ppn=16:xkhimem -l walltime=1:00:00 -q high
```

To compile, simply run the compile script provided for Blue Waters. To execute the program:
```
aprun -n 16 -d 1 ./run
```

