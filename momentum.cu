
#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#ifndef MOMENTUM_CU
#define MOMENTUM_CU


/* Compute the pressure gradient forcing for
   the W momentum equation */
__host__ __device__ pgrad_w(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    const float cp = 1005.7;

    // get dpi/dz
    float *buf0 = data->prespert;
    float dpidz = BUF4D(i, j, k, t) - BUF4D(i, j, k-1);

    // get theta_rho on W points by averaging them
    // to the staggered W level. NOTE: Need to do something
    // about k = 0. 
    buf0 = data->thrhopert;
    float thbar1 = interp1D(grid, grid->th0, grid->zh[k], false, t); 
    float thbar2 = interp1D(grid, grid->th0, grid->zh[k-1], false, t);
    float thrhopert1 = BUF4D(i, j, k, t);
    float thrhopert2 = BUF4D(i, j, k-1, t);
    float thrhow = 0.5*(thbar1 + thrhoper1) + 0.5(thbar2 + thrhopert2);

    buf0 = data->wpgrad;
    BUF4D(i, j, k, t) = -cp*thrhow*dpidz;
}


__global__ void calcpgradw(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX+1) && (j < NY+1) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_pgrad_w(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}


#endif
