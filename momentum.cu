
#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#ifndef MOMENTUM_CU
#define MOMENTUM_CU

/* Compute the pressure gradient forcing for
   the W momentum equation */
__host__ __device__ void calc_pgrad_u(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    const float cp = 1005.7;
    const float reps = 461.5 / 287.04;

    // get dpi/dz
    float *buf0 = data->pi;
    float dpidx = (BUF4D(i, j, k, t) - BUF4D(i-1, j, k, t)) / (xh(i) - xh(i-1));

    // get theta_rho on U points by averaging them
    // to the staggered U level. Base state doesnt vary in X or Y.
    buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k];
    float thbar1 = grid->th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 

    float thrhopert1 = BUF4D(i, j, k, t);
    float thrhopert2 = BUF4D(i-1, j, k, t);
    float thrhou = 0.5*(thbar1 + thrhopert1) + 0.5*(thbar1 + thrhopert2);

    buf0 = data->pgradu;
    BUF4D(i, j, k, t) = -cp*thrhou*dpidx;
}
/* Compute the pressure gradient forcing for
   the V momentum equation */
__host__ __device__ void calc_pgrad_v(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    const float cp = 1005.7;
    const float reps = 461.5 / 287.04;

    // get dpi/dz
    float *buf0 = data->pi;
    float dpidy = (BUF4D(i, j, k, t) - BUF4D(i, j-1, k, t)) / (yh(j) - yh(j-1));

    // get theta_rho on V points by averaging them
    // to the staggered V level. Base state doesnt vary in X or Y.
    buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k];
    float thbar1 = grid->th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thrhopert1 = BUF4D(i, j, k, t);
    float thrhopert2 = BUF4D(i, j-1, k, t);
    float thrhov = 0.5*(thbar1 + thrhopert1) + 0.5*(thbar1 + thrhopert2);

    buf0 = data->pgradv;
    BUF4D(i, j, k, t) = -cp*thrhov*dpidy;
}

/* Compute the pressure gradient forcing for
   the W momentum equation */
__host__ __device__ void calc_pgrad_w(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    const float cp = 1005.7;
    const float reps = 461.5 / 287.04;

    // get dpi/dz
    float *buf0 = data->pi;
    float dpidz = (BUF4D(i, j, k, t) - BUF4D(i, j, k-1, t)) / (zh(k) - zh(k-1));

    // get theta_rho on W points by averaging them
    // to the staggered W level. NOTE: Need to do something
    // about k = 0. 
    buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k];
    float qvbar2 = grid->qv0[k-1];

    float thbar1 = grid->th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = grid->th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    float thrhopert1 = BUF4D(i, j, k, t);
    float thrhopert2 = BUF4D(i, j, k-1, t);
    float thrhow = 0.5*(thbar1 + thrhopert1) + 0.5*(thbar2 + thrhopert2);

    buf0 = data->pgradw;
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
    if ((i < NX) && (j < NY+1) && (i > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_pgrad_u(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX+1) && (j < NY) && (j > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_pgrad_v(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}


#endif
