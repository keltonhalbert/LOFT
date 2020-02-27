#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#include "../calc/calcdiff6.cu"
#ifndef DIFF6_CU
#define DIFF6_CU
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the GPLV3 License.
 * Email: kthalbert@wisc.edu
*/

__global__ void doCalcDiffUXYZ(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffx_u(grid, data, idx_4D, NX, NY, NZ);
            calc_diffy_u(grid, data, idx_4D, NX, NY, NZ);
            calc_diffz_u(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void doCalcDiffVXYZ(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffx_v(grid, data, idx_4D, NX, NY, NZ);
            calc_diffy_v(grid, data, idx_4D, NX, NY, NZ);
            calc_diffz_v(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void doCalcDiffWXYZ(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffx_w(grid, data, idx_4D, NX, NY, NZ);
            calc_diffy_w(grid, data, idx_4D, NX, NY, NZ);
            calc_diffz_w(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void doCalcDiffU(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffu(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}
__global__ void doCalcDiffV(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffv(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}
__global__ void doCalcDiffW(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 4) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_diffw(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

#endif
