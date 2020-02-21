#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#include "../calc/calcdiff6.cu"
#ifndef DIFF6_CU
#define DIFF6_CU

__global__ void cuCalcDiffUXYZ(datagrid *grid, float *ustag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 3) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_u(ustag, tem1, i, j, k, NX, NY);
		calc_diffy_u(ustag, tem2, i, j, k, NX, NY);
		calc_diffz_u(ustag, grid->u0, tem3, i, j, k, NX, NY);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, NX+2, NY+2)] = -1.0*tem3[P3(i, j, 4, NX+2, NY+2)];
			tem3[P3(i, j, 1, NX+2, NY+2)] = tem3[P3(i, j, 3, NX+2, NY+2)];
			tem3[P3(i, j, 0, NX+2, NY+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiffVXYZ(datagrid *grid, float *vstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 3) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_v(vstag, tem1, i, j, k, NX, NY);
		calc_diffy_v(vstag, tem2, i, j, k, NX, NY);
		calc_diffz_v(vstag, grid->v0, tem3, i, j, k, NX, NY);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, NX+2, NY+2)] = -1.0*tem3[P3(i, j, 4, NX+2, NY+2)];
			tem3[P3(i, j, 1, NX+2, NY+2)] = tem3[P3(i, j, 3, NX+2, NY+2)];
			tem3[P3(i, j, 0, NX+2, NY+2)] = 0.0;
		}
    }
}

__global__ void doCalcDiffWXYZ(datagrid *grid, float *wstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 3) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_w(wstag, tem1, i, j, k, NX, NY);
		calc_diffy_w(wstag, tem2, i, j, k, NX, NY);
		calc_diffz_w(wstag, tem3, i, j, k, NX, NY);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, NX+2, NY+2)] = -1.0*tem3[P3(i, j, 4, NX+2, NY+2)];
			tem3[P3(i, j, 1, NX+2, NY+2)] = tem3[P3(i, j, 3, NX+2, NY+2)];
			tem3[P3(i, j, 0, NX+2, NY+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiff(datagrid *grid, float *diffx, float *diffy, float *diffz, float *difften) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    if ((i < NX+1) && (j < NY+1) && (k < NZ)) {
		calc_diff(diffx, diffy, diffz, difften, grid->dt, i, j, k, NX, NY);
    }
}

#endif
