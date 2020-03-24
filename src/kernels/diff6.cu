#include <iostream>
#include <stdio.h>
extern "C" {
#include <lofs-read.h>
#include <dirstruct.h>
#include <hdf2nc.h>
#include <limits.h>
#include <macros.h>
}
#include "../include/datastructs.h"
#include "../calc/calcdiff6.cu"
#ifndef DIFF6_CU
#define DIFF6_CU
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

__global__ void cuCalcDiffUXYZ(grid *gd, mesh *msh, sounding *snd, float *ustag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 3) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_u(ustag, tem1, i, j, k, NX, NY);
		calc_diffy_u(ustag, tem2, i, j, k, NX, NY);
		calc_diffz_u(ustag, snd->u0, tem3, i, j, k, NX, NY);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, NX+2, NY+2)] = -1.0*tem3[P3(i, j, 4, NX+2, NY+2)];
			tem3[P3(i, j, 1, NX+2, NY+2)] = tem3[P3(i, j, 3, NX+2, NY+2)];
			tem3[P3(i, j, 0, NX+2, NY+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiffVXYZ(grid *gd, mesh *msh, sounding *snd, float *vstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < NX-3) && (j < NY-3) && (k >= 3) && (k < NZ-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_v(vstag, tem1, i, j, k, NX, NY);
		calc_diffy_v(vstag, tem2, i, j, k, NX, NY);
		calc_diffz_v(vstag, snd->v0, tem3, i, j, k, NX, NY);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, NX+2, NY+2)] = -1.0*tem3[P3(i, j, 4, NX+2, NY+2)];
			tem3[P3(i, j, 1, NX+2, NY+2)] = tem3[P3(i, j, 3, NX+2, NY+2)];
			tem3[P3(i, j, 0, NX+2, NY+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiffWXYZ(grid *gd, mesh *msh, sounding *snd, float *wstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

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

__global__ void cuCalcDiff(grid *gd, mesh *msh, sounding *snd, float *diffx, float *diffy, float *diffz, float *difften) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX+1) && (j < NY+1) && (k < NZ)) {
		calc_diff(diffx, diffy, diffz, difften, gd->dt, i, j, k, NX, NY);
    }
}

#endif
