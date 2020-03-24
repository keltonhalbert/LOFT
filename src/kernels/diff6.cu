#include <iostream>
#include <stdio.h>
extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
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
    int nx = gd->NX;
    int ny = gd->NY;
    int nz = gd->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < nx-3) && (j < ny-3) && (k >= 3) && (k < nz-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_u(ustag, tem1, i, j, k, nx, ny);
		calc_diffy_u(ustag, tem2, i, j, k, nx, ny);
		calc_diffz_u(ustag, snd->u0, tem3, i, j, k, nx, ny);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, nx+2, ny+2)] = -1.0*tem3[P3(i, j, 4, nx+2, ny+2)];
			tem3[P3(i, j, 1, nx+2, ny+2)] = tem3[P3(i, j, 3, nx+2, ny+2)];
			tem3[P3(i, j, 0, nx+2, ny+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiffVXYZ(grid *gd, mesh *msh, sounding *snd, float *vstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int nx = gd->NX;
    int ny = gd->NY;
    int nz = gd->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < nx-3) && (j < ny-3) && (k >= 3) && (k < nz-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_v(vstag, tem1, i, j, k, nx, ny);
		calc_diffy_v(vstag, tem2, i, j, k, nx, ny);
		calc_diffz_v(vstag, snd->v0, tem3, i, j, k, nx, ny);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, nx+2, ny+2)] = -1.0*tem3[P3(i, j, 4, nx+2, ny+2)];
			tem3[P3(i, j, 1, nx+2, ny+2)] = tem3[P3(i, j, 3, nx+2, ny+2)];
			tem3[P3(i, j, 0, nx+2, ny+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiffWXYZ(grid *gd, mesh *msh, sounding *snd, float *wstag, float *tem1, float *tem2, float *tem3) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int nx = gd->NX;
    int ny = gd->NY;
    int nz = gd->NZ;

    // this is a pretty large stencil so we have to be careful
    if ((i < nx-3) && (j < ny-3) && (k >= 3) && (k < nz-4) && (i >= 3) && (j >= 3)) {
		calc_diffx_w(wstag, tem1, i, j, k, nx, ny);
		calc_diffy_w(wstag, tem2, i, j, k, nx, ny);
		calc_diffz_w(wstag, tem3, i, j, k, nx, ny);

		// handle lower boundary condition for diffz_u
		// we've kind of ignored the top boundary...
		if ( (k == 3)  && (zf(0) == 0.0) ) {
			tem3[P3(i, j, 2, nx+2, ny+2)] = -1.0*tem3[P3(i, j, 4, nx+2, ny+2)];
			tem3[P3(i, j, 1, nx+2, ny+2)] = tem3[P3(i, j, 3, nx+2, ny+2)];
			tem3[P3(i, j, 0, nx+2, ny+2)] = 0.0;
		}
    }
}

__global__ void cuCalcDiff(grid *gd, mesh *msh, sounding *snd, float *diffx, float *diffy, float *diffz, float *difften) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int nx = gd->NX;
    int ny = gd->NY;
    int nz = gd->NZ;

    if ((i < nx+1) && (j < ny+1) && (k < nz)) {
		calc_diff(diffx, diffy, diffz, difften, msh->dt, i, j, k, nx, ny);
    }
}

#endif
