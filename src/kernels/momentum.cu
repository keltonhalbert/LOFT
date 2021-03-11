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
#include "../calc/calcmomentum.cu"
#ifndef MOMENTUM_CU
#define MOMENTUM_CU
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

__global__ void cuCalcBuoy(grid *gd, mesh *msh, sounding *snd, float *thrhopert, float *buoy) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    if ((i < NX+1) && (j < NY+1) && (k > 0) && (k < NZ+1)) {
		    calc_buoyancy(thrhopert, snd->th0, snd->qv0, buoy, i, j, k, NX, NY);
    }
}

__global__ void cuCalcPgradU(grid *gd, mesh *msh, sounding *snd, float *pipert, float *thrhopert, float *pgradu) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

	float dx;
    if ((i < NX) && (j < NY+1) && (i > 0) && (k < NZ+1)) {
		    dx = xh(i) - xh(i-1);
		    calc_pgrad_u(pipert, thrhopert, snd->qv0, snd->th0, pgradu, dx, i, j, k, NX, NY);
    }
}

__global__ void cuCalcPgradV(grid *gd, mesh *msh, sounding *snd, float *pipert, float *thrhopert, float *pgradv) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dy;

    if ((i < NX+1) && (j < NY) && (j > 0) && (k < NZ+1)) {
		dy = yh(j) - yh(j-1);
		calc_pgrad_v(pipert, thrhopert, snd->qv0, snd->th0, pgradv, dy, i, j, k, NX, NY);
    }
}

__global__ void cuCalcPgradW(grid *gd, mesh *msh, sounding *snd, float *pipert, float *thrhopert, float *pgradw) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dz;

    if ((i < NX+1) && (j < NY+1) && (k > 0) && (k < NZ+1)) {
		dz = zh(k) - zh(k-1);
		calc_pgrad_w(pipert, thrhopert, snd->qv0, snd->th0, pgradw, dz, i, j, k, NX, NY);
    }
}

#endif
