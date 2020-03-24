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
#include "../calc/calcturb.cu"
#ifndef TURB_CU
#define TURB_CU
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

__global__ void cuCalcRf(grid *gd, mesh *msh, sounding *snd, float *rhopert, float *rhof) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX) && (j < NY) && (k < NZ+1)) {
		calcrf(rhopert, snd->rho0, rhof, i, j, k, NX, NY);
    }
}

__global__ void cuCalcStrain(grid *gd, mesh *msh, sounding *snd, float *ustag, float *vstag, float *wstag, float *rhopert, \
		                     float *rhof, float *s11, float *s12, float *s13, float *s22, float *s23, float *s33) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dx, dy, dz;

    if ((i < NX) && (j < NY) && (k < NZ)) {
		dx = xf(i+1) - xf(i);
		dy = yf(j+1) - yf(j);
		dz = zf(k+1) - zf(k);
		calcstrain1(ustag, vstag, wstag, rhopert, snd->rho0, s11, s12, s22, s33, dx, dy, dz, i, j, k, NX, NY);
    }

	if ((i < NX+1) && (j < NY+1) && (k < NZ) && (i > 1) && (j > 1) && (k > 1)) {
		dx = xf(i) - xf(i-1);
		dy = yf(j) - yf(j-1);
		dz = zf(k) - zf(k-1);
		calcstrain2(ustag, vstag, wstag, rhof, s13, s23, dx, dy, dz, i, j, k, NX, NY);
	}
}

__global__ void cuGetTau(grid *gd, mesh *msh, sounding *snd, float *km, float *t11, float *t12, float *t13, float *t22, float *t23, float *t33) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX) && (j < NY) && (k < NZ)) {
		gettau1(km, t11, t12, t22, t33, i, j, k, NX, NY);
    }
	if ((i < NX+1) && (j < NY+1) && (k < NZ) && (i > 1) && (j > 1) && (k > 1)) {
		gettau2(km, t13, t23, i, j, k, NX, NY);
	}
}

__global__ void cuCalcTurb(grid *gd, mesh *msh, sounding *snd, float *t11, float *t12, float *t13, \
		                   float *t22, float *t23, float *t33, float *rhopert, 
						   float *rhof, float *turbu, float *turbv, float *turbw) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dx, dy, dz;

    if ((i < NX+1) && (j < NY) && (k < NZ)) {
		dx = xf(i) - xf(i-1);
		dy = yf(j+1) - yf(j);
		dz = zf(k+1) - zf(k);
		calc_turbu(t11, t12, t13, rhopert, snd->rho0, turbu, dx, dy, dz, i, j, k, NX, NY);
    }

    if ((i < NX) && (j < NY+1) && (k < NZ)) {
		dx = xf(i+1) - xf(i);
		dy = yf(j) - yf(j-1);
		dz = zf(k+1) - zf(k);
		calc_turbv(t12, t22, t23, rhopert, snd->rho0, turbv, dx, dy, dz, i, j, k, NX, NY);
    }

    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
		dx = xf(i+1) - xf(i);
		dy = yf(j+1) - yf(j);
		dz = zf(k) - zf(k-1);
		calc_turbw(t13, t23, t33, rhof, turbw, dx, dy, dz, i, j, k, NX, NY);
    }
}

#endif
