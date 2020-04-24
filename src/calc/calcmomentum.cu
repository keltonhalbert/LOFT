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
#include "../include/constants.h"
#ifndef MOMENTUM_CALC
#define MOMENTUM_CALC
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

/* Compute the pressure gradient forcing for
   the W momentum equation */
__device__ inline void calc_pgrad_u(float *pipert, float *thrhopert, float *qv0, float *th0, float *pgradu, \
		                              float dx, int i, int j, int k, int nx, int ny) {
    // get dpi/dz
    float *buf0 = pipert;
    float dpidx = (BUF(i, j, k) - BUF(i-1, j, k)) / dx;

    // get theta_rho on U points by averaging them
    // to the staggered U level. Base state doesnt vary in X or Y.
    buf0 = thrhopert;
    float qvbar1 = qv0[k];
    float thbar1 = th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 

    float thrhopert1 = BUF(i, j, k);
    float thrhopert2 = BUF(i-1, j, k);
    float thrhou = 0.5*( (thbar1 + thrhopert1) + (thbar1 + thrhopert2) );

    buf0 = pgradu;
    BUF(i, j, k) = -cp*thrhou*dpidx;
}

/* Compute the pressure gradient forcing for
   the V momentum equation */
__device__ inline void calc_pgrad_v(float *pipert, float *thrhopert, float *qv0, float *th0, float *pgradv, \
		                              float dy, int i, int j, int k, int nx, int ny) {
    // get dpi/dz
    float *buf0 = pipert;
    float dpidy = (BUF(i, j, k) - BUF(i, j-1, k)) / dy;

    // get theta_rho on V points by averaging them
    // to the staggered V level. Base state doesnt vary in X or Y.
    buf0 = thrhopert;
    float qvbar1 = qv0[k];
    float thbar1 = th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thrhopert1 = BUF(i, j, k);
    float thrhopert2 = BUF(i, j-1, k);
    float thrhov = 0.5*( (thbar1 + thrhopert1) + (thbar1 + thrhopert2) );

    buf0 = pgradv;
    BUF(i, j, k) = -cp*thrhov*dpidy;
}

/* Compute the pressure gradient forcing for
   the W momentum equation */
__device__ inline void calc_pgrad_w(float *pipert, float *thrhopert, float *qv0, float *th0, float *pgradw, \
		                              float dz, int i, int j, int k, int nx, int ny) {
    // get dpi/dz
    float *buf0 = pipert;
    float dpidz = (BUF(i, j, k) - BUF(i, j, k-1)) / dz;

    // get theta_rho on W points by averaging them
    // to the staggered W level. NOTE: Need to do something
    // about k = 0. 
    buf0 = thrhopert;
    float qvbar1 = qv0[k];
    float qvbar2 = qv0[k-1];

    float thbar1 = th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    float thrhopert1 = BUF(i, j, k);
    float thrhopert2 = BUF(i, j, k-1);
    float thrhow = 0.5*( (thbar1 + thrhopert1) + (thbar2 + thrhopert2) );

    buf0 = pgradw;
    BUF(i, j, k) = -cp*thrhow*dpidz;
}

/* Compute the buoyancy forcing
   the W momentum equation */
__device__ inline void calc_buoyancy(float *thrhopert, float *th0, float *buoy, int i, int j, int k, int nx, int ny) {
    float *buf0 = thrhopert;
    // we need to get this all on staggered W grid
    // in CM1, geroge uses base state theta for buoyancy
    float buoy1 = g*(BUF(i, j, k)/th0[k]);
    float buoy2 = g*(BUF(i, j, k-1)/th0[k-1]);
    buf0 = buoy;
    BUF(i, j, k) = 0.5*(buoy1 + buoy2);
}

#endif
