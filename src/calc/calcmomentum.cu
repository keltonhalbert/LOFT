#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef MOMENTUM_CALC
#define MOMENTUM_CALC
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the GPLV3 License.
 * Email: kthalbert@wisc.edu
*/

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
    float *buf0 = data->pipert;
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
    float *buf0 = data->pipert;
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
    float *buf0 = data->pipert;
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

/* Compute the buoyancy forcing
   the W momentum equation */
__device__ void calc_buoyancy(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    const float g = 9.81;
    float *buf0 = data->thrhopert;
    // we need to get this all on staggered W grid
    // in CM1, geroge uses base state theta for buoyancy
    float buoy1 = g*(BUF4D(i, j, k, t)/grid->th0[k]);
    float buoy2 = g*(BUF4D(i, j, k-1, t)/grid->th0[k-1]);
    buf0 = data->buoy;
    BUF4D(i, j, k, t) = 0.5*(buoy1 + buoy2);
}

#endif
