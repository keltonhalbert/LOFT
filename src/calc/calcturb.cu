#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef TURB_CALC
#define TURB_CALC

__device__ void calcrf(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    // use the w staggered grid
    float *wstag = data->rhof;
    float *buf0 = data->rhopert;

    if (k >= 2) {
        WA4D(i, j, k, t) =  (0.5 * (BUF4D(i, j, k-1, t) + grid->rho0[k-1]) + 0.5 * (BUF4D(i, j, k, t) + grid->rho0[k]));
    }
    // k == 1
    else {
        WA4D(i, j, 0, t) = (1.75*(BUF4D(i, j, 1, t) + grid->rho0[1]) - (BUF4D(i, j, 2, t) +grid->rho0[2]) + 0.25*(BUF4D(i, j, 3, t) + grid->rho0[3]));
    }
}

// calculate the deformation terms for the turbulence diagnostics. They get stored in the 
// arrays later designated for tau stress tensors and variables are named according to
// tensor notation
__device__ void calcdef(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *dum0;
    float *ustag, *vstag, *wstag;
    ustag = data->ustag;
    vstag = data->vstag;
    wstag = data->wstag;

    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);
    float dz = zf(k+1) - zf(k);

    // apply the zero strain condition for free slip to out subsurface/ghost zone
    // tau 11. Derivative is du/dx and therefore the derivative on the staggered mesh results on the scalar point.
    dum0 = data->tem1;
    TEM4D(i, j, k, t) = ( ( UA4D(i+1, j, k, t) - UA4D(i, j, k, t) ) / dx ) * UH(i);

    // tau 12. Derivatives are no longer on the staggered mesh since it's du/dy and dv/dx. Therefore, and
    // averaging step must take place on the TEM array after calculation. 

    dum0 = data->tem2;
    TEM4D(i, j, k, t) = ( ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / dy ) * VF(j) ) \
                        + ( ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / dx ) * UF(i) );

    // tau 22. Once again back on the scalar mesh. 
    dum0 = data->tem3;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j+1, k, t) - VA4D(i, j, k, t) ) / dy ) * VH(j);

    // tau 33. On the scalar mesh. 
    dum0 = data->tem4;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k+1, t) - WA4D(i, j, k, t) ) / dz ) * MH(k);

    if (k == 1) {
        // we'll go ahead and apply the zero strain condition on the lower boundary/ghost zone
        // for tau 13 and tau 23
        // tau 13 boundary
        dum0 = data->tem5;
        TEM4D(i, j, 0, t) = 0.0;
        // tau 23 boundary
        dum0 = data->tem6;
        TEM4D(i, j, 0, t) = 0.0;
    }

    if (k > 1) {

        // tau 13 is not on the scalar mesh
        dum0 = data->tem5;
        TEM4D(i, j, k, t) = ( ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / dx ) * UF(i) ) \
                           +( ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / dz ) * MF(k) );

        // tau 23 is not on the scalar mesh
        dum0 = data->tem6;
        TEM4D(i, j, k, t) = ( ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / dy ) * VF(j) ) \
                           +( ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / dz ) * MF(k) );

    }
}


__device__ void gettau(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *dum0, *buf0, *wstag, *kmstag;

    kmstag = data->kmh;
    buf0 = data->rhopert;

    // NOTE: Base state arrays have a different grid index to them because there is no ghost zone.
    // For example, rho0[0] corresponds to zh[1]. We need to be careful and make sure we offset 
    // our indices appropriately

    // tau 11
    dum0 = data->tem1;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);
    // tau 22
    dum0 = data->tem3;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);
    // tau 33
    dum0 = data->tem4;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);

    // tau 12
    dum0 = data->tem2;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.03125 * \
                        ( ( ( KM4D(i-1, j-1, k, t) + KM4D(i, j, k, t) ) + ( KM4D(i-1, j, k, t) + KM4D(i, j-1, k, t) ) ) \
                         +( ( KM4D(i-1, j-1, k+1, t) + KM4D(i, j, k+1, t) ) + ( KM4D(i-1, j, k+1, t) + KM4D(i, j-1, k+1, t) ) ) ) \
                         *( ( (BUF4D(i-1, j-1, k, t) + grid->rho0[k]) + (BUF4D(i, j, k, t) + grid->rho0[k]) ) \
                           + ((BUF4D(i-1, j, k, t) + grid->rho0[k]) + (BUF4D(i, j-1, k, t) + grid->rho0[k]) ) );
    if (k == 1) {
        // we'll go ahead and apply the zero strain condition on the lower boundary/ghost zone
        // for tau 13 and tau 23
        // tau 13 boundary
        dum0 = data->tem5;
        TEM4D(i, j, 0, t) = 0.0;
        // tau 23 boundary
        dum0 = data->tem6;
        TEM4D(i, j, 0, t) = 0.0;
    }

    if ((k >= 1)) {
        // tau 13
        dum0 = data->tem5;
        wstag = data->rhof; // rather than make a new maro, we'll just use the WA4D macro
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i-1, j, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i-1, j, k, t)) + (WA4D(i, j, k, t)) ); 
        // tau 23
        dum0 = data->tem6;
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i, j-1, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i, j-1, k, t) + WA4D(i, j, k, t)) ); 
    }
}

__device__ void calc_turbu(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *ustag, *buf0, *dum0;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);
    float dz = zf(k+1) - zf(k);

    // tau 11
    dum0 = data->tem1;
    float turbx = ((TEM4D(i, j, k, t) - TEM4D(i-1, j, k, t)) / dx)*UF(i);

    // tau 12
    dum0 = data->tem2;
    float turby = ((TEM4D(i, j+1, k, t) - TEM4D(i, j, k, t)) / dy)*VH(j);

    // tau 13
    dum0 = data->tem5;
    float turbz = ((TEM4D(i, j, k+1, t) - TEM4D(i, j, k, t)) / dz)*MH(k);

    buf0 = data->rhopert;
    // calculate the momentum tendency now
    float rru0 = 1.0 / (0.5 * ((BUF4D(i-1, j, k, t) + grid->rho0[k]) + (BUF4D(i, j, k, t) + grid->rho0[k])));
    ustag = data->turbu;
    UA4D(i, j, k, t) = ( turbx + turby + turbz ) * rru0; 
}

__device__ void calc_turbv(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *vstag, *buf0, *dum0;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);
    float dz = zf(k+1) - zf(k);

    // tau 12
    dum0 = data->tem2;
    float turbx = ((TEM4D(i+1, j, k, t) - TEM4D(i, j, k, t)) / dx)*UH(i);

    // tau 22
    dum0 = data->tem3;
    float turby = ((TEM4D(i, j, k, t) - TEM4D(i, j-1, k, t)) / dy)*VF(j);

    // tau 23
    dum0 = data->tem6;
    float turbz = ((TEM4D(i, j, k+1, t) - TEM4D(i, j, k, t)) / dz)*MH(k);

    buf0 = data->rhopert;
    // calculate the momentum tendency now
    float rrv0 = 1.0 / (0.5 * ((BUF4D(i, j-1, k, t) + grid->rho0[k]) + (BUF4D(i, j, k, t) + grid->rho0[k])));
    vstag = data->turbv;
    VA4D(i, j, k, t) = ( turbx + turby + turbz ) * rrv0; 
}

__device__ void calc_turbw(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *wstag, *buf0, *dum0;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);
    float dz = zf(k+1) - zf(k);

    // tau 13
    dum0 = data->tem5;
    float turbx = ((TEM4D(i+1, j, k, t) - TEM4D(i, j, k, t)) / dx)*UH(i);

    // tau 23
    dum0 = data->tem6;
    float turby = ((TEM4D(i, j+1, k, t) - TEM4D(i, j, k, t)) / dy)*VH(j);

    // tau 33
    dum0 = data->tem4;
    float turbz = ((TEM4D(i, j, k, t) - TEM4D(i, j, k-1, t)) / dz)*MF(k);

    buf0 = data->rhopert;
    // calculate the momentum tendency now
    float rrf = 1.0 / (0.5 * (BUF4D(i, j, k-1, t) + grid->rho0[k-1] + BUF4D(i, j, k, t) + grid->rho0[k]));
    wstag = data->turbw;
    WA4D(i, j, k, t) = ( turbx + turby + turbz ) * rrf; 
}
#endif