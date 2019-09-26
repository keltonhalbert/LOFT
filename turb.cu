#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#ifndef TURB_CU
#define TURB_CU

__device__ void calcrf(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    // use the w staggered grid
    float *wstag = data->rhof_4d_chunk;
    float *buf0 = data->rho_4d_chunk;

    if (k >= 2) {
        WA4D(i, j, k, t) =  (0.5 * (BUF4D(i, j, k-1, t) + grid->rho0[k-2]) + 0.5 * (BUF4D(i, j, k, t) + grid->rho0[k-1]));
    }
    // k == 1
    else {
        WA4D(i, j, 1, t) = (1.75*(BUF4D(i, j, 1, t) + grid->rho0[0]) - (BUF4D(i, j, 2, t) +grid->rho0[1]) + 0.25*(BUF4D(i, j, 3, t) + grid->rho0[2]));
    }
}

// calculate the deformation terms for the turbulence diagnostics. They get stored in the 
// arrays later designated for tau stress tensors and variables are named according to
// tensor notation
__device__ void calcdef(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *dum0;
    float *ustag, *vstag, *wstag;
    ustag = data->u_4d_chunk;
    vstag = data->v_4d_chunk;
    wstag = data->w_4d_chunk;

    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    // apply the zero strain condition for free slip to out subsurface/ghost zone
    // tau 11. Derivative is du/dx and therefore the derivative on the staggered mesh results on the scalar point.
    dum0 = data->tem1_4d_chunk;
    TEM4D(i, j, k, t) = ( ( UA4D(i+1, j, k, t) - UA4D(i, j, k, t) ) / dx ) * UH(i);

    // tau 12. Derivatives are no longer on the staggered mesh since it's du/dy and dv/dx. Therefore, and
    // averaging step must take place on the TEM array after calculation. 

    dum0 = data->tem2_4d_chunk;
    TEM4D(i, j, k, t) = ( ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / dy ) * VF(j) ) \
                        + ( ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / dx ) * UF(i) );

    // tau 22. Once again back on the scalar mesh. 
    dum0 = data->tem3_4d_chunk;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j+1, k, t) - VA4D(i, j, k, t) ) / dy ) * VH(j);

    // tau 33. On the scalar mesh. 
    dum0 = data->tem4_4d_chunk;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k+1, t) - WA4D(i, j, k, t) ) / dz ) * MH(k);

    if (k == 1) {
        // we'll go ahead and apply the zero strain condition on the lower boundary/ghost zone
        // for tau 13 and tau 23
        // tau 13 boundary
        dum0 = data->tem5_4d_chunk;
        TEM4D(i, j, 0, t) = 0.0;
        // tau 23 boundary
        dum0 = data->tem6_4d_chunk;
        TEM4D(i, j, 0, t) = 0.0;
    }

    if (k > 1) {

        // tau 13 is not on the scalar mesh
        dum0 = data->tem5_4d_chunk;
        TEM4D(i, j, k, t) = ( ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / dx ) * UF(i) ) \
                           +( ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / dz ) * MF(k) );

        // tau 23 is not on the scalar mesh
        dum0 = data->tem6_4d_chunk;
        TEM4D(i, j, k, t) = ( ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / dy ) * VF(j) ) \
                           +( ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / dz ) * MF(k) );

    }
}


__device__ void gettau(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *dum0, *buf0, *wstag, *kmstag;

    kmstag = data->kmh_4d_chunk;
    buf0 = data->rho_4d_chunk;

    // NOTE: Base state arrays have a different grid index to them because there is no ghost zone.
    // For example, rho0[0] corresponds to zh[1]. We need to be careful and make sure we offset 
    // our indices appropriately

    // tau 11
    dum0 = data->tem1_4d_chunk;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k-1]);
    // tau 22
    dum0 = data->tem3_4d_chunk;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k-1]);
    // tau 33
    dum0 = data->tem4_4d_chunk;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k-1]);

    // tau 12
    dum0 = data->tem2_4d_chunk;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.03125 * \
                        ( ( ( KM4D(i-1, j-1, k, t) + KM4D(i, j, k, t) ) + ( KM4D(i-1, j, k, t) + KM4D(i, j-1, k, t) ) ) \
                         +( ( KM4D(i-1, j-1, k+1, t) + KM4D(i, j, k+1, t) ) + ( KM4D(i-1, j, k+1, t) + KM4D(i, j-1, k+1, t) ) ) ) \
                         *( ( (BUF4D(i-1, j-1, k, t) + grid->rho0[k-1]) + (BUF4D(i, j, k, t) + grid->rho0[k-1]) ) \
                           + ((BUF4D(i-1, j, k, t) + grid->rho0[k-1]) + (BUF4D(i, j-1, k, t) + grid->rho0[k-1]) ) );
    if (k == 1) {
        // we'll go ahead and apply the zero strain condition on the lower boundary/ghost zone
        // for tau 13 and tau 23
        // tau 13 boundary
        dum0 = data->tem5_4d_chunk;
        TEM4D(i, j, 0, t) = 0.0;
        // tau 23 boundary
        dum0 = data->tem6_4d_chunk;
        TEM4D(i, j, 0, t) = 0.0;
    }

    if ((k >= 2)) {
        // tau 13
        dum0 = data->tem5_4d_chunk;
        wstag = data->rhof_4d_chunk; // rather than make a new maro, we'll just use the WA4D macro
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i-1, j, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i-1, j, k, t)) + (WA4D(i, j, k, t)) ); 
        // tau 23
        dum0 = data->tem6_4d_chunk;
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i, j-1, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i, j-1, k, t) + WA4D(i, j, k, t)) ); 
    }
}

__device__ void calc_turbu(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *ustag, *buf0, *dum0;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    // tau 11
    dum0 = data->tem1_4d_chunk;
    float turbx = ((TEM4D(i, j, k, t) - TEM4D(i-1, j, k, t)) / dx)*UF(i);

    // tau 12
    dum0 = data->tem2_4d_chunk;
    float turby = ((TEM4D(i, j+1, k, t) - TEM4D(i, j, k, t)) / dy)*VH(j);

    // tau 13
    dum0 = data->tem5_4d_chunk;
    float turbz = ((TEM4D(i, j, k+1, t) - TEM4D(i, j, k, t)) / dz)*MH(k);

    buf0 = data->rho_4d_chunk;
    // calculate the momentum tendency now
    float rru0 = 1.0 / (0.5 * ((BUF4D(i-1, j, k, t) + grid->rho0[k-1]) + (BUF4D(i, j, k, t) + grid->rho0[k-1])));
    ustag = data->turbu_4d_chunk;
    UA4D(i, j, k, t) = ( turbx + turby + turbz ) * rru0; 
}

__device__ void calc_turbv(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *vstag, *buf0, *dum0;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    // tau 12
    dum0 = data->tem2_4d_chunk;
    float turbx = ((TEM4D(i+1, j, k, t) - TEM4D(i, j, k, t)) / dx)*UH(i);

    // tau 22
    dum0 = data->tem3_4d_chunk;
    float turby = ((TEM4D(i, j, k, t) - TEM4D(i, j-1, k, t)) / dy)*VF(j);

    // tau 23
    dum0 = data->tem6_4d_chunk;
    float turbz = ((TEM4D(i, j, k+1, t) - TEM4D(i, j, k, t)) / dz)*MH(k);

    buf0 = data->rho_4d_chunk;
    // calculate the momentum tendency now
    float rrv0 = 1.0 / (0.5 * ((BUF4D(i, j-1, k, t) + grid->rho0[k-1]) + (BUF4D(i, j, k, t) + grid->rho0[k-1])));
    vstag = data->turbv_4d_chunk;
    VA4D(i, j, k, t) = ( turbx + turby + turbz ) * rrv0; 
}

__device__ void calc_turbw(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *wstag, *buf0, *dum0;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    // tau 13
    dum0 = data->tem5_4d_chunk;
    float turbx = ((TEM4D(i+1, j, k, t) - TEM4D(i, j, k, t)) / dx)*UH(i);

    // tau 23
    dum0 = data->tem6_4d_chunk;
    float turby = ((TEM4D(i, j+1, k, t) - TEM4D(i, j, k, t)) / dy)*VH(j);

    // tau 33
    dum0 = data->tem4_4d_chunk;
    float turbz = ((TEM4D(i, j, k, t) - TEM4D(i, j, k-1, t)) / dz)*MF(k);

    buf0 = data->rho_4d_chunk;
    // calculate the momentum tendency now
    float rrf = 1.0 / (0.5 * (BUF4D(i, j, k-1, t) + grid->rho0[k-1] + BUF4D(i, j, k, t) + grid->rho0[k-1]));
    wstag = data->turbw_4d_chunk;
    WA4D(i, j, k, t) = ( turbx + turby + turbz ) * rrf; 
}

__global__ void doCalcRf(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k < NZ+1) && (k >=1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calcrf(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}
__global__ void doCalcDef(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k < NZ+1) && (i > 0) && (j > 0) && (k >=1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calcdef(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void doGetTau(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k < NZ+1) && (i > 0) && (j > 0) && (k >=1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            gettau(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}
__global__ void doCalcTurb(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k < NZ+1) && (i > 0) && (j > 0) && (k >=1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_turbu(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX) && (j < NY) && (k < NZ+1) && (j > 0) && (i > 0) && (k >=1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_turbv(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX) && (j < NY) && (k < NZ+1) && (k >=2) && (i > 0) && (j > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_turbw(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}
#endif
