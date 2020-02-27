#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef DIFF6_CALC
#define DIFF6_CALC
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the GPLV3 License.
 * Email: kthalbert@wisc.edu
*/

__device__ void calc_diffx_u(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our x diffusion of u
    // in the tem1 array for later use
    float *dum0 = data->tem1;
    float *ustag = data->ustag;

    float pval = ( 10.0*( UA4D(i  , j, k, t) - UA4D(i-1, j, k, t) ) \
                   -5.0*( UA4D(i+1, j, k, t) - UA4D(i-2, j, k, t) ) \
                       +( UA4D(i+2, j, k, t) - UA4D(i-3, j, k, t) ) );
    if ( pval*(UA4D(i,j,k,t)-UA4D(i-1,j,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffx_v(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our x diffusion of v
    // in the tem1 array for later use
    float *dum0 = data->tem1;
    float *vstag = data->vstag;

    float pval = ( 10.0*( VA4D(i  , j, k, t) - VA4D(i-1, j, k, t) ) \
                   -5.0*( VA4D(i+1, j, k, t) - VA4D(i-2, j, k, t) ) \
                       +( VA4D(i+2, j, k, t) - VA4D(i-3, j, k, t) ) );
    if ( pval*(VA4D(i,j,k,t)-VA4D(i-1,j,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffx_w(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our x diffusion of w
    // in the tem1 array for later use
    float *dum0 = data->tem1;
    float *wstag = data->wstag;

    float pval = ( 10.0*( WA4D(i  , j, k, t) - WA4D(i-1, j, k, t) ) \
                   -5.0*( WA4D(i+1, j, k, t) - WA4D(i-2, j, k, t) ) \
                       +( WA4D(i+2, j, k, t) - WA4D(i-3, j, k, t) ) );
    if ( pval*(WA4D(i,j,k,t)-WA4D(i-1,j,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffy_u(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our y diffusion of u
    // in the tem2 array for later use
    float *dum0 = data->tem2;
    float *ustag = data->ustag;

    float pval = ( 10.0*( UA4D(i, j  , k, t) - UA4D(i, j-1, k, t) ) \
                   -5.0*( UA4D(i, j+1, k, t) - UA4D(i, j-2, k, t) ) \
                       +( UA4D(i, j+2, k, t) - UA4D(i, j-3, k, t) ) );
    if ( pval*(UA4D(i,j,k,t)-UA4D(i,j-1,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffy_v(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our y diffusion of v
    // in the tem2 array for later use
    float *dum0 = data->tem2;
    float *vstag = data->vstag;

    float pval = ( 10.0*( VA4D(i, j  , k, t) - VA4D(i, j-1, k, t) ) \
                   -5.0*( VA4D(i, j+1, k, t) - VA4D(i, j-2, k, t) ) \
                       +( VA4D(i, j+2, k, t) - VA4D(i, j-3, k, t) ) );
    if ( pval*(VA4D(i,j,k,t)-VA4D(i,j-1,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffy_w(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our y diffusion of w
    // in the tem2 array for later use
    float *dum0 = data->tem2;
    float *wstag = data->wstag;

    float pval = ( 10.0*( WA4D(i, j  , k, t) - WA4D(i, j-1, k, t) ) \
                   -5.0*( WA4D(i, j+1, k, t) - WA4D(i, j-2, k, t) ) \
                       +( WA4D(i, j+2, k, t) - WA4D(i, j-3, k, t) ) );
    if ( pval*(WA4D(i,j,k,t)-WA4D(i,j-1,k,t)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;
}

__device__ void calc_diffz_u(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our z diffusion of u
    // in the tem3 array for later use
    float *dum0 = data->tem3;
    float *ustag = data->ustag;

    // We have to subtract off the base state wind. This will look
    // a little ugly, but it's better than having a seperate kernel
    // and storing it in a separate array like CM1 does. Also keep
    // in mind the base state arrays do not have a lower ghost zone,
    // so indexing is offset by 1.
    float u01 = grid->u0[k-0];
    float u02 = grid->u0[k-1];
    float u03 = grid->u0[k+1];
    float u04 = grid->u0[k-2];
    float u05 = grid->u0[k+2];
    float u06 = grid->u0[k-3];
    float pval = ( 10.0*( (UA4D(i, j, k  , t) - u01) - (UA4D(i, j, k-1, t) - u02) ) \
                   -5.0*( (UA4D(i, j, k+1, t) - u03) - (UA4D(i, j, k-2, t) - u04) ) \
                       +( (UA4D(i, j, k+2, t) - u05) - (UA4D(i, j, k-3, t) - u06) ) );
    if ( pval*( (UA4D(i,j,k,t)-u01)-(UA4D(i,j,k-1,t)-u02) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;

    // boundary conditions
    if (k == 5) {
        TEM4D(i, j, 3, t) = -pval;
    }
    if (k == 4) {
        TEM4D(i, j, 2, t) = pval;
        // zero strain boundary condition
        TEM4D(i, j, 1, t) = 0.0;
    }
}

__device__ void calc_diffz_v(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our z diffusion of v
    // in the tem3 array for later use
    float *dum0 = data->tem3;
    float *vstag = data->vstag;

    // We have to subtract off the base state wind. This will look
    // a little ugly, but it's better than having a seperate kernel
    // and storing it in a separate array like CM1 does. Also keep
    // in mind the base state arrays do not have a lower ghost zone,
    // so indexing is offset by 1.
    float v01 = grid->v0[k-0];
    float v02 = grid->v0[k-1];
    float v03 = grid->v0[k+1];
    float v04 = grid->v0[k-2];
    float v05 = grid->v0[k+2];
    float v06 = grid->v0[k-3];
    float pval = ( 10.0*( (VA4D(i, j, k  , t) - v01) - (VA4D(i, j, k-1, t) - v02) ) \
                   -5.0*( (VA4D(i, j, k+1, t) - v03) - (VA4D(i, j, k-2, t) - v04) ) \
                       +( (VA4D(i, j, k+2, t) - v05) - (VA4D(i, j, k-3, t) - v06) ) );
    if ( pval*( (VA4D(i,j,k,t)-v01)-(VA4D(i,j,k-1,t)-v02) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;

    // boundary conditions
    if (k == 5) {
        TEM4D(i, j, 3, t) = -pval;
    }
    if (k == 4) {
        TEM4D(i, j, 2, t) = pval;
        // zero strain boundary condition
        TEM4D(i, j, 1, t) = 0.0;
    }
}

__device__ void calc_diffz_w(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // we're going to store our z diffusion of w
    // in the tem3 array for later use
    float *dum0 = data->tem3;
    float *wstag = data->wstag;

    float pval = ( 10.0*( WA4D(i, j, k  , t) - WA4D(i, j, k-1, t) ) \
                   -5.0*( WA4D(i, j, k+1, t) - WA4D(i, j, k-2, t) ) \
                       +( WA4D(i, j, k+2, t) - WA4D(i, j, k-3, t) ) );
    if ( pval*( WA4D(i,j,k,t)-WA4D(i,j,k-1,t) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM4D(i, j, k, t) = pval;

    // boundary conditions
    if (k == 5) {
        TEM4D(i, j, 3, t) = -pval;
    }
    if (k == 4) {
        TEM4D(i, j, 2, t) = pval;
        // zero strain boundary condition
        TEM4D(i, j, 1, t) = 0.0;
    }
}

__device__ void calc_diffu(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float kdiff6 = 0.080;
    float dt = grid->dt; 
    float coeff = (kdiff6/64.0/dt);

    // get diffx from tem1
    float *dum0 = data->tem1;
    float diffx = coeff*(TEM4D(i+1, j, k, t)-TEM4D(i, j, k, t));

    // get diffy from tem2
    dum0 = data->tem2;
    float diffy = coeff*(TEM4D(i, j+1, k, t)-TEM4D(i, j, k, t));

    // get diffz from tem3
    dum0 = data->tem3;
    float diffz = coeff*(TEM4D(i, j, k+1, t)-TEM4D(i, j, k, t));

    // put it in our uten array
    float *ustag = data->diffu;
    UA4D(i, j, k, t) = diffx + diffy + diffz;

}

__device__ void calc_diffv(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float kdiff6 = 0.080;
    float dt = grid->dt; 
    float coeff = (kdiff6/64.0/dt);

    // get diffx from tem1
    float *dum0 = data->tem1;
    float diffx = coeff*(TEM4D(i+1, j, k, t)-TEM4D(i, j, k, t));

    // get diffy from tem2
    dum0 = data->tem2;
    float diffy = coeff*(TEM4D(i, j+1, k, t)-TEM4D(i, j, k, t));

    // get diffz from tem3
    dum0 = data->tem3;
    float diffz = coeff*(TEM4D(i, j, k+1, t)-TEM4D(i, j, k, t));

    // put it in our uten array
    float *vstag = data->diffv;
    VA4D(i, j, k, t) = diffx + diffy + diffz;

}

__device__ void calc_diffw(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float kdiff6 = 0.080;
    float dt = grid->dt; 
    float coeff = (kdiff6/64.0/dt);

    // get diffx from tem1
    float *dum0 = data->tem1;
    float diffx = coeff*(TEM4D(i+1, j, k, t)-TEM4D(i, j, k, t));

    // get diffy from tem2
    dum0 = data->tem2;
    float diffy = coeff*(TEM4D(i, j+1, k, t)-TEM4D(i, j, k, t));

    // get diffz from tem3
    dum0 = data->tem3;
    float diffz = coeff*(TEM4D(i, j, k+1, t)-TEM4D(i, j, k, t));

    // put it in our uten array
    float *wstag = data->diffw;
    WA4D(i, j, k, t) = diffx + diffy + diffz;

}

#endif
