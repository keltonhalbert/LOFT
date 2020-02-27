#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/constants.h"
#include "../include/macros.h"
#ifndef DIFF6_CALC
#define DIFF6_CALC

__device__ void calc_diffx_u(float *ustag, float *diffxu, int i, int j, int k, int NX, int NY) {
    // we're going to store our x diffusion of u
    // in the tem1 array for later use
    float *dum0 = diffxu;

    float pval = ( 10.0*( UA(i  , j, k) - UA(i-1, j, k) ) \
                   -5.0*( UA(i+1, j, k) - UA(i-2, j, k) ) \
                       +( UA(i+2, j, k) - UA(i-3, j, k) ) );
    if ( pval*(UA(i,j,k)-UA(i-1,j,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffx_v(float *vstag, float *diffxv, int i, int j , int k, int NX, int NY) {
    // we're going to store our x diffusion of v
    // in the tem1 array for later use
    float *dum0 = diffxv;

    float pval = ( 10.0*( VA(i  , j, k) - VA(i-1, j, k) ) \
                   -5.0*( VA(i+1, j, k) - VA(i-2, j, k) ) \
                       +( VA(i+2, j, k) - VA(i-3, j, k) ) );
    if ( pval*(VA(i,j,k)-VA(i-1,j,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffx_w(float *wstag, float *diffxw, int i, int j, int k, int NX, int NY) {
    // we're going to store our x diffusion of w
    // in the tem1 array for later use
    float *dum0 = diffxw;

    float pval = ( 10.0*( WA(i  , j, k) - WA(i-1, j, k) ) \
                   -5.0*( WA(i+1, j, k) - WA(i-2, j, k) ) \
                       +( WA(i+2, j, k) - WA(i-3, j, k) ) );
    if ( pval*(WA(i,j,k)-WA(i-1,j,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffy_u(float *ustag, float *diffyu, int i, int j, int k, int NX, int NY) {
    // we're going to store our y diffusion of u
    // in the tem2 array for later use
    float *dum0 = diffyu;

    float pval = ( 10.0*( UA(i, j  , k) - UA(i, j-1, k) ) \
                   -5.0*( UA(i, j+1, k) - UA(i, j-2, k) ) \
                       +( UA(i, j+2, k) - UA(i, j-3, k) ) );
    if ( pval*(UA(i,j,k)-UA(i,j-1,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffy_v(float *vstag, float *diffyv, int i, int j, int k, int NX, int NY) {
    // we're going to store our y diffusion of v
    // in the tem2 array for later use
    float *dum0 = diffyv;

    float pval = ( 10.0*( VA(i, j  , k) - VA(i, j-1, k) ) \
                   -5.0*( VA(i, j+1, k) - VA(i, j-2, k) ) \
                       +( VA(i, j+2, k) - VA(i, j-3, k) ) );
    if ( pval*(VA(i,j,k)-VA(i,j-1,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffy_w(float *wstag, float *diffyw, int i, int j, int k, int NX, int NY) {
    // we're going to store our y diffusion of w
    // in the tem2 array for later use
    float *dum0 = diffyw;

    float pval = ( 10.0*( WA(i, j  , k) - WA(i, j-1, k) ) \
                   -5.0*( WA(i, j+1, k) - WA(i, j-2, k) ) \
                       +( WA(i, j+2, k) - WA(i, j-3, k) ) );
    if ( pval*(WA(i,j,k)-WA(i,j-1,k)) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffz_u(float *ustag, float *u0, float *diffzu, int i, int j, int k, int NX, int NY) {
    // we're going to store our z diffusion of u
    // in the tem3 array for later use
    float *dum0 = diffzu;

    // We have to subtract off the base state wind. This will look
    // a little ugly, but it's better than having a seperate kernel
    // and storing it in a separate array like CM1 does. 
    float u01 = u0[k-0];
    float u02 = u0[k-1];
    float u03 = u0[k+1];
    float u04 = u0[k-2];
    float u05 = u0[k+2];
    float u06 = u0[k-3];
    float pval = ( 10.0*( (UA(i, j, k  ) - u01) - (UA(i, j, k-1) - u02) ) \
                   -5.0*( (UA(i, j, k+1) - u03) - (UA(i, j, k-2) - u04) ) \
                       +( (UA(i, j, k+2) - u05) - (UA(i, j, k-3) - u06) ) );
    if ( pval*( (UA(i,j,k)-u01)-(UA(i,j,k-1)-u02) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffz_v(float *vstag, float *v0, float *diffzv, int i, int j, int k, int NX, int NY) {
    // we're going to store our z diffusion of v
    // in the tem3 array for later use
    float *dum0 = diffzv;

    // We have to subtract off the base state wind. This will look
    // a little ugly, but it's better than having a seperate kernel
    // and storing it in a separate array like CM1 does. Also keep
    // in mind the base state arrays do not have a lower ghost zone,
    // so indexing is offset by 1.
    float v01 = v0[k-0];
    float v02 = v0[k-1];
    float v03 = v0[k+1];
    float v04 = v0[k-2];
    float v05 = v0[k+2];
    float v06 = v0[k-3];
    float pval = ( 10.0*( (VA(i, j, k  ) - v01) - (VA(i, j, k-1) - v02) ) \
                   -5.0*( (VA(i, j, k+1) - v03) - (VA(i, j, k-2) - v04) ) \
                       +( (VA(i, j, k+2) - v05) - (VA(i, j, k-3) - v06) ) );
    if ( pval*( (VA(i,j,k)-v01)-(VA(i,j,k-1)-v02) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diffz_w(float *wstag, float *diffzw, int i, int j, int k, int NX, int NY) {
    // we're going to store our z diffusion of w
    // in the tem3 array for later use
    float *dum0 = diffzw;

    float pval = ( 10.0*( WA(i, j, k  ) - WA(i, j, k-1) ) \
                   -5.0*( WA(i, j, k+1) - WA(i, j, k-2) ) \
                       +( WA(i, j, k+2) - WA(i, j, k-3) ) );
    if ( pval*( WA(i,j,k)-WA(i,j,k-1) ) <= 0.0 ) {
        pval = 0.0;
    } 
    TEM(i, j, k) = pval;
}

__device__ void calc_diff(float *diffx, float *diffy, float *diffz, float *difften, float dt, int i, int j, int k, int NX, int NY) {
    const float coeff = (kdiff6/64.0/dt);

    float *dum0 = diffx; 
    float xten = coeff*(TEM(i+1, j, k)-TEM(i, j, k));

    // get diffy from tem2
    dum0 = diffy;
    float yten = coeff*(TEM(i, j+1, k)-TEM(i, j, k));

    // get diffz from tem3
    dum0 = diffz;
    float zten = coeff*(TEM(i, j, k+1)-TEM(i, j, k));

    // it really doesn't matter which array we use, 
	// as long as it's one of the staggered macros
    float *ustag = difften;
    UA(i, j, k) = xten + yten + zten;

}

#endif
